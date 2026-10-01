# coding=utf-8
"""PCL-U representation learning and outcome bridge."""

import numpy as np
import tensorflow as tf


def safe_sqrt(x):
    """Numerically stable square root used by representation normalization."""
    return tf.sqrt(tf.clip_by_value(x, 1e-6, 1e8))


class PCLUModel(object):
    """Learn Z/X/W roles from C for treatment A and outcome Y."""

    def __init__(self, c, a, y, do_in, do_out, dims, flags):
        self.c = c
        self.a = a
        self.y = y
        self.do_in = do_in
        self.do_out = do_out
        self.num = tf.shape(self.c)[0]

        if flags.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        elif flags.nonlin.lower() == 'tanh':
            self.nonlin = tf.nn.tanh
        elif flags.nonlin.lower() == 'relu':
            self.nonlin = tf.nn.relu
        else:
            raise ValueError('Unsupported non-linearity: %s' % flags.nonlin)

        self._build_graph(dims, flags)

    def _build_graph(self, dims, flags):
        dim_input, dim_in = dims
        gamma = flags.gamma
        alpha = flags.alpha
        beta = flags.beta

        if flags.n_in < 1:
            raise ValueError('n_in must be at least 1')

        # All three latent roles receive the complete observed covariate vector.
        with tf.name_scope('representation'):
            z_rep, z_norm, z_weights = \
                self._build_representation_graph(dim_input, dim_in, flags)
            x_rep, x_norm, x_weights = \
                self._build_representation_graph(dim_input, dim_in, flags)
            w_rep, w_norm, w_weights = \
                self._build_representation_graph(dim_input, dim_in, flags)

        tf.concat([z_rep, x_rep, w_rep], axis=1)
        tf.concat([z_norm, x_norm, w_norm], axis=1)
        if beta > 0:
            num_layers = (len(z_weights) if flags.constrainedLayer == 0
                          else flags.constrainedLayer)
            max_layers = min(len(z_weights), len(x_weights), len(w_weights))
            if num_layers < 1 or num_layers > max_layers:
                raise ValueError(
                    'constrainedLayer must be 0 or between 1 and n_in '
                    'when beta > 0')

            weight_z = z_weights[0]
            weight_x = x_weights[0]
            weight_w = w_weights[0]
            for i in range(1, num_layers):
                weight_z = tf.matmul(weight_z, z_weights[i])
                weight_x = tf.matmul(weight_x, x_weights[i])
                weight_w = tf.matmul(weight_w, w_weights[i])

            importance_z = tf.reduce_mean(tf.abs(weight_z), axis=1)
            importance_x = tf.reduce_mean(tf.abs(weight_x), axis=1)
            importance_w = tf.reduce_mean(tf.abs(weight_w), axis=1)
            ortho_dist = (
                tf.reduce_sum(importance_z * importance_x) +
                tf.reduce_sum(importance_z * importance_w) +
                tf.reduce_sum(importance_x * importance_w))
            self.ortho_loss = beta * ortho_dist
        else:
            self.ortho_loss = tf.constant(0.0, dtype=tf.float32)

        # Outcome bridge h(A,W,X).  The split form creates h0(W,X), h1(W,X).
        with tf.name_scope('h_network'):
            h_covariates = tf.concat([w_norm, x_norm], axis=1)
            if flags.h_split:
                with tf.name_scope('control'):
                    h0, h0_weights, h0_pred, h0_biases, h0_pred_bias = \
                        self._build_h_graph(
                            h_covariates, 2 * dim_in, self.do_out, flags)
                with tf.name_scope('treated'):
                    h1, h1_weights, h1_pred, h1_biases, h1_pred_bias = \
                        self._build_h_graph(
                            h_covariates, 2 * dim_in, self.do_out, flags)
                h_hat = (1.0 - self.a) * h0 + self.a * h1
                h_weights = h0_weights + h1_weights
                h_biases = h0_biases + h1_biases
                h_pred = tf.concat([h0_pred, h1_pred], axis=1)
                h_pred_bias = tf.concat([h0_pred_bias, h1_pred_bias], axis=0)
            else:
                h_input = tf.concat([self.a, h_covariates], axis=1)
                h_hat, h_weights, h_pred, h_biases, h_pred_bias = \
                    self._build_h_graph(
                        h_input, 1 + 2 * dim_in, self.do_out, flags)

        # Role constraints: Z contains A but excludes Y; X contains both A
        # and Y; W contains Y but excludes A.
        if alpha > 0:
            with tf.variable_scope('mi_za'):
                lld_za, bound_za = self._mi_net(
                    z_norm, self.a, dim_in, 1, 'max', 'za')
            with tf.variable_scope('mi_zy'):
                lld_zy, bound_zy = self._mi_net(
                    z_norm, self.y, dim_in, 1, 'min', 'zy')
            with tf.variable_scope('mi_xa'):
                lld_xa, bound_xa = self._mi_net(
                    x_norm, self.a, dim_in, 1, 'max', 'xa')
            with tf.variable_scope('mi_xy'):
                lld_xy, bound_xy = self._mi_net(
                    x_norm, self.y, dim_in, 1, 'max', 'xy')
            with tf.variable_scope('mi_wa'):
                lld_wa, bound_wa = self._mi_net(
                    w_norm, self.a, dim_in, 1, 'min', 'wa')
            with tf.variable_scope('mi_wy'):
                lld_wy, bound_wy = self._mi_net(
                    w_norm, self.y, dim_in, 1, 'max', 'wy')

            loss_mi_x = lld_xa + lld_xy + bound_xa + bound_xy
            loss_mi_z = lld_za + lld_zy + bound_za + bound_zy
            loss_mi_w = lld_wa + lld_wy + bound_wa + bound_wy
            self.loss_mi_total = loss_mi_z + loss_mi_x + loss_mi_w
            mi_loss = alpha * self.loss_mi_total
        elif alpha == 0:
            self.loss_mi_total = tf.constant(0.0, dtype=tf.float32)
            mi_loss = self.loss_mi_total
        else:
            raise ValueError('alpha must be non-negative')

        if gamma <= 0:
            raise ValueError('gamma must be greater than zero for NMMR-V')

        # NMMR-V moment restriction:
        #   (Y-h(A,W,X))^T K((A,Z,X),(A,Z,X)) (Y-h(A,W,X)) / n^2.
        kernel_input = tf.concat([self.a, z_norm, x_norm], axis=1)
        self.h_v_loss = self._nmmr_v_loss(
            h_hat, self.y, kernel_input, flags.scale)
        nmmr_loss = gamma * self.h_v_loss
        self.pred_loss = tf.sqrt(tf.reduce_mean(tf.square(self.y - h_hat)))

        h_params = h_weights + [h_pred] + h_biases + [h_pred_bias]
        if flags.h_weight_decay > 0:
            h_weight_decay_loss = flags.h_weight_decay * tf.add_n(
                [tf.nn.l2_loss(param) for param in h_params])
        else:
            h_weight_decay_loss = tf.constant(0.0, dtype=tf.float32)

        self.tot_loss = (
            nmmr_loss + mi_loss + self.ortho_loss + h_weight_decay_loss)
        self.output = h_hat

    def _build_representation_graph(self, dim_input, dim_in, flags):
        weights = []
        layers = [self.c]

        for i in range(flags.n_in):
            if i == 0:
                weights.append(tf.Variable(tf.random_normal(
                    [dim_input, dim_in],
                    stddev=flags.weight_init / np.sqrt(dim_input))))
            else:
                weights.append(tf.Variable(tf.random_normal(
                    [dim_in, dim_in],
                    stddev=flags.weight_init / np.sqrt(dim_in))))

            bias = tf.Variable(tf.zeros([1, dim_in]))
            z = tf.matmul(layers[i], weights[i]) + bias
            if flags.batch_norm:
                batch_mean, batch_var = tf.nn.moments(z, [0])
                if flags.normalization == 'bn_fixed':
                    z = tf.nn.batch_normalization(
                        z, batch_mean, batch_var, 0, 1, 1e-3)
                else:
                    bn_bias = tf.Variable(tf.zeros([dim_in]))
                    bn_scale = tf.Variable(tf.ones([dim_in]))
                    z = tf.nn.batch_normalization(
                        z, batch_mean, batch_var,
                        bn_bias, bn_scale, 1e-3)
            layer = self.nonlin(z)
            layers.append(tf.nn.dropout(layer, self.do_in))

        representation = layers[-1]
        if flags.normalization == 'divide':
            normalized = representation / safe_sqrt(tf.reduce_sum(
                tf.square(representation), axis=1, keep_dims=True))
        elif flags.normalization == 'l2':
            normalized = tf.nn.l2_normalize(representation, dim=1)
        else:
            normalized = 1.0 * representation
        return representation, normalized, weights

    def _build_h_graph(self, h_input, dim_input, do_out, flags):
        h = h_input
        weights = []
        biases = []
        dim_current = dim_input

        for i in range(flags.h_network_depth):
            weight = tf.Variable(tf.random_normal(
                [dim_current, flags.h_network_width],
                stddev=flags.weight_init / np.sqrt(dim_current)),
                name='h_w_%d' % i)
            bias = tf.Variable(
                tf.zeros([1, flags.h_network_width]), name='h_b_%d' % i)
            weights.append(weight)
            biases.append(bias)
            h = tf.nn.relu(tf.matmul(h, weight) + bias)
            h = tf.nn.dropout(h, do_out)
            dim_current = flags.h_network_width

        weight_pred = tf.Variable(tf.random_normal(
            [dim_current, 1],
            stddev=flags.weight_init / np.sqrt(dim_current)), name='h_w_pred')
        bias_pred = tf.Variable(tf.zeros([1]), name='h_b_pred')
        prediction = tf.matmul(h, weight_pred) + bias_pred
        return prediction, weights, weight_pred, biases, bias_pred

    def _nmmr_v_loss(self, model_output, target, kernel_input, length_scale):
        if length_scale <= 0:
            raise ValueError('scale must be greater than zero')
        residual = target - model_output
        pairwise_difference = (
            tf.expand_dims(kernel_input, axis=1) -
            tf.expand_dims(kernel_input, axis=0))
        kernel_matrix = tf.exp(
            -tf.reduce_sum(tf.square(pairwise_difference), axis=2) /
            (2.0 * length_scale ** 2))
        n = tf.maximum(tf.cast(tf.shape(residual)[0], tf.float32), 1.0)
        loss = tf.matmul(
            tf.matmul(residual, kernel_matrix, transpose_a=True), residual)
        return tf.squeeze(loss) / tf.square(n)

    def _fc_net(self, inp, dim_out, act_fun):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.contrib.layers.fully_connected(
            inputs=inp,
            num_outputs=dim_out,
            activation_fn=act_fun,
            weights_initializer=initializer)

    def _mi_net(self, inp, outp, dim_in, dim_out, mi_min_max, name):
        h_mu = self._fc_net(inp, dim_in // 2, tf.nn.elu)
        mu = self._fc_net(h_mu, dim_out, None)
        h_var = self._fc_net(inp, dim_in // 2, tf.nn.elu)
        logvar = self._fc_net(h_var, dim_out, tf.nn.tanh)

        new_order = tf.random_shuffle(tf.range(self.num))
        outp_rand = tf.gather(outp, new_order)

        loglikeli = -tf.reduce_mean(tf.reduce_sum(
            -(outp - mu) ** 2 / tf.exp(logvar) - logvar,
            axis=-1))
        pos = -(mu - outp) ** 2 / tf.exp(logvar)
        neg = -(mu - outp_rand) ** 2 / tf.exp(logvar)

        soft_weights = 1.0 / tf.cast(self.num, tf.float32)

        if mi_min_max == 'min':
            direction = 1.0
        elif mi_min_max == 'max':
            direction = -1.0
        else:
            raise ValueError("mi_min_max must be 'min' or 'max'")

        bound = direction * tf.reduce_mean(
            soft_weights * (pos - neg))
        return loglikeli, bound
