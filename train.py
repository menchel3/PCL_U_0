# coding=utf-8
"""Train the PCL-U MI + orthogonal + NMMR-V model."""

import datetime
import os
import random
import sys
import traceback

import numpy as np
import tensorflow as tf

from pcl_u.model import PCLUModel
from pcl_u.util import load_data, log, save_config, validation_indices


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('n_in', 3, 'Number of layers in each Z/X/W representation network.')
tf.app.flags.DEFINE_integer('dim_in', 16, 'Width of each Z/X/W representation network.')
tf.app.flags.DEFINE_integer('constrainedLayer', 2, 'Layers used by the deep orthogonal regularizer; 0 uses all.')
tf.app.flags.DEFINE_float('gamma', 30.0, 'Multiplier on the NMMR-V moment loss.')
tf.app.flags.DEFINE_float('alpha', 0.05, 'Multiplier on the six mutual-information objectives.')
tf.app.flags.DEFINE_float('beta', 100.0, 'Multiplier on the deep orthogonal regularizer.')
tf.app.flags.DEFINE_integer('h_network_width', 80, 'Hidden width of the outcome bridge.')
tf.app.flags.DEFINE_integer('h_network_depth', 3, 'Hidden depth of the outcome bridge.')
tf.app.flags.DEFINE_float('h_weight_decay', 3e-7, 'Outcome-bridge L2 regularization coefficient.')
tf.app.flags.DEFINE_float('scale', 0.05, 'RBF length scale in the NMMR-V kernel.')
tf.app.flags.DEFINE_integer('h_split', 1, 'Use treatment-specific outcome-bridge networks.')
tf.app.flags.DEFINE_integer('batch_norm', 0, 'Whether representation layers use batch normalization.')
tf.app.flags.DEFINE_string('normalization', 'divide', 'Representation normalization.')
tf.app.flags.DEFINE_string('nonlin', 'elu', 'Representation non-linearity.')
tf.app.flags.DEFINE_float('weight_init', 0.1, 'Network weight initialization scale.')

tf.app.flags.DEFINE_string('outcome', 'continuous', 'Outcome type used by evaluation: binary or continuous.')
tf.app.flags.DEFINE_float('dropout_in', 1.0, 'Representation dropout keep probability.')
tf.app.flags.DEFINE_float('dropout_out', 1.0, 'Bridge-network dropout keep probability.')
tf.app.flags.DEFINE_float('lrate', 1e-3, 'Learning rate.')
tf.app.flags.DEFINE_float('decay', 0.3, 'RMSProp decay.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Training mini-batch size; 0 uses the full training split.')
tf.app.flags.DEFINE_integer('experiments', 1, 'Number of dataset replications.')
tf.app.flags.DEFINE_integer('iterations', 2000, 'Number of training iterations.')
tf.app.flags.DEFINE_float('lrate_decay', 0.97, 'Learning-rate decay every 100 iterations.')
tf.app.flags.DEFINE_string('outdir', '../results/', 'Output directory.')
tf.app.flags.DEFINE_string('datadir', '../data/', 'Data directory.')
tf.app.flags.DEFINE_string('dataform', 'train.npz', 'Training data filename.')
tf.app.flags.DEFINE_string('data_test', 'test.npz', 'Test data filename.')
tf.app.flags.DEFINE_integer('seed', 1, 'Random seed.')
tf.app.flags.DEFINE_integer('repetitions', 1, 'Repeated initializations for a single dataset.')
tf.app.flags.DEFINE_string('optimizer', 'Adam', 'Optimizer name.')
tf.app.flags.DEFINE_integer('output_delay', 100, 'Iterations between loss logs.')
tf.app.flags.DEFINE_integer('pred_output_delay', 100, 'Iterations between stored predictions.')
tf.app.flags.DEFINE_float('val_part', 0.3, 'Validation fraction.')


NUM_ITERATIONS_PER_DECAY = 100


def _make_optimizer(learning_rate):
    if FLAGS.optimizer == 'Adagrad':
        return tf.train.AdagradOptimizer(learning_rate)
    if FLAGS.optimizer == 'GradientDescent':
        return tf.train.GradientDescentOptimizer(learning_rate)
    if FLAGS.optimizer == 'Adam':
        return tf.train.AdamOptimizer(learning_rate)
    return tf.train.RMSPropOptimizer(learning_rate, FLAGS.decay)


def train(model, sess, train_step, data, valid_indices, test_data,
          logfile, i_exp):
    """Train one dataset replication and collect prediction checkpoints."""
    n = data['c'].shape[0]
    train_indices = list(set(range(n)) - set(valid_indices))
    n_train = len(train_indices)
    if n_train == 0:
        raise ValueError('The training split is empty.')
    if FLAGS.batch_size > n_train:
        raise ValueError(
            'batch_size (%d) exceeds the training split size (%d).' %
            (FLAGS.batch_size, n_train))

    factual_feed = {
        model.c: data['c'][train_indices, :],
        model.a: data['a'][train_indices, :],
        model.y: data['y'][train_indices, :],
        model.do_in: 1.0,
        model.do_out: 1.0,
    }
    valid_feed = None
    if FLAGS.val_part > 0:
        valid_feed = {
            model.c: data['c'][valid_indices, :],
            model.a: data['a'][valid_indices, :],
            model.y: data['y'][valid_indices, :],
            model.do_in: 1.0,
            model.do_out: 1.0,
        }

    counterfactual_feed = None
    if data['HAVE_TRUTH']:
        counterfactual_feed = {
            model.c: data['c'][train_indices, :],
            model.a: 1.0 - data['a'][train_indices, :],
            model.y: data['y_cf'][train_indices, :],
            model.do_in: 1.0,
            model.do_out: 1.0,
        }

    sess.run(tf.global_variables_initializer())
    predictions_train = []
    predictions_test = []
    losses = []

    def evaluate_losses():
        objective, factual_rmse, mi_raw, nmmr, ortho = sess.run(
            [model.tot_loss, model.pred_loss, model.loss_mi_total,
             model.h_v_loss, model.ortho_loss], feed_dict=factual_feed)
        counterfactual_rmse = np.nan
        if counterfactual_feed is not None:
            counterfactual_rmse = sess.run(
                model.pred_loss, feed_dict=counterfactual_feed)

        valid_rmse = np.nan
        valid_mi = np.nan
        valid_nmmr = np.nan
        if valid_feed is not None:
            valid_rmse, valid_mi, valid_nmmr = sess.run(
                [model.pred_loss, model.loss_mi_total, model.h_v_loss],
                feed_dict=valid_feed)
        losses.append([
            objective, factual_rmse, counterfactual_rmse, mi_raw,
            valid_rmse, valid_mi, valid_nmmr])
        return (objective, factual_rmse, counterfactual_rmse, mi_raw,
                nmmr, ortho, valid_rmse, valid_mi, valid_nmmr)

    evaluate_losses()
    objective_is_nan = False

    for iteration in range(FLAGS.iterations):
        if FLAGS.batch_size == 0:
            batch_indices = random.sample(range(n_train), n_train)
        else:
            batch_indices = random.sample(range(n_train), FLAGS.batch_size)

        c_batch = data['c'][train_indices, :][batch_indices, :]
        a_batch = data['a'][train_indices, :][batch_indices]
        y_batch = data['y'][train_indices, :][batch_indices]

        if not objective_is_nan:
            sess.run(train_step, feed_dict={
                model.c: c_batch,
                model.a: a_batch,
                model.y: y_batch,
                model.do_in: FLAGS.dropout_in,
                model.do_out: FLAGS.dropout_out,
            })

        if iteration % FLAGS.output_delay == 0 or iteration == FLAGS.iterations - 1:
            values = evaluate_losses()
            (objective, factual_rmse, counterfactual_rmse, mi_raw,
             nmmr, ortho, valid_rmse, valid_mi, valid_nmmr) = values
            log(logfile, (
                'Iter-%d\tObj: %.3f,\tNMMR-V: %.3e,\tF: %.3f,'
                '\tCf: %.3f,\tMI: %.3g,\tOrtho: %.3g,'
                '\tValF: %.3f,\tValMI: %.3g,\tValNMMR: %.3e') % (
                    iteration, objective, nmmr, factual_rmse,
                    counterfactual_rmse, mi_raw, ortho,
                    valid_rmse, valid_mi, valid_nmmr))
            if np.isnan(objective):
                log(logfile, 'Experiment %d: objective is NaN; stopping updates.' % i_exp)
                objective_is_nan = True

        if ((FLAGS.pred_output_delay > 0 and
             iteration % FLAGS.pred_output_delay == 0) or
                iteration == FLAGS.iterations - 1):
            factual = sess.run(model.output, feed_dict={
                model.c: data['c'], model.a: data['a'],
                model.do_in: 1.0, model.do_out: 1.0})
            counterfactual = sess.run(model.output, feed_dict={
                model.c: data['c'], model.a: 1.0 - data['a'],
                model.do_in: 1.0, model.do_out: 1.0})
            predictions_train.append(
                np.concatenate([factual, counterfactual], axis=1))

            if test_data is not None:
                factual_test = sess.run(model.output, feed_dict={
                    model.c: test_data['c'], model.a: test_data['a'],
                    model.do_in: 1.0, model.do_out: 1.0})
                counterfactual_test = sess.run(model.output, feed_dict={
                    model.c: test_data['c'],
                    model.a: 1.0 - test_data['a'],
                    model.do_in: 1.0, model.do_out: 1.0})
                predictions_test.append(np.concatenate(
                    [factual_test, counterfactual_test], axis=1))

    return losses, predictions_train, predictions_test


def _tile_single_replication(data, experiments):
    if data is None or data['c'].ndim != 3 or data['c'].shape[2] != 1:
        return
    data['c'] = np.tile(data['c'], (1, 1, experiments))
    data['a'] = np.tile(data['a'], (1, experiments))
    data['y'] = np.tile(data['y'], (1, experiments))
    if data['y_cf'] is not None:
        data['y_cf'] = np.tile(data['y_cf'], (1, experiments))


def _validate_replications(data, experiments, label):
    if data['c'].ndim != 3:
        raise ValueError('%s covariates must have shape [n, d, experiments].' % label)
    if data['c'].shape[2] < experiments:
        raise ValueError(
            '%s has %d replications but experiments=%d.' %
            (label, data['c'].shape[2], experiments))


def run(outdir):
    result_file = os.path.join(outdir, 'result')
    result_test_file = os.path.join(outdir, 'result.test')
    logfile = os.path.join(outdir, 'log.txt')
    open(logfile, 'w').close()

    train_path = os.path.join(FLAGS.datadir, FLAGS.dataform)
    test_path = None
    if FLAGS.data_test:
        test_path = os.path.join(FLAGS.datadir, FLAGS.data_test)

    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    save_config(os.path.join(outdir, 'config.txt'), FLAGS)

    log(logfile, (
        'PCL-U method: gamma=%.3g, alpha=%.3g, beta=%.3g, layers=%d, dim=%d, '
        'constrained=%d, h_width=%d, h_depth=%d, h_decay=%.3g, '
        'scale=%.3g, h_split=%d') % (
            FLAGS.gamma, FLAGS.alpha, FLAGS.beta,
            FLAGS.n_in, FLAGS.dim_in, FLAGS.constrainedLayer,
            FLAGS.h_network_width, FLAGS.h_network_depth,
            FLAGS.h_weight_decay, FLAGS.scale,
            FLAGS.h_split))
    log(logfile, (
        'Training: iterations=%d, batch_size=%d, lrate=%.3g, '
        'optimizer=%s, val_part=%.2g') % (
            FLAGS.iterations, FLAGS.batch_size, FLAGS.lrate,
            FLAGS.optimizer, FLAGS.val_part))
    log(logfile, 'Training data: ' + train_path)
    if test_path is not None:
        log(logfile, 'Test data:     ' + test_path)

    data = load_data(train_path)
    test_data = load_data(test_path) if test_path is not None else None
    _tile_single_replication(data, FLAGS.experiments)
    _tile_single_replication(test_data, FLAGS.experiments)
    _validate_replications(data, FLAGS.experiments, 'Training data')
    if test_data is not None:
        _validate_replications(test_data, FLAGS.experiments, 'Test data')
    log(logfile, 'Loaded data with shape [%d,%d]' % (data['n'], data['dim']))

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    c = tf.placeholder('float', shape=[None, data['dim']], name='C')
    a = tf.placeholder('float', shape=[None, 1], name='A')
    y = tf.placeholder('float', shape=[None, 1], name='Y')
    do_in = tf.placeholder('float', name='dropout_in')
    do_out = tf.placeholder('float', name='dropout_out')

    log(logfile, 'Defining PCL-U MI + orthogonal + NMMR-V graph...')
    model = PCLUModel(
        c, a, y, do_in, do_out, [data['dim'], FLAGS.dim_in], FLAGS)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(
        FLAGS.lrate, global_step, NUM_ITERATIONS_PER_DECAY,
        FLAGS.lrate_decay, staircase=True)
    optimizer = _make_optimizer(learning_rate)
    train_step = optimizer.minimize(model.tot_loss, global_step=global_step)

    n_experiments = FLAGS.experiments
    if FLAGS.repetitions > 1:
        if FLAGS.experiments > 1:
            raise ValueError(
                'repetitions>1 and experiments>1 cannot be used together.')
        n_experiments = FLAGS.repetitions

    all_losses = []
    all_predictions_train = []
    all_predictions_test = []
    all_valid = []

    for i_exp in range(1, n_experiments + 1):
        log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))
        if i_exp == 1 or FLAGS.experiments > 1:
            data_exp = {
                'c': data['c'][:, :, i_exp - 1],
                'a': data['a'][:, i_exp - 1:i_exp],
                'y': data['y'][:, i_exp - 1:i_exp],
                'y_cf': (data['y_cf'][:, i_exp - 1:i_exp]
                        if data['HAVE_TRUTH'] else None),
                'HAVE_TRUTH': data['HAVE_TRUTH'],
            }
            test_exp = None
            if test_data is not None:
                test_exp = {
                    'c': test_data['c'][:, :, i_exp - 1],
                    'a': test_data['a'][:, i_exp - 1:i_exp],
                    'y': test_data['y'][:, i_exp - 1:i_exp],
                    'y_cf': (test_data['y_cf'][:, i_exp - 1:i_exp]
                            if test_data['HAVE_TRUTH'] else None),
                    'HAVE_TRUTH': test_data['HAVE_TRUTH'],
                }

        valid_indices = validation_indices(data_exp, FLAGS.val_part)
        losses, predictions_train, predictions_test = train(
            model, sess, train_step, data_exp, valid_indices,
            test_exp, logfile, i_exp)

        all_losses.append(losses)
        all_predictions_train.append(predictions_train)
        all_predictions_test.append(predictions_test)
        all_valid.append(valid_indices)

        output_predictions_train = np.swapaxes(
            np.swapaxes(all_predictions_train, 1, 3), 0, 2)
        output_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)
        np.savez(
            result_file, pred=output_predictions_train,
            loss=output_losses, val=np.array(all_valid))

        if test_data is not None:
            output_predictions_test = np.swapaxes(
                np.swapaxes(all_predictions_test, 1, 3), 0, 2)
            np.savez(result_test_file, pred=output_predictions_test)

        log(logfile, 'Saving result to %s...' % outdir)


def main(argv=None):
    del argv
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S-%f')
    outdir = os.path.join(FLAGS.outdir, 'results_' + timestamp)
    os.makedirs(outdir)
    try:
        run(outdir)
    except Exception:
        with open(os.path.join(outdir, 'error.txt'), 'w') as error_file:
            error_file.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    tf.app.run()
