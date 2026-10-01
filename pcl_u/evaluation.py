import numpy as np
import os

from sklearn.metrics import roc_auc_score

from .logger import Logger as Log
from .loader import *

POL_CURVE_RES = 40

class NaNException(Exception):
    pass


def policy_range(n, res=10):
    step = int(float(n)/float(res))
    n_range = list(range(0,int(n+1),step))
    if not n_range[-1] == n:
        n_range.append(n)

    while len(n_range) > res:
        k = np.random.randint(len(n_range)-2)+1
        del n_range[k]

    return n_range

def policy_val(a, y, eff_pred, compute_policy_curve=False):
    """ Computes the value of the policy defined by predicted effect """

    if np.any(np.isnan(eff_pred)):
        return np.nan, np.nan

    policy = eff_pred>0
    treat_overlap = (policy == a) * (a > 0)
    control_overlap = (policy == a) * (a < 1)

    if np.sum(treat_overlap)==0:
        treat_value = 0
    else:
        treat_value = np.mean(y[treat_overlap])

    if np.sum(control_overlap)==0:
        control_value = 0
    else:
        control_value = np.mean(y[control_overlap])

    pit = np.mean(policy)
    policy_value = pit*treat_value + (1-pit)*control_value

    policy_curve = []

    if compute_policy_curve:
        n = a.shape[0]
        I_sort = np.argsort(-eff_pred)

        n_range = policy_range(n, POL_CURVE_RES)

        for i in n_range:
            I = I_sort[0:i]

            policy_i = 0*policy
            policy_i[I] = 1
            pit_i = np.mean(policy_i)

            treat_overlap = (policy_i > 0) * (a > 0)
            control_overlap = (policy_i < 1) * (a < 1)

            if np.sum(treat_overlap)==0:
                treat_value = 0
            else:
                treat_value = np.mean(y[treat_overlap])

            if np.sum(control_overlap)==0:
                control_value = 0
            else:
                control_value = np.mean(y[control_overlap])

            policy_curve.append(pit_i*treat_value + (1-pit_i)*control_value)

    return policy_value, policy_curve

def pdist2(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*X.dot(Y.T)
    nx = np.sum(np.square(X),1,keepdims=True)
    ny = np.sum(np.square(Y),1,keepdims=True)
    D = (C + ny.T) + nx

    return np.sqrt(D + 1e-8)

def cf_nn(c, a):
    treated_indices = np.array(np.where(a == 1))[0, :]
    control_indices = np.array(np.where(a == 0))[0, :]

    c_control = c[control_indices, :]
    c_treated = c[treated_indices, :]

    D = pdist2(c_control, c_treated)

    nn_treated = control_indices[np.argmin(D, 0)]
    nn_control = treated_indices[np.argmin(D, 1)]

    return nn_treated, nn_control

def pehe_nn(y_pred, y_cf_pred, y, c, a,
            nn_treated=None, nn_control=None):
    if nn_treated is None or nn_control is None:
        nn_treated, nn_control = cf_nn(c, a)

    treated_indices = np.array(np.where(a == 1))[0, :]

    y_cf_treated = 1.0 * y[nn_treated]
    eff_nn_treated = y_cf_treated - 1.0 * y[treated_indices]
    eff_pred_treated = (
        y_cf_pred[treated_indices] - y_pred[treated_indices])

    return np.sqrt(np.mean(np.square(eff_pred_treated - eff_nn_treated)))

def evaluate_bin_att(predictions, data, i_exp, I_subset=None,
                     compute_policy_curve=False,
                     nn_treated=None, nn_control=None):

    c = data['c'][:, :, i_exp]
    a = data['a'][:, i_exp]
    if data['e'] is None:
        e = np.ones_like(a)
    else:
        e = data['e'][:, i_exp]
    y = data['y'][:, i_exp]
    y_pred = predictions[:, 0]
    y_cf_pred = predictions[:, 1]

    att = np.mean(y[a > 0]) - np.mean(y[(1 - a + e) > 1])

    if not I_subset is None:
        c = c[I_subset, :]
        a = a[I_subset]
        e = e[I_subset]
        y_pred = y_pred[I_subset]
        y_cf_pred = y_cf_pred[I_subset]
        y = y[I_subset]

    y_pred_binary = 1.0 * (y_pred > 0.5)

    if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_cf_pred)):
        raise NaNException('NaN encountered')

  

    eff_pred = y_cf_pred - y_pred
    eff_pred[a > 0] = -eff_pred[a > 0]

    ate_pred = np.mean(eff_pred[e > 0])
    atc_pred = np.mean(eff_pred[(1 - a + e) > 1])

    att_pred = np.mean(eff_pred[(a + e) > 1])
    bias_att = att_pred - att

    pehe = np.nan
    bias_ate = np.nan

    if data['HAVE_TRUTH']:
        mu0 = data['mu0'][:, i_exp]
        mu1 = data['mu1'][:, i_exp]
        if not I_subset is None:
            mu0 = mu0[I_subset]
            mu1 = mu1[I_subset]
        eff = mu1 - mu0
        pehe = np.sqrt(np.mean(np.square(eff_pred - eff)))
        bias_ate = ate_pred - np.mean(eff)

    err_fact = np.mean(np.abs(y_pred_binary - y))

    p1_treated = np.mean(y[a > 0])
    p1_treated_pred = np.mean(y_pred[a > 0])

    lpr = np.log(p1_treated / p1_treated_pred + 0.001)

    policy_value, policy_curve = \
        policy_val(a[e > 0], y[e > 0], eff_pred[e > 0], compute_policy_curve)

    pehe_appr = np.nan

    auc = roc_auc_score(y, y_pred)

    return {'ate_pred': ate_pred, 'att_pred': att_pred,
            'bias_att': bias_att, 'atc_pred': atc_pred,
            'err_fact': err_fact, 'lpr': lpr,
            'policy_value': policy_value, 'policy_risk': 1 - policy_value,
            'policy_curve': policy_curve, 'pehe_nn': pehe_appr,
            'pehe': pehe, 'bias_ate': bias_ate, 'auc': auc}
    

def evaluate_cont_ate(predictions, data, i_exp, I_subset=None,
    compute_policy_curve=False, nn_treated=None, nn_control=None):

    c = data['c'][:, :, i_exp]
    a = data['a'][:, i_exp]
    y = data['y'][:, i_exp]
    y_cf = data['y_cf'][:, i_exp]
    mu0 = data['mu0'][:,i_exp]
    mu1 = data['mu1'][:,i_exp]
    y_pred = predictions[:, 0]
    y_cf_pred = predictions[:, 1]

    if not I_subset is None:
        c = c[I_subset,]
        a = a[I_subset]
        y_pred = y_pred[I_subset]
        y_cf_pred = y_cf_pred[I_subset]
        y = y[I_subset]
        y_cf = y_cf[I_subset]
        mu0 = mu0[I_subset]
        mu1 = mu1[I_subset]

    eff = mu1-mu0

    rmse_fact = np.sqrt(np.mean(np.square(y_pred - y)))
    rmse_cfact = np.sqrt(np.mean(np.square(y_cf_pred - y_cf)))

    eff_pred = y_cf_pred - y_pred
    eff_pred[a > 0] = -eff_pred[a > 0]

    ite_pred = y_cf_pred - y
    ite_pred[a > 0] = -ite_pred[a > 0]
    rmse_ite = np.sqrt(np.mean(np.square(ite_pred-eff)))

    ate_pred = np.mean(eff_pred)
    bias_ate = ate_pred-np.mean(eff)

    att_pred = np.mean(eff_pred[a > 0])
    bias_att = att_pred - np.mean(eff[a > 0])

    atc_pred = np.mean(eff_pred[a < 1])
    bias_atc = atc_pred - np.mean(eff[a < 1])

    pehe = np.sqrt(np.mean(np.square(eff_pred-eff)))

    pehe_appr = pehe_nn(
        y_pred, y_cf_pred, y, c, a, nn_treated, nn_control)

    auc = roc_auc_score((y > np.median(y)).astype(int), y_pred)

    return {'ate_pred': ate_pred, 'att_pred': att_pred,
            'atc_pred': atc_pred, 'bias_ate': bias_ate,
            'bias_att': bias_att, 'bias_atc': bias_atc,
            'rmse_fact': rmse_fact, 'rmse_cfact': rmse_cfact,
            'pehe': pehe, 'rmse_ite': rmse_ite, 'pehe_nn': pehe_appr, 'auc': auc}
          

def evaluate_result(result, data, validation=False,
        multiple_exps=False, binary=False):

    predictions = result['pred']

    if validation:
        I_valid = result['val']

    n_units, _, n_rep, n_outputs = predictions.shape

    compute_policy_curve = True

    eval_results = []
 
    for i_out in range(n_outputs):
        eval_results_out = []

        if not multiple_exps and not validation:
            nn_treated, nn_control = cf_nn(
                data['c'][:, :, 0], data['a'][:, 0])


     
        for i_rep in range(n_rep):

            if validation:
                I_valid_rep = I_valid[i_rep,:]
            else:
                I_valid_rep = None

            if multiple_exps:
                i_exp = i_rep
                if validation:
                    nn_treated, nn_control = cf_nn(
                        data['c'][I_valid_rep, :, i_exp],
                        data['a'][I_valid_rep, i_exp])
                else:
                    nn_treated, nn_control = cf_nn(
                        data['c'][:, :, i_exp], data['a'][:, i_exp])
            else:
                i_exp = 0

            if validation and not multiple_exps:
                nn_treated, nn_control = cf_nn(
                    data['c'][I_valid_rep, :, i_exp],
                    data['a'][I_valid_rep, i_exp])

            if binary:
                eval_result = evaluate_bin_att(predictions[:,:,i_rep,i_out],
                    data, i_exp, I_valid_rep, compute_policy_curve,
                    nn_treated=nn_treated, nn_control=nn_control)
            else:
                eval_result = evaluate_cont_ate(predictions[:,:,i_rep,i_out],
                    data, i_exp, I_valid_rep, compute_policy_curve,
                    nn_treated=nn_treated, nn_control=nn_control)

            eval_results_out.append(eval_result)

        eval_results.append(eval_results_out)

  
    eval_dict = {}
    keys = eval_results[0][0].keys()
    for k in keys:
        arr = [[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_rep)]
        v = np.array([[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_rep)])
        eval_dict[k] = v

   
    if 'loss' in result.keys() and result['loss'].shape[1]>=6:
        losses = result['loss']
        n_loss_outputs = losses.shape[0]
       
        if validation:
            objective = np.array([losses[(n_loss_outputs*i)//n_outputs,4,:] for i in range(n_outputs)]).T
        else:
            objective = np.array([losses[(n_loss_outputs*i)//n_outputs,0,:] for i in range(n_outputs)]).T

        eval_dict['objective'] = objective
  
    return eval_dict

def evaluate(output_dir, data_path_train, data_path_test=None, binary=False):

    print ('\nEvaluating experiment %s...' % output_dir)

    
    results = load_results(output_dir)

    if len(results) == 0:
        raise Exception('No finished results found.')


    configs = [r['config'] for r in results]

  
    multiple_exps = (configs[0]['experiments'] > 1)
    if Log.VERBOSE and multiple_exps:
        print ('Multiple data (experiments) detected')

   
    if Log.VERBOSE:
        print ('Loading TRAINING data %s...' % data_path_train)
    data_train = load_data(data_path_train)

   
    if data_path_test is not None:
        if Log.VERBOSE:
            print ('Loading TEST data %s...' % data_path_test)
        data_test = load_data(data_path_test)
    else:
        data_test = None

    # If multiple experiments, tile the data
    if multiple_exps:
        n_experiments = int(configs[0]['experiments'])
        if data_train['c'].ndim == 3 and data_train['c'].shape[2] == 1:
            data_train['c'] = np.tile(data_train['c'], (1, 1, n_experiments))
            data_train['a'] = np.tile(data_train['a'], (1, n_experiments))
            data_train['y'] = np.tile(data_train['y'], (1, n_experiments))
            if data_train['y_cf'] is not None:
                data_train['y_cf'] = np.tile(
                    data_train['y_cf'], (1, n_experiments))
            if 'mu0' in data_train and data_train['mu0'] is not None:
                data_train['mu0'] = np.tile(data_train['mu0'], (1, n_experiments))
            if 'mu1' in data_train and data_train['mu1'] is not None:
                data_train['mu1'] = np.tile(data_train['mu1'], (1, n_experiments))
        if data_test is not None and data_test['c'].ndim == 3 and data_test['c'].shape[2] == 1:
            data_test['c'] = np.tile(data_test['c'], (1, 1, n_experiments))
            data_test['a'] = np.tile(data_test['a'], (1, n_experiments))
            data_test['y'] = np.tile(data_test['y'], (1, n_experiments))
            if data_test['y_cf'] is not None:
                data_test['y_cf'] = np.tile(
                    data_test['y_cf'], (1, n_experiments))
            if 'mu0' in data_test and data_test['mu0'] is not None:
                data_test['mu0'] = np.tile(data_test['mu0'], (1, n_experiments))
            if 'mu1' in data_test and data_test['mu1'] is not None:
                data_test['mu1'] = np.tile(data_test['mu1'], (1, n_experiments))



  
    eval_results = []
    configs_out = []
    i = 0
    if Log.VERBOSE:
        print ('Evaluating result (out of %d): ' % len(results))
    for result in results:
        if Log.VERBOSE:
            print ('Evaluating %d...' % (i+1))

        try:
            eval_train = evaluate_result(result['train'], data_train,
                validation=False, multiple_exps=multiple_exps, binary=binary)

            eval_valid = evaluate_result(result['train'], data_train,
                validation=True, multiple_exps=multiple_exps, binary=binary)

            if data_test is not None:
                eval_test = evaluate_result(result['test'], data_test,
                    validation=False, multiple_exps=multiple_exps, binary=binary)
            else:
                eval_test = None

            eval_results.append({'train': eval_train, 'valid': eval_valid, 'test': eval_test})
            configs_out.append(configs[i])
        except NaNException as e:
            print ('WARNING: Encountered NaN exception. Skipping.')
            print (e)

        i += 1

  
    eval_dict = {'train': {}, 'test': {}, 'valid': {}}
    keys = eval_results[0]['train'].keys()

    def _stack_or_raise(vals, metric_name):
        vals = [np.asarray(v) for v in vals]
        shapes = [v.shape for v in vals]
        if any(s != shapes[0] for s in shapes[1:]):
            raise ValueError(
                "Inconsistent shapes while aggregating evaluation metric '%s': %s. "
                "This usually means the same outdir contains mixed experiment runs (different model/data settings). "
                "Rerun into a fresh outdir (or delete old results) and evaluate again." %
                (metric_name, shapes)
            )
        return np.stack(vals, axis=0)

    for k in keys:
        eval_dict['train'][k] = _stack_or_raise(
            [eval_results[i]['train'][k] for i in range(len(eval_results))],
            'train/%s' % k,
        )

        eval_dict['valid'][k] = _stack_or_raise(
            [eval_results[i]['valid'][k] for i in range(len(eval_results))],
            'valid/%s' % k,
        )

        if eval_test is not None and k in eval_results[0]['test']:
            eval_dict['test'][k] = _stack_or_raise(
                [eval_results[i]['test'][k] for i in range(len(eval_results))],
                'test/%s' % k,
            )

    return eval_dict, configs_out
