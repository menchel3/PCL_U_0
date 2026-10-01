import os
import numpy as np

from .logger import Logger as Log

def load_result_file(file):
    arr = np.load(file)

    D = dict([(k, arr[k]) for k in arr.keys()])

    return D

def load_config(cfgfile):
    """ Parses a configuration file """

    cfgf = open(cfgfile,'r')
    cfg = {}
    for l in cfgf:
        ps = [p.strip() for p in l.split(':')]
        if len(ps)==2:
            try:
                cfg[ps[0]] = float(ps[1])
            except ValueError:
                cfg[ps[0]] = ps[1]
                if cfg[ps[0]] == 'False':
                    cfg[ps[0]] = False
                elif cfg[ps[0]] == 'True':
                    cfg[ps[0]] = True
    cfgf.close()
    return cfg

def load_single_result(result_dir):
    if Log.VERBOSE:
        print ('Loading %s...' % result_dir)

    config_path = '%s/config.txt' % result_dir
    has_config = os.path.isfile(config_path)
    if not has_config:
        print ('WARNING: Could not find config.txt for %s. Skipping.' % os.path.basename(result_dir))
        config = None
    else:
        config = load_config(config_path)

    train_path = '%s/result.npz' % result_dir
    test_path = '%s/result.test.npz' % result_dir

    has_test = os.path.isfile(test_path)

    try:
        train_results = load_result_file(train_path)
    except:
        'WARNING: Couldnt load result file. Skipping'
        return None

    if has_test:
        test_results = load_result_file(test_path)
    else:
        test_results = None

    return {'train': train_results, 'test': test_results, 'config': config}

def load_results(output_dir):

    if Log.VERBOSE:
        print ('Loading results from %s...' % output_dir)

    ''' Detect results structure '''
    # Single result
    if os.path.isfile('%s/results.npz' % output_dir):
        #@TODO: Implement
        pass

    # Multiple results
    files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
    exp_dirs = [f for f in files if os.path.isdir(f)
                    if os.path.isfile('%s/result.npz' % f)]

    if Log.VERBOSE:
        print ('Found %d experiment configurations.' % len(exp_dirs))

    # Load each result folder
    results = []
    for dir in exp_dirs:
        dir_result = load_single_result(dir)
        if dir_result is not None:
            results.append(dir_result)

    return results

def load_data(datapath):
    """Load a dataset into the canonical C/A/Y/Y_cf evaluation schema."""
    arr = np.load(datapath)

    def required(canonical_key, legacy_key):
        if canonical_key in arr.files:
            return arr[canonical_key]
        if legacy_key in arr.files:
            return arr[legacy_key]
        raise KeyError(
            "Dataset must contain '%s' (or legacy '%s'): %s" %
            (canonical_key, legacy_key, datapath))

    # Public benchmark files use x/t/yf/ycf. Map those legacy storage keys to
    # C/A/Y/Y_cf here so X remains unambiguously the confounding role inside
    # the PCL-U model.
    covariates = required('c', 'x')
    treatment = required('a', 't')
    outcome = required('y', 'yf')

    HAVE_TRUTH = False
    SPARSE = False

    if len(covariates.shape)==1:
        SPARSE = True

    try:
        es = arr['e']
    except:
        es = None
    try:
        ate = np.mean(arr['ate'])
    except:
        ate = None
    try:
        ymul = arr['ymul'][0,0]
        yadd = arr['yadd'][0,0]
    except:
        ymul = 1
        yadd = 0
    if 'y_cf' in arr.files:
        counterfactual_outcome = arr['y_cf']
    elif 'ycf' in arr.files:
        counterfactual_outcome = arr['ycf']
    else:
        counterfactual_outcome = None

    try:
        mu0s = arr['mu0']
        mu1s = arr['mu1']
    except:
        mu0s = None
        mu1s = None

    if counterfactual_outcome is not None:
        HAVE_TRUTH = True
        if mu0s is None or mu1s is None:
            mu0s = np.zeros_like(outcome)
            mu1s = np.zeros_like(outcome)
            # A=1 -> Y(1)=Y, Y(0)=Y_cf
            mu1s[treatment == 1] = outcome[treatment == 1]
            mu0s[treatment == 1] = counterfactual_outcome[treatment == 1]
            # A=0 -> Y(1)=Y_cf, Y(0)=Y
            mu1s[treatment == 0] = counterfactual_outcome[treatment == 0]
            mu0s[treatment == 0] = outcome[treatment == 0]
    else:
        print ('Couldn\'t find ground truth. Proceeding...')
        counterfactual_outcome = None; mu0s = None; mu1s = None

    data = {'c': covariates, 'a': treatment, 'e': es,
            'y': outcome, 'y_cf': counterfactual_outcome, \
            'mu0':mu0s, 'mu1':mu1s, 'ate':ate, 'YMUL': ymul, \
            'YADD': yadd,  'HAVE_TRUTH': HAVE_TRUTH, \
            'SPARSE': SPARSE}

    return data
