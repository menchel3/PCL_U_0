import numpy as np

def validation_indices(data, val_fraction):
    """Return randomly selected held-out row indices."""
    n = data['c'].shape[0]
    if val_fraction <= 0:
        return []
    n_valid = int(val_fraction * n)
    return np.random.permutation(n)[n - n_valid:]


def log(logfile, message):
    with open(logfile, 'a') as log_file:
        log_file.write(message + '\n')
    print(message)


def save_config(path, flags):
    """Save TensorFlow flags beside a completed run."""
    flag_values = flags.__dict__['__flags']
    contents = '\n'.join(
        '%s: %s' % (key, str(flag_values[key]))
        for key in sorted(flag_values))
    with open(path, 'w') as config_file:
        config_file.write(contents)


def load_data(path):
    """Load NPZ data into the canonical C/A/Y/Y_cf training schema."""
    if not path.endswith('.npz'):
        raise ValueError('PCL-U training supports NPZ datasets only: %s' % path)

    source = np.load(path)

    def required(canonical_key, legacy_key):
        if canonical_key in source.files:
            return source[canonical_key]
        if legacy_key in source.files:
            return source[legacy_key]
        raise KeyError(
            "Dataset must contain '%s' (or legacy '%s'): %s" %
            (canonical_key, legacy_key, path))

    # Public benchmark files use x/t/yf/ycf. At this loading boundary only,
    # map x to the combined observed input C, t to treatment A, and yf to Y.
    covariates = required('c', 'x')
    treatment = required('a', 't')
    outcome = required('y', 'yf')
    if 'y_cf' in source.files:
        counterfactual_outcome = source['y_cf']
    elif 'ycf' in source.files:
        counterfactual_outcome = source['ycf']
    else:
        counterfactual_outcome = None

    data = {
        'c': covariates,
        'a': treatment,
        'y': outcome,
        'y_cf': counterfactual_outcome,
    }
    data['HAVE_TRUTH'] = data['y_cf'] is not None
    data['dim'] = data['c'].shape[1]
    data['n'] = data['c'].shape[0]
    return data




