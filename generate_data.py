import os

import numpy as np
from scipy.special import expit


NONLINEAR_MIXING_ALPHA = 0.3


def _uniform(low, high, size):
    return np.random.uniform(low, high, size=size)


def _linear_term(features, weights, normalize=False):
    result = features @ weights
    if normalize:
        result = result / np.sqrt(max(1, features.shape[1]))
    return result


def generate(
    p=12,
    n=10000,
    dim_u=2,
    dim_z=4,
    dim_x=4,
    dim_w=4,
    n_copy=10,
    rand_seed=1,
    *,
    confounding_strength=1.0,
    proxy_coef_range=(-2.0, 2.0),
    proxy_noise_std=0.5,
    x_coef_range=(-0.5, 0.5),
    x_noise_std=0.5,
    a_z_coef_range=(-0.5, 0.5),
    a_x_coef_range=(-0.1, 0.1),
    y_w_coef_range=(-0.5, 0.5),
    y_x_coef_range=(-0.1, 0.1),
    u_to_a_range=(0.5, 1.5),
    u_to_y_range=(0.5, 1.5),
    cate_u_coef_range=(0.1, 0.3),
    cate_x_coef_range=(0.0, 0.05),
    y_noise_std=0.05
):
    """Generate synthetic data under proximal causal assumptions."""
    assert p == dim_z + dim_x + dim_w, (
        "p must equal dim_z + dim_x + dim_w")
    assert confounding_strength >= 0, (
        "confounding_strength must be non-negative")

    original_covariates = np.zeros((n, p, n_copy))
    mixed_covariates = np.zeros((n, p, n_copy))
    latent_confounders = np.zeros((n, dim_u, n_copy))
    treatment_proxies = np.zeros((n, dim_z, n_copy))
    observed_confounders = np.zeros((n, dim_x, n_copy))
    outcome_proxies = np.zeros((n, dim_w, n_copy))
    treatments = np.zeros((n, n_copy))
    outcomes = np.zeros((n, n_copy))
    counterfactual_outcomes = np.zeros((n, n_copy))
    mu0 = np.zeros((n, n_copy))
    mu1 = np.zeros((n, n_copy))
    ate_values = []
    att_values = []

    np.random.seed(rand_seed)
    mixing_rng = np.random.RandomState(rand_seed + 104729)

    for copy_index in range(n_copy):
        u = np.random.normal(0, 1, size=(n, dim_u))
        latent_confounders[:, :, copy_index] = u

        u_to_z = _uniform(*proxy_coef_range, size=(dim_u, dim_z))
        z = _linear_term(u, u_to_z, normalize=True) + np.random.normal(
            0, proxy_noise_std, size=(n, dim_z))

        u_to_w = _uniform(*proxy_coef_range, size=(dim_u, dim_w))
        w = _linear_term(u, u_to_w, normalize=True) + np.random.normal(
            0, proxy_noise_std, size=(n, dim_w))

        u_to_x = _uniform(*x_coef_range, size=(dim_u, dim_x))
        x = _linear_term(u, u_to_x, normalize=True) + np.random.normal(
            0, x_noise_std, size=(n, dim_x))

        treatment_proxies[:, :, copy_index] = z
        observed_confounders[:, :, copy_index] = x
        outcome_proxies[:, :, copy_index] = w

        covariates = np.concatenate([z, x, w], axis=1)
        original_covariates[:, :, copy_index] = covariates

        linear_mixing, _ = np.linalg.qr(np.random.randn(p, p))
        nonlinear_mixing, _ = np.linalg.qr(mixing_rng.randn(p, p))
        linearly_mixed = np.dot(covariates, linear_mixing)
        covariates = linearly_mixed + NONLINEAR_MIXING_ALPHA * np.tanh(
            np.dot(linearly_mixed, nonlinear_mixing))
        mixed_covariates[:, :, copy_index] = covariates

        u_to_a = confounding_strength * np.abs(
            _uniform(*u_to_a_range, size=(dim_u, 1)))
        z_to_a = _uniform(*a_z_coef_range, size=(dim_z, 1))
        x_to_a = _uniform(*a_x_coef_range, size=(dim_x, 1))
        treatment_logits = (
            _linear_term(u, u_to_a, normalize=True)
            + _linear_term(z, z_to_a)
            + _linear_term(x, x_to_a)
        ).flatten()
        treatment_probability = expit(
            treatment_logits - np.mean(treatment_logits))
        treatments[:, copy_index] = np.random.binomial(
            1, treatment_probability)

        u_to_y = confounding_strength * np.abs(
            _uniform(*u_to_y_range, size=(dim_u, 1)))
        w_to_y = _uniform(*y_w_coef_range, size=(dim_w, 1))
        x_to_y = _uniform(*y_x_coef_range, size=(dim_x, 1))
        cate_u = _uniform(*cate_u_coef_range, size=(dim_u, 1))
        cate_x = _uniform(*cate_x_coef_range, size=(dim_x, 1))

        baseline_outcome = (
            _linear_term(u, u_to_y, normalize=True)
            + _linear_term(w, w_to_y)
            + _linear_term(x, x_to_y)
        ).flatten()
        treatment_effect = (
            1.0
            + _linear_term(u, cate_u, normalize=True)
            + _linear_term(x, cate_x, normalize=True)
        ).flatten()

        mu0[:, copy_index] = baseline_outcome
        mu1[:, copy_index] = baseline_outcome + treatment_effect

        ate = np.mean(mu1[:, copy_index] - mu0[:, copy_index])
        treated = np.where(treatments[:, copy_index] == 1)
        att = np.mean(
            mu1[treated, copy_index] - mu0[treated, copy_index])
        ate_values.append(ate)
        att_values.append(att)

        noise = np.random.normal(0, y_noise_std, size=n)
        treatment = treatments[:, copy_index]
        outcomes[:, copy_index] = (
            treatment * mu1[:, copy_index]
            + (1 - treatment) * mu0[:, copy_index]
            + noise
        )
        counterfactual_outcomes[:, copy_index] = (
            treatment * mu0[:, copy_index]
            + (1 - treatment) * mu1[:, copy_index]
            + noise
        )

    return {
        "c": mixed_covariates,
        "u": latent_confounders,
        "z": treatment_proxies,
        "x": observed_confounders,
        "w": outcome_proxies,
        "c_original": original_covariates,
        "c_mixed": mixed_covariates,
        "nonlinear_mixing_alpha": np.asarray(NONLINEAR_MIXING_ALPHA),
        "a": treatments,
        "mu0": mu0,
        "mu1": mu1,
        "y": outcomes,
        "y_cf": counterfactual_outcomes,
        "ate": np.mean(ate_values),
        "att": np.mean(att_values),
    }


def split_generated_data(data, train_ratio=0.8, split_seed=2026):
    """Split one generated data set while preserving its shared DGM."""
    num_samples = data["a"].shape[0]
    num_train = int(num_samples * train_ratio)
    indices = np.random.RandomState(split_seed).permutation(num_samples)
    split_indices = {
        "train": indices[:num_train],
        "test": indices[num_train:],
    }

    splits = {}
    for split_name, indices in split_indices.items():
        split = {
            key: value[indices]
            if (isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == num_samples)
            else value
            for key, value in data.items()
        }
        treatment_effect = split["mu1"] - split["mu0"]
        split["ate"] = np.mean(treatment_effect)
        split["att"] = np.mean(treatment_effect[split["a"] == 1])
        splits[split_name] = split

    return splits


if __name__ == "__main__":
    num_samples = 1000
    total_dim = 12
    dim_z = 4
    dim_x = 4
    dim_w = 4
    num_copies = 10

    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "data", "synthetic")
    if os.path.exists(output_dir):
        raise FileExistsError(
            "Refusing to overwrite existing directory: {0}".format(output_dir))
    os.makedirs(output_dir)

    experiment_configs = [
        ("dimension", "1", 1, 1.0),
        ("dimension", "2", 2, 1.0),
        ("dimension", "8", 8, 1.0),
        ("strength", "s", 2, 0.5),
        ("strength", "m", 2, 1.0),
        ("strength", "h", 2, 2.0),
    ]

    for group, setting, dim_u, strength in experiment_configs:
        setting_dir = os.path.join(output_dir, group, setting)
        os.makedirs(setting_dir, exist_ok=True)
        print("Generating {0}/{1}".format(group, setting))

        generated = generate(
            p=total_dim,
            n=num_samples,
            dim_u=dim_u,
            dim_z=dim_z,
            dim_x=dim_x,
            dim_w=dim_w,
            n_copy=num_copies,
            rand_seed=1,
            confounding_strength=strength,
        )
        splits = split_generated_data(
            generated, train_ratio=0.8, split_seed=2026)

        for split_name, split in splits.items():
            output_file = os.path.join(
                setting_dir, "{0}.npz".format(split_name))
            np.savez(output_file, **split)
            print("Saved {0} (n={1})".format(
                output_file, split["a"].shape[0]))
