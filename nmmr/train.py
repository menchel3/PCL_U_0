#!/usr/bin/env python
"""Train NMMR on the oracle Z/W/X blocks from the PCL-U synthetic data."""

import argparse
import copy
import csv
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from nmmr.kernels import calculate_kernel_matrix
from nmmr.loss import NMMR_loss
from nmmr.model import MLP_for_NMMR


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "syn_strength_s.json"
SELECTIONS = ("final", "best_val_nmmr")
KEY_ALIASES = {
    "u": ("u", "u_true"),
    "z": ("z", "z_true"),
    "x": ("x", "x_true"),
    "w": ("w", "w_true"),
    "a": ("a", "t"),
    "y": ("y", "yf"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run original NMMR with true synthetic Z/W/X proxy blocks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, help="Override n_epochs from the config.")
    parser.add_argument("--repeats", type=int, help="Override n_repeat from the config.")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run one realization for two epochs in a separate output directory.")
    return parser.parse_args()


def load_config(args):
    with args.config.resolve().open() as config_file:
        config = json.load(config_file)
    if args.epochs is not None:
        config["model"]["n_epochs"] = args.epochs
    if args.repeats is not None:
        config["data"]["n_repeat"] = args.repeats
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.smoke:
        config["experiment_name"] += "_smoke"
        config["data"]["n_repeat"] = 1
        config["model"]["n_epochs"] = 2
        config["model"]["validation_interval"] = 1
    return config


def realization(array, index, vector=False):
    if vector:
        result = array[:, index]
        return result.astype(np.float32).reshape(-1, 1)
    return array[:, :, index].astype(np.float32)


def archive_array(archive, canonical_key):
    for key in KEY_ALIASES[canonical_key]:
        if key in archive.files:
            return archive[key]
    raise KeyError(
        "archive is missing {} (accepted keys: {})".format(
            canonical_key, ", ".join(KEY_ALIASES[canonical_key])))


def validate_archive(archive, path, data_config):
    required = ("u", "z", "x", "w", "a", "y")
    missing = [key for key in required
               if not any(alias in archive.files for alias in KEY_ALIASES[key])]
    missing.extend(
        key for key in ("mu0", "mu1") if key not in archive.files)
    if missing:
        raise ValueError("{} is missing keys: {}".format(path, ", ".join(missing)))
    expected = {
        "u": data_config["dim_u"],
        "z": data_config["dim_z"],
        "x": data_config["dim_x"],
        "w": data_config["dim_w"],
    }
    for key, dimension in expected.items():
        array = archive_array(archive, key)
        if array.ndim != 3 or array.shape[1] != dimension:
            raise ValueError(
                "{} expected {} dimension {}, got shape {}".format(
                    path, key, dimension, array.shape))


def read_realization(path, index, data_config):
    with np.load(path) as archive:
        validate_archive(archive, path, data_config)
        treatment = archive_array(archive, "a")
        if index >= treatment.shape[1]:
            raise IndexError(
                "realization {} requested, but {} only contains {} realizations".format(
                    index, path, treatment.shape[1]))
        return {
            "u": realization(archive_array(archive, "u"), index),
            "z": realization(archive_array(archive, "z"), index),
            "x": realization(archive_array(archive, "x"), index),
            "w": realization(archive_array(archive, "w"), index),
            "a": realization(treatment, index, vector=True),
            "y": realization(archive_array(archive, "y"), index, vector=True),
            "mu0": realization(archive["mu0"], index, vector=True),
            "mu1": realization(archive["mu1"], index, vector=True),
        }


def subset(data, indices):
    return {key: value[indices] for key, value in data.items()}


def load_data(config, realization_index):
    data_config = config["data"]
    train_path = Path(data_config["train_path"])
    test_path = Path(data_config["test_path"])
    if not train_path.is_absolute():
        train_path = PROJECT_DIR / train_path
    if not test_path.is_absolute():
        test_path = PROJECT_DIR / test_path
    train_all = read_realization(
        train_path, realization_index, data_config)
    test = read_realization(
        test_path, realization_index, data_config)

    n_samples = train_all["a"].shape[0]
    n_valid = int(round(n_samples * data_config["validation_fraction"]))
    if n_valid <= 0 or n_valid >= n_samples:
        raise ValueError("validation_fraction must leave non-empty train and validation sets")
    rng = np.random.RandomState(data_config["split_seed"] + realization_index)
    indices = rng.permutation(n_samples)
    valid_indices = indices[:n_valid]
    train_indices = indices[n_valid:]
    return subset(train_all, train_indices), subset(train_all, valid_indices), test


def to_tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def model_inputs(data, device, treatment=None):
    a = data["a"] if treatment is None else np.full_like(data["a"], treatment)
    return torch.cat(
        (to_tensor(a, device), to_tensor(data["w"], device), to_tensor(data["x"], device)),
        dim=1)


def kernel_inputs(data, device):
    return torch.cat(
        (to_tensor(data["a"], device), to_tensor(data["z"], device), to_tensor(data["x"], device)),
        dim=1)


def effect_metrics(model, data, device):
    with torch.no_grad():
        pred0 = model(model_inputs(data, device, treatment=0.0)).cpu().numpy()
        pred1 = model(model_inputs(data, device, treatment=1.0)).cpu().numpy()
        factual = model(model_inputs(data, device)).cpu().numpy()
    effect_pred = pred1 - pred0
    effect_true = data["mu1"] - data["mu0"]
    ate_pred = float(np.mean(effect_pred))
    ate_true = float(np.mean(effect_true))
    return {
        "pehe": float(np.sqrt(np.mean(np.square(effect_pred - effect_true)))),
        "ate_pred": ate_pred,
        "ate_true": ate_true,
        "bias_ate": float(abs(ate_pred - ate_true)),
        "signed_ate_error": float(ate_pred - ate_true),
        "factual_rmse": float(np.sqrt(np.mean(np.square(factual - data["y"])))),
    }


def validation_metrics(model, data, kernel, loss_name, device):
    with torch.no_grad():
        prediction = model(model_inputs(data, device))
        target = to_tensor(data["y"], device)
        nmmr = float(NMMR_loss(prediction, target, kernel, loss_name).cpu())
    metrics = effect_metrics(model, data, device)
    metrics["nmmr"] = nmmr
    return metrics


def clone_state(model):
    return copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})


def resolve_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def train_one(config, realization_index, output_dir, device):
    runtime = config["runtime"]
    model_config = config["model"]
    seed = runtime["seed"] + realization_index
    np.random.seed(seed)
    torch.manual_seed(seed)

    train, valid, test = load_data(config, realization_index)
    input_dimension = 1 + config["data"]["dim_w"] + config["data"]["dim_x"]
    model = MLP_for_NMMR(input_dim=input_dimension, train_params=model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_config["learning_rate"],
        weight_decay=model_config["l2_penalty"])

    train_x = model_inputs(train, device)
    train_y = to_tensor(train["y"], device)
    kernel_length_scale = model_config["kernel_length_scale"]
    train_kernel = calculate_kernel_matrix(
        kernel_inputs(train, device), length_scale=kernel_length_scale)
    valid_kernel = calculate_kernel_matrix(
        kernel_inputs(valid, device), length_scale=kernel_length_scale)
    loss_name = model_config["loss_name"]

    best_values = {
        "best_val_nmmr": math.inf,
    }
    best_states = {}
    best_epochs = {}
    history = []
    n_epochs = model_config["n_epochs"]
    validation_interval = model_config["validation_interval"]

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(train_x)
        train_loss = NMMR_loss(prediction, train_y, train_kernel, loss_name)
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == n_epochs or epoch % validation_interval == 0:
            model.eval()
            val = validation_metrics(model, valid, valid_kernel, loss_name, device)
            history.append({"epoch": epoch, "train_nmmr": float(train_loss.detach().cpu()), **val})
            candidates = {
                "best_val_nmmr": val["nmmr"],
            }
            for name, value in candidates.items():
                if value < best_values[name]:
                    best_values[name] = value
                    best_states[name] = clone_state(model)
                    best_epochs[name] = epoch

    states = {"final": clone_state(model), **best_states}
    epochs = {"final": n_epochs, **best_epochs}
    results = {}
    for name in SELECTIONS:
        model.load_state_dict(states[name])
        model.eval()
        results[name] = {
            "selected_epoch": epochs[name],
            "validation": validation_metrics(model, valid, valid_kernel, loss_name, device),
            "test": effect_metrics(model, test, device),
        }

    realization_dir = output_dir / "realization_{:02d}".format(realization_index)
    realization_dir.mkdir()
    torch.save(states["final"], realization_dir / "final.pt")
    torch.save(states["best_val_nmmr"], realization_dir / "best_val_nmmr.pt")
    with (realization_dir / "metrics.json").open("w") as metrics_file:
        json.dump(results, metrics_file, indent=2)
    with (realization_dir / "history.csv").open("w", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return results, train["a"].shape[0], valid["a"].shape[0], test["a"].shape[0]


def aggregate(all_results):
    summary = {}
    for selection in SELECTIONS:
        summary[selection] = {}
        for metric in ("pehe", "bias_ate", "signed_ate_error", "factual_rmse"):
            values = np.array(
                [result[selection]["test"][metric] for result in all_results], dtype=float)
            summary[selection][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        epochs = np.array(
            [result[selection]["selected_epoch"] for result in all_results], dtype=float)
        summary[selection]["selected_epoch"] = {
            "mean": float(np.mean(epochs)),
            "std": float(np.std(epochs, ddof=1)) if len(epochs) > 1 else 0.0,
        }
    return summary


def write_summary(output_dir, config, all_results, sizes, device):
    summary = aggregate(all_results)
    payload = {
        "experiment": config["experiment_name"],
        "device": str(device),
        "n_repeat": len(all_results),
        "sample_sizes": {"train": sizes[0], "valid": sizes[1], "test": sizes[2]},
        "inputs": {
            "bridge": "[A, W, X]",
            "kernel": "[A, Z, X]",
            "kernel_length_scale": config["model"]["kernel_length_scale"],
        },
        "summary": summary,
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(payload, summary_file, indent=2)

    lines = [
        config["experiment_name"],
        "NMMR oracle: bridge=[A,W,X], kernel=[A,Z,X]",
        "repeats={}, train/valid/test={}/{}/{}, device={}".format(
            len(all_results), sizes[0], sizes[1], sizes[2], device),
        "",
        "selection                    PEHE mean +/- std     |ATE error| mean +/- std   epoch mean +/- std",
    ]
    for selection in SELECTIONS:
        values = summary[selection]
        lines.append(
            "{:<28} {:>7.4f} +/- {:<7.4f}  {:>7.4f} +/- {:<7.4f}       {:>7.1f} +/- {:.1f}".format(
                selection,
                values["pehe"]["mean"], values["pehe"]["std"],
                values["bias_ate"]["mean"], values["bias_ate"]["std"],
                values["selected_epoch"]["mean"], values["selected_epoch"]["std"]))
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    return summary


def main():
    args = parse_args()
    config = load_config(args)
    torch.set_num_threads(config["runtime"].get("num_threads", 1))
    device = resolve_device(config["runtime"].get("device", "auto"))

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_DIR / "results" / "{}_{}".format(
            config["experiment_name"], timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "config.json").open("w") as config_file:
        json.dump(config, config_file, indent=2)

    n_repeat = config["data"]["n_repeat"]
    print("PCL-U NMMR oracle experiment")
    print("bridge=[A,W,X], kernel=[A,Z,X]")
    print("device={}, repeats={}, epochs={}".format(
        device, n_repeat, config["model"]["n_epochs"]))
    print("results={}".format(output_dir))

    all_results = []
    sizes = None
    for realization_index in range(n_repeat):
        results, n_train, n_valid, n_test = train_one(
            config, realization_index, output_dir, device)
        sizes = (n_train, n_valid, n_test)
        all_results.append(results)
        selected = results["best_val_nmmr"]
        print(
            "[{}/{}] best-val-NMMR epoch={} test PEHE={:.4f}, |ATE error|={:.4f}".format(
                realization_index + 1, n_repeat, selected["selected_epoch"],
                selected["test"]["pehe"], selected["test"]["bias_ate"]))

    summary = write_summary(output_dir, config, all_results, sizes, device)
    print((output_dir / "summary.txt").read_text(), end="")
    print("[SUCCESS] Results saved to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
