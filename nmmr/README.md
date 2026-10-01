# NMMR on PCL-U Synthetic Data

This experiment runs NMMR on the PCL-U synthetic data. The outcome bridge uses
`[A, W, X]`, the kernel uses `[A, Z, X]`, and the unmeasured confounder `U` is
not given to the model.

## Environment

```bash
conda env create -f environment.yaml
conda activate NMMR
```

## Run

Run all six experiments:

```bash
bash run_all.sh
```

Run one experiment, for example:

```bash
python train.py --config configs/syn_strength_m.json
```

