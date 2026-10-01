#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

configs=(
  configs/syn_dim_1.json
  configs/syn_dim_2.json
  configs/syn_dim_8.json
  configs/syn_strength_s.json
  configs/syn_strength_m.json
  configs/syn_strength_h.json
)

for config in "${configs[@]}"; do
  python -u train.py --config "${config}"
done
