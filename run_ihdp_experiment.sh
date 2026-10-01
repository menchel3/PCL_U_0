#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python_bin="${PCL_U_PYTHON:-python}"
mpl_config_dir="${MPLCONFIGDIR:-/tmp/pcl-u-matplotlib}"
mkdir -p "${mpl_config_dir}"
export MPLCONFIGDIR="${mpl_config_dir}"
config="configs/ihdp.txt"
mode="${1:-all}"

case "${mode}" in
  train) "${python_bin}" grid_search.py "${config}" ;;
  evaluate) MPLBACKEND=Agg "${python_bin}" evaluate.py "${config}" 1 ;;
  all)
    "${python_bin}" grid_search.py "${config}"
    MPLBACKEND=Agg "${python_bin}" evaluate.py "${config}" 1
    ;;
  *) echo "Usage: $0 {train|evaluate|all}" >&2; exit 2 ;;
esac
