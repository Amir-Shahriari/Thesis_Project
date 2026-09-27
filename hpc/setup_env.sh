#!/bin/bash
# One-time environment bootstrap for NCI Gadi. Run on a LOGIN node -- compute
# nodes have no outbound internet, so every package must be installed here first.
#
#   bash hpc/setup_env.sh
#
# Creates a venv under $PROJECT_ROOT/.venv-gadi. Put the repo on /scratch or
# /g/data, never in /home: /home is ~10 GB and is not mounted the same way.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${PROJECT_ROOT}/.venv-gadi"

echo "Repo:  ${PROJECT_ROOT}"
case "${PROJECT_ROOT}" in
  /home/*) echo "WARNING: repo is under /home (small quota, and /home is not in"
           echo "         a job's -l storage list). Move it to /scratch or /g/data." ;;
esac

# Pick a python3 module. Versions change; list them with `module avail python3`
# and override with PYTHON_MODULE=python3/3.x.y if this default is gone.
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.7}"
module purge
module load "${PYTHON_MODULE}"
echo "Loaded ${PYTHON_MODULE} -> $(python3 --version)"

if [ ! -d "${VENV}" ]; then
    python3 -m venv "${VENV}"
    echo "Created ${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel

# CPU wheels: this workload is dominated by the CPU-bound QAOA oracle and by
# small-graph message passing, neither of which benefits much from a GPU.
# Swap the index-url for a CUDA build only if you have profiled a GPU win.
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install torch_geometric
python -m pip install numpy scipy pandas networkx pyyaml
python -m pip install qiskit qiskit-aer      # required by the faithful QAOA oracle
python -m pip install -e "${PROJECT_ROOT}"   # installs the qwarm package

echo
echo "Verifying..."
python - <<'PY'
import torch, torch_geometric, networkx, scipy, numpy
print("torch          ", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
try:
    import qiskit_aer
    print("qiskit-aer     ", qiskit_aer.__version__)
except ImportError:
    print("qiskit-aer      MISSING -- the full oracle pool will fail; use "
          "--oracle-pool classical_only")
import qwarm
print("qwarm           importable")
PY

echo
echo "Done. In a PBS script, activate with:"
echo "    module load ${PYTHON_MODULE}"
echo "    source ${VENV}/bin/activate"
