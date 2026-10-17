#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
CONDA_ROOT=${CONDA_ROOT:-"$HOME/miniforge3"}
ENV_PREFIX="$CONDA_ROOT/envs/nf-beng"

if [ "$#" -ne 0 ]; then
    echo "Usage: bash setup-macOS.sh (optional: CONDA_ROOT=/path/to/miniforge3)"
    exit 2
fi
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    echo "Miniforge not found at $CONDA_ROOT. Set CONDA_ROOT to its installation directory."
    exit 2
fi
if [ ! -d "$ENV_PREFIX/conda-meta" ]; then
    "$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" python=3.12 pip
fi
PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "nf-beng must use Python 3.12"'
"$PYTHON" -m pip install -r "$REPO_ROOT/requirements.txt"
"$PYTHON" -m pip check
"$PYTHON" -m ipykernel install --user --name nf-beng --display-name "Python (nf-beng)"
cd "$REPO_ROOT"
"$PYTHON" -c 'from resources import aerofoils, cases, data, nnetwork, saved; print("Project imports: OK")'
echo "Ready. Select nf-beng in VS Code and in the notebook kernel picker."
echo "Terminal activation: conda activate nf-beng"
