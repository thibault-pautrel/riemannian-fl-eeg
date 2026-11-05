#!/bin/bash
# One-off setup on a MUST user interface (lappusmb / lappui), or any Linux machine.
#
#   bash condor/setup_must.sh            # virtual environment + fast tests
#   bash condor/setup_must.sh --no-tests
#
# Creates $RFL_VENV (see condor/env.sh) with Python 3.11 or 3.12, installs PyTorch and
# requirements.txt, then runs the fast test suite. Run it on the user interface: the
# worker nodes reuse the environment through the shared filesystem.
#
# Python. yetanotherspdnet needs Python >= 3.11. The script tries $PYTHON, then
# python3.12, python3.11 and python3. If none is recent enough, install Miniforge on the
# shared filesystem and point PYTHON at its interpreter, for example:
#   bash Miniforge3-Linux-x86_64.sh -b -p /uds_data/<lab>/<user>/miniforge3
#   /uds_data/<lab>/<user>/miniforge3/bin/conda create -y -p /uds_data/<lab>/<user>/py312 python=3.12
#   PYTHON=/uds_data/<lab>/<user>/py312/bin/python bash condor/setup_must.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
TESTS=1; [ "${1:-}" = "--no-tests" ] && TESTS=""

pick_python() {
    for c in "${PYTHON:-}" python3.12 python3.11 python3; do
        [ -n "$c" ] || continue
        command -v "$c" >/dev/null 2>&1 || continue
        if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
            echo "$c"; return 0
        fi
    done
    return 1
}

if [ ! -x "$RFL_VENV/bin/python" ]; then
    PY="$(pick_python)" || { echo "FATAL: no Python >= 3.11 found (see the header of $0)." >&2; exit 1; }
    echo "Creating $RFL_VENV with $PY ($("$PY" --version))"
    "$PY" -m venv "$RFL_VENV"
fi
# shellcheck disable=SC1091
source "$RFL_VENV/bin/activate"
# pip caches its downloads (PyTorch and its CUDA libraries: several GB) next to the
# repository rather than in the small home directory.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "$RFL_REPO")/.cache/pip}"
python -m pip install --upgrade pip
# The default PyPI wheel of torch on Linux runs on CPU and on NVIDIA GPUs.
python -m pip install torch
python -m pip install -r "$RFL_REPO/requirements-dev.txt"
# The repository itself, editable: its packages import from any folder.
python -m pip install -e "$RFL_REPO" --no-deps

python - <<'PY'
import sys, torch, yetanotherspdnet, moabb
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"cuda available: {torch.cuda.is_available()}  moabb {moabb.__version__}")
PY

if [ -n "$TESTS" ]; then
    cd "$RFL_REPO"
    python -m pytest -q -m "not slow"
fi
echo "Setup done. Next: bash condor/fetch_data.sh BNCI2014_001, then bash condor/submit.sh smoke"
