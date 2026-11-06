# Environment of the jobs, sourced by every script of condor/.
#
# Edit the three paths below once, for your account. They must be on a filesystem that
# the worker nodes see (on MUST: /uds_data/... or /lapp_data/...; home directories are
# not meant for data). Every variable can also be set before sourcing this file.

# The clone of this repository (default: the parent of this condor/ folder).
export RFL_REPO="${RFL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# The Python virtual environment, created by condor/setup_must.sh.
export RFL_VENV="${RFL_VENV:-$RFL_REPO/.venv}"
# Where the runs write their results (one folder per run).
export RFL_RESULTS="${RFL_RESULTS:-$RFL_REPO/results}"
# The MOABB / MNE data folder (several GB), shared by all runs.
export MNE_DATA="${MNE_DATA:-$(dirname "$RFL_REPO")/mne_data}"

# Use the real path, not a symbolic link: every node then sees the same folder. The
# Python code also points each MOABB dataset key at this folder (one key has a hyphen
# that `export` cannot set) and ignores ~/.mne/mne-python.json (rfl/data/registry.py).
mkdir -p "$MNE_DATA" 2>/dev/null || true
MNE_DATA="$(cd "$MNE_DATA" 2>/dev/null && pwd -P || echo "$MNE_DATA")"
export MNE_DATA

# An absolute results folder, so that a relative RFL_RESULTS names the same folder on
# every node (rfl/experiments/grid.py resolves it the same way).
mkdir -p "$RFL_RESULTS" 2>/dev/null || true
RFL_RESULTS="$(cd "$RFL_RESULTS" 2>/dev/null && pwd -P || echo "$RFL_RESULTS")"
export RFL_RESULTS

# Temporary files on the shared filesystem rather than a small /tmp.
export TMPDIR="${RFL_TMPDIR:-$RFL_RESULTS/.tmp}"
mkdir -p "$TMPDIR" 2>/dev/null || true

# MOABB reads the files of each dataset's own source. With the default ("auto"), the
# datasets that have a NEMAR mirror ask NEMAR first, for every subject: about 40 s lost
# per subject without network (the nodes), and a second download of the whole dataset
# on the user interface when only the source files are on disk. For a first download
# from NEMAR: MOABB_DOWNLOAD_PROVIDER=auto bash condor/fetch_data.sh <dataset>.
export MOABB_DOWNLOAD_PROVIDER="${MOABB_DOWNLOAD_PROVIDER:-upstream}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

if [ -f "$RFL_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$RFL_VENV/bin/activate"
fi
