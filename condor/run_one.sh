#!/bin/bash
# Run ONE experiment: what each HTCondor job executes.
#
#   condor/run_one.sh <save_dir> <entry> <runner arguments...>
#
# <entry> is 'fed' (rfl/experiments/run_federated_experiment.py), 'tslr'
# (rfl/experiments/run_tslr.py) or 'central'
# (rfl/experiments/run_centralized_experiment.py). The lines of a jobs file, written by
# `python -m rfl.experiments.grid <plan> --list`, have exactly this format.
#
# A run whose result file exists is skipped, so resubmitting a plan only fills the gaps.
# The output goes through rfl/utils/log_filter.py, which removes the tensors printed by
# yetanotherspdnet's batch norm at every step (RFL_RAW_LOG=1 keeps them).
# FORCE=1 reruns it (the old result is deleted first). The run writes
# <save_dir>/job_args (the command, checked by rfl/experiments/grid.py), <save_dir>/run.log
# and <save_dir>/exit_code, and removes the 'queued' marker written by submit.sh.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <save_dir> <fed|central|tslr> [runner arguments...]" >&2
    exit 2
fi
SAVE_DIR="$1"; ENTRY="$2"; shift 2

case "$ENTRY" in
    fed)     MODULE=rfl.experiments.run_federated_experiment;  RESULT=fl_results.json ;;
    central) MODULE=rfl.experiments.run_centralized_experiment; RESULT=centralized_results.json ;;
    tslr)    MODULE=rfl.experiments.run_tslr;                  RESULT=fl_results.json ;;
    *)       echo "unknown entry '$ENTRY' (fed, central or tslr)" >&2; exit 2 ;;
esac

COMMAND="$ENTRY $*"
if [ -z "${FORCE:-}" ] && [ -f "$SAVE_DIR/job_args" ] \
        && [ "$(cat "$SAVE_DIR/job_args")" != "$COMMAND" ]; then
    echo "FATAL: $SAVE_DIR holds a run of another command (job_args). FORCE=1 overwrites it." >&2
    exit 4
fi
# Done = result file and exit code 0 (or none yet), as experiments/grid.py counts it.
if [ -z "${FORCE:-}" ] && [ -f "$SAVE_DIR/$RESULT" ] \
        && { [ ! -f "$SAVE_DIR/exit_code" ] || [ "$(cat "$SAVE_DIR/exit_code")" = "0" ]; }; then
    echo "SKIP (done) $SAVE_DIR"
    rm -f "$SAVE_DIR/queued"
    exit 0
fi
command -v python >/dev/null || { echo "FATAL: no python; run condor/setup_must.sh" >&2; exit 3; }

mkdir -p "$SAVE_DIR"
rm -f "$SAVE_DIR/exit_code" "$SAVE_DIR/$RESULT" "$SAVE_DIR/queued"
# Attempts of this run: after RFL_MAX_ATTEMPTS (3) failures, rfl/experiments/grid.py
# leaves it out ('abandoned'). FORCE=1 starts the count again.
[ -n "${FORCE:-}" ] && rm -f "$SAVE_DIR/attempts"
echo $(( $(cat "$SAVE_DIR/attempts" 2>/dev/null || echo 0) + 1 )) > "$SAVE_DIR/attempts"
echo "$COMMAND" > "$SAVE_DIR/job_args"
# One thread per requested core (set by the submit file), never the whole node.
# The nodes have no network. MOABB then reads the files of each dataset's own source
# rather than asking NEMAR first (about 40 s lost per subject offline). Checked offline
# for the five datasets. The trials usually come from the cache (rfl/data/cache.py).
export MOABB_DOWNLOAD_PROVIDER="${MOABB_DOWNLOAD_PROVIDER:-upstream}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Temporary files on the node's local scratch when HTCondor provides one.
[ -n "${_CONDOR_SCRATCH_DIR:-}" ] && export TMPDIR="$_CONDOR_SCRATCH_DIR"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "RUN  $SAVE_DIR  on $(hostname)  $(date -Is)  threads=$OMP_NUM_THREADS"
# The folder the plan was submitted from (relative paths of the plan), else here.
cd "${RFL_WORKDIR:-.}" || { echo "FATAL: cd ${RFL_WORKDIR:-.}" >&2; exit 3; }
if [ -n "${RFL_RAW_LOG:-}" ]; then
    python -m "$MODULE" "$@" --save-dir "$SAVE_DIR" > "$SAVE_DIR/run.log" 2>&1
    rc=$?
else
    python -m "$MODULE" "$@" --save-dir "$SAVE_DIR" 2>&1 \
        | python -m rfl.utils.log_filter > "$SAVE_DIR/run.log"
    rc=${PIPESTATUS[0]}
fi
echo "$rc" > "$SAVE_DIR/exit_code"
echo "END  rc=$rc  $(date -Is)"
[ "$rc" -ne 0 ] && tail -n 30 "$SAVE_DIR/run.log" >&2
# In a DAG the POST script reads exit_code: exit 0, so that DAGMan does not remove the
# other jobs of the node at the first failure.
[ -n "${RFL_DAG:-}" ] && exit 0
exit "$rc"
