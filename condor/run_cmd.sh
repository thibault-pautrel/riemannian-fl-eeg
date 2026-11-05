#!/bin/bash
# Run one command of a study DAG on a worker (a Command step of
# rfl/experiments/pipeline.py, for example the ReEig floors), in the job environment.
#
#   condor/run_cmd.sh <command and arguments...>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-1}"
[ -n "${_CONDOR_SCRATCH_DIR:-}" ] && export TMPDIR="$_CONDOR_SCRATCH_DIR"
cd "${RFL_WORKDIR:-.}" || exit 3
echo "RUN  $*  on $(hostname)  $(date -Is)"
"$@"
rc=$?
echo "END  rc=$rc  $(date -Is)"
exit "$rc"
