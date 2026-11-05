#!/bin/bash
# Submit the pending jobs of one stage of a plan to HTCondor.
#
#   bash condor/submit.sh <plan> [stage] [options]
#
# Options:
#   --cpus N           cores per job (default 4)
#   --memory SIZE      memory per job (default 8GB; long private runs may need 16GB)
#   --disk SIZE        local disk per job (default 1GB)
#   --gpu [TYPE]       request one GPU, optionally of a type (v100, t4, a100, ...)
#   --short | --long   duration class on MUST (2 days, 8 days; default 4 days, and 8
#                      days for GPU jobs)
#   --group NAME       accounting group, e.g. group_usmb.listic
#   --retry-failed     also resubmit the failed, interrupted and queued jobs. Use it
#                      only when condor_q shows no job of this plan.
#   --dry-run          write the jobs file and print the submit command only
#
# Examples:
#   bash condor/submit.sh smoke
#   bash condor/submit.sh mi_protocol sweep --memory 16GB
#   bash condor/submit.sh mi_protocol final --gpu t4
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

[ "$#" -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }
PLAN="$1"; shift
STAGE=""
if [ "$#" -ge 1 ] && [[ "$1" != --* ]]; then STAGE="$1"; shift; fi
# Jobs run in this folder, so relative paths of a plan mean what they mean here.
WORKDIR="$(pwd -P)"

CPUS=4; MEMORY=8GB; DISK=1GB; GPU=""; GPU_TYPE=""; DURATION=""; GROUP=""
RETRY=""; DRY=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpus)   CPUS="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --disk)   DISK="$2"; shift 2 ;;
        --gpu)    GPU=1; if [ "$#" -ge 2 ] && [[ "$2" != --* ]]; then GPU_TYPE="$2"; shift; fi; shift ;;
        --short)  DURATION="+isShortJob = True"; shift ;;
        --long)   DURATION="+isLongJob = True"; shift ;;
        --group)  GROUP="$2"; shift 2 ;;
        --retry-failed) RETRY="--retry-failed"; shift ;;
        --dry-run) DRY=1; shift ;;
        --stage)  STAGE="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

STAGE_ARG=(); [ -n "$STAGE" ] && STAGE_ARG=(--stage "$STAGE")
# Results folder of the plan (its NAME, for example mi_BNCI2014_001 for mi_protocol).
NS=$(python -c "import sys; from rfl.experiments.grid import plan_namespace; print(plan_namespace(sys.argv[1]))" "$PLAN")
TAG="${NS}${STAGE:+_$STAGE}"
JOBS_DIR="$RFL_RESULTS/condor/$TAG"
LOGDIR="$JOBS_DIR/logs"
mkdir -p "$LOGDIR"
JOBS_FILE="$JOBS_DIR/jobs_$(date +%Y%m%d_%H%M%S).txt"

python -m rfl.experiments.grid "$PLAN" "${STAGE_ARG[@]}" --list $RETRY > "$JOBS_FILE" || { rm -f "$JOBS_FILE"; exit 1; }
N=$(grep -c . "$JOBS_FILE" || true)
if [ "$N" -eq 0 ]; then
    echo "Nothing to submit: every job of $TAG is done or already queued (see --status; --retry-failed resubmits failed and interrupted jobs)."
    rm -f "$JOBS_FILE"
    exit 0
fi

for v in "$RFL_REPO" "$RFL_VENV" "$RFL_RESULTS" "$MNE_DATA" "$TMPDIR" "$WORKDIR"; do
    case "$v" in *[[:space:]]*) echo "paths may not contain spaces: '$v'" >&2; exit 2 ;; esac
done
APPEND=(-a "repo = $RFL_REPO" -a "logdir = $LOGDIR" -a "cpus = $CPUS"
        -a "memory = $MEMORY" -a "disk = $DISK" -a "venv = $RFL_VENV"
        -a "results = $RFL_RESULTS" -a "mnedata = $MNE_DATA" -a "tmpdir = $TMPDIR"
        -a "workdir = $WORKDIR")
if [ -n "$GPU" ]; then
    APPEND+=(-a "request_gpus = 1")
    [ -n "$GPU_TYPE" ] && APPEND+=(-a "+wantGpuType = \"$GPU_TYPE\"")
fi
[ -n "$DURATION" ] && APPEND+=(-a "$DURATION")
[ -n "$GROUP" ] && APPEND+=(-a "+wishedAcctGroup = \"$GROUP\"")

echo "$N jobs of $TAG  ->  $JOBS_FILE"
echo "logs: $LOGDIR    results: $RFL_RESULTS/$NS"
if [ -n "$DRY" ]; then
    printf 'condor_submit %s' "$HERE/rfl.sub"; printf ' %q' "${APPEND[@]}"
    printf ' -queue %q\n' "job from $JOBS_FILE"
    exit 0
fi
condor_submit "$HERE/rfl.sub" "${APPEND[@]}" -queue "job from $JOBS_FILE"
# Mark the submitted jobs, so that a second submit.sh does not submit them again.
while read -r dir _; do
    mkdir -p "$dir" && touch "$dir/queued"
done < "$JOBS_FILE"
