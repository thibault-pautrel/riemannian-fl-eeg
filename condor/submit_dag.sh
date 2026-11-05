#!/bin/bash
# Run a whole study on HTCondor: one DAG chains its steps, the selections and the grid
# extensions (rfl/experiments/pipeline.py). Nothing to do by hand until the reports.
#
#   bash condor/submit_dag.sh <plan> [options]
#
# Options: --cpus N, --memory SIZE, --disk SIZE, --gpu [TYPE], --flash | --short | --long,
# --group NAME (as condor/submit.sh), --retries N (passes per node, default 20),
# --step N (only step N, by its number or node name in `pipeline status`; repeatable:
# a study step by step), --dry-run (write the DAG, do not submit).
#
# Example, one dataset of the batch-norm study:
#   export RFL_DATASET=Cho2017
#   bash condor/submit_dag.sh mi_spdnet_bn --memory 12GB
#   python -m rfl.experiments.pipeline status mi_spdnet_bn
#
# The DAG records this shell's environment (RFL_DATASET, the folders): set it first.
# Run it from the folder where the plan's relative paths make sense: the jobs run there.
# Submitting again is safe: finished steps are skipped and finished jobs are not rerun.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
[ "$#" -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }
PLAN="$1"; shift
ARGS=(); DRY=""; STEPS=""
export RFL_WORKDIR="$(pwd -P)"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpus|--memory|--disk|--retries) ARGS+=("$1" "$2"); shift 2 ;;
        --step)   ARGS+=(--step "$2"); STEPS+="_$2"; shift 2 ;;
        --gpu)    ARGS+=(--extra "request_gpus = 1")
                  if [ "$#" -ge 2 ] && [[ "$2" != --* ]]; then
                      ARGS+=(--extra "+wantGpuType = \"$2\""); shift
                  fi; shift ;;
        --flash)  ARGS+=(--extra "+isFlash = True"); shift ;;
        --short)  ARGS+=(--extra "+isShortJob = True"); shift ;;
        --long)   ARGS+=(--extra "+isLongJob = True"); shift ;;
        --group)  ARGS+=(--extra "+wishedAcctGroup = \"$2\""); shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
DAG=$(python -m rfl.experiments.pipeline dag "$PLAN" "${ARGS[@]}")
echo "DAG: $DAG"
if [ -n "$DRY" ]; then
    echo "condor_submit_dag $DAG"
    exit 0
fi
condor_submit_dag -batch-name "${PLAN}_${RFL_DATASET:-}${STEPS:+_step${STEPS#_}}" "$DAG"
echo "Follow: python -m rfl.experiments.pipeline status $PLAN   (and condor_q)"
