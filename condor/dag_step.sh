#!/bin/bash
# PRE and POST script of the nodes of a study DAG (rfl/experiments/pipeline.py).
#
#   condor/dag_step.sh <dag_env.sh> pre|post <plan> <node index> [jobs file]
#
# Runs on the submit machine, started by DAGMan. dag_env.sh holds the environment of the
# shell that wrote the DAG (folders, RFL_DATASET), so every step sees the same study.
set -uo pipefail
# shellcheck disable=SC1090
source "$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "${RFL_WORKDIR:-.}" || exit 3
exec python -m rfl.experiments.pipeline "$@"
