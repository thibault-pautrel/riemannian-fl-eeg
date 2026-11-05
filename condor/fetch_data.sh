#!/bin/bash
# Download MOABB datasets into $MNE_DATA, on the user interface (it has network access).
#
#   bash condor/fetch_data.sh BNCI2014_001 [Cho2017 ...]
#
# Download before submitting: many jobs downloading the same files at once would
# collide, and worker nodes may not reach the dataset hosts. A dataset already present
# is not downloaded again.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
[ "$#" -ge 1 ] || { echo "usage: $0 DATASET [DATASET ...]" >&2; exit 2; }
cd "$RFL_REPO"
FAILED=()
for ds in "$@"; do
    echo "== $ds -> $MNE_DATA  $(date -Is)"
    # A dataset with missing subjects does not stop the others.
    python -m rfl.experiments.fetch_dataset --dataset "$ds" --data-dir "$MNE_DATA" \
        || FAILED+=("$ds")
done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "INCOMPLETE: ${FAILED[*]}. Run again: bash condor/fetch_data.sh ${FAILED[*]}" >&2
    exit 1
fi
echo "All datasets complete: $*"
