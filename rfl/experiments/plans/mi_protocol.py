"""Motor-imagery study with the recommended protocol: sweep, select, final runs.

The arms, search space and seeds are those of `rfl/experiments/protocol.py` (defaults).
The dataset is set by the environment variable RFL_DATASET (BNCI2014_001 by default),
the privacy budget of the DP arms by RFL_EPS (10 by default). Results go under
$RFL_RESULTS/mi_<dataset>/, one folder per dataset.

    bash condor/submit.sh mi_protocol sweep                       # 432 runs
    python -m rfl.experiments.grid mi_protocol --stage sweep --status
    python -m rfl.experiments.grid mi_protocol --stage sweep --select # sweep/selection.json
    bash condor/submit.sh mi_protocol final                       # 200 runs
    python -m rfl.analysis.report $RFL_RESULTS/mi_BNCI2014_001/final --by kind arch aggregation scope eps

RFL_DATASET must be set to the same value for every command of a study. For a study
with its own search space, copy `template_new_dataset.py`.
"""

import os

from rfl.experiments.protocol import Protocol

P = Protocol(dataset=os.environ.get("RFL_DATASET", "BNCI2014_001"),
             eps=float(os.environ.get("RFL_EPS", "10")))
NAME = P.name
STAGES = P.stages()
