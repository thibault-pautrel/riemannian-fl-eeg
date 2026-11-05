"""Template: a study on a new dataset, with its own search space.

Copy this file to rfl/experiments/plans/<my_study>.py and edit the values below. Before a
new dataset, follow docs/data.md, "Adding a dataset" (registry entry, download, ReEig floors).

This template uses the staged search: stage 1 tunes the learning rate and the horizon,
stage 2 the clip constant C (DP arms) and the BiMap dimension, at the values chosen in
stage 1. Each stage is checked, run and selected before the next one:

    python -m rfl.experiments.grid <my_study> --stage sweep1 --check
    bash condor/submit.sh <my_study> sweep1
    python -m rfl.experiments.grid <my_study> --stage sweep1 --select
    bash condor/submit.sh <my_study> sweep2          # reads the stage-1 selection
    python -m rfl.experiments.grid <my_study> --stage sweep2 --select
    bash condor/submit.sh <my_study> final           # reads the stage-2 selection

Every key of a search space is swept: `lr`, `batch_size`, `horizon` (rounds, or epochs
for the centralised arm), or any runner option written as on the command line. The
first value of each list is the starting value of the staged search.
"""

from rfl.experiments.protocol import Protocol

P = Protocol(
    dataset="Cho2017",
    eps=10,
    models=("spdnet",),
    aggregations={"spdnet": ("projavg", "rlavg", "retavg")},
    kinds=("central", "fed_shared", "fed_fedper", "dp_shared", "dp_fedper"),
    space={
        # Non-private arms.
        "nodp": {"lr": [1e-3, 1e-2], "batch_size": [32], "horizon": [100, 200]},
        # Private arms: the batch size sets the sensitivity 2C/B. It must not exceed
        # the smallest client training set (checked by --check once the dataset
        # statistics exist, see docs/data.md).
        "dp": {"lr": [1e-2, 1e-1], "batch_size": [64], "horizon": [100, 200],
               "--dp-clip-C": [1.0, 0.5, 2.0]},
    },
    # Swept for SPDNet only, on top of the space above.
    model_space={"spdnet": {"--hidden-dim": [16, 8]}},
    search="staged",
    stage_keys=[["lr", "horizon"], ["--dp-clip-C", "--hidden-dim"]],
    sweep_seeds=(1, 2, 3),
    final_seeds=tuple(range(101, 111)),
    name="cho2017_spdnet",          # results under $RFL_RESULTS/cho2017_spdnet/
)
NAME = P.name
STAGES = P.stages()
