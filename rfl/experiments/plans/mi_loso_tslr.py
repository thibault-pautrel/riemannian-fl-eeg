"""Inter-subject study, tangent space + logistic regression (see mi_loso).

One run per fold and final seed (rfl/experiments/run_tslr.py --mode loso): the pooled
model of the re-centred sources, then each target re-centred on its own trials, scored
by that model, by that model refitted with its N trials, and by a model of its N trials
alone. The regularisation is chosen inside each run, on the sources' validation or by
cross-validation on the N trials.
"""

from rfl.experiments.grid import Job
from rfl.experiments.plans.mi_loso_calib import FOLDS, NS
from rfl.experiments.plans.mi_spdnet_bn_calib import DATASET, FINAL_SEEDS

NAME = f"mi_{DATASET}_loso_tslr"


def jobs():
    return [Job(arm=f"tslr_loso_f{k}", entry="tslr", seed=s, hp={},
                args=["--dataset", DATASET, "--mode", "loso",
                      "--target-subjects", *map(str, targets),
                      "--target-ns", *map(str, NS)])
            for k, targets in enumerate(FOLDS) for s in FINAL_SEEDS]


STAGES = {"final": jobs}
