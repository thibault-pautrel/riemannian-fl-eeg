"""Data quantity, tangent space + logistic regression (see mi_datasize).

Local (one model per subject) and pooled (every subject re-centred on its own mean, one
model on all subjects, a non-federated transfer reference), for every scenario, on the
final seeds. The regularisation is chosen inside each run (rfl/experiments/run_tslr.py).
"""

from rfl.experiments.datasize import tslr_jobs
from rfl.experiments.plans.mi_datasize import DATASET, SCENARIOS
from rfl.experiments.plans.mi_spdnet_bn_calib import FINAL_SEEDS

NAME = f"mi_{DATASET}_datasize_tslr"
STAGES = {"final": lambda: tslr_jobs(["--dataset", DATASET], SCENARIOS, FINAL_SEEDS)}
