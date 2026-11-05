"""Data quantity on synthetic subjects, TS+LR baseline (see synth_datasize)."""

from rfl.experiments.datasize import tslr_jobs
from rfl.experiments.plans.synth_datasize import DATA_ARGS, FINAL_SEEDS, QUICK, SCENARIOS

NAME = "synth_datasize_quick_tslr" if QUICK else "synth_datasize_tslr"
STAGES = {"final": lambda: tslr_jobs(DATA_ARGS, SCENARIOS, FINAL_SEEDS)}
