"""Calibration stages of agg_synth (see there): central SPDNet on the synthetic data."""
from rfl.experiments.plans.agg_synth import EXTEND, calibration  # noqa: F401

NAME = "agg_synth_calib"
STAGES = {f"sweep{k + 1}": (lambda k=k: calibration().sweep(k)) for k in range(3)}
