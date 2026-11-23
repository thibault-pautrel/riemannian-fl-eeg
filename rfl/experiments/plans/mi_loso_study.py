"""The whole inter-subject study of one dataset in one DAG.

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_loso_study --memory 12GB
    python -m rfl.experiments.pipeline status mi_loso_study

Three chains side by side: SPDNet (floors per fold, calibration, mi_loso), EEGNet
(mi_loso_eegnet_calib, mi_loso_eegnet) and TS+LR (mi_loso_tslr).
"""

from rfl.experiments.plans.mi_loso import PIPELINE as _SPD
from rfl.experiments.plans.mi_loso_eegnet import PIPELINE as _EEG
from rfl.experiments.plans.mi_spdnet_bn_calib import DATASET

NAME = f"mi_{DATASET}_loso_study"
STAGES: dict = {}
PIPELINE = [list(_SPD), list(_EEG), [("mi_loso_tslr", "final")]]
