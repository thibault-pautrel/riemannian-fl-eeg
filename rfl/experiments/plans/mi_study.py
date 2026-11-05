"""Both studies of one dataset in one DAG: the batch-norm study, then the data quantity.

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_study --cpus 2 --memory 8GB
    python -m rfl.experiments.pipeline status mi_study

Three chains side by side: (1) ReEig floors, calibration (mi_spdnet_bn_calib), the 18
arms of mi_spdnet_bn, then the SPDNet arms of mi_datasize, (2) the EEGNet arms of
mi_datasize, (3) the TS+LR baseline. The calibration runs once and serves both studies. The results go to the folders of each plan (mi_<dataset>_bn,
mi_<dataset>_datasize, ...), as with the separate DAGs.
"""

from rfl.experiments.plans.mi_datasize import DATASET
from rfl.experiments.plans.mi_spdnet_bn import PIPELINE as _BN

NAME = f"mi_{DATASET}_study"
STAGES: dict = {}
PIPELINE = [
    list(_BN) + [("mi_datasize", "sweep1"), ("mi_datasize", "sweep2"),
                 ("mi_datasize", "final")],
    [("mi_datasize_eegnet", "sweep1"), ("mi_datasize_eegnet", "sweep2"),
     ("mi_datasize_eegnet", "final")],
    [("mi_datasize_tslr", "final")],
]
