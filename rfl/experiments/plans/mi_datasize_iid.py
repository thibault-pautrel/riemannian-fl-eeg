"""Data quantity with IID clients: the control of the data-quantity study.

The scenarios, arms, selection and training of mi_datasize (SPDNet) and
mi_datasize_eegnet (EEGNet), with the trials dealt at random to the clients
(--client-partition iid). Each subject is first reduced to N trials per class with the
draw of the subject clients, so that both partitions hold the same trials and the same
test set (rfl/federated/server.py). Without heterogeneity between the clients,
federated training should beat local training at every N: the crossover of the subject
clients then comes from their heterogeneity.

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_datasize_iid --cpus 1 --memory 6GB

The memory is that of mi_study, which runs the same arms with subject clients. Two
chains: the SPDNet arms after the calibration of the batch-norm study, and the EEGNet
arms. Submit once `pipeline status mi_study` shows the calibration done (N1, N2): its
steps are then skipped, while a calibration still running in another DAG would be
submitted a second time. The arms are those
of mi_datasize with the prefix iid_ in their normalisation (iid_bn, iid_bnshared).
"""

from rfl.experiments.plans import mi_datasize as _DS
from rfl.experiments.plans.mi_datasize import DATASET, EXTEND, SELECTION  # noqa: F401

NAME = f"mi_{DATASET}_datasize_iid"
IID = ("--client-partition", "iid")


def study():
    return _DS.study(IID, "iid_", NAME)


STAGES = {"sweep1": lambda: study().sweep(0), "final": lambda: study().final()}
PIPELINE = [
    list(_DS.PIPELINE[:3]) + [("mi_datasize_iid", "sweep1"), ("mi_datasize_iid", "final")],
    [("mi_datasize_iid_eegnet", "sweep1"), ("mi_datasize_iid_eegnet", "final")],
]
