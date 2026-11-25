"""Data quantity with IID clients, EEGNet arms (see mi_datasize_iid)."""

from rfl.experiments.plans import mi_datasize_eegnet as _DS
from rfl.experiments.plans.mi_datasize import DATASET, EXTEND, SELECTION  # noqa: F401
from rfl.experiments.plans.mi_datasize_iid import IID

NAME = f"mi_{DATASET}_datasize_iid_eegnet"


def study():
    return _DS.study(IID, "iid_", NAME)


STAGES = {"sweep1": lambda: study().sweep(0), "final": lambda: study().final()}
