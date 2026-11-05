"""The optimiser of every training step (config.optimizer, rfl/utils/config.py)."""
from __future__ import annotations

from typing import Iterable, Optional

import torch


def make_optimizer(params: Iterable[torch.nn.Parameter], config, lr: float,
                   weight_decay: Optional[float] = None) -> torch.optim.Optimizer:
    """Adam or SGD (with config.sgd_momentum) over `params`.

    `weight_decay` defaults to config.weight_decay. Used by the centralised loop, the
    federated clients (with and without DP) and the head re-fit.
    """
    wd = float(config.weight_decay if weight_decay is None else weight_decay)
    if config.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=float(config.sgd_momentum),
                               weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)
