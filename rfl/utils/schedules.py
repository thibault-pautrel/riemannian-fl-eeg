"""Learning-rate schedules, shared by the federated server and the centralised loop.

One function computes the rate of round (or epoch) t from the schedule, the budget and,
for 'plateau', the history of the selection metric. The two pipelines call it with the
same arguments, so a schedule means exactly the same thing in both.
"""

from __future__ import annotations

import math
from typing import List, Optional


def compute_round_lr(base_lr: float, round_num: int, total_rounds: int,
                     schedule: str = 'none', lr_min: float = 1e-4,
                     val_history: Optional[List[float]] = None,
                     plateau_patience: int = 20,
                     plateau_factor: float = 0.5,
                     mode: str = 'max') -> float:
    """Return the learning rate of round `round_num` (1-indexed).

    Schedules, with T = total_rounds:

        'none'     base_lr
        'cosine'   lr_min + (base_lr - lr_min) * (1 + cos(pi t)) / 2,
                   t = (round_num - 1) / (T - 1)
        'step'     base_lr for rounds 1..max(1, int(0.7 T)), then 0.3 base_lr
        'plateau'  base_lr multiplied by plateau_factor each time plateau_patience
                   consecutive entries of val_history fail to strictly improve on the
                   best so far. This is the federated ReduceLROnPlateau.

    'cosine', 'step' and 'plateau' are floored at lr_min. With T <= 1, 'cosine' and
    'step' return base_lr.

    Args:
        base_lr: initial learning rate.
        round_num: current round, starting at 1.
        total_rounds: round budget T.
        schedule: 'none', 'cosine', 'step' or 'plateau'.
        lr_min: floor of the learning rate.
        val_history: selection-metric values of the previous rounds ('plateau' only).
        plateau_patience: rounds without improvement before a decay.
        plateau_factor: multiplicative decay.
        mode: 'min' if a lower metric is better (loss), 'max' otherwise (accuracy).

    Raises:
        ValueError: for an unknown schedule, or 'plateau' without val_history.
    """
    if schedule == 'plateau':
        if val_history is None:
            raise ValueError("lr_schedule='plateau' needs val_history; passing None "
                             "would silently degrade it to a constant rate.")
        minimize = str(mode).lower() == 'min'
        lr = base_lr
        best = float('inf') if minimize else -float('inf')
        stale = 0
        for v in val_history:
            if (v < best) if minimize else (v > best):
                best, stale = v, 0
            else:
                stale += 1
                if stale >= plateau_patience:
                    lr, stale = max(lr * plateau_factor, lr_min), 0
        return max(lr, lr_min)

    if schedule == 'none' or total_rounds <= 1:
        return base_lr

    t = (round_num - 1) / max(total_rounds - 1, 1)

    if schedule == 'cosine':
        lr = lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))
    elif schedule == 'step':
        # The first 70% of the rounds (at least one) at the base rate.
        lr = base_lr if round_num <= max(1, int(0.7 * total_rounds)) else base_lr * 0.3
    else:
        raise ValueError(f"Unknown lr_schedule: {schedule!r} "
                         "(choose 'none', 'cosine', 'step' or 'plateau')")
    return max(lr, lr_min)


def best_position(series, lower_is_better: bool):
    """Index of the best finite value (earliest on ties), or None if there is none.

    NaN and infinite values are never selected. A plain min/max with a key would keep a
    NaN in first position, since every comparison with NaN is False.
    """
    finite = [i for i, v in enumerate(series)
              if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    pick = min if lower_is_better else max
    return pick(finite, key=lambda i: series[i])
