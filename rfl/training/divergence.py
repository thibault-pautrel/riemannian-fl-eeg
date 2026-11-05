"""A diverged run is a result, not a failure.

Too large a learning rate makes the training loss non-finite, and the eigenvalue
decompositions of SPDNet then fail (torch.linalg.LinAlgError), or puts a RetAvg client
where the polar retraction cannot be inverted (RetractionNotInvertible). The runners stop such a
run and write its result file with `"diverged": true` and NaN metrics, and exit 0. The
selection then leaves its cell out (rfl/experiments/grid.py:select needs every seed of a
cell), and the run is not submitted again.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict

import torch

from rfl.geometry.stiefel import RetractionNotInvertible


class TrainingDiverged(RuntimeError):
    """The training loss became non-finite."""


# A non-finite loss, a failed eigendecomposition, or a RetAvg client too far to invert
# the polar retraction: all three come from steps too large.
DIVERGENCE_ERRORS = (TrainingDiverged, torch.linalg.LinAlgError, RetractionNotInvertible)


def check_finite(loss: float, where: str) -> None:
    """Raise TrainingDiverged when `loss` is NaN or infinite."""
    if not math.isfinite(float(loss)):
        raise TrainingDiverged(f"non-finite training loss ({loss}) at {where}")


def write_diverged(save_dir: str, config: Any, result_file: str, error: BaseException) -> str:
    """Write the result file of a diverged run. Returns its path."""
    nan = float("nan")
    out: Dict[str, Any] = {"config": config.to_dict(), "diverged": True,
                           "error": f"{type(error).__name__}: {str(error)[:300]}"}
    if result_file == "fl_results.json":
        out.update(final_metrics={k: nan for k in (
            "global_val_acc", "global_val_f1", "global_val_loss", "global_test_acc",
            "global_test_f1", "global_test_loss")},
            last_round_metrics={}, round_history={}, report_rule=config.report_rule)
    else:
        out.update(training={"report_rule": config.report_rule, "final_val_acc": nan,
                             "final_val_loss": nan},
                   test_metrics={"accuracy": nan, "f1_macro": nan})
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, result_file)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"DIVERGED: {out['error']}\nresult written: {path}")
    return path
