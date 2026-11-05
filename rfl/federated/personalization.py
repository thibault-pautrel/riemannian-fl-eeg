"""Post-training re-fit of the classifier head on each client (personalisation).

After the last round each client fits its head on its own training records, with the
rest of the model frozen and in eval mode (normalisation statistics and dropout off).
The model constraints (EEGNet max-norm) apply after each step, as in training.
It starts from the head of the reported model: the global head, or the client's
personal head with `fl_local_fc`. This gives FedAvg + fine-tuning on a shared head,
and the post-hoc fit of a head pinned during the rounds
(`fl_local_head_mode='frozen'`).

Privacy. The re-fit happens after the last release and sends nothing, so the DP
guarantee on the transcript is unchanged (post-processing). The re-fitted head is
computed from the raw records: it is not covered by the guarantee if it is published.

Selection. The head is trained once to the largest budget of the epoch grid, and the
states at the grid epochs are kept as candidates, for each weight decay. Each client
keeps the candidate with the best validation accuracy, the loss breaking ties
(`fl_head_refit_selection`). Ties go to the stronger weight decay and the shorter
budget. A short grid keeps the selection noise low on small validation splits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from rfl.training.optim import make_optimizer

logger = logging.getLogger(__name__)

__all__ = ["parse_epoch_grid", "parse_weight_decays", "refit_head",
           "refit_and_evaluate", "SELECTION_MODES"]

SELECTION_MODES = ("fixed", "wd_only", "wd_and_epochs")


def parse_epoch_grid(spec: Any) -> Tuple[int, ...]:
    """Parse '25,50,100' (or an iterable) into a sorted tuple of positive epochs.

    An empty spec, None or '0' gives an empty tuple, which disables the re-fit.

    Raises:
        ValueError: on a negative or non-numeric entry.
    """
    if spec is None:
        return ()
    if isinstance(spec, str):
        vals = [int(float(v)) for v in spec.replace(";", ",").split(",") if v.strip()]
    else:
        vals = [int(v) for v in spec]
    if any(v < 0 for v in vals):
        raise ValueError(f"head re-fit epochs must be non-negative, got {spec!r}")
    return tuple(sorted({v for v in vals if v > 0}))


def parse_weight_decays(spec: Any) -> Tuple[float, ...]:
    """Parse '0,1e-3' (or an iterable) into a tuple of weight decays. Empty gives (0.0,).

    Raises:
        ValueError: on a negative entry.
    """
    if spec is None:
        return (0.0,)
    if isinstance(spec, str):
        vals = [float(v) for v in spec.replace(";", ",").split(",") if v.strip()]
    else:
        vals = [float(v) for v in spec]
    if any(v < 0 for v in vals):
        raise ValueError(f"head re-fit weight decays must be non-negative, got {spec!r}")
    return tuple(vals) if vals else (0.0,)


@torch.no_grad()
def _loss_acc(model: nn.Module, loader, criterion: nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """(mean loss, accuracy) of `model` on `loader`. (inf, 0) on an empty loader."""
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        total_loss += float(criterion(out, y)) * int(y.numel())
        correct += int(out.argmax(1).eq(y).sum())
        total += int(y.numel())
    if not total:
        return float("inf"), 0.0
    return total_loss / total, correct / total


def _better(cand: Dict[str, Any], best: Dict[str, Any]) -> bool:
    """True if `cand` strictly beats `best`: higher accuracy, then lower loss.

    With NaN scores (selection 'fixed') every comparison is False, so the first
    candidate is kept.
    """
    if not best:
        return True
    if cand["val_acc"] > best["val_acc"]:
        return True
    return cand["val_acc"] == best["val_acc"] and cand["val_loss"] < best["val_loss"]


def refit_head(client: Any, state_dict: Dict[str, torch.Tensor], criterion: nn.Module,
               device: torch.device, *, lr: float, epoch_grid: Sequence[int],
               weight_decays: Sequence[float] = (0.0,),
               selection: str = "wd_and_epochs") -> Dict[str, Any]:
    """Re-fit the head of one client and leave the chosen head in `client.local_model`.

    Args:
        client: an FLClient with data and model.
        state_dict: the reported global state. The client keeps its local keys.
        criterion: loss function.
        device: torch device.
        lr: Adam learning rate.
        epoch_grid: candidate budgets, in epochs over the client's training records.
        weight_decays: candidate weight decays.
        selection: one of SELECTION_MODES.

    Returns:
        A diagnostic dict: chosen weight_decay and epochs, their validation loss and
        accuracy, and the trace of all candidates.

    Raises:
        ValueError: on an unknown selection, an empty grid, a model without head, or a
            client without validation records when the selection needs them.
    """
    from rfl.federated.param_groups import identify_classifier_param_names

    if selection not in SELECTION_MODES:
        raise ValueError(f"selection must be one of {SELECTION_MODES}, got {selection!r}")
    egrid = tuple(sorted(set(int(e) for e in epoch_grid if int(e) > 0)))
    if not egrid:
        raise ValueError("refit_head needs a non-empty epoch grid")
    grid = tuple(float(w) for w in weight_decays) or (0.0,)
    if selection == "fixed":
        grid, egrid = (grid[0],), (egrid[-1],)
    elif selection == "wd_only":
        egrid = (egrid[-1],)
    score = selection != "fixed"
    if score and len(client.val_loader.dataset) == 0:
        raise ValueError(f"client {client.client_id} has no validation records, so the "
                         "re-fit cannot be selected. Use fl_head_refit_selection='fixed'.")

    client.receive_global_model(state_dict)
    model = client.local_model.to(device)
    names = identify_classifier_param_names(model)
    head = [p for n, p in model.named_parameters() if n in names]
    if not head:
        raise ValueError("refit_head: the model has no classifier head")

    was_training = model.training
    start = [p.detach().clone() for p in head]
    head_ids = {id(p) for p in head}
    saved_flags = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)
    model.eval()

    best: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
    wanted = set(egrid)
    for wd in sorted(grid, reverse=True):            # strongest first: ties stay regularised
        with torch.no_grad():
            for p, p0 in zip(head, start):
                p.copy_(p0)
        optimizer = make_optimizer(head, client.config, lr, weight_decay=wd)
        for epoch in range(1, egrid[-1] + 1):
            for x, y in client.train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                criterion(model(x), y).backward()
                optimizer.step()
                if hasattr(model, "apply_constraints"):
                    model.apply_constraints()            # EEGNet max-norm on the head
            if epoch in wanted:
                if score:
                    loss, acc = _loss_acc(model, client.val_loader, criterion, device)
                else:
                    loss, acc = float("nan"), float("nan")
                cand = {"weight_decay": wd, "epochs": epoch, "val_loss": loss,
                        "val_acc": acc, "state": [p.detach().clone() for p in head]}
                trace.append({k: cand[k] for k in ("weight_decay", "epochs",
                                                   "val_loss", "val_acc")})
                if _better(cand, best):
                    best = cand

    with torch.no_grad():
        for p, p_best in zip(head, best["state"]):
            p.copy_(p_best)
    for p, flag in saved_flags:
        p.requires_grad_(flag)
    model.train(was_training)

    return {"weight_decay": best["weight_decay"], "epochs": int(best["epochs"]),
            "val_loss": best["val_loss"], "val_acc": best["val_acc"],
            "selection": selection, "trace": trace}


def _pooled(values: Sequence[float], sizes: Sequence[int]) -> float:
    total = float(sum(sizes))
    return float(sum(v * n for v, n in zip(values, sizes)) / total) if total > 0 else 0.0


def refit_and_evaluate(clients: Sequence[Any], state_dict: Dict[str, torch.Tensor],
                       criterion: nn.Module, device: torch.device, *, lr: float,
                       epoch_grid: Sequence[int], weight_decays: Sequence[float] = (0.0,),
                       selection: str = "wd_and_epochs",
                       verbose: bool = True) -> Dict[str, Any]:
    """Re-fit every client's head, then evaluate the personalised models.

    The metrics are pooled over the clients' predictions, as in FLServer, and prefixed
    'personalized_' so that they sit next to the federated metrics in fl_results.json.
    """
    fits = [refit_head(c, state_dict, criterion, device, lr=lr, epoch_grid=epoch_grid,
                       weight_decays=weight_decays, selection=selection)
            for c in clients]

    sums: Dict[str, Dict[str, list]] = {s: {"loss": [], "n": [], "pred": [], "lab": [],
                                            "acc": []} for s in ("val", "test")}
    for client in clients:
        evals = client.evaluate_local_model(criterion, splits=("val", "test"))
        for split in ("val", "test"):
            e = evals[split]
            sums[split]["loss"].append(e["loss"])
            sums[split]["acc"].append(e["acc"])
            sums[split]["n"].append(e["n_samples"])
            sums[split]["pred"].extend(e["predictions"])
            sums[split]["lab"].extend(e["labels"])

    out: Dict[str, Any] = {}
    for split in ("val", "test"):
        d = sums[split]
        n = sum(d["n"])
        out[f"personalized_{split}_loss"] = _pooled(d["loss"], d["n"])
        out[f"personalized_{split}_acc"] = _pooled(d["acc"], d["n"])
        out[f"personalized_{split}_f1"] = (float(f1_score(d["lab"], d["pred"],
                                                          average="macro", zero_division=0))
                                           if n else 0.0)
        out[f"personalized_per_client_{split}_acc"] = [float(a) for a in d["acc"]]
    out.update({
        "personalized_lr": float(lr),
        "personalized_epoch_grid": [int(e) for e in epoch_grid],
        "personalized_weight_decay_grid": [float(w) for w in weight_decays],
        "personalized_selection": selection,
        "personalized_fits": fits,
    })
    if verbose:
        chosen = [(f["weight_decay"], f["epochs"]) for f in fits]
        logger.info(f"[head re-fit] (weight_decay, epochs) per client: {chosen} -> test acc "
              f"{out['personalized_test_acc'] * 100:.2f}%, macro-F1 "
              f"{out['personalized_test_f1']:.4f}")
    return out
