"""Exact running statistics of the batch norms, computed once after training.

During training each batch is normalised with its own statistics, and the running
statistics used at evaluation follow a momentum average. With little data the average
has few updates: with one batch per round and momentum 0.01, the SPD running mean
covers about 26 % of the way from its initial value I after 30 rounds. The evaluation
then normalises the data with statistics the training never used.

`recalibrate` replaces them by the exact statistics of each layer's inputs over the
training data, the model in evaluation mode (no dropout), layer after layer in forward
order, so that the inputs of a layer are computed with the recalibrated layers before
it ("precise BN", Wu and Johnson, 2021). Only buffers are written: the libraries are
used as they are.

    SPD batch norm (SPDNet)   running_mean = the layer's own mean of its inputs
                              (`module.mean_fun`, the geometry of --bn-mean-type)
    BatchNorm (EEGNet)        running_mean and running_var = mean and unbiased variance
                              of each channel over the batch and spatial positions

In federated runs (`recalibrate_clients`):

    FedBN and local-only   each client recalibrates its own layers on its own data
    shared batch norm      each client computes its statistics, the server combines
                           them as the aggregation combines running statistics (SPD:
                           weighted mean in the layer's geometry, rfl/models/
                           spd_batchnorm.py; BatchNorm: weighted average), and every
                           client receives the same values
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader


def _is_spd_bn(module: nn.Module) -> bool:
    from rfl.models.spdnet import BatchNormSPD
    return BatchNormSPD is not None and isinstance(module, BatchNormSPD)


def batchnorm_layers(model: nn.Module, loader) -> List[tuple]:
    """(name, module) of the batch norms of `model` (SPD and torch), in call order."""
    names = {id(m): n for n, m in model.named_modules()}
    order: List[tuple] = []
    hooks = [m.register_forward_pre_hook(
        lambda mod, args: order.append((names[id(mod)], mod))
        if all(mod is not o[1] for o in order) else None)
        for m in model.modules() if isinstance(m, _BatchNorm) or _is_spd_bn(m)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x, _ = next(iter(loader))
            model(x[:2])
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return order


def spd_batchnorm_layers(model: nn.Module) -> List[tuple]:
    """(name, module) of the SPD batch norms of an SPDNet, in forward order."""
    out = []
    for i in range(int(getattr(model, "depth", 0))):
        name = "batch_norm" + ("" if i == 0 else f"_{i + 1}")
        mod = getattr(model, name, None)
        if mod is not None and _is_spd_bn(mod):
            out.append((name, mod))
    return out


def _sequential(loader) -> DataLoader:
    """Every training record once, in a fixed order (no shuffle, no dropped batch)."""
    return DataLoader(loader.dataset, batch_size=256, shuffle=False)


@torch.no_grad()
def layer_statistics(model: nn.Module, module: nn.Module, loader, device
                     ) -> Dict[str, torch.Tensor]:
    """Exact statistics of the inputs of `module` over `loader` (module docstring)."""
    inputs: List[torch.Tensor] = []
    handle = module.register_forward_pre_hook(
        lambda _m, args: inputs.append(args[0].detach()))
    was_training = model.training
    model.eval()
    try:
        for x, _ in _sequential(loader):
            model(x.to(device))
    finally:
        handle.remove()
        model.train(was_training)
    X = torch.cat(inputs)
    if _is_spd_bn(module):
        return {"running_mean": module.mean_fun(X.to(module.running_mean.dtype)).detach()}
    dims = [0] + list(range(2, X.dim()))
    return {"running_mean": X.mean(dim=dims), "running_var": X.var(dim=dims, unbiased=True)}


def _combine(module: nn.Module, stats: Sequence[Dict[str, torch.Tensor]],
             weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """The shared statistics of several clients (module docstring)."""
    w = torch.tensor([float(x) for x in weights], dtype=torch.float64)
    w = w / w.sum()
    if _is_spd_bn(module):
        from rfl.models.spd_batchnorm import weighted_spd_mean
        return {"running_mean": weighted_spd_mean([s["running_mean"] for s in stats],
                                                  list(weights), module.mean_type)}
    return {k: sum(wi * s[k].double() for wi, s in zip(w, stats)).to(stats[0][k].dtype)
            for k in stats[0]}


def _write(module: nn.Module, stats: Dict[str, torch.Tensor]) -> None:
    for k, v in stats.items():
        getattr(module, k).copy_(v.to(getattr(module, k).dtype))


@torch.no_grad()
def recalibrate(model: nn.Module, loader, device) -> Dict[str, torch.Tensor]:
    """Set the running statistics of every batch norm of `model` from its training data.

    Returns {state-dict key: new value}.
    """
    out = {}
    for name, mod in batchnorm_layers(model, loader):
        stats = layer_statistics(model, mod, loader, device)
        _write(mod, stats)
        out.update({f"{name}.{k}": getattr(mod, k).detach().clone() for k in stats})
    return out


@torch.no_grad()
def recalibrate_clients(clients: Sequence, shared: bool, weights: Sequence[float],
                        device, states: Sequence[Dict[str, torch.Tensor]] = ()) -> None:
    """Recalibrate the reported model of every client (module docstring).

    Args:
        clients: FLClient objects holding the reported model (`local_model`).
        shared: the batch norm is shared (--no-fedbn): one combined value per layer.
        weights: aggregation weights of the clients (shared batch norm).
        states: global state dicts updated with the combined values (shared batch norm).
    """
    if not clients:
        return
    order = [n for n, _ in batchnorm_layers(clients[0].local_model, clients[0].train_loader)]
    for name in order:                     # call order: layer k after the layers before it
        mods = [dict(c.local_model.named_modules())[name] for c in clients]
        stats = [layer_statistics(c.local_model, m, c.train_loader, device)
                 for c, m in zip(clients, mods)]
        if shared:
            common = _combine(mods[0], stats, weights)
            stats = [common] * len(clients)
            for st in states:
                for k, v in common.items():
                    key = f"{name}.{k}"
                    if key in st:
                        st[key] = v.detach().clone().to(st[key].dtype)
        for m, s in zip(mods, stats):
            _write(m, s)
