"""Per-sample Riemannian gradient privatisation (the DP-RFedProj mechanism).

One local step on the federated parameters x uses

    g~ = (1/B) sum_{l in batch} clip_C(P_x grad f(x; z_l)) + P_x xi,   xi ~ N(0, sigma^2 I),

where P_x is the orthogonal projection onto the tangent space at x (BiMap weights,
the identity for Euclidean parameters) and clip_C rescales the joint norm over all
federated parameters to at most C. Under substitution adjacency the sensitivity of the
average is Delta = 2C/B, so the step is mu0-GDP with mu0 = 2C / (B * sigma).

The tangent part is computed in orthonormal coordinates of the tangent space at
uf(W), in float64 (`geometry.stiefel.StiefelChart`): each per-sample gradient becomes
a coordinate vector, the joint norm is clipped and the Gaussian noise is added in these
coordinates, and only then is the result mapped back to a matrix. The step is then
exactly a Gaussian mechanism in R^dim with sensitivity 2C/B. A projection computed at
W itself would not be: when W is orthonormal only up to round-off (float32), P_W is not
an exact orthogonal projector, and it shrinks some directions of the gradient and of the
noise by the same small factor, which inflates the effective mu0.

A per-sample gradient that is not finite contributes zero (as if clipped to 0), so one
corrupted record cannot turn the whole update into NaN.

Three more points make this sound and efficient.

1. Clip after projecting. P_x is fixed by the current iterate, so it is the same for
   neighbouring datasets and Delta = 2C/B still holds. Clipping the ambient gradient
   instead would spend the clip budget on the normal component that P_x discards.
2. The per-sample forward must not depend on the rest of the batch. Batch
   normalisation would couple every per-sample gradient to every record, so DP refuses
   any normalisation layer with running statistics (`assert_dp_safe_norm_layers`,
   Appendix D.2 of the DP-RFedProj paper).
3. Dropout stays live during the per-sample pass, so that DP runs train at the same
   dropout rate as the non-private ones. The masks are data-independent randomness,
   so a substitution still moves one summand and Delta is unchanged.

Per-sample gradients use a microbatch loop with `torch.autograd.grad` by default. It
works with the custom autograd of yetanotherspdnet (BiMap, ReEig, LogEig), which
`torch.func.vmap` does not support. When every federated parameter is Euclidean (EEGNet)
and `vectorized=True`, a vmap pass is used instead, after a one-off check against the
loop.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from rfl.geometry.stiefel import StiefelChart

__all__ = [
    "FedParam",
    "federated_param_groups",
    "tangent_noise",
    "per_sample_fed_norms",
    "privatized_fed_grads",
    "dp_local_step",
    "assert_dp_safe_norm_layers",
]

# (state-dict name, parameter tensor, is_stiefel) for one federated parameter.
FedParam = Tuple[str, nn.Parameter, bool]

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                  nn.AlphaDropout, nn.FeatureAlphaDropout)


def _is_bimap(module: nn.Module) -> bool:
    """True for BiMap modules, whose weight lies on the Stiefel manifold."""
    return "bimap" in type(module).__name__.lower()


# --------------------------------------------------------------------------- #
#  Forward mode for the per-sample pass
# --------------------------------------------------------------------------- #
class _forward_mode:
    """Context manager: model in eval() with dropout layers optionally left in train().

    Dropout is live in DP training and off only for the deterministic vmap check.
    Every module's training flag is restored on exit.
    """

    def __init__(self, model: nn.Module, dropout: bool):
        self.model = model
        self.dropout = bool(dropout)
        self._saved: List[Tuple[nn.Module, bool]] = []

    def __enter__(self) -> nn.Module:
        self._saved = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        if self.dropout:
            for m in self.model.modules():
                if isinstance(m, _DROPOUT_TYPES):
                    m.train()
        return self.model

    def __exit__(self, *exc):
        for module, was_training in self._saved:
            module.training = was_training
        return False


# --------------------------------------------------------------------------- #
#  Scope
# --------------------------------------------------------------------------- #
def federated_param_groups(model: nn.Module) -> List[FedParam]:
    """List the parameters inside the DP scope.

    Every trainable parameter, shared or personal: a personal head trained during the
    rounds is privatised with the trunk even though it is never uploaded. Parameters
    with requires_grad=False (frozen trunk or head) are skipped, since they do not
    depend on the data. Normalisation parameters are skipped too, but DP refuses models
    that carry them (`assert_dp_safe_norm_layers`).

    Args:
        model: the local model.

    Returns:
        A list of (name, parameter, is_stiefel) triples. is_stiefel is True for BiMap
        weights, where P_x is the Stiefel tangent projection.
    """
    from rfl.federated.param_groups import identify_bn_param_names

    local = set(identify_bn_param_names(model))

    groups: List[FedParam] = []
    for mod_name, module in model.named_modules():
        is_stiefel = _is_bimap(module)
        for p_name, p in module.named_parameters(recurse=False):
            full = f"{mod_name}.{p_name}" if mod_name else p_name
            if full in local or not p.requires_grad:
                continue
            groups.append((full, p, is_stiefel))
    return groups


def assert_dp_safe_norm_layers(model: nn.Module) -> None:
    """Refuse DP on a model with batch normalisation or other running statistics.

    Batch statistics couple the records of a batch, and running statistics (torch
    BatchNorm, or SPDNet's BatchNormSPDMean, which keeps a running Frechet mean) would
    be updated on private data. Either way Delta = 2C/B no longer holds.

    Raises:
        ValueError: if such a module is found.
    """
    bad: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, _BN_TYPES):
            bad.append(f"{name or '<root>'} ({type(module).__name__})")
            continue
        tname = type(module).__name__.lower()
        if "batchnorm" in tname or "runningmean" in tname:
            bad.append(f"{name or '<root>'} ({type(module).__name__})")
            continue
        for b_name, buf in module.named_buffers(recurse=False):
            if buf is not None and ("running" in b_name or "mean" in b_name):
                bad.append(f"{name or '<root>'}.{b_name} ({type(module).__name__})")
    if bad:
        raise ValueError(
            "DP is enabled but the model carries batch normalisation: "
            + ", ".join(sorted(set(bad)))
            + ". It couples the records of a batch, which breaks the Delta = 2C/B "
            "per-sample sensitivity and invalidates the accounted (eps, delta). Use "
            "--eegnet-norm-type none for EEGNet and no --use-batch-norm for SPDNet.")


# --------------------------------------------------------------------------- #
#  Noise
# --------------------------------------------------------------------------- #
def _charts(groups: Sequence[FedParam]) -> List[Optional[StiefelChart]]:
    """One tangent chart per Stiefel parameter (None for Euclidean parameters)."""
    return [StiefelChart(p) if is_stiefel else None for (_, p, is_stiefel) in groups]


def _coords(chart: Optional[StiefelChart], g: torch.Tensor) -> torch.Tensor:
    """Coordinates of the tangent part of g (float64, flat)."""
    return chart.coords(g) if chart is not None else g.detach().to(torch.float64).reshape(-1)


def _matrix(chart: Optional[StiefelChart], c: torch.Tensor, like: torch.Tensor):
    """The parameter-shaped tensor with coordinates c, in the dtype of `like`."""
    out = chart.vector(c) if chart is not None else c.reshape(like.shape)
    return out.to(device=like.device, dtype=like.dtype)


def tangent_noise(W: torch.Tensor, sigma: float, is_stiefel: bool,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Isotropic Gaussian noise of std sigma in the tangent space at uf(W) (P_x xi).

    Drawn in float64 on CPU, in the orthonormal coordinates of the tangent space
    (identity coordinates for a Euclidean parameter), then mapped to W's shape, dtype
    and device. Isotropic in the tangent space, so mu0 = 2C/(B sigma).
    """
    chart = StiefelChart(W) if is_stiefel else None
    dim = chart.dim if chart is not None else W.numel()
    xi = torch.randn(dim, generator=generator, dtype=torch.float64) * sigma
    return _matrix(chart, xi.to(W.device), W)


# --------------------------------------------------------------------------- #
#  Per-sample tangent gradients
# --------------------------------------------------------------------------- #
def _euclidean_only(groups: Sequence[FedParam]) -> bool:
    """True when no federated parameter is a Stiefel weight."""
    return all(not is_stiefel for (_, _, is_stiefel) in groups)


def _per_sample_tangent_grads_loop(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                                   criterion: nn.Module, groups: Sequence[FedParam],
                                   charts: Optional[List[Optional[StiefelChart]]] = None,
                                   ) -> List[List[torch.Tensor]]:
    """Per-sample tangent gradients by a microbatch loop (reference path, any model).

    Stiefel components are projected at uf(W) (`StiefelChart.project`).
    """
    params = [p for (_, p, _) in groups]
    charts = _charts(groups) if charts is None else charts
    out: List[List[torch.Tensor]] = []
    for i in range(x.shape[0]):
        try:
            loss_i = criterion(model(x[i:i + 1]), y[i:i + 1])
            grads = torch.autograd.grad(loss_i, params, retain_graph=False)
        except (RuntimeError, ValueError):
            # A record on which the model fails (for example a non-finite matrix in
            # an eigendecomposition) gets a NaN gradient, which the mechanism turns
            # into a zero contribution: the step never fails because of one record.
            grads = [torch.full_like(p, float("nan")) for p in params]
        out.append([ch.project(g) if ch is not None else g
                    for ch, g in zip(charts, grads)])
    return out


def _per_sample_grads_vmap(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                           criterion: nn.Module, groups: Sequence[FedParam],
                           ) -> List[List[torch.Tensor]]:
    """Per-sample gradients with torch.func.vmap, for Euclidean federated parameters only.

    Requires a model without batch normalisation, so that it is a pure function of its
    parameters.
    randomness='different' gives each sample its own dropout mask, as in the loop.
    """
    from torch.func import functional_call, grad, vmap

    fed_names = [n for (n, _, _) in groups]
    all_params = dict(model.named_parameters())
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    differentiable = {n: all_params[n].detach() for n in fed_names}
    fixed = {n: p.detach() for n, p in all_params.items() if n not in differentiable}

    def loss_i(dparams, xi, yi):
        logits = functional_call(model, ({**fixed, **dparams}, buffers), (xi.unsqueeze(0),))
        return criterion(logits, yi.unsqueeze(0))

    grads = vmap(grad(loss_i), in_dims=(None, 0, 0),
                 randomness="different")(differentiable, x, y)     # {name: (B, *shape)}
    return [[grads[n][i] for n in fed_names] for i in range(x.shape[0])]


def _vmap_matches_loop(model, x, y, criterion, groups, *, rtol=1e-4, atol=1e-6) -> bool:
    """True when the vmap gradients match the loop on this batch, with dropout off.

    Dropout is off because two passes would otherwise draw different masks.
    """
    try:
        with _forward_mode(model, dropout=False):
            loop = _per_sample_tangent_grads_loop(model, x, y, criterion, groups)
            vec = _per_sample_grads_vmap(model, x, y, criterion, groups)
    except Exception:
        return False
    if len(loop) != len(vec):
        return False
    return all(torch.allclose(a, b, rtol=rtol, atol=atol)
               for row_a, row_b in zip(loop, vec)
               for a, b in zip(row_a, row_b))


def _per_sample_tangent_grads(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                              criterion: nn.Module, groups: Sequence[FedParam],
                              *, vectorized: bool = False, dropout: bool = True,
                              ) -> List[List[torch.Tensor]]:
    """Per-sample tangent gradients, one list of tensors per sample.

    The model runs in eval() with dropout live unless `dropout=False`. The vmap path is
    used for Euclidean-only groups when `vectorized=True` and the one-off check
    `_vmap_matches_loop` passed. The check result is cached on the model as
    `_dp_vmap_ok`.
    """
    if not (vectorized and _euclidean_only(groups)):
        with _forward_mode(model, dropout):
            return _per_sample_tangent_grads_loop(model, x, y, criterion, groups)

    ok = getattr(model, "_dp_vmap_ok", None)
    if ok is None:                                   # decide once, cache on the model
        ok = _vmap_matches_loop(model, x, y, criterion, groups)      # dropout OFF
        try:
            model._dp_vmap_ok = bool(ok)
        except Exception:
            pass

    with _forward_mode(model, dropout):
        if ok:
            return _per_sample_grads_vmap(model, x, y, criterion, groups)
        return _per_sample_tangent_grads_loop(model, x, y, criterion, groups)


def per_sample_fed_norms(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         criterion: nn.Module, groups: Sequence[FedParam],
                         *, vectorized: bool = False, dropout: bool = True,
                         ) -> List[float]:
    """Joint per-sample norms ||P_x grad f(x; z_l)|| over the federated parameters.

    This is the quantity the mechanism clips.

    Args:
        model: the local model.
        x: input batch.
        y: label batch.
        criterion: loss function.
        groups: federated parameters, from `federated_param_groups`.
        vectorized: allow the vmap path for Euclidean-only groups.
        dropout: keep dropout live, as in DP training.

    Returns:
        One norm per sample.
    """
    charts = _charts(groups)
    norms: List[float] = []
    for tg in _per_sample_tangent_grads(model, x, y, criterion, groups,
                                        vectorized=vectorized, dropout=dropout):
        norms.append(float(torch.sqrt(sum((_coords(ch, t) ** 2).sum()
                                           for ch, t in zip(charts, tg)))))
    return norms


# --------------------------------------------------------------------------- #
#  The mechanism
# --------------------------------------------------------------------------- #
def privatized_fed_grads(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         criterion: nn.Module, groups: Sequence[FedParam],
                         C: float, sigma: float,
                         generator: Optional[torch.Generator] = None,
                         *, vectorized: bool = False, dropout: bool = True,
                         ) -> Dict[str, torch.Tensor]:
    """Privatised gradient g~ for each federated parameter.

        g~ = (1/B) sum_l clip_C(P_x grad f(x; z_l)) + P_x xi,   xi ~ N(0, sigma^2 I)

    The clip acts on the joint norm across all federated parameters, after projection.
    sigma = 0 disables the noise. C = inf disables clipping, which the tests use to
    recover the plain tangent minibatch gradient. DP runs always use a finite C.

    Args:
        model: the local model.
        x: input batch of size B.
        y: label batch.
        criterion: loss function.
        groups: federated parameters, from `federated_param_groups`.
        C: clip constant.
        sigma: noise std on the averaged gradient.
        generator: CPU generator for the noise.
        vectorized: allow the vmap path for Euclidean-only groups.
        dropout: keep dropout live in the per-sample pass.

    Returns:
        {parameter name: g~}. p.grad is not modified.
    """
    B = x.shape[0]
    charts = _charts(groups)
    dims = [ch.dim if ch is not None else p.numel() for ch, (_, p, _) in zip(charts, groups)]
    accum = [torch.zeros(d, dtype=torch.float64) for d in dims]
    clipping = math.isfinite(C)

    for tg in _per_sample_tangent_grads(model, x, y, criterion, groups,
                                        vectorized=vectorized, dropout=dropout):
        cs = [_coords(ch, t).cpu() for ch, t in zip(charts, tg)]
        sq = sum(float((c * c).sum()) for c in cs)
        if not math.isfinite(sq):
            continue                       # a non-finite record contributes zero
        scale = min(1.0, C / math.sqrt(sq)) if clipping and sq > 0 else 1.0
        for i, c in enumerate(cs):
            accum[i] += scale * c

    out: Dict[str, torch.Tensor] = {}
    for i, (ch, (name, p, _)) in enumerate(zip(charts, groups)):
        g = accum[i] / B
        if sigma > 0:                      # isotropic noise in the same coordinates
            g = g + sigma * torch.randn(dims[i], generator=generator, dtype=torch.float64)
        out[name] = _matrix(ch, g.to(ch.X.device if ch is not None else p.device), p)
    return out


def dp_local_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                  x: torch.Tensor, y: torch.Tensor, criterion: nn.Module,
                  groups: Sequence[FedParam], C: float, sigma: float,
                  generator: Optional[torch.Generator] = None,
                  apply_constraints: bool = False,
                  *, vectorized: bool = False) -> Tuple[float, torch.Tensor]:
    """One DP local optimisation step.

    (a) If some trainable parameter is outside the DP scope, run an ordinary batched
        backward for it. In the configurations `DPFLClient` accepts there is none, so a
        no-grad forward only provides the logits.
    (b) Compute g~ for the federated parameters with `privatized_fed_grads`.
    (c) Write g~ into p.grad and step. The optimiser update, `apply_constraints` (EEGNet
        max-norm) and `project_stiefel` (SPDNet) are post-processing of g~.

    Args:
        model: the local model, without batch normalisation.
        optimizer: optimiser over the model parameters.
        x: input batch of exactly B records.
        y: label batch.
        criterion: loss function.
        groups: federated parameters, from `federated_param_groups`.
        C: clip constant.
        sigma: noise std on the averaged gradient.
        generator: CPU generator for the noise.
        apply_constraints: call `model.apply_constraints()` after the step, if defined.
        vectorized: allow the vmap path for Euclidean-only groups.

    Returns:
        (batch loss, logits) of the forward in (a).
    """
    fed_ids = {id(p) for (_, p, _) in groups}
    has_local_grads = any(p.requires_grad and id(p) not in fed_ids
                          for p in model.parameters())

    optimizer.zero_grad(set_to_none=False)

    if has_local_grads:
        model.train()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
    else:
        with _forward_mode(model, dropout=True), torch.no_grad():
            try:
                logits = model(x)
                loss = criterion(logits, y)
            except (RuntimeError, ValueError):     # diagnostic only, see above
                logits = torch.full((x.shape[0], model.classifier.out_features),
                                    float("nan"))
                loss = torch.tensor(float("nan"))

    g_tilde = privatized_fed_grads(model, x, y, criterion, groups, C, sigma,
                                   generator, vectorized=vectorized, dropout=True)

    for (name, p, _) in groups:
        p.grad = g_tilde[name].detach().clone()
    optimizer.step()

    if apply_constraints and hasattr(model, "apply_constraints"):
        model.apply_constraints()
    if hasattr(model, "project_stiefel"):
        model.project_stiefel()

    return float(loss.detach()), logits.detach()