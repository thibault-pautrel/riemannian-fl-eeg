"""Server-side aggregation rules for SPDNet (Stiefel BiMap weights) and EEGNet.

With w_i the normalised client weights, W_i the client BiMap weights and W_t the
current global BiMap weight, the rules are:

    fedavg    sum_i w_i W_i                              no projection
    projavg   uf(sum_i w_i W_i)                          RFedProj
    rlavg     uf(W_t + sum_i w_i P_{W_t}(W_i - W_t))     RFedRL
    retavg    R_{W_t}(sum_i w_i R_{W_t}^{-1}(W_i))       PriRFed (Huang et al., Eq. 8)

Here uf is the orthogonal polar factor, P_W the tangent projection and R_W the polar
retraction on St(n, p) (rfl/geometry/stiefel.py). A BiMap weight is a tall matrix W of
shape (n_in, n_out) with W^T W = I, and the layer computes Sigma -> W^T Sigma W.

Only the last three rules map back onto the Stiefel manifold, and SPDNet uses one of
them (FLConfig refuses SPDNet with fedavg, whose extrinsic mean leaves the manifold).
fedavg is the rule of EEGNet, whose parameters are all Euclidean.

The rules are looked up by name in AGGREGATION_RULES (end of this module): the server,
FLConfig and the --aggregation option read it. A new mean on St(n, p) is a function

    mean(Ws, weights, base) -> W

of the client matrices Ws (each of shape (n_in, n_out)), their normalised weights and
the current global matrix `base` (W_t, for rules that need a base point). Registered
with `register_stiefel_mean("mymean", mean)`, it is applied to every BiMap weight, the
other parameters are averaged as above, FedBN and FedPer are unchanged, and the runs
take `--aggregation mymean`. Differential privacy is unaffected: the clients privatise
their updates and the server rule is post-processing.
`test_a_registered_mean_is_a_full_rule` (tests/test_aggregation.py) shows a mean
registered and checked against ProjAvg.

All non-Stiefel parameters (classifier, convolutions) are averaged in Euclidean space
under every rule.

Normalisation layers are local by default (FedBN, Li et al. 2021): their keys are left
out of the aggregated state dict, and clients keep their own values on receive
(rfl/federated/client.py). With `fedbn=False` (`FLConfig.fl_fedbn`) they are averaged in
Euclidean space like the other parameters, running statistics included. For an SPD
batch norm the server then replaces the running means by their mean in the geometry
of the layer (FLServer._share_spd_batchnorm). They are found from the key names because this module only sees
state dicts. rfl/federated/param_groups.py applies the matching rule to the module tree.

References: McMahan et al. (2017), Absil et al. (2008), Li et al. (2021),
Huang et al. (2026).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rfl.geometry.stiefel import (
    inverse_polar_retraction, lifting, polar_projection, retraction,
)

__all__ = [
    # re-exported so that callers need a single import
    "polar_projection", "lifting", "retraction", "inverse_polar_retraction",
    # aggregation rules
    "fedavg_aggregate", "projavg_aggregate", "rlavg_aggregate", "retavg_aggregate",
    # registry of the rules by name
    "AggregationRule", "AGGREGATION_RULES", "register_stiefel_mean", "stiefel_rule",
    "get_rule", "rule_names",
    # key and matrix identification
    "identify_stiefel_keys", "identify_batchnorm_keys",
    "extract_stiefel_matrices_from_model", "extract_stiefel_matrices_from_models",
]


# --------------------------------------------------------------------------- #
#  Key and matrix identification
# --------------------------------------------------------------------------- #
def identify_stiefel_keys(state_dict: Dict[str, torch.Tensor]) -> set:
    """Return the keys of BiMap Stiefel weights (names containing 'bimap', ending in '.weight')."""
    return {k for k in state_dict
            if "bimap" in k.lower() and k.endswith(".weight")}


def identify_batchnorm_keys(state_dict: Dict[str, torch.Tensor]) -> set:
    """Return the normalisation keys to exclude from aggregation (FedBN).

    Only key names are available here. A key is local when:

    * its name contains 'batchnorm', 'batch_norm' or '.bn', or
    * it ends in weight, bias, running_mean, running_var or num_batches_tracked and a
      sibling key holds running_mean or running_var.

    BiMap weights and the classifier match neither test.

    Args:
        state_dict: a client or global state dict.

    Returns:
        The set of keys that stay local.
    """
    bn_keys = set()
    suffixes = ("running_mean", "running_var", "num_batches_tracked", "weight", "bias")
    for key in state_dict:
        lowered = key.lower()
        if "batchnorm" in lowered or "batch_norm" in lowered or ".bn" in lowered:
            bn_keys.add(key)
        elif any(key.endswith(f".{s}") for s in suffixes):
            base = key.rsplit(".", 1)[0]
            if f"{base}.running_mean" in state_dict or f"{base}.running_var" in state_dict:
                bn_keys.add(key)
    return bn_keys


def extract_stiefel_matrices_from_model(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return detached copies of the BiMap weights of `model`, keyed as in its state dict.

    A module counts as a BiMap when its class name contains 'BiMap'.
    """
    out: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if "BiMap" in type(module).__name__:
            key = f"{name}.weight" if name else "weight"
            out[key] = module.weight.detach().clone()
    return out


def extract_stiefel_matrices_from_models(models: List[nn.Module]) -> List[Dict[str, torch.Tensor]]:
    """Apply `extract_stiefel_matrices_from_model` to each model."""
    return [extract_stiefel_matrices_from_model(m) for m in models]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _normalize_weights(weights: Optional[List[float]], n: int) -> List[float]:
    """Client weights summing to 1. Uniform when `weights` is None or sums to 0."""
    if n == 0:
        raise ValueError("No client updates to aggregate")
    if weights is None:
        return [1.0 / n] * n
    total = float(sum(weights))
    if total == 0.0:
        return [1.0 / n] * n
    return [w / total for w in weights]


def _weighted_average(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """Weighted sum of tensors. Integer tensors return the first client's value."""
    # Integer buffers such as num_batches_tracked are counters. Their weighted mean
    # would be a float and be truncated on load, so the first value is kept. The key
    # stays present and the dtype is preserved.
    if not tensors[0].is_floating_point():
        return tensors[0].clone()
    result = tensors[0].clone() * weights[0]
    for t, w in zip(tensors[1:], weights[1:]):
        result = result + t.clone() * w
    return result


def _local_keys(state_dict: Dict[str, torch.Tensor], fedbn: bool) -> set:
    """Keys left out of the aggregation: the normalisation keys under FedBN, else none."""
    return identify_batchnorm_keys(state_dict) if fedbn else set()


def _euclidean_keys(state_dict: Dict[str, torch.Tensor], bn_keys: set,
                    stiefel_keys: set) -> List[str]:
    """Keys averaged in Euclidean space, in state-dict order."""
    return [k for k in state_dict if k not in bn_keys and k not in stiefel_keys]


# --------------------------------------------------------------------------- #
#  Aggregation rules
# --------------------------------------------------------------------------- #
def fedavg_aggregate(
    client_state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    fedbn: bool = True,
) -> Dict[str, torch.Tensor]:
    """Weighted Euclidean average of every non-local key.

    BiMap weights are averaged like any other tensor and not re-projected, so for
    SPDNet the result generally lies off the Stiefel manifold.

    Args:
        client_state_dicts: one state dict per client, all with the same keys.
        weights: client weights, normalised internally. None means uniform.
        fedbn: leave the normalisation keys out (FedBN). False averages them.

    Returns:
        The aggregated state dict, without the local normalisation keys under FedBN.
    """
    n = len(client_state_dicts)
    weights = _normalize_weights(weights, n)
    bn_keys = _local_keys(client_state_dicts[0], fedbn)

    agg = OrderedDict()
    for name in client_state_dicts[0]:
        if name in bn_keys:
            continue
        agg[name] = _weighted_average([sd[name] for sd in client_state_dicts], weights)
    return agg


def projavg_aggregate(
    client_stiefel_matrices: List[Dict[str, torch.Tensor]],
    client_state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    fedbn: bool = True,
) -> Dict[str, torch.Tensor]:
    """ProjAvg: W_{t+1} = uf(sum_i w_i W_i) on the Stiefel keys.

    No base point is needed. The polar factor is well defined whenever the weighted
    average has full column rank. Other keys are averaged in Euclidean space.

    Args:
        client_stiefel_matrices: BiMap weights per client
            (from `extract_stiefel_matrices_from_model`).
        client_state_dicts: one state dict per client.
        weights: client weights, normalised internally. None means uniform.
        fedbn: leave the normalisation keys out (FedBN). False averages them.

    Returns:
        The aggregated state dict, without the local normalisation keys under FedBN.
    """
    n = len(client_stiefel_matrices)
    weights = _normalize_weights(weights, n)
    stiefel_keys = set(client_stiefel_matrices[0]) if client_stiefel_matrices else set()
    bn_keys = _local_keys(client_state_dicts[0], fedbn)

    agg = OrderedDict()
    for name in sorted(stiefel_keys):
        avg = _weighted_average([m[name] for m in client_stiefel_matrices], weights)
        agg[name] = polar_projection(avg)
    for name in _euclidean_keys(client_state_dicts[0], bn_keys, stiefel_keys):
        agg[name] = _weighted_average([sd[name] for sd in client_state_dicts], weights)
    return agg


def rlavg_aggregate(
    client_stiefel_matrices: List[Dict[str, torch.Tensor]],
    client_state_dicts: List[Dict[str, torch.Tensor]],
    global_model: nn.Module,
    weights: Optional[List[float]] = None,
    fedbn: bool = True,
) -> Dict[str, torch.Tensor]:
    """RLAvg: W_{t+1} = uf(W_t + sum_i w_i P_{W_t}(W_i - W_t)) on the Stiefel keys.

    Each client displacement is lifted to the tangent space at the current global
    point W_t (first-order lifting), averaged there and retracted. W_t is read from
    `global_model`. Other keys are averaged in Euclidean space.

    Args:
        client_stiefel_matrices: BiMap weights per client.
        client_state_dicts: one state dict per client.
        global_model: current global model, which provides W_t.
        weights: client weights, normalised internally. None means uniform.
        fedbn: leave the normalisation keys out (FedBN). False averages them.

    Returns:
        The aggregated state dict, without the local normalisation keys under FedBN.
    """
    n = len(client_stiefel_matrices)
    weights = _normalize_weights(weights, n)
    global_stiefel = extract_stiefel_matrices_from_model(global_model)
    stiefel_keys = set(global_stiefel)
    bn_keys = _local_keys(client_state_dicts[0], fedbn)

    agg = OrderedDict()
    for name in sorted(stiefel_keys):
        base = global_stiefel[name].clone()
        tangent = torch.zeros_like(base)
        for w, matrices in zip(weights, client_stiefel_matrices):
            tangent = tangent + w * lifting(base, matrices[name].clone())
        agg[name] = retraction(base, tangent)
    for name in _euclidean_keys(client_state_dicts[0], bn_keys, stiefel_keys):
        agg[name] = _weighted_average([sd[name] for sd in client_state_dicts], weights)
    return agg


def retavg_aggregate(
    client_stiefel_matrices: List[Dict[str, torch.Tensor]],
    client_state_dicts: List[Dict[str, torch.Tensor]],
    global_model: nn.Module,
    weights: Optional[List[float]] = None,
    fedbn: bool = True,
) -> Dict[str, torch.Tensor]:
    """RetAvg (PriRFed baseline): W_{t+1} = R_{W_t}(sum_i w_i R_{W_t}^{-1}(W_i)).

    Same structure as RLAvg, with the exact inverse polar retraction in place of the
    first-order lifting. Each inverse retraction solves a continuous Lyapunov equation as
    a dense p^2 x p^2 linear system, O(p^6) in the BiMap output dimension p, per client
    and per Stiefel key. This cost is
    what `agg_seconds` in the round history measures.

    Args:
        client_stiefel_matrices: BiMap weights per client.
        client_state_dicts: one state dict per client.
        global_model: current global model, which provides W_t.
        weights: client weights, normalised internally. None means uniform.
        fedbn: leave the normalisation keys out (FedBN). False averages them.

    Returns:
        The aggregated state dict, without the local normalisation keys under FedBN.

    Raises:
        ValueError: if a client weight lies outside the neighbourhood of W_t where the
            polar retraction is invertible (geometry.stiefel.inverse_polar_retraction).
    """
    n = len(client_stiefel_matrices)
    weights = _normalize_weights(weights, n)
    global_stiefel = extract_stiefel_matrices_from_model(global_model)
    stiefel_keys = set(global_stiefel)
    bn_keys = _local_keys(client_state_dicts[0], fedbn)

    agg = OrderedDict()
    for name in sorted(stiefel_keys):
        base = global_stiefel[name].clone()
        tangent = torch.zeros_like(base)
        for w, matrices in zip(weights, client_stiefel_matrices):
            tangent = tangent + w * inverse_polar_retraction(
                base, matrices[name].clone())
        agg[name] = retraction(base, tangent)
    for name in _euclidean_keys(client_state_dicts[0], bn_keys, stiefel_keys):
        agg[name] = _weighted_average([sd[name] for sd in client_state_dicts], weights)
    return agg


# --------------------------------------------------------------------------- #
#  Registry of the rules by name
# --------------------------------------------------------------------------- #
# mean(Ws, weights, base) -> W: weighted mean of the client matrices Ws on St(n, p).
# `weights` are normalised (sum 1). `base` is the current global matrix W_t.
StiefelMean = Callable[[List[torch.Tensor], List[float], torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AggregationRule:
    """A rule as the server calls it.

    `aggregate(client_stiefel_matrices, client_state_dicts, global_model, weights,
    fedbn)` returns the aggregated state dict. `stiefel`: the rule keeps the BiMap
    weights on St(n, p), so it suits SPDNet and not EEGNet (fedavg is the converse).
    """
    aggregate: Callable[..., Dict[str, torch.Tensor]]
    stiefel: bool = True
    description: str = ""


def stiefel_rule(mean: StiefelMean) -> Callable[..., Dict[str, torch.Tensor]]:
    """The full aggregation rule built from a mean on St(n, p).

    The mean is applied to each BiMap weight separately. Every other parameter is
    averaged in Euclidean space, and the normalisation keys stay local under FedBN,
    as in the rules above.
    """
    def aggregate(client_stiefel_matrices, client_state_dicts, global_model,
                  weights=None, fedbn=True):
        n = len(client_stiefel_matrices)
        w = _normalize_weights(weights, n)
        base = extract_stiefel_matrices_from_model(global_model)
        stiefel_keys = set(base)
        bn_keys = _local_keys(client_state_dicts[0], fedbn)
        agg = OrderedDict()
        for name in sorted(stiefel_keys):
            agg[name] = mean([m[name].clone() for m in client_stiefel_matrices], w,
                             base[name].clone())
        for name in _euclidean_keys(client_state_dicts[0], bn_keys, stiefel_keys):
            agg[name] = _weighted_average([sd[name] for sd in client_state_dicts], w)
        return agg
    return aggregate


AGGREGATION_RULES: Dict[str, AggregationRule] = {
    "fedavg": AggregationRule(
        lambda S, sds, g, w=None, fedbn=True: fedavg_aggregate(sds, w, fedbn=fedbn),
        stiefel=False, description="sum_i w_i x_i (EEGNet)"),
    "projavg": AggregationRule(
        lambda S, sds, g, w=None, fedbn=True: projavg_aggregate(S, sds, w, fedbn=fedbn),
        description="uf(sum_i w_i W_i) (RFedProj)"),
    "rlavg": AggregationRule(
        lambda S, sds, g, w=None, fedbn=True: rlavg_aggregate(S, sds, g, w, fedbn=fedbn),
        description="uf(W_t + sum_i w_i P_{W_t}(W_i - W_t)) (RFedRL)"),
    "retavg": AggregationRule(
        lambda S, sds, g, w=None, fedbn=True: retavg_aggregate(S, sds, g, w, fedbn=fedbn),
        description="R_{W_t}(sum_i w_i R_{W_t}^{-1}(W_i)) (PriRFed)"),
}


def register_stiefel_mean(name: str, mean: StiefelMean, description: str = "",
                          replace: bool = False) -> None:
    """Make `mean` available as the aggregation rule `name` (module docstring)."""
    if not name.isidentifier() or name != name.lower():
        raise ValueError(f"rule name {name!r}: lower case letters, digits and '_'")
    if name in AGGREGATION_RULES and not replace:
        raise ValueError(f"aggregation rule {name!r} exists (replace=True to replace it)")
    AGGREGATION_RULES[name] = AggregationRule(stiefel_rule(mean), True, description)


def get_rule(name: str) -> AggregationRule:
    try:
        return AGGREGATION_RULES[str(name).lower()]
    except KeyError:
        raise ValueError(f"unknown aggregation rule {name!r}; known: "
                         f"{', '.join(AGGREGATION_RULES)}") from None


def rule_names(stiefel: Optional[bool] = None) -> Tuple[str, ...]:
    """Names of the rules, all of them or only the Stiefel ones (True) or not (False)."""
    return tuple(k for k, r in AGGREGATION_RULES.items()
                 if stiefel is None or r.stiefel == stiefel)

