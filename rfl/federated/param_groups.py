"""Identification of the parameter groups that are not federated, and freezing helpers.

Two groups can stay on the client:

* Normalisation layers (FedBN). Their parameters (affine) and buffers (running
  statistics) are never aggregated. DP runs refuse them
  (dp_grad.assert_dp_safe_norm_layers), so this only concerns non-private runs.
* The classifier head, in local-head mode (FedPer, `FLConfig.fl_local_fc`). It is
  removed from the uploaded state dict.

A module is a local normalisation layer when

1. it is a torch BatchNorm, or
2. it is a custom class whose name contains 'batchnorm' (such as yetanotherspdnet's
   BatchNormSPDMean).

`identify_batchnorm_keys` in rfl/federated/aggregation.py applies the matching rule to
state-dict keys, and the two must agree. rfl/models/eegnet.py names its BatchNorm layers
`batchnorm_*` accordingly. This module is imported by both the non-private client and
the DP code, so it depends on neither.
"""

from __future__ import annotations

from typing import Iterable, Set

import torch.nn as nn

__all__ = [
    "identify_bn_param_names",
    "identify_bn_buffer_names",
    "identify_classifier_param_names",
    "freeze_trunk",
    "freeze_params",
]

_BN_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def _is_local_norm(module: nn.Module) -> bool:
    """True for a normalisation layer that stays on the client (rules 1 and 2 above)."""
    if isinstance(module, _BN_TYPES):
        return True
    return "batchnorm" in type(module).__name__.lower()


def identify_bn_param_names(model: nn.Module) -> Set[str]:
    """Return the parameter names of local normalisation layers.

    The parameters of their child modules are included. yetanotherspdnet's SPD batch
    norm keeps its bias in a parametrization (`batch_norm.parametrizations.Covbias.original`).
    """
    names: Set[str] = set()
    for mod_name, module in model.named_modules():
        if _is_local_norm(module):
            for p_name, _ in module.named_parameters():
                names.add(f"{mod_name}.{p_name}" if mod_name else p_name)
    return names


def identify_bn_buffer_names(model: nn.Module) -> Set[str]:
    """Return the buffer names (running_mean, running_var, num_batches_tracked) of local normalisation layers, child modules included."""
    names: Set[str] = set()
    for mod_name, module in model.named_modules():
        if _is_local_norm(module):
            for b_name, buf in module.named_buffers():
                if buf is not None:
                    names.add(f"{mod_name}.{b_name}" if mod_name else b_name)
    return names


def identify_classifier_param_names(model: nn.Module) -> Set[str]:
    """Return the parameter names of the classifier head.

    Uses the nn.Linear submodule named 'classifier' (EEGNet and SPDNet both have one).
    Otherwise falls back to the last nn.Linear in the module tree.

    Returns:
        The head parameter names, or an empty set if the model has no nn.Linear.
    """
    names: Set[str] = set()

    for mod_name, module in model.named_modules():
        if mod_name.split(".")[-1] == "classifier" and isinstance(module, nn.Linear):
            for p_name, _ in module.named_parameters(recurse=False):
                names.add(f"{mod_name}.{p_name}")
            return names

    last_linear = None
    for mod_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = (mod_name, module)
    if last_linear is not None:
        mod_name, module = last_linear
        for p_name, _ in module.named_parameters(recurse=False):
            names.add(f"{mod_name}.{p_name}" if mod_name else p_name)
    return names


def freeze_params(model: nn.Module, names: Iterable[str]) -> int:
    """Set requires_grad=False on the named parameters.

    Returns:
        The number of parameter tensors that were trainable and are now frozen.
    """
    wanted = set(names)
    n = 0
    for name, p in model.named_parameters():
        if name in wanted and p.requires_grad:
            p.requires_grad_(False)
            n += 1
    return n


def freeze_trunk(model: nn.Module) -> int:
    """Set requires_grad=False on every parameter except the classifier head.

    The random-feature control of both pipelines (`config.freeze_trunk`).

    Returns:
        The number of frozen parameter tensors.

    Raises:
        ValueError: if the model has no classifier head.
    """
    head = identify_classifier_param_names(model)
    if not head:
        raise ValueError("freeze_trunk: the model has no classifier head to train.")
    trunk = [name for name, _ in model.named_parameters() if name not in head]
    return freeze_params(model, trunk)
