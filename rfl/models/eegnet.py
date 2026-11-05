"""EEGNet for EEG motor-imagery classification (Lawhern et al., 2018).

Follows the original Keras model, including dropout in both blocks and the two max-norm
constraints (depthwise spatial conv at 1.0, dense layer at `norm_rate`). These apply in
every setting: centralised, federated and DP.

Architecture::

    Block 1  temporal conv -> norm -> depthwise spatial conv -> norm -> ELU -> avgpool -> dropout
    Block 2  separable conv (depthwise + pointwise) -> norm -> ELU -> avgpool -> dropout
    Head     flatten -> Linear(n_classes)

`norm_type` selects the normalisation layer: 'none' (nn.Identity, no normalisation,
the default) or 'batch' (BatchNorm2d, as in the original
model). In federated runs BatchNorm stays on the client and is never aggregated (FedBN).
The federated code sees state dicts only, so it finds these layers by their module
names. The three slots ('temporal', 'spatial', 'separable') are named as follows::

    norm_type  module name        treatment
    none       identity_{slot}    no parameters, no buffers
    batch      batchnorm_{slot}   kept on the client (FedBN), non-private runs only

rfl/federated/aggregation.py and rfl/federated/param_groups.py rely on these names. Renaming
the modules changes the aggregation scope.

DP compatibility. BatchNorm statistics couple the samples of a batch, so they break the
per-sample sensitivity Delta = 2C/B (Appendix D.2 of the DP-RFedProj paper). DP
therefore refuses norm_type='batch'.
The DP per-sample pass runs in eval() and turns
dropout back on (dp_grad._forward_mode), so DP and non-private runs train with the same
dropout.

References:
    Lawhern et al. (2018), EEGNet: a compact convolutional neural network for
    EEG-based brain-computer interfaces. https://github.com/vlawhern/arl-eegmodels
    Li et al. (2021), FedBN.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["EEGNet", "create_eegnet", "NORM_TYPES"]

NORM_TYPES = ("none", "batch")


class EEGNet(nn.Module):
    """EEGNet.

    Input shape (batch, 1, n_channels, n_timepoints) or (batch, n_channels,
    n_timepoints). Output shape (batch, n_classes).

    Args:
        n_classes: Number of output classes.
        n_channels: Number of EEG channels.
        n_timepoints: Number of time samples per trial.
        dropout_rate: Dropout probability (0.5 in the original model).
        kern_length: Temporal kernel length, about 500 ms of samples.
        F1: Number of temporal filters.
        D: Depth multiplier of the spatial filters.
        F2: Number of pointwise filters. Defaults to F1 * D.
        norm_rate: Max-norm bound on the rows of the dense layer (0.25 originally).
        pool1: First temporal average-pooling size.
        pool2: Second temporal average-pooling size.
        separable_kernel: Temporal kernel of the depthwise separable conv.
        dropout_type: 'Dropout' or 'SpatialDropout2D'.
        norm_type: 'none' or 'batch' (see the module docstring).

    Raises:
        ValueError: On an unknown norm_type or dropout_type.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        n_timepoints: int,
        dropout_rate: float = 0.5,
        kern_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: Optional[int] = None,
        norm_rate: float = 0.25,
        pool1: int = 4,
        pool2: int = 8,
        separable_kernel: int = 16,
        dropout_type: str = 'Dropout',
        norm_type: str = 'none',
    ):
        super().__init__()

        F2 = F1 * D if F2 is None else F2

        norm_type = str(norm_type).lower()
        if norm_type not in NORM_TYPES:
            raise ValueError(f"norm_type must be one of {NORM_TYPES}, got {norm_type!r}")

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.F1, self.D, self.F2 = F1, D, F2
        self.norm_rate = norm_rate
        self.pool1, self.pool2 = pool1, pool2
        self.separable_kernel = separable_kernel
        self.norm_type = norm_type

        dt = dropout_type.lower()
        if "spatial" in dt:
            dropout_class = nn.Dropout2d
        elif dt == "dropout":
            dropout_class = nn.Dropout
        else:
            raise ValueError("dropout_type must be 'Dropout' or 'SpatialDropout2D'")

        # ---- Block 1: temporal conv -> depthwise spatial conv ----
        self.conv_temporal = nn.Conv2d(1, F1, kernel_size=(1, kern_length),
                                       padding='same', bias=False)
        self._norm_temporal_attr = self._make_norm("temporal", F1)

        self.conv_spatial = nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                                      groups=F1, bias=False)
        self._norm_spatial_attr = self._make_norm("spatial", F1 * D)
        self.activation_1 = nn.ELU()
        self.pool_1 = nn.AvgPool2d(kernel_size=(1, pool1))
        self.dropout_1 = dropout_class(dropout_rate)

        # ---- Block 2: separable conv (depthwise + pointwise) ----
        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, separable_kernel),
            padding='same', groups=F1 * D, bias=False)
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self._norm_separable_attr = self._make_norm("separable", F2)
        self.activation_2 = nn.ELU()
        self.pool_2 = nn.AvgPool2d(kernel_size=(1, pool2))
        self.dropout_2 = dropout_class(dropout_rate)

        # ---- Head ----
        self._calculate_feature_dim()
        self.classifier = nn.Linear(self.feature_dim, n_classes)

        self._initialize_weights()

    def _make_norm(self, slot: str, num_features: int) -> str:
        """Register the normalisation layer of `slot` and return its attribute name.

        The name encodes the type (see the module docstring). The module is registered
        once only, so its state-dict keys are not duplicated.
        """
        if self.norm_type == "none":
            attr = f"identity_{slot}"
            module: nn.Module = nn.Identity()
        else:
            attr = f"batchnorm_{slot}"
            module = nn.BatchNorm2d(num_features)
        setattr(self, attr, module)
        return attr

    @property
    def norm_temporal(self) -> nn.Module:
        return getattr(self, self._norm_temporal_attr)

    @property
    def norm_spatial(self) -> nn.Module:
        return getattr(self, self._norm_spatial_attr)

    @property
    def norm_separable(self) -> nn.Module:
        return getattr(self, self._norm_separable_attr)

    def norm_module_names(self) -> tuple:
        """The three normalisation attribute names, in forward order."""
        return (self._norm_temporal_attr, self._norm_spatial_attr,
                self._norm_separable_attr)

    def _initialize_weights(self):
        """Xavier-uniform conv weights, norm affine at (1, 0)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _calculate_feature_dim(self):
        """Measure the flattened feature width on a dummy input.

        In eval mode, so that the probe leaves the BatchNorm running statistics at
        their initial values.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            self.feature_dim = self._forward_features(dummy).numel()
        self.train(was_training)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extractor. Expects a 4-D input (B, 1, C, T)."""
        x = self.conv_temporal(x)
        x = self.norm_temporal(x)
        x = self.conv_spatial(x)
        x = self.norm_spatial(x)
        x = self.activation_1(x)
        x = self.pool_1(x)
        x = self.dropout_1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.norm_separable(x)
        x = self.activation_2(x)
        x = self.pool_2(x)
        x = self.dropout_2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, C, T) or (B, 1, C, T) trials to logits (B, n_classes)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)                      # (B, C, T) -> (B, 1, C, T)
        elif x.dim() != 4:
            raise ValueError("EEGNet expects (B, C, T) or (B, 1, C, T)")
        x = self._forward_features(x)
        return self.classifier(x.view(x.size(0), -1))

    def apply_constraints(self):
        """Apply the Keras max-norm constraints in place.

        Each output row of the depthwise spatial conv is renormalised to norm <= 1.0,
        and each row of the dense layer to norm <= norm_rate. Called after every
        optimiser step by both training loops. The map is data-independent, so under
        DP it is post-processing. Frozen parameters (requires_grad=False, see
        `freeze_trunk` and `fl_local_head_mode='frozen'`) are left as they are, so that
        they stay exactly at their initialisation.
        """
        with torch.no_grad():
            if self.conv_spatial.weight.requires_grad:
                self.conv_spatial.weight.data = torch.renorm(
                    self.conv_spatial.weight.data, p=2, dim=0, maxnorm=1.0)
            if self.classifier.weight.requires_grad:
                self.classifier.weight.data = torch.renorm(
                    self.classifier.weight.data, p=2, dim=0, maxnorm=self.norm_rate)

    def extra_repr(self) -> str:
        if self.norm_type == "none":
            return "norm=none (no normalisation layer)"
        return "norm=batch (local, FedBN)"


def create_eegnet(
    n_classes: int,
    n_channels: int,
    n_timepoints: int,
    sampling_rate: Optional[int] = None,
    F1: int = 8,
    D: int = 2,
    F2: Optional[int] = None,
    dropout_rate: float = 0.5,
    dropout_type: str = 'Dropout',
    kern_length: Optional[int] = None,
    separable_kernel: Optional[int] = None,
    pool1: int = 4,
    pool2: int = 8,
    norm_type: str = 'none',
    **kwargs,
) -> EEGNet:
    """Build an EEGNet, sizing the temporal kernels from the sampling rate.

    When not given, `kern_length` is max(round(0.5 fs), 32), a 500 ms window, and
    `separable_kernel` is max(round(fs / (2 pool1)), 8), a 500 ms window at the pooled
    rate fs / pool1. Without a sampling rate they default to 64 and 16 samples. Pass
    the rate the data was loaded at (128 Hz after the default resampling, which gives
    kern_length = 64).

    Args:
        n_classes: Number of output classes.
        n_channels: Number of EEG channels.
        n_timepoints: Number of time samples per trial.
        sampling_rate: Sampling rate of the input in Hz, used to size the kernels.
        F1, D, F2, dropout_rate, dropout_type, pool1, pool2, norm_type: See `EEGNet`.
        kern_length: Temporal kernel length. Derived from `sampling_rate` if None.
        separable_kernel: Separable-conv kernel. Derived from `sampling_rate` if None.
        **kwargs: Forwarded to `EEGNet` (for example `norm_rate`).

    Returns:
        An `EEGNet`.
    """
    if kern_length is None:
        if sampling_rate is not None:
            kern_length = max(int(round(0.5 * sampling_rate)), 32)
            logger.info(f"[EEGNet] fs={sampling_rate} Hz -> temporal kernel {kern_length} "
                  f"samples (~{kern_length / sampling_rate * 1000:.0f} ms)")
        else:
            kern_length = 64
            logger.info(f"[EEGNet] no sampling rate given; temporal kernel {kern_length} samples")

    if separable_kernel is None:
        if sampling_rate is not None:
            separable_kernel = max(int(round(sampling_rate / (2 * pool1))), 8)
            post_pool_fs = sampling_rate / pool1
            logger.info(f"[EEGNet] separable kernel {separable_kernel} samples "
                  f"(~{separable_kernel / post_pool_fs * 1000:.0f} ms post-pool)")
        else:
            separable_kernel = 16

    if str(norm_type).lower() == "batch":
        logger.info("[EEGNet] BatchNorm in all three slots (original model, non-private only)")

    return EEGNet(
        n_classes=n_classes,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        dropout_rate=dropout_rate,
        kern_length=kern_length,
        F1=F1,
        D=D,
        F2=F1 * D if F2 is None else F2,
        pool1=pool1,
        pool2=pool2,
        separable_kernel=separable_kernel,
        dropout_type=dropout_type,
        norm_type=norm_type,
        **kwargs,
    )
