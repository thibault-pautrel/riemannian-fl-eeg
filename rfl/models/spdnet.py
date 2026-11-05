"""SPDNet for EEG motor-imagery and synthetic SPD classification, with L blocks.

Architecture::

    SPD input (d x d)
      -> [trace normalisation]
      -> BiMap(d -> h_1)    Sigma -> W^T Sigma W,  W in St(d, h_1)
      -> ReEig(eps_1) -> [SPD batch norm]
      -> BiMap(h_1 -> h_2) -> ReEig(eps_2) -> [SPD batch norm] -> ...   (L >= 2 only)
      -> LogEig -> Vech
      -> Linear(h_L (h_L + 1) / 2, n_classes)

The ReEig thresholds must strictly increase with depth. ReEig is
sigma(x) = max(eps, x), so after block 1 every eigenvalue is >= eps_1. By Poincare's
separation theorem a congruence by a semi-orthogonal W_2 keeps the eigenvalues inside
[lambda_min, lambda_max], so they stay >= eps_1. If eps_2 <= eps_1 the second ReEig is
the identity and the stack collapses to a single block (Oswald and Massart,
arXiv:2606.02490, Prop. 2 and Cor. 2). eps_2 > eps_1 is necessary for depth to add
expressivity, but not sufficient: eigenvalues must actually fall in [eps_1, eps_2).
`validate_epsilons` rejects a non-increasing schedule. `track_clipping` measures how
much each ReEig changes its input (see `SPDNet.reeig_activity`); the federated runner
records it on the validation split at the end of each run.

The argument needs the output of a block to be the input of the next BiMap. A batch
norm between two blocks (`use_batch_norm` with bn_position 'after_reeig', or
'before_reeig' with L >= 2) re-centres the spectrum, so the eigenvalues entering block
2 are no longer bounded below by eps_1. The schedule is then not checked, and equal
floors are allowed.

State-dict names. Block 1 uses the names `bimap` and `reig`, later blocks `bimap_2`,
`reig_2`, ... A depth-1 model therefore has the keys `bimap.weight`,
`classifier.weight`, `classifier.bias`, which existing checkpoints use. Every BiMap key
contains "bimap", which is how rfl/federated/aggregation.py finds the Stiefel parameters.

Stiefel constraint (`stiefel`). Two ways, both from yetanotherspdnet:

* 'projection' (default). BiMap weights are unparametrized (`BiMap(parametrized=False)`),
  so the library neither projects their gradient nor retracts them. The non-private
  training loops take an optimiser step on the Euclidean gradient, then restore
  orthonormality with the polar factor (`SPDNet.project_stiefel`). The private path
  first projects each per-sample gradient on the tangent space
  (rfl/federated/privacy/dp_grad.py). DP needs this mode.
* 'trivialization'. `BiMap(parametrized=True)`, the library default and the choice of
  Gallet et al.: PyTorch's orthogonal parametrization (Householder map for a
  rectangular W, dynamic trivialization with a base point). The optimiser runs on an
  unconstrained matrix and W is orthonormal at every step. The state dict of the model
  still holds W itself under `bimap.weight` (`state_dict` and `load_state_dict` below),
  so aggregation, FedBN and the saved files work as with 'projection'. Loading a W
  re-centres the parametrization on it (PyTorch `right_inverse`): the base point becomes
  W, completed by random orthonormal columns drawn from a fixed seed. Each client thus
  starts every round from a fresh chart centred on the global W.

SPD batch norm (`use_batch_norm`), after Gallet, Mian, Bouchard and Ginolhac, "On
batch normalization for SPDnet" (Signal Processing, 2027). One layer per block, from
yetanotherspdnet:
`BatchNormSPDMean` (bn_type 'mean_only': centring on the batch SPD mean, then a learned
SPD bias) or `BatchNormSPDMeanScalarVariance` ('mean_var_scalar': also a scalar
dispersion). As in the library's own SPDnet, the layer sits after the ReEig of each
block (bn_position 'after_reeig'). 'before_reeig' places it between the BiMap and the
ReEig, as in SPDNetBN (Brooks et al., NeurIPS 2019). The output of the batch norm has
its mean at the learned bias, near the identity, so with 'before_reeig' the ReEig
floors calibrated on trace-normalised spectra are far too small and must be
recalibrated. The same holds for the floors of blocks 2 and later with 'after_reeig':
they see the output of a batch norm, not trace-normalised matrices (the paper uses
floors of 0.01 to 0.8 with batch norm). The layers are those of yetanotherspdnet,
used as they are, and run in float64, as the library recommends: the input is cast
to float64 before the layer and back to float32 after it. Block 1 uses the name
`batch_norm`, later blocks `batch_norm_2`, ... Every key contains "batch_norm", which is
how rfl/federated/aggregation.py keeps these layers on the client (FedBN). DP refuses them,
since their running mean is updated on the private batch.

Convention (as in rfl/geometry/stiefel.py and yetanotherspdnet): a BiMap weight is tall,
of shape (n_in, n_out) with n_out <= n_in and W^T W = I. The model has no dropout and
no constraint on the classifier.

Reference: https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from rfl.geometry.stiefel import polar_projection

logger = logging.getLogger(__name__)

try:
    from yetanotherspdnet.nn import BiMap, LogEig, ReEig, Vech
    try:
        from yetanotherspdnet.nn import BatchNormSPDMean, BatchNormSPDMeanScalarVariance
        BatchNormSPD = BatchNormSPDMean                  # base class of both layers
        BATCHNORM_AVAILABLE = True
    except ImportError:
        BatchNormSPD = BatchNormSPDMean = BatchNormSPDMeanScalarVariance = None
        BATCHNORM_AVAILABLE = False
        warnings.warn("BatchNormSPDMean not found; SPD batch norm unavailable.")
    SPDNET_AVAILABLE = True
except ImportError:
    BiMap = ReEig = LogEig = Vech = None
    BatchNormSPD = BatchNormSPDMean = BatchNormSPDMeanScalarVariance = None
    SPDNET_AVAILABLE = False
    BATCHNORM_AVAILABLE = False
    warnings.warn("yetanotherspdnet not found; SPDNet models unavailable.")

# Options of the SPD batch norm, with their defaults. The library's own SPDnet uses the
# same mean (geometric-arithmetic-harmonic) and position. The momentum is 0.1 rather
# than the library's 0.01: a federated round runs only a few local steps, and with 0.01
# the running mean used at evaluation would trail the batch means for many rounds.
BN_DEFAULTS = {
    "bn_type": "mean_only",                          # 'mean_only' | 'mean_var_scalar'
    "bn_mean_type": "geometric_arithmetic_harmonic",
    "bn_momentum": 0.1,
    "bn_norm_and_bias_mode": "whitening",            # 'whitening' | 'parallel_transport'
    "bn_position": "after_reeig",                    # 'after_reeig' | 'before_reeig'
}
BN_TYPES = ("mean_only", "mean_var_scalar")
BN_MEAN_TYPES = ("affine_invariant", "log_euclidean", "arithmetic", "harmonic",
                 "geometric_arithmetic_harmonic",
                 "adaptive_geometric_arithmetic_harmonic", "bures_wasserstein")
BN_POSITIONS = ("after_reeig", "before_reeig")
BN_NORM_AND_BIAS_MODES = ("whitening", "parallel_transport")


def validate_bn_options(bn_type: str, bn_mean_type: str, bn_momentum: float,
                        bn_norm_and_bias_mode: str, bn_position: str) -> None:
    """Check the SPD batch-norm options (see BN_DEFAULTS).

    Raises:
        ValueError: on an unknown value or a combination the library does not offer.
    """
    if bn_type not in BN_TYPES:
        raise ValueError(f"bn_type must be one of {BN_TYPES}, got {bn_type!r}")
    if bn_mean_type not in BN_MEAN_TYPES:
        raise ValueError(f"bn_mean_type must be one of {BN_MEAN_TYPES}, got "
                         f"{bn_mean_type!r}")
    if not 0.0 < float(bn_momentum) <= 1.0:
        raise ValueError(f"bn_momentum must lie in (0, 1], got {bn_momentum}")
    if bn_norm_and_bias_mode not in BN_NORM_AND_BIAS_MODES:
        raise ValueError(f"bn_norm_and_bias_mode must be one of "
                         f"{BN_NORM_AND_BIAS_MODES}, got {bn_norm_and_bias_mode!r}")
    if bn_position not in BN_POSITIONS:
        raise ValueError(f"bn_position must be one of {BN_POSITIONS}, got "
                         f"{bn_position!r}")
    if bn_type == "mean_var_scalar" and bn_mean_type == "adaptive_geometric_arithmetic_harmonic":
        raise ValueError("bn_type='mean_var_scalar' has no dispersion for the adaptive "
                         "GAH mean in yetanotherspdnet. Use another mean or 'mean_only'.")
    if bn_norm_and_bias_mode == "parallel_transport" and (
            bn_type == "mean_var_scalar" or bn_mean_type == "bures_wasserstein"):
        raise ValueError("norm_and_bias_mode='parallel_transport' is available for "
                         "bn_type='mean_only' and a mean other than bures_wasserstein.")


def parse_int_list(spec, default=None) -> Optional[List[int]]:
    """Parse '16,8' or [16, 8] into [16, 8]. Returns `default` for None or ''."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return default
    if isinstance(spec, str):
        return [int(float(v)) for v in spec.replace(";", ",").split(",") if v.strip()]
    return [int(v) for v in spec]


def parse_float_list(spec, default=None) -> Optional[List[float]]:
    """Parse '1e-4,1e-2' or [1e-4, 1e-2] into floats. Returns `default` for None or ''."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return default
    if isinstance(spec, str):
        return [float(v) for v in spec.replace(";", ",").split(",") if v.strip()]
    return [float(v) for v in spec]


def validate_epsilons(epsilons: Sequence[float], strict: bool = True) -> None:
    """Check that ReEig thresholds strictly increase across blocks.

    With semi-orthogonal BiMap weights, a threshold that does not exceed the previous
    one makes that ReEig the identity (see the module docstring).

    Args:
        epsilons: One threshold per block.
        strict: Raise if True, otherwise only warn.

    Raises:
        ValueError: If `strict` and the schedule is not strictly increasing.
    """
    for i in range(1, len(epsilons)):
        if epsilons[i] <= epsilons[i - 1]:
            msg = (f"ReEig thresholds must strictly increase: eps_{i+1}="
                   f"{epsilons[i]:g} <= eps_{i}={epsilons[i-1]:g}. With semi-orthogonal "
                   "BiMap weights every eigenvalue after block %d is already >= %g "
                   "(Poincare separation), so this ReEig is the identity and the stack "
                   "collapses to a single block." % (i, epsilons[i - 1]))
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)


if SPDNET_AVAILABLE:

    class SPDNet(nn.Module):
        """SPDNet: (BiMap -> ReEig) x L -> [BatchNormSPD] -> LogEig -> Vech -> Linear.

        Args:
            input_dim: Dimension d of the input SPD matrices.
            n_classes: Number of output classes.
            hidden_dims: BiMap output dimension per block, non-increasing and all
                <= input_dim. An int gives a single block.
            use_batch_norm: Add one SPD batch norm per block (module docstring). Not
                allowed under DP, because its running mean would be updated on the
                private batch (rfl/federated/privacy/dp_grad.py).
            epsilons: ReEig threshold per block, strictly increasing. A scalar is
                broadcast to every block, which only passes validation at depth 1.
            seed: Seed of the BiMap initialisation. None uses the global RNG.
            trace_normalize: Scale each input to unit trace.
            track_clipping: Record the relative change made by each ReEig (one
                Frobenius norm per block per forward).
            strict_epsilons: Raise rather than warn on a non-increasing schedule.
            bn_type, bn_mean_type, bn_momentum, bn_norm_and_bias_mode, bn_position:
                SPD batch-norm options, see BN_DEFAULTS. Used with use_batch_norm.
            stiefel: 'projection' or 'trivialization' (module docstring).

        Raises:
            ValueError: On empty `hidden_dims`, an increasing dimension, a length
                mismatch between `hidden_dims` and `epsilons`, or (when strict) a
                non-increasing threshold schedule.
        """

        def __init__(
            self,
            input_dim: int,
            n_classes: int,
            hidden_dims=16,
            use_batch_norm: bool = False,
            epsilons=1e-4,
            seed: Optional[int] = 42,
            trace_normalize: bool = True,
            track_clipping: bool = False,
            strict_epsilons: bool = True,
            bn_type: str = BN_DEFAULTS["bn_type"],
            bn_mean_type: str = BN_DEFAULTS["bn_mean_type"],
            bn_momentum: float = BN_DEFAULTS["bn_momentum"],
            bn_norm_and_bias_mode: str = BN_DEFAULTS["bn_norm_and_bias_mode"],
            bn_position: str = BN_DEFAULTS["bn_position"],
            stiefel: str = "projection",
        ):
            super().__init__()
            if stiefel not in ("projection", "trivialization"):
                raise ValueError("stiefel must be 'projection' or 'trivialization', got "
                                 f"{stiefel!r}")
            self.stiefel = stiefel

            dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
            if not dims:
                raise ValueError("hidden_dims must contain at least one block")
            eps = ([float(epsilons)] * len(dims) if isinstance(epsilons, (int, float))
                   else [float(e) for e in epsilons])
            if len(eps) != len(dims):
                raise ValueError(
                    f"epsilons has {len(eps)} entries for {len(dims)} blocks")

            prev = input_dim
            for i, h in enumerate(dims):
                if h > prev:
                    raise ValueError(
                        f"hidden_dims must be non-increasing and <= input_dim: block "
                        f"{i+1} maps {prev} -> {h}, but BiMap requires n_out <= n_in.")
                prev = h
            if use_batch_norm and not BATCHNORM_AVAILABLE:
                warnings.warn("BatchNormSPDMean requested but unavailable; disabling.")
                use_batch_norm = False
            # A batch norm between blocks breaks the separation argument (docstring).
            if not use_batch_norm:
                validate_epsilons(eps, strict=strict_epsilons)

            self.input_dim = input_dim
            self.n_classes = n_classes
            self.hidden_dims = dims
            self.hidden_dim = dims[-1]          # output dimension of the last block
            self.epsilons = eps
            self.epsilon = eps[0]
            self.depth = len(dims)
            self.seed = seed
            self.trace_normalize = trace_normalize
            self.track_clipping = track_clipping
            self._clip_activity: List[float] = [0.0] * self.depth
            self._clip_calls = 0

            self.use_batch_norm = bool(use_batch_norm)
            self.bn_position = bn_position
            if self.use_batch_norm:
                validate_bn_options(bn_type, bn_mean_type, bn_momentum,
                                    bn_norm_and_bias_mode, bn_position)
                self.bn_options = dict(bn_type=bn_type, bn_mean_type=bn_mean_type,
                                       bn_momentum=float(bn_momentum),
                                       bn_norm_and_bias_mode=bn_norm_and_bias_mode,
                                       bn_position=bn_position)
            else:
                self.bn_options = {}

            generator = None
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(seed)

            # Block 1 uses the flat names `bimap`/`reig` (see the module docstring).
            n_in = input_dim
            for i, h in enumerate(dims):
                suffix = "" if i == 0 else f"_{i+1}"
                # The parametrization draws the completion of its base point from the
                # global RNG: a fixed seed keeps it reproducible and leaves the global
                # stream (data order) as in 'projection' mode.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(0 if seed is None else seed)
                    bimap = BiMap(n_in=n_in, n_out=h,
                                  parametrized=(stiefel == "trivialization"),
                                  dtype=torch.float32, generator=generator)
                # torch.Generator is not picklable on some torch versions, and
                # FLClient.set_model deep-copies the model.
                if getattr(bimap, "generator", None) is not None:
                    bimap.generator = None
                setattr(self, f"bimap{suffix}", bimap)
                setattr(self, f"reig{suffix}", ReEig(eps=eps[i], dim=h))
                setattr(self, f"batch_norm{suffix}",
                        self._make_batch_norm(h) if self.use_batch_norm else None)
                n_in = h

            self.logeig = LogEig()
            self.vech = Vech()

            # Measure the Vech width, h (h + 1) / 2, on a probe. Batch norm is skipped
            # so that its running statistics are untouched at initialisation.
            with torch.no_grad():
                probe = torch.eye(dims[-1], dtype=torch.float32).unsqueeze(0)
                self.feature_dim = int(self.vech(self.logeig(probe)).shape[-1])
            self.classifier = nn.Linear(self.feature_dim, n_classes, dtype=torch.float32)

        def _make_batch_norm(self, n_features: int) -> nn.Module:
            """One SPD batch-norm layer, in float64 (module docstring)."""
            o = self.bn_options
            common = dict(n_features=n_features, mean_type=o["bn_mean_type"],
                          momentum=o["bn_momentum"], dtype=torch.float64)
            if o["bn_type"] == "mean_only":
                return BatchNormSPDMean(norm_and_bias_mode=o["bn_norm_and_bias_mode"],
                                        **common)
            return BatchNormSPDMeanScalarVariance(**common)

        def _block(self, i: int):
            """(BiMap, ReEig) modules of block i (0-based)."""
            suffix = "" if i == 0 else f"_{i+1}"
            return getattr(self, f"bimap{suffix}"), getattr(self, f"reig{suffix}")

        def batch_norms(self) -> List[nn.Module]:
            """SPD batch-norm modules in forward order (empty without batch norm)."""
            if not self.use_batch_norm:
                return []
            return [getattr(self, "batch_norm" + ("" if i == 0 else f"_{i+1}"))
                    for i in range(self.depth)]

        def _bn(self, i: int, x: torch.Tensor) -> torch.Tensor:
            """Apply the batch norm of block i in float64, return float32."""
            bn = getattr(self, "batch_norm" + ("" if i == 0 else f"_{i+1}"))
            return self._f32(bn(x.double()))

        def bimaps(self) -> List[nn.Module]:
            """BiMap modules in forward order."""
            return [self._block(i)[0] for i in range(self.depth)]

        @staticmethod
        def _f32(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == torch.float32 else x.float()

        @torch.no_grad()
        def project_stiefel(self) -> None:
            """Replace every BiMap weight W by its polar factor uf(W).

            Called after each optimiser step by both training loops. The map is
            data-independent, so under DP it is post-processing. Frozen BiMap weights
            (requires_grad=False) are skipped: they are orthonormal already. Nothing to
            do with stiefel='trivialization', where W is orthonormal by construction.
            """
            if self.stiefel == "trivialization":
                return
            for bimap in self.bimaps():
                if bimap.weight.requires_grad:      # a frozen BiMap stays exactly as it is
                    bimap.weight.data.copy_(polar_projection(bimap.weight.data))

        # ---- state dict with W itself under `bimap.weight` (stiefel='trivialization')
        def _bimap_names(self) -> List[str]:
            return ["bimap" + ("" if i == 0 else f"_{i+1}") for i in range(self.depth)]

        def state_dict(self, *args, **kwargs):
            """The usual state dict. With stiefel='trivialization', each BiMap appears as
            `<name>.weight` = W (orthonormal) instead of its parametrization tensors."""
            sd = super().state_dict(*args, **kwargs)
            if self.stiefel != "trivialization":
                return sd
            prefix = kwargs.get("prefix", "")
            keep_vars = kwargs.get("keep_vars", False)
            raw = {f"{prefix}{n}.parametrizations.weight.": n for n in self._bimap_names()}
            out = type(sd)()
            for k, v in sd.items():
                head = next((r for r in raw if k.startswith(r)), None)
                if head is None:
                    out[k] = v
                elif k == head + "original":          # W takes the place of its chart
                    w = getattr(self, raw[head]).weight
                    out[f"{prefix}{raw[head]}.weight"] = w if keep_vars else w.detach()
            if hasattr(sd, "_metadata"):
                out._metadata = sd._metadata
            return out

        def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
            """Load a state dict of `state_dict`. With stiefel='trivialization', each
            `<name>.weight` = W re-centres the parametrization on W (module docstring)."""
            if self.stiefel != "trivialization":
                return super().load_state_dict(state_dict, strict=strict, assign=assign)
            sd = dict(state_dict)
            weights = {n: sd.pop(f"{n}.weight") for n in self._bimap_names()
                       if f"{n}.weight" in sd}
            current = super().state_dict()
            for n in weights:          # placeholders, replaced by the assignment below
                for k, v in current.items():
                    if k.startswith(f"{n}.parametrizations.weight."):
                        sd.setdefault(k, v)
            result = super().load_state_dict(sd, strict=strict, assign=assign)
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                for n, w in weights.items():
                    bimap = getattr(self, n)
                    ref = bimap.parametrizations.weight.original
                    bimap.weight = w.to(device=ref.device, dtype=ref.dtype)
            return result

        def reset_clip_stats(self) -> None:
            """Reset the statistics returned by `reeig_activity`."""
            self._clip_activity = [0.0] * self.depth
            self._clip_calls = 0

        def reeig_activity(self) -> List[float]:
            """Mean relative change ||ReEig(X) - X||_F / ||X||_F per block.

            Zero means the ReEig of that block acts as the identity on the data seen.
            Requires `track_clipping=True`, otherwise returns NaN per block. The
            Frobenius norms are taken over the whole batch.
            """
            if not self._clip_calls:
                return [float("nan")] * self.depth
            return [a / self._clip_calls for a in self._clip_activity]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Map a batch of SPD matrices (B, d, d) to logits (B, n_classes)."""
            x = self._f32(x)
            if self.trace_normalize:
                trace = x.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-8)
                x = x / trace.unsqueeze(-1).unsqueeze(-1)

            for i in range(self.depth):
                bimap, reig = self._block(i)
                x = self._f32(bimap(x))
                if self.use_batch_norm and self.bn_position == "before_reeig":
                    x = self._bn(i, x)
                if self.track_clipping:
                    before = x
                    x = self._f32(reig(x))
                    with torch.no_grad():
                        num = torch.norm(x - before)
                        den = torch.norm(before).clamp(min=1e-12)
                        self._clip_activity[i] += float(num / den)
                else:
                    x = self._f32(reig(x))
                if self.use_batch_norm and self.bn_position == "after_reeig":
                    x = self._bn(i, x)
            if self.track_clipping:
                self._clip_calls += 1

            x = self._f32(self.logeig(x))
            x = self._f32(self.vech(x))
            return self.classifier(x)

    def create_spdnet(
        n_classes: int,
        n_channels: int,
        hidden_dim: Optional[int] = None,
        use_batch_norm: bool = False,
        epsilon: float = 1e-4,
        seed: Optional[int] = 42,
        trace_normalize: bool = True,
        hidden_dims=None,
        epsilons=None,
        track_clipping: bool = False,
        strict_epsilons: bool = True,
        **bn_options,
    ) -> "SPDNet":
        """Build an SPDNet.

        The list arguments `hidden_dims` and `epsilons` (lists or comma-separated
        strings) override the scalars `hidden_dim` and `epsilon` and select a deeper
        model. The scalars are what `ExperimentConfig.get_model_params` passes.

        Args:
            n_classes: Number of output classes.
            n_channels: Input dimension d (number of EEG channels).
            hidden_dim: Single-block output dimension. Defaults to
                max(2, n_channels // 2).
            use_batch_norm: See `SPDNet`.
            epsilon: ReEig threshold, broadcast to every block when `epsilons` is
                not given.
            seed: See `SPDNet`.
            trace_normalize: See `SPDNet`.
            hidden_dims: Per-block output dimensions.
            epsilons: Per-block ReEig thresholds.
            track_clipping: See `SPDNet`.
            strict_epsilons: See `SPDNet`.
            **bn_options: SPD batch-norm options of `SPDNet` (bn_type, bn_mean_type,
                bn_momentum, bn_norm_and_bias_mode, bn_position).

        Returns:
            An `SPDNet`.

        Raises:
            ValueError: If several blocks are requested with a single threshold, or
                if `SPDNet` rejects the configuration.
        """
        dims = parse_int_list(hidden_dims, None)
        if dims is None:
            if hidden_dim is None:
                hidden_dim = max(2, n_channels // 2)
                logger.info(f"[SPDNet] hidden_dim not given; using {n_channels} -> {hidden_dim}")
            dims = [int(hidden_dim)]

        eps = parse_float_list(epsilons, None)
        if eps is None:
            eps = [float(epsilon)] * len(dims)
        if len(eps) == 1 and len(dims) > 1 and not use_batch_norm:
            raise ValueError(
                f"{len(dims)} blocks but a single ReEig threshold: with semi-orthogonal "
                "weights the later ReEig layers would be the identity and depth would "
                "buy nothing. Pass one strictly increasing threshold per block.")

        return SPDNet(
            input_dim=n_channels, n_classes=n_classes, hidden_dims=dims,
            use_batch_norm=use_batch_norm, epsilons=eps, seed=seed,
            trace_normalize=trace_normalize, track_clipping=track_clipping,
            strict_epsilons=strict_epsilons, **bn_options,
        )

    def fix_batch_norm_dtype(model: nn.Module, device: torch.device) -> None:
        """Move SPD batch-norm parameters and buffers to `device` in float64, in place.

        The layers compute in float64 (module docstring). Child modules are included:
        the bias lives in a parametrization.
        """
        if not BATCHNORM_AVAILABLE:
            return
        for module in model.modules():
            if BatchNormSPD is not None and isinstance(module, BatchNormSPD):
                for sub in module.modules():
                    for name, buf in list(sub.named_buffers(recurse=False)):
                        if buf is not None and torch.is_floating_point(buf):
                            setattr(sub, name, buf.to(device).double())
                    for p in sub.parameters(recurse=False):
                        p.data = p.data.to(device).double()

else:
    def create_spdnet(*args, **kwargs):
        """Unavailable: yetanotherspdnet is not installed."""
        raise ImportError("yetanotherspdnet is not installed.")

    def fix_batch_norm_dtype(*args, **kwargs):
        """No-op without yetanotherspdnet."""


