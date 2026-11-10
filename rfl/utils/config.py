"""Base experiment configuration shared by the centralised and federated pipelines.

`ExperimentConfig` holds the data, model, training and protocol settings that both
pipelines use, so that a centralised run and a federated run at the same settings see
the same data, splits and model. rfl/federated/fl_config.py extends it with the federated
and DP settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Hyperparameters and settings for one experiment.

    `__post_init__` derives `experiment_dir` (save_root/experiment_name/run_name) and
    `config_hash`, then validates the fields (ValueError on invalid values).

    Example:
        >>> config = ExperimentConfig(
        ...     experiment_name='motor_imagery',
        ...     dataset='BNCI2014_001',
        ...     model_type='eegnet',
        ...     learning_rate=1e-3,
        ... )
        >>> config.save()
    """

    # ==================== Identification ====================
    experiment_name: str = 'default_experiment'
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    description: str = ''

    # ==================== Reproducibility ====================
    seed: int = 42                      # model init + training randomness
    data_split_seed: int = 42           # train/val/test split
    deterministic: bool = True

    # ==================== Dataset ====================
    dataset: str = 'BNCI2014_001'
    subjects: List[int] = field(default_factory=lambda: [1, 2, 3])
    # The synthetic .npz file of a synthetic run (absolute path), part of the hash.
    data_path: str = ""
    split_type: str = 'random'          # stratified random split, as in the federated runs
    # 'raw' (EEGNet) | 'covariance' (SPDNet on real EEG) | 'spd' (SPDNet on synthetic SPD
    # matrices). None derives it from model_type ('raw' or 'covariance').
    data_type: Optional[str] = None
    task: str = 'all_classes'           # 'binary' | 'all_classes' | 'no_rest'

    # Preprocessing
    resample: Optional[float] = 128.0   # target rate (Hz), <= 0 or None keeps native
    fmin: float = 8.0                   # band-pass low cutoff (Hz)
    fmax: float = 32.0                  # band-pass high cutoff (Hz)

    # Normalisation of the raw trials (EEGNet): 'per_trial', each trial and channel
    # standardised on its own (the papers), or 'per_trial_global', each trial
    # standardised over all its channels, which keeps the relative channel powers. Both
    # share no statistic between trials or splits, so they leak nothing from validation
    # or test and suit record-level DP.
    normalization_type: str = 'per_trial'

    # Covariance estimation (SPDNet, from raw EEG). Per trial, hence DP-compatible.
    cov_estimator: str = 'scm'          # 'scm' | 'lwf' (fixed shrinkage, see rfl/data/datasets.py)
    cov_shrinkage: float = 0.0          # shrinkage intensity for 'lwf', in (0, 1]
    cov_regularization: float = 0.0     # diagonal floor for strict positive definiteness

    # ==================== Data loading ====================
    batch_size: int = 64
    num_workers: int = 0
    val_size: float = 0.1
    test_size: float = 0.15
    # Data quantity (rfl/data/reduction.py): N calibration trials per class and subject,
    # drawn from the subject's training and validation trials (None: all trials).
    train_trials_per_class: Optional[int] = None
    calibration_val_fraction: float = 0.2      # share of the N trials for validation
    calibration_order: str = 'random'          # 'random' | 'chronological'
    reduced_subjects: int = 0                  # reduce K subjects only (0: all)

    # ==================== Model ====================
    model_type: str = 'spdnet'          # 'spdnet' | 'eegnet'

    # EEGNet. Dropout and the max-norm constraints are always active, including under DP.
    eegnet_F1: int = 8                  # temporal filters
    eegnet_D: int = 2                   # depth multiplier
    eegnet_F2: Optional[int] = None     # pointwise filters (default F1 * D)
    eegnet_kern_length: Optional[int] = None   # auto from the sampling rate if None
    eegnet_dropout: float = 0.5
    eegnet_dropout_type: str = 'Dropout'       # 'Dropout' | 'SpatialDropout2D'

    # EEGNet normalisation layer. See rfl/models/eegnet.py.
    #   'none'   no normalisation layer (default). Valid with and without DP.
    #   'batch'  BatchNorm, as in the original model. Non-private runs only: FLConfig
    #            refuses it under DP. In federated runs it stays on the client (FedBN).
    # Ignored for SPDNet.
    eegnet_norm_type: str = 'none'             # 'none' | 'batch'

    # SPDNet, one BiMap(d -> hidden_dim) block. None takes the value of the dataset in
    # rfl/data/registry.py (DATASET_DEFAULTS), else lets create_spdnet use
    # max(2, n_channels // 2). Deeper models: spdnet_hidden_dims and spdnet_epsilons.
    spdnet_hidden_dim: Optional[int] = None
    # SPD batch norm, one layer per block (rfl/models/spdnet.py). Rejected under DP: its
    # running mean is updated on the private batch. In federated runs it stays on the
    # client (FedBN).
    spdnet_use_batch_norm: bool = False
    spdnet_bn_type: str = 'mean_only'          # 'mean_only' | 'mean_var_scalar'
    spdnet_bn_mean_type: str = 'geometric_arithmetic_harmonic'   # see rfl/models/spdnet.py
    spdnet_bn_momentum: float = 0.1            # running-statistics update rate (also EEGNet)
    spdnet_bn_norm_and_bias_mode: str = 'whitening'   # 'whitening' | 'parallel_transport'
    spdnet_bn_position: str = 'after_reeig'    # 'after_reeig' | 'before_reeig'
    # After training, set the running statistics of every batch norm (SPD batch norm, and
    # EEGNet BatchNorm) to the exact statistics of the layer's inputs on the training
    # data (rfl/models/bn_recalibration.py), instead of the momentum average.
    spdnet_bn_recalibrate: bool = False
    spdnet_epsilon: Optional[float] = None     # ReEig floor. None: registry, else 1e-4
    spdnet_trace_normalize: bool = True
    # Several BiRe blocks, as comma-separated lists, for example "16,8" and "1e-4,1e-2",
    # one ReEig floor per block. Empty strings give the single-block model above. The
    # floors must strictly increase: with semi-orthogonal BiMap weights the spectrum
    # after a block lies inside the spectrum before it (Poincare separation), so a
    # non-increasing floor makes the next ReEig the identity and the blocks collapse
    # into one (Oswald & Massart, arXiv:2606.02490). With spdnet_use_batch_norm the batch
    # norm between blocks re-centres the spectrum, and the check is not applied.
    spdnet_hidden_dims: str = ""
    spdnet_epsilons: str = ""
    # How the BiMap weights stay on the Stiefel manifold (rfl/models/spdnet.py):
    #   'projection'      plain weights, an optimiser step on the Euclidean gradient,
    #                     then the polar factor uf(W) (default; the only choice under DP).
    #   'trivialization'  the library parametrization (BiMap(parametrized=True), PyTorch
    #                     orthogonal map, Householder by default), as in Gallet et al.:
    #                     the optimiser runs on an unconstrained matrix and W is always
    #                     orthonormal. State dicts still hold W under `bimap.weight`.
    spdnet_stiefel: str = 'projection'
    # Percentile q of --reeig-percentile, when the floors spdnet_epsilons come from the
    # floors file of analyze_covariances (None: floors given directly or by default).
    spdnet_reeig_percentile: Optional[float] = None

    # ==================== Training ====================
    epochs: int = 300                   # centralised only (FL runs use fl_rounds)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    # Optimiser of every training step (local steps, centralised epochs, head re-fit):
    # 'adam' or 'sgd'. sgd_momentum is the heavy-ball momentum of SGD (0: plain SGD).
    optimizer: str = 'adam'
    sgd_momentum: float = 0.0

    # Protocol shared by the centralised loop (rfl/training/train_loop.py) and the federated
    # server (rfl/federated/server.py, pooled over clients). One validation metric drives the
    # plateau schedule, early stopping and the epoch or round that is reported.
    #   'val_loss'  validation cross-entropy (default). Continuous, so it does not stall
    #               on the ties of a small validation set.
    #   'val_acc'   validation accuracy.
    selection_metric: str = 'val_acc'   # 'val_acc' | 'val_loss'

    # Epochs without a strict improvement before stopping (0 runs the full budget).
    # Centralised only. The federated counterpart is FLConfig.early_stop_patience.
    early_stopping_patience: int = 0

    # Plateau schedule (both pipelines): multiply the rate by lr_plateau_factor after
    # lr_plateau_patience epochs or rounds without a strict improvement, floored at lr_min.
    lr_plateau_patience: int = 20
    lr_plateau_factor: float = 0.5
    # None: learning_rate / 100.
    lr_min: Optional[float] = None

    # Schedule over epochs (centralised) or rounds (federated), rfl/utils/schedules.py:
    # 'cosine' (from learning_rate to lr_min), 'plateau' (see above), 'step' or 'none'.
    lr_schedule: str = 'cosine'

    # Evaluated model: 'selected' (best selection_metric) or 'last' (last epoch or
    # round). 'last' requires early stopping to be off.
    report_rule: str = 'last'

    # Random-feature control: keep every parameter except the classifier head at its
    # (data-independent) initialisation and train only the head. Both pipelines
    # (rfl/training/train_loop.py, rfl/federated/client.py), with and without DP. Normalisation
    # running statistics still update in non-private training.
    freeze_trunk: bool = False

    # ==================== Loss ====================
    # Both runners train with the plain cross-entropy loss. No loss option is exposed.

    # ==================== Device / paths ====================
    device: str = 'auto'               # 'auto' (cuda if available, else cpu), 'cpu', 'cuda'
    data_root: str = './data'
    save_root: str = './results'

    # ==================== Lifecycle ====================
    def __post_init__(self):
        self._resolve_defaults()
        self.experiment_dir = os.path.join(self.save_root, self.experiment_name,
                                           self.run_name)
        self.config_hash = self.get_hash()
        self._validate()

    def _resolve_defaults(self):
        """Fill the fields whose default depends on other fields (before the hash)."""
        if self.data_type is None:
            self.data_type = 'raw' if self.model_type.lower() == 'eegnet' else 'covariance'
        if self.lr_min is None:
            self.lr_min = self.learning_rate / 100.0
        if self.device == 'auto':        # resolved where the configuration is built
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        from rfl.data.registry import dataset_default
        if self.spdnet_hidden_dim is None:
            self.spdnet_hidden_dim = dataset_default(self.dataset, "spdnet_hidden_dim")
        if self.spdnet_epsilon is None:
            self.spdnet_epsilon = dataset_default(self.dataset, "spdnet_reeig", 1e-4)

    def _validate(self):
        """Raise ValueError on invalid field values."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if len(set(self.subjects)) != len(self.subjects):
            raise ValueError(f"subjects has duplicates: {self.subjects}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.optimizer not in ('adam', 'sgd'):
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {self.optimizer!r}")
        if not 0.0 <= self.sgd_momentum < 1.0:
            raise ValueError(f"sgd_momentum must be in [0, 1), got {self.sgd_momentum}")
        if self.sgd_momentum and self.optimizer != 'sgd':
            raise ValueError("sgd_momentum only acts with optimizer='sgd'")
        if self.spdnet_stiefel not in ('projection', 'trivialization'):
            raise ValueError("spdnet_stiefel must be 'projection' or 'trivialization', got "
                             f"{self.spdnet_stiefel!r}")
        expected = ('raw',) if self.model_type.lower() == 'eegnet' else ('covariance', 'spd')
        if self.model_type.lower() in ('eegnet', 'spdnet') and self.data_type not in expected:
            raise ValueError(f"model_type={self.model_type!r} reads data_type in {expected}, "
                             f"got {self.data_type!r}")
        if not 0 <= self.fmin < self.fmax:
            raise ValueError(f"the band-pass needs 0 <= fmin < fmax, got {self.fmin}-{self.fmax} Hz")
        rate = self._effective_rate()
        if rate is not None and self.fmax >= rate / 2:
            raise ValueError(f"fmax={self.fmax} Hz is not below the Nyquist frequency "
                             f"{rate / 2:g} Hz of the {rate:g} Hz signal (--resample).")
        if self.cov_estimator not in ('scm', 'lwf'):
            raise ValueError(f"cov_estimator must be 'scm' or 'lwf', got {self.cov_estimator!r}")
        if not 0.0 <= self.cov_shrinkage <= 1.0:
            raise ValueError(f"cov_shrinkage must lie in [0, 1], got {self.cov_shrinkage}")
        if self.cov_estimator == 'lwf' and self.cov_shrinkage == 0.0:
            raise ValueError("cov_estimator='lwf' with cov_shrinkage=0 is the sample "
                             "covariance: set the shrinkage intensity (for example 0.1).")
        if self.cov_regularization < 0:
            raise ValueError("cov_regularization must be non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.model_type.lower() not in ('eegnet', 'spdnet'):
            raise ValueError(f"Unknown model_type: {self.model_type!r} "
                             "(must be 'eegnet' or 'spdnet')")
        if self.normalization_type not in ('per_trial', 'per_trial_global'):
            raise ValueError("normalization_type must be 'per_trial' or 'per_trial_global', "
                             f"got {self.normalization_type!r}")
        if self.eegnet_norm_type not in ('batch', 'none'):
            raise ValueError(f"Unknown eegnet_norm_type: {self.eegnet_norm_type!r} "
                             "(must be 'batch' or 'none')")
        if self.selection_metric not in ('val_loss', 'val_acc'):
            raise ValueError(f"Unknown selection_metric: {self.selection_metric!r} "
                             "(must be 'val_loss' or 'val_acc')")
        if self.lr_schedule not in ('none', 'cosine', 'step', 'plateau'):
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule!r}")
        if self.report_rule not in ('selected', 'last'):
            raise ValueError(f"Unknown report_rule: {self.report_rule!r} "
                             "(must be 'selected' or 'last')")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative, got "
                             f"{self.early_stopping_patience}")
        if self.report_rule == 'last' and self.early_stopping_patience > 0:
            raise ValueError("report_rule='last' needs early_stopping_patience=0: with "
                             "early stopping, the last epoch is chosen by the validation "
                             "data. Use report_rule='selected'.")
        if self.lr_plateau_patience < 1:
            raise ValueError("lr_plateau_patience must be at least 1, got "
                             f"{self.lr_plateau_patience}")
        if not 0 < self.lr_plateau_factor < 1:
            raise ValueError("lr_plateau_factor must lie in (0, 1), got "
                             f"{self.lr_plateau_factor}")
        if self.lr_min <= 0:
            raise ValueError(f"lr_min must be positive, got {self.lr_min}")
        if not 0.0 < self.val_size < 1.0 or not 0.0 < self.test_size < 1.0:
            raise ValueError("val_size and test_size must be in (0, 1)")
        if self.val_size + self.test_size >= 1.0:
            raise ValueError("val_size + test_size must leave a non-empty training split")
        from rfl.data import reduction
        reduction.validate(self)
        if self.spdnet_bn_recalibrate and not self.has_batch_norm():
            raise ValueError("spdnet_bn_recalibrate needs a batch norm: SPDNet with "
                             "spdnet_use_batch_norm, or EEGNet with eegnet_norm_type "
                             "'batch'")
        if self.model_type.lower() == 'spdnet' and self.spdnet_use_batch_norm:
            from rfl.models.spdnet import validate_bn_options
            if self.batch_size < 2:
                raise ValueError("SPD batch norm needs batches of at least 2 matrices: "
                                 "a single matrix has no batch statistics.")
            validate_bn_options(self.spdnet_bn_type, self.spdnet_bn_mean_type,
                                self.spdnet_bn_momentum,
                                self.spdnet_bn_norm_and_bias_mode,
                                self.spdnet_bn_position)
        if self.model_type.lower() == 'spdnet' and str(self.spdnet_hidden_dims).strip():
            from rfl.models.spdnet import parse_float_list, parse_int_list, validate_epsilons
            dims = parse_int_list(self.spdnet_hidden_dims, [])
            eps = parse_float_list(self.spdnet_epsilons, [])
            if len(eps) != len(dims):
                raise ValueError(
                    f"spdnet_epsilons has {len(eps)} entries for {len(dims)} blocks: "
                    "give one strictly increasing ReEig threshold per block.")
            # With a batch norm between blocks the floors may be equal (rfl/models/spdnet.py).
            if not self.spdnet_use_batch_norm:
                validate_epsilons(eps, strict=True)

    def _effective_rate(self) -> Optional[float]:
        """Sampling rate after resampling (never upsampled), or None if unknown."""
        from rfl.data.registry import get_sampling_rate
        native = get_sampling_rate(self.dataset)
        target = self.resample if self.resample and self.resample > 0 else None
        if native is None:
            return target
        return min(native, target) if target else native

    def torch_device(self):
        """The torch device of the run. Raises RuntimeError if CUDA is asked for and
        absent (no silent fall-back to the CPU: config.json records the real device)."""
        import torch
        if str(self.device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f"device={self.device!r}, but CUDA is not available on "
                               f"this machine. Use --device cpu, or auto.")
        return torch.device(self.device)

    # ==================== Model parameters ====================
    def get_model_params(self) -> Dict[str, Any]:
        """Keyword arguments for create_eegnet or create_spdnet, without None values.

        `seed`, `n_classes`, `n_channels`, `n_timepoints` and `sampling_rate` are not
        included: the runners pass them from the config seed and from the data.

        Raises:
            ValueError: On an unknown model_type.
        """
        if self.model_type.lower() == 'eegnet':
            params = {
                'F1': self.eegnet_F1,
                'D': self.eegnet_D,
                'F2': self.eegnet_F2,
                'kern_length': self.eegnet_kern_length,
                'dropout_rate': self.eegnet_dropout,
                'dropout_type': self.eegnet_dropout_type,
                'norm_type': self.eegnet_norm_type,
            }
            # One momentum field for both batch norms (--bn-momentum), so that the
            # configuration hash of the runs without it is unchanged.
            if str(self.eegnet_norm_type).lower() == 'batch':
                params['bn_momentum'] = self.spdnet_bn_momentum
        elif self.model_type.lower() == 'spdnet':
            params = {
                'hidden_dim': self.spdnet_hidden_dim,
                'use_batch_norm': self.spdnet_use_batch_norm,
                'epsilon': self.spdnet_epsilon,
                'trace_normalize': self.spdnet_trace_normalize,
                'stiefel': self.spdnet_stiefel,
            }
            if self.spdnet_use_batch_norm:
                params.update(bn_type=self.spdnet_bn_type,
                              bn_mean_type=self.spdnet_bn_mean_type,
                              bn_momentum=self.spdnet_bn_momentum,
                              bn_norm_and_bias_mode=self.spdnet_bn_norm_and_bias_mode,
                              bn_position=self.spdnet_bn_position)
            # A block list overrides the single-block hidden_dim and epsilon.
            if str(self.spdnet_hidden_dims).strip():
                params['hidden_dims'] = self.spdnet_hidden_dims
                params['epsilons'] = self.spdnet_epsilons
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

        # Drop None so that the factory defaults apply.
        return {k: v for k, v in params.items() if v is not None}

    # ==================== Serialization ====================
    def has_batch_norm(self) -> bool:
        """The model has a batch norm: SPD batch norm, or EEGNet BatchNorm."""
        m = str(self.model_type).lower()
        return ((m == 'spdnet' and bool(self.spdnet_use_batch_norm))
                or (m == 'eegnet' and str(self.eegnet_norm_type).lower() == 'batch'))

    def get_hash(self) -> str:
        """8-character MD5 hash of the configuration, used to identify a run.

        `experiment_dir`, `run_name`, `save_root` and `config_hash` are excluded, so the
        hash does not depend on where the run is saved.
        """
        config_dict = self.to_dict()
        for key in ('experiment_dir', 'run_name', 'save_root', 'config_hash'):
            config_dict.pop(key, None)
        return hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        """All dataclass fields as a dict."""
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like attribute access with a default."""
        return getattr(self, key, default)

    def save(self, directory: Optional[str] = None) -> str:
        """Write config.json to `directory` (default `experiment_dir`) and return its path."""
        directory = directory or self.experiment_dir
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'config.json')
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to: {path}  (hash {self.config_hash})")
        return path

    @classmethod
    def load(cls, directory: str) -> 'ExperimentConfig':
        """Load `directory`/config.json, ignoring keys that are not dataclass fields.

        Unknown keys are dropped, so a config.json with extra fields still loads for
        analysis.

        Raises:
            FileNotFoundError: If config.json is missing.
        """
        import dataclasses

        path = os.path.join(directory, 'config.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            config_dict = json.load(f)

        for key in ('experiment_dir', 'config_hash'):
            config_dict.pop(key, None)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in config_dict.items() if k in known})

    def update(self, **kwargs):
        """Set existing fields (warn on unknown ones), then recompute the dir and hash.

        Validation is not re-run.
        """
        import warnings
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown configuration parameter: {key}")
        self.experiment_dir = os.path.join(self.save_root, self.experiment_name,
                                           self.run_name)
        self.config_hash = self.get_hash()
