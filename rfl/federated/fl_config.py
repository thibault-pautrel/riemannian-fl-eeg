"""Configuration of federated experiments, with optional personalisation and DP.

`FLConfig` extends `utils.config.ExperimentConfig` with the federated, personalisation
and DP settings. Personalisation and DP are independent: `_validate_fl` accepts every
combination that keeps the DP guarantee and refuses the others with a message.
Clients are formed by grouping consecutive subjects (subject-based, non-IID), so the
number of clients is ceil(n_subjects / fl_subjects_per_client).

The train/val/test split sizes are the base config's `val_size` and `test_size`. The
federated split is therefore identical to the centralized split at the same
`data_split_seed`, which makes the two pipelines directly comparable. Under DP the
private phase touches the training records only: the clip constant is fixed a priori
and the models carry no normalisation layer with running statistics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from dataclasses import dataclass
from typing import List, Literal

from rfl.utils.config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class FLConfig(ExperimentConfig):
    """Federated and DP experiment configuration.

    Each field is documented by the comment above it. Cross-field constraints are
    checked in `__post_init__` and raise `ValueError` (or warn) on invalid combinations.

    Example:
        >>> config = FLConfig(
        ...     experiment_name='fl_spdnet',
        ...     model_type='spdnet',
        ...     fl_aggregation='projavg',
        ...     fl_rounds=50,
        ...     dp_enabled=True,
        ...     dp_target_eps=3.0,
        ... )
    """

    # ==================== Federated learning ====================
    # Number of communication rounds T.
    fl_rounds: int = 50
    # Local epochs per round. Without DP a client runs ceil(m_i / B) steps per epoch over
    # a shuffled DataLoader. Under DP it runs floor(m_i / B) steps per epoch, each on a
    # fresh minibatch drawn without replacement, and the accountant composes
    # tau = epochs * ceil(m_max / B) steps per round, an upper bound for every client.
    fl_local_epochs: int = 1
    # Number of consecutive subjects grouped into one client.
    fl_subjects_per_client: int = 1
    # How the trials are dealt to the clients (real EEG only). The number of clients
    # and the train/val/test split are the same in both cases:
    #   'subject'  each client holds the trials of its subjects (non-IID: one subject,
    #              one recording setup and one brain per client).
    #   'iid'      the trials of every split are pooled and dealt at random to the
    #              clients, stratified by class (seed data_split_seed). Every client then
    #              holds trials of every subject, and the clients are IID.
    fl_client_partition: Literal['subject', 'iid'] = 'subject'

    # Aggregation rule (rfl/federated/aggregation.py), with W_i the client BiMap weights:
    #   'fedavg'   sum_i w_i x_i: EEGNet only (its parameters are Euclidean)
    #   'projavg'  uf(sum_i w_i W_i)                           (RFedProj)
    #   'rlavg'    uf(W_t + sum_i w_i P_{W_t}(W_i - W_t))      (RFedRL)
    #   'retavg'   R_{W_t}(sum_i w_i R_{W_t}^{-1}(W_i))        (PriRFed baseline)
    # SPDNet takes one of the last three, which keep the BiMap weights on the Stiefel
    # manifold (the other SPDNet parameters are averaged). EEGNet takes fedavg. Any
    # other pairing is refused. None: 'projavg' for SPDNet, 'fedavg' for EEGNet.
    # Other means on St(n, p) are registered by name in aggregation.AGGREGATION_RULES.
    fl_aggregation: str | None = None

    # Client weights, the same for every rule: uniform 1/k (False) or proportional to
    # the client training-set size (True).
    fl_weight_by_samples: bool = False

    # Fraction of clients sampled per round, in (0, 1]. Participation is visible to an
    # honest-but-curious server, so it gives no privacy amplification.
    fl_client_fraction: float = 1.0

    # Scope of the normalisation layers (EEGNet with eegnet_norm_type='batch', SPDNet
    # with spdnet_use_batch_norm; non-private runs only):
    #   True   FedBN (default): affine parameters, SPD bias and running statistics stay
    #          on each client and are never aggregated.
    #   False  shared normalisation. Trainable parameters (affine, SPD bias) are
    #          averaged like every other parameter. EEGNet running statistics are
    #          averaged. SPDNet running means are combined in the geometry of their
    #          layer (rfl/federated/server.py, rfl/models/spd_batchnorm.py).
    # aggregation.py finds these layers from the state-dict keys and param_groups.py
    # from the module tree. Without a normalisation layer the setting has no effect.
    fl_fedbn: bool = True

    # ==================== Personalisation ====================
    # Every option below is independent of dp_enabled, and every combination that
    # FLConfig accepts is covered by the DP guarantee (see _validate_fl).
    #
    # Personal classifier head (FedPer). The trunk is aggregated, the head stays on the
    # client and is never uploaded.
    fl_local_fc: bool = False

    # How a personal head is trained during the rounds:
    #   'train'   with the trunk, on the client's records, at every local step. Under DP
    #             the head is inside the privatised gradient: its per-sample gradient is
    #             clipped jointly with the trunk's and noised. This is required, since
    #             a head trained on raw records would carry them into the released
    #             trunk updates through backpropagation. The head is still never
    #             uploaded.
    #   'frozen'  pinned at its data-independent initialisation (the same on every
    #             client) during the rounds, then re-fitted locally after the last
    #             round (fl_head_refit_epochs). Under DP the head is outside the
    #             privatised vector, so the clip budget goes to the trunk only.
    fl_local_head_mode: Literal['train', 'frozen'] = 'train'

    # Purely local training: no broadcast and no aggregation. Each client trains, stops
    # and is evaluated on its own model. A baseline, never combined with DP.
    fl_local_only: bool = False

    # Post-training head re-fit (personalisation by fine-tuning). After the last round,
    # each client fits its head on its own training records, with the rest of the model
    # frozen, starting from the head of the reported model (the global head, or its
    # personal head). The epoch grid is comma-separated, for example "25,50,100,200".
    # An empty string disables the re-fit. The head is trained once to the largest
    # budget, and the intermediate states are kept as candidates.
    # Under DP the re-fit releases nothing, so the guarantee on the transcript is
    # unchanged (post-processing). The re-fitted head itself is computed from the raw
    # records and is not covered by the guarantee if it is published.
    fl_head_refit_epochs: str = ""
    # Weight decays tried by the re-fit, comma-separated.
    fl_head_refit_weight_decays: str = "0"
    # Adam learning rate of the re-fit. None uses learning_rate.
    fl_head_refit_lr: float | None = None
    # Candidate selection, per client, on its validation accuracy (loss breaks ties):
    #   'fixed'          first weight decay, last epoch of the grid, no selection.
    #   'wd_only'        weight decay selected, last epoch of the grid.
    #   'wd_and_epochs'  both selected.
    fl_head_refit_selection: Literal['fixed', 'wd_only', 'wd_and_epochs'] = 'wd_and_epochs'

    # The learning-rate schedule (lr_schedule, inherited, 'cosine' by default) is
    # evaluated per round by rfl/utils/schedules.py:compute_round_lr:
    #   'none'     constant learning_rate
    #   'cosine'   lr(t) = lr_min + (lr - lr_min) * (1 + cos(pi t)) / 2,
    #              t = (round - 1) / (T - 1)
    #   'step'     lr for the first 70% of the rounds (at least one), then 0.3 lr
    #   'plateau'  lr *= lr_plateau_factor after lr_plateau_patience rounds without a
    #              strict improvement of selection_metric, floored at lr_min
    # Adam is rebuilt every round, so the schedule applies exactly.

    # ==================== Differential privacy ====================
    # Record-level DP over the trained parameters, substitution adjacency. Each local
    # step draws a size-B minibatch without replacement, projects the per-sample
    # gradients on the tangent space at the current iterate, clips their joint norm to
    # C, averages them and adds tangent Gaussian noise:
    #
    #     g~ = (1/B) sum_l clip_C(P_x grad f(x; z_l)) + P_x xi,   xi ~ N(0, sigma^2 I)
    #     Delta = 2C/B,   mu0 = 2C/(B sigma),   q = B / m_star
    #
    # The transcript is C_q(G_mu0)^{(x) n_steps}-DP with
    # n_steps = T * local_epochs * ceil(m_max / B). See rfl/federated/privacy/.
    dp_enabled: bool = False
    # Target (eps, delta) used to calibrate sigma.
    dp_target_eps: float = 10.0
    dp_target_delta: float = 1e-5

    # Clip constant C, fixed a priori for the whole run (C = 1 in the DP-RFedProj
    # paper). It uses no data, so it costs no privacy.
    dp_clip_C: float = 1.0

    # Noise std added to the averaged clipped gradient. Written by
    # orchestration.setup_dp after calibration. 0 means no noise. Not a user setting.
    dp_sigma: float = 0.0

    # Lattice spacing of the privacy-loss distribution for the FFT composition. The
    # accountant refines it to min(h, mu0/40), since the discretisation error is O(h)
    # per step and accumulates over the composition.
    dp_accountant_h: float = 0.01

    # Accounting used to set sigma: 'ours' (exact T*tau-fold composition) or 'prirfed'
    # (PriRFed Theorem 7 at rho = 1, for the comparison baseline). The mechanism is the
    # same, only sigma differs.
    dp_accounting: str = "ours"

    # Also record in fl_results.json the noise PriRFed's accounting would need for the
    # same target (reporting only, the trained sigma is unchanged). Off by default: it
    # doubles the calibration time. Figure 4b comes from rfl/experiments/accounting_comparison.py.
    dp_compare_prirfed: bool = False

    # Per-sample gradients with torch.func.vmap for purely Euclidean models (EEGNet),
    # checked against the microbatch loop on first use. SPDNet always uses the loop,
    # because the custom autograd of BiMap, ReEig and LogEig does not compose with vmap.
    dp_vectorized_grads: bool = True

    # Rounds without a strict improvement of selection_metric (pooled global_val_acc
    # by default) before the run stops. 0 runs the full budget. lr_plateau_patience,
    # lr_plateau_factor, lr_min and selection_metric are inherited from
    # ExperimentConfig and mean the same thing as in the centralized pipeline.
    early_stop_patience: int = 0

    # The reported round follows report_rule (inherited): 'last' (default) or
    # 'selected' (best selection_metric).

    # ==================== Validation ====================
    def __post_init__(self):
        if self.fl_aggregation is None:
            self.fl_aggregation = ('fedavg' if self.model_type.lower() == 'eegnet'
                                   else 'projavg')
        super().__post_init__()
        self._validate_fl()

    def _validate_fl(self):
        """Check the federated and DP settings. Some fields are adjusted in place."""
        if self.fl_rounds < 1:
            raise ValueError("fl_rounds must be at least 1")
        if self.fl_local_epochs < 1:
            raise ValueError("fl_local_epochs must be at least 1")
        if self.fl_subjects_per_client < 1:
            raise ValueError("fl_subjects_per_client must be at least 1")
        if self.fl_client_partition not in ('subject', 'iid'):
            raise ValueError("fl_client_partition must be 'subject' or 'iid', got "
                             f"{self.fl_client_partition!r}")
        if self.train_trials_per_class and self.fl_client_partition == 'iid':
            raise ValueError("--train-trials-per-class reduces subjects: use it with "
                             "--client-partition subject")
        if not 0.0 < self.fl_client_fraction <= 1.0:
            raise ValueError("fl_client_fraction must be in (0, 1]")
        if (self.freeze_trunk and self.model_type.lower() == 'spdnet'
                and self.spdnet_stiefel == 'trivialization'):
            raise ValueError("freeze_trunk needs --stiefel projection in federated runs "
                             "(the server keeps frozen weights by parameter name).")
        if (not self.fl_fedbn and self.model_type.lower() == 'spdnet'
                and self.spdnet_use_batch_norm):
            from rfl.models.spd_batchnorm import SHARED_MEAN_TYPES
            if (self.spdnet_bn_type != 'mean_only'
                    or self.spdnet_bn_mean_type not in SHARED_MEAN_TYPES):
                raise ValueError(
                    "fl_fedbn=False shares the SPD batch norm, which needs a running "
                    "mean that can be combined across clients: bn_type 'mean_only' with "
                    f"a mean in {SHARED_MEAN_TYPES}. Keep FedBN for the other layers.")
        from rfl.federated.aggregation import rule_names
        if self.fl_aggregation not in rule_names():
            raise ValueError(f"fl_aggregation must be one of {', '.join(rule_names())} "
                             f"(rfl/federated/aggregation.py); got {self.fl_aggregation!r}")

        self._validate_personalisation()

        # Learning-rate schedule.
        if self.lr_schedule not in ('none', 'cosine', 'step', 'plateau'):
            raise ValueError("lr_schedule must be one of 'none', 'cosine', 'step', "
                             f"'plateau'; got {self.lr_schedule!r}")
        if self.lr_min <= 0:
            raise ValueError(f"lr_min must be positive, got {self.lr_min}")
        if self.lr_schedule == 'plateau':
            if self.lr_plateau_patience < 1:
                raise ValueError("lr_plateau_patience must be at least 1, got "
                                 f"{self.lr_plateau_patience}")
            if not 0 < self.lr_plateau_factor < 1:
                raise ValueError("lr_plateau_factor must lie in (0, 1), got "
                                 f"{self.lr_plateau_factor}")
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be non-negative, got "
                             f"{self.early_stop_patience}")
        if self.report_rule == 'last' and self.early_stop_patience > 0:
            raise ValueError("report_rule='last' needs early_stop_patience=0: with early "
                             "stopping, the last round is chosen by the validation data.")
        if self.lr_schedule == 'cosine' and self.lr_min >= self.learning_rate:
            warnings.warn(
                f"lr_min ({self.lr_min}) >= learning_rate ({self.learning_rate}): "
                "the cosine schedule will stay flat.", stacklevel=2)

        # Aggregation rule and model: Riemannian rules for SPDNet, fedavg for EEGNet.
        from rfl.federated.aggregation import get_rule, rule_names
        on_stiefel = get_rule(self.fl_aggregation).stiefel
        if self.model_type.lower() == 'spdnet' and not on_stiefel:
            raise ValueError(
                f"SPDNet with fl_aggregation={self.fl_aggregation!r}: the mean of Stiefel "
                "matrices is off the manifold, so the BiMap weights would no longer be "
                f"orthonormal. Use one of {', '.join(rule_names(stiefel=True))}.")
        if self.model_type.lower() == 'eegnet' and on_stiefel:
            raise ValueError(
                f"EEGNet with fl_aggregation={self.fl_aggregation!r}: the Riemannian "
                "rules act on Stiefel weights, and EEGNet has none, so the run would be "
                "FedAvg under another name. Use fedavg.")

        # Differential privacy.
        if not self.dp_enabled:
            return

        if not 0.0 < self.dp_target_delta < 1.0:
            raise ValueError("dp_target_delta must be in (0, 1)")
        if self.dp_target_eps <= 0:
            raise ValueError("dp_target_eps must be positive")
        C = float(self.dp_clip_C)
        if not (math.isfinite(C) and C > 0):
            raise ValueError(f"dp_clip_C must be a positive finite number, got {C}")
        if self.dp_accountant_h <= 0:
            raise ValueError("dp_accountant_h must be positive")
        if self.dp_accounting not in ("ours", "prirfed"):
            raise ValueError("dp_accounting must be 'ours' or 'prirfed'")

        # The validation records are outside the guarantee. Warn when they drive a
        # choice that the released model depends on.
        driven = [n for n, on in (("report_rule='selected'", self.report_rule == 'selected'),
                                  ("lr_schedule='plateau'", self.lr_schedule == 'plateau'),
                                  ("early_stop_patience > 0", self.early_stop_patience > 0))
                  if on]
        if driven:
            warnings.warn(
                f"dp_enabled with {', '.join(driven)}: the validation records drive the "
                "training, and the guarantee does not cover them (docs/design.md, Limitations).",
                stacklevel=3)

        # Settings where DP protects less than the transcript. They are valid, and kept
        # so that privacy and personalisation can be switched independently.
        if self.fl_local_only:
            warnings.warn(
                "dp_enabled with fl_local_only: nothing is sent, so the guarantee covers "
                "each client's own model (private local training).", stacklevel=3)
        elif self.freeze_trunk and self.fl_local_fc:
            if self.fl_local_head_mode == 'frozen':
                warnings.warn(
                    "dp_enabled with freeze_trunk and a frozen local head: nothing trains "
                    "during the rounds, so no private step is run.", stacklevel=3)
            else:
                warnings.warn(
                    "dp_enabled with freeze_trunk and a local head: the uploads do not "
                    "depend on the data, and the noise only protects the personal heads.",
                    stacklevel=3)

        # The private step clips and noises the gradient of W in orthonormal tangent
        # coordinates at uf(W) (rfl/federated/privacy/dp_grad.py): it needs plain BiMap
        # weights.
        if self.model_type.lower() == 'spdnet' and self.spdnet_stiefel != 'projection':
            raise ValueError(
                "DP needs --stiefel projection: the private step works on the BiMap "
                "weights W themselves, not on a parametrization of W.")
        # Batch normalisation couples the records of a batch, so the per-sample
        # sensitivity 2C/B no longer holds (Appendix D.2 of the DP-RFedProj paper).
        if self.model_type.lower() == 'spdnet' and self.spdnet_use_batch_norm:
            raise ValueError(
                "DP + SPD batch norm is unsound: BatchNormSPDMean's running Frechet "
                "mean is updated on the private batch, which breaks the 2C/B "
                "per-sample sensitivity. Disable --use-batch-norm.")
        if self.model_type.lower() == 'eegnet' and self.eegnet_norm_type != 'none':
            raise ValueError(
                "DP + EEGNet BatchNorm is unsound: batch statistics couple the records "
                "of a batch, which breaks the 2C/B per-sample sensitivity. Use "
                "--eegnet-norm-type none.")

    def _validate_personalisation(self):
        """Check the personalisation settings (fl_local_fc, fl_local_only, re-fit)."""
        if self.fl_local_head_mode not in ('train', 'frozen'):
            raise ValueError("fl_local_head_mode must be 'train' or 'frozen', got "
                             f"{self.fl_local_head_mode!r}")
        if self.fl_head_refit_selection not in ('fixed', 'wd_only', 'wd_and_epochs'):
            raise ValueError("fl_head_refit_selection must be 'fixed', 'wd_only' or "
                             f"'wd_and_epochs', got {self.fl_head_refit_selection!r}")
        if self.fl_local_only and self.fl_client_fraction < 1.0:
            raise ValueError("fl_local_only trains every client at every round: client "
                             "sampling would count rounds without training in each "
                             "client's stopping rule. Use fl_client_fraction=1.")
        if self.fl_local_only and self.fl_local_fc:
            raise ValueError("fl_local_only already keeps every parameter local. Use "
                             "either --local-only or --local-fc, not both.")
        if self.fl_local_head_mode == 'frozen' and not self.fl_local_fc:
            raise ValueError("fl_local_head_mode='frozen' applies to a personal head: "
                             "add --local-fc.")

        from rfl.federated.personalization import parse_epoch_grid, parse_weight_decays
        epochs = parse_epoch_grid(self.fl_head_refit_epochs)
        parse_weight_decays(self.fl_head_refit_weight_decays)
        if self.fl_head_refit_lr is not None and self.fl_head_refit_lr <= 0:
            raise ValueError("fl_head_refit_lr must be positive")
        if epochs and self.fl_local_only:
            raise ValueError("The head re-fit acts on a federated model. In local-only "
                             "mode the head is already trained locally.")

        frozen_head = self.fl_local_fc and self.fl_local_head_mode == 'frozen'
        if frozen_head and self.freeze_trunk and not epochs:
            raise ValueError("freeze_trunk with a frozen local head trains nothing. "
                             "Set fl_head_refit_epochs, or use fl_local_head_mode='train'.")
        if frozen_head and not epochs:
            warnings.warn("fl_local_head_mode='frozen' without fl_head_refit_epochs: the "
                          "personal heads stay at their random initialisation.",
                          stacklevel=3)

    @property
    def head_refit_enabled(self) -> bool:
        """True when a post-training head re-fit is configured."""
        from rfl.federated.personalization import parse_epoch_grid
        return bool(parse_epoch_grid(self.fl_head_refit_epochs))

    def personalisation_name(self) -> str:
        """Short name of the personalisation scheme, as used in the result tables.

        'shared' (trunk and head), 'fedper' (local head trained during the rounds),
        'fedper_frozen' (local head pinned during the rounds) or 'local' (nothing
        shared), with '_rnd' when the trunk is frozen and '+refit' with a head re-fit.
        """
        if self.fl_local_only:
            name = 'local'
        elif self.fl_local_fc:
            name = 'fedper' if self.fl_local_head_mode == 'train' else 'fedper_frozen'
        else:
            name = 'shared'
        if self.freeze_trunk:
            name += '_rnd'
        if self.head_refit_enabled:
            name += '+refit'
        return name

    # ==================== Serialization ====================
    def get_hash(self) -> str:
        """Return an 8-character MD5 hash of the configuration.

        Runtime fields are excluded: `experiment_dir`, `run_name`, `save_root`,
        `config_hash` and `dp_sigma`. `dp_sigma` is written by `setup_dp` after
        construction, and every input of that calibration (C, B, eps, delta, T, tau, h)
        is hashed already.
        """
        config_dict = self.to_dict()
        for key in ('experiment_dir', 'run_name', 'save_root', 'config_hash', 'dp_sigma'):
            config_dict.pop(key, None)
        return hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]

    def print_fl_summary(self):
        """Print the federated and DP settings."""
        logger.info("\n" + "-" * 70)
        logger.info("Federated learning:")
        logger.info(f"  Subjects per client:  {self.fl_subjects_per_client}")
        logger.info(f"  Client partition:     {self.fl_client_partition}")
        logger.info(f"  Rounds:               {self.fl_rounds}")
        logger.info(f"  Local epochs:         {self.fl_local_epochs}")
        logger.info(f"  Client fraction:      {self.fl_client_fraction} "
              f"(no privacy amplification: participation is observable)")
        logger.info(f"  Aggregation:          {self.fl_aggregation}"
              f"{' (uniform)' if not self.fl_weight_by_samples else ' (sample-weighted)'}")
        logger.info(f"  Personalisation:      {self.personalisation_name()}")
        if self.fl_local_fc:
            logger.info(f"  Local head:           {self.fl_local_head_mode}"
                  + (" (inside the privatised gradient)"
                     if self.dp_enabled and self.fl_local_head_mode == 'train' else ""))
        if self.head_refit_enabled:
            logger.info(f"  Head re-fit:          epochs {self.fl_head_refit_epochs}, "
                  f"weight decays {self.fl_head_refit_weight_decays}, "
                  f"selection {self.fl_head_refit_selection}")
        has_norm = ((self.model_type.lower() == 'eegnet' and self.eegnet_norm_type == 'batch')
                    or (self.model_type.lower() == 'spdnet' and self.spdnet_use_batch_norm))
        if has_norm:
            logger.info(f"  Normalisation:        {'local (FedBN)' if self.fl_fedbn else 'shared (averaged)'}")
        logger.info(f"  LR schedule:          {self.lr_schedule}"
              + (f" (lr_min={self.lr_min})" if self.lr_schedule == 'cosine' else ""))
        logger.info(f"  Split:                {1 - self.val_size - self.test_size:.2f}/"
              f"{self.val_size}/{self.test_size} (shared with centralized)")
        if self.dp_enabled:
            logger.info("\nDifferential privacy (record-level, substitution adjacency):")
            logger.info(f"  Target:               (eps={self.dp_target_eps}, "
                  f"delta={self.dp_target_delta:.0e})")
            logger.info(f"  Clip C:               {self.dp_clip_C} (fixed a priori)")
        logger.info("-" * 70)


def create_client_subject_mapping(
    subjects: List[int],
    subjects_per_client: int = 5,
) -> List[List[int]]:
    """Group consecutive subjects into clients (subject-based, non-IID partitioning).

    When the number of subjects is not a multiple of `subjects_per_client`, the last
    client is smaller than the others.

    Args:
        subjects: subject IDs, in the order they are assigned.
        subjects_per_client: maximum number of subjects per client.

    Returns:
        One list of subject IDs per client.

    Raises:
        ValueError: if `subjects_per_client` < 1.

    Example:
        >>> create_client_subject_mapping([1, 2, 3, 4, 5, 6, 7, 8], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8]]
    """
    if subjects_per_client < 1:
        raise ValueError("subjects_per_client must be at least 1")
    return [subjects[i:i + subjects_per_client]
            for i in range(0, len(subjects), subjects_per_client)]
