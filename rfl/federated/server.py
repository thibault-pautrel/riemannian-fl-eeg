"""Federated server: client sampling, aggregation, evaluation and model selection.

One call to `FLServer.run_round` does the following.

1. Sample the participating clients and broadcast the global state to them. Clients
   keep their local keys (normalisation layers, personal head).
2. Each participant trains locally: Adam in `FLClient`, or the DP step of
   `DPFLClient` (per-sample tangent clipping plus noise) when `config.dp_enabled`.
3. Aggregate the returned state dicts with `config.fl_aggregation` (fedavg, projavg,
   rlavg or retavg, see rfl/federated/aggregation.py) and update the global model.
4. Evaluate the global model on the validation and test split of every client.

Metrics. The main metrics are pooled over the union of the clients' predictions
(`global_test_acc`, `global_test_f1`, `global_test_loss`). They use the same estimator
as the centralized runs. Client-uniform means (`avg_test_acc`, ...) weight each client
equally whatever its test size, and are reported as secondary statistics.

Model selection. `config.selection_metric` ('val_acc' by default, pooled as
`global_val_acc`) drives the plateau schedule, the early-stopping rule and the
selected round, as in rfl/training/train_loop.py. The selected round is the first one that
reaches the best value. The runner decides whether that round or the last one is
reported (`config.report_rule`).

The module also builds the clients: `create_fl_clients` for MOABB data and
`create_fl_clients_synthetic` for precomputed SPD data.
"""

from __future__ import annotations

import copy
import gc
import logging
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from rfl.data.data_utils import filter_binary_task, load_moabb_dataset
from rfl.federated.aggregation import (
    extract_stiefel_matrices_from_model,
    get_rule,
    identify_stiefel_keys,
    polar_projection,
)
from rfl.federated.client import DATASET_SAMPLING_RATES, FLClient
from rfl.utils.schedules import compute_round_lr  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)


def _client_class(config):
    """Return DPFLClient when `config.dp_enabled`, FLClient otherwise."""
    if bool(getattr(config, "dp_enabled", False)):
        from rfl.federated.privacy.dp_client import DPFLClient
        return DPFLClient
    return FLClient


class FLServer:
    """Holds the global model, runs rounds and records their metrics.

    Per-round metrics are appended to `round_history` (a dict of lists keyed by metric
    name). The best round for the selection metric is tracked in `best_round`,
    `best_score` and `best_state_dict` (None in local-only mode).

    Args:
        global_model: initial global model, deep-copied.
        config: an FLConfig.
        device: torch device.
    """

    def __init__(self, global_model: nn.Module, config: Any, device: torch.device):
        self.config = config
        self.device = device

        self.global_model = copy.deepcopy(global_model).to(device)
        self.global_state_dict = copy.deepcopy(self.global_model.state_dict())

        self.aggregation_strategy = str(config.fl_aggregation).lower()
        self.is_spdnet = config.model_type.lower() == 'spdnet'

        # BiMap weights are extracted for every SPDNet run, for the Riemannian rules
        # and the geometry diagnostics.
        self.needs_stiefel = self.is_spdnet

        # Dedicated RNG for client sampling, independent of the global numpy state.
        self._participation_rng = np.random.default_rng(int(config.seed) + 9_973)

        # Best round for the selection metric, with a copy of its global state.
        self.best_round: Optional[int] = None
        self.best_score: Optional[float] = None
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        # The clients' local keys (personal heads, local normalisation) at that round,
        # so that the selected model can be rebuilt exactly on every client.
        self.best_local_states: Dict[Any, Dict[str, torch.Tensor]] = {}
        # Local-only mode: each client's own best state (its own validation metric).
        self._local_best: Dict[Any, Dict[str, torch.Tensor]] = {}

        sel_key, sel_min = self.selection_rule()

        logger.info(f"[Server] Model: {'SPDNet' if self.is_spdnet else 'EEGNet'}")
        logger.info(f"[Server] Aggregation: {self.aggregation_strategy}")
        self.fedbn = bool(config.fl_fedbn)
        logger.info("[Server] Normalisation layers, if any: "
              + ("local to each client (FedBN, never aggregated)" if self.fedbn
                 else "shared (parameters averaged, SPD running means combined in "
                      "their geometry)"))
        uses = [u for u, on in (("the plateau schedule", config.lr_schedule == 'plateau'),
                                ("early stopping", config.early_stop_patience > 0),
                                ("the reported round", config.report_rule == 'selected'))
                if on]
        logger.info(f"[Server] Selection: {sel_key} ({'minimised' if sel_min else 'maximised'})"
                    + (f" -- drives {', '.join(uses)}" if uses
                       else " -- recorded only (last round reported)"))
        if bool(config.dp_enabled):
            logger.info("[Server] Local training: DP-RFedProj steps (record-level DP)")
        if bool(config.fl_local_fc):
            logger.info("[Server] Scope: SHARED TRUNK, PERSONAL HEAD (FedPer, head "
                  f"{config.fl_local_head_mode})")
        elif bool(config.fl_local_only):
            logger.info("[Server] Scope: PURELY LOCAL -- no broadcast, no aggregation")
        if self.is_spdnet:
            keys = sorted(identify_stiefel_keys(self.global_state_dict))
            logger.info(f"[Server] Stiefel parameters ({len(keys)}): {', '.join(keys)}")

        self.round_history: Dict[str, List] = {
            'round': [],
            'current_lr': [],
            # local training (participating clients)
            'avg_train_loss': [], 'std_train_loss': [],
            'avg_train_acc': [], 'std_train_acc': [],
            # pooled estimators of the aggregated model (primary)
            'global_val_acc': [], 'global_val_f1': [], 'global_val_loss': [],
            'global_test_acc': [], 'global_test_f1': [], 'global_test_loss': [],
            # client-uniform means (secondary)
            'avg_val_acc': [], 'std_val_acc': [],
            'avg_test_acc': [], 'std_test_acc': [],
            'avg_test_f1': [], 'std_test_f1': [],
            'per_client_test_acc': [], 'per_client_test_f1': [],
            'per_client_val_acc': [],
            # local-only mode: clients still training, and per-client learning rates
            'n_active_clients': [], 'per_client_lr': [],
            # cost + geometry
            'agg_seconds': [],
            'proj_defect': [], 'client_disp': [], 'head_norm': [],
        }

    # ------------------------------------------------------------------ #
    #  Global state
    # ------------------------------------------------------------------ #
    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of the global state dict."""
        return copy.deepcopy(self.global_state_dict)

    # In local-only mode the schedule and the stopping rule are per client. A pooled
    # rule would let other clients' validation data decide when a client stops.
    def _local_state(self, client) -> Dict[str, Any]:
        """Per-client selection history and stopped flag (local-only mode)."""
        if not hasattr(self, '_client_hist'):
            self._client_hist: Dict[Any, List[float]] = {}
            self._client_stopped: Dict[Any, bool] = {}
        cid = client.client_id
        self._client_hist.setdefault(cid, [])
        self._client_stopped.setdefault(cid, False)
        return {'hist': self._client_hist[cid], 'stopped': self._client_stopped[cid]}

    @property
    def all_clients_stopped(self) -> bool:
        """True once every client has met its own stopping rule (local-only mode)."""
        return bool(getattr(self, '_client_stopped', {})) and \
            all(self._client_stopped.values())

    def broadcast_global_model(self, clients: List[FLClient]):
        """Send the global state to `clients`, which keep their local keys."""
        for client in clients:
            client.receive_global_model(self.global_state_dict)

    def update_global_model(self, aggregated_state: Dict[str, torch.Tensor]):
        """Overwrite the aggregated keys of the global model. Other keys are unchanged."""
        state = self.global_model.state_dict()
        for key, val in aggregated_state.items():
            state[key] = copy.deepcopy(val)
        self.global_state_dict = state
        self.global_model.load_state_dict(state)

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #
    def aggregate_updates(
        self,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        client_stiefel_matrices: Optional[List[Dict[str, torch.Tensor]]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates with the configured rule.

        For EEGNet the Riemannian rules fall back to fedavg. Local normalisation keys
        are excluded by every rule.

        Args:
            client_state_dicts: uploaded state dicts.
            client_stiefel_matrices: BiMap weights per client (SPDNet), extracted
                beforehand so that full client models need not be kept.
            client_weights: unnormalised client weights. None means uniform.

        Returns:
            The aggregated state dict.

        Raises:
            ValueError: for an unknown rule, or a Riemannian rule without
                `client_stiefel_matrices`.
        """
        strategy = self.aggregation_strategy

        if get_rule(strategy).stiefel and not self.is_spdnet:
            logger.info(f"[Warning] {strategy} is Stiefel-only (SPDNet); falling back to "
                  "fedavg for EEGNet. On a flat manifold the rules coincide, so the "
                  "fallback is loss-free.")
            strategy = 'fedavg'

        agg = self._aggregate_rule(strategy, client_state_dicts,
                                   client_stiefel_matrices, client_weights)
        if not self.fedbn:
            self._share_spd_batchnorm(agg, client_state_dicts, client_weights)
        if self.config.freeze_trunk:
            # Frozen parameters keep their initial value exactly: the projection of the
            # mean of identical Stiefel matrices would move them by float round-off.
            from rfl.federated.param_groups import identify_classifier_param_names
            head = identify_classifier_param_names(self.global_model)
            for name, _ in self.global_model.named_parameters():
                if name not in head and name in agg and name in self.global_state_dict:
                    agg[name] = self.global_state_dict[name].clone()
        return agg

    def _aggregate_rule(self, strategy, client_state_dicts, client_stiefel_matrices,
                        client_weights):
        """Apply the aggregation rule `strategy` (aggregation.AGGREGATION_RULES)."""
        rule = get_rule(strategy)
        if rule.stiefel and client_stiefel_matrices is None:
            raise ValueError(f"{strategy} requires the client BiMap weights")
        return rule.aggregate(client_stiefel_matrices, client_state_dicts,
                              self.global_model, client_weights, fedbn=self.fedbn)

    def _share_spd_batchnorm(self, agg, client_state_dicts, client_weights) -> None:
        """Replace the Euclidean average of SPD batch-norm running means, in place.

        With shared normalisation (`fl_fedbn=False`) each running mean is combined in
        the geometry of its layer (rfl/models/spd_batchnorm.py). The bias and other trainable parameters keep the average of the rule,
        taken in the parameter space where they are optimised.
        """
        from rfl.models.spd_batchnorm import weighted_spd_mean
        from rfl.models.spdnet import BatchNormSPD
        if BatchNormSPD is None:
            return
        w = (list(client_weights) if client_weights is not None
             else [1.0] * len(client_state_dicts))
        for name, module in self.global_model.named_modules():
            if not isinstance(module, BatchNormSPD):
                continue
            key = f"{name}.running_mean"
            agg[key] = weighted_spd_mean([sd[key] for sd in client_state_dicts], w,
                                         module.mean_type)

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _geometry_diagnostics(client_stiefel_matrices) -> Dict[str, float]:
        """Per-round dispersion of the client BiMap weights, averaged over BiMap keys.

        proj_defect: ||mean_i W_i - uf(mean_i W_i)||_F, the distance of the uniform
            extrinsic mean to the Stiefel manifold. Zero when the clients agree.
        client_disp: mean over pairs i < j of ||W_i W_i^T - W_j W_j^T||_F, the chordal
            distance between the spanned subspaces. Projectors are used because W and
            W Q (Q orthogonal) span the same subspace. Only with two or more clients.

        Returns:
            A dict with these keys, or an empty dict when there are no matrices.
        """
        diag: Dict[str, float] = {}
        if not client_stiefel_matrices:
            return diag

        defects, disps = [], []
        for name in client_stiefel_matrices[0]:
            Ws = [m[name].detach().float() for m in client_stiefel_matrices]
            W_bar = torch.stack(Ws).mean(0)
            defects.append(float(torch.linalg.norm(W_bar - polar_projection(W_bar))))

            if len(Ws) > 1:
                projectors = [W @ W.T for W in Ws]
                pair = [float(torch.linalg.norm(projectors[i] - projectors[j]))
                        for i in range(len(Ws)) for j in range(i + 1, len(Ws))]
                disps.append(float(np.mean(pair)))

        diag['proj_defect'] = float(np.mean(defects))
        if disps:
            diag['client_disp'] = float(np.mean(disps))
        return diag

    def _head_norm(self) -> Optional[float]:
        """Frobenius norm of the global classifier weight, or None.

        None in local-FC and local-only mode, where the server's head is never
        updated. Useful under DP, where a noise-dominated head can drift in norm.
        """
        if (bool(self.config.fl_local_fc)
                or bool(self.config.fl_local_only)):
            return None
        with torch.no_grad():
            for name, p in self.global_model.named_parameters():
                if name.endswith('classifier.weight'):
                    return float(torch.linalg.norm(p.detach().float()))
        return None

    # ------------------------------------------------------------------ #
    #  One round
    # ------------------------------------------------------------------ #
    def selection_rule(self):
        """Return (round_history key, minimize) for `config.selection_metric`.

        'val_loss' maps to ('global_val_loss', True), anything else to
        ('global_val_acc', False). Used by the schedule, the stopping rule and the
        round selection.
        """
        m = str(self.config.selection_metric).lower()
        if m in ('val_loss', 'global_val_loss', 'loss'):
            return 'global_val_loss', True
        return 'global_val_acc', False

    def _select_participants(self, clients: List[FLClient],
                             round_num: int, verbose: bool) -> List[FLClient]:
        """Sample ceil(fraction * n_clients) participants without replacement.

        With every client selected the RNG is not used. Participation is visible to the
        server, so sampling gives no privacy amplification.
        """
        fraction = float(self.config.fl_client_fraction)
        n = max(1, math.ceil(len(clients) * fraction))
        if n >= len(clients):
            return clients
        idx = self._participation_rng.choice(len(clients), size=n, replace=False)
        if verbose:
            logger.info(f"[Round {round_num}] Participating clients: {sorted(idx.tolist())}")
        return [clients[i] for i in idx]

    def run_round(self, round_num: int, clients: List[FLClient],
                  criterion: nn.Module, verbose: bool = True) -> Dict[str, float]:
        """Run one round: train, aggregate, evaluate and record (see module docstring).

        In local-only mode (`config.fl_local_only`) there is no broadcast and no
        aggregation, each client is evaluated on its own model, and each client follows
        its own learning-rate schedule and stopping rule on its own validation metric.

        Args:
            round_num: current round, starting at 1.
            clients: all clients. Only the sampled ones train, all are evaluated.
            criterion: loss function.
            verbose: print progress.

        Returns:
            The scalar metrics of this round, also appended to `round_history`.
        """
        if verbose:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"ROUND {round_num}/{self.config.fl_rounds}")
            logger.info(f"{'=' * 70}")

        # Local-only clients keep their local_model across rounds: skipping the broadcast
        # is enough.
        local_only = bool(self.config.fl_local_only)

        participants = self._select_participants(clients, round_num, verbose)
        if not local_only:
            self.broadcast_global_model(participants)

        sel_key, sel_min = self.selection_rule()
        current_lr = compute_round_lr(
            base_lr=self.config.learning_rate,
            round_num=round_num,
            total_rounds=self.config.fl_rounds,
            schedule=self.config.lr_schedule,
            lr_min=self.config.lr_min,
            val_history=list(self.round_history.get(sel_key, [])),
            plateau_patience=int(self.config.lr_plateau_patience),
            plateau_factor=float(self.config.lr_plateau_factor),
            mode='min' if sel_min else 'max',
        )

        # ---- 1. local training -------------------------------------------------
        client_state_dicts: List[Dict[str, torch.Tensor]] = []
        client_stiefel: Optional[List[Dict[str, torch.Tensor]]] = [] if self.needs_stiefel else None
        client_sizes: List[float] = []
        train_losses, train_accs = [], []
        checked_losses = []        # non-private clients: a NaN loss there is a divergence

        per_client_lr: List[float] = []
        for client in participants:
            client_lr = current_lr
            if local_only:
                st = self._local_state(client)
                if st['stopped']:
                    per_client_lr.append(0.0)
                    continue                      # met its own stopping rule
                client_lr = compute_round_lr(
                    base_lr=self.config.learning_rate,
                    round_num=round_num,
                    total_rounds=self.config.fl_rounds,
                    schedule=self.config.lr_schedule,
                    lr_min=self.config.lr_min,
                    val_history=st['hist'],
                    plateau_patience=int(self.config.lr_plateau_patience),
                    plateau_factor=float(self.config.lr_plateau_factor),
                    mode='min' if sel_min else 'max',
                )
            per_client_lr.append(client_lr)
            state_dict, metrics = client.train_local(
                n_epochs=self.config.fl_local_epochs,
                criterion=criterion,
                learning_rate=client_lr,
                is_spdnet=self.is_spdnet,
                apply_constraints=True,
                verbose=False,
            )
            client_state_dicts.append(state_dict)
            client_sizes.append(float(client.n_samples))
            if self.needs_stiefel:
                # Keep only the BiMap weights, not the client model.
                client_stiefel.append(extract_stiefel_matrices_from_model(client.local_model))

            train_losses.append(metrics['train_loss'])
            train_accs.append(metrics['train_acc'])
            # Under DP the loss is a diagnostic and is NaN by design when nothing trains
            # or a batch forward fails (dp_client.py, dp_grad.py).
            if 'dp_local_steps' not in metrics:
                checked_losses.append(metrics['train_loss'])
            if verbose:
                logger.info(f"  [Client {client.client_id}] train loss={metrics['train_loss']:.4f} "
                      f"acc={metrics['train_acc']:.4f}")

        # ---- 2. aggregate ------------------------------------------------------
        # Same client weights for every rule.
        weights = (client_sizes if self.config.fl_weight_by_samples
                   else [1.0] * len(client_state_dicts))

        t0 = time.perf_counter()
        aggregated = None if local_only else self.aggregate_updates(
            client_state_dicts, client_stiefel, weights)
        agg_seconds = time.perf_counter() - t0

        diag = self._geometry_diagnostics(client_stiefel)

        del client_state_dicts, client_stiefel, client_sizes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if aggregated is not None:
            self.update_global_model(aggregated)
        del aggregated
        gc.collect()

        head_norm = self._head_norm()          # of the aggregated model
        if head_norm is not None:
            diag['head_norm'] = head_norm

        # ---- 3. evaluate the aggregated model on all clients -------------------
        ev = self.evaluate_clients(clients, criterion, local_only)
        val_accs, test_accs, test_f1s = ev['val_accs'], ev['test_accs'], ev['test_f1s']
        val_losses = ev['val_losses']
        global_val_acc, global_val_f1 = ev['global_val_acc'], ev['global_val_f1']
        global_val_loss = ev['global_val_loss']
        global_test_acc, global_test_f1 = ev['global_test_acc'], ev['global_test_f1']
        global_test_loss = ev['global_test_loss']

        if checked_losses:
            from rfl.training.divergence import check_finite
            check_finite(float(np.mean(checked_losses)), f"round {round_num}")
        metrics = {
            'round': round_num,
            'current_lr': current_lr,
            # No client trained when all are stopped (local-only mode): report zeros.
            'avg_train_loss': float(np.mean(train_losses)) if train_losses else 0.0,
            'std_train_loss': float(np.std(train_losses)) if train_losses else 0.0,
            'avg_train_acc': float(np.mean(train_accs)) if train_accs else 0.0,
            'std_train_acc': float(np.std(train_accs)) if train_accs else 0.0,
            'global_val_acc': global_val_acc,
            'global_val_f1': global_val_f1,
            'global_val_loss': global_val_loss,
            'global_test_acc': global_test_acc,
            'global_test_f1': global_test_f1,
            'global_test_loss': global_test_loss,
            'avg_val_acc': float(np.mean(val_accs)),
            'std_val_acc': float(np.std(val_accs)),
            'avg_test_acc': float(np.mean(test_accs)),
            'std_test_acc': float(np.std(test_accs)),
            'avg_test_f1': float(np.mean(test_f1s)),
            'std_test_f1': float(np.std(test_f1s)),
            'agg_seconds': agg_seconds,
            **diag,
        }

        for key, value in metrics.items():
            self.round_history.setdefault(key, []).append(value)
        self.round_history['per_client_test_acc'].append([float(a) for a in test_accs])
        self.round_history['per_client_test_f1'].append([float(f) for f in test_f1s])
        self.round_history['per_client_val_acc'].append([float(a) for a in val_accs])

        if local_only:
            stop_patience = int(self.config.early_stop_patience or 0)
            # Per-client stopping rule, on the client's own validation metric.
            per_client_sel = val_losses if sel_min else val_accs
            for client, v in zip(clients, per_client_sel):
                st = self._local_state(client)
                prev = st['hist']
                if not self._client_stopped[client.client_id] and (
                        v == v and (not any(x == x for x in prev) or (
                            v < min(x for x in prev if x == x) if sel_min
                            else v > max(x for x in prev if x == x)))):
                    # The client's own selected model (strict improvement, NaN never
                    # selected), loaded at the end with report_rule='selected'.
                    self._local_best[client.client_id] = copy.deepcopy(
                        client.local_model.state_dict())
                st['hist'].append(float(v))
                self._client_hist[client.client_id] = st['hist']
                if stop_patience > 0 and not self._client_stopped[client.client_id]:
                    # Rounds since the last strict improvement, as for the plateau
                    # schedule.
                    best, stale = (float('inf') if sel_min else -float('inf')), 0
                    for x in st['hist']:
                        if (x < best) if sel_min else (x > best):
                            best, stale = x, 0
                        else:
                            stale += 1
                    if stale >= stop_patience:
                        self._client_stopped[client.client_id] = True
                        if verbose:
                            logger.info(f"  [early stop] client {client.client_id} frozen at "
                                  f"round {round_num} ({stale} rounds without a new best)")
            self.round_history['n_active_clients'].append(
                int(sum(1 for s in self._client_stopped.values() if not s)))
            self.round_history['per_client_lr'].append(
                [float(x) for x in per_client_lr])

        # ---- 4. track the selected round (strict improvement, ties to earliest) -
        score = metrics.get(sel_key)
        if score is not None and score == score:          # score == score rejects NaN
            score = float(score)
            better = (self.best_score is None
                      or ((score < self.best_score) if sel_min
                          else (score > self.best_score)))
            if better:
                self.best_score = score
                self.best_round = round_num
                # No global model to keep in local-only mode.
                self.best_state_dict = (None if local_only
                                        else copy.deepcopy(self.global_state_dict))
                self.best_local_states = ({} if local_only else
                                          {c.client_id: c.local_state_dict()
                                           for c in clients})

        if verbose:
            logger.info(f"\n[Round {round_num}] lr={current_lr:.2e}")
            logger.info(f"  train  loss={metrics['avg_train_loss']:.4f}  "
                  f"acc={metrics['avg_train_acc']:.4f} +/- {metrics['std_train_acc']:.4f}")
            logger.info(f"  val    acc={global_val_acc:.4f}  f1={global_val_f1:.4f}   (pooled)")
            logger.info(f"  test   acc={global_test_acc:.4f}  f1={global_test_f1:.4f}   (pooled)"
                  f"  |  per-client acc {metrics['avg_test_acc']:.4f} "
                  f"+/- {metrics['std_test_acc']:.4f}")
            if diag:
                logger.info("  diag   " + "  ".join(f"{k}={v:.4g}" for k, v in sorted(diag.items())))

        return metrics

    def evaluate_clients(self, clients: List[FLClient], criterion: nn.Module,
                         local_only: bool) -> Dict[str, Any]:
        """Evaluate every client and pool the metrics over clients.

        Each client evaluates the global weights combined with its local keys, or its
        own model in local-only mode. Pooled accuracy and F1 use all predictions, the
        pooled loss is sum_i n_i L_i / sum_i n_i.
        """
        val_accs, val_f1s, val_losses, val_sizes = [], [], [], []
        test_accs, test_f1s, test_losses, test_sizes = [], [], [], []
        val_preds, val_labels, test_preds, test_labels = [], [], [], []

        for client in clients:                 # every client, not just the participants
            # In local-only mode the global state is never updated.
            if local_only:
                evals = client.evaluate_local_model(criterion, splits=("val", "test"))
            else:
                evals = client.evaluate_global_model(
                    self.global_state_dict, criterion, splits=("val", "test"))

            v, t = evals['val'], evals['test']
            val_losses.append(v['loss']); val_accs.append(v['acc']); val_f1s.append(v['f1_macro'])
            test_losses.append(t['loss']); test_accs.append(t['acc']); test_f1s.append(t['f1_macro'])
            val_sizes.append(v['n_samples']); test_sizes.append(t['n_samples'])

            val_preds.extend(v['predictions']);  val_labels.extend(v['labels'])
            test_preds.extend(t['predictions']); test_labels.extend(t['labels'])

        # Pooled estimators. Clients report mean losses, so the pooled loss is
        # sum_i n_i L_i / sum_i n_i.
        def _pooled_loss(losses, sizes) -> float:
            total = float(sum(sizes))
            if total <= 0:
                return 0.0
            return float(sum(l * n for l, n in zip(losses, sizes)) / total)

        global_val_acc = float(accuracy_score(val_labels, val_preds)) if val_labels else 0.0
        global_val_f1 = float(f1_score(val_labels, val_preds, average='macro',
                                       zero_division=0)) if val_labels else 0.0
        global_val_loss = _pooled_loss(val_losses, val_sizes)

        global_test_acc = float(accuracy_score(test_labels, test_preds)) if test_labels else 0.0
        global_test_f1 = float(f1_score(test_labels, test_preds, average='macro',
                                        zero_division=0)) if test_labels else 0.0
        global_test_loss = _pooled_loss(test_losses, test_sizes)
        return {
            'val_accs': val_accs, 'val_f1s': val_f1s, 'val_losses': val_losses,
            'test_accs': test_accs, 'test_f1s': test_f1s, 'test_losses': test_losses,
            'global_val_acc': global_val_acc, 'global_val_f1': global_val_f1,
            'global_val_loss': global_val_loss, 'global_test_acc': global_test_acc,
            'global_test_f1': global_test_f1, 'global_test_loss': global_test_loss,
        }

    # ------------------------------------------------------------------ #
    #  Reporting
    # ------------------------------------------------------------------ #
    def get_final_metrics(self) -> Dict[str, float]:
        """Return the last value of every non-empty `round_history` entry.

        Empty entries (such as SPDNet diagnostics for EEGNet) are skipped. Returns an
        empty dict before the first round.
        """
        if not self.round_history['round']:
            return {}
        return {k: v[-1] for k, v in self.round_history.items() if v}

    def get_metrics_at_round(self, round_num) -> Dict[str, Any]:
        """Return every `round_history` entry at round number `round_num`.

        The row is found by round number, not position. Returns an empty dict if the
        round is not in the history.
        """
        rounds = list(self.round_history.get('round') or [])
        if round_num not in rounds:
            return {}
        pos = rounds.index(round_num)
        return {k: v[pos] for k, v in self.round_history.items()
                if v and len(v) > pos}

    def get_selected_round(self, select_metric: Optional[str] = None,
                           report_metrics=('global_test_acc', 'global_test_f1')):
        """Return the round with the best selection metric and its report metrics.

        Ties go to the earliest round. Falls back to 'global_val_acc' when the
        selection series is empty.

        Args:
            select_metric: a round_history key overriding `config.selection_metric`.
                Keys containing 'loss' are minimised, others maximised.
            report_metrics: round_history keys to return at the selected round.

        Returns:
            (round number, {metric: value}), or (None, {}) without history.
        """
        key, minimize = self.selection_rule()
        if select_metric is not None:
            key, minimize = select_metric, ('loss' in select_metric.lower())
        series = list(self.round_history.get(key) or [])
        if not series:
            key, minimize = 'global_val_acc', False
            series = list(self.round_history.get(key) or [])
        if not series:
            return None, {}
        from rfl.utils.schedules import best_position
        pos = best_position(series, minimize)
        if pos is None:                     # no finite value: nothing to select
            return None, {}
        return (self.round_history['round'][pos],
                {m: self.round_history[m][pos] for m in report_metrics
                 if self.round_history.get(m)})

    def load_reported_model(self, clients: List[FLClient], report_rule: str) -> None:
        """Put the reported model on every client: global state plus its local keys.

        'last' loads the last global state, with the clients' current local keys.
        'selected' loads the global state of the selected round, with the local keys
        each client had at that round, so that the model is the one that was evaluated.
        Local-only clients keep their own model: their last state, or with
        'selected' the state of their own best validation round.
        """
        if self.config.fl_local_only:
            # Each client keeps its own model: the last one, or with 'selected' the one
            # of its own best validation round.
            if report_rule == 'selected':
                for c in clients:
                    if c.client_id in self._local_best:
                        c.local_model.load_state_dict(self._local_best[c.client_id])
            return
        if report_rule == 'selected' and self.best_state_dict is not None:
            for c in clients:
                c.receive_global_model(self.best_state_dict)
                if c.client_id in self.best_local_states:
                    c.load_local_state(self.best_local_states[c.client_id])
        else:
            for c in clients:
                c.receive_global_model(self.global_state_dict)

    def save_global_model(self, path: str):
        """Save the last and the selected global state, with the round history.

        The checkpoint holds model_state_dict (last round), best_state_dict (selected
        round), best_round, best_score, selection_metric, round_history and
        config_hash.
        """
        torch.save({
            'model_state_dict': self.global_state_dict,       # last round
            'best_state_dict': self.best_state_dict,          # selected round
            'best_round': self.best_round,
            'best_score': self.best_score,
            'selection_metric': self.selection_rule()[0],
            'round_history': self.round_history,
            'config_hash': getattr(self.config, 'config_hash', None),
        }, path)
        logger.info(f"[Server] Global model saved to: {path} "
              f"(selected round {self.best_round})")

    def load_global_model(self, path: str):
        """Load the last-round state (and the history) saved by `save_global_model`."""
        ckpt = torch.load(path, map_location=self.device)
        self.global_state_dict = ckpt['model_state_dict']
        self.global_model.load_state_dict(self.global_state_dict)
        if 'round_history' in ckpt:
            self.round_history = ckpt['round_history']
        logger.info(f"[Server] Global model loaded from: {path}")


# =========================================================================== #
#                              Client creation
# =========================================================================== #
def iid_partition(indices, y, n_clients: int, seed: int, tag: int) -> List[List[int]]:
    """Deal `indices` to `n_clients` clients at random, stratified by the labels `y`.

    The indices of each class are shuffled, the classes are put one after the other,
    and the k-th index goes to client k mod n_clients. Every client then receives
    floor or ceil of n_c / n_clients trials of each class c, and the client sizes differ
    by at most one. The draw depends on (seed, tag) only, so the train, validation and
    test splits (tags 0, 1, 2) are dealt independently and reproducibly.
    """
    rng = np.random.default_rng([int(seed), int(tag)])
    idx = np.asarray(indices)
    order = np.concatenate([rng.permutation(idx[y[idx] == c])
                            for c in np.unique(y[idx])]) if len(idx) else idx
    return [order[k::n_clients].tolist() for k in range(n_clients)]


def create_fl_clients(dataset_name: str, all_subjects: List[int],
                      subjects_per_client: int, config: Any,
                      device: torch.device, verbose: bool = True) -> List[FLClient]:
    """Create FL clients from a MOABB dataset. See `create_fl_clients_by_subject`."""
    return create_fl_clients_by_subject(
        dataset_name=dataset_name, all_subjects=all_subjects,
        subjects_per_client=subjects_per_client, config=config,
        device=device, verbose=verbose)


def create_fl_clients_by_subject(dataset_name: str, all_subjects: List[int],
                                 subjects_per_client: int, config: Any,
                                 device: torch.device,
                                 verbose: bool = True) -> List[FLClient]:
    """Load a MOABB dataset and build one client per group of subjects.

    The data is loaded once and split once, stratified by label, with the
    (val_size, test_size, data_split_seed) of the config. Each trial then goes to the
    client that owns its subject, or, with config.fl_client_partition == 'iid', to a
    client drawn by `iid_partition` (the same number of clients). The union of the
    client splits is therefore the centralized split in both cases.

    Args:
        dataset_name: MOABB dataset name (rfl/data/registry.py).
        all_subjects: subject IDs to load.
        subjects_per_client: consecutive subjects per client.
        config: an FLConfig.
        device: torch device.
        verbose: print the partition and split sizes.

    Returns:
        The clients (FLClient, or DPFLClient under DP), with their data loaders.
    """
    from sklearn.model_selection import train_test_split

    from rfl.data.data_utils import resolve_resample
    from rfl.data.reduction import reduce_pool, reduced_positions, subject_rng
    from rfl.federated.fl_config import create_client_subject_mapping

    client_subjects = create_client_subject_mapping(all_subjects, subjects_per_client)
    iid = config.fl_client_partition == 'iid'
    if verbose:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Creating {len(client_subjects)} FL clients "
                    + ("(IID: trials pooled and dealt by class)" if iid
                       else "(subject-based, non-IID)"))
        logger.info(f"{'=' * 70}")
        if not iid:
            for i, subs in enumerate(client_subjects):
                logger.info(f"  Client {i}: subjects {subs}")

    native_sfreq = DATASET_SAMPLING_RATES[dataset_name]
    target_sfreq = resolve_resample(config.resample, native_sfreq)
    if verbose and target_sfreq != native_sfreq:
        logger.info(f"[Data] Resampling {native_sfreq} Hz -> {target_sfreq} Hz")

    X_all, y_all, meta_all = load_moabb_dataset(
        dataset_name=dataset_name, subjects=all_subjects,
        resample=target_sfreq, fmin=config.fmin, fmax=config.fmax, verbose=verbose)

    X_all, y_all, meta_all = filter_binary_task(
        X=X_all, y=y_all, meta=meta_all,
        task=config.task,
        dataset_name=dataset_name, verbose=verbose)

    # ---- one global split, identical to the centralized one -------------------
    val_size = float(config.val_size)
    test_size = float(config.test_size)
    split_seed = int(config.data_split_seed)

    indices = np.arange(len(y_all))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=split_seed, stratify=y_all)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size / (1.0 - test_size),
        random_state=split_seed, stratify=y_all[train_val_idx])

    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    if verbose:
        logger.info(f"\n[Split] seed={split_seed}  train={len(train_idx)} "
              f"val={len(val_idx)} test={len(test_idx)} (matches centralized)")

    meta = meta_all.reset_index(drop=True)
    use_local_fc = bool(config.fl_local_fc)
    if iid:
        k = len(client_subjects)
        parts = [iid_partition(idx, y_all, k, split_seed, tag)
                 for tag, idx in enumerate((train_idx, val_idx, test_idx))]
        subject_of = meta['subject'].to_numpy()

    # Data quantity (rfl/data/reduction.py): fewer calibration trials per subject.
    n_cal = config.train_trials_per_class
    if n_cal and iid:
        raise ValueError("--train-trials-per-class reduces subjects: use it with "
                         "--client-partition subject")
    reduced_pos = (reduced_positions(len(all_subjects), int(config.reduced_subjects),
                                     int(config.seed)) if n_cal else set())
    position = {int(s): i for i, s in enumerate(all_subjects)}

    clients: List[FLClient] = []
    for client_id, subs in enumerate(client_subjects):
        reduced = False
        if iid:
            tr, va, te = (sorted(p[client_id]) for p in parts)
            subs = sorted({int(s) for s in subject_of[tr + va + te]})
        else:
            tr, va, te = [], [], []
            for sub in subs:
                owned = meta[meta['subject'] == sub].index.tolist()
                tr_s = [i for i in owned if i in train_set]
                va_s = [i for i in owned if i in val_set]
                te += [i for i in owned if i in test_set]
                if position[int(sub)] in reduced_pos:
                    tr_s, va_s, info = reduce_pool(
                        tr_s, va_s, y_all, int(n_cal), config.calibration_val_fraction,
                        config.calibration_order, subject_rng(config.seed, position[int(sub)]))
                    reduced = True
                    if info["short_classes"] and verbose:
                        logger.info(f"[Data] subject {sub}: {info['short_classes']} "
                                    f"class(es) with fewer than {n_cal} trials in the "
                                    f"pool of {info['pool']}: all of them are kept")
                tr += tr_s
                va += va_s
            tr, va, te = sorted(tr), sorted(va), sorted(te)

        if verbose:
            logger.info(f"[Client {client_id}] subjects {subs}: "
                  f"train={len(tr)} val={len(va)} test={len(te)}"
                  + ("  (reduced)" if reduced else ""))

        client = _client_class(config)(
            client_id=client_id, subjects=subs, dataset_name=dataset_name,
            config=config, device=device,
            use_fedbn=bool(config.fl_fedbn),
            use_local_fc=use_local_fc,
        )
        if not tr:
            raise ValueError(f"client {client_id} (subjects {subs}) has no training "
                             "trial: a subject failed to load or holds too few trials.")
        client.prepare_data_from_splits(
            X_all[tr], y_all[tr], X_all[va], y_all[va], X_all[te], y_all[te],
            verbose=False)
        client.reduced = reduced
        clients.append(client)

    if verbose:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"{len(clients)} clients | "
              f"train={sum(c.n_samples for c in clients)} "
              f"test={sum(len(c.test_loader.dataset) for c in clients)} "
              f"(both match centralized)")
        logger.info(f"{'=' * 70}")
    return clients


def create_fl_clients_synthetic(client_splits, config, device,
                                dataset_name: str = 'synthetic',
                                verbose: bool = True) -> List[FLClient]:
    """Build one client per pre-generated synthetic split (rfl/data/synthetic.py).

    Args:
        client_splits: one dict per client with keys X_train, y_train, X_val, y_val,
            X_test and y_test.
        config: an FLConfig, normally with data_type='spd'.
        device: torch device.
        dataset_name: name stored on the clients.
        verbose: print split sizes.

    Returns:
        The clients, with their data loaders.
    """
    from rfl.data.reduction import reduce_synthetic
    client_splits, reduced_pos = reduce_synthetic(list(client_splits), config)
    use_local_fc = bool(config.fl_local_fc)
    clients: List[FLClient] = []
    for cid, split in enumerate(client_splits):
        client = _client_class(config)(
            client_id=cid, subjects=[f'synthetic_{cid}'], dataset_name=dataset_name,
            config=config, device=device,
            use_fedbn=bool(config.fl_fedbn),
            use_local_fc=use_local_fc)
        client.prepare_data_from_splits(
            split['X_train'], split['y_train'],
            split['X_val'], split['y_val'],
            split['X_test'], split['y_test'],
            verbose=verbose)
        client.reduced = cid in reduced_pos
        clients.append(client)
    if verbose:
        logger.info(f"Created {len(clients)} synthetic-SPD clients (data_type=spd)")
    return clients