"""Federated learning client: local data, local model, local training and evaluation.

Some state never leaves the client:

* Local normalisation layers (FedBN). Their parameters and running statistics are left
  out of the aggregation (rfl/federated/aggregation.py) and are not overwritten on receive.
  Which layers are local is decided by rfl/federated/param_groups.py from module types and
  class names. `config.fl_fedbn=False` shares them instead.
* The classifier head, when `use_local_fc` is set (FedPer). It is stripped from the
  uploaded state dict, so each client keeps its own head.

The key sets are computed once, in `set_model`, which also freezes the parameters that
do not train during the rounds: the trunk with `config.freeze_trunk`, and a personal
head with `config.fl_local_head_mode == 'frozen'`.

A non-private round of local training is `n_epochs` passes over a shuffled DataLoader
of batch size B, so ceil(m_i / B) steps per epoch, the last one on the remaining
m_i mod B records. SPDNet steps are Adam on the Euclidean gradient followed by the
polar projection of the BiMap weights (`SPDNet.project_stiefel`).
rfl/federated/privacy/dp_client.py subclasses this class and overrides `train_local` only.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from rfl.data.datasets import build_dataset, make_loader
from rfl.data.registry import DATASET_CONFIGS as _DATASET_META
from rfl.data.registry import get_sampling_rate

logger = logging.getLogger(__name__)

# Native sampling rate (Hz) per dataset, from rfl/data/registry.py.
DATASET_SAMPLING_RATES = {name: cfg['sampling_rate']
                          for name, cfg in _DATASET_META.items()}


class FLClient:
    """One federated client: the data of a fixed group of subjects and a local model.

    Typical use: `prepare_data_from_splits`, then `set_model`, then once per round
    `receive_global_model`, `train_local` and `evaluate_global_model`.

    Args:
        client_id: unique identifier.
        subjects: subject IDs owned by this client.
        dataset_name: MOABB dataset name, or 'synthetic' for precomputed SPD data.
        config: an FLConfig (or any object exposing the same attributes).
        device: torch device.
        use_fedbn: keep local normalisation layers on the client, as decided by
            rfl/federated/param_groups.py (`FLConfig.fl_fedbn`, True by default). False
            shares them: the received global values overwrite them.
        use_local_fc: keep the classifier head local and out of the aggregation.
    """

    def __init__(self, client_id: int, subjects: List[Any], dataset_name: str,
                 config: Any, device: torch.device,
                 use_fedbn: bool = True, use_local_fc: bool = False):
        self.client_id = client_id
        self.subjects = subjects
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        self.use_fedbn = use_fedbn
        self.use_local_fc = use_local_fc

        # Local key sets, filled by set_model().
        self._bn_param_names: Set[str] = set()        # normalisation weight / bias
        self._bn_buffer_names: Set[str] = set()       # running_mean / running_var / count
        self._local_fc_param_names: Set[str] = set()  # classifier weight / bias

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.n_samples = 0
        self.reduced = False          # fewer calibration trials (rfl/data/reduction.py)
        self._optimizer = None        # kept across rounds in local-only mode
        self.info: Dict[str, Any] = {}

        self.local_model: nn.Module | None = None

    # ------------------------------------------------------------------ #
    #  Data
    # ------------------------------------------------------------------ #
    def prepare_data_from_splits(self, X_train, y_train, X_val, y_val,
                                 X_test, y_test, verbose: bool = True) -> Dict[str, Any]:
        """Build the train, validation and test loaders from this client's arrays.

        The global split is made once by the server (`create_fl_clients_by_subject`).
        This method only wraps the arrays with `data.datasets.build_dataset`, using
        `config.data_type` ('raw', 'covariance' or 'spd').

        Returns:
            The `info` dict: n_channels, n_timepoints, n_classes, classes, n_train,
            n_val, n_test and sampling_rate (the rate after resampling, None for
            synthetic data).
        """
        from rfl.data.data_utils import resolve_resample

        # Rate the data was loaded at. EEGNet sizes its temporal kernels from it.
        sfreq = resolve_resample(self.config.resample,
                                 get_sampling_rate(self.dataset_name))

        kind = self.config.data_type.lower()
        ds_kwargs = dict(
            normalization_type=self.config.normalization_type,
            estimator=self.config.cov_estimator,
            shrinkage=self.config.cov_shrinkage,
            regularization=self.config.cov_regularization,
        )
        train_ds = build_dataset(X_train, y_train, kind, **ds_kwargs)
        val_ds = build_dataset(X_val, y_val, kind, **ds_kwargs)
        test_ds = build_dataset(X_test, y_test, kind, **ds_kwargs)

        bs = int(self.config.batch_size)
        # A batch of one record has no batch statistics (SPD batch norm), or noisy ones
        # (BatchNorm over the time axis only): dropped for both models alike.
        self.train_loader = make_loader(train_ds, bs, shuffle=True,
                                        avoid_single_batch=self.config.has_batch_norm())
        self.val_loader = make_loader(val_ds, bs, shuffle=False)
        self.test_loader = make_loader(test_ds, bs, shuffle=False)
        self.n_samples = len(train_ds)

        all_labels = (np.concatenate([y_train, y_val, y_test])
                      if len(y_train) else np.asarray(y_train))
        self.info = {
            'n_channels': X_train.shape[1] if len(X_train) else 0,
            'n_timepoints': X_train.shape[2] if len(X_train) else 0,
            'n_classes': len(np.unique(all_labels)) if len(all_labels) else 0,
            'classes': sorted(int(c) for c in np.unique(all_labels)) if len(all_labels) else [],
            'n_train': len(train_ds),
            'n_val': len(val_ds),
            'n_test': len(test_ds),
            'sampling_rate': sfreq,
        }
        if verbose:
            logger.info(f"[Client {self.client_id}] train={self.info['n_train']} "
                  f"val={self.info['n_val']} test={self.info['n_test']}")
        return self.info

    # ------------------------------------------------------------------ #
    #  Model
    # ------------------------------------------------------------------ #
    def set_model(self, model: nn.Module):
        """Store a deep copy of `model` as the local model and compute the local key sets."""
        from rfl.federated.param_groups import (
            identify_bn_buffer_names, identify_bn_param_names,
            identify_classifier_param_names,
        )

        from rfl.federated.param_groups import freeze_params, freeze_trunk

        self.local_model = copy.deepcopy(model).to(self.device)

        if self.use_fedbn:
            self._bn_param_names = identify_bn_param_names(self.local_model)
            self._bn_buffer_names = identify_bn_buffer_names(self.local_model)
        else:
            self._bn_param_names = set()
            self._bn_buffer_names = set()

        if self.use_local_fc:
            self._local_fc_param_names = identify_classifier_param_names(self.local_model)
            if self._local_fc_param_names:
                n = sum(p.numel() for k, p in self.local_model.named_parameters()
                        if k in self._local_fc_param_names)
                logger.info(f"  [Client {self.client_id}] local-FC: "
                      f"{sorted(self._local_fc_param_names)} ({n} scalars) kept local")
        else:
            self._local_fc_param_names = set()

        # Parameters that do not train during the rounds.
        if bool(getattr(self.config, "freeze_trunk", False)):
            freeze_trunk(self.local_model)
        if (self._local_fc_param_names
                and getattr(self.config, "fl_local_head_mode", "train") == "frozen"):
            freeze_params(self.local_model, self._local_fc_param_names)

    def _skip_keys(self) -> Set[str]:
        """State-dict keys that a received global state must not overwrite.

        These are the local normalisation parameters and buffers and, in local-FC mode,
        the head. The server's own copies of local keys are never updated, so loading
        them would reset the client.
        """
        return (self._local_fc_param_names
                | self._bn_param_names | self._bn_buffer_names)

    def _selective_load(self, global_state_dict: Dict[str, torch.Tensor]):
        """Load the global state, preserving the local keys."""
        skip = self._skip_keys()
        if not skip:
            self.local_model.load_state_dict(copy.deepcopy(global_state_dict), strict=True)
            return

        state = self.local_model.state_dict()
        for key, val in global_state_dict.items():
            if key not in skip:
                state[key] = copy.deepcopy(val)
        # strict=False because the aggregated dict may lack keys, e.g. the classifier
        # in local-FC mode.
        self.local_model.load_state_dict(state, strict=False)

    def local_state_dict(self) -> Dict[str, torch.Tensor]:
        """Copy of the keys that stay on the client (personal head, local normalisation)."""
        state = self.local_model.state_dict()
        return {k: state[k].detach().clone() for k in self._skip_keys() if k in state}

    def load_local_state(self, local_state: Dict[str, torch.Tensor]) -> None:
        """Restore keys saved by `local_state_dict`, leaving the shared keys as they are."""
        state = self.local_model.state_dict()
        for k, v in local_state.items():
            state[k] = v.clone()
        self.local_model.load_state_dict(state, strict=True)

    def upload_state(self) -> Dict[str, torch.Tensor]:
        """Copy of the local state dict that is sent to the server.

        The personal head is removed in local-FC mode. Local normalisation keys are
        sent but left out by every aggregation rule (rfl/federated/aggregation.py).
        """
        state_dict = copy.deepcopy(self.local_model.state_dict())
        if self._local_fc_param_names:
            state_dict = {k: v for k, v in state_dict.items()
                          if k not in self._local_fc_param_names}
        return state_dict

    def receive_global_model(self, global_state_dict: Dict[str, torch.Tensor]):
        """Load the global weights, keeping the local keys (see `_skip_keys`)."""
        if self.local_model is None:
            raise RuntimeError("Local model not set. Call set_model() first.")
        self._selective_load(global_state_dict)

    # ------------------------------------------------------------------ #
    #  Local training
    # ------------------------------------------------------------------ #
    def train_local(self, n_epochs: int, criterion: nn.Module, learning_rate: float,
                    is_spdnet: bool = False, apply_constraints: bool = True,
                    verbose: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Train the local model for `n_epochs` with the configured optimiser.

        In federated mode the optimiser is rebuilt on every call (every round starts
        from the global model). In local-only mode it is kept, so its state carries over
        as in plain training. The learning rate of the round applies in both cases.
        After each step the model's `apply_constraints()` (EEGNet max-norm, if
        `apply_constraints`) and `project_stiefel()` (SPDNet polar re-projection) are
        called when they exist.

        Frozen parameters (see `set_model`) are not trained. If nothing is trainable,
        the round runs forward passes only, in eval mode, so that BatchNorm statistics
        do not move either.

        Args:
            n_epochs: number of local epochs.
            criterion: loss function.
            learning_rate: Adam learning rate for this round.
            is_spdnet: not used by this class. The server passes it to every client class.
            apply_constraints: apply EEGNet's max-norm constraints after each step.
            verbose: show a progress bar.

        Returns:
            (state_dict, metrics). The state dict to upload has the local head removed
            in local-FC mode. metrics holds
            train_loss, train_acc (averages over all local steps) and n_samples.

        Raises:
            RuntimeError: if `set_model` or `prepare_data_from_splits` was not called.
        """
        if self.local_model is None:
            raise RuntimeError("Local model not set. Call set_model() first.")
        if self.train_loader is None:
            raise RuntimeError("Data not prepared. Call prepare_data_from_splits() first.")

        model = self.local_model

        # If nothing is trainable, backward() would fail and train() would still update
        # BatchNorm statistics, so the round runs in eval mode, forward only.
        trainable = any(p.requires_grad for p in model.parameters())
        model.train() if trainable else model.eval()

        from rfl.training.optim import make_optimizer
        if bool(getattr(self.config, "fl_local_only", False)):
            # Local-only training is plain training on the client's data: the optimiser
            # state (Adam moments, SGD momentum) carries over from round to round, as in
            # the centralised loop. Only the learning rate follows the schedule.
            if self._optimizer is None:
                self._optimizer = make_optimizer(model.parameters(), self.config,
                                                 learning_rate)
            optimizer = self._optimizer
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        else:
            # Federated rounds: each round starts from the received global model with a
            # new optimiser, as in FedAvg.
            optimizer = make_optimizer(model.parameters(), self.config, learning_rate)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(n_epochs):
            batches = self.train_loader
            if verbose:
                batches = tqdm(batches, leave=False,
                               desc=f"Client {self.client_id} epoch {epoch + 1}/{n_epochs}")

            for inputs, labels in batches:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if trainable:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    # Forward only, so that training metrics are still reported.
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                if apply_constraints and hasattr(model, 'apply_constraints'):
                    model.apply_constraints()
                if hasattr(model, 'project_stiefel'):
                    model.project_stiefel()

                bs = labels.size(0)
                total_correct += int(outputs.argmax(1).eq(labels).sum())
                total_samples += bs
                total_loss += float(loss.detach()) * bs

        metrics = {
            'train_loss': total_loss / total_samples if total_samples else 0.0,
            'train_acc': total_correct / total_samples if total_samples else 0.0,
            'n_samples': self.n_samples,
        }

        return self.upload_state(), metrics

    # ------------------------------------------------------------------ #
    #  Evaluation of the aggregated model
    # ------------------------------------------------------------------ #
    def evaluate_global_model(self, global_state_dict: Dict[str, torch.Tensor],
                              criterion: nn.Module,
                              splits: Sequence[str] = ("val", "test"),
                              ) -> Dict[str, Dict[str, Any]]:
        """Evaluate the global model, combined with this client's local keys.

        The global state is loaded into the local model once (keeping `_skip_keys`),
        so the evaluated model is the one this client would deploy. Side effect: the
        local model keeps the loaded weights.

        Args:
            global_state_dict: the aggregated state dict.
            criterion: loss function.
            splits: any of 'train', 'val', 'test'.

        Returns:
            {split: {loss, acc, f1_macro, n_samples, predictions, labels}}, where loss
            is the mean over the split. The server pools these by n_samples.

        Raises:
            RuntimeError: if `set_model` was not called.
            ValueError: for an unknown split name.
        """
        if self.local_model is None:
            raise RuntimeError("Local model not set. Call set_model() first.")
        self._selective_load(global_state_dict)
        return self.evaluate_local_model(criterion, splits)

    def evaluate_local_model(self, criterion: nn.Module,
                             splits: Sequence[str] = ("val", "test"),
                             ) -> Dict[str, Dict[str, Any]]:
        """Evaluate the local model as it is, without loading any global state.

        Used after the head re-fit (rfl/federated/personalization.py) and in local-only
        mode. Arguments and return value as in `evaluate_global_model`.
        """
        if self.local_model is None:
            raise RuntimeError("Local model not set. Call set_model() first.")
        self.local_model.eval()

        loaders = {'train': self.train_loader, 'val': self.val_loader,
                   'test': self.test_loader}
        out: Dict[str, Dict[str, Any]] = {}

        for split in splits:
            loader = loaders.get(split)
            if loader is None:
                raise ValueError(f"Unknown split {split!r} (expected train/val/test)")

            total_loss = 0.0
            correct = 0
            total = 0
            preds: List[int] = []
            labels_all: List[int] = []

            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    outputs = self.local_model(inputs)
                    loss = criterion(outputs, labels)
                    predicted = outputs.argmax(1)

                    bs = labels.size(0)
                    correct += int(predicted.eq(labels).sum())
                    total += bs
                    total_loss += float(loss.detach()) * bs
                    preds.extend(predicted.cpu().numpy().tolist())
                    labels_all.extend(labels.cpu().numpy().tolist())

            out[split] = {
                'loss': total_loss / total if total else 0.0,
                'acc': correct / total if total else 0.0,
                'f1_macro': float(f1_score(labels_all, preds, average='macro',
                                           zero_division=0)) if total else 0.0,
                'n_samples': total,
                'predictions': preds,
                'labels': labels_all,
            }
        return out