"""DPFLClient: record-level differentially private local training.

`DPFLClient` subclasses `federated.client.FLClient` and overrides only `train_local`.
Data handling, model exchange and evaluation are inherited.

One private round on a client with m_i training records:

1. Check that the model carries no normalisation layer with running statistics. Such
   a layer would couple the records of a batch and break the sensitivity 2C/B
   (Appendix D.2 of the DP-RFedProj paper).
2. Run local_epochs * floor(m_i / B) steps. This is at most
   tau = local_epochs * ceil(m_max / B), the count the accountant composes, so the
   reported budget is an upper bound for every client. Each step draws its own size-B
   minibatch uniformly without replacement, independently of the other steps. This is
   the sampling the accountant assumes (q = B / m_star). A shuffled DataLoader
   partitions each epoch into disjoint batches instead, and a short last batch would
   raise the sensitivity above 2C/B.
3. Each step applies `dp_grad.dp_local_step`: per-sample tangent gradients, joint clip
   to C, average, projected Gaussian noise of std sigma, optimiser step, then the
   model constraints (Stiefel projection, max-norm) as post-processing.

The privatised vector holds every parameter that trains during the round, shared or
personal (`dp_grad.federated_param_groups`). A personal head trained during the rounds
(`fl_local_head_mode='train'`) is therefore clipped and noised with the trunk, and
never uploaded. A frozen head or trunk is data-independent and stays out of the vector.
Every parameter of the model is thus a function of the privatised gradients, and the
upload is post-processing of them.

C and sigma are read from `config.dp_clip_C` and `config.dp_sigma`. C is fixed a
priori, sigma is written by `orchestration.setup_dp`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from rfl.federated.client import FLClient
from rfl.federated.privacy.dp_grad import (
    assert_dp_safe_norm_layers,
    dp_local_step,
    federated_param_groups,
)

__all__ = ["DPFLClient"]


class DPFLClient(FLClient):
    """FLClient whose `train_local` runs the DP-RFedProj local steps.

    Takes the same arguments as `FLClient`. Two CPU generators, seeded from
    `config.seed` and the client id, drive the noise and the minibatch sampling.
    """

    def __init__(self, *args, **kwargs):
        """See `FLClient`."""
        super().__init__(*args, **kwargs)
        seed = int(getattr(self.config, "seed", 42))
        # Independent reproducible streams for the noise and the minibatch draws, per
        # client: NumPy's SeedSequence hashes (seed, client, purpose), so no two
        # streams share a state. The noise is reproducible from the seed, which suits a
        # simulation. A deployment would draw it from a secret source.
        import numpy as np
        noise_seed, sample_seed = (
            int(np.random.SeedSequence([seed, int(self.client_id), tag])
                .generate_state(1, dtype=np.uint64)[0] >> 1)
            for tag in (1, 2))
        self._dp_noise_gen = torch.Generator(device="cpu")
        self._dp_noise_gen.manual_seed(noise_seed)
        self._dp_sample_gen = torch.Generator(device="cpu")
        self._dp_sample_gen.manual_seed(sample_seed)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _dp_active(self) -> bool:
        """True when DP is enabled."""
        return bool(getattr(self.config, "dp_enabled", False))

    def _sample_wor_batch(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw B training records uniformly without replacement and collate them."""
        dataset = self.train_loader.dataset
        m = len(dataset)
        idx = torch.randperm(m, generator=self._dp_sample_gen)[:B].tolist()
        return self.train_loader.collate_fn([dataset[i] for i in idx])

    # ------------------------------------------------------------------ #
    #  DP local training
    # ------------------------------------------------------------------ #
    def train_local(self, n_epochs: int, criterion: nn.Module, learning_rate: float,
                    is_spdnet: bool = False, apply_constraints: bool = True,
                    verbose: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Run one round of private local training (see the module docstring).

        Falls back to `FLClient.train_local` when DP is off. Arguments and return value
        are those of `FLClient.train_local`. The metrics also include `dp_local_steps`.

        Raises:
            ValueError: if B > m_i, if floor(m_i / B) * n_epochs = 0, or if the model
                carries a normalisation layer with running statistics.
        """
        if not self._dp_active():
            return super().train_local(n_epochs, criterion, learning_rate,
                                       is_spdnet, apply_constraints, verbose)

        if self.local_model is None or self.train_loader is None:
            raise RuntimeError("Model/data not prepared.")

        # The (C, sigma) pair the accountant was called with.
        C = float(self.config.dp_clip_C)
        sigma = float(self.config.dp_sigma)
        B = int(self.config.batch_size)

        m_i = len(self.train_loader.dataset)
        if B > m_i:
            raise ValueError(
                f"[client {self.client_id}] batch_size B={B} > m_i={m_i}: a DP step "
                "consumes exactly B records, and the sensitivity 2C/B assumes it. "
                "setup_dp already requires B <= min_i m_i.")

        model = self.local_model
        # Refuse batch normalisation (module docstring, step 1).
        assert_dp_safe_norm_layers(model)

        # The DP forwards set their own mode in dp_grad._forward_mode.
        model.eval()
        from rfl.training.optim import make_optimizer
        optimizer = make_optimizer(model.parameters(), self.config, learning_rate)

        groups = federated_param_groups(model)
        vectorized = bool(getattr(self.config, "dp_vectorized_grads", True))

        # Nothing trains (frozen trunk and frozen head): no step, the data are not read.
        if not groups:
            return self.upload_state(), {"train_loss": float("nan"),
                                         "train_acc": float("nan"),
                                         "n_samples": self.n_samples,
                                         "dp_local_steps": 0}

        # local_epochs * floor(m_i / B) steps, never more than the accountant charges
        # (local_epochs * ceil(m_max / B)).
        n_planned = n_epochs * (m_i // B)

        # Zero steps would return the received model and still be aggregated. Refuse.
        if n_planned == 0:
            raise ValueError(
                f"[DP][client {self.client_id}] batch_size={B} exceeds this client's "
                f"training set (m_i={m_i}), so floor(m_i/B)=0 and it would take ZERO "
                f"optimiser steps while still being aggregated. Lower --batch-size below "
                f"{m_i}, raise --local-epochs, or use fewer subjects per client.")

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(n_planned):
            inputs, labels = self._sample_wor_batch(B)
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            loss, logits = dp_local_step(
                model, optimizer, inputs, labels, criterion, groups,
                C=C, sigma=sigma, generator=self._dp_noise_gen,
                apply_constraints=apply_constraints, vectorized=vectorized)

            # Training accuracy from the step's own logits.
            with torch.no_grad():
                total_correct += int(logits.argmax(1).eq(labels).sum())
            total_samples += B
            total_loss += loss * B

        # train_loss and train_acc are computed on the private minibatches. They are
        # simulation diagnostics, logged in round_history: a deployed client would not
        # send them, and they are outside the (eps, delta) guarantee.
        metrics = {
            "train_loss": total_loss / total_samples if total_samples else 0.0,
            "train_acc": total_correct / total_samples if total_samples else 0.0,
            "n_samples": self.n_samples,
            "dp_local_steps": n_planned,
        }
        return self.upload_state(), metrics