"""Centralised training loop for EEGNet and SPDNet.

One validation metric, `config.selection_metric` ('val_acc' by default), drives three
decisions:

* the plateau schedule: the rate is multiplied by `config.lr_plateau_factor` after
  `config.lr_plateau_patience` epochs without a strict improvement, floored at
  `config.lr_min`,
* early stopping after `config.early_stopping_patience` epochs without a strict
  improvement (0 runs the full budget),
* the reported model: the best checkpoint is the one the centralised runner reloads
  for the test evaluation (unless `config.report_rule == 'last'`).

rfl/federated/server.py applies the same rules to the pooled validation metric. With
the default protocol (last epoch reported, no plateau, no early stopping) the metric
only matters for the record of the best epoch. With these rules on, 'val_loss' is
often the better choice: accuracy on a small validation split ties often, and a tie
counts as no improvement.

SPDNet is trained with Adam on the Euclidean gradient of the BiMap weights, followed by
the polar projection of the weights onto the Stiefel manifold after every step
(`SPDNet.project_stiefel`). The gradient itself is not projected on the tangent space.
The federated non-private clients use the same step (rfl/federated/client.py).

Schedules ('none', 'cosine', 'step', 'plateau') come from rfl/utils/schedules.py, the
function the federated server uses per round, so each one means the same thing in both
pipelines. The plateau rule lowers the rate on the `lr_plateau_patience`-th epoch
without a strict improvement. It is recomputed from the history at every epoch, so a
resumed run follows the same schedule as an uninterrupted one.
"""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from rfl.models.spdnet import fix_batch_norm_dtype
from rfl.utils.schedules import best_position, compute_round_lr

logger = logging.getLogger(__name__)


def setup_training(
    model: nn.Module,
    config: Any,  # ExperimentConfig
    device: torch.device,
    is_spdnet: bool = False
) -> Tuple[Optimizer, Any]:
    """Build the Adam optimiser. The learning rate is set per epoch by `train_model`.

    Args:
        model: Model to train.
        config: ExperimentConfig (learning_rate, weight_decay, lr_schedule and the
            plateau fields are read).
        device: Device, used to fix SPD batch-norm dtypes for SPDNet.
        is_spdnet: Whether the model is an SPDNet.

    Returns:
        (optimizer, None). The second element is kept for API compatibility.

    Raises:
        ValueError: If lr_schedule is not 'none', 'cosine', 'step' or 'plateau'.
    """
    lr = config.learning_rate
    weight_decay = config.weight_decay

    from rfl.training.optim import make_optimizer
    optimizer = make_optimizer(model.parameters(), config, lr, weight_decay)
    logger.info(f"Optimizer: {config.optimizer}"
                + (f", Stiefel weights by {config.spdnet_stiefel}" if is_spdnet else ""))
    if is_spdnet:
        fix_batch_norm_dtype(model, device)

    schedule = str(config.lr_schedule).lower()
    if schedule not in ('none', 'cosine', 'step', 'plateau'):
        raise ValueError(f"lr_schedule={schedule!r} is not supported (use 'none', "
                         "'cosine', 'step' or 'plateau')")
    # The rate is set at the start of every epoch by utils.schedules.compute_round_lr,
    # the function the federated server uses for its rounds.
    logger.info(f"LR schedule: {schedule} from {lr:g} (floor {config.lr_min:g}, "
          f"{config.epochs} epochs)")
    return optimizer, None


def epoch_lr(base_lr: float, epoch: int, total_epochs: int, lr_min: float) -> float:
    """Cosine learning rate of epoch `epoch` (1-indexed), as in rfl/utils/schedules.py.

    lr = lr_min + (base_lr - lr_min) (1 + cos(pi t)) / 2 with
    t = (epoch - 1) / (total_epochs - 1), so the first epoch runs at base_lr and the
    last at lr_min.
    """
    return compute_round_lr(base_lr, epoch, total_epochs, 'cosine', lr_min)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    apply_constraints: bool = False,
    desc: str = "Training",
) -> Tuple[float, float]:
    """Train for one epoch.

    After each optimiser step, applies `model.apply_constraints()` when
    `apply_constraints` is set (EEGNet max-norm) and `model.project_stiefel()` when the
    model has it (SPDNet).

    Args:
        model: Model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimiser.
        device: Device.
        apply_constraints: Apply the model's max-norm constraints after each step.
        desc: Progress-bar label.

    Returns:
        (mean training loss, training accuracy) over the epoch, computed on the fly
        during training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if apply_constraints and hasattr(model, 'apply_constraints'):
            model.apply_constraints()
        if hasattr(model, 'project_stiefel'):
            model.project_stiefel()

        _, predicted = outputs.max(1)
        batch_size = labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total += batch_size
        running_loss += loss.item() * batch_size

        try:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        except BlockingIOError:
            pass  # stdout buffer full in some batch environments

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Any,  # ExperimentConfig
    device: torch.device,
    checkpoint_manager: Any,  # CheckpointManager
    evaluator: Any,  # Evaluator
    is_spdnet: bool = False,
    test_loader: Any = None,
) -> Dict[str, Any]:
    """Train with the protocol described in the module docstring.

    Resumes from the latest checkpoint of `checkpoint_manager` if one exists. Every
    epoch is checkpointed, and the epoch with the best selection metric is saved as
    the best checkpoint. Ties go to the earliest epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader (drives schedule, stopping and selection).
        criterion: Loss function.
        config: ExperimentConfig.
        device: Device.
        checkpoint_manager: training.checkpointing.CheckpointManager.
        evaluator: training.evaluation.Evaluator.
        is_spdnet: Whether the model is an SPDNet (no max-norm constraints).
        test_loader: Optional test loader. Its per-epoch metrics are recorded in the
            history for plotting only and never used for any decision.

    Returns:
        Dict with keys 'history' (per-epoch lists), 'selection_metric', 'best_epoch'
        and 'best_val_loss', 'best_val_acc', 'best_val_f1' (all at the selected epoch),
        'final_val_loss', 'final_val_acc' (last epoch), 'lr_schedule', 'report_rule',
        'final_epoch' and 'stopped_early'.

    Raises:
        ValueError: On an unknown selection metric, or report_rule='last' with early
            stopping enabled (the last epoch would then depend on validation data).
    """
    model = model.to(device)
    if is_spdnet:
        fix_batch_norm_dtype(model, device)
        logger.info(f"SPDNet BatchNorm buffers fixed to {device}")

    optimizer, scheduler = setup_training(
        model, config, device, is_spdnet
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_macro': [],
        'test_loss': [],
        'test_acc': [],
        'learning_rate': [],
        'epoch': []
    }

    selection_metric = str(config.selection_metric).lower()
    if selection_metric not in ('val_loss', 'val_acc'):
        raise ValueError("selection_metric must be 'val_loss' or 'val_acc', got "
                         f"{selection_metric!r}")
    lower_is_better = selection_metric == 'val_loss'

    # best_val_* are the metrics at the selected epoch, not separate maxima.
    best_val_loss = float('inf')
    best_val_acc = -float('inf')
    best_val_f1 = -float('inf')
    best_score = float('inf') if lower_is_better else -float('inf')
    best_epoch = None
    patience_counter = 0
    early_stopping_patience = int(config.early_stopping_patience)
    schedule = str(config.lr_schedule).lower()
    report_rule = str(config.report_rule).lower()
    if report_rule == 'last' and early_stopping_patience > 0:
        raise ValueError("report_rule='last' needs early_stopping_patience=0: with "
                         "early stopping, the last epoch is chosen by the validation "
                         "data.")

    start_epoch = checkpoint_manager.load_latest(model, optimizer, device)
    if start_epoch > 1:
        logger.info(f"Resumed from epoch {start_epoch - 1}")
        saved_history = checkpoint_manager.load_history()
        if saved_history:
            history = saved_history
            series = history['val_loss'] if lower_is_better else history['val_acc']
            pos = best_position(series, lower_is_better)
            if pos is not None:
                best_val_loss = history['val_loss'][pos]
                best_val_acc = history['val_acc'][pos]
                best_val_f1 = history['val_f1_macro'][pos]
                best_epoch = history['epoch'][pos]
                best_score = best_val_loss if lower_is_better else best_val_acc
                patience_counter = len(series) - 1 - pos
            logger.info(f"Restored selected epoch {best_epoch} on {selection_metric} - "
                  f"Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")

    logger.info(f"\n{'='*70}")
    logger.info(f"Starting training for {config.epochs} epochs")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Selection metric: {selection_metric}")
    logger.info(f"Early stopping patience: {early_stopping_patience}"
          + ("  (0 = disabled, the full budget is run)"
             if early_stopping_patience == 0 else ""))
    logger.info(f"{'='*70}\n")

    # `epoch` stays defined even when a resumed run has nothing left to do. A run that
    # had already met its stopping rule before the restart does not train again.
    epoch = start_epoch - 1
    if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
        logger.info("The resumed run had already stopped early: nothing left to train.")
        start_epoch = config.epochs + 1
    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{config.epochs}")
        logger.info("-" * 50)
        lr_now = compute_round_lr(
            base_lr=config.learning_rate, round_num=epoch, total_rounds=config.epochs,
            schedule=schedule, lr_min=config.lr_min,
            val_history=list(history['val_loss'] if lower_is_better
                             else history['val_acc']),
            plateau_patience=int(config.lr_plateau_patience),
            plateau_factor=float(config.lr_plateau_factor),
            mode='min' if lower_is_better else 'max')
        for group in optimizer.param_groups:
            group['lr'] = lr_now

        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            apply_constraints=(not is_spdnet),
            desc=f"Epoch {epoch} [Train]",
        )

        from rfl.training.divergence import check_finite
        check_finite(train_loss, f"epoch {epoch}")

        val_loss, val_acc, val_metrics = evaluator.evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Epoch {epoch} [Val]"
        )

        # Test curve for plotting only. No decision reads it.
        if test_loader is not None:
            test_loss, test_acc, _ = evaluator.evaluate(
                model=model, dataloader=test_loader, criterion=criterion,
                device=device, desc=f"Epoch {epoch} [Test]")
        else:
            test_loss, test_acc = float('nan'), float('nan')

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history.setdefault('test_loss', []).append(test_loss)
        history.setdefault('test_acc', []).append(test_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")
        logger.info(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        val_f1 = val_metrics['f1_macro']
        score = val_loss if lower_is_better else val_acc
        is_best = (score < best_score) if lower_is_better else (score > best_score)
        if is_best:
            best_score = score
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            logger.info(f"New best on {selection_metric}: Val Loss {val_loss:.4f}, "
                  f"Val Acc {val_acc:.4f}, F1 {val_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement on {selection_metric} for {patience_counter}"
                  f"/{early_stopping_patience} epochs")

        # Save before the stopping test, so the last epoch is in the history and can
        # still be the best checkpoint.
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={'val_acc': val_acc, 'val_loss': val_loss},
            history=history,
            is_best=is_best
        )

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            logger.info(f"\nEarly stopping after {epoch} epochs "
                  f"({patience_counter} without improvement on {selection_metric})")
            logger.info(f"Selected epoch: {best_epoch}")
            logger.info(f"Validation loss at that epoch: {best_val_loss:.4f}")
            logger.info(f"Validation accuracy at that epoch: {best_val_acc:.4f}")
            break

    logger.info(f"\n{'='*70}")
    logger.info("Training completed!")
    logger.info(f"Selected epoch ({selection_metric}): {best_epoch} of {epoch} run")
    logger.info(f"Validation loss at that epoch:     {best_val_loss:.4f}")
    logger.info(f"Validation accuracy at that epoch: {best_val_acc:.4f}")
    if early_stopping_patience > 0 and epoch >= config.epochs:
        logger.info("NOTE: the epoch budget was exhausted without the stopping rule firing. "
              "This run has not plateaued; raise --epochs before reporting it.")
    logger.info(f"{'='*70}\n")

    # Recompute the selected epoch from the full history, so that a resumed run reports
    # what an uninterrupted one would. min/max return the earliest epoch on ties.
    series = history['val_loss'] if lower_is_better else history['val_acc']
    best_pos = best_position(series, lower_is_better)
    if best_pos is not None:
        best_epoch = history['epoch'][best_pos]
        best_val_loss = history['val_loss'][best_pos]
        best_val_acc = history['val_acc'][best_pos]
        best_val_f1 = history['val_f1_macro'][best_pos]

    return {
        'history': history,
        'selection_metric': selection_metric,
        # Metrics at the selected epoch.
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        # Metrics of the last epoch (report_rule='last', and the sweep selection score).
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('nan'),
        'final_val_acc': history['val_acc'][-1] if history['val_acc'] else float('nan'),
        'lr_schedule': schedule,
        'report_rule': report_rule,
        'final_epoch': epoch,
        'stopped_early': bool(early_stopping_patience > 0
                              and patience_counter >= early_stopping_patience),
    }


