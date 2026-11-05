"""Evaluation of a classifier on a data loader: loss, accuracy, macro and per-class metrics."""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """Computes classification metrics and optionally saves them as JSON.

    Args:
        num_classes: Number of classes (rows of the confusion matrix).
        class_names: Class names, used for printing.
        save_dir: If given, `evaluate_and_save` writes to `save_dir/metrics/`.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        save_dir: Optional[str] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.save_dir = save_dir

        if save_dir:
            os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        desc: str = "Evaluating",
        return_predictions: bool = False
    ):
        """Evaluate `model` in eval mode on every batch of `dataloader`.

        Args:
            model: Model to evaluate.
            dataloader: Data loader.
            criterion: Loss function (mean reduction).
            device: Device.
            desc: Progress-bar label.
            return_predictions: Also return predictions and labels.

        Returns:
            (mean loss, accuracy, metrics) or, with `return_predictions`,
            (mean loss, accuracy, metrics, predictions, labels). `metrics` is the dict
            returned by `_compute_metrics` with macro averaging.
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        pbar = tqdm(dataloader, desc=desc, leave=False, disable=None)   # off when not a terminal (job logs)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            batch_size = labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total += batch_size
            running_loss += loss.item() * batch_size

            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        avg_loss = running_loss / total
        accuracy = correct / total

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        metrics = self._compute_metrics(all_predictions, all_labels)

        if return_predictions:
            return avg_loss, accuracy, metrics, all_predictions, all_labels
        return avg_loss, accuracy, metrics

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """Accuracy, averaged precision/recall/F1 and per-class metrics.

        Returns:
            Dict with 'accuracy', 'precision_<average>', 'recall_<average>',
            'f1_<average>' and the lists 'per_class_precision', 'per_class_recall',
            'per_class_f1', 'per_class_support' (over the classes present in the data).
        """
        preds_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()

        accuracy = (preds_np == labels_np).mean()

        precision, recall, f1, support = precision_recall_fscore_support(
            labels_np, preds_np, average=average, zero_division=0
        )
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(labels_np, preds_np, average=None, zero_division=0)

        return {
            'accuracy': float(accuracy),
            f'precision_{average}': float(precision),
            f'recall_{average}': float(recall),
            f'f1_{average}': float(f1),
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'per_class_support': per_class_support.tolist(),
        }

    def compute_confusion_matrix(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        """Confusion matrix of shape (num_classes, num_classes), rows = true labels."""
        return confusion_matrix(
            labels.cpu().numpy(),
            predictions.cpu().numpy(),
            labels=list(range(self.num_classes))
        )

    def evaluate_and_save(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        split_name: str,
        run_name: str
    ) -> Dict[str, float]:
        """Evaluate, print a summary and save the metrics.

        If `save_dir` was given, writes `save_dir/metrics/<run_name>_<split_name>_results.json`.

        Args:
            model: Model to evaluate.
            dataloader: Data loader.
            criterion: Loss function.
            device: Device.
            split_name: Split label, for example 'test'.
            run_name: Run label used in the file name.

        Returns:
            The metrics of `evaluate` plus 'loss', 'accuracy' and 'confusion_matrix'.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating on {split_name} set")
        logger.info(f"{'='*70}")

        loss, acc, metrics, preds, labels = self.evaluate(
            model, dataloader, criterion, device,
            desc=f"Evaluating {split_name}",
            return_predictions=True
        )

        logger.info(f"Loss: {loss:.4f}")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")

        logger.info("\nPer-class F1 scores:")
        for name, f1 in zip(self.class_names, metrics['per_class_f1']):
            logger.info(f"  {name}: {f1:.4f}")

        cm = self.compute_confusion_matrix(preds, labels)

        results = {
            'loss': loss,
            'accuracy': acc,
            **metrics,
            'confusion_matrix': cm.tolist()
        }

        if self.save_dir:
            results_path = os.path.join(
                self.save_dir, 'metrics', f'{run_name}_{split_name}_results.json'
            )
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to {results_path}")

        return results
