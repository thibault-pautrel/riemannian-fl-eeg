"""Checkpoint saving, loading and resuming for the centralised training loop."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Saves and loads training checkpoints for one run.

    Files are written to `checkpoint_dir` as `<run_name>_best.pt` (the selected
    epoch) and `<run_name>_epoch_<k>.pt` (periodic checkpoints). Each file holds the
    epoch, model and optimiser state dicts, metrics, history and config.

    Args:
        checkpoint_dir: Output directory (created if needed).
        run_name: Prefix of the file names.
        config: ExperimentConfig, stored in each checkpoint.
        keep_last_n: Number of periodic checkpoints kept. None keeps all.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        run_name: str,
        config: Any,
        keep_last_n: Optional[int] = 3
    ):
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.config = config
        self.keep_last_n = keep_last_n

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        history: Dict[str, List[float]],
        is_best: bool = False
    ) -> str:
        """Save a checkpoint.

        A periodic checkpoint is written at every epoch (the oldest beyond
        `keep_last_n` are deleted), and the best file is overwritten when `is_best`.
        Each checkpoint stores the configuration hash, so that `load_latest` never
        resumes a run of another configuration.

        Args:
            model: Model whose state is saved.
            optimizer: Optimiser whose state is saved.
            epoch: Current epoch.
            metrics: Metrics of this epoch.
            history: Training history so far.
            is_best: Whether this epoch is the best so far.

        Returns:
            Path of the written file.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'history': history,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
            'config_hash': self._config_hash(),
            # Random states, so that a resumed run continues the same random stream
            # (shuffle order, dropout masks) as an uninterrupted one.
            'rng': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
        }

        save_path = os.path.join(self.checkpoint_dir, f'{self.run_name}_epoch_{epoch}.pt')
        torch.save(checkpoint, save_path)
        if self.keep_last_n is not None:
            self._cleanup_old_checkpoints()
        if is_best:
            save_path = os.path.join(self.checkpoint_dir, f'{self.run_name}_best.pt')
            torch.save(checkpoint, save_path)
        return save_path

    def _config_hash(self):
        """Hash of the configuration, without the output location (None if unavailable)."""
        return self.config.get_hash() if hasattr(self.config, 'get_hash') else None

    def _compatible(self, checkpoint: Dict[str, Any]) -> bool:
        """True if `checkpoint` was written by a run of the same configuration."""
        mine = self._config_hash()
        return mine is not None and checkpoint.get('config_hash') == mine

    def _set_aside(self, reason: str) -> None:
        """Move this run's checkpoints to stale/, so that no later step reads them."""
        stale = Path(self.checkpoint_dir) / 'stale'
        stale.mkdir(exist_ok=True)
        for f in Path(self.checkpoint_dir).glob(f'{self.run_name}_*.pt'):
            f.replace(stale / f.name)
        logger.info(f"[checkpoint] existing checkpoints moved to {stale}: {reason}")

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load the best checkpoint into `model` (and `optimizer` if given).

        Returns:
            The checkpoint dict.

        Raises:
            FileNotFoundError: If no best checkpoint exists.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.run_name}_best.pt')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> int:
        """Load the latest periodic checkpoint, to resume training.

        Returns:
            The epoch to start from: loaded epoch + 1, or 1 if there is no checkpoint.
        """
        checkpoint_files = list(Path(self.checkpoint_dir).glob(f'{self.run_name}_epoch_*.pt'))

        if not checkpoint_files:
            logger.info("No existing checkpoints found. Starting from scratch.")
            return 1

        latest_checkpoint = max(
            checkpoint_files,
            key=lambda p: int(p.stem.split('_')[-1])
        )

        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        if not self._compatible(checkpoint):
            self._set_aside("they belong to a run of another configuration")
            logger.info("Starting from scratch.")
            return 1
        model.load_state_dict(checkpoint['model_state_dict'])
        rng = checkpoint.get('rng')
        if rng:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if rng.get('cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from {latest_checkpoint.name} (epoch {epoch})")
        return epoch + 1

    def load_history(self) -> Optional[Dict[str, List[float]]]:
        """History stored in the latest periodic checkpoint, or None if there is none."""
        checkpoint_files = list(Path(self.checkpoint_dir).glob(f'{self.run_name}_epoch_*.pt'))

        if not checkpoint_files:
            return None

        latest_checkpoint = max(
            checkpoint_files,
            key=lambda p: int(p.stem.split('_')[-1])
        )
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        if not self._compatible(checkpoint):
            return None
        return checkpoint.get('history', None)

    def _cleanup_old_checkpoints(self):
        """Delete periodic checkpoints beyond the `keep_last_n` most recent ones."""
        checkpoint_files = sorted(
            Path(self.checkpoint_dir).glob(f'{self.run_name}_epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        if len(checkpoint_files) > self.keep_last_n:
            for checkpoint_file in checkpoint_files[:-self.keep_last_n]:
                checkpoint_file.unlink()
