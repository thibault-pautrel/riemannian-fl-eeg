"""Seeding, deterministic PyTorch settings and environment logging."""

import json
import logging
import os
import platform
import random
import subprocess
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True, warn: bool = True):
    """Seed Python, NumPy and PyTorch (CPU and CUDA), and set cuDNN mode.

    With `deterministic`, also enables cuDNN determinism,
    `torch.use_deterministic_algorithms(True)` and sets CUBLAS_WORKSPACE_CONFIG.
    Without it, enables cuDNN benchmarking.

    Args:
        seed: Seed value.
        deterministic: Request deterministic PyTorch kernels (slower).
        warn: Print the chosen mode, and warn if deterministic algorithms cannot be
            enabled.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError as e:
                if warn:
                    warnings.warn(
                        f"Could not enable deterministic algorithms: {e}\n"
                        "Some operations may still be non-deterministic."
                    )
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        if warn:
            logger.info(f"Deterministic mode enabled (seed={seed})")
            logger.info("Note: This may reduce performance but ensures reproducibility")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if warn:
            logger.info(f"Seed set to {seed}")
            logger.info("Note: cuDNN benchmark enabled for better performance")


def get_environment_info() -> Dict[str, Any]:
    """System, Python and library versions (torch, numpy, and moabb, mne, sklearn,
    geoopt when installed), plus CUDA devices when available."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_implementation': platform.python_implementation(),
        },
        'python': {
            'version': sys.version,
            'executable': sys.executable,
        },
        'libraries': {
            'numpy': np.__version__,
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
    }

    if torch.cuda.is_available():
        info['cuda'] = {
            'version': torch.version.cuda,
            'device_count': torch.cuda.device_count(),
            'devices': [
                {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i),
                    'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
                }
                for i in range(torch.cuda.device_count())
            ],
            'current_device': torch.cuda.current_device(),
        }

    try:
        import sklearn
        info['libraries']['sklearn'] = sklearn.__version__
    except ImportError:
        pass

    try:
        import moabb
        info['libraries']['moabb'] = moabb.__version__
    except ImportError:
        pass

    try:
        import mne
        info['libraries']['mne'] = mne.__version__
    except ImportError:
        pass

    return info


def get_git_info() -> Dict[str, Optional[str]]:
    """Commit hash, branch, dirty flag and remote URL of the current Git checkout.

    Values are None when Git or the repository is unavailable.
    """
    git_info = {
        'commit_hash': None,
        'branch': None,
        'has_uncommitted_changes': None,
        'remote_url': None,
    }

    try:
        git_info['commit_hash'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        git_info['branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_info['has_uncommitted_changes'] = len(status) > 0

        try:
            git_info['remote_url'] = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            pass

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not installed, or not a Git checkout.
        pass

    return git_info


def log_environment(save_dir: str, filename: str = 'environment.json') -> Dict[str, Any]:
    """Write environment and Git information to `save_dir/filename` as JSON.

    Warns if the working tree has uncommitted changes.

    Returns:
        Dict with keys 'environment' and 'git'.
    """
    os.makedirs(save_dir, exist_ok=True)

    env_info = {
        'environment': get_environment_info(),
        'git': get_git_info(),
    }

    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(env_info, f, indent=2)

    logger.info(f"Environment information logged to: {save_path}")

    # Warn about uncommitted changes
    if env_info['git']['has_uncommitted_changes']:
        warnings.warn(
            "Git repository has uncommitted changes. "
            "Consider committing changes for full reproducibility."
        )

    return env_info


