"""Shared test fixtures."""
import numpy as np
import pytest
import torch


def pytest_sessionstart(session):
    """Stop at once, with the fix, when torch cannot exchange arrays with NumPy."""
    try:
        torch.from_numpy(np.zeros(1))
    except RuntimeError as err:
        pytest.exit(
            f"torch {torch.__version__} cannot use NumPy {np.__version__} ({err}). "
            "PyTorch builds before 2.3, such as 2.2.2 on Intel Macs, need numpy<2: "
            "pip install 'numpy<2'",
            returncode=1)


@pytest.fixture(autouse=True)
def restore_torch_determinism():
    """rfl.utils.reproducibility.set_seed switches deterministic algorithms on globally."""
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)
