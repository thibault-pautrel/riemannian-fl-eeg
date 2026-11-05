"""Manifold primitives: Stiefel (PyTorch) for the EEG models, Grassmann (NumPy) for the
synthetic experiments.

The Stiefel functions are imported on first use, so that the Grassmann code and the
synthetic experiments run without PyTorch.
"""
__all__ = ["polar_projection", "tangent_projection", "lifting", "retraction",
           "inverse_polar_retraction"]


def __getattr__(name):
    if name in __all__:
        from rfl.geometry import stiefel
        return getattr(stiefel, name)
    raise AttributeError(f"module 'geometry' has no attribute {name!r}")
