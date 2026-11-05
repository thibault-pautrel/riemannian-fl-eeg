"""riemannian-fl-eeg: federated SPDNet and EEGNet, with optional personalisation and
record-level differential privacy.

Subpackages: data (MOABB loading, registry, synthetic data), models (SPDNet, EEGNet),
geometry (Stiefel, Grassmann), federated (client, server, aggregation, privacy),
training (centralised loop), analysis (result tables and tests), experiments (entry
points and plans, run with `python -m rfl.experiments.<name>`), utils, synthetic_pca.
"""

__version__ = "1.0.0"
