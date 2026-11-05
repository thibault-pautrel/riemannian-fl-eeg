"""Gradients of SPDNet against finite differences, in float64.

The custom backward passes of the SPD layers (BiMap, ReEig, LogEig, batch norm) are
checked end to end: the directional derivative of the loss along random directions of
all parameters must match a central finite difference.

The `Vech` of yetanotherspdnet (commit 5c08ebc) copies the gradient of each
off-diagonal entry onto both triangles. The model uses the library layer as it is.
The end-to-end check below replaces it by an autograd `Vech` to test the other
layers, and `test_library_vech_backward` records the library behaviour (xfail).
The backward of the library whitening (batch norm, `whitening` mode) prints its two
gradients at every step: `test_library_whitening_backward_is_silent` records it (xfail).
"""
import pytest
import torch
import torch.nn as nn

pytest.importorskip("yetanotherspdnet")

from rfl.models import spdnet as S                                     # noqa: E402


class _AutogradVech(nn.Module):
    """Same forward as the library Vech (lower triangle, row by row), autograd backward."""

    def forward(self, x):
        n = x.shape[-1]
        i, j = torch.tril_indices(n, n, device=x.device)
        return x[..., i, j]


def _spd(n=10, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, d, 3 * d, generator=g, dtype=torch.float64)
    return A @ A.transpose(-1, -2) / (3 * d) + 1e-2 * torch.eye(d, dtype=torch.float64)


def _directional_check(model, x, y, n_dirs=3, h=1e-6):
    crit = nn.CrossEntropyLoss()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    # The SPD bias is parametrised by a symmetric matrix: perturb it symmetrically.
    sym = [("parametrizations" in n and p.dim() == 2) for n, p in named]
    model.zero_grad()
    crit(model(x), y).backward()
    grads = [p.grad.detach().clone() for p in params]
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    for _ in range(n_dirs):
        dirs = [torch.randn(p.shape, generator=g, dtype=p.dtype) for p in params]
        dirs = [0.5 * (d + d.T) if s else d for d, s in zip(dirs, sym)]
        analytic = sum(float((gr * d).sum()) for gr, d in zip(grads, dirs))
        with torch.no_grad():
            for p, d in zip(params, dirs):
                p.add_(h * d)
            lp = float(crit(model(x), y))
            for p, d in zip(params, dirs):
                p.sub_(2 * h * d)
            lm = float(crit(model(x), y))
            for p, d in zip(params, dirs):
                p.add_(h * d)
        numeric = (lp - lm) / (2 * h)
        worst = max(worst, abs(numeric - analytic) / max(1e-8, abs(numeric)))
    return worst


@pytest.mark.parametrize("kw", [
    dict(),
    dict(hidden_dims=[6, 4], epsilons=[1e-3, 1e-2]),
    dict(use_batch_norm=True),
    dict(use_batch_norm=True, bn_mean_type="affine_invariant"),
    dict(use_batch_norm=True, bn_mean_type="harmonic", bn_position="before_reeig"),
    dict(use_batch_norm=True, bn_type="mean_var_scalar", bn_mean_type="log_euclidean"),
    dict(use_batch_norm=True, bn_norm_and_bias_mode="parallel_transport",
         hidden_dims=[6, 4], epsilons=[0.05, 0.05]),
])
def test_spdnet_gradient_matches_finite_differences(kw, monkeypatch):
    monkeypatch.setattr(S.SPDNet, "_f32", staticmethod(lambda x: x))  # stay in float64
    monkeypatch.setattr(S, "Vech", _AutogradVech)
    opts = dict(n_classes=3, n_channels=8, seed=0)
    if "hidden_dims" not in kw:
        opts["hidden_dim"] = 5
    model = S.create_spdnet(**opts, **kw).double()
    with torch.no_grad():                    # move the BN biases away from the identity
        for name, p in model.named_parameters():
            if "Covbias" in name or "stdScalarbias" in name:
                p.add_(0.05 * torch.randn_like(p))
                if p.dim() == 2:
                    p.copy_(0.5 * (p + p.T))
    model.train()
    x = _spd()
    y = torch.arange(10) % 3
    assert _directional_check(model, x, y) < 1e-4


@pytest.mark.xfail(strict=True, reason="yetanotherspdnet Vech: off-diagonal gradient "
                   "copied onto both triangles (reported upstream, not patched here)")
def test_library_vech_backward():
    X = _spd(2, 4).requires_grad_(True)
    torch.autograd.gradcheck(lambda A: S.Vech()(A + A.transpose(-1, -2)), (X,))


@pytest.mark.xfail(strict=True, reason="yetanotherspdnet Whitening.backward prints its "
                   "gradients (debug print in functions/spd_linalg.py, not patched here)")
def test_library_whitening_backward_is_silent(capsys):
    from yetanotherspdnet.functions.spd_linalg import Whitening
    X = _spd(4, 3).requires_grad_(True)
    G = _spd(1, 3)[0].requires_grad_(True)
    Whitening.apply(X, G).sum().backward()
    assert capsys.readouterr().out == ""
