"""Tests of the per-sample privatisation mechanism (rfl/federated/privacy/dp_grad.py).

    g~ = (1/B) sum_l clip_C(P_x grad f(x; z_l)) + P_x xi,   xi ~ N(0, sigma^2 I)

Small hand-written models cover the Euclidean path (dropout, vmap), the refusal of
batch normalisation and the Stiefel path (a BiMap-like layer). The real SPDNet, which needs yetanotherspdnet, is
tested at the end.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from rfl.federated.privacy.dp_grad import (
    _forward_mode,
    _per_sample_grads_vmap,
    _per_sample_tangent_grads,
    _per_sample_tangent_grads_loop,
    assert_dp_safe_norm_layers,
    dp_local_step,
    federated_param_groups,
    per_sample_fed_norms,
    privatized_fed_grads,
    tangent_noise,
)

CRIT = nn.CrossEntropyLoss()
CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
#  Small models
# --------------------------------------------------------------------------- #
class TinyEEGNet(nn.Module):
    """Euclidean model with dropout and no normalisation (takes the vmap path)."""

    def __init__(self, p=0.5, batch_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, (1, 5), bias=False)
        self.bn = nn.BatchNorm2d(4) if batch_norm else nn.Identity()
        self.act = nn.ELU()
        self.drop = nn.Dropout(p)
        self.classifier = nn.Linear(4 * 8 * 4, 2)

    def forward(self, x):
        return self.classifier(self.drop(self.act(self.bn(self.conv(x)))).flatten(1))


class BiMap(nn.Module):
    """Stand-in for the SPDNet BiMap layer: X -> W^T X W, W on the Stiefel manifold."""

    def __init__(self, n, m):
        super().__init__()
        self.W = nn.Parameter(torch.linalg.qr(torch.randn(n, m))[0])

    def forward(self, X):
        return self.W.transpose(-1, -2) @ X @ self.W


class TinySPDNet(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.bimap = BiMap(6, 4)
        self.drop = nn.Dropout(p)
        self.classifier = nn.Linear(16, 2)

    def forward(self, X):
        H = torch.linalg.cholesky(self.bimap(X) + 1e-3 * torch.eye(4))
        return self.classifier(self.drop(H.flatten(1)))


def eeg_batch(B=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 1, 8, 8, generator=g), torch.randint(0, 2, (B,), generator=g)


def spd_batch(B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(B, 6, 6, generator=g)
    return A @ A.transpose(-1, -2) + 6 * torch.eye(6), torch.randint(0, 2, (B,), generator=g)


@pytest.fixture
def eegnet():
    torch.manual_seed(0)
    return TinyEEGNet()


def _max_diff(rows_a, rows_b):
    return max(float((a - b).abs().max()) for ra, rb in zip(rows_a, rows_b)
               for a, b in zip(ra, rb))


# --------------------------------------------------------------------------- #
#  Scope
# --------------------------------------------------------------------------- #
def test_scope_covers_every_parameter(eegnet):
    full = {n for n, _, _ in federated_param_groups(eegnet)}
    assert full == {"conv.weight", "classifier.weight", "classifier.bias"}


def test_stiefel_flag():
    groups = {n: st for n, _, st in federated_param_groups(TinySPDNet())}
    assert groups == {"bimap.W": True, "classifier.weight": False, "classifier.bias": False}


# --------------------------------------------------------------------------- #
#  Batch normalisation is refused under DP
# --------------------------------------------------------------------------- #
def test_dp_safe_norm_layers_guard():
    assert_dp_safe_norm_layers(TinyEEGNet())             # no normalisation: accepted
    with pytest.raises(ValueError):
        assert_dp_safe_norm_layers(TinyEEGNet(batch_norm=True))   # torch BatchNorm

    class BatchNormSPDMean(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_mean", torch.eye(4))

        def forward(self, X):
            return X

    model = TinySPDNet()
    model.bn_spd = BatchNormSPDMean()
    with pytest.raises(ValueError):
        assert_dp_safe_norm_layers(model)


# --------------------------------------------------------------------------- #
#  Per-sample gradients
# --------------------------------------------------------------------------- #
def test_dropout_is_live_only_when_asked(eegnet):
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()

    def spread(dropout):
        torch.manual_seed(1)
        a = _per_sample_tangent_grads(eegnet, x, y, CRIT, groups,
                                      vectorized=True, dropout=dropout)
        torch.manual_seed(2)
        b = _per_sample_tangent_grads(eegnet, x, y, CRIT, groups,
                                      vectorized=True, dropout=dropout)
        return _max_diff(a, b)

    assert spread(True) > 1e-6
    assert spread(False) == 0.0


def test_vmap_path_is_taken_and_matches_loop(eegnet):
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()
    _per_sample_tangent_grads(eegnet, x, y, CRIT, groups, vectorized=True)
    assert eegnet._dp_vmap_ok is True
    with _forward_mode(eegnet, dropout=False):
        loop = _per_sample_tangent_grads_loop(eegnet, x, y, CRIT, groups)
        vec = _per_sample_grads_vmap(eegnet, x, y, CRIT, groups)
    assert _max_diff(loop, vec) < 1e-5


@pytest.mark.parametrize("vectorized", [True, False])
def test_dropout_masks_are_per_sample(eegnet, vectorized):
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()
    xr, yr = x[:1].repeat(8, 1, 1, 1), y[:1].repeat(8)     # 8 identical records
    torch.manual_seed(3)
    g = _per_sample_tangent_grads(eegnet, xr, yr, CRIT, groups,
                                  vectorized=vectorized, dropout=True)
    spread = max(float((g[0][j] - g[i][j]).abs().max())
                 for i in range(1, 8) for j in range(len(groups)))
    assert spread > 1e-6


def test_forward_mode_restores_flags(eegnet):
    eegnet.train()
    with _forward_mode(eegnet, dropout=True):
        assert not eegnet.training and eegnet.drop.training
    assert eegnet.training and eegnet.drop.training


def test_result_does_not_depend_on_prior_mode(eegnet):
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()
    eegnet.train()
    a = privatized_fed_grads(eegnet, x, y, CRIT, groups, 1.0, 0.0,
                             vectorized=True, dropout=False)
    eegnet.eval()
    b = privatized_fed_grads(eegnet, x, y, CRIT, groups, 1.0, 0.0,
                             vectorized=True, dropout=False)
    assert all(torch.equal(a[n], b[n]) for n, _, _ in groups)


def test_stiefel_per_sample_grads_are_tangent():
    model = TinySPDNet()
    groups = federated_param_groups(model)
    X, Y = spd_batch()
    W = model.bimap.W.detach()
    for row in _per_sample_tangent_grads(model, X, Y, CRIT, groups, dropout=False):
        G = row[0]
        assert float((W.T @ G + G.T @ W).abs().max()) < 1e-5


# --------------------------------------------------------------------------- #
#  Clipping, sensitivity and noise
# --------------------------------------------------------------------------- #
def test_clipped_average_has_norm_at_most_C(eegnet):
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()
    raw = per_sample_fed_norms(eegnet, x, y, CRIT, groups, dropout=False)
    C = 0.5 * float(np.median(raw))
    g = privatized_fed_grads(eegnet, x, y, CRIT, groups, C, 0.0, dropout=False)
    norm = math.sqrt(sum(float((g[n] ** 2).sum()) for n, _, _ in groups))
    assert norm <= C + 1e-6


def test_no_clip_no_noise_is_the_minibatch_gradient(eegnet):
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()
    g = privatized_fed_grads(eegnet, x, y, CRIT, groups, math.inf, 0.0,
                             dropout=False)
    with _forward_mode(eegnet, dropout=False):
        ref = torch.autograd.grad(CRIT(eegnet(x), y), [p for _, p, _ in groups])
    for (n, _, _), r in zip(groups, ref):
        assert torch.allclose(g[n], r, atol=1e-6), n


@pytest.mark.parametrize("C", [0.2, 1.0])
def test_substitution_sensitivity(eegnet, C):
    """Conditional on the dropout masks, ||g(D) - g(D')|| <= 2C/B for one substitution."""
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()
    xp, yp = x.clone(), y.clone()
    xp[3] = torch.randn_like(xp[3])
    yp[3] = 1 - yp[3]
    torch.manual_seed(7)
    a = privatized_fed_grads(eegnet, x, y, CRIT, groups, C, 0.0, vectorized=True)
    torch.manual_seed(7)
    b = privatized_fed_grads(eegnet, xp, yp, CRIT, groups, C, 0.0, vectorized=True)
    d = math.sqrt(sum(float(((a[n] - b[n]) ** 2).sum()) for n, _, _ in groups))
    assert d <= 2 * C / x.shape[0] + 1e-6


def test_noise_law_and_tangency():
    model = TinySPDNet()
    groups = federated_param_groups(model)
    X, Y = spd_batch()
    sigma = 0.05
    gen = torch.Generator().manual_seed(0)
    g0 = privatized_fed_grads(model, X, Y, CRIT, groups, math.inf, 0.0, dropout=False)
    res = []
    for _ in range(200):
        g1 = privatized_fed_grads(model, X, Y, CRIT, groups, math.inf, sigma,
                                  generator=gen, dropout=False)
        res.append(torch.cat([(g1[n] - g0[n]).flatten() for n, _, st in groups if not st]))
    R = torch.stack(res)
    assert abs(float(R.mean())) < 0.01
    assert abs(float(R.std()) - sigma) < 0.01
    W = model.bimap.W.detach()
    G = privatized_fed_grads(model, X, Y, CRIT, groups, 1.0, 0.1, generator=gen,
                             dropout=False)["bimap.W"]
    assert float((W.T @ G + G.T @ W).abs().max()) < 1e-5


def test_tangent_noise():
    W = torch.linalg.qr(torch.randn(10, 4))[0]
    gen = torch.Generator().manual_seed(0)
    xi = tangent_noise(W, 1.0, True, gen)
    assert float((W.T @ xi + xi.T @ W).abs().max()) < 1e-5
    eu = tangent_noise(W, 1.0, False, torch.Generator().manual_seed(0))
    assert not torch.allclose(eu, xi)                 # identity projection keeps the normal part


def test_noise_is_reproducible_from_the_generator(eegnet):
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()
    a = privatized_fed_grads(eegnet, x, y, CRIT, groups, 1.0, 0.5,
                             generator=torch.Generator().manual_seed(5), dropout=False)
    b = privatized_fed_grads(eegnet, x, y, CRIT, groups, 1.0, 0.5,
                             generator=torch.Generator().manual_seed(5), dropout=False)
    assert all(torch.equal(a[n], b[n]) for n, _, _ in groups)


# --------------------------------------------------------------------------- #
#  Real SPDNet (custom autograd from yetanotherspdnet)
# --------------------------------------------------------------------------- #
def _spdnet(d=12, hidden=6, k=4):
    pytest.importorskip("yetanotherspdnet")
    from rfl.models.spdnet import create_spdnet
    return create_spdnet(n_classes=k, n_channels=d, hidden_dim=hidden, seed=0)


def _spd_data(n=8, d=12, k=4, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, d, d).astype("float32")
    X = np.einsum("nij,nkj->nik", A, A) + 0.1 * np.eye(d, dtype="float32")[None]
    return torch.from_numpy(X), torch.from_numpy(rng.randint(0, k, n).astype("int64"))


def test_spdnet_scope():
    m = _spdnet()
    full = {n: st for n, _, st in federated_param_groups(m)}
    assert full == {"bimap.weight": True, "classifier.weight": False,
                    "classifier.bias": False}


def test_spdnet_per_sample_grads_through_custom_ops():
    m = _spdnet()
    X, y = _spd_data()
    norms = per_sample_fed_norms(m, X, y, CRIT, federated_param_groups(m))
    assert len(norms) == 8 and all(math.isfinite(v) and v > 0 for v in norms)


def test_spdnet_privatized_bimap_grad_is_tangent():
    m = _spdnet()
    X, y = _spd_data()
    g = privatized_fed_grads(m, X, y, CRIT, federated_param_groups(m), C=1.0, sigma=0.5,
                             generator=torch.Generator().manual_seed(1))["bimap.weight"]
    W = m.bimap.weight.detach()
    assert torch.isfinite(g).all()
    assert torch.norm(W.T @ g + g.T @ W) < 1e-4


def test_spdnet_dp_step_stays_on_stiefel():
    m = _spdnet()
    X, y = _spd_data()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    loss, logits = dp_local_step(m, opt, X, y, CRIT, federated_param_groups(m),
                                 C=1.0, sigma=0.5,
                                 generator=torch.Generator().manual_seed(2))
    W = m.bimap.weight.detach()
    assert math.isfinite(loss) and logits.shape == (8, 4)
    assert torch.norm(W.T @ W - torch.eye(W.shape[1])) < 1e-4


# ---- soundness off the manifold and with non-finite records ---------------------
def _off_manifold_W(n=8, p=3, spread=0.3, seed=0):
    """A mean of Stiefel matrices, off the manifold: a stress test of the tangent chart."""
    from rfl.geometry.stiefel import polar_projection
    g = torch.Generator().manual_seed(seed)
    base = polar_projection(torch.randn(n, p, generator=g, dtype=torch.float64))
    Ws = [polar_projection(base + spread * torch.randn(n, p, generator=g,
                                                       dtype=torch.float64))
          for _ in range(5)]
    return (sum(Ws) / 5).float()


def test_noise_is_isotropic_in_the_tangent_space_even_off_the_manifold():
    from rfl.geometry.stiefel import StiefelChart
    W = _off_manifold_W()
    assert float(torch.linalg.svdvals(W.double()).min()) < 0.95          # off-manifold
    ch = StiefelChart(W)
    gen = torch.Generator().manual_seed(0)
    C = torch.stack([ch.coords(tangent_noise(W, 2.0, True, gen)) for _ in range(4000)])
    cov = C.T @ C / C.shape[0]
    ev = torch.linalg.eigvalsh(cov)
    assert 3.4 < float(ev.min()) and float(ev.max()) < 4.6                # sigma^2 = 4


def test_clip_bounds_the_contribution_of_any_record_off_the_manifold():
    """One record moves g~ by at most 2C/B in the tangent norm, however large it is."""
    pytest.importorskip("yetanotherspdnet")
    from rfl.geometry.stiefel import StiefelChart
    from rfl.models.spdnet import create_spdnet
    torch.manual_seed(0)
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dim=3, seed=0)
    with torch.no_grad():
        m.bimap.weight.copy_(_off_manifold_W(8, 3))
    groups = federated_param_groups(m)
    A = torch.randn(4, 8, 24)
    X = A @ A.transpose(-1, -2) / 24 + 1e-3 * torch.eye(8)
    y = torch.tensor([0, 1, 0, 1])
    X2 = X.clone()
    X2[0] = X[0] * 1e6                                     # an extreme record
    C, B = 0.5, 4
    g1 = privatized_fed_grads(m, X, y, CRIT, groups, C, 0.0, dropout=False)
    g2 = privatized_fed_grads(m, X2, y, CRIT, groups, C, 0.0, dropout=False)
    ch = StiefelChart(m.bimap.weight)
    d = 0.0
    for name, p, is_st in groups:
        diff = g1[name].double() - g2[name].double()
        d += float((ch.coords(diff) ** 2).sum()) if is_st else float((diff ** 2).sum())
    assert math.sqrt(d) <= 2 * C / B * (1 + 1e-5)


def test_non_finite_record_contributes_zero(eegnet):
    """An infinite input gives an infinite per-sample gradient: it must not reach g~."""
    groups = federated_param_groups(eegnet)
    x, y = eeg_batch()
    x = x.clone()
    x[0].view(-1)[0] = float("inf")
    g = privatized_fed_grads(eegnet, x, y, CRIT, groups, 1.0, 0.5,
                             torch.Generator().manual_seed(0), dropout=False)
    assert all(torch.isfinite(v).all() for v in g.values())
    # without that record, the clipped sum is the same
    g0 = privatized_fed_grads(eegnet, x[1:], y[1:], CRIT, groups, 1.0, 0.0, dropout=False)
    g1 = privatized_fed_grads(eegnet, x, y, CRIT, groups, 1.0, 0.0, dropout=False)
    B, B0 = x.shape[0], x.shape[0] - 1
    for k in g0:
        assert torch.allclose(g1[k] * B, g0[k] * B0, atol=1e-6)


def test_a_record_that_breaks_spdnet_contributes_zero_instead_of_crashing():
    """A non-finite matrix makes eigh fail in ReEig: the step still runs, and the
    record contributes nothing (the run cannot reveal its presence by failing)."""
    pytest.importorskip("yetanotherspdnet")
    from rfl.federated.privacy.dp_grad import dp_local_step
    from rfl.models.spdnet import create_spdnet
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dim=3, seed=0)
    groups = federated_param_groups(m)
    A = torch.randn(4, 8, 24, generator=torch.Generator().manual_seed(0))
    X = A @ A.transpose(-1, -2) / 24 + 1e-3 * torch.eye(8)
    y = torch.tensor([0, 1, 0, 1])
    bad = X.clone()
    bad[0] = float("nan")
    g_bad = privatized_fed_grads(m, bad, y, CRIT, groups, 1.0, 0.0, dropout=False)
    g_ref = privatized_fed_grads(m, X[1:], y[1:], CRIT, groups, 1.0, 0.0, dropout=False)
    for k in g_ref:
        assert torch.allclose(g_bad[k] * 4, g_ref[k] * 3, atol=1e-6)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    dp_local_step(m, opt, bad, y, CRIT, groups, 1.0, 0.5, torch.Generator().manual_seed(1))
    assert all(torch.isfinite(p).all() for p in m.parameters())
