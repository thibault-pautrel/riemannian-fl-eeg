"""Tests for rfl/geometry/stiefel.py (pure geometry, no SPDNet dependency)."""
import pytest
import torch

from rfl.geometry.stiefel import (
    RetractionNotInvertible, inverse_polar_retraction, lifting, polar_projection,
    pseudo_exp, pseudo_log, retraction, tangent_projection,
)

DT = torch.float64


def _rand_stiefel(n, p, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, p, generator=g, dtype=DT))
    return Q


def _near(W, eps, seed=0):
    """A Stiefel point at distance about eps from W, along a random tangent direction."""
    g = torch.Generator().manual_seed(seed)
    xi = tangent_projection(W, torch.randn(W.shape, generator=g, dtype=DT))
    return retraction(W, eps * xi / xi.norm())


# ---- polar projection ---------------------------------------------------------
def test_polar_projection_is_orthonormal_and_idempotent():
    g = torch.Generator().manual_seed(0)
    U = polar_projection(torch.randn(22, 8, generator=g, dtype=DT))
    assert torch.norm(U.T @ U - torch.eye(8, dtype=DT)) < 1e-10
    assert torch.norm(polar_projection(U) - U) < 1e-10


def test_polar_projection_recovers_consensus_point():
    """ProjAvg of identical client points returns that point."""
    W = _rand_stiefel(22, 8, seed=1)
    avg = sum(W.clone() for _ in range(5)) / 5
    assert torch.norm(polar_projection(avg) - W) < 1e-10


def test_polar_projection_rejects_wide_matrix():
    with pytest.raises(ValueError):
        polar_projection(torch.randn(4, 10))


# ---- tangent projection -------------------------------------------------------
def test_tangent_projection_is_tangent_idempotent_and_contracting():
    W = _rand_stiefel(22, 8, seed=2)
    g = torch.Generator().manual_seed(3)
    X = torch.randn(22, 8, generator=g, dtype=DT)
    V = tangent_projection(W, X)
    assert torch.norm(W.T @ V + V.T @ W) < 1e-10
    assert torch.norm(tangent_projection(W, V) - V) < 1e-10
    assert torch.norm(V) <= torch.norm(X) + 1e-12


# ---- lifting and retraction (RLAvg) -------------------------------------------
def test_retraction_of_lifting_stays_on_manifold():
    W, W2 = _rand_stiefel(22, 8, seed=4), _rand_stiefel(22, 8, seed=5)
    R = retraction(W, lifting(W, W2))
    assert torch.norm(R.T @ R - torch.eye(8, dtype=DT)) < 1e-10


# ---- inverse polar retraction (RetAvg) ----------------------------------------
def test_inverse_at_base_is_zero():
    W = _rand_stiefel(22, 11, seed=1)
    assert torch.norm(inverse_polar_retraction(W, W)) < 1e-10


@pytest.mark.parametrize("eps", [1e-3, 1e-2, 5e-2, 1e-1, 2e-1])
def test_inverse_round_trip_is_exact(eps):
    W = _rand_stiefel(22, 11, seed=2)
    Y = _near(W, eps, seed=3)
    assert torch.norm(retraction(W, inverse_polar_retraction(W, Y)) - Y) < 1e-9


@pytest.mark.parametrize("eps", [1e-2, 1e-1, 2e-1])
def test_inverse_returns_tangent_vector(eps):
    W = _rand_stiefel(18, 7, seed=4)
    V = inverse_polar_retraction(W, _near(W, eps, seed=5))
    assert torch.norm(W.T @ V + V.T @ W) < 1e-8


def test_lifting_matches_inverse_to_first_order():
    """The gap between lifting and the exact inverse is small and grows with the step."""
    W = _rand_stiefel(22, 11, seed=6)
    rels = []
    for eps in (1e-3, 1e-2, 1e-1):
        Y = _near(W, eps, seed=7)
        v_inv = inverse_polar_retraction(W, Y)
        rels.append(float((v_inv - lifting(W, Y)).norm() / v_inv.norm()))
    assert rels[0] < 1e-4
    assert rels[0] < rels[1] < rels[2]


def test_inverse_rejects_far_target():
    W = _rand_stiefel(22, 11, seed=8)
    Y = _rand_stiefel(22, 11, seed=999)
    with pytest.raises(ValueError, match="real part"):
        inverse_polar_retraction(W, Y)


def test_inverse_accepts_targets_beyond_the_symmetric_part_condition():
    """sym(W^T Y) > 0 is sufficient, not necessary: the exact condition is used."""
    from rfl.geometry.stiefel import retraction
    g = torch.Generator().manual_seed(1)
    found = 0
    for _ in range(4000):
        W = _rand_stiefel(6, 3, seed=int(torch.randint(0, 10**6, (1,), generator=g)))
        Y = _rand_stiefel(6, 3, seed=int(torch.randint(0, 10**6, (1,), generator=g)))
        M = W.T @ Y
        if (float(torch.linalg.eigvalsh(0.5 * (M + M.T)).min()) < 0
                and float(torch.linalg.eigvals(M).real.min()) > 1e-3):
            V = inverse_polar_retraction(W, Y)
            assert torch.allclose(retraction(W, V), Y, atol=1e-8)
            found += 1
    assert found > 0


def test_inverse_rejects_bad_shapes():
    W = _rand_stiefel(10, 4, seed=10)
    with pytest.raises(ValueError):
        inverse_polar_retraction(W, _rand_stiefel(10, 5, seed=11))
    with pytest.raises(ValueError):
        inverse_polar_retraction(torch.randn(4, 10, dtype=DT), torch.randn(4, 10, dtype=DT))


# ---- pseudo-exponential and pseudo-logarithm (PseAvg) ---------------------------
def test_pseudo_exp_stays_on_manifold():
    W = _rand_stiefel(22, 11, seed=20)
    g = torch.Generator().manual_seed(2)
    Y = pseudo_exp(W, tangent_projection(W, torch.randn(W.shape, generator=g, dtype=DT)))
    assert torch.norm(Y.T @ Y - torch.eye(11, dtype=DT)) < 1e-10


def test_pseudo_log_at_base_is_zero():
    W = _rand_stiefel(22, 11, seed=3)
    assert torch.norm(pseudo_log(W, W)) < 1e-10


@pytest.mark.parametrize("eps", [1e-3, 1e-2, 1e-1, 2e-1, 1.0])
def test_pseudo_round_trip_is_exact(eps):
    W = _rand_stiefel(22, 11, seed=4)
    Y = _near(W, eps, seed=24)
    assert torch.norm(pseudo_exp(W, pseudo_log(W, Y)) - Y) < 1e-8


@pytest.mark.parametrize("eps", [1e-2, 1e-1, 2e-1])
def test_pseudo_log_returns_tangent_vector(eps):
    W = _rand_stiefel(18, 7, seed=5)
    V = pseudo_log(W, _near(W, eps, seed=6))
    assert torch.norm(W.T @ V + V.T @ W) < 1e-8


def test_pseudo_log_rejects_orthogonal_subspaces():
    W = torch.cat([torch.eye(3, dtype=DT), torch.zeros(3, 3, dtype=DT)], dim=0)
    Y = torch.cat([torch.zeros(3, 3, dtype=DT), torch.eye(3, dtype=DT)], dim=0)
    with pytest.raises(RetractionNotInvertible, match="Grassmannian"):
        pseudo_log(W, Y)


def test_pseudo_log_rejects_opposite_orientation():
    W = _rand_stiefel(22, 3, seed=7)
    with pytest.raises(RetractionNotInvertible, match="orientation"):
        pseudo_log(W, -W)


def test_pseudo_log_rejects_a_half_turn():
    W = _rand_stiefel(22, 3, seed=8)
    with pytest.raises(RetractionNotInvertible, match="angle"):
        pseudo_log(W, W * torch.tensor([-1., -1., 1.], dtype=DT))


def test_pseudo_log_rejects_bad_shapes():
    W = _rand_stiefel(10, 4, seed=9)
    with pytest.raises(ValueError):
        pseudo_log(W, _rand_stiefel(10, 5, seed=30))
    with pytest.raises(ValueError):
        pseudo_log(torch.randn(4, 10, dtype=DT), torch.randn(4, 10, dtype=DT))


def test_pseudo_exp_matches_retraction_to_first_order():
    """The gap between pseudo_exp and the polar retraction is o(||V||)."""
    W = _rand_stiefel(22, 11, seed=10)
    g = torch.Generator().manual_seed(11)
    xi = tangent_projection(W, torch.randn(W.shape, generator=g, dtype=DT))
    rels = [float((pseudo_exp(W, e * xi / xi.norm()) - retraction(W, e * xi / xi.norm()))
                  .norm() / e) for e in (1e-3, 1e-2, 1e-1)]
    assert rels[0] < 1e-2 and rels[0] < rels[1] < rels[2]


def test_pseudo_maps_keep_the_dtype_of_the_weights():
    W = _rand_stiefel(16, 8, seed=12).float()
    Y = _near(W.double(), 0.1, seed=13).float()
    assert pseudo_log(W, Y).dtype == torch.float32
    assert pseudo_exp(W, pseudo_log(W, Y)).dtype == torch.float32


def test_inverse_polar_retraction_for_a_wide_bimap():
    """p = 32 (a 64 -> 32 BiMap): the O(p^3) Lyapunov solve stays exact."""
    W = _rand_stiefel(64, 32, seed=14)
    Y = _near(W, 0.3, seed=15)
    assert torch.norm(retraction(W, inverse_polar_retraction(W, Y)) - Y) < 1e-9
