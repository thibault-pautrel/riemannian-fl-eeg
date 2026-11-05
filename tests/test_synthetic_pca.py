"""Checks of the geometry, the gradients, the noise, the aggregators and one short run."""
import numpy as np
import pytest

from rfl.geometry import grassmann as gr
from rfl.synthetic_pca.algorithms import AGGREGATORS, Config, _clipped_grad, aggregate, metrics, run
from rfl.synthetic_pca.data import make_federation

RNG = np.random.default_rng(0)
D, P_ = 12, 3


def _point():
    return gr.random_point(D, P_, RNG)


def _tangent(P, scale=0.3):
    return scale * gr.tangent_proj(P, gr.sym(RNG.standard_normal((D, D))))


def test_tangent_projector():
    P = _point()
    X = gr.sym(RNG.standard_normal((D, D)))
    Y = gr.sym(RNG.standard_normal((D, D)))
    T = gr.tangent_proj(P, X)
    I = np.eye(D)
    assert np.allclose(P @ T @ P, 0) and np.allclose((I - P) @ T @ (I - P), 0)
    assert np.allclose(gr.tangent_proj(P, T), T)                                  # idempotent
    assert np.isclose(np.sum(gr.tangent_proj(P, X) * Y), np.sum(X * gr.tangent_proj(P, Y)))


def test_projection():
    P = _point()
    assert np.allclose(gr.proj(P, P_), P)
    Q = gr.proj(P + 0.1 * gr.sym(RNG.standard_normal((D, D))), P_)
    assert np.allclose(Q @ Q, Q) and np.isclose(np.trace(Q), P_)
    u = 1e-4 * gr.sym(RNG.standard_normal((D, D)))      # Pi(P + u) = P + P_P(u) + O(|u|^2)
    assert np.linalg.norm(gr.proj(P + u, P_) - P - gr.tangent_proj(P, u)) < 1e-6


def test_retraction_and_its_inverse():
    P = _point()
    X = _tangent(P)
    Q = gr.retraction(P, X, P_)
    assert np.allclose(Q @ Q, Q) and np.isclose(np.trace(Q), P_)
    assert np.allclose(gr.inverse_retraction(P, Q, P_), X)                       # R^{-1} R = id
    assert np.allclose(gr.retraction(P, np.zeros((D, D)), P_), P)
    u = 1e-5 * X                                                                  # first order
    assert np.linalg.norm(gr.retraction(P, u, P_) - P - u) < 1e-3 * np.linalg.norm(u)
    Q2 = gr.proj(P + _tangent(P), P_)                                             # R R^{-1} = id
    assert np.allclose(gr.retraction(P, gr.inverse_retraction(P, Q2, P_), P_), Q2)


def test_noise_is_isotropic_and_tangent_noise_has_d_M_directions():
    sigma, n = 0.7, 4000
    P = _point()
    E = np.zeros((D, D))
    E[0, 1] = E[1, 0] = 1 / np.sqrt(2)                       # a unit vector of S_d
    xs = [gr.sym_gaussian(D, sigma, RNG) for _ in range(n)]
    assert np.isclose(np.var([np.sum(x * E) for x in xs]), sigma ** 2, rtol=0.1)
    sq = np.mean([np.sum(gr.tangent_proj(P, x) ** 2) for x in xs])
    assert np.isclose(sq, sigma ** 2 * gr.intrinsic_dim(D, P_), rtol=0.05)     # E||P_P(N)||^2 = sigma^2 d_M


def test_per_sample_gradient_and_clipping():
    P = _point()
    z = RNG.standard_normal((1, D))
    z /= np.linalg.norm(z)
    expected = gr.tangent_proj(P, -0.5 * np.outer(z[0], z[0]))
    assert np.allclose(_clipped_grad(z, P, 10.0), expected)
    assert np.linalg.norm(expected) <= 1 / (2 * np.sqrt(2)) + 1e-12   # C = 1/(2 sqrt 2) never clips
    C = 0.5 * np.linalg.norm(expected)
    assert np.isclose(np.linalg.norm(_clipped_grad(z, P, C)), C)


def test_federation_and_metrics():
    fed = make_federation(n=4, m=50, d=D, p=P_, seed=1)
    assert max(np.linalg.norm(Z, axis=1).max() for Z in fed.clients) <= 1 + 1e-12
    gap, grad2, sin2 = metrics(fed, fed.P_star)
    assert abs(gap) < 1e-12 and grad2 < 1e-20 and abs(sin2) < 1e-12
    gap, grad2, sin2 = metrics(fed, _point())
    assert gap > 0 and grad2 > 0 and sin2 > 0


@pytest.mark.parametrize("agg", AGGREGATORS)
def test_aggregators_fix_a_consensus_and_agree_to_first_order(agg):
    P = _point()
    assert np.allclose(aggregate(P, [P, P, P], P_, agg), P)
    local = [gr.proj(P + _tangent(P, 1e-4), P_) for _ in range(3)]
    ref = aggregate(P, local, P_, "projavg")
    assert np.linalg.norm(aggregate(P, local, P_, agg) - ref) < 1e-6           # agree with ProjAvg to first order


@pytest.mark.parametrize("agg", AGGREGATORS)
def test_non_private_run_converges(agg):
    fed = make_federation(n=6, m=60, d=D, p=P_, seed=2)
    h = run(fed, Config(aggregator=agg, T=150, k=3, tau=3, B=20, alpha=2.0))
    assert h.gap[-1] < 0.05 * h.gap[0]


def test_tikz_writer(tmp_path):
    from rfl.analysis.tikz import grid_figure
    out = tmp_path / "fig.tex"
    pts = dict(style="ours", x=[1, 2], y=[1e-3, 2e-3], err_lo=[1e-4, 1e-4], err_hi=[2e-4, 2e-4])
    line = dict(style="prediction", x=[1, 2], y=[1e-3, 2e-3])
    panel = dict(title="(a)", xlabel="$k$", ylabel="cost", xticks=[1, 2], ymin=1e-4,
                 ymax=1e-2, series=[pts, line])
    grid_figure(out, [panel] * 4, [("DP-RFedProj", "ours"), ("prediction", "prediction")])
    text = out.read_text()
    assert text.count("\\nextgroupplot") == 4 and "\\begin{tikzpicture}" in text
    assert text.count("\\addlegendentry") == 2                  # one legend only
    assert "legend columns=-1" in text and "pgfplotslegendfromname" in text   # below
