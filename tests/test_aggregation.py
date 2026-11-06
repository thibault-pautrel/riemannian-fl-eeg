"""Aggregation rules: fedavg, projavg, rlavg and retavg.

Uses a toy model with a BiMap-named module, so SPDNet is not required.
"""

import pytest
import torch
import torch.nn as nn

from rfl.federated.aggregation import (
    extract_stiefel_matrices_from_model, fedavg_aggregate, identify_batchnorm_keys,
    identify_stiefel_keys, projavg_aggregate, retavg_aggregate, rlavg_aggregate,
)
from rfl.geometry.stiefel import polar_projection, retraction, tangent_projection

DT = torch.float64


class _BiMap(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.weight = nn.Parameter(W)


class _TinySPDNet(nn.Module):
    def __init__(self, W, k=4):
        super().__init__()
        self.bimap = _BiMap(W)
        self.classifier = nn.Linear(W.shape[1], k)


def _rand_stiefel(n, p, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, p, generator=g, dtype=DT))
    return Q


def _clients_near(Wt, k, step, seed=0):
    """k points R_{Wt}(step * xi_i) with unit tangent directions xi_i."""
    out = []
    for i in range(k):
        g = torch.Generator().manual_seed(seed + i)
        xi = tangent_projection(Wt, torch.randn(Wt.shape, generator=g, dtype=DT))
        out.append(retraction(Wt, step * xi / xi.norm()))
    return out


def _client_dicts(client_Ws, k_classes=4):
    """(Stiefel matrices, state dicts) as the server passes them."""
    csm = [{"bimap.weight": W.float()} for W in client_Ws]
    csd = [{"bimap.weight": W.float(),
            "classifier.weight": torch.full((k_classes, W.shape[1]), float(i)),
            "classifier.bias": torch.full((k_classes,), float(i))}
           for i, W in enumerate(client_Ws)]
    return csm, csd


def _ortho_err(W):
    W = W.double()
    return float(torch.norm(W.T @ W - torch.eye(W.shape[1], dtype=DT)))


# --------------------------------------------------------------------------- #
#  Key identification
# --------------------------------------------------------------------------- #
def test_identify_keys_on_a_bn_model():
    sd = {"bimap.weight": torch.zeros(4, 2),
          "classifier.weight": torch.zeros(3, 2), "classifier.bias": torch.zeros(3),
          "batchnorm_a.weight": torch.zeros(2), "batchnorm_a.running_mean": torch.zeros(2),
          "norm.weight": torch.zeros(2), "norm.running_var": torch.zeros(2)}
    assert identify_stiefel_keys(sd) == {"bimap.weight"}
    assert identify_batchnorm_keys(sd) == {
        "batchnorm_a.weight", "batchnorm_a.running_mean",
        "norm.weight", "norm.running_var"}


def test_extract_stiefel_matrices_reads_bimap_modules():
    W = _rand_stiefel(6, 3).float()
    out = extract_stiefel_matrices_from_model(_TinySPDNet(W))
    assert set(out) == {"bimap.weight"}
    assert torch.equal(out["bimap.weight"], W)


# --------------------------------------------------------------------------- #
#  fedavg
# --------------------------------------------------------------------------- #
def test_fedavg_is_weighted_mean_and_skips_local_norm():
    sds = [{"w": torch.tensor([0.0, 2.0]), "bn.running_mean": torch.tensor([1.0]),
            "bn.weight": torch.tensor([1.0])},
           {"w": torch.tensor([4.0, 6.0]), "bn.running_mean": torch.tensor([3.0]),
            "bn.weight": torch.tensor([3.0])}]
    agg = fedavg_aggregate(sds, weights=[1.0, 3.0])
    assert set(agg) == {"w"}
    assert torch.allclose(agg["w"], torch.tensor([3.0, 5.0]))
    # None and all-zero weights both mean uniform.
    assert torch.allclose(fedavg_aggregate(sds)["w"], torch.tensor([2.0, 4.0]))
    assert torch.allclose(fedavg_aggregate(sds, [0.0, 0.0])["w"], torch.tensor([2.0, 4.0]))


def test_fedavg_keeps_integer_tensors_of_the_first_client():
    sds = [{"counter": torch.tensor(5), "w": torch.tensor([1.0])},
           {"counter": torch.tensor(9), "w": torch.tensor([3.0])}]
    agg = fedavg_aggregate(sds)
    assert agg["counter"].dtype == torch.int64
    assert int(agg["counter"]) == 5
    assert torch.allclose(agg["w"], torch.tensor([2.0]))


def test_fedavg_leaves_the_stiefel_manifold():
    Ws = _clients_near(_rand_stiefel(10, 4, seed=5), 4, 0.5, seed=50)
    _, csd = _client_dicts(Ws)
    assert _ortho_err(fedavg_aggregate(csd)["bimap.weight"]) > 1e-3


def test_no_clients_raises():
    with pytest.raises(ValueError):
        fedavg_aggregate([])


# --------------------------------------------------------------------------- #
#  Riemannian rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rule", ["projavg", "rlavg", "retavg"])
def test_riemannian_rules_stay_on_stiefel(rule):
    n, p, k = 22, 11, 6
    Wt = _rand_stiefel(n, p, seed=1)
    gm = _TinySPDNet(Wt.float())
    csm, csd = _client_dicts(_clients_near(Wt, k, 0.05, seed=10))
    if rule == "projavg":
        out = projavg_aggregate(csm, csd)
    else:
        out = {"rlavg": rlavg_aggregate, "retavg": retavg_aggregate}[rule](csm, csd, gm)
    assert _ortho_err(out["bimap.weight"]) < 1e-4
    # Euclidean keys are the uniform mean of 0, 1, ..., k-1.
    assert torch.allclose(out["classifier.bias"], torch.full((4,), (k - 1) / 2))


def test_projavg_recovers_consensus():
    W = _rand_stiefel(12, 5, seed=7)
    csm, csd = _client_dicts([W] * 4)
    out = projavg_aggregate(csm, csd)["bimap.weight"].double()
    assert torch.norm(out - W) < 1e-5


def test_projavg_equals_polar_factor_of_mean():
    Ws = _clients_near(_rand_stiefel(12, 5, seed=8), 3, 0.2, seed=80)
    csm, csd = _client_dicts(Ws)
    expected = polar_projection(sum(W.float() for W in Ws) / 3)
    assert torch.allclose(projavg_aggregate(csm, csd)["bimap.weight"], expected, atol=1e-5)


def test_retavg_consensus_recovery():
    """Clients equal to a common point Wc near the base point give back Wc."""
    n, p, k = 22, 11, 5
    Wt = _rand_stiefel(n, p, seed=2)
    gm = _TinySPDNet(Wt.float())
    Wc = _clients_near(Wt, 1, 0.04, seed=20)[0]
    csm, csd = _client_dicts([Wc] * k)
    out = retavg_aggregate(csm, csd, gm)["bimap.weight"].double()
    assert torch.norm(out - Wc) < 1e-4
    # Clients equal to the base point give back the base point.
    csm0, csd0 = _client_dicts([Wt] * k)
    out0 = retavg_aggregate(csm0, csd0, gm)["bimap.weight"].double()
    assert torch.norm(out0 - Wt) < 1e-4


def test_rules_agree_to_first_order_for_small_steps():
    n, p, k = 22, 11, 8
    Wt = _rand_stiefel(n, p, seed=3)
    gm = _TinySPDNet(Wt.float())
    csm, csd = _client_dicts(_clients_near(Wt, k, 0.03, seed=30))
    out_proj = projavg_aggregate(csm, csd)["bimap.weight"].double()
    out_rl = rlavg_aggregate(csm, csd, gm)["bimap.weight"].double()
    out_ret = retavg_aggregate(csm, csd, gm)["bimap.weight"].double()
    assert torch.norm(out_ret - out_rl) < 5e-2
    assert torch.norm(out_ret - out_proj) < 5e-2


def test_retavg_rejects_points_outside_the_invertibility_domain():
    W = _rand_stiefel(6, 3, seed=9)
    gm = _TinySPDNet(W.float())
    csm, csd = _client_dicts([-W])          # M = W^T Y = -I is not positive definite
    with pytest.raises(ValueError):
        retavg_aggregate(csm, csd, gm)


# --------------------------------------------------------------------------- #
#  Registry of the rules (a new mean on St(n, p) in one function)
# --------------------------------------------------------------------------- #
def test_registry_holds_the_rules_and_their_kind():
    from rfl.federated.aggregation import get_rule, rule_names
    assert set(rule_names()) >= {"fedavg", "projavg", "rlavg", "retavg"}
    assert "fedavg" not in rule_names(stiefel=True) and not get_rule("fedavg").stiefel
    with pytest.raises(ValueError, match="unknown aggregation rule"):
        get_rule("nope")


def test_a_registered_mean_is_a_full_rule(monkeypatch):
    import rfl.federated.aggregation as A
    monkeypatch.setattr(A, "AGGREGATION_RULES", dict(A.AGGREGATION_RULES))
    seen = {}

    def mean(Ws, weights, base):              # the polar factor of the mean: ProjAvg
        seen["base"], seen["weights"] = base, weights
        return polar_projection(sum(w * W for w, W in zip(weights, Ws)))

    A.register_stiefel_mean("mymean", mean, "test")
    with pytest.raises(ValueError, match="exists"):
        A.register_stiefel_mean("mymean", mean)
    Wt = _rand_stiefel(12, 5, seed=3)
    csm, csd = _client_dicts(_clients_near(Wt, 3, 0.2, seed=30))
    out = A.get_rule("mymean").aggregate(csm, csd, _TinySPDNet(Wt.float()), [1, 1, 2])
    ref = projavg_aggregate(csm, csd, [1, 1, 2])
    assert set(out) == set(ref)
    for k in ref:
        assert torch.allclose(out[k], ref[k], atol=1e-6), k
    assert torch.allclose(seen["base"], Wt.float()) and abs(sum(seen["weights"]) - 1) < 1e-12
