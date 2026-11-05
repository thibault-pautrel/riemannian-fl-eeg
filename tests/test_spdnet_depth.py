"""SPDNet with several BiRe blocks (depth), list parsing and ReEig-floor validation."""
import pytest
import torch
import torch.nn as nn


def _spd_batch(n=4, d=22, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, d, generator=g)
    return x @ x.transpose(-2, -1) + 0.1 * torch.eye(d)


@pytest.fixture
def spdnet():
    pytest.importorskip("yetanotherspdnet")
    from rfl.models import spdnet
    return spdnet


def test_spdnet_depth1_state_dict_keys(spdnet):
    model = spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dim=11, seed=0)
    assert model.depth == 1 and model.hidden_dim == 11
    assert set(model.state_dict()) == {"bimap.weight", "classifier.weight",
                                       "classifier.bias"}
    assert model.bimap.weight.shape == (22, 11)
    assert model(_spd_batch()).shape == (4, 4)


def test_spdnet_default_hidden_dim_is_half_the_channels(spdnet):
    model = spdnet.create_spdnet(n_classes=2, n_channels=22, seed=0)
    assert model.hidden_dim == 11


def test_spdnet_depth2_forward_and_bimap_keys(spdnet):
    deep = spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dims=[16, 8],
                                epsilons=[1e-4, 1e-2], seed=0, track_clipping=True)
    keys = set(deep.state_dict())
    assert deep.depth == 2
    assert {"bimap.weight", "bimap_2.weight"} <= keys
    assert deep.bimap.weight.shape == (22, 16) and deep.bimap_2.weight.shape == (16, 8)
    assert deep(_spd_batch()).shape == (4, 4)
    # Aggregation finds Stiefel parameters by the 'bimap' substring.
    assert len([k for k in keys if "bimap" in k and k.endswith(".weight")]) == 2
    act = deep.reeig_activity()
    assert len(act) == 2 and all(a >= 0 for a in act)


def test_spdnet_hidden_dims_accept_comma_strings(spdnet):
    deep = spdnet.create_spdnet(n_classes=2, n_channels=22, hidden_dims="16,8",
                                epsilons="1e-4,1e-2", seed=0)
    assert deep.hidden_dims == [16, 8] and deep.epsilons == [1e-4, 1e-2]


def test_spdnet_project_stiefel_restores_orthonormality(spdnet):
    deep = spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dims=[16, 8],
                                epsilons=[1e-4, 1e-2], seed=0)
    with torch.no_grad():
        for b in deep.bimaps():
            b.weight.add_(0.1 * torch.randn_like(b.weight))
    deep.project_stiefel()
    for b in deep.bimaps():
        W = b.weight.detach()
        assert float(torch.norm(W.T @ W - torch.eye(W.shape[1]))) < 1e-4


def test_spdnet_same_seed_gives_same_init(spdnet):
    a = spdnet.create_spdnet(n_classes=2, n_channels=10, hidden_dim=5, seed=3)
    b = spdnet.create_spdnet(n_classes=2, n_channels=10, hidden_dim=5, seed=3)
    assert torch.equal(a.bimap.weight, b.bimap.weight)


@pytest.mark.parametrize("bad", [[1e-2, 1e-2], [1e-2, 1e-4]])
def test_spdnet_rejects_non_increasing_epsilons(spdnet, bad):
    with pytest.raises(ValueError):
        spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dims=[16, 8], epsilons=bad)


def test_spdnet_rejects_single_epsilon_for_deep_model(spdnet):
    with pytest.raises(ValueError):
        spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dims=[16, 8], epsilon=1e-4)


def test_spdnet_rejects_increasing_dimension(spdnet):
    with pytest.raises(ValueError):
        spdnet.create_spdnet(n_classes=2, n_channels=8, hidden_dims=[6, 7],
                             epsilons=[1e-4, 1e-2])


def test_spdnet_flat_epsilons_allowed_when_not_strict(spdnet):
    with pytest.warns(UserWarning):
        spdnet.create_spdnet(n_classes=2, n_channels=8, hidden_dims=[6, 4],
                             epsilons=[1e-3, 1e-3], strict_epsilons=False)


def test_spdnet_has_no_dropout(spdnet):
    deep = spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dims=[16, 8],
                                epsilons=[1e-4, 1e-2], seed=0)
    assert not any(isinstance(m, nn.Dropout) for m in deep.modules())


def test_parse_lists():
    from rfl.models.spdnet import parse_float_list, parse_int_list
    assert parse_int_list("16,8") == [16, 8]
    assert parse_int_list("16;8") == [16, 8]
    assert parse_int_list("", [3]) == [3]
    assert parse_float_list("1e-4, 1e-2") == [1e-4, 1e-2]
    assert parse_float_list(None) is None
