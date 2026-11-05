"""Tests for rfl/models/spdnet.py and rfl/models/eegnet.py."""
import pytest
import torch
import torch.nn as nn

from rfl.federated.aggregation import identify_batchnorm_keys
from rfl.federated.param_groups import identify_bn_param_names
from rfl.models.eegnet import create_eegnet

SLOTS = ("temporal", "spatial", "separable")
BN_SUFFIXES = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


# =========================================================================== #
#  SPDNet
# =========================================================================== #
def _spd_batch(n=4, d=22, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, d, generator=g)
    return x @ x.transpose(-2, -1) + 0.1 * torch.eye(d)


@pytest.fixture
def spdnet():
    pytest.importorskip("yetanotherspdnet")
    from rfl.models import spdnet
    return spdnet


def test_spdnet_state_dict_keys(spdnet):
    model = spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dim=11, seed=0)
    assert model.hidden_dim == 11
    assert list(model.state_dict()) == ["bimap.weight", "classifier.weight",
                                        "classifier.bias"]
    assert model.bimap.weight.shape == (22, 11)
    assert model(_spd_batch()).shape == (4, 4)


def test_spdnet_project_stiefel_restores_orthonormality(spdnet):
    model = spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dim=8, seed=0)
    with torch.no_grad():
        model.bimap.weight.add_(0.1 * torch.randn_like(model.bimap.weight))
    model.project_stiefel()
    W = model.bimap.weight.detach()
    assert float(torch.norm(W.T @ W - torch.eye(W.shape[1]))) < 1e-4


def test_spdnet_rejects_increasing_dimension(spdnet):
    with pytest.raises(ValueError):
        spdnet.create_spdnet(n_classes=2, n_channels=8, hidden_dim=9)


def test_spdnet_has_no_dropout(spdnet):
    model = spdnet.create_spdnet(n_classes=4, n_channels=22, hidden_dim=8, seed=0)
    assert not any(isinstance(m, nn.Dropout) for m in model.modules())


# =========================================================================== #
#  EEGNet
# =========================================================================== #
NORM_TYPES = ["batch", "none"]


def _eegnet(norm_type="batch", n_ch=22, n_t=512, n_classes=4):
    return create_eegnet(n_classes=n_classes, n_channels=n_ch, n_timepoints=n_t,
                         sampling_rate=128, norm_type=norm_type)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


def _local_keys(model):
    """Keys kept local by the aggregation, and parameters kept local by param_groups."""
    return identify_batchnorm_keys(model.state_dict()), identify_bn_param_names(model)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("n_ch", [14, 22, 64])
def test_eegnet_forward_shapes(norm_type, n_ch):
    torch.manual_seed(0)
    m = _eegnet(norm_type, n_ch=n_ch, n_classes=2)
    out = m(torch.randn(2, 1, n_ch, 512))
    assert out.shape == (2, 2) and torch.isfinite(out).all()
    assert m(torch.randn(2, n_ch, 512)).shape == (2, 2)          # 3-D input


def test_eegnet_kernel_sizes_follow_sampling_rate():
    m = _eegnet()
    assert m.conv_temporal.kernel_size == (1, 64)          # 0.5 * 128 Hz
    assert m.separable_kernel == 16                        # 128 / (2 * 4)


def test_eegnet_batchnorm_keys():
    keys = set(_eegnet().state_dict())
    for slot in SLOTS:
        for sfx in BN_SUFFIXES:
            assert f"batchnorm_{slot}.{sfx}" in keys
    assert not any(k.startswith("_norm") for k in keys)


def test_eegnet_local_norm_contract():
    """aggregation (state-dict keys) and param_groups (module tree) agree."""
    agg, pg = _local_keys(_eegnet("batch"))
    assert len(agg) == 15 and len(pg) == 6
    agg, pg = _local_keys(_eegnet("none"))
    assert agg == set() and pg == set()


def test_eegnet_none_has_no_norm_state():
    nonorm, bn = _eegnet("none"), _eegnet("batch")
    keys = set(nonorm.state_dict())
    assert not any(k.startswith("batchnorm") or "running" in k for k in keys)
    assert _n_params(bn) - _n_params(nonorm) == 2 * (bn.F1 + bn.F1 * bn.D + bn.F2)


def test_eegnet_train_and_eval_agree_without_batchnorm():
    torch.manual_seed(0)
    model = _eegnet("none")
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.p = 0.0
    x = torch.randn(4, 1, 22, 512)
    with torch.no_grad():
        model.eval()
        y_eval = model(x)
        model.train()
        y_train = model(x)
    assert torch.allclose(y_eval, y_train, atol=1e-6)


def test_eegnet_max_norm_constraints():
    model = _eegnet()
    with torch.no_grad():
        model.conv_spatial.weight.mul_(10.0)
        model.classifier.weight.mul_(10.0)
    model.apply_constraints()
    assert (model.conv_spatial.weight.flatten(1).norm(dim=1) <= 1.0 + 1e-5).all()
    assert (model.classifier.weight.norm(dim=1) <= model.norm_rate + 1e-5).all()


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_eegnet_keeps_dropout(norm_type):
    model = _eegnet(norm_type)
    assert any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in model.modules())


def test_eegnet_rejects_unknown_options():
    with pytest.raises(ValueError):
        _eegnet("group")


def test_eegnet_batchnorm_starts_with_untouched_running_statistics():
    import torch.nn as nn

    from rfl.models.eegnet import create_eegnet
    m = create_eegnet(n_classes=2, n_channels=8, n_timepoints=128, sampling_rate=128,
                      norm_type="batch")
    bns = [b for b in m.modules() if isinstance(b, nn.BatchNorm2d)]
    assert bns and all(int(b.num_batches_tracked) == 0 for b in bns)
    assert all(torch.equal(b.running_var, torch.ones_like(b.running_var)) for b in bns)
    assert m.training


def test_trivialization_keeps_w_in_the_state_dict_and_loads_it_exactly():
    import copy

    from rfl.models.spdnet import create_spdnet
    kw = dict(n_classes=3, n_channels=8, hidden_dims="6,4", epsilons="1e-4,1e-2",
              use_batch_norm=True, seed=1)
    m = create_spdnet(stiefel="trivialization", **kw)
    p = create_spdnet(stiefel="projection", **kw)
    assert list(m.state_dict()) == list(p.state_dict())          # same keys, same order
    assert torch.allclose(m.bimap.weight, p.bimap.weight, atol=1e-6)   # same init
    W = torch.linalg.qr(torch.randn(8, 6))[0]
    sd = m.state_dict()
    sd["bimap.weight"] = W
    m2 = copy.deepcopy(m)
    m2.load_state_dict(sd)
    assert torch.allclose(m2.state_dict()["bimap.weight"], W, atol=1e-6)
    opt = torch.optim.SGD(m2.parameters(), lr=0.5)
    X = torch.randn(10, 8, 30)
    X = X @ X.transpose(1, 2) / 30
    m2.train()
    m2(X).sum().backward()
    opt.step()
    for b in m2.bimaps():
        Wb = b.weight.detach().double()
        assert torch.allclose(Wb.T @ Wb, torch.eye(Wb.shape[1], dtype=Wb.dtype), atol=1e-5)
    assert not torch.allclose(m2.bimap.weight, W, atol=1e-4)     # it trained
