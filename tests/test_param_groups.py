"""Identification of local normalisation layers and of the classifier head, freezing."""

import torch.nn as nn

from rfl.federated.aggregation import identify_batchnorm_keys
from rfl.federated.param_groups import identify_bn_buffer_names, identify_bn_param_names


class _ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.classifier = nn.Linear(4, 2)


class _ToySPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.bimap = nn.Linear(8, 4, bias=False)     # stands in for a BiMap
        self.classifier = nn.Linear(4, 2)


class _NamedBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.batchnorm_a = nn.BatchNorm2d(4)
        self.classifier = nn.Linear(4, 2)


def test_batchnorm_is_local_with_its_buffers():
    model = _ConvBN()
    assert identify_bn_param_names(model) == {"bn.weight", "bn.bias"}
    assert identify_bn_buffer_names(model) == {
        "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}


def test_bimap_weight_and_head_are_not_norm():
    spd = _ToySPD()
    assert identify_bn_param_names(spd) == set()
    assert identify_bn_buffer_names(spd) == set()


def test_module_rules_agree_with_state_dict_rules():
    """param_groups (module tree) and aggregation (state-dict keys) select the same keys."""
    for model in (_ConvBN(), _ToySPD(), _NamedBN()):
        by_module = identify_bn_param_names(model) | identify_bn_buffer_names(model)
        by_key = identify_batchnorm_keys(model.state_dict())
        assert by_module == by_key, type(model).__name__


class _NoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.last = nn.Linear(4, 2)


def test_classifier_head_identification():
    from rfl.federated.param_groups import identify_classifier_param_names
    assert identify_classifier_param_names(_ConvBN()) == {"classifier.weight",
                                                          "classifier.bias"}
    assert "bimap.weight" not in identify_classifier_param_names(_ToySPD())
    assert identify_classifier_param_names(_NoClassifier()) == {"last.weight",
                                                                "last.bias"}


def test_freeze_trunk_keeps_only_the_head_trainable():
    from rfl.federated.param_groups import freeze_trunk
    model = _ConvBN()
    assert freeze_trunk(model) == sum(1 for n, _ in model.named_parameters()
                                      if not n.startswith("classifier."))
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {
        "classifier.weight", "classifier.bias"}
