"""ReEig floors calibrated where each ReEig acts (rfl/experiments/analyze_covariances.py)."""
import json
import types

import numpy as np
import pytest

from rfl.experiments.analyze_covariances import architecture_floors, write_floors


@pytest.fixture(scope="module")
def covs():
    rng = np.random.default_rng(0)
    d = 24
    A = rng.normal(size=(300, d, 3 * d)) * np.exp(rng.normal(0, 1, d))[None, :, None]
    return A @ A.transpose(0, 2, 1) / (3 * d)


@pytest.mark.parametrize("dims", [[8], [16, 8], [16, 8, 4]])
@pytest.mark.parametrize("bn", [False, True])
def test_each_floor_clamps_the_target_share_of_its_own_spectrum(covs, dims, bn):
    r = architecture_floors(covs, dims, 15.0, bn)
    assert len(r["floors"]) == len(dims)
    for c in r["clamped"]:
        assert 0.10 < c < 0.20                   # 15 %, up to the rounding of the floor
    if not bn:
        assert all(b > a for a, b in zip(r["floors"], r["floors"][1:]))


def test_the_first_floor_ignores_the_batch_norm_and_later_ones_see_whitened_matrices(covs):
    plain = architecture_floors(covs, [16, 8], 15.0, False)
    bn = architecture_floors(covs, [16, 8], 15.0, True)
    assert plain["floors"][0] == bn["floors"][0]
    # After the GAH whitening the spectrum is centred near the identity, while without
    # batch norm it keeps the scale of a trace-normalised covariance (sum 1).
    assert 0.05 < bn["floors"][1] < 1.0 and plain["floors"][1] < 0.05


def test_floors_are_measured_after_the_congruence(covs):
    """The input-frame percentile clamps much less than 15 % after a reduction."""
    x = covs / np.trace(covs, axis1=1, axis2=2)[:, None, None]
    input_p15 = np.percentile(np.linalg.eigvalsh(x), 15)
    r = architecture_floors(covs, [8], 15.0, False)
    assert r["floors"][0] > input_p15


def _args(tmp_path):
    return types.SimpleNamespace(epsilon_clamp_percentile=15.0, percentiles=[5, 15, 30],
                                 trace_normalize=True, floors_dir=str(tmp_path),
                                 max_subjects=0, max_samples=2000, task="all_classes",
                                 estimator="scm", resample=128.0)


def test_write_floors_file(covs, tmp_path):
    path = write_floors("BNCI2014_001", covs, ["8", "16,8"], _args(tmp_path))
    d = json.load(open(path))
    assert set(d["architectures"]) == {"8", "16,8"}
    assert set(d["architectures"]["16,8"]["bn"]) == {"5", "15", "30"}
    assert len(d["architectures"]["16,8"]["bn"]["15"]["floors"]) == 2
    f5 = d["architectures"]["8"]["nobn"]["5"]["floors"][0]
    f30 = d["architectures"]["8"]["nobn"]["30"]["floors"][0]
    assert f5 < f30


@pytest.mark.parametrize("runner", ["fed", "central"])
def test_runners_read_the_floors_of_a_percentile(covs, tmp_path, runner):
    import contextlib
    import io

    from rfl.experiments import run_centralized_experiment as C
    from rfl.experiments import run_federated_experiment as F
    mod = F if runner == "fed" else C
    path = write_floors("BNCI2014_001", covs, ["16,8"], _args(tmp_path))
    want = json.load(open(path))["architectures"]["16,8"]
    base = ["--dataset", "BNCI2014_001", "--hidden-dims", "16,8", "--floors-file", path]
    _, a, _ = mod.prepare_args(base + ["--reeig-percentile", "30"])
    assert a.epsilons == ",".join(f"{x:g}" for x in want["nobn"]["30"]["floors"])
    _, a, _ = mod.prepare_args(base + ["--reeig-percentile", "5", "--use-batch-norm"])
    assert a.epsilons == ",".join(f"{x:g}" for x in want["bn"]["5"]["floors"])
    cfg = (F.build_config(a) if runner == "fed" else C.config_from_args(a, "BNCI2014_001"))
    assert cfg.spdnet_reeig_percentile == 5 and cfg.spdnet_epsilons == a.epsilons
    for bad, needle in ((["--reeig-percentile", "40", "--floors-file", path],
                         "40th percentile"),
                        (["--reeig-percentile", "5", "--epsilons", "1e-4,1e-3"], "drop"),
                        (["--floors-file", path], "only acts with --reeig-percentile")):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
            mod.prepare_args(["--dataset", "BNCI2014_001", "--hidden-dims", "16,8"] + bad)
        assert needle in err.getvalue()


def test_covariances_are_loaded_one_subject_at_a_time(monkeypatch):
    import rfl.data.data_utils as du
    from rfl.experiments.analyze_covariances import load_covariances
    calls = []

    def fake_load(name, subjects, **kw):
        calls.append(list(subjects))
        rng = np.random.default_rng(subjects[0])
        return rng.normal(size=(40, 4, 64)), np.arange(40) % 4, None
    monkeypatch.setattr(du, "load_moabb_dataset", fake_load)
    monkeypatch.setattr(du, "filter_binary_task", lambda X, y, meta, **kw: (X, y, meta))
    out = load_covariances("BNCI2014_001", [1, 2, 3], "all_classes", "scm", 0.0, 0.0,
                           128.0, max_samples=30)
    assert calls == [[1], [2], [3]]
    assert out["covs"].shape == (30, 4, 4) and out["n_classes"] == 4


def test_one_percentile_per_block(covs, tmp_path):
    """--per-block-percentiles: every pair of percentiles of a two-block architecture,
    the first floor set by q1 alone, the second by q2 on the whitened spectrum."""
    args = _args(tmp_path)
    args.per_block_percentiles = True
    path = write_floors("BNCI2014_001", covs, ["16,8"], args)
    bn = json.load(open(path))["architectures"]["16,8"]["bn"]
    assert {"5", "15", "30", "5,15", "30,5"} <= set(bn) and len(bn) == 9
    assert bn["5,30"]["floors"][0] == bn["5"]["floors"][0]
    assert bn["30,5"]["floors"][1] < bn["30"]["floors"][1]
    single = architecture_floors(covs, [16, 8], [30.0, 5.0], True)
    assert single["floors"] == bn["30,5"]["floors"]


def test_writing_floors_keeps_the_other_architectures(covs, tmp_path):
    write_floors("BNCI2014_001", covs, ["8"], _args(tmp_path))
    path = write_floors("BNCI2014_001", covs, ["16,8"], _args(tmp_path))
    assert set(json.load(open(path))["architectures"]) == {"8", "16,8"}


@pytest.mark.parametrize("runner", ["fed", "central"])
def test_runners_read_one_percentile_per_block(covs, tmp_path, runner):
    import contextlib
    import io

    from rfl.experiments import run_centralized_experiment as C
    from rfl.experiments import run_federated_experiment as F
    mod = F if runner == "fed" else C
    args = _args(tmp_path)
    args.per_block_percentiles = True
    path = write_floors("BNCI2014_001", covs, ["16,8"], args)
    want = json.load(open(path))["architectures"]["16,8"]["bn"]["5,30"]["floors"]
    base = ["--dataset", "BNCI2014_001", "--hidden-dims", "16,8", "--floors-file", path,
            "--use-batch-norm", "--reeig-percentile", "5"]
    _, a, _ = mod.prepare_args(base + ["--reeig-percentile-2", "30"])
    assert a.epsilons == ",".join(f"{x:g}" for x in want)
    cfg = (F.build_config(a) if runner == "fed" else C.config_from_args(a, "BNCI2014_001"))
    assert cfg.spdnet_reeig_percentile_2 == 30
    assert cfg.to_dict()["spdnet_reeig_percentile_2"] == 30
    _, a, _ = mod.prepare_args(base)
    cfg = (F.build_config(a) if runner == "fed" else C.config_from_args(a, "BNCI2014_001"))
    assert "spdnet_reeig_percentile_2" not in cfg.to_dict()     # same hash as before
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
        mod.prepare_args(["--dataset", "BNCI2014_001", "--hidden-dims", "16",
                          "--floors-file", path, "--reeig-percentile", "5",
                          "--reeig-percentile-2", "30"])
    assert "two blocks" in err.getvalue()
