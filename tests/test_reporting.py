"""Tests of the result loader, the summary tables and the paired tests (analysis/)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from rfl.analysis import report, stats
from rfl.analysis.results import arch_name, load_curve, load_runs, varying_columns


def _write_fed(root, name, *, seed, acc, agg="projavg", eps=None, model="spdnet",
               norm="batch", rounds=5, lr=0.01):
    cfg = {"dataset": "BNCI2014_001", "model_type": model, "eegnet_norm_type": norm,
           "fl_aggregation": agg, "fl_client_fraction": 1.0, "dp_enabled": eps is not None,
           "dp_target_eps": eps if eps is not None else 3.0, "dp_accounting": "ours",
           "fl_rounds": rounds, "learning_rate": lr, "batch_size": 32, "seed": seed}
    curve = list(np.linspace(0.3, acc, rounds))
    r = {"config": cfg,
         "final_metrics": {"global_test_acc": acc, "global_test_f1": acc - 0.01,
                           "global_val_acc": acc, "global_val_loss": 1.0},
         "last_round_metrics": {"global_test_acc": acc},
         "round_history": {"round": list(range(1, rounds + 1)), "global_test_acc": curve,
                           "agg_seconds": [0.002] * rounds},
         "dp_budget": ({"C": 1.0, "sigma": 0.5, "mu0": 0.1, "realized_eps": eps}
                       if eps is not None else None)}
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "fl_results.json"), "w") as f:
        json.dump(r, f)
    return d


def _write_central(root, name, *, seed, acc):
    cfg = {"dataset": "BNCI2014_001", "model_type": "eegnet", "eegnet_norm_type": "none",
           "epochs": 4, "learning_rate": 1e-3, "batch_size": 64, "seed": seed}
    r = {"config": cfg,
         "training": {"report_rule": "last", "final_val_acc": acc, "final_val_loss": 0.9,
                      "final_epoch": 4, "test_acc_history": [0.3, 0.4, 0.5, acc]},
         "test_metrics": {"accuracy": acc, "f1_macro": acc - 0.02}}
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "centralized_results.json"), "w") as f:
        json.dump(r, f)


@pytest.fixture
def results_dir(tmp_path):
    for s, (a, b) in enumerate([(0.70, 0.60), (0.72, 0.61), (0.69, 0.62),
                                (0.75, 0.60), (0.71, 0.63), (0.73, 0.59)]):
        _write_fed(tmp_path, f"proj{s}", seed=s, acc=a, agg="projavg")
        _write_fed(tmp_path, f"ret{s}", seed=s, acc=b, agg="retavg")
        _write_fed(tmp_path, f"priv{s}", seed=s, acc=b - 0.1, agg="projavg", eps=5.0)
    _write_central(tmp_path, "central0", seed=0, acc=0.8)
    return tmp_path


def test_arch_names():
    assert arch_name({"model_type": "spdnet"}) == "spdnet"
    assert arch_name({"model_type": "eegnet", "eegnet_norm_type": "batch"}) == "eegnet_bn"
    assert arch_name({"model_type": "eegnet", "eegnet_norm_type": "none"}) == "eegnet_nobn"


def test_load_runs_reads_both_kinds(results_dir):
    df = load_runs(str(results_dir))
    assert len(df) == 19
    assert set(df["kind"]) == {"federated", "centralized"}
    central = df[df["kind"] == "centralized"].iloc[0]
    assert central["arch"] == "eegnet_nobn" and central["test_acc"] == 0.8
    priv = df[df["eps"] == 5.0]
    assert len(priv) == 6 and (priv["realized_eps"] == 5.0).all()
    assert df[df["kind"] == "federated"]["agg_ms"].iloc[0] == pytest.approx(2.0)
    assert "seed" not in varying_columns(df)


def test_summary_table_and_filters(results_dir, capsys):
    df = report.apply_filters(load_runs(str(results_dir)), ["kind=federated", "eps=nan"])
    assert len(df) == 12
    summary = report.summarize(df, ["aggregation"], "test_acc")
    proj = summary.set_index("aggregation").loc["projavg"]
    assert proj["n"] == 6
    assert proj["mean"] == pytest.approx(np.mean([0.70, 0.72, 0.69, 0.75, 0.71, 0.73]))

    report.main([str(results_dir), "--where", "kind=federated",
                 "--by", "aggregation", "--pivot", "eps"])
    out = capsys.readouterr().out
    assert "projavg" in out and "± " in out


def test_learning_curves_are_saved(results_dir, tmp_path):
    out = tmp_path / "figs"
    report.main([str(results_dir), "--where", "kind=federated", "--curves", str(out)])
    assert any(p.suffix == ".png" for p in out.iterdir())
    assert load_curve(str(results_dir / "proj0")).shape == (5,)


def test_paired_tests_detect_a_consistent_difference(results_dir):
    df = report.apply_filters(load_runs(str(results_dir)), ["kind=federated", "eps=nan"])
    pw = stats.paired_pairwise(df, "test_acc", "aggregation", [])
    row = pw.iloc[0]
    assert row["n_pairs"] == 6 and row["win_rate"] in (0.0, 1.0)
    assert row["p_value"] == pytest.approx(0.03125)        # exact two-sided, n = 6
    assert stats.rank_biserial(np.array([1.0, 2.0, -0.5])) == pytest.approx(
        (5 - 1) / 6)


def test_holm_bonferroni():
    assert stats.holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_tests_skip_groups_with_too_few_pairs(results_dir):
    df = load_runs(str(results_dir))
    df = df[(df["kind"] == "federated") & (df["seed"] < 3)]
    pw = stats.paired_pairwise(df, "test_acc", "aggregation", ["eps"], min_pairs=6)
    assert pw["p_value"].isna().all()
    assert isinstance(pw, pd.DataFrame)


def test_smooth_is_a_centred_moving_average():
    import numpy as np
    from rfl.analysis.report import _smooth
    c = np.array([0.0, 3.0, 6.0, 9.0])
    assert np.allclose(_smooth(c, 1), c)
    assert np.allclose(_smooth(c, 3), [1.5, 3.0, 6.0, 7.5])


def test_scope_names():
    from rfl.analysis.results import scope_name
    assert scope_name({}) == "shared"
    assert scope_name({"fl_local_fc": True}) == "fedper"
    assert scope_name({"fl_local_fc": True, "fl_local_head_mode": "frozen",
                       "fl_head_refit_epochs": "25,50"}) == "fedper_frozen+refit"
    assert scope_name({"fl_local_fc": True, "freeze_trunk": True}) == "fedper_rnd"
    assert scope_name({"fl_local_only": True, "freeze_trunk": True}) == "local_rnd"
    assert scope_name({"fl_head_refit_epochs": "0"}) == "shared"
    assert scope_name({"fl_head_refit_epochs": "10"}) == "shared+refit"


def test_scope_name_matches_the_config():
    from rfl.analysis.results import scope_name
    from rfl.federated.fl_config import FLConfig
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="spd", model_type="spdnet", device="cpu")
    for kw in (dict(), dict(fl_local_fc=True), dict(fl_local_only=True),
               dict(freeze_trunk=True, fl_local_fc=True),
               dict(fl_local_fc=True, fl_local_head_mode="frozen",
                    fl_head_refit_epochs="5"),
               dict(fl_head_refit_epochs="5", dp_enabled=True)):
        cfg = FLConfig(**base, **kw)
        assert scope_name(cfg.to_dict()) == cfg.personalisation_name()


def test_depth_runs_get_their_own_arch():
    assert arch_name({"model_type": "spdnet", "spdnet_hidden_dims": "16,8",
                      "spdnet_epsilons": "1e-4,2e-4"}) == "spdnet_16,8_1e-4,2e-4"


def test_duplicate_seeds_are_detected_and_refused_by_the_paired_tests(tmp_path):
    from rfl.analysis.results import duplicate_seeds
    for s, (a, b) in enumerate([(0.7, 0.6), (0.72, 0.61), (0.69, 0.62)]):
        _write_fed(tmp_path, f"proj{s}", seed=s, acc=a, agg="projavg")
        _write_fed(tmp_path, f"ret{s}", seed=s, acc=b, agg="retavg")
    # Same seed, another learning rate: a second configuration of the same arm.
    d = _write_fed(tmp_path, "proj0_lr", seed=0, acc=0.9, agg="projavg")
    r = json.load(open(os.path.join(d, "fl_results.json")))
    r["config"]["learning_rate"] = 0.1
    json.dump(r, open(os.path.join(d, "fl_results.json"), "w"))
    df = load_runs(str(tmp_path))
    dups = duplicate_seeds(df, ["aggregation"])
    assert len(dups) == 1 and int(dups["n_configs"].iloc[0]) == 2
    with pytest.raises(ValueError):
        stats.paired_pairwise(df, "test_acc", "aggregation", [], min_pairs=2)
    assert duplicate_seeds(df, ["aggregation", "lr"]).empty


def test_final_runs_are_paired_across_arms_with_their_own_hyperparameters(tmp_path):
    """Each arm keeps its selected lr: the arm and the lr must not split the pairs."""
    for s, (a, b) in enumerate([(0.70, 0.60), (0.72, 0.61), (0.69, 0.62),
                                (0.75, 0.60), (0.71, 0.63), (0.73, 0.59)]):
        _write_fed(tmp_path, f"spdnet_projavg/lr0.1/seed{s}", seed=s, acc=a, lr=0.1)
        _write_fed(tmp_path, f"spdnet_retavg/lr0.01/seed{s}", seed=s, acc=b, agg="retavg")
    df = load_runs(str(tmp_path))
    assert set(df["arm"]) == {"spdnet_projavg", "spdnet_retavg"}
    for col in ("hidden_dim", "clip_C", "weight_decay", "local_epochs"):
        assert col in df.columns
    out = tmp_path / "out"
    assert stats.main([str(tmp_path), "--out-dir", str(out)]) in (0, None)
    pw = pd.read_csv(out / "pairwise_test_acc.csv")
    assert len(pw) == 1 and pw["n_pairs"].iloc[0] == 6


def test_per_client_accuracy_columns(tmp_path):
    d = _write_fed(tmp_path, "run", seed=0, acc=0.7)
    path = os.path.join(d, "fl_results.json")
    r = json.load(open(path))
    r["final_metrics"]["per_client_test_acc"] = [0.9, 0.6, 0.6]
    json.dump(r, open(path, "w"))
    row = load_runs(str(tmp_path)).iloc[0]
    assert row["client_mean_acc"] == pytest.approx(0.7) and row["client_min_acc"] == 0.6
    assert row["clients"] == "subject"
