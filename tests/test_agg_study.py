"""The aggregation study (plans mi_agg_calib and mi_agg): floors, calibration, arms."""
import importlib
import json

import numpy as np
import pytest

from rfl.experiments import grid as G


@pytest.fixture
def plans(monkeypatch):
    """The two plans, imported for one dataset and a subset of the factors."""
    import rfl.experiments.plans.mi_agg as main
    import rfl.experiments.plans.mi_agg_calib as calib

    def load(dataset, **env):
        monkeypatch.setenv("RFL_DATASET", dataset)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        importlib.reload(calib)
        importlib.reload(main)
        return calib, main
    yield load
    for k in ("RFL_DATASET", "RFL_AGG_RULES", "RFL_AGG_CLIENTS", "RFL_AGG_PARTICIPATION",
              "RFL_AGG_EPOCHS"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(calib)
    importlib.reload(main)


def _parse(job):
    if job.entry == "fed":
        from rfl.experiments.run_federated_experiment import _build_arg_parser
    else:
        from rfl.experiments.run_centralized_experiment import _build_arg_parser
    return _build_arg_parser().parse_args(job.runner_args() + ["--save-dir", "x"])


def _write_floors(dataset, archs):
    """A floors file as analyze_covariances --per-block-percentiles writes it."""
    from rfl.experiments.analyze_covariances import PERCENTILES, percentile_key
    pairs = [(a, b) for a in PERCENTILES for b in PERCENTILES]
    out = {a: {"nobn": {}, "bn": {percentile_key([q1, q2]): {"floors": [1e-4 * q1, 0.01 * q2]}
                                  for q1, q2 in pairs}} for a in archs}
    path = G.results_root() / "reeig_floors" / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": {}, "architectures": out}))


def _score(j):   # best: lr 0.1, batch 32, q1 5, q2 30, momentum 0.3, blocks 32,16
    hp = j.hp
    return (0.5 + 0.1 * (j.lr == 0.1) + 0.05 * (j.batch_size == 32)
            + 0.01 * (hp.get("--reeig-percentile") == 5)
            + 0.01 * (hp.get("--reeig-percentile-2") == 30)
            + 0.01 * (hp.get("--bn-momentum") == 0.3) + 0.03 * ("a32_16" in j.arm))


def _fake(job, val_acc):
    job.save_dir.mkdir(parents=True, exist_ok=True)
    if job.entry == "fed":
        r = {"config": {"dataset": "D", "model_type": "spdnet", "learning_rate": job.lr,
                        "batch_size": job.batch_size, "fl_rounds": job.horizon,
                        "seed": job.seed},
             "final_metrics": {"global_val_acc": val_acc, "global_test_acc": 0.5},
             "last_round_metrics": {}, "round_history": {}}
        (job.save_dir / "fl_results.json").write_text(json.dumps(r))
        return
    r = {"config": {"dataset": "D", "model_type": "spdnet", "learning_rate": job.lr,
                    "batch_size": job.batch_size, "epochs": job.horizon, "seed": job.seed},
         "training": {"report_rule": "selected", "best_val_acc": val_acc,
                      "best_val_loss": 1.0, "final_epoch": job.horizon},
         "test_metrics": {"accuracy": 0.5, "f1_macro": 0.5}}
    (job.save_dir / "centralized_results.json").write_text(json.dumps(r))


def _run_stage(plan, stage, n):
    jobs = G.plan_jobs(plan, stage, check=True)
    assert len(jobs) == n, (plan, stage, len(jobs))
    for j in jobs:
        _fake(j, _score(j))
    assert G.main([plan, "--stage", stage, "--select"]) == 0
    return jobs


def test_schedule_keeps_the_reference_rule_and_scales_with_participation_and_epochs():
    from rfl.experiments.plans.mi_agg import schedule
    assert schedule(100, 1, 52) == {"plateau": 20, "stop": 150, "budget": 3000}
    assert schedule(100, 10, 52) == {"plateau": 5, "stop": 30, "budget": 300}
    assert schedule(100, 100, 52) == {"plateau": 5, "stop": 30, "budget": 300}
    # 9 clients at 10 %: one client per round, pe = 1/9.
    assert schedule(10, 1, 9) == {"plateau": 180, "stop": 1350, "budget": 27000}
    assert schedule(10, 100, 9) == {"plateau": 45, "stop": 270, "budget": 2700}
    # 52 clients at 25 %: 13 per round.
    assert schedule(25, 1, 52) == {"plateau": 80, "stop": 600, "budget": 12000}


def test_calibration_then_every_arm_tuned_alike(plans):
    calib, main = plans("Cho2017", RFL_AGG_RULES="projavg,pseavg", RFL_AGG_CLIENTS="subj,iid",
                        RFL_AGG_PARTICIPATION="25,100", RFL_AGG_EPOCHS="1,10")
    archs = calib.STUDY["Cho2017"]["architectures"]
    with pytest.raises(SystemExit, match="no per-block ReEig floors"):
        G.plan_jobs("mi_agg_calib", "sweep1")
    assert "--per-block-percentiles" in calib.floors_command()
    _write_floors("Cho2017", archs)
    assert calib.floors_ready()
    with pytest.raises(SystemExit, match="not found"):
        G.plan_jobs("mi_agg", "sweep1")                     # calibrate first
    c1 = _run_stage("mi_agg_calib", "sweep1", 4 * 12 * 3)
    for j in c1:
        ns = _parse(j)
        assert ns.reeig_percentile == 15 and ns.reeig_percentile_2 == 15
        assert ns.bn_momentum == 0.1 and ns.epochs == 3000 and ns.early_stopping_patience == 150
    c2 = _run_stage("mi_agg_calib", "sweep2", 4 * 9 * 3)
    assert {(j.hp["--reeig-percentile"], j.hp["--reeig-percentile-2"]) for j in c2} == \
        {(a, b) for a in (15, 5, 30) for b in (15, 5, 30)}
    # The edges extend by the listed percentiles; finish the extended cells.
    from rfl.experiments.pipeline import advance
    while (todo := advance("mi_agg_calib", "sweep2", verbose=False)):
        for j in todo:
            _fake(j, _score(j))
    _run_stage("mi_agg_calib", "sweep3", 4 * 3 * 3)
    while (todo := advance("mi_agg_calib", "sweep3", verbose=False)):
        for j in todo:
            _fake(j, _score(j))
    s1 = G.plan_jobs("mi_agg", "sweep1", check=True)
    arms = {j.arm for j in s1}
    assert len(arms) == 2 * 2 * 2 * 2 and len(s1) == len(arms) * 12 * 3
    assert "fed_spdnet_pseavg_fedper_iid_p25_e10" in arms
    for j in s1:
        ns = _parse(j)
        assert ns.hidden_dims == "32,16" and ns.use_batch_norm and ns.fedbn and ns.local_fc
        assert ns.reeig_percentile == 5 and ns.bn_momentum == 0.3
        assert ns.reeig_percentile_2 == 30
        assert ns.optimizer == "sgd" and ns.stiefel == "trivialization"
        assert ns.lr_schedule == "plateau" and ns.selection_metric == "val_loss"
        assert ns.report == "selected"
        pct, e = (int(x[1:]) for x in j.arm.split("_")[-2:])
        s = main.schedule(pct, e, 52)
        assert ns.rounds == s["budget"] and ns.early_stop_patience == s["stop"]
        assert ns.lr_plateau_patience == s["plateau"]
        assert ns.client_fraction == pct / 100 and ns.local_epochs == e
        assert (ns.client_partition == "iid") == ("_iid_" in j.arm)
    for j in s1:
        _fake(j, _score(j))
    assert G.main(["mi_agg", "--stage", "sweep1", "--select"]) == 0
    final = G.plan_jobs("mi_agg", "final")
    assert len(final) == len(arms) * 10
    assert all(j.lr == 0.1 and j.batch_size == 32 for j in final)


def test_a_subset_leaves_the_other_arms_alone(plans):
    calib, main = plans("BNCI2014_001", RFL_AGG_EPOCHS="1", RFL_AGG_PARTICIPATION="100")
    assert set(main.variants(9)) == {"subj_p100_e1", "iid_p100_e1"}
    calib, main = plans("BNCI2014_001", RFL_AGG_CLIENTS="nope")
    with pytest.raises(SystemExit, match="subj or iid"):
        main.variants(9)


def test_accuracy_at_a_round_budget():
    from rfl.analysis.aggregation import accuracy_at_budgets
    h = {"round": [1, 2, 3, 4, 5], "global_val_loss": [3.0, 2.0, float("nan"), 1.0, 1.5],
         "per_client_test_acc": [[.1, .3], [.4, .4], [.9, .9], [.6, .8], [.0, .0]]}
    assert accuracy_at_budgets(h, [1, 2, 3, 4, 100]) == pytest.approx([.2, .4, .4, .7, .7])
    assert np.isnan(accuracy_at_budgets({"round": [1], "global_val_loss": [float("nan")],
                                         "per_client_test_acc": [[.5]]}, [1])[0])


def test_rounds_to_plateau_ignores_the_slow_tail_of_the_validation_loss():
    from rfl.analysis.aggregation import rounds_to_plateau
    acc = [.3, .5, .70, .72, .715, .72, .725, .72]
    h = {"round": list(range(1, 9)), "global_val_loss": list(np.linspace(2, 1, 8)),
         "per_client_test_acc": [[a] for a in acc]}
    assert rounds_to_plateau(h, tol=0.01) == 4      # within 1 point of 0.72 from round 4
    assert rounds_to_plateau(h, tol=0.03) == 3


def _fake_fed_run(root, rule, part, e, pct, seed, lr=0.1, batch=32, diverged=False,
                  rng=None):
    d = root / f"fed_spdnet_{rule}_fedper_{part}_p{pct}_e{e}" / f"lr{lr}_B{batch}" / f"seed{seed}"
    d.mkdir(parents=True)
    cfg = {"dataset": "Cho2017", "model_type": "spdnet", "fl_aggregation": rule,
           "fl_client_partition": part, "fl_client_fraction": pct / 100,
           "fl_local_epochs": e, "fl_local_fc": True, "seed": seed,
           "learning_rate": lr, "batch_size": batch}
    acc = [[0.6 + 0.1 * rng.random() for _ in range(4)] for _ in range(50)]
    loss = list(np.linspace(2, 1, 50))
    r = {"config": cfg, "diverged": diverged,
         "final_metrics": {"per_client_test_acc": acc[-1], "global_val_acc": 0.6 + lr},
         "selection": {"round": 50, "hit_budget": False},
         "round_history": {"round": list(range(1, 51)), "global_val_loss": loss,
                           "per_client_test_acc": acc}}
    (d / "fl_results.json").write_text(json.dumps(r))


def test_summary_table_and_figures(tmp_path):
    from rfl.analysis.aggregation import budget_table, latex, lr_table, main, summary
    from rfl.analysis.results import load_runs
    rng = np.random.default_rng(0)
    final, sweep = tmp_path / "final", tmp_path / "sweep1"
    for rule in ("projavg", "rlavg", "retavg", "pseavg"):
        for part in ("subject", "iid"):
            for e in (1, 10):
                for pct in (25, 100):
                    for seed in (101, 102, 103):
                        _fake_fed_run(final, rule, part, e, pct, seed, rng=rng,
                                      diverged=rule == "retavg" and e == 10 and seed == 101)
                    for lr in (0.01, 0.1, 0.3):
                        for batch in (16, 32):
                            for seed in (1, 2):
                                _fake_fed_run(sweep, rule, part, e, pct, seed, lr, batch,
                                              diverged=rule == "pseavg" and lr == 0.3,
                                              rng=rng)
    runs = load_runs(str(final))
    s = summary(runs, "client_mean_acc")
    assert len(s) == 4 * 2 * 2 * 2 and set(s["n"]) == {3}
    row = s[(s["rule"] == "retavg") & (s["local_epochs"] == 10)]
    assert set(row["n_diverged"]) == {1}
    assert "$^{1}$" in latex(s) and r"\textbf" in latex(s)
    b = budget_table(runs, [10, 50])
    assert len(b) == 2 * len(s) and set(b[b["rule"] == "projavg"]["n"]) == {3}
    lr = lr_table(load_runs(str(sweep)))
    pse = lr[(lr["rule"] == "pseavg") & (lr["lr"] == 0.3)]
    assert set(pse["diverged_share"]) == {1.0} and pse["best_val_acc"].isna().all()
    assert set(lr[lr["rule"] == "projavg"]["best_val_acc"].round(2)) == {0.61, 0.7, 0.9}
    assert main([str(final), "--sweep", str(sweep), "--out", str(tmp_path / "agg"),
                 "--report"]) == 0
    tex = (tmp_path / "agg_report.tex").read_text()
    assert r"\input{agg_tests.tex}" in tex and r"agg_lr_div.pdf" in tex
    assert "fewer than 6 pairs" in (tmp_path / "agg_tests.csv").read_text()
    for suffix in (".pdf", "_budget.pdf", "_lr_acc.pdf", "_lr_div.pdf", "_lr.csv",
                   "_rounds.tex", "_tests.tex"):
        assert (tmp_path / f"agg{suffix}").stat().st_size > 0
