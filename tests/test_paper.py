"""rfl/analysis/paper.py on a small fake results folder."""
import json

import numpy as np

from rfl.analysis import paper

SUBJECTS = [1, 2, 3]


def _run(root, plan, arm, seed, cfg, acc, refit=None, central=False):
    d = root / plan / "final" / arm / "hp" / f"seed{seed}"
    d.mkdir(parents=True)
    cfg = {"dataset": "DS", "seed": seed, "learning_rate": 0.1, "batch_size": 8, **cfg}
    if central:
        (d / "centralized_results.json").write_text(json.dumps(
            {"config": cfg, "test_metrics": {"accuracy": float(np.mean(acc))}}))
        return
    res = {"config": cfg,
           "fl_info": {"client_subjects": [[s] for s in SUBJECTS]},
           "final_metrics": {"per_client_test_acc": acc,
                             "global_test_acc": float(np.mean(acc))},
           "personalized_metrics": ({"personalized_per_client_test_acc": refit}
                                    if refit else None),
           "round_history": {"avg_test_acc": list(np.linspace(0.25, np.mean(acc), 50))}}
    (d / "fl_results.json").write_text(json.dumps(res))


def _fake(root, rng):
    norms = {"nobn": {}, "bn": {"use_bn": True}, "bnshared": {"use_bn": True, "fl_fedbn": False}}
    scopes = {"local": {"fl_local_only": True}, "fedper": {"fl_local_fc": True},
              "shared": {"fl_head_refit_epochs": "25,50"}}

    def net(model, norm):
        bn = norms[norm].get("use_bn", False)
        cfg = {"model_type": model, "fl_fedbn": norms[norm].get("fl_fedbn", True)}
        if model == "eegnet":
            cfg["eegnet_norm_type"] = "batch" if bn else "none"
        else:
            cfg.update(spdnet_hidden_dims="4", spdnet_reeig_percentile=15,
                       spdnet_use_batch_norm=bn)
        return cfg

    for model, plan in (("spdnet", "mi_DS_bn"), ("eegnet", "mi_DS_bn_eegnet")):
        for seed in (1, 2, 3):
            for norm in norms:
                for scope, s_cfg in scopes.items():
                    if scope == "local" and norm == "bnshared":
                        continue
                    acc = list(rng.uniform(0.4, 0.8, 3))
                    refit = list(rng.uniform(0.4, 0.8, 3)) if scope == "shared" else None
                    _run(root, plan, f"fed_{model}_{scope}_{norm}", seed,
                         {**net(model, norm), **s_cfg}, acc, refit)
                if norm != "bnshared":
                    _run(root, plan, f"central_{model}_{norm}", seed, net(model, norm),
                         list(rng.uniform(0.4, 0.8, 3)), central=True)
            for n in (5, 10, None):
                tag = f"U{n}" if n else "full"
                for norm in ("nobn", "bn"):
                    _run(root, f"mi_DS_datasize{'_eegnet' if model == 'eegnet' else ''}",
                         f"fed_{model}_fedper_{norm}_{tag}", seed,
                         {**net(model, norm), "fl_local_fc": True,
                          "train_trials_per_class": n}, list(rng.uniform(0.3, 0.7, 3)))


def test_paper_writes_every_table_and_figure(tmp_path):
    _fake(tmp_path / "res", np.random.default_rng(0))
    out = tmp_path / "out"
    out.mkdir()
    (out / "tab_stale.tex").write_text("old")
    paper.main(["DS", "--results", str(tmp_path / "res"), "--out", str(out)])
    assert not (out / "tab_stale.tex").exists()
    names = {p.name for p in out.iterdir()}
    assert {"tab_bn.tex", "tab_bn_clients_spdnet.tex", "tab_bn_clients_eegnet.tex",
                     "fig_bn_rounds.tex", "tab_datasize.tex", "fig_datasize.tex",
                     "clients.csv", "results.tex"} <= names
    doc = (out / "results.tex").read_text()
    assert doc.count("\\input{") == 6 and "\\section{Data quantity}" in doc
    bn = (out / "tab_bn.tex").read_text()
    assert "Shared head &" in bn and "re-fitted" not in bn
    assert "Centralised" in bn and "\\mathbf" in bn
    per = (out / "tab_bn_clients_spdnet.tex").read_text()
    assert all(f"\n{s} &" in per for s in SUBJECTS) and "Mean &" in per
    ds = (out / "tab_datasize.tex").read_text()
    assert "\\emph{SPDNet}" in ds and "TS+LR" not in ds and "& all \\\\" in ds
    assert "xticklabels={5,10,all}" in (out / "fig_datasize.tex").read_text()
    rounds = (out / "fig_bn_rounds.tex").read_text()
    assert "\\addlegendentry{FedPer}" in rounds and "\\addlegendentry{Local}" not in rounds
