#!/usr/bin/env python3
"""Results of the aggregation study (plan mi_agg): rules x clients x participation x E.

Three outputs, written with --out PREFIX:

1. At convergence (final runs). For each cell (client partition, local epochs E,
   participation) and each rule, over the final seeds: mean and standard deviation of the
   metric (client_mean_acc by default: every client weighs the same), the number of
   diverged runs (a RetAvg or PseAvg client outside the domain of the inverse map), the
   median selected round and the runs that hit the round budget. A diverged run has no
   accuracy: it counts in `n_diverged`, not in the mean.
   PREFIX.csv, PREFIX.tex (one row per cell, one column per rule), PREFIX.pdf.

2. At a fixed communication budget (final runs, --budgets). The accuracy a run would
   report had it been stopped after B rounds: the client mean test accuracy at its round
   of lowest validation loss among the first B rounds. A run that stopped before B keeps
   its reported model. PREFIX_budget.csv and PREFIX_budget.pdf (accuracy against B).

3. Sensitivity to the learning rate (sweep runs, --sweep ROOT). For each cell, rule and
   learning rate: the mean validation accuracy over the seeds of the best batch size,
   and the share of diverged runs over every batch size. PREFIX_lr.csv,
   PREFIX_lr_acc.pdf and PREFIX_lr_div.pdf.

    R=$RFL_RESULTS/mi_Cho2017_agg
    python -m rfl.analysis.aggregation $R/final --sweep $R/sweep1 --out agg_Cho2017
    python -m rfl.analysis.stats $R/final --metric client_mean_acc \\
        --factor aggregation --baseline projavg --by clients local_epochs participation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from rfl.analysis.report import apply_filters
from rfl.analysis.results import load_runs
from rfl.utils.schedules import best_position

RULES = ("projavg", "rlavg", "retavg", "pseavg")
CELL = ["clients", "local_epochs", "participation"]
BUDGETS = (25, 50, 100, 200, 500, 1000, 3000)
# Categorical slots 1-4 of the dataviz reference palette (validated, light mode), with
# one marker per rule so that identity never rests on color alone.
STYLE = {"projavg": ("#257de0", "o"), "rlavg": ("#f47230", "s"),
         "retavg": ("#00b894", "^"), "pseavg": ("#f2a600", "D")}
LABELS = {"client_mean_acc": "mean client accuracy", "test_acc": "pooled test accuracy"}


def _federated(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["kind"] == "federated"].copy()
    df["local_epochs"] = df["local_epochs"].astype(int)
    df["rule"] = df["aggregation"]
    return df


def _sort(out: pd.DataFrame, keys: list) -> pd.DataFrame:
    order = {r: i for i, r in enumerate(RULES)}
    out["_o"] = out["rule"].map(lambda r: order.get(r, len(order)))
    return out.sort_values(CELL + ["_o"] + keys).drop(columns="_o").reset_index(drop=True)


# ---------------------------------------------------------------- 1. convergence
def summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per cell and rule (module docstring, output 1)."""
    rows = []
    for key, g in _federated(df).groupby(CELL + ["rule"], dropna=False):
        ok = g[~g["diverged"]]
        rows.append(dict(zip(CELL + ["rule"], key), n=len(g),
                         n_diverged=int(g["diverged"].sum()),
                         mean=ok[metric].mean(), std=ok[metric].std(ddof=1),
                         median_round=ok["sel_round"].median(),
                         n_hit_budget=int(ok["hit_budget"].fillna(False).sum())))
    return _sort(pd.DataFrame(rows), [])


def latex(s: pd.DataFrame) -> str:
    """One row per cell, one column per rule: mean (sd) in %, the best mean in bold,
    the diverged runs as a superscript."""
    rules = [r for r in RULES if r in set(s["rule"])]
    lines = [r"\begin{tabular}{ll r" + "c" * len(rules) + "}", r"\toprule",
             "Clients & $E$ & $p$ (\\%) & " + " & ".join(rules) + r" \\", r"\midrule"]
    for key, g in s.groupby(CELL, sort=False):
        by = g.set_index("rule")
        best = by["mean"].max()
        cells = []
        for r in rules:
            if r not in by.index or np.isnan(by.loc[r, "mean"]):
                cells.append("--")
                continue
            m, sd, nd = by.loc[r, "mean"], by.loc[r, "std"], by.loc[r, "n_diverged"]
            txt = f"{100 * m:.1f} ({100 * sd:.1f})"
            txt = rf"\textbf{{{txt}}}" if m == best else txt
            cells.append(txt + (rf"$^{{{nd}}}$" if nd else ""))
        clients, e, p = key
        lines.append(f"{clients} & {e} & {100 * p:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 2. fixed budget
def accuracy_at_budgets(history: dict, budgets: Sequence[int]) -> list:
    """Client mean test accuracy at the round of lowest validation loss among the first
    B rounds, for each B. NaN when no round of the first B has a finite loss."""
    rounds = list(history.get("round") or [])
    loss = list(history.get("global_val_loss") or [])
    accs = [np.mean(a) if a else np.nan for a in history.get("per_client_test_acc") or []]
    out = []
    for b in budgets:
        n = sum(1 for r in rounds if r <= b)
        pos = best_position(loss[:n], True)
        out.append(float(accs[pos]) if pos is not None and pos < len(accs) else np.nan)
    return out


def budget_table(df: pd.DataFrame, budgets: Sequence[int]) -> pd.DataFrame:
    """One row per cell, rule and budget (module docstring, output 2)."""
    rows = []
    for _, run in _federated(df).iterrows():
        accs = [np.nan] * len(budgets)
        if not run["diverged"]:
            with open(os.path.join(run["path"], "fl_results.json")) as f:
                accs = accuracy_at_budgets(json.load(f).get("round_history") or {}, budgets)
        rows += [dict(**{k: run[k] for k in CELL + ["rule", "seed"]}, budget=b, acc=a)
                 for b, a in zip(budgets, accs)]
    runs = pd.DataFrame(rows)
    out = (runs.groupby(CELL + ["rule", "budget"])["acc"]
           .agg(n="count", mean="mean", std=lambda x: x.std(ddof=1)).reset_index())
    return _sort(out, ["budget"])


# ---------------------------------------------------------------- 3. learning rate
def lr_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cell, rule and learning rate (module docstring, output 3)."""
    rows = []
    for key, g in _federated(df).groupby(CELL + ["rule", "lr"], dropna=False):
        per_batch = g[~g["diverged"]].groupby("batch_size")["val_acc"].mean()
        rows.append(dict(zip(CELL + ["rule", "lr"], key), n=len(g),
                         diverged_share=float(g["diverged"].mean()),
                         best_val_acc=per_batch.max() if len(per_batch) else np.nan,
                         best_batch=per_batch.idxmax() if len(per_batch) else np.nan))
    return _sort(pd.DataFrame(rows), ["lr"])


# ---------------------------------------------------------------- figures
def _panels(data: pd.DataFrame, x: str, y: str, path: str, xlabel: str, ylabel: str,
            by_participation: bool, logx: bool = False, scale: float = 100.0,
            err: str | None = None) -> None:
    """One line per rule. Panels: rows = client partitions and columns = E, or with
    `by_participation` rows = participation and columns = (partition, E). The legend
    sits under the panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [(c, e) for c in dict.fromkeys(data["clients"]) for e in sorted(set(data["local_epochs"]))]
    if by_participation:
        rows = sorted(set(data["participation"]))
        sel = [[(lambda d, c=c, e=e, p=p: d[(d["clients"] == c) & (d["local_epochs"] == e)
                                           & (d["participation"] == p)],
                 f"{c}, E = {e}, p = {100 * p:g} %") for c, e in cols] for p in rows]
    else:
        parts = list(dict.fromkeys(data["clients"]))
        epochs = sorted(set(data["local_epochs"]))
        sel = [[(lambda d, c=c, e=e: d[(d["clients"] == c) & (d["local_epochs"] == e)],
                 f"{c} clients, E = {e}") for e in epochs] for c in parts]
    nr, nc = len(sel), len(sel[0])
    fig, axes = plt.subplots(nr, nc, squeeze=False, sharey=True,
                             figsize=(3.2 * nc, 2.5 * nr))
    rules = [r for r in RULES if r in set(data["rule"])]
    for i, row in enumerate(sel):
        for j, (pick, title) in enumerate(row):
            ax = axes[i][j]
            sub = pick(data)
            for k, r in enumerate(rules):
                g = sub[sub["rule"] == r].sort_values(x)
                color, marker = STYLE.get(r, ("#666666", "o"))
                xs = g[x].to_numpy(dtype=float)
                if not logx and x == "participation":
                    xs = 100 * xs + 1.5 * (k - (len(rules) - 1) / 2)    # dodge
                ax.errorbar(xs, scale * g[y], yerr=None if err is None else scale * g[err],
                            color=color, marker=marker, markersize=4, linewidth=1.5,
                            elinewidth=0.8, capsize=0, label=r)
            if logx:
                ax.set_xscale("log")
            elif x == "participation":
                ax.set_xticks(sorted({round(100 * v) for v in data["participation"]}))
            ax.set_title(title, fontsize=8)
            ax.grid(alpha=0.3, linewidth=0.5)
            if i == nr - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.6 / (2.5 * nr), 1, 1))
    fig.savefig(path)
    plt.close(fig)


def figure(s: pd.DataFrame, path: str, metric: str) -> None:
    """Accuracy at convergence against participation."""
    _panels(s, "participation", "mean", path, "participation (%)",
            LABELS.get(metric, metric) + " (%)", by_participation=False, err="std")


# ---------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="Directories of the final runs.")
    p.add_argument("--sweep", nargs="+", default=[], metavar="ROOT",
                   help="Directories of the sweep runs, for the learning-rate outputs.")
    p.add_argument("--metric", default="client_mean_acc")
    p.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS),
                   help="Round budgets of output 2.")
    p.add_argument("--where", nargs="+", default=[], metavar="COL=VALUE")
    p.add_argument("--out", default=None, help="Prefix of the output files.")
    a = p.parse_args(argv)
    fmt = {"display.width": 160, "display.max_rows": None}
    runs = apply_filters(load_runs(*a.roots), a.where)
    s = summary(runs, a.metric)
    b = budget_table(runs, a.budgets)
    lr = lr_table(apply_filters(load_runs(*a.sweep), a.where)) if a.sweep else None
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        print(s.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print("\nAccuracy at a fixed round budget (client mean test accuracy):")
        print(b.pivot_table(index=CELL + ["rule"], columns="budget", values="mean")
              .to_string(float_format=lambda x: f"{100 * x:.1f}"))
        if lr is not None:
            print("\nLearning rate: best mean validation accuracy, share of diverged runs:")
            print(lr.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    if a.out:
        s.to_csv(f"{a.out}.csv", index=False)
        with open(f"{a.out}.tex", "w") as f:
            f.write(latex(s))
        figure(s, f"{a.out}.pdf", a.metric)
        b.to_csv(f"{a.out}_budget.csv", index=False)
        _panels(b, "budget", "mean", f"{a.out}_budget.pdf", "round budget",
                LABELS["client_mean_acc"] + " (%)", by_participation=True, logx=True)
        written = [f"{a.out}{x}" for x in (".csv", ".tex", ".pdf", "_budget.csv",
                                           "_budget.pdf")]
        if lr is not None:
            lr.to_csv(f"{a.out}_lr.csv", index=False)
            _panels(lr, "lr", "best_val_acc", f"{a.out}_lr_acc.pdf", "learning rate",
                    "best mean validation accuracy (%)", by_participation=True, logx=True)
            _panels(lr, "lr", "diverged_share", f"{a.out}_lr_div.pdf", "learning rate",
                    "diverged runs (%)", by_participation=True, logx=True)
            written += [f"{a.out}{x}" for x in ("_lr.csv", "_lr_acc.pdf", "_lr_div.pdf")]
        print("written: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
