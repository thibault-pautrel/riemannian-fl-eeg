#!/usr/bin/env python3
"""Results of the aggregation study (plan mi_agg): rules x clients x participation x E.

For each cell (client partition, local epochs E, participation) and each rule, over the
final seeds: mean and standard deviation of the metric (client_mean_acc by default:
every subject weighs the same), the number of diverged runs (a RetAvg or PseAvg client
outside the domain of the inverse map), the median selected round (communication
rounds to the best validation loss) and the runs that hit the round budget. A diverged
run has no accuracy: it counts in `n_diverged`, not in the mean.

Writes <out>.csv (one row per cell and rule), <out>.tex (one row per cell, one column
per rule) and <out>.pdf (accuracy against participation, one line per rule, one panel
per partition and E). The paired tests between rules are those of rfl/analysis/stats.py:

    python -m rfl.analysis.aggregation $RFL_RESULTS/mi_Cho2017_agg/final --out agg_Cho2017
    python -m rfl.analysis.stats $RFL_RESULTS/mi_Cho2017_agg/final --metric client_mean_acc \\
        --factor aggregation --baseline projavg --by clients local_epochs participation
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from rfl.analysis.report import apply_filters
from rfl.analysis.results import load_runs

RULES = ("projavg", "rlavg", "retavg", "pseavg")
CELL = ["clients", "local_epochs", "participation"]
# Categorical slots 1-4 of the dataviz reference palette (validated, light mode), with
# one marker per rule so that identity never rests on color alone.
STYLE = {"projavg": ("#257de0", "o"), "rlavg": ("#f47230", "s"),
         "retavg": ("#00b894", "^"), "pseavg": ("#f2a600", "D")}


LABELS = {"client_mean_acc": "mean client accuracy", "test_acc": "pooled test accuracy"}


def summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per cell and rule (module docstring)."""
    df = df[df["kind"] == "federated"].copy()
    df["local_epochs"] = df["local_epochs"].astype(int)
    rows = []
    for key, g in df.groupby(CELL + ["aggregation"], dropna=False):
        ok = g[~g["diverged"]]
        rows.append(dict(zip(CELL + ["rule"], key), n=len(g),
                         n_diverged=int(g["diverged"].sum()),
                         mean=ok[metric].mean(), std=ok[metric].std(ddof=1),
                         median_round=ok["sel_round"].median(),
                         n_hit_budget=int(ok["hit_budget"].fillna(False).sum())))
    out = pd.DataFrame(rows)
    order = {r: i for i, r in enumerate(RULES)}
    out["_o"] = out["rule"].map(lambda r: order.get(r, len(order)))
    return out.sort_values(CELL + ["_o"]).drop(columns="_o").reset_index(drop=True)


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


def figure(s: pd.DataFrame, path: str, metric: str) -> None:
    """Accuracy against participation, one line per rule, one panel per (clients, E).
    The legend sits under the panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = list(dict.fromkeys(s["clients"]))
    epochs = sorted(set(s["local_epochs"]))
    fig, axes = plt.subplots(len(parts), len(epochs), squeeze=False, sharey=True,
                             figsize=(3.2 * len(epochs), 2.6 * len(parts)))
    for i, c in enumerate(parts):
        for j, e in enumerate(epochs):
            ax = axes[i][j]
            rules = [r for r in RULES if r in set(s["rule"])]
            for k, r in enumerate(rules):
                g = s[(s["clients"] == c) & (s["local_epochs"] == e) & (s["rule"] == r)]
                g = g.sort_values("participation")
                color, marker = STYLE[r]
                dodge = 1.5 * (k - (len(rules) - 1) / 2)     # error bars side by side
                ax.errorbar(100 * g["participation"] + dodge, 100 * g["mean"],
                            yerr=100 * g["std"], color=color, marker=marker, markersize=5,
                            linewidth=1.5, elinewidth=0.8, capsize=0, label=r)
            ax.set_xticks(sorted({round(100 * x) for x in s["participation"]}))
            ax.set_title(f"{'subject' if c == 'subject' else c} clients, E = {e}",
                         fontsize=9)
            ax.grid(alpha=0.3, linewidth=0.5)
            if i == len(parts) - 1:
                ax.set_xlabel("participation (%)")
            if j == 0:
                ax.set_ylabel(LABELS.get(metric, metric) + " (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="Directories searched for result files.")
    p.add_argument("--metric", default="client_mean_acc")
    p.add_argument("--where", nargs="+", default=[], metavar="COL=VALUE")
    p.add_argument("--out", default=None, help="Prefix of the .csv, .tex and .pdf files.")
    a = p.parse_args(argv)
    s = summary(apply_filters(load_runs(*a.roots), a.where), a.metric)
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(s.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if a.out:
        s.to_csv(f"{a.out}.csv", index=False)
        with open(f"{a.out}.tex", "w") as f:
            f.write(latex(s))
        figure(s, f"{a.out}.pdf", a.metric)
        print(f"written: {a.out}.csv, {a.out}.tex, {a.out}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
