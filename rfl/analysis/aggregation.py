#!/usr/bin/env python3
"""Results of the aggregation study (plan mi_agg): rules x clients x participation x E.

Three outputs, written with --out PREFIX:

1. At convergence (final runs). For each cell (client partition, local epochs E,
   participation) and each rule, over the final seeds: mean and standard deviation of the
   metric (client_mean_acc by default: every client weighs the same), the number of
   diverged runs (a RetAvg or PseAvg client outside the domain of the inverse map), the
   median selected round and the runs that hit the round budget. A diverged run has no
   accuracy: it counts in `n_diverged`, not in the mean. `plateau_round` is the median
   number of rounds after which the accuracy the run would report stays within 1 point
   of its final value (`rounds_to_plateau`). The selected round itself can come much
   later: the validation loss keeps falling slowly on the plateau.
   PREFIX.csv, PREFIX.tex (one row per cell, one column per rule), PREFIX.pdf.

2. At a fixed communication budget (final runs, --budgets). The accuracy a run would
   report had it been stopped after B rounds: the client mean test accuracy at its round
   of lowest validation loss among the first B rounds. A run that stopped before B keeps
   its reported model. PREFIX_budget.csv and PREFIX_budget.pdf (accuracy against B).

3. Sensitivity to the learning rate (sweep runs, --sweep ROOT). For each cell, rule and
   learning rate: the mean validation accuracy over the seeds of the best batch size,
   and the share of diverged runs over every batch size. PREFIX_lr.csv,
   PREFIX_lr_acc.pdf and PREFIX_lr_div.pdf.

With --report, PREFIX_report.tex gathers the tables, the paired tests against ProjAvg
(Wilcoxon, Holm correction within each cell, rfl/analysis/stats.py) and the figures,
and is compiled to PREFIX_report.pdf when pdflatex is available.

    R=$RFL_RESULTS/mi_Cho2017_agg
    python -m rfl.analysis.aggregation $R/final --sweep $R/sweep1 --out agg_Cho2017 --report
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
    # A synthetic IID federation is a 'subject' partition of IID data: the arm names it.
    iid = df["arm"].fillna("").str.contains("_iid_")
    df.loc[iid, "clients"] = "iid"
    df["local_epochs"] = df["local_epochs"].astype(int)
    df["rule"] = df["aggregation"]
    return df


def _sort(out: pd.DataFrame, keys: list) -> pd.DataFrame:
    order = {r: i for i, r in enumerate(RULES)}
    out["_o"] = out["rule"].map(lambda r: order.get(r, len(order)))
    return out.sort_values(CELL + ["_o"] + keys).drop(columns="_o").reset_index(drop=True)


# ---------------------------------------------------------------- 1. convergence
def _history(path: str) -> dict:
    with open(os.path.join(path, "fl_results.json")) as f:
        return json.load(f).get("round_history") or {}


def _reported_accuracy(history: dict) -> tuple:
    """(rounds, accuracy the run would report if stopped at each round): the client mean
    test accuracy at the round of lowest validation loss so far."""
    rounds = list(history.get("round") or [])
    loss = list(history.get("global_val_loss") or [])
    accs = [np.mean(a) if a else np.nan for a in history.get("per_client_test_acc") or []]
    best, out = None, []
    for i in range(min(len(rounds), len(loss), len(accs))):
        if np.isfinite(loss[i]) and (best is None or loss[i] < loss[best]):
            best = i
        out.append(accs[best] if best is not None else np.nan)
    return rounds[:len(out)], out


def rounds_to_plateau(history: dict, tol: float = 0.01) -> float:
    """First round from which the reported accuracy stays within `tol` of its final
    value. NaN without history."""
    rounds, acc = _reported_accuracy(history)
    if not acc or not np.isfinite(acc[-1]):
        return np.nan
    first = len(acc) - 1
    while first > 0 and np.isfinite(acc[first - 1]) and abs(acc[first - 1] - acc[-1]) <= tol:
        first -= 1
    return float(rounds[first])


def summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per cell and rule (module docstring, output 1)."""
    rows = []
    for key, g in _federated(df).groupby(CELL + ["rule"], dropna=False):
        ok = g[~g["diverged"]]
        plateau = [rounds_to_plateau(_history(p)) for p in ok["path"]]
        rows.append(dict(zip(CELL + ["rule"], key), n=len(g),
                         n_diverged=int(g["diverged"].sum()),
                         mean=ok[metric].mean(), std=ok[metric].std(ddof=1),
                         plateau_round=np.nanmedian(plateau) if plateau else np.nan,
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


def latex_rounds(s: pd.DataFrame) -> str:
    """One row per cell, one column per rule: median rounds to the plateau, then in
    parentheses the median selected round, and as a superscript the runs that hit the
    round budget."""
    rules = [r for r in RULES if r in set(s["rule"])]
    lines = [r"\begin{tabular}{ll r" + "c" * len(rules) + "}", r"\toprule",
             "Clients & $E$ & $p$ (\\%) & " + " & ".join(rules) + r" \\", r"\midrule"]
    for (clients, e, p), g in s.groupby(CELL, sort=False):
        by = g.set_index("rule")
        cells = []
        for r in rules:
            if r not in by.index or np.isnan(by.loc[r, "plateau_round"]):
                cells.append("--")
                continue
            txt = f"{by.loc[r, 'plateau_round']:.0f} ({by.loc[r, 'median_round']:.0f})"
            hb = int(by.loc[r, "n_hit_budget"])
            cells.append(txt + (rf"$^{{{hb}}}$" if hb else ""))
        lines.append(f"{clients} & {e} & {100 * p:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def paired_tests(df: pd.DataFrame, metric: str, baseline: str = "projavg") -> pd.DataFrame:
    """Every rule against `baseline` within each cell, paired on the seeds
    (rfl/analysis/stats.py: Wilcoxon signed-rank, Holm correction within the cell)."""
    from rfl.analysis.stats import paired_pairwise
    d = _federated(df)
    d = d[~d["diverged"]]
    return paired_pairwise(d, metric, "aggregation", CELL, baseline=baseline)


def latex_tests(pw: pd.DataFrame) -> str:
    """One row per cell and rule: mean paired difference with the baseline (points),
    share of seeds where the rule wins, Holm-adjusted p-value (bold below 0.05)."""
    lines = [r"\begin{tabular}{ll r l r r r r}", r"\toprule",
             r"Clients & $E$ & $p$ (\%) & Rule & pairs & $\Delta$ (pt) & wins & $p_{\mathrm{Holm}}$ \\",
             r"\midrule"]
    for _, r in pw.iterrows():
        if not np.isfinite(r.get("p_value", np.nan)):
            diff, wins, ph = "--", "--", "--"
        else:
            diff = f"{100 * r['mean_diff']:+.1f}"
            wins = f"{100 * r['win_rate']:.0f}\\%"
            ph = f"{r['p_holm']:.3f}"
            ph = rf"\textbf{{{ph}}}" if r["p_holm"] < 0.05 else ph
        lines.append(f"{r['clients']} & {r['local_epochs']} & {100 * r['participation']:g} & "
                     f"{r['level']} & {r['n_pairs']} & {diff} & {wins} & {ph} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 2. fixed budget
def accuracy_at_budgets(history: dict, budgets: Sequence[int]) -> list:
    """Client mean test accuracy at the round of lowest validation loss among the first
    B rounds, for each B. NaN when no round of the first B has a finite loss."""
    rounds, acc = _reported_accuracy(history)
    out = []
    for b in budgets:
        n = sum(1 for r in rounds if r <= b)
        out.append(float(acc[n - 1]) if n else np.nan)
    return out


def budget_table(df: pd.DataFrame, budgets: Sequence[int]) -> pd.DataFrame:
    """One row per cell, rule and budget (module docstring, output 2)."""
    rows = []
    for _, run in _federated(df).iterrows():
        accs = [np.nan] * len(budgets)
        if not run["diverged"]:
            accs = accuracy_at_budgets(_history(run["path"]), budgets)
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


# ---------------------------------------------------------------- report
REPORT = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs, graphicx, caption}
\captionsetup{font=small}
\title{Aggregation study: %(title)s}
\date{\today}
\begin{document}
\maketitle

Two-block SPDNet, personal head (FedPer) and local batch norms (FedBN): the server
averages the BiMap weights $W_1$ and $W_2$ only. Factors: aggregation rule, client
partition (one subject per client or IID clients), participation $p$ per round and
local epochs $E$. Learning rate and batch size are tuned per arm on the validation
accuracy, then %(seeds)s final seeds are run. Each run reports the model of its round of
lowest pooled validation loss. Metric: %(metric)s.

\section{Accuracy at convergence}
\begin{table}[h]\centering\small
\caption{%(metric_cap)s (\%%), mean (standard deviation) over the final seeds. Best rule
of each row in bold. Superscript: diverged runs, left out of the mean.}
\input{%(base)s.tex}
\end{table}

\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{%(base)s.pdf}
\caption{%(metric_cap)s against the participation, one panel per partition and $E$
(mean and standard deviation over the final seeds).}
\end{figure}

\begin{table}[h]\centering\small
\caption{Rounds to the plateau: first round after which the reported accuracy stays within
one point of its final value (median over the seeds). In parentheses: median selected round
(lowest validation loss). Superscript: runs that hit the round budget.}
\input{%(base)s_rounds.tex}
\end{table}

\clearpage
\section{Paired comparisons with ProjAvg}
\begin{table}[h]\centering\small
\caption{Each rule against ProjAvg within each cell, paired on the seeds (same splits and
same sampled clients). $\Delta$: mean difference in points. Wins: share of seeds where the
rule is better. Wilcoxon signed-rank test, Holm correction within the cell, at least 6
pairs (bold: $p < 0.05$).}
\input{%(base)s_tests.tex}
\end{table}

\clearpage
\section{Accuracy at a fixed communication budget}
\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{%(base)s_budget.pdf}
\caption{Accuracy each run would report if stopped after $B$ rounds (model of lowest
validation loss among the first $B$ rounds), mean over the final seeds.}
\end{figure}
%(lr)s
\end{document}
"""
REPORT_LR = r"""
\clearpage
\section{Sensitivity to the learning rate}
\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{%(base)s_lr_acc.pdf}
\caption{Tuning runs: mean validation accuracy over the seeds, best batch size, against
the learning rate.}
\end{figure}
\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{%(base)s_lr_div.pdf}
\caption{Tuning runs: share of diverged runs against the learning rate (every batch size).}
\end{figure}
"""


def write_report(prefix: str, s: pd.DataFrame, pw: pd.DataFrame, metric: str, title: str,
                 seeds: int, with_lr: bool) -> str:
    """PREFIX_report.tex, compiled with pdflatex when available. Returns the .tex path."""
    import shutil
    import subprocess
    out_dir, base = os.path.dirname(os.path.abspath(prefix)), os.path.basename(prefix)
    with open(f"{prefix}_rounds.tex", "w") as f:
        f.write(latex_rounds(s))
    with open(f"{prefix}_tests.tex", "w") as f:
        f.write(latex_tests(pw))
    cap = LABELS.get(metric, metric).capitalize()
    tex = REPORT % {"title": title.replace("_", r"\_"), "base": base, "seeds": seeds,
                    "metric": LABELS.get(metric, metric), "metric_cap": cap,
                    "lr": REPORT_LR % {"base": base} if with_lr else ""}
    path = f"{prefix}_report.tex"
    with open(path, "w") as f:
        f.write(tex)
    if shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            os.path.basename(path)], cwd=out_dir,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path


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
    p.add_argument("--report", action="store_true",
                   help="Also write PREFIX_report.tex and compile it (needs --out).")
    p.add_argument("--title", default=None, help="Title of the report (default: dataset).")
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
        if a.report:
            pw = paired_tests(runs, a.metric)
            pw.to_csv(f"{a.out}_tests.csv", index=False)
            title = a.title or ", ".join(sorted(map(str, runs["dataset"].dropna().unique())))
            seeds = int(s["n"].max()) if len(s) else 0
            tex = write_report(a.out, s, pw, a.metric, title, seeds, lr is not None)
            pdf = tex[:-4] + ".pdf"
            written += [tex] + ([pdf] if os.path.exists(pdf) else [])
        print("written: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
