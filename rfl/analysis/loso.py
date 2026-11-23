"""Tables, figure and paired tests of the inter-subject study (rfl/experiments/loso.py).

    python -m rfl.analysis.paper BNCI2014_001        # writes them with the other studies

Reads the final runs of mi_<dataset>_loso, _loso_eegnet and _loso_tslr. The unit is the
test accuracy of one target subject. Every subject is a target once per seed (in its
fold), so a seed gives one accuracy per subject: the tables average them over the
subjects, then give the mean and standard deviation over the seeds, as the other
tables of rfl/analysis/paper.py.

    tab_loso.tex                  rows: source arm x adaptation, columns: N, with the
                                  within-subject accuracy of the arm
    fig_loso.tex                  accuracy against N, one panel per network and source
                                  arm, with the local references
    tab_loso_subjects_<model>_N<n>.tex   one row per target subject, at the primary
                                  statistics (PRIMARY)
    tab_loso_tests.tex, loso_tests.csv   the primary paired tests (see `comparisons`)

A seed enters a mean only when every target subject has a result for it: a missing or
diverged run would otherwise change the set of subjects behind the mean. Such seeds
are dropped, with a warning, and the tables show the number of seeds left.

A row of the long table (`records`) is one (run, target, N, stats, head):

    source   central | shared | fedper (the arms trained on the sources), local (the
             target trained alone on its N trials), tslr
    norm     bn (FedBN, the centralised batch norm) | shared (shared batch norm) | ''
    stats    none | pool | labeled (the target statistics, or the TS+LR re-centring)
    head     shared | shared_ft | local (networks), scratch (local arms), pooled |
             pooled_plus | local (TS+LR)
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MODELS = {"spdnet": "SPDNet", "eegnet": "EEGNet", "tslr": "TS+LR"}
SOURCES = {"central": "Centralised", "shared": "Shared head", "fedper": "FedPer",
           "local": "Local from scratch", "tslr": "TS+LR"}
NORMS = {"bn": "FedBN", "shared": "shared BN", "": ""}
HEADS = {"shared": "as trained", "shared_ft": "shared, refitted", "local": "new head",
         "scratch": "", "pooled": "sources", "pooled_plus": "sources + target"}
TSLR_HEADS = {**HEADS, "local": "target only"}
STATS = {"none": "none", "pool": "pool", "labeled": "labeled", "": ""}
ORDER = {"source": list(SOURCES), "norm": ["bn", "shared", ""],
         "head": ["shared", "shared_ft", "local", "scratch", "pooled", "pooled_plus"],
         "stats": ["none", "pool", "labeled", ""]}
# The target statistics of the primary comparisons: the unlabeled pool when there is
# no label (Kobler et al.), the N labeled trials otherwise (a cued calibration).
PRIMARY = {0: "pool"}
PRIMARY_LABELED = "labeled"
# Colour: head. Line and mark: target statistics.
COLOURS = {"rflShared": "2A78D6", "rflRefit": "8E44AD", "rflNew": "1BAF7A",
           "rflLocal": "EB6834", "rflTslr": "52514E"}
HEAD_COLOUR = {"shared": "rflShared", "shared_ft": "rflRefit", "local": "rflNew"}
STATS_LINE = {"none": ("solid", "*"), "pool": ("dashed", "square*"),
              "labeled": ("densely dotted", "triangle*")}


def _fold(arm: str) -> Optional[int]:
    m = re.search(r"_f(\d+)$", str(arm))
    return int(m.group(1)) if m else None


def primary_stats(n: int) -> str:
    return PRIMARY.get(int(n), PRIMARY_LABELED)


def records(df: pd.DataFrame) -> pd.DataFrame:
    """The long table of the module docstring, accuracies in %, from load_runs rows."""
    from rfl.analysis.paper import norm_of
    rows = []
    for _, r in df.iterrows():
        name = "centralized_results.json" if r.kind == "centralized" else "fl_results.json"
        with open(os.path.join(r.path, name)) as f:
            res = json.load(f)
        base = {"model": r.model, "fold": _fold(r.arm), "seed": r.seed, "arm": r.arm}
        if r.model == "tslr":
            base.update(source="tslr", norm="")
        elif r.kind == "centralized":
            base.update(source="central", norm=norm_of(r.arch))
        else:
            base.update(source=str(r.scope).replace("+refit", ""), norm=norm_of(r.arch))
        tm = res.get("target_metrics")
        if tm:
            for x in tm["records"]:
                rows.append({**base, "subject": int(x["subject"]), "n": int(x["n"]),
                             "stats": x["stats"], "head": x["head"], "acc": 100 * x["acc"]})
        elif base["source"] == "local" and (res.get("config") or {}).get("target_subjects"):
            subs = (res.get("fl_info") or {}).get("client_subjects") or []
            accs = (res.get("final_metrics") or {}).get("per_client_test_acc") or []
            for s, a in zip(subs, accs):
                rows.append({**base, "subject": int(s[0]), "n": int(r.n_per_class),
                             "stats": "", "head": "scratch", "acc": 100 * a})
    cols = ["model", "source", "norm", "stats", "head", "n", "fold", "seed", "subject",
            "acc", "arm"]
    return pd.DataFrame(rows, columns=cols)


KEY = ["model", "source", "norm", "head", "stats"]


def complete(rec: pd.DataFrame) -> pd.DataFrame:
    """The rows of the (curve, N, seed) that cover every target subject of the study
    (module docstring). Prints the dropped ones."""
    n_all = rec.subject.nunique()
    size = rec.groupby(KEY + ["n", "seed"], dropna=False).subject.transform("nunique")
    gone = rec[size < n_all].drop_duplicates(KEY + ["n", "seed"])
    if len(gone):
        print(f"warning: {len(gone)} (curve, N, seed) without every target subject left "
              "out of the means: " + ", ".join(sorted({f"{r.arm} seed {r.seed}"
                                                       for r in gone.itertuples()})[:5]))
    return rec[size == n_all]


def by_seed(rec: pd.DataFrame) -> pd.DataFrame:
    """Mean over the target subjects of each (curve, N, seed) that covers them all."""
    return (complete(rec).groupby(KEY + ["n", "seed"], dropna=False).acc.mean()
            .reset_index())


def summary(rec: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, standard error and count over the seeds, per curve and N."""
    s = by_seed(rec).groupby(KEY + ["n"], dropna=False).acc.agg(["mean", "std", "count"])
    s = s.reset_index()
    s["se"] = s["std"] / np.sqrt(s["count"].clip(lower=1))
    return s


def _norm_label(source: str, norm: str) -> str:
    """'BN' for the arms without clients (centralised, local), else FedBN or shared BN."""
    return "BN" if source in ("central", "local") and norm == "bn" else NORMS.get(norm, norm)


def _sort_key(row) -> tuple:
    return tuple(ORDER[k].index(row[k]) if row[k] in ORDER[k] else 99
                 for k in ("source", "norm", "head", "stats"))


def within_subject(bn: Optional[pd.DataFrame]) -> Dict[tuple, float]:
    """(model, source, norm) -> mean test accuracy of the batch-norm study (subject
    clients), the reference of each source arm."""
    if bn is None:
        return {}
    from rfl.analysis.paper import bn_cells
    c = bn_cells(bn)
    c = c[c.clients.isin(["subject", "central"])]
    return {(m, h, n): float(g.acc.mean()) for (m, h, n), g in
            c.groupby(["model", "head", "norm"])}


# --------------------------------------------------------------------------- #
#  Tables
# --------------------------------------------------------------------------- #
def main_table(s: pd.DataFrame, within: Dict[tuple, float]) -> List[str]:
    """Rows: source arm x adaptation, by network. Columns: N, then the within-subject
    accuracy of the arm. Best mean of each column and network in bold."""
    from rfl.analysis.paper import _cell
    ns = sorted(s.n.unique())
    n_max = int(s["count"].max())
    lines = [f"\\begin{{tabular}}{{llll{'c' * len(ns)}c}}", "\\toprule",
             " & & & & \\multicolumn{" + str(len(ns)) + "}{c}{labeled target trials per "
             "class} & \\\\", f"\\cmidrule(lr){{5-{4 + len(ns)}}}",
             "Arm & Norm. & Head & Stats & " + " & ".join(f"{n:g}" for n in ns)
             + " & Within \\\\"]
    for model, title in MODELS.items():
        b = s[s.model == model]
        if b.empty:
            continue
        lines += ["\\midrule", f"\\multicolumn{{{5 + len(ns)}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        best = b.groupby("n")["mean"].max()
        keys = sorted({tuple(r[k] for k in KEY[1:]) for _, r in b.iterrows()},
                      key=lambda k: _sort_key(dict(zip(KEY[1:], k))))
        shown = set()
        for src, norm, head, stats in keys:
            g = b[(b.source == src) & (b.norm == norm) & (b["head"] == head)
                  & (b.stats == stats)]
            by_n = {r.n: r for _, r in g.iterrows()}
            cells = [_cell(by_n.get(n), n in by_n and by_n[n]["mean"] == best[n], n_max)
                     for n in ns]
            w = (within.get((model, src, norm))
                 if (src, norm) not in shown and src != "local" else None)
            shown.add((src, norm))
            norm_label = _norm_label(src, norm)
            heads = TSLR_HEADS if src == "tslr" else HEADS
            lines.append(f"{SOURCES[src]} & {norm_label} & {heads.get(head, head)} & "
                         f"{STATS.get(stats, stats)} & " + " & ".join(cells)
                         + f" & {'' if w is None else f'{w:.1f}'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def subject_table(rec: pd.DataFrame, model: str, n: int) -> Optional[List[str]]:
    """Rows: target subjects, then their mean. Columns: source arm and head, at one N
    and the primary target statistics, with the local reference of the same N."""
    from rfl.analysis.paper import _best, _cell, _stats
    stats = primary_stats(n)
    d = rec[(rec.model == model) & (rec.n == n)
            & ((rec.stats == stats) | (rec.source == "local"))]
    if d.empty or not (d.stats == stats).any():
        return None
    cols = sorted({(r.source, r.norm, r["head"]) for _, r in d.iterrows()},
                  key=lambda c: _sort_key({"source": c[0], "norm": c[1], "head": c[2],
                                           "stats": stats}))
    per = {c: d[(d.source == c[0]) & (d.norm == c[1]) & (d["head"] == c[2])] for c in cols}
    n_max = int(max(per[c].groupby("subject").acc.count().max() for c in cols))
    head = ["Subject"] + [f"{SOURCES[c[0]]}" + (f", {NORMS[c[1]]}" if c[1] and c[0] not in
                                               ("central", "local") else "")
                          + (f", {HEADS[c[2]]}" if HEADS.get(c[2]) else "") for c in cols]
    lines = [f"\\begin{{tabular}}{{l{'c' * len(cols)}}}", "\\toprule",
             " & ".join(f"\\rotatebox{{70}}{{{h}}}" if i else h for i, h in enumerate(head))
             + " \\\\", "\\midrule"]

    def row(name, vals, k):
        best = _best(vals)
        return name + " & " + " & ".join(
            _cell(v, v is not None and v["mean"] == best, k) for v in vals) + " \\\\"

    for s in sorted(d.subject.unique()):
        lines.append(row(str(s), [_stats(per[c][per[c].subject == s].acc) for c in cols],
                         n_max))
    means = [_stats(complete(per[c]).groupby("seed").acc.mean()) for c in cols]
    lines += ["\\midrule", row("Mean", means, max(int(m["count"]) for m in means)),
              "\\bottomrule", "\\end{tabular}"]
    return lines


# --------------------------------------------------------------------------- #
#  Paired tests on the subjects
# --------------------------------------------------------------------------- #
def _per_subject(rec: pd.DataFrame) -> pd.DataFrame:
    """Mean over the seeds of each (curve, N, subject)."""
    return rec.groupby(KEY + ["n", "subject"], dropna=False).acc.mean().reset_index()


def comparisons(rec: pd.DataFrame) -> List[tuple]:
    """The primary comparisons of the plan, (family, level, reference) curve keys (KEY +
    n), at the primary statistics of each N (PRIMARY):

        recalibration  primary statistics vs those of the sources ('none'), for the
                       shared head as trained at N = 0 and the new head at N > 0
        new head       new head vs the refitted shared head, N > 0
        federation     new head vs the target trained from scratch, N > 0

    With 9 or 10 targets the smallest exact two-sided Wilcoxon p-value is 2 / 2^9 or
    2 / 2^10, so Holm over k tests can only reject when k < 25 (9 targets) or 51: the
    families are corrected per network and N, a handful of tests each. Other contrasts
    stay descriptive (tab_loso).
    """
    keys = {tuple(r) for r in rec[KEY + ["n"]].drop_duplicates().itertuples(index=False)}
    out = []
    for k in sorted(keys, key=str):
        model, src, norm, head, stats, n = k
        if src in ("local", "tslr") or stats != primary_stats(n):
            continue
        if (head == "shared") == (n == 0) and head in ("shared", "local"):
            ref = (model, src, norm, head, "none", n)
            if ref in keys:
                out.append(("recalibration", k, ref))
        if head == "local" and n > 0:
            ref = (model, src, norm, "shared_ft", stats, n)
            if ref in keys:
                out.append(("new head", k, ref))
            ref = (model, "local", "bn", "scratch", "", n)
            if ref in keys:
                out.append(("federation", k, ref))
    return out


def paired_tests(rec: pd.DataFrame, min_pairs: int = 6) -> pd.DataFrame:
    """Wilcoxon signed-rank tests of the primary comparisons on the target subjects, the
    accuracy of a subject averaged over the seeds. Holm within each network, family and
    N. Differences are level - reference, in percentage points."""
    from scipy.stats import wilcoxon

    from rfl.analysis.stats import holm_bonferroni, rank_biserial
    ps = _per_subject(rec).set_index(KEY + ["n", "subject"]).acc
    rows = []
    for fam, a, b in comparisons(rec):
        va, vb = ps.loc[a], ps.loc[b]
        common = sorted(set(va.index) & set(vb.index))
        d = va.loc[common].to_numpy() - vb.loc[common].to_numpy()
        row = {"family": fam, **dict(zip(KEY + ["n"], a)), "vs_source": b[1],
               "vs_head": b[3], "vs_stats": b[4], "n_subjects": len(common),
               "mean_diff": float(d.mean()) if len(d) else math.nan,
               "median_diff": float(np.median(d)) if len(d) else math.nan,
               "win_rate": float(np.mean(d > 0)) if len(d) else math.nan,
               "rank_biserial": rank_biserial(d) if len(d) else math.nan,
               "p_value": math.nan}
        if len(common) >= min_pairs:
            try:
                row["p_value"] = float(wilcoxon(d, zero_method="wilcox").pvalue)
            except ValueError:                       # every difference is zero
                row["p_value"] = 1.0
        rows.append(row)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    fam = ["recalibration", "new head", "federation"]
    t = t.assign(_k=[(list(MODELS).index(r.model), fam.index(r.family),
                      *_sort_key(r._asdict()), r.n) for r in t.itertuples()])
    t = t.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    t["p_holm"] = math.nan
    for _, g in t.groupby(["model", "family", "n"]):
        ok = g.p_value.notna()
        if ok.any():
            t.loc[g.index[ok], "p_holm"] = holm_bonferroni(g.p_value[ok].tolist())
    return t


def tests_table(t: pd.DataFrame) -> List[str]:
    """One row per test: network, family, arm and adaptation, N, mean difference, win
    rate and Holm-adjusted p-value."""
    lines = ["\\begin{tabular}{lllllcrrr}", "\\toprule",
             "Net. & Comparison & Arm & Head & Stats & N & $\\Delta$ (pp) & win & "
             "$p_\\mathrm{Holm}$ \\\\", "\\midrule"]
    for _, r in t.iterrows():
        vs = {"recalibration": f"vs {r.vs_stats}", "new head": "vs refitted",
              "federation": "vs local"}[r.family]
        arm = SOURCES[r.source] + (f", {NORMS[r.norm]}" if r.norm and r.source not in
                                   ("central", "local") else "")
        p = "--" if not np.isfinite(r.p_holm) else (
            f"\\textbf{{{r.p_holm:.3f}}}" if r.p_holm < 0.05 else f"{r.p_holm:.3f}")
        lines.append(f"{MODELS[r.model]} & {r.family} ({vs}) & {arm} & "
                     f"{HEADS.get(r['head'], r['head'])} & {r.stats} & {int(r.n)} & "
                     f"{r.mean_diff:+.1f} & {r.win_rate:.2f} & {p} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
def figure(s: pd.DataFrame, path: Path) -> bool:
    """Accuracy against N: one panel per network and source arm, one curve per head and
    target statistics, with the local-from-scratch and TS+LR local references."""
    from rfl.analysis.tikz import grid_figure
    nets = [m for m in ("spdnet", "eegnet") if m in set(s.model)]
    if not nets:
        return False
    ns = sorted(s.n.unique())
    arms = [a for a in (("central", "bn"), ("shared", "bn"), ("shared", "shared"),
                        ("fedper", "bn"), ("fedper", "shared"))
            if ((s.source == a[0]) & (s.norm == a[1])).any()]
    panels, used = [], set()

    def series(g, style):
        g = g.sort_values("n")
        if list(g.n) == [0]:                 # the head as trained, without labels: flat
            g = pd.concat([g.assign(n=n) for n in ns])
        return {"style": style, "x": g.n.tolist(), "y": g["mean"].tolist(),
                "err_lo": g.se.fillna(0).tolist(), "err_hi": g.se.fillna(0).tolist()}

    for m in nets:
        refs = [(s[(s.model == m) & (s.source == "local")], "rflLocal, thick, mark=o"),
                (s[(s.model == "tslr") & (s["head"] == "local") & (s.stats == "labeled")],
                 "rflTslr, thick, mark=diamond*")]
        for src, norm in arms:
            sub = s[(s.model == m) & (s.source == src) & (s.norm == norm)]
            if sub.empty:
                continue
            out = []
            for (head, stats), g in sub.groupby(["head", "stats"]):
                dash, mark = STATS_LINE[stats]
                out.append(series(g, f"{HEAD_COLOUR[head]}, thick, {dash}, mark={mark}, "
                                     "mark size=1.5pt, mark options={solid}"))
                used.add((head, stats))
            out += [series(g, st) for g, st in refs if not g.empty]
            allv = pd.concat([sub] + [g for g, _ in refs])
            title = SOURCES[src] + ("" if src == "central" else f", {NORMS[norm]}")
            panels.append({"title": f"{MODELS[m]}: {title}",
                           "xlabel": "labeled target trials per class",
                           "ylabel": "test accuracy (\\%)", "xticks": ns, "xlog": False,
                           "ylog": False, "xmin": -1, "xmax": max(ns) + 1,
                           "ymin": math.floor((allv["mean"] - allv.se.fillna(0)).min() / 5) * 5,
                           "ymax": math.ceil((allv["mean"] + allv.se.fillna(0)).max() / 5) * 5,
                           "series": out})
    if not panels:
        return False
    legend = [(f"head {HEADS[h]}", f"{HEAD_COLOUR[h]}, line width=2pt")
              for h in ("shared", "shared_ft", "local") if any(u[0] == h for u in used)]
    legend += [(f"stats {st}", f"black!70, thick, {STATS_LINE[st][0]}, "
                               f"mark={STATS_LINE[st][1]}, mark options={{solid}}")
               for st in STATS_LINE if any(u[1] == st for u in used)]
    legend += [("local from scratch", "rflLocal, thick, mark=o"),
               ("TS+LR, target only", "rflTslr, thick, mark=diamond*")]
    grid_figure(path, panels, legend, cols=min(3, len(panels)), legend_name="losolegend",
                colours=COLOURS, legend_columns=4, width="0.27\\linewidth",
                comment="Inter-subject study: mean test accuracy of the target subjects "
                        "against N, mean and standard error over the seeds.")
    print(f"written: {path}")
    return True


# --------------------------------------------------------------------------- #
#  Entry point of rfl/analysis/paper.py
# --------------------------------------------------------------------------- #
def write_all(root: Path, dataset: str, out: Path) -> Optional[pd.DataFrame]:
    """Write every output of the module docstring. Returns the long table (for
    clients.csv), or None without final runs."""
    from rfl.analysis.paper import _write, final_runs
    runs = final_runs(root, dataset, ("loso", "loso_eegnet", "loso_tslr"))
    if runs is None:
        return None
    rec = records(runs)
    if rec.empty:
        return None
    s = summary(rec)
    _write(out / "tab_loso.tex",
           main_table(s, within_subject(final_runs(root, dataset, ("bn", "bn_eegnet")))))
    figure(s, out / "fig_loso.tex")
    for m in ("spdnet", "eegnet"):
        for n in sorted(rec.n.unique()):
            lines = subject_table(rec, m, int(n))
            if lines:
                _write(out / f"tab_loso_subjects_{m}_N{int(n)}.tex", lines)
    t = paired_tests(rec)
    if not t.empty:
        t.to_csv(out / "loso_tests.csv", index=False)
        print(f"written: {out / 'loso_tests.csv'}")
        _write(out / "tab_loso_tests.tex", tests_table(t))
    return rec


def subject_captions(out: Path) -> Dict[str, str]:
    """Captions of the per-subject tables found in `out`."""
    caps = {}
    for p in sorted(out.glob("tab_loso_subjects_*.tex")):
        m = re.match(r"tab_loso_subjects_(\w+?)_N(\d+)$", p.stem)
        if m:
            caps[p.stem] = (f"Inter-subject study, {MODELS[m.group(1)]}, $N = {m.group(2)}$, "
                            f"target statistics: {primary_stats(int(m.group(2)))}. Test "
                            "accuracy of each target "
                            "subject (\\%), mean $\\pm$ std over the seeds. Best mean of "
                            "each row in bold.")
    return caps


CAPTIONS = {
    "tab_loso": "Inter-subject study: mean test accuracy of the target subjects (\\%), "
                "mean $\\pm$ std over the seeds, against the labeled trials per class of "
                "the target. Stats: batch-norm statistics of the target (none: those of "
                "the sources; pool: its unlabeled calibration trials; labeled: its $N$ "
                "labeled trials), re-centring for TS+LR. Within: the arm in the "
                "batch-norm study. Best mean of each column and block in bold.",
    "fig_loso": "Inter-subject study: mean test accuracy of the target subjects against "
                "$N$, mean and standard error over the seeds.",
    "tab_loso_tests": "Inter-subject study, primary comparisons: paired Wilcoxon tests "
                      "on the target subjects (accuracy of a subject averaged over the "
                      "seeds). $\\Delta$: mean difference; win: share of subjects "
                      "improved; Holm correction within each comparison, network and $N$.",
}
