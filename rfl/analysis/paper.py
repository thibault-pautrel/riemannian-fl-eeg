"""LaTeX tables and pgfplots figures of the batch-norm and data-quantity studies.

    python -m rfl.analysis.paper BNCI2014_001 --out paper/BNCI2014_001

Reads the final runs of one dataset under $RFL_RESULTS (mi_<dataset>_bn, _bn_eegnet,
_datasize, _datasize_eegnet, _datasize_tslr) and writes, for the studies that have runs:

    tab_bn.tex                 batch norm: client mean test accuracy of every arm
    tab_bn_clients_<model>.tex batch norm: test accuracy of each subject
    fig_bn_rounds.tex          batch norm: client mean test accuracy against the round
    tab_datasize.tex           data quantity: accuracy against N (scenario U)
    fig_datasize.tex           data quantity: the same as a figure, one panel per network
    clients.csv                every final run: test accuracy of each client
    results.tex, results.pdf   one document with all of them (pdf when pdflatex is found)

Accuracies are in %. Tables: mean ± standard deviation over the seeds, best mean in bold
(per row and network for the batch norm, per column and block for the data quantity), the
number of seeds in brackets when some are missing (diverged runs, left out). Figures:
mean over the seeds, with the standard error as error bars. "Shared head, re-fitted" is
the shared-head arm after each client re-fits its head. A centralised model is scored on
the pooled test trials.

Each tab_/fig_ file holds a bare `tabular` or `tikzpicture`, to \\input in a table or
figure. The tables need \\usepackage{booktabs}, the figures \\usepackage{pgfplots} and
\\usepgfplotslibrary{groupplots}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from rfl.analysis import datasize as D
from rfl.analysis.results import load_runs
from rfl.analysis.tikz import grid_figure

MODELS = {"spdnet": "SPDNet", "eegnet": "EEGNet"}
HEADS = {"local": "Local", "fedper": "FedPer", "shared": "Shared head",
         "refit": "Shared head, re-fitted", "central": "Centralised"}
NORMS = {"nobn": "no BN", "bn": "BN (FedBN)", "shared": "shared BN"}
# Colour: paradigm and head. Line and mark: normalisation.
COLOURS = {"rflTslr": "52514E", "rflPooled": "A3A29C", "rflLocal": "EB6834",
           "rflRefit": "EDA100", "rflShared": "2A78D6", "rflFedper": "1BAF7A"}
HEAD_COLOUR = {"local": "rflLocal", "fedper": "rflFedper", "shared": "rflShared",
               "refit": "rflRefit"}
NORM_LINE = {"bn": ("solid", "*"), "shared": ("dashed", "square*"),
             "nobn": ("densely dotted", "triangle*"), "": ("solid", "diamond*")}


def norm_of(arch: str) -> str:
    """Normalisation of an architecture label (rfl/analysis/results.py:arch_name)."""
    arch = str(arch)
    return "shared" if arch.endswith("bnshared") else "bn" if arch.endswith("_bn") else "nobn"


def final_runs(root: Path, dataset: str, studies: Sequence[str]) -> Optional[pd.DataFrame]:
    """The final runs of the plans `mi_<dataset>_<study>`, diverged runs left out."""
    dirs = [str(d) for s in studies if (d := root / f"mi_{dataset}_{s}" / "final").is_dir()]
    try:
        df = load_runs(*dirs) if dirs else None
    except FileNotFoundError:
        return None
    if df is None:
        return None
    div = df.diverged.fillna(False).astype(bool)
    if div.any():
        print(f"warning: {int(div.sum())} diverged run(s) left out: "
              + ", ".join(sorted(set(df.arm[div].astype(str)))))
    return df[~div].copy()


# --------------------------------------------------------------------------- #
#  Table helpers
# --------------------------------------------------------------------------- #
def _cell(stats: Optional[pd.Series], bold: bool, n_max: int) -> str:
    """'$m \\pm s$', the mean in bold, the number of seeds when below `n_max`."""
    if stats is None or not stats["count"]:
        return "--"
    m = f"\\mathbf{{{stats['mean']:.1f}}}" if bold else f"{stats['mean']:.1f}"
    s = stats["std"] if stats["count"] > 1 else 0.0
    n = f"\\,{{\\scriptsize ({int(stats['count'])})}}" if stats["count"] < n_max else ""
    return f"${m} \\pm {s:.1f}${n}"


def _best(values: List[Optional[pd.Series]]) -> Optional[float]:
    means = [v["mean"] for v in values if v is not None and v["count"]]
    return max(means) if means else None


def _stats(acc: pd.Series) -> pd.Series:
    return pd.Series({"mean": acc.mean(), "std": acc.std(), "count": acc.count()})


def _write(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"written: {path}")


# --------------------------------------------------------------------------- #
#  Batch-norm study
# --------------------------------------------------------------------------- #
def bn_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run and head: model, clients, head, norm, seed, acc (%)."""
    rows = []
    for _, r in df.iterrows():
        norm = norm_of(r.arch)
        if r.kind == "centralized":
            rows.append((r.model, "central", "central", norm, r.seed, r.test_acc))
            continue
        scope = str(r.scope)
        rows.append((r.model, r.clients, scope.replace("+refit", ""), norm, r.seed,
                     r.client_mean_acc))
        if scope.endswith("+refit"):
            rows.append((r.model, r.clients, "refit", norm, r.seed,
                         r.reduced_refit_test_acc))
    t = pd.DataFrame(rows, columns=["model", "clients", "head", "norm", "seed", "acc"])
    t = t.dropna(subset=["acc"])
    t["acc"] *= 100
    return t


def bn_table(cells: pd.DataFrame) -> List[str]:
    """Rows: client partition and head. Columns: network and normalisation."""
    models = [m for m in MODELS if m in set(cells.model)]
    stats = {k: _stats(g.acc) for k, g in cells.groupby(["model", "clients", "head", "norm"])}
    n_max = int(max(s["count"] for s in stats.values()))
    lines = [f"\\begin{{tabular}}{{ll{'ccc' * len(models)}}}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{MODELS[m]}}}" for m in models)
             + " \\\\",
             "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(models))),
             "Clients & Head & " + " & ".join(list(NORMS.values()) * len(models)) + " \\\\",
             "\\midrule"]
    groups = [(c, [h for h in ("local", "fedper", "shared", "refit")])
              for c in ("subject", "iid") if c in set(cells.clients)]
    if "central" in set(cells.clients):
        groups.append(("central", ["central"]))
    for gi, (clients, heads) in enumerate(groups):
        if gi:
            lines.append("\\midrule")
        first = True
        for head in heads:
            parts = []
            for m in models:
                vals = [stats.get((m, clients, head, n)) for n in NORMS]
                best = _best(vals)
                parts += [_cell(v, v is not None and v["mean"] == best, n_max) for v in vals]
            if all(p == "--" for p in parts):
                continue
            name = "" if not first else {"subject": "subject", "iid": "IID",
                                         "central": "--"}[clients]
            lines.append(f"{name} & {HEADS[head]} & " + " & ".join(parts) + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def client_details(df: pd.DataFrame) -> pd.DataFrame:
    """One row per federated run and client: its subjects and test accuracies (%)."""
    rows = []
    for _, r in df[df.kind == "federated"].iterrows():
        with open(os.path.join(r.path, "fl_results.json")) as f:
            res = json.load(f)
        subjects = (res.get("fl_info") or {}).get("client_subjects") or []
        acc = (res.get("final_metrics") or {}).get("per_client_test_acc") or []
        refit = (res.get("personalized_metrics") or {}).get(
            "personalized_per_client_test_acc") or []
        for i, a in enumerate(acc):
            s = subjects[i] if i < len(subjects) else []
            rows.append({"arm": r.arm, "model": r.model, "arch": r.arch,
                         "clients": r.clients, "scope": r.scope,
                         "n_per_class": r.n_per_class, "seed": r.seed, "client": i,
                         "subjects": "+".join(str(x) for x in s),
                         "test_acc": 100 * a if a is not None else math.nan,
                         "refit_test_acc": (100 * refit[i] if i < len(refit)
                                            and refit[i] is not None else math.nan)})
    return pd.DataFrame(rows)


def client_table(det: pd.DataFrame, model: str) -> Optional[List[str]]:
    """Rows: subjects, then their mean. Columns: head and normalisation (subject clients)."""
    d = det[(det.model == model) & (det.clients == "subject")].copy()
    if d.empty:
        return None
    d["norm"] = d.arch.map(norm_of)
    d["base"] = d.scope.str.replace("+refit", "", regex=False)
    cols = [("local", "nobn"), ("local", "bn"), ("fedper", "nobn"), ("fedper", "bn"),
            ("fedper", "shared"), ("refit", "nobn"), ("refit", "bn"), ("refit", "shared")]

    def acc_of(head: str, norm: str) -> pd.DataFrame:
        if head == "refit":
            sel = d[(d.base == "shared") & d.scope.str.endswith("+refit") & (d.norm == norm)]
            return sel.assign(acc=sel.refit_test_acc)
        sel = d[(d.base == head) & (d.norm == norm)]
        return sel.assign(acc=sel.test_acc)

    per = {c: acc_of(*c) for c in cols}
    cols = [c for c in cols if not per[c].empty]
    subjects = sorted(set(d.subjects), key=lambda s: [int(x) for x in s.split("+") if x])
    n_max = int(max(per[c].groupby("subjects").acc.count().max() for c in cols))
    spans = [(h, sum(1 for c in cols if c[0] == h)) for h in ("local", "fedper", "refit")]
    spans = [(h, k) for h, k in spans if k]
    head = ["Subject"] + [f"\\multicolumn{{{k}}}{{c}}{{{HEADS[h]}}}" for h, k in spans]
    rules, start = [], 2
    for _, k in spans:
        rules.append(f"\\cmidrule(lr){{{start}-{start + k - 1}}}")
        start += k
    lines = [f"\\begin{{tabular}}{{l{'c' * len(cols)}}}", "\\toprule",
             " & ".join(head) + " \\\\", "".join(rules),
             " & " + " & ".join(NORMS[n] for _, n in cols) + " \\\\", "\\midrule"]

    def row(name: str, vals: List[Optional[pd.Series]], n: int) -> str:
        best = _best(vals)
        return name + " & " + " & ".join(
            _cell(v, v is not None and v["mean"] == best, n) for v in vals) + " \\\\"

    for s in subjects:
        vals = [_stats(per[c][per[c].subjects == s].acc) for c in cols]
        lines.append(row(s, vals, n_max))
    # Mean over the subjects of each seed, then over the seeds: the client mean accuracy.
    means = [_stats(per[c].groupby("seed").acc.mean()) for c in cols]
    lines += ["\\midrule", row("Mean", means, max(int(m["count"]) for m in means)),
              "\\bottomrule", "\\end{tabular}"]
    return lines


def _points(length: int, n: int = 200) -> np.ndarray:
    """At most `n` evenly spaced indices of a curve of `length` rounds, the last included."""
    return np.unique(np.linspace(0, length - 1, min(n, length)).round().astype(int))


def bn_rounds(df: pd.DataFrame, path: Path) -> bool:
    """Client mean test accuracy against the round, one panel per network.

    A seed that stopped early keeps its last value up to the longest seed of its arm, so
    each curve runs until the last seed stops and ends on its plateau.
    """
    fed = df[(df.kind == "federated") & (df.clients == "subject")]
    panels, used = [], set()
    for m in [m for m in MODELS if m in set(fed.model)]:
        series, top = [], 0
        for (scope, arch), g in fed[fed.model == m].groupby(["scope", "arch"]):
            curves = []
            for p in g.path:
                with open(os.path.join(p, "fl_results.json")) as f:
                    h = (json.load(f).get("round_history") or {}).get("avg_test_acc")
                if h:
                    curves.append(np.asarray(h, float))
            if not curves:
                continue
            n = max(len(c) for c in curves)
            mean = 100 * np.mean([np.pad(c, (0, n - len(c)), mode="edge") for c in curves],
                                 axis=0)
            idx = _points(n)
            head, norm = scope.replace("+refit", ""), norm_of(arch)
            dash, _ = NORM_LINE[norm]
            series.append({"style": f"{HEAD_COLOUR[head]}, thick, {dash}",
                           "x": (idx + 1).tolist(), "y": mean[idx].tolist()})
            used.add((head, norm))
            top = max(top, n)
        if series:
            ys = [y for s in series for y in s["y"]]
            step = 10 ** math.floor(math.log10(top)) / (2 if top < 2.5 * 10 ** math.floor(
                math.log10(top)) else 1)
            ticks = [int(step * k) for k in range(int(top // step) + 1)]
            panels.append({"title": MODELS[m], "xlabel": "communication round",
                           "ylabel": "test accuracy (\\%)", "xticks": ticks,
                           "xmin": 0, "xmax": top, "xlog": False, "ylog": False,
                           "ymin": math.floor(min(ys) / 5) * 5,
                           "ymax": math.ceil(max(ys) / 5) * 5, "series": series})
    if not panels:
        return False
    legend = [(HEADS[h], f"{HEAD_COLOUR[h]}, line width=2pt")
              for h in ("local", "fedper", "shared") if any(u[0] == h for u in used)]
    legend += [(NORMS[n], f"black!70, thick, {NORM_LINE[n][0]}")
               for n in NORMS if any(u[1] == n for u in used)]
    grid_figure(path, panels, legend, cols=len(panels), legend_name="bnroundslegend",
                colours=COLOURS, legend_columns=-1,
                comment="Client mean test accuracy against the round, subject clients, "
                        "mean over the final seeds.")
    print(f"written: {path}")
    return True


# --------------------------------------------------------------------------- #
#  Data-quantity study
# --------------------------------------------------------------------------- #
def _datasize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """rfl/analysis/datasize.py:table, scenario U, with a head key and a norm key."""
    t = D.table(df)
    t = t[t.scenario == "U"].copy()

    def head(f: str) -> str:
        if "Pooled" in f:
            return "pooled"
        if "TS+LR" in f:
            return "tslr"
        if f.startswith("Local"):
            return "local"
        return "refit" if "re-fitted" in f else "shared" if "shared head" in f else "fedper"

    t["head"] = t.family.map(head)
    t["normkey"] = t.norm.map({"": "", "BN": "bn", "FedBN": "bn", "shared BN": "shared",
                               "no BN": "nobn"})
    return t


DS_HEADS = {"tslr": "Local", "pooled": "Pooled re-centred", "local": "Local",
            "fedper": "FedPer", "shared": "Shared head", "refit": "Shared head, re-fitted"}
DS_COLOUR = {"tslr": "rflTslr", "pooled": "rflPooled", **HEAD_COLOUR}
NORM_ORDER = ["", "nobn", "bn", "shared"]


def datasize_table(t: pd.DataFrame) -> List[str]:
    """Rows: curves, by block (TS+LR, then each network). Columns: N, then all."""
    ns = sorted(t.N.unique())
    names = ["all" if not np.isfinite(n) else f"{n:g}" for n in ns]
    n_max = int(t["count"].max())
    lines = [f"\\begin{{tabular}}{{ll{'c' * len(ns)}}}", "\\toprule",
             " & & \\multicolumn{" + str(len(ns)) + "}{c}{trials per class and subject} \\\\",
             f"\\cmidrule(lr){{3-{2 + len(ns)}}}",
             "Head & Normalisation & " + " & ".join(names) + " \\\\"]
    for model, title in (("tslr", "TS+LR"), *MODELS.items()):
        b = t[t.model == model]
        if b.empty:
            continue
        lines += ["\\midrule", f"\\multicolumn{{{2 + len(ns)}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        best = b.groupby("N")["mean"].max()
        for (h, nk), g in sorted(b.groupby(["head", "normkey"]),
                                 key=lambda kv: (list(DS_HEADS).index(kv[0][0]),
                                                 NORM_ORDER.index(kv[0][1]))):
            by_n = {r.N: r for _, r in g.iterrows()}
            cells = [_cell(by_n.get(n), n in by_n and by_n[n]["mean"] == best[n], n_max)
                     for n in ns]
            norm = NORMS.get(nk, "") if h != "local" else {"bn": "BN", "nobn": "no BN"}[nk]
            lines.append(f"{DS_HEADS[h]} & {norm} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def datasize_figure(t: pd.DataFrame, path: Path) -> bool:
    """Accuracy against N, one panel per network, with the TS+LR curves on each."""
    nets = [m for m in MODELS if m in set(t.model)]
    if not nets:
        return False
    ns = sorted(n for n in t.N.unique() if np.isfinite(n))
    has_all = bool((~np.isfinite(t.N)).any())
    x_all = 2 * max(ns)
    ticks = ns + ([x_all] if has_all else [])
    labels = [f"{n:g}" for n in ns] + (["all"] if has_all else [])
    panels, used = [], set()
    for m in nets:
        sub = t[t.model.isin([m, "tslr"])]
        series = []
        for (h, nk), g in sub.groupby(["head", "normkey"]):
            g = g.sort_values("N")
            dash, mark = NORM_LINE[nk]
            series.append({"style": f"{DS_COLOUR[h]}, thick, {dash}, mark={mark}, "
                                    "mark size=1.5pt, mark options={solid}",
                           "x": [x_all if not np.isfinite(n) else n for n in g.N],
                           "y": g["mean"].tolist(), "err_lo": g["se"].tolist(),
                           "err_hi": g["se"].tolist()})
            used.add((h, nk))
        panels.append({"title": MODELS[m], "xlabel": "trials per class and subject",
                       "ylabel": "test accuracy (\\%)", "xticks": ticks,
                       "xticklabels": labels, "xmin": min(ns) / 1.3, "xmax": x_all * 1.3,
                       "ylog": False,
                       "ymin": math.floor((sub["mean"] - sub["se"].fillna(0)).min() / 5) * 5,
                       "ymax": math.ceil((sub["mean"] + sub["se"].fillna(0)).max() / 5) * 5,
                       "series": series})
    legend = [(("TS+LR, " + DS_HEADS[h].lower()) if h in ("tslr", "pooled") else DS_HEADS[h],
               f"{DS_COLOUR[h]}, line width=2pt")
              for h in DS_HEADS if any(u[0] == h for u in used)]
    legend += [(NORMS[n], f"black!70, thick, {NORM_LINE[n][0]}, mark={NORM_LINE[n][1]}, "
                          "mark options={solid}")
               for n in NORMS if any(u[1] == n for u in used)]
    grid_figure(path, panels, legend, cols=len(panels), legend_name="datasizelegend",
                colours=COLOURS, legend_columns=4,
                comment="Client mean test accuracy against the trials per class and "
                        "subject (scenario U), mean and standard error over the seeds.")
    print(f"written: {path}")
    return True


CAPTIONS = {
    "tab_bn": "Batch-norm study: client mean test accuracy (\\%), mean $\\pm$ std over "
              "the seeds. Best mean of each row and network in bold.",
    "tab_bn_clients_spdnet": "Batch-norm study, SPDNet, subject clients: test accuracy of "
                             "each subject (\\%). Best mean of each row in bold.",
    "tab_bn_clients_eegnet": "Batch-norm study, EEGNet, subject clients: test accuracy of "
                             "each subject (\\%). Best mean of each row in bold.",
    "fig_bn_rounds": "Batch-norm study, subject clients: client mean test accuracy against "
                     "the communication round, mean over the seeds. A seed that stopped "
                     "keeps its last value.",
    "tab_datasize": "Data quantity (every subject has $N$ trials per class): client mean "
                    "test accuracy (\\%). Best mean of each column and block in bold.",
    "fig_datasize": "Data quantity: client mean test accuracy against $N$, mean and "
                    "standard error over the seeds.",
}


def document(out: Path, dataset: str) -> Path:
    """results.tex: every table and figure of `out`, by study."""
    name = dataset.replace("_", "\\_")
    lines = ["\\documentclass[a4paper]{article}",
             "\\usepackage[margin=1.5cm]{geometry}",
             "\\usepackage{booktabs,graphicx,pgfplots}",
             "\\usepgfplotslibrary{groupplots}", "\\pgfplotsset{compat=1.16}",
             f"\\title{{Results on {name}}}", "\\date{\\today}",
             "\\begin{document}", "\\maketitle"]
    for title, names in (("Batch norm", ("tab_bn", "tab_bn_clients_spdnet",
                                         "tab_bn_clients_eegnet", "fig_bn_rounds")),
                         ("Data quantity", ("tab_datasize", "fig_datasize"))):
        names = [n for n in names if (out / f"{n}.tex").exists()]
        if not names:
            continue
        lines.append(f"\\section{{{title}}}")
        for n in names:
            caption = f"\\caption{{{CAPTIONS[n]}}}"
            if n.startswith("tab"):          # caption above a table, below a figure
                lines += ["\\begin{table}[htbp]\\centering", caption,
                          "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}"
                          f"{{\\input{{{n}}}}}", "\\end{table}"]
            else:
                lines += ["\\begin{figure}[htbp]\\centering", f"\\input{{{n}}}", caption,
                          "\\end{figure}"]
        lines.append("\\clearpage")
    lines.append("\\end{document}")
    path = out / "results.tex"
    _write(path, lines)
    return path


def compile_pdf(tex: Path) -> None:
    """Compile `tex` with pdflatex when it is installed."""
    if not shutil.which("pdflatex"):
        print(f"pdflatex not found: compile {tex} elsewhere")
        return
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                       cwd=tex.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = tex.with_suffix(".pdf")
    print(f"written: {pdf}" if r.returncode == 0
          else f"pdflatex failed: see {tex.with_suffix('.log')}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataset", help="for example BNCI2014_001")
    p.add_argument("--out", default=None, help="output folder (default: paper/<dataset>)")
    p.add_argument("--results", default=os.environ.get("RFL_RESULTS", "results"),
                   help="results folder (default: $RFL_RESULTS)")
    a = p.parse_args(argv)
    root, out = Path(a.results), Path(a.out or f"paper/{a.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    details = []

    bn = final_runs(root, a.dataset, ("bn", "bn_eegnet"))
    if bn is not None:
        _write(out / "tab_bn.tex", bn_table(bn_cells(bn)))
        det = client_details(bn)
        details.append(det.assign(study="bn"))
        for m in MODELS:
            lines = client_table(det, m) if not det.empty else None
            if lines:
                _write(out / f"tab_bn_clients_{m}.tex", lines)
        bn_rounds(bn, out / "fig_bn_rounds.tex")

    ds = final_runs(root, a.dataset, ("datasize", "datasize_eegnet", "datasize_tslr"))
    if ds is not None:
        t = _datasize_rows(ds)
        _write(out / "tab_datasize.tex", datasize_table(t))
        datasize_figure(t, out / "fig_datasize.tex")
        details.append(client_details(ds).assign(study="datasize"))

    if not details:
        raise SystemExit(f"no final run of {a.dataset} under {root}")
    pd.concat(details).to_csv(out / "clients.csv", index=False)
    print(f"written: {out / 'clients.csv'}")
    compile_pdf(document(out, a.dataset))


if __name__ == "__main__":
    main()
