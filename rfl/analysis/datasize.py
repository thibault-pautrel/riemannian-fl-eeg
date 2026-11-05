"""Accuracy against the number of calibration trials: table and figure.

Reads the final runs of the data-quantity plans (mi_datasize, mi_datasize_eegnet,
mi_datasize_tslr, synth_datasize) and plots, for each scenario, the test accuracy of
every arm against N (calibration trials per class and subject, log scale):

    U   every subject has N trials: mean accuracy over the subjects
    H   a few subjects have N trials, the others all theirs: mean accuracy of the
        subjects with N trials

Each point is the mean over the seeds of `reduced_test_acc` (the mean accuracy of the
reduced clients, every client in U and full), with its standard error over the seeds.
The .csv holds every curve of both network models: compare SPDNet and EEGNet there.
One figure per network model (SPDNet, EEGNet), with the TS+LR baselines on each. The
colour gives the paradigm and the head (local, federated with a shared head, a shared
head re-fitted on each client, a personal head), the line style the normalisation
(solid: batch norm kept on each client, dashed: shared batch norm, dotted: none). The
table names each curve in full, for example "Federated · SPDNet · shared head · FedBN".
The point "all" (no reduction) is drawn at the right of the axis. With --pairs, the per-seed differences (federated minus the best
local arm of the same seed) give a 95 % interval on the gain.

    python -m rfl.analysis.datasize $RFL_RESULTS/mi_BNCI2014_001_datasize*/final \\
        --out datasize_BNCI2014_001
"""

from __future__ import annotations

import argparse
import math
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from rfl.analysis.results import load_runs

MODELS = {"spdnet": "SPDNet", "eegnet": "EEGNet", "tslr": "TS+LR"}
HEADS = {"shared": "shared head", "shared+refit": "shared head, re-fitted",
         "fedper": "personal head (FedPer)", "local": "", "pooled": ""}
NORMS = {"nobn": "no BN", "bn": "FedBN", "bnshared": "shared BN"}


def parse_arm(arm: str) -> Dict[str, Optional[str]]:
    """Scenario, normalisation and N from an arm name of the data-quantity plans."""
    m = re.search(r"_(nobn|bn|bnshared)_(U\d+|H\d+|full)$", arm or "")
    if m:
        norm, scen = m.group(1), m.group(2)
    else:
        t = re.search(r"_(U\d+|H\d+|full)$", arm or "")
        norm, scen = None, (t.group(1) if t else None)
    return {"norm": norm, "scenario": scen}


def family(model: str, scope: str) -> str:
    """Paradigm, model and head of a curve (its colour in the figure)."""
    m = MODELS.get(model, model)
    if scope == "pooled":
        return f"Pooled re-centred · {m} (not federated)"
    if scope == "local":
        return f"Local · {m}"
    return f"Federated · {m} · {HEADS.get(scope, scope)}"


def norm_name(model: str, scope: str, norm: Optional[str]) -> str:
    """Normalisation of a curve (its line style in the figure)."""
    if model == "tslr" or norm is None:
        return ""
    if scope == "local":
        return "BN" if norm == "bn" else "no BN"
    return NORMS.get(norm, norm)


def label(model: str, scope: str, norm: Optional[str]) -> str:
    """Curve name: paradigm, model, head, normalisation."""
    n = norm_name(model, scope, norm)
    return family(model, scope) + (f" · {n}" if n else "")


def table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, curve, N): mean and standard error over the seeds."""
    rows = []
    for _, r in df.iterrows():
        p = parse_arm(r["arm"])
        if p["scenario"] is None:
            continue
        kind = p["scenario"][0] if p["scenario"] != "full" else "full"
        n = math.inf if p["scenario"] == "full" else int(p["scenario"][1:])
        scope = str(r["scope"]).replace("+refit", "")
        base = {"scenario": kind, "N": n, "seed": r["seed"], "model": r["model"],
                "norm": norm_name(r["model"], scope, p["norm"])}
        rows.append({**base, "family": family(r["model"], scope),
                     "curve": label(r["model"], scope, p["norm"]),
                     "acc": r["reduced_test_acc"]})
        if str(r["scope"]).endswith("+refit") and not pd.isna(r["reduced_refit_test_acc"]):
            rows.append({**base, "family": family(r["model"], "shared+refit"),
                         "curve": label(r["model"], "shared+refit", p["norm"]),
                         "acc": r["reduced_refit_test_acc"]})
    t = pd.DataFrame(rows)
    # 'full' belongs to both scenarios (every subject has all its trials).
    full = t[t.scenario == "full"]
    t = pd.concat([t[t.scenario != "full"], full.assign(scenario="U"),
                   full.assign(scenario="H")])
    g = t.groupby(["scenario", "model", "family", "norm", "curve", "N"])["acc"]
    out = g.agg(["mean", "count", "std"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"])
    out[["mean", "std", "se"]] *= 100
    return out


# Colour: paradigm and head. Line and marker: normalisation.
FAMILY_ORDER = ("Local · TS+LR", "Pooled re-centred", "Local", "shared head, re-fitted",
                "shared head", "personal head")
FAMILY_COLORS = ("#52514e", "#a3a29c", "#eb6834", "#eda100", "#2a78d6", "#1baf7a")
NORM_STYLES = {"": ("-", "D"), "FedBN": ("-", "o"), "BN": ("-", "o"),
               "shared BN": ("--", "s"), "no BN": (":", "^")}


def _family_rank(f: str) -> int:
    for i, key in enumerate(FAMILY_ORDER):
        if key in f and not (key == "Local" and "TS+LR" in f) \
                and not (key == "shared head" and "re-fitted" in f):
            return i
    return len(FAMILY_ORDER)


def plot(tab: pd.DataFrame, path: str, title: str = "") -> None:
    """One figure, two panels (U, H). `tab` holds one network model and the baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    fams = sorted(tab.family.unique(), key=_family_rank)
    color = {f: FAMILY_COLORS[min(_family_rank(f), len(FAMILY_COLORS) - 1)] for f in fams}
    ns = sorted(n for n in tab.N.unique() if np.isfinite(n))
    x_full = (max(ns) * 2) if ns else 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, scen, ylab in ((axes[0], "U", "test accuracy, all subjects (%)"),
                           (axes[1], "H", "test accuracy, subjects with N trials (%)")):
        sub = tab[tab.scenario == scen]
        for (fam, norm), d in sub.groupby(["family", "norm"]):
            d = d.sort_values("N")
            x = [x_full if not np.isfinite(n) else n for n in d.N]
            ls, mk = NORM_STYLES.get(norm, ("-", "o"))
            ax.errorbar(x, d["mean"], yerr=d["se"], color=color[fam], marker=mk, ls=ls,
                        lw=1.8, ms=6, capsize=2)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        has_full = bool((~np.isfinite(tab.N)).any())
        ax.set_xticks(ns + ([x_full] if has_full else []))
        ax.set_xticklabels([f"{n:g}" for n in ns] + (["all"] if has_full else []))
        ax.set_xlabel("calibration trials per class and subject (N)")
        ax.set_ylabel(ylab)
        ax.set_title("every subject has N trials" if scen == "U" else
                     "some subjects have N trials, the others all theirs", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=color[f], lw=3) for f in fams]
    labels = list(fams)
    present = set(tab.norm)
    norms = [n for n in ("FedBN", "shared BN", "no BN")
             if n in present or (n == "FedBN" and "BN" in present)]
    handles += [Line2D([], [], color="#52514e", ls=NORM_STYLES[n][0],
                       marker=NORM_STYLES[n][1], lw=1.8) for n in norms]
    labels += [{"FedBN": "BN on each client (FedBN, local)"}.get(n, n) for n in norms]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    nrow = math.ceil(len(labels) / 3)
    fig.tight_layout(rect=(0, 0.04 * nrow + 0.02, 1, 1))
    if title:
        fig.text(0.5, 0.0, title, ha="center", va="top", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def paired_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Per scenario and N: each federated curve (re-fitted heads included) minus the best
    local curve of the scenario (TS+LR or a network, chosen on its mean), paired by seed."""
    t = []
    for _, r in df.iterrows():
        p = parse_arm(r["arm"])
        if p["scenario"] in (None, "full"):
            continue
        scope = str(r["scope"]).replace("+refit", "")
        t.append({"scenario": p["scenario"], "seed": r["seed"],
                  "curve": label(r["model"], scope, p["norm"]),
                  "acc": 100 * r["reduced_test_acc"]})
        if str(r["scope"]).endswith("+refit") and not pd.isna(r["reduced_refit_test_acc"]):
            t.append({"scenario": p["scenario"], "seed": r["seed"],
                      "curve": label(r["model"], "shared+refit", p["norm"]),
                      "acc": 100 * r["reduced_refit_test_acc"]})
    t = pd.DataFrame(t)
    if t.empty:
        return t
    out = []
    for scen, d in t.groupby("scenario"):
        piv = d.pivot_table(index="seed", columns="curve", values="acc")
        local = [c for c in piv.columns if c.startswith("Local")]
        if not local:
            continue
        best_local = piv[local].mean().idxmax()
        for c in piv.columns:
            if c.startswith("Federated"):
                diff = (piv[c] - piv[best_local]).dropna()
                if len(diff) >= 2:
                    se = diff.std(ddof=1) / math.sqrt(len(diff))
                    out.append({"scenario": scen, "curve": c, "vs": best_local,
                                "gain": diff.mean(), "ci95": 1.96 * se, "n": len(diff)})
    return pd.DataFrame(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("roots", nargs="+", help="final folders of the data-quantity plans")
    p.add_argument("--out", default="datasize", help="prefix of the .csv and .png files")
    p.add_argument("--title", default="", help="line under the figure")
    p.add_argument("--pairs", action="store_true", help="also print the paired gains")
    a = p.parse_args(argv)
    df = load_runs(*a.roots)
    dup = df[df.duplicated(["arm", "seed"], keep=False)]
    if not dup.empty:
        raise SystemExit(
            f"{len(dup)} runs share an arm and a seed (final runs of an earlier selection "
            "left in the final folder?), for example:\n  "
            + "\n  ".join(dup.path.head(4)) + "\nMove the stale folders out, then rerun.")
    div = df[df.diverged]
    if not div.empty:
        print(f"warning: {len(div)} final run(s) diverged and are left out: "
              + ", ".join(sorted(set(div.arm))[:6]))
    df = df[~df.diverged]
    seeds = df.groupby("arm").seed.nunique()
    if seeds.nunique() > 1:
        print(f"warning: the arms have {seeds.min()} to {seeds.max()} final seeds "
              "(abandoned or diverged runs): the means are over different seeds")
    tab = table(df)
    tab.to_csv(f"{a.out}.csv", index=False)
    figures = []
    nets = [m for m in ("spdnet", "eegnet") if m in set(tab.model)]
    for m in nets or [None]:
        sub = tab[tab.model.isin([m, "tslr"])] if m else tab
        path = f"{a.out}_{m}.png" if m else f"{a.out}.png"
        plot(sub, path, a.title)
        figures.append(path)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        for scen in ("U", "H"):
            sub = tab[tab.scenario == scen]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="curve", columns="N", values="mean").round(1)
            piv.columns = ["all" if not np.isfinite(c) else f"{c:g}" for c in piv.columns]
            print(f"\nscenario {scen} (mean test accuracy %, "
                  f"{'all subjects' if scen == 'U' else 'subjects with N trials'})")
            print(piv.to_string())
        if a.pairs:
            gains = paired_gain(df)
            print("\nfederated minus the best local curve, paired by seed (95 % interval):")
            print(gains.round(2).to_string(index=False) if not gains.empty
                  else "  (needs at least two seeds per arm)")
    print(f"\nwritten: {a.out}.csv " + " ".join(figures))


if __name__ == "__main__":
    main()
