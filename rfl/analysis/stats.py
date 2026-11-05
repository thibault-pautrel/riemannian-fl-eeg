#!/usr/bin/env python3
"""Paired significance tests between the levels of one factor, matched by seed.

Within a seed, every arm shares the same data split and the same initialisation, so the
runs of two arms are paired. For each group of runs (``--by``), the script compares the
levels of one factor (``--factor``, the aggregation rule by default):

* pairwise Wilcoxon signed-rank tests on the seeds common to both levels, with the
  Holm-Bonferroni correction within each group, and three effect sizes: median paired
  difference, win rate, matched-pairs rank-biserial correlation,
* a Friedman test across all levels when there are at least three, on the seeds common
  to all of them, with a Nemenyi post-hoc test if scikit-posthocs is installed.

Usage (from the repository root)::

    python -m rfl.analysis.stats results/federated --factor aggregation --by dataset eps
    python -m rfl.analysis.stats results/federated --factor scope --baseline shared \\
        --by dataset arch --out-dir stats/

Requires scipy. Runs are read with rfl/analysis/results.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


from rfl.analysis.report import apply_filters
from rfl.analysis.results import HP_COLUMNS, ID_COLUMNS, load_runs, varying_columns


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (W+ - W-) / (W+ + W-) on non-zero diffs."""
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    w_plus, w_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


def paired_values(a: pd.DataFrame, b: pd.DataFrame,
                  metric: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """Values of two levels on their common seeds where both are finite.

    Raises:
        ValueError: if a level holds two runs of the same seed. They differ in more
            than the seed, and pairing would pick one of them arbitrarily.
    """
    for name, d in (("first", a), ("second", b)):
        if d["seed"].duplicated().any():
            raise ValueError(
                f"the {name} level holds several runs per seed (configurations "
                f"{sorted(d['config_id'].unique()) if 'config_id' in d else '?'}). Add "
                "grouping columns (--by) or filter the runs (--where).")
    ma = a.set_index("seed")[metric]
    mb = b.set_index("seed")[metric]
    seeds = [s for s in sorted(set(ma.index) & set(mb.index))
             if np.isfinite(ma[s]) and np.isfinite(mb[s])]
    return ma.loc[seeds].to_numpy(), mb.loc[seeds].to_numpy(), seeds


def paired_pairwise(df: pd.DataFrame, metric: str, factor: str, by: List[str], *,
                    baseline: Optional[str] = None, min_pairs: int = 6) -> pd.DataFrame:
    """Pairwise paired Wilcoxon tests between the levels of ``factor`` within each group.

    Args:
        df: per-run table from :func:`analysis.results.load_runs`.
        metric: column compared (higher is better for the win rate).
        factor: column whose levels are compared.
        by: columns defining the groups.
        baseline: if given and present in a group, compare every level to it only.
        min_pairs: pairs with fewer common seeds are reported without a test.

    Returns:
        One row per (group, level, other level), with ``median_diff``, ``mean_diff``,
        ``win_rate``, ``rank_biserial``, ``W``, ``p_value`` and ``p_holm``. Differences
        are ``level - other``.
    """
    rows: List[dict] = []
    for gkey, g in df.groupby(by, dropna=False) if by else [((), df)]:
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        levels = sorted(g[factor].dropna().unique(), key=str)
        if len(levels) < 2:
            continue
        pairs = ([(a, baseline) for a in levels if a != baseline]
                 if baseline in levels else list(combinations(levels, 2)))
        group_rows = []
        for a, b in pairs:
            va, vb, seeds = paired_values(g[g[factor] == a], g[g[factor] == b], metric)
            rec = dict(zip(by, gkey), metric=metric, level=a, vs=b, n_pairs=len(seeds))
            if len(seeds) < min_pairs:
                rec.update(p_value=np.nan, note=f"fewer than {min_pairs} pairs")
                group_rows.append(rec)
                continue
            diff = va - vb
            try:
                res = wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox")
                W, p = float(res.statistic), float(res.pvalue)
            except ValueError:                # every difference is zero
                W, p = np.nan, 1.0
            rec.update(median_diff=float(np.median(diff)), mean_diff=float(diff.mean()),
                       win_rate=float(np.mean(diff > 0)), rank_biserial=rank_biserial(diff),
                       W=W, p_value=p, note="")
            group_rows.append(rec)
        tested = [r for r in group_rows if np.isfinite(r["p_value"])]
        for r, padj in zip(tested, holm_bonferroni([r["p_value"] for r in tested])):
            r["p_holm"] = padj
        rows += group_rows
    cols = by + ["metric", "level", "vs", "n_pairs", "median_diff", "mean_diff",
                 "win_rate", "rank_biserial", "W", "p_value", "p_holm", "note"]
    return pd.DataFrame(rows, columns=cols)


def friedman_omnibus(df: pd.DataFrame, metric: str, factor: str, by: List[str], *,
                     min_common: int = 6) -> Tuple[pd.DataFrame, Dict[Tuple, pd.DataFrame]]:
    """Friedman test across the levels of ``factor`` within each group of three or more.

    Returns:
        A summary table (``chi2``, ``p_value``, mean ranks, 1 = best) and a dict of
        Nemenyi p-value matrices per group, empty if scikit-posthocs is missing.
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        sp = None
    rows, nemenyi = [], {}
    for gkey, g in df.groupby(by, dropna=False) if by else [((), df)]:
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        if g[factor].nunique() < 3:
            continue
        if g.duplicated(["seed", factor]).any():
            raise ValueError("a level holds several runs per seed: add grouping "
                             "columns (--by) or filter the runs (--where).")
        wide = g.pivot_table(index="seed", columns=factor, values=metric,
                             aggfunc="first").dropna(how="any")
        rec = dict(zip(by, gkey), metric=metric, n_levels=wide.shape[1],
                   n_common_seeds=len(wide))
        if len(wide) < min_common:
            rec.update(chi2=np.nan, p_value=np.nan, note=f"fewer than {min_common} seeds")
            rows.append(rec)
            continue
        res = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
        ranks = wide.rank(axis=1, ascending=False).mean(axis=0).sort_values()
        rec.update(chi2=float(res.statistic), p_value=float(res.pvalue),
                   mean_ranks=json.dumps({str(k): round(v, 2) for k, v in ranks.items()}),
                   note="")
        rows.append(rec)
        if sp is not None:
            mat = sp.posthoc_nemenyi_friedman(wide.to_numpy())
            mat.index = mat.columns = wide.columns
            nemenyi[gkey] = mat
    return pd.DataFrame(rows), nemenyi


def print_pairwise(pw: pd.DataFrame, by: List[str]) -> None:
    for gkey, g in pw.groupby(by, dropna=False) if by else [((), pw)]:
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        print("\n[" + (", ".join(f"{c}={v}" for c, v in zip(by, gkey)) or "all runs") + "]")
        for _, r in g.iterrows():
            head = f"  {str(r['level']):>14} vs {str(r['vs']):<14} n={int(r['n_pairs'])}"
            if not np.isfinite(r["p_value"]):
                print(f"{head}  ({r['note']})")
                continue
            star = "*" if r["p_holm"] < 0.05 else " "
            print(f"{head}  median diff={r['median_diff']:+.4f}  win={r['win_rate']:.2f}  "
                  f"r={r['rank_biserial']:+.2f}  p={r['p_value']:.4f}  "
                  f"p_holm={r['p_holm']:.4f} {star}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="Directories searched for result files.")
    p.add_argument("--metric", default="test_acc")
    p.add_argument("--factor", default="aggregation", choices=ID_COLUMNS,
                   help="Column whose levels are compared.")
    p.add_argument("--by", nargs="+", default=None, choices=ID_COLUMNS,
                   help="Group columns (default: every varying column except the factor, "
                        "the seed, the plan arm and the hyperparameters).")
    p.add_argument("--where", nargs="+", default=[], metavar="COL=VALUE")
    p.add_argument("--baseline", default=None,
                   help="Compare every level to this one only (default: all pairs).")
    p.add_argument("--min-pairs", type=int, default=6)
    p.add_argument("--out-dir", default=None, help="Write the result tables as CSV here.")
    args = p.parse_args(argv)

    df = apply_filters(load_runs(*args.roots), args.where)
    if args.factor == "eps":
        # A non-private run is eps = inf: compare it with the private levels.
        df["eps"] = df["eps"].fillna(float("inf"))
    # Hyperparameters and plan arms differ between levels by design: not group columns.
    exclude = ("seed", args.factor, "arm", *HP_COLUMNS)
    by = args.by if args.by is not None else varying_columns(df, exclude=exclude)
    baseline = args.baseline
    if baseline is not None and df[args.factor].dtype.kind == "f":
        baseline = float(baseline)

    try:
        pw = paired_pairwise(df, args.metric, args.factor, by, baseline=baseline,
                             min_pairs=args.min_pairs)
        fr, nemenyi = friedman_omnibus(df, args.metric, args.factor, by,
                                       min_common=args.min_pairs)
    except ValueError as e:
        raise SystemExit(f"stats: {e}") from None
    print(f"Paired Wilcoxon on {args.metric} across {args.factor}, Holm-corrected per "
          f"group (* p_holm < 0.05), {len(df)} runs")
    print_pairwise(pw, by)
    if not fr.empty:
        print(f"\nFriedman test across {args.factor}")
        print(fr.drop(columns=["metric"]).to_string(index=False))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        pw.to_csv(os.path.join(args.out_dir, f"pairwise_{args.metric}.csv"), index=False)
        fr.to_csv(os.path.join(args.out_dir, f"friedman_{args.metric}.csv"), index=False)
        for gkey, mat in nemenyi.items():
            tag = "_".join(str(v) for v in gkey) or "all"
            mat.to_csv(os.path.join(args.out_dir, f"nemenyi_{args.metric}_{tag}.csv"))
        print(f"\nTables written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
