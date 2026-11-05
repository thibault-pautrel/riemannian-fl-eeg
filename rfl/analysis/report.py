#!/usr/bin/env python3
"""Summary tables and learning curves from finished runs.

Reads every ``fl_results.json`` and ``centralized_results.json`` under the given
directories (see rfl/analysis/results.py), groups the runs and prints the mean, the standard
deviation (ddof = 1) and the number of runs of a metric for each group.

By default the runs are grouped by every identification column that varies, except the
seed. ``--by`` sets the grouping explicitly, and ``--pivot`` spreads one column across the
table (for instance one column per privacy budget). ``--where`` keeps a subset of runs.

Usage (from the repository root)::

    python -m rfl.analysis.report results/federated
    python -m rfl.analysis.report results --where kind=federated participation=1 \\
        --by dataset arch scope --pivot eps
    python -m rfl.analysis.report results --metric refit_test_acc --csv summary.csv
    python -m rfl.analysis.report results/federated --curves figures/ --curve-figure dataset arch

``--runs runs.csv`` writes the full per-run table, for further analysis in pandas.
``--curves DIR`` saves one PNG per figure group (``--curve-figure``), with one line per
remaining group: mean over seeds of the per-round (or per-epoch) test accuracy, with a
band of one standard deviation.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Sequence

import numpy as np
import pandas as pd


from rfl.analysis.results import ID_COLUMNS, load_curve, load_runs, varying_columns


def _show(v) -> str:
    """Display value of a table entry: '-' for a missing value."""
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def apply_filters(df: pd.DataFrame, where: Sequence[str]) -> pd.DataFrame:
    """Keep the rows matching every ``column=value`` condition.

    Values are compared as strings after a numeric round trip, so ``eps=20`` matches
    ``20.0`` and ``eps=nan`` selects the non-private runs.
    """
    for cond in where:
        col, _, value = cond.partition("=")
        if col not in df.columns:
            raise SystemExit(f"--where: unknown column {col!r}")

        def norm(x):
            try:
                f = float(x)
                return "nan" if math.isnan(f) else f"{f:g}"
            except (TypeError, ValueError):
                return str(x)

        df = df[df[col].map(norm) == norm(value)]
    return df


def summarize(df: pd.DataFrame, by: List[str], metric: str) -> pd.DataFrame:
    """Mean, standard deviation and count of ``metric`` per group."""
    g = df.groupby(by, dropna=False)[metric]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    return out.sort_values(by).reset_index(drop=True)


def format_table(summary: pd.DataFrame, by: List[str], pivot: str | None,
                 scale: float, digits: int) -> str:
    """Text table with one ``mean ± std (n)`` cell per group."""
    def cell(row) -> str:
        if not row["n"]:
            return "-"
        std = row["std"] if row["n"] > 1 else 0.0
        return f"{scale * row['mean']:.{digits}f} ± {scale * std:.{digits}f} ({int(row['n'])})"

    s = summary.copy()
    s["value"] = s.apply(cell, axis=1)
    if pivot:
        index = [c for c in by if c != pivot]
        s = s.pivot_table(index=index or None, columns=pivot, values="value",
                          aggfunc="first", dropna=False).reset_index()
        s.columns = ["-" if str(c) == "nan" else str(c) for c in s.columns]
    else:
        s = s[by + ["value"]]
    return s.astype(object).where(s.notna(), "-").to_string(index=False)


def _smooth(curve: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over `window` points (shorter at the ends)."""
    if window <= 1:
        return curve
    half = window // 2
    return np.array([curve[max(0, i - half):i + half + 1].mean()
                     for i in range(len(curve))])


def plot_curves(df: pd.DataFrame, by: List[str], figure_by: List[str], out_dir: str,
                key: str, scale: float, smooth: int = 1) -> List[str]:
    """Save one PNG per ``figure_by`` group, one mean curve per remaining group.

    With ``smooth`` > 1 each seed's curve is smoothed by a centred moving average over
    that many rounds before averaging.

    Returns:
        The paths of the written figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    line_by = [c for c in by if c not in figure_by]
    written = []
    for fig_key, fig_df in df.groupby(figure_by, dropna=False) if figure_by else [((), df)]:
        fig_key = fig_key if isinstance(fig_key, tuple) else (fig_key,)
        fig, ax = plt.subplots(figsize=(6, 4))
        groups = fig_df.groupby(line_by, dropna=False) if line_by else [((), fig_df)]
        for line_key, line_df in groups:
            curves = [c for c in (load_curve(p, key) for p in line_df["path"]) if c.size]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            stack = scale * np.stack([_smooth(c[:n], smooth) for c in curves])
            mean, std = stack.mean(0), stack.std(0, ddof=1) if len(curves) > 1 else 0 * stack[0]
            x = np.arange(1, n + 1)
            line_key = line_key if isinstance(line_key, tuple) else (line_key,)
            label = ", ".join(f"{c}={_show(v)}" for c, v in zip(line_by, line_key)) or "all"
            ax.plot(x, mean, label=f"{label} (n={len(curves)})")
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("round" if (fig_df["kind"] == "federated").all() else "round / epoch")
        ax.set_ylabel(key + (" (%)" if scale == 100.0 else ""))
        ax.set_title(", ".join(f"{c}={_show(v)}" for c, v in zip(figure_by, fig_key)))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        name = "_".join(str(v) for v in fig_key) or "curves"
        path = os.path.join(out_dir, f"{name}_{key}.png".replace("/", "-"))
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="Directories searched for result files.")
    p.add_argument("--metric", default="test_acc",
                   help="Column summarised (default: test_acc). See rfl/analysis/results.py.")
    p.add_argument("--by", nargs="+", default=None, choices=ID_COLUMNS,
                   help="Grouping columns (default: every varying column except seed).")
    p.add_argument("--pivot", default=None, choices=ID_COLUMNS,
                   help="Column spread across the table.")
    p.add_argument("--where", nargs="+", default=[], metavar="COL=VALUE",
                   help="Keep only the runs matching all conditions.")
    p.add_argument("--raw", action="store_true",
                   help="Print metrics as stored (default: accuracies and F1 in percent).")
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--csv", default=None, help="Write the summary table to this CSV file.")
    p.add_argument("--runs", default=None, help="Write the per-run table to this CSV file.")
    p.add_argument("--smooth", type=int, default=1,
                   help="Moving-average window (rounds) applied to each curve.")
    p.add_argument("--curves", default=None, metavar="DIR",
                   help="Save learning curves (PNG) into this directory.")
    p.add_argument("--curve-key", default="test_acc",
                   help="Curve to plot: test_acc, test_loss, val_acc or val_loss.")
    p.add_argument("--curve-figure", nargs="+", default=["dataset", "arch"],
                   choices=ID_COLUMNS, help="Columns defining one figure each.")
    args = p.parse_args(argv)

    df = apply_filters(load_runs(*args.roots), args.where)
    if df.empty:
        raise SystemExit("no run left after --where")
    by = args.by or varying_columns(df) or ["kind"]
    if args.pivot and args.pivot not in by:
        by = by + [args.pivot]

    percent = not args.raw and (args.metric.endswith("_acc") or args.metric.endswith("_f1"))
    scale = 100.0 if percent else 1.0
    from rfl.analysis.results import duplicate_seeds
    dups = duplicate_seeds(df, by)
    if len(dups):
        print(f"[warn] {len(dups)} (group, seed) pairs hold several runs: these groups "
              "mix configurations that differ in more than the seed. Add columns to "
              "--by (for example lr batch_size horizon) or filter with --where. "
              f"First one: {dups.iloc[0].to_dict()}")
    summary = summarize(df, by, args.metric)
    unit = " (%)" if percent else ""
    print(f"{args.metric}{unit}: mean ± std (number of runs), {len(df)} runs\n")
    print(format_table(summary, by, args.pivot, scale, args.digits))

    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"\nSummary written to {args.csv}")
    if args.runs:
        df.to_csv(args.runs, index=False)
        print(f"Per-run table written to {args.runs}")
    if args.curves:
        curve_scale = 100.0 if (not args.raw and args.curve_key.endswith("_acc")) else 1.0
        paths = plot_curves(df, by, [c for c in args.curve_figure if c in by] or [],
                            args.curves, args.curve_key, curve_scale, args.smooth)
        print(f"{len(paths)} figure(s) written to {args.curves}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
