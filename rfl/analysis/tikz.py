"""Writer of pgfplots figures (TikZ) with the data inlined, in the style of the DP-RFedProj paper.

`grid_figure` writes one `tikzpicture` with a grid of log-log panels (pgfplots
`groupplots`) and a single legend below the figure, on one line. The file only needs, in
the preamble,

    \\usepackage{pgfplots}
    \\usepgfplotslibrary{groupplots}

The colours C0, C3 and C7 are provided if the document does not define them.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

COLOURS = {"C0": "1B6CA8", "C1": "E1701A", "C2": "2C8B57", "C3": "A02C2C", "C7": "444444"}

# Style of each kind of series.
STYLES = {
    "ours": "C0, thick, mark=*, mark size=1.6pt",
    "prirfed": "C3, thick, mark=triangle*, mark size=2pt",
    "prediction": "C7, thick, dashed",
    "nonprivate": "C7, thick, dotted",
    "rlavg": "C2, thick, mark=square, mark size=1.8pt",
    "retavg": "C1, thick, mark=diamond*, mark size=2.2pt",
}


def _num(v: float) -> str:
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.6g}"


def _series(s: Dict) -> List[str]:
    style = STYLES.get(s["style"], s["style"]) + ", forget plot"
    if "err_lo" in s:
        lines = [f"\\addplot+[{style}, error bars/.cd, y dir=both, y explicit] coordinates {{"]
        lines += [f"  ({_num(x)},{_num(y)}) += (0,{_num(hi)}) -= (0,{_num(lo)})"
                  for x, y, lo, hi in zip(s["x"], s["y"], s["err_lo"], s["err_hi"])]
    else:
        lines = [f"\\addplot[{style}] coordinates {{"]
        lines += [f"  ({_num(x)},{_num(y)})" for x, y in zip(s["x"], s["y"])]
    return lines + ["};"]


def grid_figure(path: Path, panels: List[Dict], legend: Sequence[Tuple[str, str]], *,
                cols: int = 2, width: str = "0.47\\linewidth", height: str = "4.4cm",
                legend_name: str = "synthlegend", comment: str = "") -> None:
    """Write a grid of panels with one legend below the figure, on one line.

    Each panel is a dict with `title`, `xlabel`, `ylabel`, `xticks`, `ymin`, `ymax` and
    `series`, and optionally `xlog` and `ylog` (default True). A series has `style` (a key
    of STYLES), `x`, `y` and optionally `err_lo`, `err_hi` (error bars). `legend` lists
    (label, style) pairs.
    """
    rows = (len(panels) + cols - 1) // cols
    lines = [f"% {comment}",
             "% Needs \\usepackage{pgfplots} and \\usepgfplotslibrary{groupplots}."]
    lines += [f"\\providecolor{{{c}}}{{HTML}}{{{h}}}" for c, h in COLOURS.items()]
    lines += [
        "\\begin{tikzpicture}",
        "\\begin{groupplot}[",
        f"  group style={{group size={cols} by {rows}, horizontal sep=1.6cm, "
        "vertical sep=1.9cm},",
        f"  width={width}, height={height},",
        "  tick label style={font=\\scriptsize}, label style={font=\\small},",
        "  title style={font=\\small},",
        "  grid=major, grid style={black!10},",
        "  legend style={font=\\small, draw=none, legend columns=-1, column sep=1.2ex},",
        "  legend cell align=left,",
        "]",
    ]
    for i, pan in enumerate(panels):
        opts = [f"title={{{pan['title']}}}",
                f"xlabel={{{pan['xlabel']}}}", f"ylabel={{{pan['ylabel']}}}",
                f"ymin={_num(pan['ymin'])}", f"ymax={_num(pan['ymax'])}",
                "xtick={" + ",".join(_num(t) for t in pan["xticks"]) + "}",
                "xticklabels={" + ",".join(f"{t:g}" for t in pan["xticks"]) + "}",
                "minor xtick={}", "xmode=" + ("log" if pan.get("xlog", True) else "linear"),
                "ymode=" + ("log" if pan.get("ylog", True) else "linear")]
        for key in ("xmin", "xmax"):
            if key in pan:
                opts.append(f"{key}={_num(pan[key])}")
        if i == 0:
            opts.append(f"legend to name={legend_name}")
        lines.append("\\nextgroupplot[" + ", ".join(opts) + "]")
        for s in pan["series"]:
            lines += _series(s)
        if i == 0:
            for label, style in legend:
                lines.append(f"\\addlegendimage{{{STYLES[style]}}}")
                lines.append(f"\\addlegendentry{{{label}}}")
    lines += ["\\end{groupplot}",
              f"\\path (group c1r{rows}.outer south) -- (group c{cols}r{rows}.outer south)",
              "  node[midway, anchor=north, yshift=-3pt] "
              f"{{\\pgfplotslegendfromname{{{legend_name}}}}};",
              "\\end{tikzpicture}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
