#!/usr/bin/env python3
"""
=============================================================================
Green Software Metrics Survey - descriptive figure generator
=============================================================================
Draws the ranked item profiles of Section V-A as TikZ bar charts and writes
one file per chart to  ../../figures/generated/*.tex , each of which the paper
\\input's inside its own column-width figure.  Every bar length and every
printed percentage is computed from the survey export here, so a data cut
redraws the figures and no number in a figure is typed by hand.

One file per chart rather than grouped panels: a column-width figure is placed
in the column it is declared in, where a figure* is deferred to the top of a
later page, and a chart the author would rather state in prose can be dropped
without disturbing the others.

All eight charts share one geometry: one label column, one bar origin, one
percentage scale.  The scale is the reason -- per-chart scaling made a 30% bar
in one figure as long as a 59% bar in the next, so bar lengths could not be
compared across figures.  The shared label column and origin follow from it, so
the eight charts read as one instrument rather than eight.

Why TikZ rather than an exported image: the two research-model figures are
TikZ, so the fonts, rules and greys match, the output is a text diff a reviewer
can read, and the repository stays free of binary artifacts.

The item sets, column indices and tier definitions come from config.py and the
tier ladder from hypothesis_test.build_dimension_scores, so a figure cannot
disagree with a test about what an item is.  Only the display labels below are
defined here: they are typography, not data.

Usage:
    python gen_figures.py [path/to/export.csv]

Run it after every data cut, in the same pass as gen_latex.py, then recompile.
=============================================================================
"""
import os
import re
import sys

import pandas as pd

import config
import hypothesis_test as H
from config import (
    SURVEY_DIMENSIONS, ROLE_COLS, DOMAIN_COLS,
    DO_TIER_PROXY, DO_TIER_PHYSICAL, DO_TIER_ENVIRONMENTAL,
    CAN_OTHER_BARRIER_COL,
)

CAN_OTHER_ENABLER_COL = 70   # free-text "Other" beside the six listed enablers

# =============================================================================
# GEOMETRY
# All lengths in cm.  A chart is laid out to fill one text column of the
# document class in use, so COLUMN_W must match its \columnwidth: 8.75 cm under
# IEEEtran, 238.25444pt = 8.374 cm under cas-dc.  Leaving the old value behind
# after the class change made every chart overhang its column by 10.7pt.  Probe
# the current width with \typeout{\the\columnwidth} inside chapters/results.tex.
# =============================================================================
COLUMN_W  = 8.37    # cas-dc single-column width (\columnwidth = 238.25pt)
ROW_PITCH = 0.36    # vertical distance between two bars
BAR_H     = 0.24    # bar thickness
LABEL_GAP = 0.10    # between the item label and the start of its bar
VALUE_W   = 0.62    # reserved to the right of a bar for its printed value
GROUP_GAP = 0.20    # extra space where one chart is split into tiers
RULE      = "0.45pt"   # matches figures/tikz_research_model.tex

# One grey for every bar in every chart.  Monochrome by decision (see the
# research-model figures): a printed greyscale copy and a colour-blind reader
# must both keep every distinction.  An earlier version shaded the metric tiers
# light/mid/dark, which read top-to-bottom as a gradient rather than as three
# categories; fig:metrics now groups its bars by tier instead.
PLAIN = "gray!55"
RULE_GREY = "gray!70"   # baseline and tier separators

# =============================================================================
# DISPLAY LABELS  (typography only -- the item keys come from config.py)
# =============================================================================
LBL_SIZE = {"1\u201349": "1--49", "50\u2013249": "50--249", "250\u2013999": "250--999",
            "1000\u20134999": "1{,}000--4{,}999", "5000+": "5{,}000+"}
LBL_DOMAIN = {
    "Business_Enterprise": "Business/Enterprise", "Platform_Infra": "Platform/Infrastructure",
    "Web": "Web", "IoT_Embedded": "IoT/Embedded", "Mobile": "Mobile",
}
LBL_ROLE = {"Management": "Management", "Technical": "Technical",
            "Non-Technical": "Non-technical"}
LBL_BARRIER = {
    "Lack_ground_truth": "No ground-truth power data", "Unclear_resp": "Unclear responsibilities",
    "Skills_gaps": "Skills gaps", "Restricted_access": "Restricted provider access",
    "Lack_methods": "No standardized methods", "Data_integration": "Data integration",
    "Cost_instrument": "Cost of instrumentation", "High_tooling": "Tooling overhead",
    "Inconsistent_results": "Inconsistent results", "Other": "Other (free text)",
}
LBL_ENABLER = {
    "Std_metrics": "Standardized metrics", "Procedures": "Clear procedures",
    "Ref_dashboards": "Reference dashboards", "Granularity": "Finer granularity",
    "CI_CD_patterns": "CI/CD patterns", "Uncertainty_report": "Uncertainty reporting",
    "Other": "Other (free text)",
}
LBL_METRIC = {
    "CPU_util": "CPU utilization", "Memory_util": "Memory", "Network_util": "Network",
    "Latency": "Latency", "Disk_IO": "Disk I/O", "Throughput": "Throughput",
    "Error_rate": "Error rate", "Cost_proxy": "Cost proxy",
    "Energy_kWh": "Energy (kWh)", "Power_W": "Power draw (W)",
    "CO2e": r"{\coo}", "Water": "Water", "SCI": "SCI",
}
LBL_AREA = {
    "Sustainability_reporting": "Sustainability reporting", "Architecture": "Architecture",
    "Hardware": "Hardware", "Cloud_region": "Cloud region", "Right_sizing": "Right-sizing",
    "Libraries": "Libraries", "Source_code": "Source code", "Procurement": "Procurement",
    "Requirements": "Requirements",
}
LBL_TIER = {0: "No metric tracked (Tier 0)", 1: "Resource proxies only (Tier 1)",
            2: "+ physical energy (Tier 2)", 3: "+ environmental (Tier 3)"}

METRIC_BAND = {**{k: "proxy" for k in DO_TIER_PROXY},
               **{k: "physical" for k in DO_TIER_PHYSICAL},
               **{k: "environmental" for k in DO_TIER_ENVIRONMENTAL}}


# =============================================================================
# DATA
# =============================================================================
def _pct(k, n):
    return 100.0 * k / n


def _shares(df, cols, extra=None):
    """[(key, pct)] for a Yes/No multi-select item set, unordered."""
    n = len(df)
    rows = [(key, _pct(int((df.iloc[:, c] == "Yes").sum()), n)) for key, c in cols.items()]
    if extra is not None:                      # free-text "Other" column
        col, key = extra
        txt = df.iloc[:, col].dropna().astype(str).str.strip()
        rows.append((key, _pct(int(txt.ne("").sum()), n)))
    return rows


def multiselect_rows(df, cols, labels, extra=None):
    """Ranked [(label, pct)] for a Yes/No multi-select item set."""
    rows = sorted(_shares(df, cols, extra), key=lambda r: r[1], reverse=True)
    return [(labels[k], p) for k, p in rows], ()


def tiered_metric_rows(df, cols, labels):
    """The metric items ranked *within* their tier, proxies first.

    The tiers used to be a light/mid/dark shading of one ranked list, which read
    as a gradient rather than as three kinds of thing, and put a pale proxy bar
    (cost) below dark environmental ones.  Grouping carries the same three
    tiers with a single grey, so this chart matches the other seven.  The
    returned break indices are the rows that open a new tier.
    """
    shares = dict(_shares(df, cols))
    rows, breaks = [], set()
    for tier in ("proxy", "physical", "environmental"):
        block = sorted(((k, shares[k]) for k in cols if METRIC_BAND[k] == tier),
                       key=lambda r: r[1], reverse=True)
        if rows:
            breaks.add(len(rows))
        rows.extend((labels[k], p) for k, p in block)
    return rows, breaks


def ordinal_rows(df, col, order, labels, n):
    counts = df.iloc[:, col].value_counts()
    return [(labels[lbl], _pct(int(counts.get(lbl, 0)), n)) for lbl in order], ()


def build_charts(df):
    n = len(df)
    scores = H.build_dimension_scores(df)
    breadth = scores["do_breadth"].value_counts()

    size_cfg = SURVEY_DIMENSIONS["context_size"]
    barriers = {k: v for k, v in SURVEY_DIMENSIONS["can_barriers"]["items"].items()
                if k != "No_barriers"}          # sentinel, not a barrier

    return {
        # One chart per file, and one float per chart in the paper.  A figure*
        # can only be placed at the top of a page, which pushed the earlier
        # grouped versions away from the text they serve; a column-width figure
        # sits in the column it is declared in.  Splitting also means a chart
        # can be commented out and replaced by prose without touching the rest.
        # The chart carries no title of its own -- the caption names it.
        # Each entry is (rows, break indices); only fig:metrics has breaks.
        "size": ordinal_rows(df, list(size_cfg["items"].values())[0], size_cfg["order"],
                             LBL_SIZE, n),
        "domain": multiselect_rows(df, DOMAIN_COLS, LBL_DOMAIN),
        "role": multiselect_rows(df, ROLE_COLS, LBL_ROLE),
        "barriers": multiselect_rows(df, barriers, LBL_BARRIER,
                                     extra=(CAN_OTHER_BARRIER_COL, "Other")),
        "enablers": multiselect_rows(df, SURVEY_DIMENSIONS["can_enablers"]["items"],
                                     LBL_ENABLER, extra=(CAN_OTHER_ENABLER_COL, "Other")),
        "metrics": tiered_metric_rows(df, SURVEY_DIMENSIONS["do_metrics"]["items"],
                                      LBL_METRIC),
        "coverage": ([(LBL_TIER[t], _pct(int(breadth.get(t, 0)), n))
                      for t in (0, 1, 2, 3)], ()),
        "areas": multiselect_rows(df, SURVEY_DIMENSIONS["outlook_impact"]["items"], LBL_AREA),
    }


# =============================================================================
# DRAWING
# =============================================================================
# Labels carry TeX markup, which sets no width.  \coo is the paper's own macro
# for the emission unit, used here so the figure and the prose cannot drift
# apart on how it is written.
_MACRO_WIDTH = {r"\coo": "CO2-eq"}


def vislen(label):
    """Printed length of a label, ignoring markup that occupies no width."""
    for macro, shown in _MACRO_WIDTH.items():
        label = label.replace(macro, shown)
    return len(re.sub(r"\\[a-zA-Z]+|[{}]", "", label))


def label_width(charts):
    """Width reserved for the item labels, from the longest label in any chart.

    One width for all eight charts, so every bar starts at the same x and the
    charts stack as a grid rather than as eight unrelated pictures.

    A crude per-character estimate rather than a real text measurement: TeX
    sets the labels, so this only has to be generous enough that no bar starts
    inside a label.  Verified by the overfull-box check on the build.
    """
    longest = max(vislen(lbl) for rows, _ in charts.values() for lbl, _ in rows)
    return round(min(0.105 * longest + 0.10, 3.30), 2)


def scale_max(charts):
    """The percentage the bars run to, shared by all eight charts.

    Per-chart scaling made bar length meaningless across figures: a 30% bar
    filled one chart as completely as a 59% bar filled the next.  One scale,
    the next full ten percent above the largest share anywhere, keeps the bars
    long enough to read while making them comparable from figure to figure.
    """
    return 10 * (int(max(p for rows, _ in charts.values() for _, p in rows) / 10) + 1)


def _row_ys(nrows, breaks):
    """Baseline y of each row, with GROUP_GAP inserted before each tier break."""
    ys, offset = [], 0.0
    for i in range(nrows):
        if i in breaks:
            offset += GROUP_GAP
        y = -(i * ROW_PITCH + offset)
        ys.append(0.0 if y == 0 else y)     # keep the top row as 0.00, not -0.00
    return ys


def chart(rows, lw, pxmax, breaks=()):
    """One horizontal bar chart, the width of a text column.

    Each chart is its own float with its own caption, so it carries no title of
    its own.  The scale and the label column are passed in and are the same for
    every chart, and each bar prints its own value, so nothing has to be read
    off a gridline and no axis is drawn.  A chart split into tiers gets a gap
    and a hairline where one tier ends and the next begins.
    """
    bx = lw + LABEL_GAP
    px = COLUMN_W - VALUE_W                 # right edge of the plotting area
    sx = (px - bx) / pxmax
    if sx <= 0:
        raise ValueError("labels do not fit a column -- shorten them")

    ys = _row_ys(len(rows), set(breaks))
    bottom = ys[-1] - BAR_H

    lines = [f"\\begin{{tikzpicture}}[line width={RULE}]",
             _bbox(COLUMN_W, BAR_H, bottom)]
    for i, (lbl, pct) in enumerate(rows):
        y = ys[i]
        w = pct * sx
        lines.append(f"  \\node[anchor=east,font=\\scriptsize] at ({lw:.2f}cm,{y:.2f}cm) {{{lbl}}};")
        lines.append(f"  \\fill[{PLAIN}] ({bx:.2f}cm,{y - BAR_H / 2:.2f}cm) "
                     f"rectangle ({bx + w:.2f}cm,{y + BAR_H / 2:.2f}cm);")
        lines.append(f"  \\node[anchor=west,font=\\scriptsize] at ({bx + w + 0.06:.2f}cm,{y:.2f}cm) "
                     f"{{{pct:.0f}\\%}};")
    lines.append(f"  \\draw[{RULE_GREY}] ({bx:.2f}cm,{BAR_H:.2f}cm) -- "
                 f"({bx:.2f}cm,{bottom:.2f}cm);")
    for i in sorted(set(breaks)):
        y = (ys[i - 1] - BAR_H / 2 + ys[i] + BAR_H / 2) / 2
        lines.append(f"  \\draw[{RULE_GREY}] ({bx:.2f}cm,{y:.2f}cm) -- ({px:.2f}cm,{y:.2f}cm);")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def _bbox(width, top, bottom):
    """Pin the picture to the printable width.

    Label widths are estimated rather than measured by TeX, so a long label can
    overhang by a fraction of a millimetre.  Without this the float reports an
    overfull box on a difference no reader can see.
    """
    return (f"  \\path[use as bounding box] (0cm,{top:.2f}cm) "
            f"rectangle ({width:.2f}cm,{bottom:.2f}cm);")


HEADER = ("% =============================================================\n"
          "% AUTO-GENERATED by replication/scripts/gen_figures.py -- DO NOT EDIT BY HAND.\n"
          "% Regenerate after each data cut:  python gen_figures.py\n"
          "% Bar lengths and printed values are computed from the survey\n"
          "% export; the caption lives with the float in chapters/results.tex.\n"
          "% All eight descriptive charts share one label column and one\n"
          "% 0--{pxmax}% scale, so bar lengths compare across figures.\n"
          "% =============================================================\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv = args[0] if args else H.DEFAULT_CSV
    df = config.load_survey(csv)
    out_dir = config.FIGURES_TEX_DIR
    os.makedirs(out_dir, exist_ok=True)
    charts = build_charts(df)
    lw = label_width(charts)          # one label column for all eight charts
    pxmax = scale_max(charts)         # one percentage scale for all eight
    for name, (rows, breaks) in charts.items():
        path = os.path.join(out_dir, f"fig_{name}.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(HEADER.format(pxmax=pxmax) + chart(rows, lw, pxmax, breaks))
        print(f"wrote {path}  ({len(rows)} bars, n={len(df)})")
    print(f"shared geometry: label column {lw:.2f}cm, bars scaled 0--{pxmax}%")


if __name__ == "__main__":
    main()
