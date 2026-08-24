#!/usr/bin/env python3
"""
Response funnel and mid-survey dropout bias.

Backs the external-validity claim in chapters/discussion.tex: that dropouts
matched completers on maturity, size and role composition but were markedly
more likely to fall in the narrow domain-breadth group.  The paper states that
direction in words and prints no figure, so this analysis is not part of the
macro pipeline -- but the claim is empirical, so the numbers behind it must
stay runnable.

Reads the '-all' export (incomplete responses included); every other analysis
in the package reads the completes-only export.

Usage:  python dropout_analysis.py [path/to/export.csv]
        The '-all' companion is derived from that path by suffix.
"""
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

import config
from config import (
    CANONICAL_COLUMNS, CONTEXT_MATURITY_ORDINAL_MAP, CONTEXT_SIZE_GROUPS,
    DOMAIN_COLS, ROLE_COLS, SURVEY_DIMENSIONS, load_survey,
)

DEFAULT_CSV = config.data_path("results-survey668719.csv")


def load_all_responses(csv_path):
    """The '-all' export, aligned to CANONICAL_COLUMNS.

    Fails loudly rather than degrading: a missing or mismatched export would
    otherwise produce a dropout comparison against the wrong data cut.
    """
    all_path = csv_path.replace(".csv", "-all.csv")
    if not os.path.exists(all_path):
        raise FileNotFoundError(
            f"full export not found: {all_path}\n"
            "  This analysis needs the LimeSurvey export that includes "
            "incomplete responses ('-all.csv'), alongside the completes-only one.")
    da = pd.read_csv(all_path, encoding="utf-8-sig")
    missing = [h for h in CANONICAL_COLUMNS if h not in da.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} expected column(s) missing in {all_path} "
            f"(first: {missing[0]!r}) -- exports from different questionnaires?")
    return da.loc[:, CANONICAL_COLUMNS]


def domain_count(d):
    """Number of application domains ticked, per respondent."""
    return pd.concat([(d.iloc[:, c] == "Yes").astype(int)
                      for c in DOMAIN_COLS.values()], axis=1).sum(axis=1)


def fisher(a, b):
    """Two-sided Fisher exact on two boolean Series, as a 2x2 of counts."""
    return stats.fisher_exact([[int(a.sum()), int((~a).sum())],
                               [int(b.sum()), int((~b).sum())]])[1]


def fmt_p(p):
    """Three decimals, or '<.001' -- matching how the reports print p-values."""
    return "<.001" if p < .001 else f"= {p:.3f}"


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    df = load_survey(csv)
    da = load_all_responses(csv)

    last_page = pd.to_numeric(da["Last page"], errors="coerce")
    submitted = da["Date submitted"].notna()
    n_opened = len(da)
    n_started = int((last_page >= 1).sum())     # answered >= first question group
    n_completed = int(submitted.sum())
    if n_completed != len(df):
        raise ValueError(
            f"completes mismatch: {n_completed} submitted in the '-all' export "
            f"vs n={len(df)} in the analysis export -- different data cuts.")

    partial = da[(~submitted) & (last_page >= 1)]
    comp = da[submitted]

    print("\nRESPONSE FUNNEL")
    print(f"  opened                             {n_opened:>5}")
    print(f"  answered the first question group  {n_started:>5}")
    print(f"  completed                          {n_completed:>5}"
          f"   ({100 * n_completed / n_started:.1f}% of starters,"
          f" {100 * n_completed / n_opened:.1f}% of openers)")
    print(f"  mid-survey dropouts compared       {len(partial):>5}")

    # --- The one contrast that came out non-null: domain breadth -------------
    broad_c = domain_count(comp) >= 2
    broad_p = domain_count(partial) >= 2
    print("\nDROPOUTS vs COMPLETERS")
    print(f"  broad footprint (>=2 domains)   completers {100 * broad_c.mean():.1f}%"
          f"   dropouts {100 * broad_p.mean():.1f}%"
          f"   p {fmt_p(fisher(broad_c, broad_p))}")


    # --- The checks that came out null, reported so the claim is bounded -----
    mat_idx = SURVEY_DIMENSIONS["context_maturity"]["items"]["Sustainability_maturity"]
    mat_c = comp.iloc[:, mat_idx].fillna("None").map(CONTEXT_MATURITY_ORDINAL_MAP).dropna()
    mat_p = partial.iloc[:, mat_idx].fillna("None").map(CONTEXT_MATURITY_ORDINAL_MAP).dropna()
    p_maturity = stats.mannwhitneyu(mat_c, mat_p, alternative="two-sided")[1]

    size_idx = SURVEY_DIMENSIONS["context_size"]["items"]["Org_size"]
    sme = CONTEXT_SIZE_GROUPS["SME (1–249)"]
    p_size = fisher(comp.iloc[:, size_idx].isin(sme),
                    partial.iloc[:, size_idx].isin(sme))

    checks = [("sustainability maturity", p_maturity),
              ("organization size (SME / Enterprise)", p_size)]
    for label, rcol in ROLE_COLS.items():
        checks.append((f"role: {label}",
                       fisher(comp.iloc[:, rcol] == "Yes",
                              partial.iloc[:, rcol] == "Yes")))
    for label, p in checks:
        print(f"  {label:<62}p {fmt_p(p)}")

    print("\n  Domain breadth is the predictor that carries H3, and dropout occurred")
    print("  before the practice items, so the direction of the bias is identifiable")
    print("  but its mechanism is not.\n")


if __name__ == "__main__":
    main()
