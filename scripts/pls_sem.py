#!/usr/bin/env python3
"""
Full PLS-SEM fit, for inspection.

The model is defined once, in pls_bootstrap.py; this script imports it rather
than restating it, so the two cannot drift.  What it adds is everything the
bootstrap loop does not print: the outer model, reliability and validity for
the reflective block, formative collinearity, R^2, and goodness of fit.

The confidence intervals the paper reports come from pls_bootstrap.py, not from
the `bootstrap=True` run below -- plspm's own resampling is unseeded, so its
intervals move between runs.  Treat the intervals printed here as indicative
and the cached ones as authoritative.

Measurement and structural model: see make_cfg() in pls_bootstrap.py.

Usage:  python pls_sem.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pls_bootstrap as PB
from plspm.plspm import Plspm
from plspm.scheme import Scheme


def banner(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def main():
    data = PB.build_data()
    cfg = PB.make_cfg()
    print(f"Structural model on listwise-complete n={len(data)} "
          f"(excludes Want item non-response and maturity 'I do not know').")

    pls = Plspm(data, cfg, Scheme.PATH, iterations=300, tolerance=1e-7,
                bootstrap=True, bootstrap_iterations=500, processes=1)

    banner("MEASUREMENT MODEL — outer model (loadings for Mode A, weights for Mode B)")
    print(pls.outer_model().round(3).to_string())

    banner("RELIABILITY / VALIDITY (reflective: Want)")
    print(pls.unidimensionality().round(3).to_string())

    # Two indicators per formative block, so VIF reduces to 1/(1-r^2).
    def vif2(a, b):
        r = data[a].corr(data[b])
        return 1.0 / (1.0 - r ** 2)

    print("\n  Formative VIF (rule: < 5, ideally < 3):")
    print(f"    Do      (do_breadth, do_inst):   VIF = {vif2('do_breadth', 'do_inst'):.2f}")
    print(f"    Outlook (ol_intent, ol_aware):   VIF = {vif2('ol_intent', 'ol_aware'):.2f}")

    banner("STRUCTURAL MODEL — R^2 (endogenous constructs)")
    print(pls.inner_summary().round(3).to_string())

    banner("PATH COEFFICIENTS (direct) — unseeded intervals, see module docstring")
    print(pls.bootstrap().paths().round(3).to_string())

    banner("GOODNESS OF FIT")
    print(f"  GoF = {pls.goodness_of_fit():.3f}")

    print("\n  Note: the single-indicator blocks (Context, Can) have loading = 1 "
          "and AVE = 1 by construction.\n")


if __name__ == "__main__":
    main()
