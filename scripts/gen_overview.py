#!/usr/bin/env python3
"""Generate HypothesisTestOverview.md from the SAME scoring/test functions used
by hypothesis_test.py, so the reference table can never drift from the report.

Hypothesis numbers here are the PAPER's (H1-H8, chapters/hypotheses.tex), not
the run_* function names in hypothesis_test.py, which carry a historical offset
documented in that file's header.  The mapping is applied once, in H_NUMBER
below, so the two numbering schemes meet in exactly one place.

Usage:  python gen_overview.py [path/to/export.csv]
Writes: ../reports/HypothesisTestOverview.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hypothesis_test as H
import config
from config import load_survey

OUT = config.report_path("HypothesisTestOverview.md")

# Paper hypothesis numbers, keyed by the relationship each row tests.
H_CTX_WANT, H_CTX_CAN = "H1", "H2"
H_WANT_DO, H_CAN_DO, H_DO_OUTLOOK = "H4", "H5", "H8"


def fmt_p(p):
    """Three decimals, or '<0.001'."""
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def mw_status(role, p_raw, p_adj):
    """Status glyph. Primary rows are judged on the Holm-adjusted p, others on raw p."""
    if role == "Primary":
        return "**✓**" if (p_adj is not None and p_adj < 0.05) else "~"
    return "✓" if (p_raw is not None and p_raw < 0.05) else "~"


def sp_status(role, rho, p, pred, sensitivity=False):
    """Status glyph. `pred` is the predicted sign; a result against it is marked ✗."""
    if sensitivity:
        return "⚠"
    if rho is None:
        return "~"
    if pred is not None and ((pred > 0) != (rho > 0)) and rho != 0:
        return "✗"
    if p is not None and p < 0.05:
        return "**✓**" if role == "Primary" else "✓"
    return "~"


def main():
    csv = config.csv_arg(sys.argv, H.DEFAULT_CSV)
    df = load_survey(csv)
    n_total = len(df)
    scores = H.build_dimension_scores(df)
    masks = H.build_context_masks(df)

    L = []
    L.append("# Hypothesis Test Overview — Green Software Metrics Enterprise Survey")
    L.append("")
    L.append("Reference table for the pairwise tests behind H1, H2, H4, H5 and H8 of the "
             "Want-Can-Do-Outlook-Context model. Hypothesis numbers are the paper's "
             "(chapters/hypotheses.tex). H3 (Context → Do), H6 and H7 are assessed jointly "
             "in the PLS-SEM and are not pairwise rows; see `reports/robustness_report.txt`. "
             f"n={n_total}, significance threshold p < 0.05.")
    L.append("")
    L.append("**Auto-generated** from `scripts/gen_overview.py` (same scoring/test "
             "functions as `hypothesis_test.py`). Do not edit by hand — rerun the generator.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Part 1 — Dimensions & Operationalization")
    L.append("")
    L.append("For Context rows the **Scale / Groups** column shows the two compared groups "
             "(A = reference, B = comparison); for all other rows it shows the measurement scale range.")
    L.append("")
    L.append("| Dimension | Variable | Scale / Groups | Definition |")
    L.append("|---|---|---|---|")
    L.append("| **Context** | `size_250` | SME / Enterprise | ≤249 employees = SME; ≥250 = Enterprise (EU SME definition) — primary H1/H2 split |")
    L.append("| | `size_1000` | SME / Enterprise | ≤999 = SME; ≥1000 = Enterprise (alternative threshold; sensitivity only) |")
    L.append("| | `domain_web` | Non-Web / Web | Operates in the Web/Internet domain — primary H1 moderator |")
    L.append("| | `domain_breadth` | Narrow / Broad | ≤1 domain = Narrow; ≥2 = Broad (tech-stack heterogeneity proxy — primary H2 moderator) |")
    L.append("| | `domain_count` | 0–5 | Number of domains per respondent (ordinal robustness complement to `domain_breadth`) |")
    L.append("| | `domain_Business_Enterprise` | Non / Yes | Operates in Business/Enterprise software domain |")
    L.append("| | `domain_Platform_Infra` | Non / Yes | Operates in Platform/Infrastructure domain |")
    L.append("| | `domain_Mobile` | Non / Yes | Operates in Mobile domain |")
    L.append("| | `domain_IoT_Embedded` | Non / Yes | Operates in IoT/Embedded domain |")
    L.append("| | `maturity_strict` | Low / High | Low = none or ad hoc; High = KPIs or SDLC-integrated; \"I do not know\" excluded |")
    L.append("| | `maturity_ord` | 0–3 | Ordinal maturity (None→Ad hoc→KPIs→SDLC); complement using full gradient |")
    L.append("| **Want** | `want_env` | 1–5 | mean(CO₂ reduction, Energy reduction) |")
    L.append("| | `want_ops` | 1–5 | mean(Cost reduction, Performance, Compliance) |")
    L.append("| | `want_all` | 1–5 | mean of all five objectives |")
    L.append("| **Can** | `can_constraint` | 0–2 | 0 structurally blocked / 1 addressable-only / 2 clear; primary for H2 |")
    L.append("| | `can_effort` | 1–5 | \"Reducing energy is worth the required effort\" Likert; primary for H4 |")
    L.append("| | `can_score` | 0–10 | 10 − number of barriers reported; legacy, H2 sensitivity only |")
    L.append("| **Do** | `do_breadth` | 0–3 | Tier ladder — 1: proxy/resource, 2: +physical energy, 3: +environmental (CO₂); strict |")
    L.append("| | `do_institutionalized` | 0–3 | Decision frequency (0/1/2) + green requirements in place (0/1) |")
    L.append("| | `do_composite` | 0–6 | `do_breadth` + `do_institutionalized` |")
    L.append("| **Outlook** | `outlook_intent` | 1–5 | Likelihood of adopting structured measurement guidelines |")
    L.append("| | `outlook_awareness` | 0–9 | Count of decision areas recognised as impacted |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Legend")
    L.append("")
    L.append("**Tests** — Mann-Whitney U: two-group score comparison (H1, H2). "
             "Spearman ρ: monotonic association (H1–H5). "
             "**Sign (MW rows):** ΔMdn = Mdn_B − Mdn_A (A = reference, B = comparison).")
    L.append("")
    L.append("**Role** — Primary: pre-specified, Holm-Bonferroni corrected (H1/H2) or confirmatory pair (H3–H5). "
             "Complementary: secondary operationalization. Sensitivity: alternative threshold/operationalization. "
             "Exploratory: uncorrected breadth row.")
    L.append("")
    L.append("**Status** — **✓** primary surviving Holm · ✓ significant uncorrected (p<0.05) · "
             "~ correct direction, n.s. · ✗ wrong direction · ⚠ caveat (not interpretable as intended).")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Part 2 — Statistical Tests")
    L.append("")
    L.append("Primary pairs for H1 and H2 are Holm-Bonferroni corrected (k=3 per hypothesis); "
             "secondary, sensitivity, and complementary rows are uncorrected. Primary tests are "
             "confirmatory. Effective n varies: Want pairs exclude item non-response; "
             "`maturity` rows exclude \"I do not know\".")
    L.append("")

    wants = [("want_env", "want_env"), ("want_ops", "want_ops"), ("want_all", "want_all")]

    # ---- H1 / H2 Mann-Whitney ------------------------------------------------
    L.append("### Mann-Whitney U — Context group comparisons (H1, H2)")
    L.append("")
    L.append("| H | Context split | Outcome | Role | n_A | n_B | ΔMdn | p_raw | p_adj | Status |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|:---:|")

    def mw_row(h, split, outcome, role, p_adj=None):
        r = H.mw_test(scores[outcome], masks[split], "")
        if r is None:
            L.append(f"| {h} | `{split}` | {outcome} | {role} | – | – | – | – | – | n/a |")
            return
        status = mw_status(role, r["p"], p_adj)
        padj_s = fmt_p(p_adj) if p_adj is not None else "—"
        cells = [h, f"`{split}`", outcome, role, str(r["n1"]), str(r["n2"]),
                 f"{r['delta_mdn']:+.2f}", fmt_p(r["p"]), padj_s, status]
        if status == "**✓**":
            cells = [f"**{c}**" for c in cells[:-1]] + [status]
        L.append("| " + " | ".join(cells) + " |")

    # H1 primaries with Holm
    h1_primary = [("size_250", "want_ops"), ("domain_web", "want_env"), ("maturity_strict", "want_env")]
    h1_praw = [H.mw_test(scores[o], masks[s], "")["p"] for s, o in h1_primary]
    h1_padj = H.holm_bonferroni(h1_praw)
    for (s, o), pa in zip(h1_primary, h1_padj):
        mw_row(H_CTX_WANT, s, o, "Primary", p_adj=pa)
    # H1 exploratory core (remaining 6 of 3x3)
    for s in ["size_250", "domain_web", "maturity_strict"]:
        for _, o in wants:
            if (s, o) in h1_primary:
                continue
            mw_row(H_CTX_WANT, s, o, "Exploratory")
    # H1 sensitivity size_1000
    for _, o in wants:
        mw_row(H_CTX_WANT, "size_1000", o, "Sensitivity")
    # H1 individual domain breakdowns (excluding Web = primary split)
    for d in ["domain_Business_Enterprise", "domain_Platform_Infra", "domain_Mobile", "domain_IoT_Embedded"]:
        for _, o in wants:
            mw_row(H_CTX_WANT, d, o, "Exploratory")

    # H2 primaries with Holm (can_constraint)
    h2_primary = ["domain_breadth", "size_250", "maturity_strict"]
    h2_praw = [H.mw_test(scores["can_constraint"], masks[s], "")["p"] for s in h2_primary]
    h2_padj = H.holm_bonferroni(h2_praw)
    for s, pa in zip(h2_primary, h2_padj):
        mw_row(H_CTX_CAN, s, "can_constraint", "Primary", p_adj=pa)
    # H2 complementary can_effort
    for s in h2_primary:
        mw_row(H_CTX_CAN, s, "can_effort", "Complementary")
    # H2 sensitivity: legacy can_score (uncorrected)
    for s in h2_primary:
        mw_row(H_CTX_CAN, s, "can_score", "Sensitivity")
    # H2 individual domain breakdowns (can_constraint)
    for d in ["domain_Business_Enterprise", "domain_Platform_Infra", "domain_Web",
              "domain_Mobile", "domain_IoT_Embedded"]:
        mw_row(H_CTX_CAN, d, "can_constraint", "Exploratory")

    L.append("")

    # ---- Spearman ------------------------------------------------------------
    L.append("### Spearman ρ — Monotonic associations (H1, H2, H4, H5, H8)")
    L.append("")
    L.append("| H | Predictor | Outcome | Role | n | ρ | p | Status |")
    L.append("|---|---|---|---|---:|---:|---:|:---:|")

    def sp_row(h, x, y, role, pred, sensitivity=False):
        r = H.spearman_test(scores[x], scores[y], x, y)
        status = sp_status(role, r["rho"], r["p"], pred, sensitivity)
        rho_s = "n/a" if r["rho"] is None else f"{r['rho']:+.3f}"
        cells = [h, f"`{x}`", f"`{y}`", role, str(r["n"]), rho_s, fmt_p(r["p"]), status]
        if status == "**✓**":
            cells = [f"**{c}**" for c in cells[:-1]] + [status]
        L.append("| " + " | ".join(cells) + " |")

    # H1 maturity ordinal
    sp_row(H_CTX_WANT, "maturity_ordinal", "want_env", "Complementary", +1)
    sp_row(H_CTX_WANT, "maturity_ordinal", "want_ops", "Complementary", +1)
    sp_row(H_CTX_WANT, "maturity_ordinal", "want_all", "Complementary", +1)
    # H2 maturity ordinal + domain_count
    sp_row(H_CTX_CAN, "maturity_ordinal", "can_constraint", "Complementary", +1)
    sp_row(H_CTX_CAN, "maturity_ordinal", "can_effort", "Complementary", +1)
    sp_row(H_CTX_CAN, "domain_count", "can_constraint", "Complementary", -1)
    sp_row(H_CTX_CAN, "domain_count", "can_effort", "Complementary", None)
    # H4 want x do (primary: want_env x do_institutionalized)
    do_keys = [("do_breadth", None), ("do_institutionalized", None), ("do_composite", None)]
    for wk, _ in wants:
        for dk, _ in do_keys:
            role = "Primary" if (wk == "want_env" and dk == "do_institutionalized") else "Exploratory"
            sp_row(H_WANT_DO, wk, dk, role, +1)
    # H5 can_effort x do (primary) + can_score sensitivity
    sp_row(H_CAN_DO, "can_effort", "do_composite", "Primary", +1)
    sp_row(H_CAN_DO, "can_effort", "do_breadth", "Primary", +1)
    sp_row(H_CAN_DO, "can_effort", "do_institutionalized", "Primary", +1)
    sp_row(H_CAN_DO, "can_score", "do_composite", "Sensitivity", +1, sensitivity=True)
    # H8 do_composite x outlook
    sp_row(H_DO_OUTLOOK, "do_composite", "outlook_intent", "Primary", +1)
    sp_row(H_DO_OUTLOOK, "do_composite", "outlook_awareness", "Primary", +1)

    L.append("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}  (n={n_total})")


if __name__ == "__main__":
    main()
