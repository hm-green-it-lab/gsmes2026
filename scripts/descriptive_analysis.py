"""
=============================================================================
Green Software Metrics Survey – Descriptive & Reliability Analysis
=============================================================================
Usage:   python descriptive_analysis.py [path/to/export.csv]
Output:  ../reports/descriptive_report.txt  |  ../reports/descriptive_plots.png

What this script does
---------------------
  • Per-dimension descriptives (median, mean, SD, IQR, min, max, missing)
  • Cronbach's alpha with bootstrapped 95% CI and alpha-if-deleted table
  • Spearman inter-item correlations with significance and factor structure note
  • Corrected item-total correlations
  • Ordinal / binary / multiselect frequency tables with Wilson CIs
  • Item variance quality (CV%, skewness, kurtosis)
  • Wilcoxon signed-rank tests vs. scale midpoint
  • Role-group comparisons (Mann-Whitney U, Kruskal-Wallis)

What this script does NOT do
-----------------------------
  • Cross-dimensional hypothesis tests (H1–H5) → see hypothesis_test.py

Requirements:  pip install pandas numpy scipy matplotlib
References:    Bujang et al. (2024), Aithal & Aithal (2020)
=============================================================================
"""

import sys
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import csv_arg, data_path, display_path, report_path
from config import (
    SURVEY_DIMENSIONS, EFFORT_MAP, OUTLOOK_MAP, ROLE_COLS, T,
    DOMAIN_COLS,
    DO_TIER_PROXY, DO_TIER_PHYSICAL, DO_TIER_ENVIRONMENTAL,
    DO_DECISION_MAP, DO_DECISION_COL, DO_REQUIREMENTS_COL,
    RUNTIME_COLS, INFRASTRUCTURE_COLS, ARCH_STYLE_COL, AUTHORITY_COL,
    GREEN_METRICS_ROLE_COLS, WORKLOAD_COLS,
    CAN_OTHER_BARRIER_COL, CAN_STRUCTURAL_BARRIERS, CAN_ADDRESSABLE_BARRIERS,
    load_survey,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

DEFAULT_CSV = data_path("results-survey668719.csv")


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def sep(char="─", w=65):
    return char * w


def sig_stars(p):
    if p < T["p_sig3"]: return "***"
    if p < T["p_sig2"]: return "** "
    if p < T["p_sig"]:  return "*  "
    return "n.s."


def alpha_verdict(a):
    if a is None:                  return "n/a"
    if a >= T["alpha_good"]:       return f"GOOD (α≥{T['alpha_good']}) ✓"
    if a >= T["alpha_pilot"]:      return f"acceptable for pilot (α≥{T['alpha_pilot']}) ✓"
    if a >= T["alpha_weak"]:       return f"weak (α≥{T['alpha_weak']}) – report with caveat"
    return f"poor (α<{T['alpha_weak']}) – check construct structure"


def r_it_verdict(r):
    if abs(r) >= T["r_it_good"]:        return f"good (r≥{T['r_it_good']})"
    if abs(r) >= T["r_it_problematic"]: return f"moderate (r≥{T['r_it_problematic']})"
    return f"problematic (r<{T['r_it_problematic']}) ⚠ – consider revising item"


def cv_verdict(cv):
    if cv >= T["cv_good"]:     return f"good spread (CV≥{T['cv_good']:.0f}%) ✓"
    if cv >= T["cv_moderate"]: return f"moderate spread (CV≥{T['cv_moderate']:.0f}%)"
    return f"low spread (CV<{T['cv_moderate']:.0f}%) ⚠ – item barely discriminates"


def d_verdict(d):
    a = abs(d)
    if a >= T["d_large"]:  return f"large (d≥{T['d_large']})"
    if a >= T["d_medium"]: return f"medium (d≥{T['d_medium']})"
    if a >= T["d_small"]:  return f"small (d≥{T['d_small']})"
    return f"negligible (d<{T['d_small']})"


def bujang_min_n(k):
    table = {2:68,3:50,4:44,5:41,6:39,7:38,8:37,9:36,10:36,15:34,20:34,25:33,30:33}
    return table.get(k, table[min(table, key=lambda x: abs(x - k))])


def cronbach_alpha(data):
    d = data.dropna()
    n, k = d.shape
    if k < 2 or n < 2: return None, n, k
    item_vars = d.var(axis=0, ddof=1)
    total_var = d.sum(axis=1).var(ddof=1)
    if total_var == 0: return None, n, k
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var), n, k


def cronbach_alpha_ci(data, n_boot=None):
    if n_boot is None: n_boot = T["alpha_ci_boot"]
    d = data.dropna().values
    n, k = d.shape
    if k < 2 or n < 4: return None, None, None
    obs_alpha, _, _ = cronbach_alpha(data)
    boot_alphas = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        sample = d[idx]
        sv = sample.var(axis=0, ddof=1)
        tv = sample.sum(axis=1).var(ddof=1)
        if tv > 0:
            boot_alphas.append((k / (k - 1)) * (1 - sv.sum() / tv))
    boot_alphas = np.array(boot_alphas)
    return obs_alpha, np.percentile(boot_alphas, 2.5), np.percentile(boot_alphas, 97.5)


def spearman_brown(r):
    return (2 * r) / (1 + r) if r > -1 else None


def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p      = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0, centre - margin) * 100, min(1, centre + margin) * 100


def mannwhitney_r(u, n1, n2):
    n      = n1 + n2
    mu_u   = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n + 1) / 12)
    z      = (u - mu_u) / sigma_u if sigma_u > 0 else 0
    return abs(z) / np.sqrt(n)


def out(lines, report_lines):
    print("\n".join(lines))
    report_lines += lines + [""]


def get_likert_series(df):
    """Canonical dict of all Likert numeric series, keyed by item name."""
    series = {}
    for name, col in SURVEY_DIMENSIONS["want_objectives"]["items"].items():
        series[name] = pd.to_numeric(df.iloc[:, col], errors="coerce")
    effort_col   = list(SURVEY_DIMENSIONS["want_effort"]["items"].values())[0]
    outlook_col  = list(SURVEY_DIMENSIONS["outlook_likely"]["items"].values())[0]
    series["Effort"]         = df.iloc[:, effort_col].map(EFFORT_MAP)
    series["Outlook_likely"] = df.iloc[:, outlook_col].map(OUTLOOK_MAP)
    return series


# =============================================================================
# DIMENSION HANDLERS
# =============================================================================

def analyse_likert(dim_key, config, df, report_lines):
    items     = config["items"]
    data      = pd.DataFrame({n: pd.to_numeric(df.iloc[:, i], errors="coerce")
                               for n, i in items.items()})
    n_total   = len(data)
    n_complete = data.dropna().shape[0]
    d         = data.dropna()

    L = [sep("═"), f"  {config['label']}",
         f"  n={n_total}  |  complete (listwise)={n_complete}", sep()]

    # Descriptives
    L += ["",
          f"  {'Item':<22} {'Mean':>6} {'SD':>5} {'Mdn':>5} {'IQR':>5} "
          f"{'min':>4} {'max':>4}  {'missing':>7}",
          "  " + sep("─", 62)]
    for col in data.columns:
        s            = data[col].dropna()
        q1, mdn, q3  = s.quantile([0.25, 0.5, 0.75])
        L.append(f"  {col:<22} {s.mean():>6.2f} {s.std(ddof=1):>5.2f} "
                 f"{mdn:>5.2f} {q3-q1:>5.2f} "
                 f"{s.min():>4.0f} {s.max():>4.0f}  {n_total - len(s):>7}")
    L.append("  Note: Median and IQR are the primary descriptors for ordinal data.")

    # Cronbach's alpha
    alpha, ci_lo, ci_hi = cronbach_alpha_ci(data)
    _, n_a, k_a = cronbach_alpha(data)
    min_n   = bujang_min_n(k_a)
    min_adj = int(np.ceil(min_n / 0.8))
    n_ok    = n_a >= min_adj
    ci_str  = (f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]"
               if ci_lo is not None else "CI not computable")
    L += ["", "  CRONBACH'S ALPHA  (Spearman-Brown corrected for 2-item sub-factors below)",
          "  " + sep("─", 52),
          f"  α = {alpha:.3f}   {ci_str}   {alpha_verdict(alpha)}",
          f"  n={n_a}, k={k_a}   Bujang (2024) minimum: n≥{min_adj} "
          f"(n≥{min_n} +20% non-response)  →  "
          f"{'met ✓' if n_ok else 'NOT met – treat result as preliminary ⚠'}"]

    # Alpha-if-deleted
    L += ["", f"  {'Item':<22} {'α if deleted':>13} {'Δα':>7}  action"]
    removals = []
    for col in data.columns:
        a_del, _, _ = cronbach_alpha(d.drop(columns=[col]))
        if a_del is not None and alpha is not None:
            delta  = a_del - alpha
            action = (f"⚠ removing improves α by {delta:+.3f} – consider revising"
                      if delta > T["alpha_if_del_flag"] else "keep")
            if delta > T["alpha_if_del_flag"]: removals.append(col)
            L.append(f"  {col:<22} {a_del:>13.3f} {delta:>+7.3f}  {action}")

    # Inter-item correlations
    corr = d.corr(method="spearman")
    cols = list(corr.columns)
    L += ["", "  INTERITEM CORRELATIONS  (Spearman ρ – appropriate for ordinal Likert data)",
          f"  Threshold: ρ≥{T['r_strong']} strong | ρ≥{T['r_moderate']} moderate | "
          f"ρ≥{T['r_weak']} weak | ρ<{T['r_weak']} independent",
          "  " + sep("─", 52),
          f"  {'':20}" + "".join(f"{c[:7]:>9}" for c in cols)]
    for row in cols:
        rs = f"  {row:<20}"
        for col in cols:
            rv = corr.loc[row, col]
            if row == col:
                rs += f"{'—':>9}"
            else:
                t_stat = rv * np.sqrt((len(d) - 2) / max(1 - rv**2, 1e-10))
                p_val  = 2 * stats.t.sf(abs(t_stat), df=len(d) - 2)
                rs += f"{rv:>5.2f}{sig_stars(p_val).strip():>3}".rjust(9)
        L.append(rs)
    L.append("  *** p<.001  ** p<.01  * p<.05")

    # Strong / near-zero pair detection
    strong_pairs, near_zero_pairs = [], []
    for i, r in enumerate(cols):
        for j, c in enumerate(cols):
            if j <= i: continue
            rv     = corr.loc[r, c]
            t_stat = rv * np.sqrt((len(d) - 2) / max(1 - rv**2, 1e-10))
            p_val  = 2 * stats.t.sf(abs(t_stat), df=len(d) - 2)
            if abs(rv) >= T["r_strong"] and p_val < T["p_sig"]:
                strong_pairs.append((r, c, rv))
            if abs(rv) < T["r_weak"]:
                near_zero_pairs.append((r, c, rv))

    # Corrected item-total correlations
    L += ["", "  CORRECTED ITEM-TOTAL CORRELATIONS  (Spearman; item excluded from rest-score)",
          f"  Threshold: r≥{T['r_it_good']} good | r≥{T['r_it_problematic']} moderate | "
          f"r<{T['r_it_problematic']} problematic",
          "  " + sep("─", 52),
          f"  {'Item':<22} {'ρ_it':>6}   verdict"]
    problematic_items = []
    for col in d.columns:
        rest    = d.drop(columns=[col]).sum(axis=1)
        rho, _  = stats.spearmanr(d[col], rest)
        verdict = r_it_verdict(rho)
        L.append(f"  {col:<22} {rho:>6.3f}   {verdict}")
        if abs(rho) < T["r_it_problematic"]:
            problematic_items.append(col)

    # Factor structure note
    if strong_pairs or near_zero_pairs:
        L += ["", "  FACTOR STRUCTURE NOTE", "  " + sep("─", 52)]
        if strong_pairs:
            L.append("  Strongly correlated pairs (likely same sub-factor):")
            for r, c, rv in strong_pairs:
                sb = spearman_brown(rv)
                L.append(f"    {r} ↔ {c}:  ρ={rv:.3f}  "
                         f"Spearman-Brown α={sb:.2f}  {alpha_verdict(sb)}")
        if near_zero_pairs:
            L.append(f"  Near-zero cross-correlations detected (ρ<{T['r_weak']}).")
            L.append(f"  → α across all {len(cols)} items may be misleading.")

    # Per-cluster alpha
    clusters = config.get("clusters", {})
    if clusters:
        L += ["", "  PER-CLUSTER RELIABILITY", "  " + sep("─", 52)]
        for cluster_name, cluster_items in clusters.items():
            available = [c for c in cluster_items if c in d.columns]
            if len(available) < 2:
                L.append(f"  {cluster_name} cluster: insufficient items ({available})")
                continue
            c_data = d[available]
            if len(available) == 2:
                rv = c_data.corr(method="spearman").iloc[0, 1]
                sb = spearman_brown(rv)
                L.append(f"  {cluster_name} ({', '.join(available)})")
                L.append(f"    Spearman-Brown α = {sb:.3f}   {alpha_verdict(sb)}")
            else:
                c_alpha, c_ci_lo, c_ci_hi = cronbach_alpha_ci(c_data)
                c_ci_str = (f"95% CI [{c_ci_lo:.3f}, {c_ci_hi:.3f}]"
                            if c_ci_lo is not None else "CI not computable")
                L.append(f"  {cluster_name} ({', '.join(available)})")
                L.append(f"    α = {c_alpha:.3f}   {c_ci_str}   {alpha_verdict(c_alpha)}")

    # Recommendations
    L += ["", "  RECOMMENDATIONS", "  " + sep("─", 52)]
    if alpha is not None:
        if alpha >= T["alpha_good"]:
            L.append(f"  ✓ α={alpha:.3f} {ci_str} meets the good threshold.")
        elif alpha >= T["alpha_pilot"]:
            L.append(f"  ✓ α={alpha:.3f} {ci_str} acceptable for pilot. "
                     f"Full validation requires larger n.")
        elif alpha >= T["alpha_weak"]:
            L.append(f"  ⚠ α={alpha:.3f} {ci_str} is weak. Report with explicit caveat.")
        else:
            L.append(f"  ✗ α={alpha:.3f} {ci_str} is poor. Do not report as reliability evidence.")
            if strong_pairs:
                L.append("    Consider reporting sub-factor α values instead (see above).")
    if not n_ok:
        L.append(f"  ⚠ n={n_a} below Bujang minimum (n≥{min_adj}). "
                 f"Defer confirmatory testing to main study.")
    if removals:
        L.append(f"  ⚠ Removing improves α: {', '.join(removals)}. "
                 f"Review item wording or consider splitting dimension.")
    if problematic_items:
        L.append(f"  ⚠ Negligible item-total correlation: {', '.join(problematic_items)}. "
                 f"Item does not align with the block.")
    out(L, report_lines)


def analyse_likert_text(dim_key, config, df, report_lines):
    mapping = config["mapping"]
    col_idx = list(config["items"].values())[0]
    s       = df.iloc[:, col_idx].map(mapping)
    n       = s.notna().sum()
    q1, mdn, q3 = s.quantile([0.25, 0.5, 0.75])

    L = [sep(), f"  {config['label']}", sep("─"),
         f"  n={n}   Mean={s.mean():.2f}   SD={s.std(ddof=1):.2f}   "
         f"Mdn={mdn:.1f}   IQR={q3-q1:.1f}   min={int(s.min())}   max={int(s.max())}", ""]
    for v in sorted(mapping.values()):
        lbl = next(k for k, vv in mapping.items() if vv == v)
        c   = (s == v).sum()
        bar = "█" * int(c / n * 40)
        L.append(f"  {v}  {lbl:<35} {c:2d} ({c/n*100:3.0f}%)  {bar}")
    L += ["", "  Single item – no alpha possible. Report Mean, SD, Median, IQR."]
    out(L, report_lines)


def analyse_binary(dim_key, config, df, report_lines):
    pos = config.get("positive_value", "Yes")
    L   = [sep(), f"  {config['label']}", sep("─")]
    for name, col_idx in config["items"].items():
        s     = df.iloc[:, col_idx].dropna()
        n_pos = s.str.startswith(pos, na=False).sum()
        n     = len(s)
        pct   = n_pos / n * 100
        lo, hi = wilson_ci(n_pos, n)
        bar   = "█" * int(pct / 5)
        L.append(f"  {n_pos}/{n} = {pct:.0f}%  95% CI [{lo:.0f}%, {hi:.0f}%]  {bar}")
    out(L, report_lines)


def analyse_multiselect(dim_key, config, df, report_lines):
    items   = config["items"]
    n_total = len(df)
    prev    = {}
    for name, col_idx in items.items():
        k = (df.iloc[:, col_idx] == "Yes").sum()
        lo, hi = wilson_ci(k, n_total)
        prev[name] = (k / n_total * 100, lo, hi)
    sorted_items = sorted(prev.items(), key=lambda x: x[1][0], reverse=True)
    max_pct = max(v[0] for v in prev.values())

    L = [sep(), f"  {config['label']}", sep("─"),
         f"  Sorted by prevalence  |  n={n_total}  |  95% Wilson CI shown", ""]
    for name, (pct, lo, hi) in sorted_items:
        bar  = "█" * int(pct / 5)
        flag = " ◀ highest" if pct == max_pct else ""
        L.append(f"  {name:<26} {pct:>4.0f}%  [{lo:>3.0f}%, {hi:>3.0f}%]  {bar}{flag}")
    out(L, report_lines)


def analyse_ordinal(dim_key, config, df, report_lines):
    order   = config.get("order", [])
    col_idx = list(config["items"].values())[0]
    s       = df.iloc[:, col_idx]
    # If "None" is an explicit answer option, NaN in the CSV represents that
    # choice (LimeSurvey leaves the cell empty when the respondent picks "None").
    # Fill before counting so those responses are not silently excluded.
    if "None" in order:
        s = s.fillna("None")
    else:
        s = s.dropna()
    n       = len(s)

    L = [sep(), f"  {config['label']}  (n={n})", sep("─")]
    for cat in order + [c for c in s.unique() if c not in order]:
        c   = (s == cat).sum()
        pct = c / n * 100
        bar = "█" * int(pct / 5)
        L.append(f"  {cat[:42]:<44} {c:2d} ({pct:3.0f}%)  {bar}")
    out(L, report_lines)


# =============================================================================
# DOMAIN PROFILE  (DOMAIN_COLS – outside SURVEY_DIMENSIONS, used in H1/H2)
# =============================================================================

def analyse_domain_profile(df, report_lines):
    n_total = len(df)

    # Multiselect frequency table
    prev = {}
    for name, col_idx in DOMAIN_COLS.items():
        k = int((df.iloc[:, col_idx] == "Yes").sum())
        lo, hi = wilson_ci(k, n_total)
        prev[name] = (k, k / n_total * 100, lo, hi)
    sorted_items = sorted(prev.items(), key=lambda x: x[1][1], reverse=True)
    max_pct = max(v[1] for v in prev.values())

    L = [sep(), "  CONTEXT – Application domain (multi-select)",
         sep("─"),
         f"  Sorted by prevalence  |  n={n_total}  |  95% Wilson CI shown", ""]
    for name, (k, pct, lo, hi) in sorted_items:
        bar  = "█" * int(pct / 5)
        flag = " ◀ highest" if pct == max_pct else ""
        L.append(f"  {name:<26} {pct:>4.0f}%  [{lo:>3.0f}%, {hi:>3.0f}%]  {bar}{flag}")

    # Domain breadth distribution
    dom_count = pd.concat(
        [(df.iloc[:, c] == "Yes").astype(int) for c in DOMAIN_COLS.values()], axis=1
    ).sum(axis=1)

    L += ["",
          "  DOMAIN BREADTH  (number of domains checked per respondent)",
          "  " + sep("─", 52)]
    for k_val in range(len(DOMAIN_COLS) + 1):
        c = int((dom_count == k_val).sum())
        if c == 0:
            continue
        pct = c / n_total * 100
        bar = "█" * int(pct / 5)
        L.append(f"  {k_val} domain(s)  {c:2d} ({pct:3.0f}%)  {bar}")

    narrow_n = int((dom_count <= 1).sum())
    broad_n  = int((dom_count >= 2).sum())
    L += ["",
          "  Hypothesis split used in H2  (domain_breadth):",
          f"    Narrow (≤1 domain):  n={narrow_n:2d}  ({narrow_n / n_total * 100:.0f}%)",
          f"    Broad  (≥2 domains): n={broad_n:2d}  ({broad_n  / n_total * 100:.0f}%)"]

    web_col  = DOMAIN_COLS["Web"]
    web_n    = int((df.iloc[:, web_col] == "Yes").sum())
    nonweb_n = n_total - web_n
    L += ["",
          "  Primary domain split used in H1  (domain_web):",
          f"    Web:      n={web_n:2d}  ({web_n    / n_total * 100:.0f}%)",
          f"    Non-Web:  n={nonweb_n:2d}  ({nonweb_n / n_total * 100:.0f}%)"]

    out(L, report_lines)


# =============================================================================
# SYSTEM CONTEXT  (RUNTIME_COLS, INFRASTRUCTURE_COLS, ARCH_STYLE_COL, etc.)
# =============================================================================

def _multiselect_table(df, cols_dict, label, report_lines):
    n_total = len(df)
    prev = {}
    for name, col_idx in cols_dict.items():
        k = int((df.iloc[:, col_idx] == "Yes").sum())
        lo, hi = wilson_ci(k, n_total)
        prev[name] = (k, k / n_total * 100, lo, hi)
    sorted_items = sorted(prev.items(), key=lambda x: x[1][1], reverse=True)
    max_pct = max(v[1] for v in prev.values())

    L = [sep(), f"  {label}",
         sep("─"),
         f"  Sorted by prevalence  |  n={n_total}  |  95% Wilson CI shown", ""]
    for name, (k, pct, lo, hi) in sorted_items:
        bar  = "█" * int(pct / 5)
        flag = " ◀ highest" if pct == max_pct else ""
        L.append(f"  {name:<30} {pct:>4.0f}%  [{lo:>3.0f}%, {hi:>3.0f}%]  {bar}{flag}")
    out(L, report_lines)


def analyse_system_context(df, report_lines):
    # --- Workload types (multi-select) ---
    _multiselect_table(df, WORKLOAD_COLS,
                       "CONTEXT – Workload types (multi-select)", report_lines)

    # --- Runtime environment (multi-select) ---
    _multiselect_table(df, RUNTIME_COLS,
                       "CONTEXT – Runtime environment (multi-select)", report_lines)

    # --- Deployment infrastructure (multi-select) ---
    _multiselect_table(df, INFRASTRUCTURE_COLS,
                       "CONTEXT – Deployment infrastructure (multi-select)", report_lines)

    # --- Architectural style (ordinal, single-answer) ---
    s = df.iloc[:, ARCH_STYLE_COL].dropna()
    n = len(s)
    L = [sep(), f"  CONTEXT – Architectural style  (n={n})", sep("─")]
    for cat in sorted(s.unique()):
        c   = int((s == cat).sum())
        pct = c / n * 100
        lo, hi = wilson_ci(c, n)
        bar = "█" * int(pct / 5)
        L.append(f"  {cat:<40} {c:2d} ({pct:3.0f}%)  [{lo:>2.0f}%, {hi:>2.0f}%]  {bar}")
    out(L, report_lines)

    # --- Decision authority (ordinal, single-answer) ---
    s = df.iloc[:, AUTHORITY_COL].dropna()
    n = len(s)
    L = [sep(), f"  CONTEXT – Sustainability decision authority  (n={n})", sep("─")]
    for cat in sorted(s.unique()):
        c   = int((s == cat).sum())
        pct = c / n * 100
        lo, hi = wilson_ci(c, n)
        bar = "█" * int(pct / 5)
        L.append(f"  {cat[:50]:<52} {c:2d} ({pct:3.0f}%)  [{lo:>2.0f}%, {hi:>2.0f}%]  {bar}")
    out(L, report_lines)

    # --- Roles involved in green metrics activities (multi-select) ---
    _multiselect_table(df, GREEN_METRICS_ROLE_COLS,
                       "CONTEXT – Roles involved in green metrics activities (multi-select)",
                       report_lines)


# =============================================================================
# COMPUTED DIMENSION SCORES  (derived for H1–H5, not in SURVEY_DIMENSIONS)
# =============================================================================

def _compute_hypothesis_scores(df):
    """Replicates build_dimension_scores from hypothesis_test.py."""
    scores = pd.DataFrame(index=df.index)

    # WANT sub-scales
    want_items = SURVEY_DIMENSIONS["want_objectives"]["items"]
    clusters   = SURVEY_DIMENSIONS["want_objectives"]["clusters"]

    def to_num(name):
        return pd.to_numeric(df.iloc[:, want_items[name]], errors="coerce")

    scores["want_env"] = pd.concat([to_num(n) for n in clusters["Environmental"]], axis=1).mean(axis=1)
    scores["want_ops"] = pd.concat([to_num(n) for n in clusters["Operational"]],   axis=1).mean(axis=1)
    scores["want_all"] = pd.concat([to_num(n) for n in want_items],                axis=1).mean(axis=1)

    # CAN – capability constraint. Same definition as hypothesis_test.py:
    # can_constraint is the 3-level severity ordinal (primary), can_score the
    # legacy 10 − barrier count kept only as a sensitivity comparison.
    barrier_cols = SURVEY_DIMENSIONS["can_barriers"]["items"]
    no_barrier   = df.iloc[:, barrier_cols["No_barriers"]] == "Yes"
    non_sentinel = {k: v for k, v in barrier_cols.items() if k != "No_barriers"}

    def _any_barrier(names):
        m = pd.Series(False, index=df.index)
        for nm in names:
            m = m | (df.iloc[:, barrier_cols[nm]] == "Yes")
        return m

    structural_present  = _any_barrier(CAN_STRUCTURAL_BARRIERS)
    addressable_present = _any_barrier(CAN_ADDRESSABLE_BARRIERS)
    other_txt     = df.iloc[:, CAN_OTHER_BARRIER_COL].astype(str).str.strip()
    other_present = (other_txt != "") & (other_txt.str.lower() != "nan")
    addressable_present = addressable_present | other_present

    scores["can_constraint"] = pd.Series(
        np.select(
            [structural_present,
             (~structural_present) & addressable_present,
             (~structural_present) & (~addressable_present) & no_barrier],
            [0.0, 1.0, 2.0],
            default=np.nan),
        index=df.index)

    raw_count = pd.Series(0.0, index=df.index)
    for _, c in non_sentinel.items():
        raw_count += (df.iloc[:, c] == "Yes").astype(float)
    raw_count += other_present.astype(float)
    raw_count = raw_count.where(~no_barrier, other=0.0)
    scores["can_score"] = (len(non_sentinel) + 1) - raw_count   # 9 listed + Other

    # DO – tier breadth
    metric_cols = SURVEY_DIMENSIONS["do_metrics"]["items"]
    has_proxy   = pd.Series(False, index=df.index)
    has_phys    = pd.Series(False, index=df.index)
    has_env     = pd.Series(False, index=df.index)
    for name, col in metric_cols.items():
        t = (df.iloc[:, col] == "Yes").astype(int)
        if name in DO_TIER_PROXY:         has_proxy |= t.astype(bool)
        if name in DO_TIER_PHYSICAL:      has_phys  |= t.astype(bool)
        if name in DO_TIER_ENVIRONMENTAL: has_env   |= t.astype(bool)
    breadth = pd.Series(0, index=df.index)
    breadth = breadth.where(~has_proxy,                        other=1)
    breadth = breadth.where(~(has_proxy & has_phys),           other=2)
    breadth = breadth.where(~(has_proxy & has_phys & has_env), other=3)
    scores["do_breadth"] = breadth

    # DO – institutionalization & composite
    freq_score = df.iloc[:, DO_DECISION_COL].map(DO_DECISION_MAP)
    req_score  = (df.iloc[:, DO_REQUIREMENTS_COL] == "Yes").astype(float)
    scores["do_institutionalized"] = freq_score + req_score
    scores["do_composite"]         = scores["do_breadth"] + scores["do_institutionalized"]

    # OUTLOOK – awareness count
    impact_cols = SURVEY_DIMENSIONS["outlook_impact"]["items"]
    scores["outlook_awareness"] = pd.concat(
        [(df.iloc[:, c] == "Yes").astype(int) for c in impact_cols.values()], axis=1
    ).sum(axis=1).astype(float)

    return scores


def analyse_computed_scores(df, report_lines):
    scores = _compute_hypothesis_scores(df)

    score_meta = [
        ("want_env",             "Want (environmental)    [1–5]",
         "mean(CO2_reduction, Energy_reduction)"),
        ("want_ops",             "Want (operational)      [1–5]",
         "mean(Cost_reduction, Performance, Compliance)"),
        ("want_all",             "Want (all objectives)   [1–5]",
         "mean of all five want items"),
        ("can_constraint",       "Can  (constraint 0–2)   [0–2]",
         "0 structural / 1 addressable / 2 clear  (primary for H2)"),
        ("can_score",            "Can  (barrier-inverted) [0–10]",
         "10 − n_barriers_checked  (legacy, H2 sensitivity only)"),
        ("do_breadth",           "Do   (tier breadth)     [0–3]",
         "strict tier ladder: proxy ≥ physical ≥ environmental"),
        ("do_institutionalized", "Do   (institutionalized)[0–3]",
         "decision_freq (0/1/2) + green_req (0/1)"),
        ("do_composite",         "Do   (composite)        [0–6]",
         "do_breadth + do_institutionalized"),
        ("outlook_awareness",    "Outlook (awareness)     [0–9]",
         "count of decision areas recognised as impacted"),
    ]

    L = [sep("═"),
         "  COMPUTED DIMENSION SCORES  (derived variables used in H1–H5)",
         "  Note: can_effort and outlook_intent are reported as single-item",
         "  Likert questions above; the other derived scores are shown here.",
         sep(),
         f"  {'Score':<36} {'n':>4} {'Mdn':>6} {'Mean':>6} {'SD':>6} {'Min':>5} {'Max':>5}",
         "  " + sep("─", 72)]
    for key, label, formula in score_meta:
        s = scores[key].dropna()
        if len(s) == 0:
            continue
        L.append(f"  {label:<36} {len(s):>4} {s.median():>6.2f} {s.mean():>6.2f} "
                 f"{s.std(ddof=1):>6.2f} {s.min():>5.1f} {s.max():>5.1f}")
        L.append(f"    = {formula}")

    # DO breadth tier breakdown
    tier_labels = {
        0: "Tier 0  no metrics tracked",
        1: "Tier 1  proxy/resource metrics only",
        2: "Tier 2  + physical energy (Power_W / Energy_kWh)",
        3: "Tier 3  + environmental (CO₂e / SCI / Water)",
    }
    L += ["",
          f"  DO BREADTH TIER DISTRIBUTION  (n={len(scores)})",
          "  " + sep("─", 62)]
    for k in range(4):
        c   = int((scores["do_breadth"] == k).sum())
        pct = c / len(scores) * 100
        bar = "█" * int(pct / 5)
        L.append(f"  {tier_labels[k]:<50} {c:2d} ({pct:3.0f}%)  {bar}")

    # DO institutionalized breakdown
    L += ["",
          "  DO INSTITUTIONALIZED DISTRIBUTION  (decision_freq 0/1/2 + green_req 0/1)",
          "  " + sep("─", 62)]
    for k in range(4):
        c   = int((scores["do_institutionalized"] == k).sum())
        pct = c / len(scores) * 100
        bar = "█" * int(pct / 5)
        L.append(f"  Score {k}  {c:2d} ({pct:3.0f}%)  {bar}")

    # DO composite breakdown
    L += ["",
          "  DO COMPOSITE DISTRIBUTION  [0–6]",
          "  " + sep("─", 62)]
    for k in range(7):
        c   = int((scores["do_composite"] == k).sum())
        if c == 0:
            continue
        pct = c / len(scores) * 100
        bar = "█" * int(pct / 5)
        L.append(f"  Score {k}  {c:2d} ({pct:3.0f}%)  {bar}")

    # Outlook awareness breakdown
    L += ["",
          "  OUTLOOK AWARENESS DISTRIBUTION  [0–9 decision areas]",
          "  " + sep("─", 62)]
    for k in range(10):
        c   = int((scores["outlook_awareness"] == k).sum())
        if c == 0:
            continue
        pct = c / len(scores) * 100
        bar = "█" * int(pct / 5)
        L.append(f"  {k} areas  {c:2d} ({pct:3.0f}%)  {bar}")

    out(L, report_lines)


# =============================================================================
# EXTENDED ANALYSES
# =============================================================================

def analyse_variance_quality(df, report_lines):
    """CV%, skewness, kurtosis per Likert item."""
    items = get_likert_series(df)
    L = [sep("═"), "  ITEM VARIANCE QUALITY",
         f"  CV%>{T['cv_good']:.0f} good | {T['cv_moderate']:.0f}–{T['cv_good']:.0f} moderate | "
         f"<{T['cv_moderate']:.0f} item barely discriminates",
         f"  |Skew|>{T['skew_flag']} response pile-up (ceiling/floor)   "
         f"|Kurt|>{T['kurt_flag']} extreme shape",
         sep(),
         f"  {'Item':<18} {'Mean':>6} {'Mdn':>5} {'SD':>5} {'IQR':>5} {'CV%':>6} "
         f"{'Skew':>6} {'Kurt':>6}  Verdict",
         "  " + sep("─", 78)]

    flags_found = []
    for name, series in items.items():
        s           = series.dropna()
        m, sd       = s.mean(), s.std(ddof=1)
        q1, mdn, q3 = s.quantile([0.25, 0.5, 0.75])
        cv          = sd / m * 100 if m else float("nan")
        skew, kurt  = s.skew(), s.kurt()
        flags       = [cv_verdict(cv)]
        if abs(skew) > T["skew_flag"]: flags.append(f"skewed ({skew:+.2f}) ⚠")
        if abs(kurt) > T["kurt_flag"]: flags.append(f"kurtosis ({kurt:+.2f}) ⚠")
        if len(flags) > 1: flags_found.append(name)
        L.append(f"  {name:<18} {m:>6.2f} {mdn:>5.2f} {sd:>5.2f} {q3-q1:>5.2f} {cv:>6.1f} "
                 f"{skew:>6.2f} {kurt:>6.2f}  {', '.join(flags)}")

    low_cv = [n for n, s in items.items()
              if s.dropna().std(ddof=1) / s.dropna().mean() * 100 < T["cv_moderate"]]
    L += ["", "  RECOMMENDATIONS", "  " + sep("─", 52)]
    if low_cv:
        L.append(f"  ⚠ Low variance items: {', '.join(low_cv)}")
        L.append("    Limited discriminatory value – consider scale range revision.")
    else:
        L.append(f"  ✓ All items CV≥{T['cv_moderate']:.0f}% – sufficient spread.")
    for name in flags_found:
        s    = items[name].dropna()
        skew = s.skew()
        kurt = s.kurt()
        if abs(kurt) > T["kurt_flag"]:
            L.append(f"  ⚠ {name}: |Kurt|={abs(kurt):.2f} – possibly bimodal. "
                     f"Report as substantive finding if respondents are genuinely split.")
        elif abs(skew) > T["skew_flag"]:
            direction = "ceiling" if skew < 0 else "floor"
            L.append(f"  ⚠ {name}: skew={skew:+.2f} – {direction} effect.")
    out(L, report_lines)


def analyse_midpoint_test(df, report_lines):
    """Wilcoxon signed-rank test vs. scale midpoint. Non-parametric."""
    items    = get_likert_series(df)
    midpoint = 3.0
    L = [sep("═"),
         f"  WILCOXON SIGNED-RANK TEST  vs. scale midpoint (μ₀={midpoint})",
         "  Non-parametric alternative to one-sample t-test.",
         "  Appropriate for ordinal Likert data regardless of distribution.",
         f"  H₀: median={midpoint} (neutral)  |  H₁: median≠{midpoint}",
         "  Cohen's d reported as supplementary effect size for comparability.",
         f"  d: <{T['d_small']} negligible | {T['d_small']}–{T['d_medium']} small | "
         f"{T['d_medium']}–{T['d_large']} medium | >{T['d_large']} large",
         sep(),
         f"  {'Item':<18} {'Mdn':>5} {'Mean':>6} {'W':>8} {'p':>7} {'Sig':>4} {'d':>6}  Result",
         "  " + sep("─", 78)]

    sig_items, nonsig_items = [], []
    all_results = {}  # name → dict for summary block
    for name, series in items.items():
        s = series.dropna()
        if len(s) < 5: continue
        m, sd, mdn = s.mean(), s.std(ddof=1), s.median()
        d          = (m - midpoint) / sd
        diffs      = (s - midpoint).pipe(lambda x: x[x != 0])
        if len(diffs) < 4:
            L.append(f"  {name:<18} {mdn:>5.2f} {m:>6.2f} {'n/a':>8} {'n/a':>7} "
                     f"{'n/a':>4} {d:>6.2f}  too many ties")
            all_results[name] = dict(mdn=mdn, m=m, d=d, p=None, sig=False,
                                     direction="too many ties")
            continue
        w_stat, p = stats.wilcoxon(diffs, alternative="two-sided")
        is_sig    = p < T["p_sig"]
        direction = ("above neutral ↑" if mdn > midpoint else "below neutral ↓")
        all_results[name] = dict(mdn=mdn, m=m, d=d, p=p, sig=is_sig,
                                 direction=direction if is_sig else "n.s.")
        if is_sig:
            sig_items.append((name, mdn, m, d, p, direction))
        else:
            nonsig_items.append((name, mdn, m, d, p))
        L.append(f"  {name:<18} {mdn:>5.2f} {m:>6.2f} {w_stat:>8.1f} {p:>7.3f} "
                 f"{sig_stars(p):>4} {d:>6.2f}  "
                 f"{direction if is_sig else 'no clear trend'}")

    L += ["", "  RECOMMENDATIONS", "  " + sep("─", 52)]
    if sig_items:
        L.append("  Significant results (p<0.05) – report in paper:")
        for name, mdn, m, d, p, direction in sorted(sig_items, key=lambda x: x[4]):
            L.append(f"    • {name:<20} Mdn={mdn:.1f}  Mean={m:.2f}  d={d:.2f} "
                     f"({d_verdict(d)})  p={p:.3f}{sig_stars(p).strip()}  {direction}")
    else:
        L.append("  No items reached significance. Report descriptive medians only.")
    if nonsig_items:
        L.append("  Non-significant – report descriptively, do not over-claim:")
        for name, mdn, m, d, p in sorted(nonsig_items, key=lambda x: x[4]):
            L.append(f"    • {name:<20} Mdn={mdn:.1f}  Mean={m:.2f}  d={d:.2f}  "
                     f"p={p:.3f}  no clear trend")

    # ── Slide-ready importance rating summary (objectives only) ──────────────
    importance_order = list(SURVEY_DIMENSIONS["want_objectives"]["items"].keys())
    importance_rows  = {k: v for k, v in all_results.items() if k in importance_order}
    if importance_rows:
        L += ["", "  IMPORTANCE RATING SUMMARY  (objectives only – slide reference)",
              "  " + sep("─", 62),
              f"  {'Item':<22} {'Mdn':>5}   {'p':>7}   {'d':>5}  {'size':>6}  result"]
        for name in importance_order:
            if name not in importance_rows:
                continue
            r = importance_rows[name]
            size  = d_verdict(r["d"]) if r["p"] is not None else "—"
            if r["p"] is None:
                p_str = "n/a"
            elif r["p"] < 0.001:
                p_str = "p < .001"
            else:
                p_str = f"p = {r['p']:.3f}"
            result = r["direction"]
            L.append(f"  {name:<22} {r['mdn']:>5.1f}   {p_str:<8}   {r['d']:>5.2f}  "
                     f"{size:<8}  {result}")

    out(L, report_lines)


def analyse_group_comparisons(df, report_lines):
    """Role-group comparisons: Mann-Whitney U + Kruskal-Wallis."""
    items  = get_likert_series(df)
    masks  = {role: df.iloc[:, col] == "Yes" for role, col in ROLE_COLS.items()}
    ns     = {role: m.sum() for role, m in masks.items()}
    overlap = (masks["Management"] & masks["Technical"]).sum()

    L = [sep("═"),
         "  GROUP COMPARISONS  " +
         "  |  ".join(f"{r} n={ns[r]}" for r in ROLE_COLS),
         f"  ⚠ Roles are multi-select (Management∩Technical overlap: n={overlap}). "
         "Treat all results as exploratory.",
         sep()]

    # Shapiro-Wilk normality check
    L += ["  [1] SHAPIRO-WILK  normality prerequisite check",
          f"  p>{T['p_sig']} = normally distributed  |  p≤{T['p_sig']} → use Mann-Whitney",
          "  " + sep("─", 52)]
    non_normal = []
    for name, series in items.items():
        g1 = series[masks["Management"]].dropna()
        g2 = series[masks["Technical"]].dropna()
        if len(g1) < 3 or len(g2) < 3: continue
        _, p1 = stats.shapiro(g1)
        _, p2 = stats.shapiro(g2)
        if p1 <= T["p_sig"] or p2 <= T["p_sig"]:
            non_normal.append(name)
    if non_normal:
        L.append(f"  Non-normal in ≥1 group: {', '.join(non_normal)}")
        L.append("  → Mann-Whitney-U applied to all items (conservative choice)")
    else:
        L.append("  → All items normally distributed; Mann-Whitney used regardless (ordinal data)")

    # Mann-Whitney U
    mw_results = {}
    L += ["", "  [2] MANN-WHITNEY-U  Management vs. Technical",
          f"  H₀: no difference  |  p<{T['p_sig']} significant",
          "  Effect size r: <0.1 negligible | 0.1–0.3 small | 0.3–0.5 moderate | >0.5 large",
          "  " + sep("─", 72),
          f"  {'Item':<18} {'Mean_Mgmt':>10} {'Mean_Tech':>10} {'Δ':>6} "
          f"{'p':>7} {'Sig':>4}  {'r':>5}  effect"]
    for name, series in items.items():
        g1 = series[masks["Management"]].dropna()
        g2 = series[masks["Technical"]].dropna()
        if len(g1) < 3 or len(g2) < 3: continue
        u_stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        delta     = g1.mean() - g2.mean()
        r_eff     = mannwhitney_r(u_stat, len(g1), len(g2))
        r_label   = ("large" if r_eff >= 0.5 else "moderate" if r_eff >= 0.3
                     else "small" if r_eff >= 0.1 else "negligible")
        mw_results[name] = dict(m1=g1.mean(), m2=g2.mean(), n1=len(g1),
                                n2=len(g2), delta=delta, p=p, r=r_eff)
        L.append(f"  {name:<18} {g1.mean():>10.2f} {g2.mean():>10.2f} "
                 f"{delta:>+6.2f} {p:>7.3f} {sig_stars(p):>4}  {r_eff:>5.3f}  {r_label}")

    # Kruskal-Wallis
    kw_results = {}
    L += ["", "  [3] KRUSKAL-WALLIS  all 3 role groups",
          f"  H₀: all groups equal  |  p<{T['p_sig']} at least one group differs",
          "  " + sep("─", 68),
          f"  {'Item':<18} {'Mean_Mgmt':>10} {'Mean_Tech':>10} {'Mean_NTec':>10} "
          f"{'p':>7}  Sig."]
    for name, series in items.items():
        g1 = series[masks["Management"]].dropna()
        g2 = series[masks["Technical"]].dropna()
        g3 = series[masks["Non-Technical"]].dropna()
        if len(g1) < 3 or len(g2) < 3 or len(g3) < 3: continue
        _, p = stats.kruskal(g1, g2, g3)
        kw_results[name] = dict(m1=g1.mean(), m2=g2.mean(), m3=g3.mean(), p=p)
        L.append(f"  {name:<18} {g1.mean():>10.2f} {g2.mean():>10.2f} "
                 f"{g3.mean():>10.2f} {p:>7.3f}  {sig_stars(p)}")

    # Recommendations
    L += ["", "  RECOMMENDATIONS", "  " + sep("─", 52)]
    sig_mw = [(n, v) for n, v in mw_results.items() if v["p"] < T["p_sig"]]
    if sig_mw:
        L.append("  Significant group differences (Mann-Whitney):")
        for name, v in sig_mw:
            higher = "Management" if v["delta"] > 0 else "Technical"
            L.append(f"    • {name}: {higher} rates higher "
                     f"(Δ={v['delta']:+.2f}, p={v['p']:.3f}, r={v['r']:.3f})")
    else:
        L.append(f"  No significant differences (all p>{T['p_sig']}).")
        min_n_grp = min(ns.values())
        if min_n_grp < 30:
            L.append(f"  ⚠ Smallest group n={min_n_grp} – insufficient power to detect "
                     "medium effects. Do not conclude absence of differences.")
            L.append("    Re-test with full dataset.")
    trends = [(n, v) for n, v in mw_results.items()
              if v["p"] >= T["p_sig"] and abs(v["delta"]) >= T["trend_delta"]]
    if trends:
        L.append(f"  Directional trends (n.s. but |Δ|≥{T['trend_delta']} – monitor):")
        for name, v in sorted(trends, key=lambda x: abs(x[1]["delta"]), reverse=True):
            higher = "Management" if v["delta"] > 0 else "Technical"
            L.append(f"    • {name}: {higher} tends higher "
                     f"({v['m1']:.2f} vs {v['m2']:.2f}, Δ={v['delta']:+.2f}, r={v['r']:.3f})")
    out(L, report_lines)


# =============================================================================
# VISUALISATION
# =============================================================================

def create_plots(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Green Software Metrics Survey – Descriptive Overview",
                 fontsize=13, fontweight="bold", y=0.98)

    want_cols   = SURVEY_DIMENSIONS["want_objectives"]["items"]
    want        = pd.DataFrame({name: pd.to_numeric(df.iloc[:, col], errors="coerce")
                                for name, col in want_cols.items()}).dropna()
    short_labels = ["CO₂", "Energy", "Cost", "Perf.", "Compl."]
    want.columns = short_labels

    # Panel A – Spearman inter-item correlations
    ax   = axes[0, 0]
    corr = want.corr(method="spearman")
    ni   = len(corr)
    for i in range(ni):
        for j in range(ni):
            v = corr.iloc[i, j]
            c = ("#CCCCCC" if i == j else
                 "#1a6e3c" if v >= T["r_strong"] else
                 "#52a87a" if v >= T["r_moderate"] else
                 "#a8d5b5" if v >= T["r_weak"] else
                 "#f0f7f2" if v >= 0 else "#e8b4b4")
            ax.add_patch(mpatches.FancyBboxPatch(
                (j - .45, ni - 1 - i - .45), .9, .9,
                boxstyle="round,pad=0.05", facecolor=c, edgecolor="white", linewidth=1.5))
            if i != j:
                t_s = v * np.sqrt((len(want) - 2) / max(1 - v**2, 1e-10))
                p_v = 2 * stats.t.sf(abs(t_s), df=len(want) - 2)
                ax.text(j, ni - 1 - i, f"{v:.2f}{sig_stars(p_v).strip()}",
                        ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color="white" if abs(v) >= T["r_strong"] else "#222")
    ax.set_xlim(-.5, ni - .5); ax.set_ylim(-.5, ni - .5)
    ax.set_xticks(range(ni)); ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_yticks(range(ni)); ax.set_yticklabels(reversed(short_labels), fontsize=8.5)
    ax.set_title("A  Spearman ρ  Inter-Item Correlations", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left",
              handles=[mpatches.Patch(facecolor=c, label=l) for c, l in
                       [("#1a6e3c", f"ρ≥{T['r_strong']}"),
                        ("#52a87a", f"ρ≥{T['r_moderate']}"),
                        ("#a8d5b5", f"ρ≥{T['r_weak']}"),
                        ("#e8b4b4", "ρ<0")]])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(length=0)

    # Panel B – Medians ± IQR with Wilcoxon significance
    ax2   = axes[0, 1]
    mdns  = [want[c].median()       for c in short_labels]
    q1s   = [want[c].quantile(0.25) for c in short_labels]
    q3s   = [want[c].quantile(0.75) for c in short_labels]
    w_sigs = []
    for c in short_labels:
        diffs = (want[c].dropna() - 3.0).pipe(lambda x: x[x != 0])
        if len(diffs) >= 4:
            _, p = stats.wilcoxon(diffs, alternative="two-sided")
            w_sigs.append(sig_stars(p).strip())
        else:
            w_sigs.append("")
    ax2.barh(short_labels[::-1], mdns[::-1],
             xerr=[[m - q for m, q in zip(mdns[::-1], q1s[::-1])],
                   [q - m for m, q in zip(mdns[::-1], q3s[::-1])]],
             color="#457b9d", alpha=0.85,
             error_kw=dict(ecolor="#444", capsize=4, lw=1.5))
    ax2.axvline(3, color="#888", linestyle="--", lw=1, label="neutral (3.0)")
    ax2.set_xlim(1, 7.5)
    ax2.set_xlabel("Median  (error bars = IQR,  scale 1–5)", fontsize=8.5)
    ax2.set_title("B  Medians ± IQR  (Wilcoxon vs. μ₀=3)", fontsize=10, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    for i, (m, q3, sig) in enumerate(zip(mdns[::-1], q3s[::-1], w_sigs[::-1])):
        ax2.text(q3 + 0.15, i, f"{m:.1f} {sig}", va="center", fontsize=8.5)
    ax2.legend(fontsize=8, loc="lower right")

    # Panel C – CAN barriers sorted
    ax3   = axes[1, 0]
    bcols = {name: col for name, col in SURVEY_DIMENSIONS["can_barriers"]["items"].items()
             if name != "No_barriers"}
    bdata = sorted([(name, (df.iloc[:, col] == "Yes").sum() / len(df) * 100)
                    for name, col in bcols.items()], key=lambda x: x[1], reverse=True)
    bn, bp = zip(*bdata)
    bars   = ax3.barh(bn, bp, color="#e76f51", alpha=0.8)
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("% respondents", fontsize=8.5)
    ax3.set_title("C  CAN – Barriers (sorted by prevalence)", fontsize=10, fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)
    for bar, pct in zip(bars, bp):
        ax3.text(pct + 1, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.0f}%", va="center", fontsize=8)

    # Panel D – single-item distributions (Effort, Outlook)
    ax4    = axes[1, 1]
    ef_col = list(SURVEY_DIMENSIONS["want_effort"]["items"].values())[0]
    ol_col = list(SURVEY_DIMENSIONS["outlook_likely"]["items"].values())[0]
    ef     = df.iloc[:, ef_col].map(EFFORT_MAP).dropna()
    ol     = df.iloc[:, ol_col].map(OUTLOOK_MAP).dropna()
    ax4.set_xlim(.5, 2.5); ax4.set_ylim(.5, 5.5)
    for pos, (lbl, series) in enumerate([("Effort\n(want)", ef), ("Adopt\n(outlook)", ol)]):
        q1, med, q3 = series.quantile([.25, .5, .75])
        x = pos + 1
        ax4.add_patch(mpatches.FancyBboxPatch(
            (x - .25, q1), .5, q3 - q1,
            boxstyle="round,pad=0.02", facecolor="#a8dadc",
            edgecolor="#1d3557", linewidth=1.5))
        ax4.plot([x - .25, x + .25], [med, med], color="#1d3557", lw=2)
        ax4.plot([x, x], [series.min(), q1], color="#1d3557", lw=1.2)
        ax4.plot([x, x], [q3, series.max()], color="#1d3557", lw=1.2)
        ax4.scatter(x, series.mean(), color="#e63946", zorder=5, s=50,
                    label="Mean" if pos == 0 else "")
        ax4.text(x + .3, series.mean(), f"Mdn={med:.1f}", va="center", fontsize=8.5)
    ax4.axhline(3, color="#888", linestyle="--", lw=1, label="neutral")
    ax4.set_xticks([1, 2]); ax4.set_xticklabels(["Effort\n(want)", "Adopt\n(outlook)"], fontsize=9)
    ax4.set_yticks(range(1, 6))
    ax4.set_yticklabels(["1\nstrongly\ndisagree", "2", "3", "4", "5\nstrongly\nagree"], fontsize=7.5)
    ax4.set_title("D  Single-item distributions (box=IQR, dot=Mean)", fontsize=10, fontweight="bold")
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    print(f"  → Plot saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    csv_path = csv_arg(sys.argv, DEFAULT_CSV)
    print(f"\nLoading: {csv_path}")
    df = load_survey(csv_path)
    print(f"n={len(df)}  |  {df.shape[1]} columns\n")

    report_lines = [
        "GREEN SOFTWARE METRICS SURVEY – DESCRIPTIVE & RELIABILITY REPORT",
        f"File: {display_path(csv_path)}  |  n={len(df)}",
        sep("═"), "",
    ]

    handlers = {
        "likert":      analyse_likert,
        "likert_text": analyse_likert_text,
        "binary":      analyse_binary,
        "multiselect": analyse_multiselect,
        "ordinal":     analyse_ordinal,
    }
    for dim_key, spec in SURVEY_DIMENSIONS.items():
        h = handlers.get(spec["type"])
        if h:
            h(dim_key, spec, df, report_lines)
        if dim_key == "context_maturity":
            analyse_domain_profile(df, report_lines)
            analyse_system_context(df, report_lines)

    analyse_computed_scores(df, report_lines)
    analyse_variance_quality(df, report_lines)
    analyse_midpoint_test(df, report_lines)
    analyse_group_comparisons(df, report_lines)

    out_path = report_path("descriptive_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  → Report saved: {out_path}")

    create_plots(df, report_path("descriptive_plots.png"))
    print()


if __name__ == "__main__":
    main()