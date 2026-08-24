"""
=============================================================================
Green Software Metrics Survey – Hypothesis Tests (H1 – H8)
=============================================================================
Usage:   python hypothesis_test.py [path/to/export.csv]
Output:  ../reports/hypothesis_report.txt

Construct definitions (all driven by config.py):

  CONTEXT (grouping variables)
    Primary    : size_250 (SME <=249 / Ent >=250, EU SME definition)
                 domain_web (Web flag)
                 domain_breadth (<=1 / >=2 domains)
                 maturity_strict ("I do not know" excluded)
    Sensitivity: size_1000 (SME <=999 / Ent >=1000)

  WANT
    want_env  = mean(CO2_reduction, Energy_reduction)         [environmental]
    want_ops  = mean(Cost_reduction, Performance, Compliance) [operational]
    want_all  = mean of all five objectives                   [overall]

  CAN
    can_constraint (0/1/2)          Primary for H2. 3-level capability-constraint
                ordinal on the structural/addressable barrier split:
                0 structurally blocked, 1 addressable-only, 2 clear, higher =
                more capable. A skipped barrier item is non-informative (NaN),
                not a maximum score, and a structural barrier dominates.
    can_effort  (Likert 1-5)        Primary for H5, complementary for H2.
                Direct self-report of perceived worthwhileness.
    can_score   (10 - barrier count) Legacy flat count, retained only as the
                H2 sensitivity comparison. Superseded by can_constraint because
                it flattens severity and scores a skipped item as fully capable.

  DO
    do_breadth           [0-3]  tier ladder; higher tier requires ALL lower
                                tiers present (strict cumulative reading)
    do_institutionalized [0-3]  decision_frequency (0/1/2) + green_req (0/1)
    do_composite         [0-6]  = do_breadth + do_institutionalized

  OUTLOOK
    outlook_intent     Likert 1-5  ("Strongly unlikely" ... "Strongly likely")
    outlook_awareness  count of decision areas impacted            [0-9]

Hypothesis tests
----------------
  H1  Context -> Want    Mann-Whitney U; Holm-Bonferroni across k=2
                         confirmatory pairs (size_250 x want_ops,
                         maturity_strict x want_env).  domain_web x want_env
                         was selected after screening all five domain
                         categories, so it sits outside the correction family
                         and is reported uncorrected with a Bonferroni bound
                         over those five.  Full 3x3 matrix reported
                         uncorrected.
  H2  Context -> Can     Mann-Whitney U over {domain_breadth, size_250,
                         maturity_strict} x can_constraint (primary, Holm k=3);
                         can_effort complementary; can_score as sensitivity.
  H3  Context -> Do      Mann-Whitney U over {size_250, maturity_strict,
                         domain_breadth} x do_composite (run_ctx_do).
  H4  Want -> Do         Spearman rho, 3x3 Want x Do matrix.
                         Primary pair: want_env x do_institutionalized.
  H5  Can -> Do          Spearman rho.
                         Primary:     can_effort x do_composite
                         Sensitivity: can_score  x do_composite  (caveat)
  H6  Want -> Outlook    Assessed jointly via PLS-SEM structural model.
  H7  Can  -> Outlook    Assessed jointly via PLS-SEM structural model.
  H8  Do   -> Outlook    Spearman rho on do_composite x {outlook_intent,
                                                         outlook_awareness}

  NOTE ON NUMBERING: run_* names and local variables use paper H-numbers.
    run_h1  = H1 Ctx->Want,   run_h2    = H2 Ctx->Can,
    run_ctx_do = H3 Ctx->Do,  run_h3    = H4 Want->Do,
    run_h4  = H5 Can->Do,     run_h5    = H8 Do->Outlook.
    H6/H7 (Want/Can->Outlook) are assessed in the PLS-SEM, not here;
    their reported intervals come from pls_bootstrap.py.
  LaTeX macros are relationship-named (gsmWantDo*, gsmCtxDo*, ...), so the
  number mapping lives only in the .tex display, not here.
=============================================================================
"""

import sys
import warnings
import pandas as pd
import numpy as np
from scipy import stats

import config
from config import (
    SURVEY_DIMENSIONS, OUTLOOK_MAP, EFFORT_MAP, T,
    CONTEXT_SIZE_GROUPS, CONTEXT_SIZE_GROUPS_LEGACY_1000,
    CONTEXT_MATURITY_LOW, CONTEXT_MATURITY_HIGH,
    CONTEXT_MATURITY_ORDINAL_MAP,
    CONTEXT_DOMAIN_PRIMARY,
    DOMAIN_COLS,
    DO_METRIC_WEIGHTS, DO_TIER_PROXY, DO_TIER_PHYSICAL, DO_TIER_ENVIRONMENTAL,
    DO_DECISION_MAP, DO_DECISION_COL, DO_REQUIREMENTS_COL,
    ROLE_COLS,
    CAN_OTHER_BARRIER_COL,
    CAN_STRUCTURAL_BARRIERS, CAN_ADDRESSABLE_BARRIERS,
    load_survey,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

DEFAULT_CSV = config.data_path("results-survey668719.csv")
MIN_GROUP_N = 10   # groups smaller than this receive a low-power flag


# =============================================================================
# UTILITIES
# =============================================================================

def sep(char="─", w=65):
    return char * w


def sig_stars(p):
    if p is None:       return "    "
    if p < T["p_sig3"]: return "***"
    if p < T["p_sig2"]: return "** "
    if p < T["p_sig"]:  return "*  "
    return "n.s."


def mannwhitney_r(u, n1, n2):
    n       = n1 + n2
    mu_u    = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n + 1) / 12)
    z       = (u - mu_u) / sigma_u if sigma_u > 0 else 0
    return abs(z) / np.sqrt(n)


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values (parallel list)."""
    m = len(p_values)
    if m == 0:
        return []
    order       = sorted(range(m), key=lambda i: p_values[i])
    adjusted    = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj         = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def _r_label(r):
    return ("large"    if r >= 0.5 else
            "moderate" if r >= 0.3 else
            "small"    if r >= 0.1 else
            "negligible")


def fmt_p(p):
    if p is None:  return "   n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _fmt_padj(p_adj):
    if p_adj is None: return "      "
    return f"{p_adj:>6.3f}"


def out(lines, report_lines):
    print("\n".join(lines))
    report_lines += lines + [""]


# =============================================================================
# DIMENSION SCORE COMPUTATION
# =============================================================================

def build_dimension_scores(df):
    """One numeric score per respondent per dimension concept.
    All scoring rules driven by config.py – no magic numbers here."""
    scores = pd.DataFrame(index=df.index)

    # ── WANT ─────────────────────────────────────────────────────────────────
    want_items = SURVEY_DIMENSIONS["want_objectives"]["items"]
    clusters   = SURVEY_DIMENSIONS["want_objectives"]["clusters"]
    env_names  = clusters["Environmental"]
    ops_names  = clusters["Operational"]

    def to_num(name):
        return pd.to_numeric(df.iloc[:, want_items[name]], errors="coerce")

    scores["want_env"] = pd.concat([to_num(n) for n in env_names], axis=1).mean(axis=1)
    scores["want_ops"] = pd.concat([to_num(n) for n in ops_names], axis=1).mean(axis=1)
    scores["want_all"] = pd.concat([to_num(n) for n in want_items], axis=1).mean(axis=1)

    # ── CAN (two operationalizations reported side-by-side) ──────────────────
    # can_effort: direct Likert measure of perceived worthwhileness of effort
    effort_col = SURVEY_DIMENSIONS["want_effort"]["items"]["Effort_worthwhile"]
    scores["can_effort"] = df.iloc[:, effort_col].map(EFFORT_MAP)

    # can_constraint: 3-level capability-constraint ordinal (PRIMARY for H2),
    # built on the structural/addressable barrier split in config.py.
    # Higher = more capable (fewer and less severe constraints).
    #   2 = clear             (respondent affirmatively checked "No barriers")
    #   1 = addressable-only  (only fixable barriers: skills, tooling, methods)
    #   0 = structurally blocked (a barrier the org cannot remove on its own)
    #   NaN = non-informative (checked nothing and did not affirm "no barriers")
    # The structural level dominates: any structural barrier forces level 0,
    # so its presence cannot be masked by the absence of addressable ones.
    barrier_cols = SURVEY_DIMENSIONS["can_barriers"]["items"]
    no_barrier   = df.iloc[:, barrier_cols["No_barriers"]] == "Yes"

    def _any_barrier(names):
        m = pd.Series(False, index=df.index)
        for nm in names:
            m = m | (df.iloc[:, barrier_cols[nm]] == "Yes")
        return m

    structural_present  = _any_barrier(CAN_STRUCTURAL_BARRIERS)
    addressable_present = _any_barrier(CAN_ADDRESSABLE_BARRIERS)

    # A non-empty free-text "Other" barrier is treated as an addressable barrier,
    # so an Other-only respondent is not mis-scored as having no barriers.
    other_txt = df.iloc[:, CAN_OTHER_BARRIER_COL].astype(str).str.strip()
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

    # Legacy inverted barrier count (10 − raw_count), retained ONLY as a
    # sensitivity comparison for H2. Superseded by can_constraint because the
    # flat count flattens severity and scores a skipped item as maximum
    # capability. Barrier universe = 9 listed + Other = 10.
    non_sentinel = {k: v for k, v in barrier_cols.items() if k != "No_barriers"}
    raw_count = pd.Series(0.0, index=df.index)
    for _, c in non_sentinel.items():
        raw_count += (df.iloc[:, c] == "Yes").astype(float)
    raw_count += other_present.astype(float)
    raw_count = raw_count.where(~no_barrier, other=0.0)
    n_possible_barriers     = len(non_sentinel) + 1   # 9 listed + Other
    scores["can_score"]     = n_possible_barriers - raw_count
    scores["barrier_count"] = raw_count

    # ── DO – breadth (strict tier ladder: higher tier requires lower tiers) ──
    metric_cols = SURVEY_DIMENSIONS["do_metrics"]["items"]
    has_proxy = pd.Series(False, index=df.index)
    has_phys  = pd.Series(False, index=df.index)
    has_env   = pd.Series(False, index=df.index)
    weighted  = pd.Series(0.0, index=df.index)
    raw       = pd.Series(0,   index=df.index)

    for name, col in metric_cols.items():
        t = (df.iloc[:, col] == "Yes").astype(int)
        raw      += t
        weighted += t * DO_METRIC_WEIGHTS.get(name, 1)
        if name in DO_TIER_PROXY:         has_proxy |= t.astype(bool)
        if name in DO_TIER_PHYSICAL:      has_phys  |= t.astype(bool)
        if name in DO_TIER_ENVIRONMENTAL: has_env   |= t.astype(bool)

    breadth = pd.Series(0, index=df.index)
    breadth = breadth.where(~has_proxy,                        other=1)
    breadth = breadth.where(~(has_proxy & has_phys),           other=2)
    breadth = breadth.where(~(has_proxy & has_phys & has_env), other=3)
    scores["do_breadth"]  = breadth
    scores["do_weighted"] = weighted
    scores["do_raw"]      = raw

    # Sensitivity variant: highest tier reached, without the cumulative rule.
    # The cumulative ladder treats an environmental indicator reported without
    # underlying instrumentation as modeled rather than measured and scores it
    # at the tier actually supported; this variant credits the highest tier
    # claimed.  The two disagree for a non-trivial share of respondents, so the
    # choice is reported as a sensitivity check rather than assumed away.
    highest = pd.Series(0, index=df.index)
    highest[has_proxy] = 1
    highest[has_phys]  = 2
    highest[has_env]   = 3
    scores["do_highest"] = highest

    # ── DO – institutionalization & composite ────────────────────────────────
    freq_score = df.iloc[:, DO_DECISION_COL].map(DO_DECISION_MAP)
    req_score  = (df.iloc[:, DO_REQUIREMENTS_COL] == "Yes").astype(float)
    scores["do_institutionalized"] = freq_score + req_score
    scores["do_composite"]         = scores["do_breadth"] + scores["do_institutionalized"]

    # ── OUTLOOK ──────────────────────────────────────────────────────────────
    outlook_col = list(SURVEY_DIMENSIONS["outlook_likely"]["items"].values())[0]
    scores["outlook_intent"] = df.iloc[:, outlook_col].map(OUTLOOK_MAP)

    impact_cols = SURVEY_DIMENSIONS["outlook_impact"]["items"]
    scores["outlook_awareness"] = pd.concat(
        [(df.iloc[:, c] == "Yes").astype(int) for c in impact_cols.values()], axis=1
    ).sum(axis=1).astype(float)

    # ── CONTEXT – maturity ordinal (for Spearman analyses) ───────────────────
    # LimeSurvey stores "None" answer as an empty cell → fillna before mapping
    mat_col = df.iloc[:, SURVEY_DIMENSIONS["context_maturity"]["items"]["Sustainability_maturity"]]
    scores["maturity_ordinal"] = mat_col.fillna("None").map(CONTEXT_MATURITY_ORDINAL_MAP)

    # ── CONTEXT – domain count ordinal 0–5 (complement to binary domain_breadth)
    scores["domain_count"] = pd.concat(
        [(df.iloc[:, c] == "Yes").astype(int) for c in DOMAIN_COLS.values()], axis=1
    ).sum(axis=1).astype(float)

    return scores


# =============================================================================
# CONTEXT GROUP MASK BUILDERS
# =============================================================================

def build_context_masks(df):
    """Return dict of {split_name: {group_label: boolean_mask}}.
    Each split defines exactly two groups used in a Mann-Whitney test."""
    masks = {}
    size_col = df.iloc[:, SURVEY_DIMENSIONS["context_size"]["items"]["Org_size"]]

    # PRIMARY: EU SME threshold (≤249 vs ≥250)
    masks["size_250"] = {
        lbl: size_col.isin(vals) for lbl, vals in CONTEXT_SIZE_GROUPS.items()
    }
    # SENSITIVITY: alternative size threshold (≤999 vs ≥1000)
    masks["size_1000"] = {
        lbl: size_col.isin(vals) for lbl, vals in CONTEXT_SIZE_GROUPS_LEGACY_1000.items()
    }

    # Maturity strict (excludes "I do not know")
    mat_col = df.iloc[:, SURVEY_DIMENSIONS["context_maturity"]["items"]["Sustainability_maturity"]]
    mat_col_filled = mat_col.fillna("None")   # LimeSurvey stores "None" answer as empty cell
    masks["maturity_strict"] = {
        "Low maturity":  mat_col_filled.isin(CONTEXT_MATURITY_LOW),
        "High maturity": mat_col_filled.isin(CONTEXT_MATURITY_HIGH),
    }

    # Domain – Web flag (single-domain moderator)
    web_col_idx = DOMAIN_COLS[CONTEXT_DOMAIN_PRIMARY]
    web_flag    = df.iloc[:, web_col_idx]
    masks["domain_web"] = {
        f"Non-{CONTEXT_DOMAIN_PRIMARY}": web_flag == "No",
        f"{CONTEXT_DOMAIN_PRIMARY}":     web_flag == "Yes",
    }

    # Domain breadth (≤1 vs ≥2 domains)
    dom_count = pd.concat(
        [(df.iloc[:, c] == "Yes").astype(int) for c in DOMAIN_COLS.values()], axis=1
    ).sum(axis=1)
    masks["domain_breadth"] = {
        "Narrow (≤1)": dom_count <= 1,
        "Broad (≥2)":  dom_count >= 2,
    }

    # Per-domain individual binary splits (exploratory, for H1 domain breakdown)
    for domain_name, col_idx in DOMAIN_COLS.items():
        flag = df.iloc[:, col_idx]
        masks[f"domain_{domain_name}"] = {
            f"Non-{domain_name}": flag == "No",
            f"{domain_name}":     flag == "Yes",
        }

    return masks


def model_frame(df):
    """The listwise-complete frame the structural model is fitted on.

    Every column is an indicator of a latent variable in
    pls_bootstrap.make_cfg(), so the dropna() defines the analysis sample
    exactly: it excludes Want item non-response and maturity "I do not know".

    It lives here rather than in pls_bootstrap.py because the count of rows it
    returns is the paper's listwise-complete n, which gen_latex.py must be able
    to compute without importing the PLS stack.
    """
    S = build_dimension_scores(df)
    wi = SURVEY_DIMENSIONS["want_objectives"]["items"]
    return pd.DataFrame({
        "maturity":   S["maturity_ordinal"],
        "co2":        pd.to_numeric(df.iloc[:, wi["CO2_reduction"]], errors="coerce"),
        "energy":     pd.to_numeric(df.iloc[:, wi["Energy_reduction"]], errors="coerce"),
        "can_effort": S["can_effort"],
        "do_breadth": S["do_breadth"],
        "do_inst":    S["do_institutionalized"],
        "ol_intent":  S["outlook_intent"],
        "ol_aware":   S["outlook_awareness"],
    }).dropna().reset_index(drop=True)


# =============================================================================
# STATISTICAL PRIMITIVES
# =============================================================================

def spearman_test(x, y, label_x, label_y, positive_prediction=True):
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    n      = len(paired)
    if n < 5:
        return {"lx": label_x, "ly": label_y, "n": n,
                "rho": None, "p": None, "positive_pred": positive_prediction}
    rho, p = stats.spearmanr(paired["x"], paired["y"])
    return {"lx": label_x, "ly": label_y, "n": n,
            "rho": rho, "p": p, "positive_pred": positive_prediction}


def spearman_tost(x, y, delta):
    """Two one-sided tests for equivalence of a Spearman correlation.

    A non-significant correlation cannot distinguish "no association" from
    "too small a sample to see one".  TOST turns the question round and asks
    whether an association as large as +/-delta can be *rejected*.  Both
    one-sided tests must reject, which is equivalent to the 90% interval
    falling entirely inside (-delta, +delta).

    delta is a smallest effect size of interest and must come from theory, not
    from the sample: here, the correlation two indicators of a single construct
    would be expected to reach.

    Fisher-z scale with the Bonett-Wright standard error for Spearman's rho.
    """
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(d)
    rho = stats.spearmanr(d["x"], d["y"]).statistic
    z = np.arctanh(rho)
    se = 1.06 / np.sqrt(n - 3)
    p_lower = 1 - stats.norm.cdf((z - np.arctanh(-delta)) / se)   # H0: rho <= -delta
    p_upper = stats.norm.cdf((z - np.arctanh(delta)) / se)        # H0: rho >= +delta
    return {"n": n, "rho": float(rho), "delta": delta,
            "p": float(max(p_lower, p_upper)),
            "lo90": float(np.tanh(z - 1.645 * se)),
            "hi90": float(np.tanh(z + 1.645 * se))}


def disattenuate(rho, rel_x, rel_y=1.0):
    """Correlation corrected for attenuation due to measurement error.

    Unreliable measures pull an observed correlation toward zero, so a null
    can be an artifact of noise rather than an absence of association.  The
    correction gives the value the correlation would take with perfectly
    reliable measures, and is therefore an upper bound on how much of a null
    unreliability can explain.
    """
    return float(rho / np.sqrt(rel_x * rel_y))


# ---------------------------------------------------------------------------
# BATCHED RESAMPLING PRIMITIVES
#
# The three bootstraps below each draw thousands of resamples.  Calling
# stats.spearmanr or stats.mannwhitneyu once per draw spends nearly all of its
# time in argument validation rather than arithmetic, so the draws are taken as
# one index matrix and the statistic is evaluated a row at a time.
#
# Both helpers reproduce their scipy counterpart exactly, not approximately:
#   * spearmanr ranks its inputs with rankdata and returns
#     np.corrcoef(vstack((rank_x, rank_y)))[1, 0].  _rho_rows makes that same
#     call per row, on ranks from the same rankdata.
#   * mannwhitneyu ranks the concatenated samples and returns
#     sum(ranks of the first sample) - n1(n1+1)/2.  _u_rows repeats it.
# A block draw rng.integers(0, n, (n_boot, n)) is the same stream as n_boot
# successive rng.integers(0, n, n) calls, so the resamples are unchanged too.
# Every interval these functions produce is therefore bit-for-bit what the
# per-draw scipy loop produced.
# ---------------------------------------------------------------------------

def _rank_rows(m):
    """Average ranks within each row, as stats.spearmanr ranks its inputs."""
    return stats.rankdata(m, axis=1)


def _rho_rows(rank_a, rank_b):
    """Row-wise Spearman rho from two matrices of ranks."""
    out = np.empty(len(rank_a))
    for i in range(len(rank_a)):
        out[i] = np.corrcoef(np.vstack((rank_a[i], rank_b[i])), rowvar=True)[1, 0]
    return out


def _u_rows(rank_ab, n1):
    """Row-wise Mann-Whitney U for the first sample, from concatenated ranks."""
    return rank_ab[:, :n1].sum(axis=-1) - n1 * (n1 + 1) / 2


def spearman_ci(x, y, n_boot=10000, seed=7):
    """Percentile bootstrap interval for a Spearman correlation.

    A p-value says whether an association is distinguishable from zero; the
    interval says how precisely it is pinned down, which at this sample size is
    the more informative quantity.  Resampling is over cases, so it carries the
    ordinal scale and its ties through unchanged.
    """
    d = pd.DataFrame({"x": x, "y": y}).dropna().reset_index(drop=True)
    n = len(d)
    if n < 10:
        return None
    rho = stats.spearmanr(d["x"], d["y"]).statistic
    rng = np.random.default_rng(seed)
    xs, ys = d["x"].values, d["y"].values
    idx = rng.integers(0, n, (n_boot, n))
    bs = _rho_rows(_rank_rows(xs[idx]), _rank_rows(ys[idx]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"n": n, "rho": rho, "lo": float(lo), "hi": float(hi)}


def hodges_lehmann(g1, g2, alpha=0.05):
    """Hodges-Lehmann shift estimate and its distribution-free interval.

    The Mann-Whitney test reports whether two groups differ; the difference in
    sample medians reported alongside it is a point estimate with no stated
    precision, and is not the quantity the test is actually about.  The
    Hodges-Lehmann estimator -- the median of all pairwise between-group
    differences -- is that quantity, and it admits an exact interval derived
    from the same rank distribution as the test.
    """
    a = np.asarray(pd.Series(g1).dropna(), dtype=float)
    b = np.asarray(pd.Series(g2).dropna(), dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return None
    diffs = np.sort((a[:, None] - b[None, :]).ravel())
    est = float(np.median(diffs))
    N = n1 * n2
    z = stats.norm.ppf(1 - alpha / 2)
    # Rank of the interval bound under the null distribution of U.
    k = int(np.floor(N / 2 - z * np.sqrt(N * (n1 + n2 + 1) / 12)))
    if k < 0:
        k = 0
    lo = float(diffs[k]) if k < N else float(diffs[0])
    hi = float(diffs[N - 1 - k]) if k < N else float(diffs[-1])
    return {"n1": n1, "n2": n2, "est": est, "lo": lo, "hi": hi}


def mannwhitney_r_ci(score_col, group_mask_pair, n_boot=5000, seed=7):
    """Percentile bootstrap interval for the standardized rank effect size r."""
    groups = {lbl: score_col[mask].dropna() for lbl, mask in group_mask_pair.items()}
    keys = list(groups.keys())
    if len(keys) != 2:
        return None
    a, b = groups[keys[0]].values, groups[keys[1]].values
    if len(a) < 3 or len(b) < 3:
        return None

    def eff(x, y):
        u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
        return mannwhitney_r(u, len(x), len(y))

    n1, n2 = len(a), len(b)
    # The two groups are resampled independently, so their draws alternate:
    # the index matrices are filled in that same order to keep the stream.
    rng = np.random.default_rng(seed)
    ia = np.empty((n_boot, n1), dtype=np.int64)
    ib = np.empty((n_boot, n2), dtype=np.int64)
    for i in range(n_boot):
        ia[i] = rng.integers(0, n1, n1)
        ib[i] = rng.integers(0, n2, n2)
    u = _u_rows(_rank_rows(np.hstack((a[ia], b[ib]))), n1)
    bs = mannwhitney_r(u, n1, n2)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"r": eff(a, b), "lo": float(lo), "hi": float(hi)}


def ordinal_logit(outcome, predictors):
    """Proportional-odds ordinal logistic regression on standardized predictors.

    This is the paper's joint estimator.  The outcome scores are ordinal, so an
    ordered logit models each on its own scale without compositing anything,
    and reports odds ratios with analytic confidence intervals.  It needs no
    measurement model, which matters here because three of the five model
    constructs rest on a single score each and a latent-variable estimator
    would have nothing to estimate for them.  (A PLS-SEM fit of the same
    relationships is retained in this package as a supplementary check; see
    pls_bootstrap.py.  It agrees on every substantive point.)

    `predictors` is a dict of {label: series}.  All series are standardized so
    the odds ratios are per standard deviation and comparable to one another.
    Returns None if the model does not converge.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    frame = pd.DataFrame({"__y": outcome, **predictors}).dropna()
    if len(frame) < 20:
        return None
    y = frame["__y"].astype(int)
    X = frame.drop(columns="__y")
    X = (X - X.mean()) / X.std(ddof=0)
    try:
        fit = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    except Exception:
        return None
    out = {"n": len(frame), "prsq": float(getattr(fit, "prsquared", float("nan"))),
           "terms": {}}
    for name in X.columns:
        b, se = fit.params[name], fit.bse[name]
        out["terms"][name] = {
            "or": float(np.exp(b)),
            "lo": float(np.exp(b - 1.96 * se)),
            "hi": float(np.exp(b + 1.96 * se)),
            "p": float(fit.pvalues[name]),
        }
    return out


def dependent_rho_diff(x1, x2, y, n_boot=10000, seed=7):
    """Difference between two dependent, overlapping Spearman correlations.

    Compares rho(x1, y) against rho(x2, y) -- the two predictors share the same
    outcome y, so the correlations are dependent and cannot be compared with an
    independent-samples test.  A non-parametric percentile bootstrap over cases
    is used rather than Williams' t, because the latter assumes bivariate
    normality while the paper's scores are ordinal.

    Returns the observed difference, its percentile CI, and a two-sided
    bootstrap p-value.  All three series are aligned listwise first, so the
    reported n is the listwise-complete count for the triple.
    """
    paired = pd.DataFrame({"x1": x1, "x2": x2, "y": y}).dropna().reset_index(drop=True)
    n = len(paired)
    if n < 10:
        return None
    r1 = stats.spearmanr(paired["x1"], paired["y"]).statistic
    r2 = stats.spearmanr(paired["x2"], paired["y"]).statistic
    r_pred = stats.spearmanr(paired["x1"], paired["x2"]).statistic
    rng = np.random.default_rng(seed)
    v1, v2, vy = (paired["x1"].values, paired["x2"].values, paired["y"].values)
    idx = rng.integers(0, n, (n_boot, n))
    ry = _rank_rows(vy[idx])
    diffs = (_rho_rows(_rank_rows(v1[idx]), ry)
             - _rho_rows(_rank_rows(v2[idx]), ry))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"n": n, "rho1": r1, "rho2": r2, "rho_predictors": r_pred,
            "diff": r1 - r2, "ci_low": lo, "ci_high": hi, "p": min(p, 1.0)}


def mw_test(score_col, group_mask_pair, label):
    groups = {lbl: score_col[mask].dropna() for lbl, mask in group_mask_pair.items()}
    keys   = list(groups.keys())
    if len(keys) != 2 or any(len(groups[k]) < 3 for k in keys):
        return None
    g1, g2 = groups[keys[0]], groups[keys[1]]
    u, p   = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    r      = mannwhitney_r(u, len(g1), len(g2))
    return {
        "label":     label,
        "g1_lbl":    keys[0],      "g2_lbl":    keys[1],
        "n1":        len(g1),      "n2":        len(g2),
        "mdn1":      g1.median(),  "mdn2":      g2.median(),
        "delta_mdn": g2.median() - g1.median(),   # B − A (direction of effect)
        "U": u, "p": p, "r": r,
    }


# =============================================================================
# H1 – CONTEXT → WANT (primary + sensitivity)
# =============================================================================

def run_h1(scores, ctx_masks, L):
    # Confirmatory pairs only.  domain_web was selected after screening all five
    # domain categories, so a Holm p_adj computed over it would not account for
    # the selection step; it is reported uncorrected as exploratory instead
    # (see EXPLORATORY_PAIRS below).
    THEORY_PAIRS = [
        ("size_250",        "want_ops", "SME vs Enterprise × Want(ops)"),
        ("maturity_strict", "want_env", "Low vs High maturity × Want(env)"),
    ]
    EXPLORATORY_PAIRS = [
        ("domain_web",      "want_env", "Non-Web vs Web × Want(env)  [post-hoc split]"),
    ]
    SENSITIVITY_SPLITS = [
        ("size_1000",       "SME (1–999) vs Enterprise (1000+)  [alternative threshold]"),
    ]
    ALL_SPLITS = [
        ("size_250",        "SME (1–249) vs Enterprise (250+)"),
        ("domain_web",      "Non-Web vs Web domain"),
        ("maturity_strict", "Low vs High maturity (strict)"),
    ]
    WANT_SUBS = [("want_env","Want(env)"), ("want_ops","Want(ops)"), ("want_all","Want(all)")]

    L += [sep("─"),
          "  H1  Context → Want  –  Mann-Whitney U",
          "  Primary (Holm k=2)  size_250 × want_ops         larger orgs face compliance/cost pressure",
          "                      maturity_strict × want_env  governance maturity strengthens intent",
          "  Post-hoc            domain_web × want_env       selected after screening all 5 domains;",
          "                                                  reported uncorrected (see below)",
          "  Exploratory         full 3×3 split × want-subscale matrix        (uncorrected)",
          "  Sensitivity         size_1000, alternative threshold             (uncorrected)",
          ""]

    # PRIMARY: 2 confirmatory theory-driven pairs (Holm k=2)
    primary_results = []
    for split_name, score_key, label in THEORY_PAIRS:
        r = mw_test(scores[score_key], ctx_masks[split_name], f"{split_name}×{score_key}")
        primary_results.append(r)

    valid_p = [(i, r["p"]) for i, r in enumerate(primary_results) if r is not None]
    adj     = holm_bonferroni([p for _, p in valid_p])
    p_adj   = {i: a for (i, _), a in zip(valid_p, adj)}

    L.append("  PRIMARY MODEL  (confirmatory pairs, Holm-Bonferroni k=2)")
    L.append(f"  {'Pair':<44} {'n_A':>5} {'n_B':>5}  {'Mdn_A':>6} {'Mdn_B':>6}  "
             f"{'ΔMdn':>6}  {'p_raw':>6} {'p_adj':>6}  effect")
    L.append("  " + sep("─", 100))
    for i, (split_name, score_key, pair_label) in enumerate(THEORY_PAIRS):
        r = primary_results[i]
        if r is None:
            L.append(f"  {pair_label:<44}  – skipped (group too small)")
            continue
        pa   = p_adj.get(i)
        mark = "*" if (pa is not None and pa < T["p_sig"]) else \
               ("." if r["p"] < T["p_sig"] else "")
        lp   = " ⚠" if min(r["n1"], r["n2"]) < MIN_GROUP_N else ""
        L.append(f"  {pair_label:<44} {r['n1']:>5} {r['n2']:>5}  "
                 f"{r['mdn1']:>6.2f} {r['mdn2']:>6.2f}  "
                 f"{r['delta_mdn']:>+6.2f}  {r['p']:>6.3f} {_fmt_padj(pa)}  "
                 f"{_r_label(r['r']):<10} {mark}{lp}")
    L.append("")

    # POST-HOC: domain split chosen after screening; uncorrected, plus a
    # conservative Bonferroni bound over the 5 screened domain categories.
    L.append("  POST-HOC SPLIT  (uncorrected; selection-aware Bonferroni bound over 5 domains)")
    for split_name, score_key, pair_label in EXPLORATORY_PAIRS:
        r = mw_test(scores[score_key], ctx_masks[split_name], f"{split_name}×{score_key}")
        if r is None:
            L.append(f"  {pair_label:<44}  – skipped (group too small)")
            continue
        bound = min(1.0, r["p"] * len(DOMAIN_COLS))
        L.append(f"  {pair_label:<44} {r['n1']:>5} {r['n2']:>5}  "
                 f"{r['mdn1']:>6.2f} {r['mdn2']:>6.2f}  "
                 f"{r['delta_mdn']:>+6.2f}  p_raw={r['p']:.4f}  "
                 f"selection-bound={bound:.3f}  {_r_label(r['r'])}")
    L.append("")

    def render_block_expl(split_name, split_pretty):
        pair = ctx_masks[split_name]
        keys = list(pair.keys())
        n1 = int(pair[keys[0]].sum()); n2 = int(pair[keys[1]].sum())
        lp_note = "  ⚠ low power" if min(n1, n2) < MIN_GROUP_N else ""
        L.append(f"  {split_pretty}{lp_note}")
        L.append(f"    n: {keys[0]}={n1}, {keys[1]}={n2}")
        L.append(f"  {'Sub-scale':<12} {'Mdn_A':>7} {'Mdn_B':>7}  {'ΔMdn':>6}  "
                 f"{'p_raw':>6}  effect")
        L.append("  " + sep("─", 65))
        for score_key, score_lbl in WANT_SUBS:
            r = mw_test(scores[score_key], pair, "")
            if r is None:
                L.append(f"  {score_lbl:<12}  – skipped (group too small)")
                continue
            is_primary = (split_name, score_key) in [(s, k) for s, k, _ in THEORY_PAIRS]
            tag  = "  ← primary" if is_primary else ""
            mark = "*" if r["p"] < T["p_sig"] else " "
            L.append(f"  {score_lbl:<12} {r['mdn1']:>7.2f} {r['mdn2']:>7.2f}  "
                     f"{r['delta_mdn']:>+6.2f}  {r['p']:>6.3f}  "
                     f"{_r_label(r['r']):<10} {mark}{tag}")
        L.append("")

    L.append("  EXPLORATORY MATRIX  (all 3 splits × 3 want subscales, uncorrected)")
    L.append("  Holm correction applies only to the confirmatory pairs above; the Web split is post-hoc.")
    L.append("")
    for split_name, split_pretty in ALL_SPLITS:
        render_block_expl(split_name, split_pretty)

    L.append("  SENSITIVITY (not included in Holm correction)")
    for split_name, split_pretty in SENSITIVITY_SPLITS:
        render_block_expl(split_name, split_pretty)

    L.append("  DOMAIN BREAKDOWN (each domain vs rest – exploratory, uncorrected)")
    L.append(f"  All {len(DOMAIN_COLS)} domains tested regardless of group size")
    L.append("  (⚠ = underpowered, not skipped). Sorted by prevalence, descending.")
    L.append("  Significance is driven by ΔMdn, not by how many respondents selected a domain.")
    L.append(f"  {CONTEXT_DOMAIN_PRIMARY} is the primary split above; shown here for direct comparison.")
    L.append("")
    domain_order = sorted(DOMAIN_COLS.keys(),
                          key=lambda d: ctx_masks[f"domain_{d}"][d].sum(), reverse=True)
    env_summary = []
    for domain_name in domain_order:
        split_key = f"domain_{domain_name}"
        primary_note = "  ← primary" if domain_name == CONTEXT_DOMAIN_PRIMARY else ""
        render_block_expl(split_key, f"Non-{domain_name} vs {domain_name}{primary_note}")
        env_summary.append((domain_name, mw_test(scores["want_env"], ctx_masks[split_key], "")))

    n_want_env = int(scores["want_env"].notna().sum())
    n_missing_want_env = len(scores) - n_want_env
    L.append("  WANT_ENV SUMMARY (prevalence order)")
    L.append(f"  n counts use want_env effective sample (n={n_want_env}; {n_missing_want_env} missing excluded per group).")
    L.append(f"  {'Domain':<24} {'n(dom)':>6} {'n(rest)':>7}  {'ΔMdn':>6}  {'p_raw':>6}  effect")
    L.append("  " + sep("─", 72))
    for domain_name, r in env_summary:
        if r is None:
            L.append(f"  {domain_name:<24}  – skipped")
            continue
        lp = "⚠" if r["n2"] < MIN_GROUP_N else " "
        mark = " *" if r["p"] < T["p_sig"] else "  "
        tag = " ← primary" if domain_name == CONTEXT_DOMAIN_PRIMARY else ""
        L.append(f"  {domain_name:<24} {r['n2']:>5}{lp} {r['n1']:>7}  "
                 f"{r['delta_mdn']:>+6.2f}  {r['p']:>6.3f}  {_r_label(r['r']):<10}{mark}{tag}")
    L.append("")

    # ── MATURITY ORDINAL (Spearman ρ) ────────────────────────────────────────
    n_ord = int(scores["maturity_ordinal"].notna().sum())
    L.append("  MATURITY ORDINAL – Spearman ρ  (richer use of 4-level scale)")
    L.append(f"  Ordinal 0–3 (None→Ad hoc→KPIs→SDLC-integrated), n={n_ord}  (excl. 'I do not know').")
    L.append("  Complement to binary split: uses full ordinal gradient, more statistical power.")
    L.append(f"  {'Pair':<44} {'n':>3}  {'ρ':>6}   {'p':>7}  direction")
    L.append("  " + sep("─", 72))
    for score_key, score_lbl in WANT_SUBS:
        r = spearman_test(scores["maturity_ordinal"], scores[score_key],
                          "maturity_ordinal", score_key, positive_prediction=True)
        if r["rho"] is None:
            L.append(f"  {'maturity_ord × '+score_lbl:<44} {r['n']:>3}   n/a")
            continue
        dirn  = "pos +" if r["rho"] > 0 else "neg −"
        stars = sig_stars(r["p"]).strip()
        L.append(f"  {'maturity_ord × '+score_lbl:<44} {r['n']:>3}   {r['rho']:>+6.3f}  "
                 f"{r['p']:>6.3f} {stars:<4} {dirn}")
    L.append("")

    # ── Verdict ───────────────────────────────────────────────────────────────
    n_sig = sum(1 for i, _ in valid_p if p_adj[i] < T["p_sig"])
    n_dir = sum(1 for i, _ in valid_p
                if p_adj[i] >= T["p_sig"]
                and primary_results[i] is not None
                and abs(primary_results[i]["delta_mdn"]) >= T["trend_delta"])
    if n_sig >= 1:
        verdict = f"SUPPORTED ✓  ({n_sig}/{len(valid_p)} confirmatory pairs survive Holm)"
    elif n_dir >= 2:
        verdict = f"DIRECTIONAL  ({n_dir}/{len(valid_p)} show |ΔMdn|≥{T['trend_delta']}, n.s.)"
    else:
        verdict = "NOT SUPPORTED"
    L.append(f"  → H1 VERDICT:  {verdict}")
    L.append("")
    return verdict, primary_results, p_adj


# =============================================================================
# H2 – CONTEXT → CAN (primary + sensitivity)
# =============================================================================
#
# Scoring note: can_constraint (3-level severity ordinal) is the PRIMARY score
# for H2 because H2 asks whether Context groups *differ in perceived
# constraints*. It encodes barrier severity through the structural/addressable
# split instead of a flat count, and treats a skipped item as non-informative
# rather than as maximum capability. can_effort is reported alongside as a
# complementary measure, and the legacy can_score count as a sensitivity check.
#
# (For H5, can_effort is the primary score because barrier reporting is
# confounded by measurement engagement – see H5 construct-validity note.)

def run_h2(scores, ctx_masks, L):
    PRIMARY_SPLITS = [
        ("domain_breadth",  "Narrow (≤1) vs Broad (≥2) domain footprint"),
        ("size_250",        "SME (1–249) vs Enterprise (250+)"),
        ("maturity_strict", "Low vs High maturity (strict)"),
    ]

    L += [sep("─"),
          "  H2  Context → Can  –  Mann-Whitney U",
          "  Primary score:    can_constraint (0 structural / 1 addressable / 2 clear)",
          "                                   higher = fewer and less severe constraints",
          "  Complementary:    can_effort     ('effort worthwhile' Likert 1–5)",
          "  Sensitivity:      can_score      (legacy 10 − barrier count, superseded)",
          "",
          "  Prediction: broad domain footprint → heterogeneous stacks → severer barriers.",
          "              Size and maturity are secondary moderators.",
          ""]

    # Primary set (can_constraint) with Holm correction
    primary_results = []
    for split_name, _ in PRIMARY_SPLITS:
        r = mw_test(scores["can_constraint"], ctx_masks[split_name], f"{split_name}×can_constraint")
        primary_results.append(r)

    valid_p = [(i, r["p"]) for i, r in enumerate(primary_results) if r is not None]
    adj = holm_bonferroni([p for _, p in valid_p])
    p_adj = {i: a for (i, _), a in zip(valid_p, adj)}

    L.append("  PRIMARY: can_constraint  (Holm-Bonferroni across the 3 splits)")
    L.append(f"  {'Split':<46} {'Mdn_A':>6} {'Mdn_B':>6}  {'ΔMdn':>6}  "
             f"{'p_raw':>6} {'p_adj':>6}  effect")
    L.append("  " + sep("─", 92))
    for i, (split_name, split_pretty) in enumerate(PRIMARY_SPLITS):
        r = primary_results[i]
        if r is None:
            L.append(f"  {split_pretty:<46}  – skipped")
            continue
        pa = p_adj.get(i)
        mark = "*" if (pa is not None and pa < T["p_sig"]) else \
               ("." if r["p"] < T["p_sig"] else "")
        lp = " ⚠" if min(r["n1"], r["n2"]) < MIN_GROUP_N else ""
        L.append(f"  {split_pretty:<46} {r['mdn1']:>6.2f} {r['mdn2']:>6.2f}  "
                 f"{r['delta_mdn']:>+6.2f}  {r['p']:>6.3f} {_fmt_padj(pa)}  "
                 f"{_r_label(r['r']):<10} {mark}{lp}")
        L.append(f"    n: {r['g1_lbl']}={r['n1']}, {r['g2_lbl']}={r['n2']}")
    L.append("")

    # Complementary (can_effort) – same splits, uncorrected
    L.append("  COMPLEMENTARY: can_effort × same splits (Likert, uncorrected)")
    L.append(f"  {'Split':<46} {'Mdn_A':>6} {'Mdn_B':>6}  {'ΔMdn':>6}  "
             f"{'p_raw':>6}  effect")
    L.append("  " + sep("─", 92))
    for split_name, split_pretty in PRIMARY_SPLITS:
        r = mw_test(scores["can_effort"], ctx_masks[split_name], "")
        if r is None:
            L.append(f"  {split_pretty:<46}  – skipped"); continue
        mark = "*" if r["p"] < T["p_sig"] else ""
        L.append(f"  {split_pretty:<46} {r['mdn1']:>6.2f} {r['mdn2']:>6.2f}  "
                 f"{r['delta_mdn']:>+6.2f}  {r['p']:>6.3f}  "
                 f"{_r_label(r['r']):<10} {mark}")
    L.append("")

    # Sensitivity: legacy can_score (10 − barrier count) on the 3 primary splits,
    # shown so the reader can see the re-operationalization did not manufacture
    # the verdict. Reported uncorrected; the verdict rests on can_constraint.
    L.append("  SENSITIVITY: can_score  (legacy 10 − barrier count, same 3 splits, uncorrected)")
    L.append(f"  {'Split':<46} {'Mdn_A':>6} {'Mdn_B':>6}  {'ΔMdn':>6}  {'p_raw':>6}  effect")
    L.append("  " + sep("─", 92))
    for split_name, split_pretty in PRIMARY_SPLITS:
        r = mw_test(scores["can_score"], ctx_masks[split_name], "")
        if r is None:
            L.append(f"  {split_pretty:<46}  – skipped"); continue
        mark = "*" if r["p"] < T["p_sig"] else ""
        L.append(f"  {split_pretty:<46} {r['mdn1']:>6.2f} {r['mdn2']:>6.2f}  "
                 f"{r['delta_mdn']:>+6.2f}  {r['p']:>6.3f}  "
                 f"{_r_label(r['r']):<10} {mark}")
    L.append("")

    # Per-domain breakdown: can_constraint for each domain vs. rest (exploratory, uncorrected)
    L.append("  DOMAIN BREAKDOWN: can_constraint by individual domain (exploratory, uncorrected)")
    L.append(f"  All {len(DOMAIN_COLS)} domains tested regardless of group size (⚠ = underpowered, not skipped).")
    L.append("  Sorted by prevalence, descending. domain_breadth is the primary split above.")
    L.append("")
    L.append(f"  {'Split':<46} {'Mdn_A':>6} {'Mdn_B':>6}  {'ΔMdn':>6}  {'p_raw':>6}  effect")
    L.append("  " + sep("─", 92))
    domain_order_h2 = sorted(DOMAIN_COLS.keys(),
                              key=lambda d: ctx_masks[f"domain_{d}"][d].sum(), reverse=True)
    for domain_name in domain_order_h2:
        split_key = f"domain_{domain_name}"
        r = mw_test(scores["can_constraint"], ctx_masks[split_key], "")
        split_lbl = f"Non-{domain_name} vs {domain_name}"
        if r is None:
            L.append(f"  {split_lbl:<46}  – skipped"); continue
        lp = " ⚠" if min(r["n1"], r["n2"]) < MIN_GROUP_N else ""
        mark = "*" if r["p"] < T["p_sig"] else ""
        L.append(f"  {split_lbl:<46} {r['mdn1']:>6.2f} {r['mdn2']:>6.2f}  "
                 f"{r['delta_mdn']:>+6.2f}  {r['p']:>6.3f}  "
                 f"{_r_label(r['r']):<10} {mark}{lp}")
        L.append(f"    n: {r['g1_lbl']}={r['n1']}, {r['g2_lbl']}={r['n2']}")
    L.append("")

    # ── MATURITY ORDINAL (Spearman ρ) ────────────────────────────────────────
    n_ord = int(scores["maturity_ordinal"].notna().sum())
    L.append("  MATURITY ORDINAL – Spearman ρ  (richer use of 4-level scale)")
    L.append(f"  Ordinal 0–3 (None→Ad hoc→KPIs→SDLC-integrated), n={n_ord}  (excl. 'I do not know').")
    L.append("  Complement to binary maturity split; reported as descriptive comparison.")
    L.append(f"  {'Pair':<44} {'n':>3}  {'ρ':>6}   {'p':>7}  direction")
    L.append("  " + sep("─", 72))
    for score_key, score_lbl in [("can_constraint", "Can(constraint)"), ("can_effort", "Can(effort)")]:
        r = spearman_test(scores["maturity_ordinal"], scores[score_key],
                          "maturity_ordinal", score_key, positive_prediction=True)
        if r["rho"] is None:
            L.append(f"  {'maturity_ord × '+score_lbl:<44} {r['n']:>3}   n/a")
            continue
        dirn  = "pos +" if r["rho"] > 0 else "neg −"
        stars = sig_stars(r["p"]).strip()
        L.append(f"  {'maturity_ord × '+score_lbl:<44} {r['n']:>3}   {r['rho']:>+6.3f}  "
                 f"{r['p']:>6.3f} {stars:<4} {dirn}")
    L.append("")

    # ── DOMAIN BREADTH ORDINAL (Spearman ρ) – robustness complement ───────────
    # Tests whether the binary Narrow/Broad effect holds under the full 0–5
    # gradient. Reported as a robustness check, not as the primary (the binary
    # split is the pre-specified primary and is marginally stronger).
    n_db = int(scores["domain_count"].notna().sum())
    L.append("  DOMAIN BREADTH ORDINAL – Spearman ρ  (robustness over full 0–5 gradient)")
    L.append(f"  Ordinal domain count 0–5 (number of domains per respondent), n={n_db}.")
    L.append("  Complement to binary Narrow/Broad split; primary remains the binary split.")
    L.append(f"  {'Pair':<44} {'n':>3}  {'ρ':>6}   {'p':>7}  direction")
    L.append("  " + sep("─", 72))
    for score_key, score_lbl in [("can_constraint", "Can(constraint)"), ("can_effort", "Can(effort)")]:
        r = spearman_test(scores["domain_count"], scores[score_key],
                          "domain_count", score_key, positive_prediction=False)
        if r["rho"] is None:
            L.append(f"  {'domain_count × '+score_lbl:<44} {r['n']:>3}   n/a")
            continue
        dirn  = "pos +" if r["rho"] > 0 else "neg −"
        stars = sig_stars(r["p"]).strip()
        L.append(f"  {'domain_count × '+score_lbl:<44} {r['n']:>3}   {r['rho']:>+6.3f}  "
                 f"{r['p']:>6.3f} {stars:<4} {dirn}")
    L.append("")

    n_sig = sum(1 for i, _ in valid_p if p_adj[i] < T["p_sig"])
    n_dir = sum(1 for i, _ in valid_p
                if p_adj[i] >= T["p_sig"]
                and primary_results[i] is not None
                and abs(primary_results[i]["delta_mdn"]) >= T["trend_delta"])
    if n_sig >= 1:
        verdict = f"SUPPORTED ✓  ({n_sig}/{len(valid_p)} primary tests survive Holm)"
    elif n_dir >= 1:
        verdict = f"DIRECTIONAL  ({n_dir}/{len(valid_p)} show |ΔMdn|≥{T['trend_delta']}, n.s.)"
    else:
        verdict = "NOT SUPPORTED"
    L.append(f"  → H2 VERDICT:  {verdict}")
    L.append("")
    return verdict, primary_results, p_adj


# =============================================================================
# H3 – CONTEXT → DO
# Mirrors the H2 Mann-Whitney design, with do_composite as the dependent
# variable. Primary framing is coercive pressure (size / CSRD threshold).
# =============================================================================

def run_ctx_do(scores, ctx_masks, L):
    PRIMARY_SPLITS = [
        ("size_250",        "SME (1–249) vs Enterprise (250+)"),
        ("maturity_strict", "Low vs High maturity (strict)"),
        ("domain_breadth",  "Narrow (≤1) vs Broad (≥2) domain footprint"),
    ]

    L += [sep("─"),
          "  H3  Context → Do  –  Mann-Whitney U",
          "  Dependent score:  do_composite (0–6)",
          "",
          "  Prediction: context shapes enacted practice directly. Coercive pressure",
          "              (size / CSRD threshold) primary; maturity and breadth secondary.",
          ""]

    primary_results = []
    for split_name, _ in PRIMARY_SPLITS:
        r = mw_test(scores["do_composite"], ctx_masks[split_name],
                    f"{split_name}×do_composite")
        primary_results.append(r)

    valid_p = [(i, r["p"]) for i, r in enumerate(primary_results) if r is not None]
    adj = holm_bonferroni([p for _, p in valid_p])
    p_adj = {i: a for (i, _), a in zip(valid_p, adj)}

    L.append("  PRIMARY: do_composite  (Holm-Bonferroni across the 3 splits)")
    L.append(f"  {'Split':<46} {'Mdn_A':>6} {'Mdn_B':>6}  {'ΔMdn':>6}  "
             f"{'p_raw':>6} {'p_adj':>6}  effect")
    L.append("  " + sep("─", 92))
    for i, (split_name, split_pretty) in enumerate(PRIMARY_SPLITS):
        r = primary_results[i]
        if r is None:
            L.append(f"  {split_pretty:<46}  – skipped")
            continue
        pa = p_adj.get(i)
        mark = "*" if (pa is not None and pa < T["p_sig"]) else \
               ("." if r["p"] < T["p_sig"] else "")
        lp = " ⚠" if min(r["n1"], r["n2"]) < MIN_GROUP_N else ""
        L.append(f"  {split_pretty:<46} {r['mdn1']:>6.2f} {r['mdn2']:>6.2f}  "
                 f"{r['delta_mdn']:>+6.2f}  {r['p']:>6.3f} {_fmt_padj(pa)}  "
                 f"{_r_label(r['r']):<10} {mark}{lp}")
        L.append(f"    n: {r['g1_lbl']}={r['n1']}, {r['g2_lbl']}={r['n2']}")
    L.append("")

    n_sig = sum(1 for i, _ in valid_p if p_adj[i] < T["p_sig"])
    n_dir = sum(1 for i, _ in valid_p
                if p_adj[i] >= T["p_sig"]
                and primary_results[i] is not None
                and abs(primary_results[i]["delta_mdn"]) >= T["trend_delta"])
    if n_sig >= 1:
        verdict = f"SUPPORTED ✓  ({n_sig}/{len(valid_p)} primary tests survive Holm)"
    elif n_dir >= 1:
        verdict = f"DIRECTIONAL  ({n_dir}/{len(valid_p)} show |ΔMdn|≥{T['trend_delta']}, n.s.)"
    else:
        verdict = "NOT SUPPORTED"
    L.append(f"  → H3 VERDICT:  {verdict}")
    L.append("")
    return verdict, primary_results, p_adj


# =============================================================================
# H4 – WANT → DO
# =============================================================================

def run_h3(scores, L):
    want_keys = [("want_env","Want(env)"), ("want_ops","Want(ops)"), ("want_all","Want(all)")]
    do_keys   = [("do_breadth","Do(breadth)"),
                 ("do_institutionalized","Do(inst)"),
                 ("do_composite","Do(comp)")]
    primary_key = ("want_env", "do_institutionalized")   # primary confirmatory pair

    L += [sep("─"),
          "  H4  Want → Do  –  Spearman ρ",
          "",
          "  Prediction: stronger sustainability objectives → more extensive or more",
          "              embedded practice (positive ρ).",
          f"  Primary pair (for verdict):  {primary_key[0]} × {primary_key[1]}",
          "             environmental motivation most directly predicts formalised",
          "             practice (decisions + green requirements embedded in process).",
          ""]
    L.append(f"  {'Pair':<40} {'n':>3}  {'ρ':>6}   {'p':>7}  direction   verdict")
    L.append("  " + sep("─", 82))

    results = []
    for wk, wl in want_keys:
        for dk, dl in do_keys:
            r = spearman_test(scores[wk], scores[dk], wk, dk, positive_prediction=True)
            results.append(r)
            if r["rho"] is None:
                L.append(f"  {wl+' × '+dl:<40} {r['n']:>3}   n/a"); continue
            correct = r["rho"] > 0
            dirn    = "pos +" if r["rho"] > 0 else "neg −"
            stars   = sig_stars(r["p"]).strip()
            if r["p"] < T["p_sig"] and correct:   vrd = "SUPPORTED ✓"
            elif r["p"] < T["p_sig"]:             vrd = "CONTRADICTED"
            elif correct:                         vrd = "directional n.s."
            else:                                 vrd = "wrong direction"
            primary_mark = "  ← primary" if (wk, dk) == primary_key else ""
            L.append(f"  {wl+' × '+dl:<40} {r['n']:>3}   {r['rho']:>+6.3f}  "
                     f"{r['p']:>6.3f} {stars:<4} {dirn:<7}  {vrd}{primary_mark}")

    primary_result = next(r for r in results
                          if r["lx"] == primary_key[0] and r["ly"] == primary_key[1])
    n_sig_total   = sum(1 for r in results
                        if r["p"] is not None and r["p"] < T["p_sig"] and r["rho"] > 0)
    n_directional = sum(1 for r in results
                        if r["rho"] is not None and r["rho"] > 0)

    L.append("")
    if (primary_result["p"] is not None and primary_result["p"] < T["p_sig"]
            and primary_result["rho"] > 0):
        verdict = (f"SUPPORTED ✓  primary ρ={primary_result['rho']:+.3f} "
                   f"({fmt_p(primary_result['p'])}); "
                   f"{n_sig_total}/9 pairs significant, {n_directional}/9 in predicted direction")
    elif n_directional >= 7:
        verdict = (f"DIRECTIONAL  {n_directional}/9 pairs in predicted direction; "
                   f"primary pair n.s. ({fmt_p(primary_result['p'])})")
    else:
        verdict = "NOT SUPPORTED"
    L.append(f"  → H4 VERDICT:  {verdict}")
    L.append("")
    return verdict, results, primary_key


# =============================================================================
# H5 – CAN → DO (primary: can_effort; sensitivity: can_score)
# =============================================================================

def run_h4(scores, L):
    no_barrier_mask  = scores["barrier_count"] == 0
    n_no_barrier     = int(no_barrier_mask.sum())
    n_with_barrier   = int((~no_barrier_mask & scores["barrier_count"].notna()).sum())
    mdn_no_barrier   = scores.loc[no_barrier_mask,  "do_breadth"].median()
    mdn_with_barrier = scores.loc[~no_barrier_mask, "do_breadth"].median()

    L += [sep("─"),
          "  H5  Can → Do  –  Spearman ρ",
          "",
          "  Prediction: higher perceived capability → more extensive or embedded",
          "              practice (positive ρ). Primary score can_effort asks directly",
          "              about worthwhileness; the barrier-count sensitivity below is",
          "              confounded by engagement (note follows that block).",
          ""]

    res_primary        = spearman_test(
        scores["can_effort"], scores["do_composite"],
        "can_effort", "do_composite", positive_prediction=True)
    res_effort_breadth = spearman_test(
        scores["can_effort"], scores["do_breadth"], "can_effort", "do_breadth", True)
    res_effort_inst    = spearman_test(
        scores["can_effort"], scores["do_institutionalized"],
        "can_effort", "do_institutionalized", True)
    res_sens           = spearman_test(
        scores["can_score"], scores["do_composite"],
        "can_score", "do_composite", positive_prediction=True)

    L.append("  PRIMARY: can_effort × Do  ('effort worthwhile' Likert)")
    L.append(f"  {'Pair':<40} {'n':>3}  {'ρ':>6}   {'p':>7}  direction   verdict")
    L.append("  " + sep("─", 82))
    for r, pretty in [
        (res_primary,        "can_effort × do_composite"),
        (res_effort_breadth, "can_effort × do_breadth"),
        (res_effort_inst,    "can_effort × do_inst"),
    ]:
        correct = (r["rho"] is not None and r["rho"] > 0)
        dirn = ("pos +" if (r["rho"] is not None and r["rho"] > 0)
                else ("neg −" if r["rho"] is not None else "—"))
        stars = sig_stars(r["p"]).strip()
        if r["p"] is None:                         vrd = "insufficient data"
        elif r["p"] < T["p_sig"] and correct:      vrd = "SUPPORTED ✓"
        elif r["p"] < T["p_sig"]:                  vrd = "CONTRADICTED"
        elif correct:                              vrd = "directional n.s."
        else:                                      vrd = "wrong direction"
        primary_mark = "  ← primary" if pretty == "can_effort × do_composite" else ""
        rho_s = f"{r['rho']:>+6.3f}" if r["rho"] is not None else "   n/a"
        L.append(f"  {pretty:<40} {r['n']:>3}   {rho_s}  "
                 f"{fmt_p(r['p']):>7} {stars:<4} {dirn:<7}  {vrd}{primary_mark}")

    L.append("")
    L.append("  SENSITIVITY: can_score × do_composite  (inverted barrier count)")
    rho_s = f"{res_sens['rho']:>+6.3f}" if res_sens["rho"] is not None else "   n/a"
    L.append(f"  can_score × do_composite                 n={res_sens['n']}   "
             f"{rho_s}  {fmt_p(res_sens['p'])}")
    L.append("")
    L.append("  ─ Construct-validity note on the sensitivity result ─────────────")
    L.append("  Barrier count is confounded by engagement: respondents who never")
    L.append("  attempted to measure cannot report barriers. In this sample, the")
    L.append(f"  {n_no_barrier} respondent(s) endorsing 'no barriers' show higher "
             f"do_breadth (mdn={mdn_no_barrier:.0f})")
    L.append(f"  than the {n_with_barrier} reporting barriers (mdn={mdn_with_barrier:.0f})"
             " – the opposite of the")
    L.append("  theoretical prediction. The primary can_effort measure avoids")
    L.append("  this confound by asking directly about perceived worthwhileness.")
    L.append("")

    if (res_primary["p"] is not None and res_primary["p"] < T["p_sig"]
            and res_primary["rho"] > 0):
        verdict = (f"SUPPORTED ✓  primary ρ={res_primary['rho']:+.3f} "
                   f"({fmt_p(res_primary['p'])})")
    elif res_primary["rho"] is not None and res_primary["rho"] > 0:
        verdict = (f"DIRECTIONAL  primary ρ={res_primary['rho']:+.3f} "
                   f"({fmt_p(res_primary['p'])}), correct direction but n.s.")
    else:
        verdict = "NOT SUPPORTED"
    L.append(f"  → H5 VERDICT:  {verdict}")
    L.append("")
    return verdict, [res_primary, res_sens]


# =============================================================================
# H8 – DO → OUTLOOK
# =============================================================================

def run_h5(scores, L):
    L += [sep("─"),
          "  H8  Do → Outlook  –  Spearman ρ",
          "",
          "  Prediction: more extensive and embedded practice → more proactive outlook",
          "              (positive ρ). × intent: drives adoption plans? × awareness:",
          "              broadens recognition of future impact?",
          ""]

    pairs = [
        ("outlook_intent",    "Outlook(intent)"),
        ("outlook_awareness", "Outlook(awareness)"),
    ]
    results = []
    L.append(f"  {'Pair':<40} {'n':>3}  {'ρ':>6}   {'p':>7}  direction   verdict")
    L.append("  " + sep("─", 82))
    for ok, ol in pairs:
        r = spearman_test(scores["do_composite"], scores[ok],
                          "do_composite", ok, positive_prediction=True)
        results.append(r)
        if r["rho"] is None:
            L.append(f"  {'Do(comp) × '+ol:<40} {r['n']:>3}   n/a"); continue
        correct = r["rho"] > 0
        dirn    = "pos +" if r["rho"] > 0 else "neg −"
        stars   = sig_stars(r["p"]).strip()
        if r["p"] < T["p_sig"] and correct:   vrd = "SUPPORTED ✓"
        elif r["p"] < T["p_sig"]:             vrd = "CONTRADICTED"
        elif correct:                         vrd = "directional n.s."
        else:                                 vrd = "wrong direction"
        L.append(f"  {'Do(comp) × '+ol:<40} {r['n']:>3}   {r['rho']:>+6.3f}  "
                 f"{r['p']:>6.3f} {stars:<4} {dirn:<7}  {vrd}")

    n_sig = sum(1 for r in results
                if r["p"] is not None and r["p"] < T["p_sig"] and r["rho"] > 0)
    if n_sig == len(results):
        verdict = f"SUPPORTED ✓  ({n_sig}/{len(results)} significant, predicted direction)"
    elif n_sig >= 1:
        verdict = f"PARTIALLY SUPPORTED  ({n_sig}/{len(results)} significant)"
    else:
        verdict = "NOT SUPPORTED"
    L.append("")
    L.append(f"  → H8 VERDICT:  {verdict}")
    L.append("")
    return verdict, results


# =============================================================================
# ROBUSTNESS – non-cumulative Do breadth
# =============================================================================

def run_breadth_robustness(scores, df, L):
    """Re-derive Do breadth with a NON-cumulative ladder (highest tier reached,
    regardless of whether lower tiers are present) and re-test the practice
    associations (H3/H4/H5). Makes the 'cumulative scoring is not an artefact'
    claim re-runnable rather than asserted."""
    metric_cols = SURVEY_DIMENSIONS["do_metrics"]["items"]
    has_proxy = pd.Series(False, index=df.index)
    has_phys  = pd.Series(False, index=df.index)
    has_env   = pd.Series(False, index=df.index)
    for name, col in metric_cols.items():
        t = (df.iloc[:, col] == "Yes")
        if name in DO_TIER_PROXY:         has_proxy |= t
        if name in DO_TIER_PHYSICAL:      has_phys  |= t
        if name in DO_TIER_ENVIRONMENTAL: has_env   |= t
    nc = pd.Series(0, index=df.index)
    nc = nc.mask(has_proxy, 1).mask(has_phys, 2).mask(has_env, 3)   # highest tier present wins
    do_comp_nc = (nc + scores["do_institutionalized"]).astype(float)

    assoc = [
        ("H4  want_env x Do(comp)",       scores["want_env"],     scores["do_composite"],       scores["want_env"],   do_comp_nc),
        ("H5  can_effort x Do(comp)",     scores["can_effort"],   scores["do_composite"],       scores["can_effort"], do_comp_nc),
        ("H8  Do(comp) x outlook_intent", scores["do_composite"], scores["outlook_intent"],     do_comp_nc,           scores["outlook_intent"]),
        ("H8  Do(comp) x outlook_aware",  scores["do_composite"], scores["outlook_awareness"],  do_comp_nc,           scores["outlook_awareness"]),
    ]
    L += [sep("─"),
          "  ROBUSTNESS – non-cumulative Do breadth (highest tier reached)",
          "  Practice associations re-tested with breadth = highest tier present",
          "  (not requiring all lower tiers), confirming the cumulative tier ladder",
          "  is not an artefact.",
          "",
          f"  {'Association':<32} {'rho cum':>8} {'rho n-c':>8} {'p n-c':>8}  verdict",
          "  " + sep("─", 72)]
    all_hold = True
    for lbl, xc, yc, xn, yn in assoc:
        rc = spearman_test(xc, yc, "", "")
        rn = spearman_test(xn, yn, "", "")
        holds = (rn["rho"] is not None and rc["rho"] is not None
                 and (rn["rho"] > 0) == (rc["rho"] > 0) and rn["p"] < T["p_sig"])
        all_hold = all_hold and holds
        L.append(f"  {lbl:<32} {rc['rho']:>+8.3f} {rn['rho']:>+8.3f} {rn['p']:>8.3f}  "
                 f"{'holds' if holds else 'CHANGES ⚠'}")
    L.append("")
    L.append("  -> Non-cumulative breadth: " +
             ("all practice associations remain positive and significant."
              if all_hold else "SOME associations change — review before reporting."))
    L.append("")


# =============================================================================
# ROLE DESCRIPTIVES
# =============================================================================

def run_role_descriptives(scores, df, L):
    """Descriptive comparison of can_effort and do scores by respondent role."""
    L += [sep("─"),
          "  ROLE DESCRIPTIVES  (exploratory, descriptive only)",
          "  Respondents may hold multiple roles; groups are not mutually exclusive.",
          ""]

    score_specs = [
        ("can_effort",  "Can(effort)"),
        ("do_breadth",  "Do(breadth)"),
        ("do_composite","Do(comp)"),
    ]

    L.append(f"  {'Role':<22}  {'n':>4}  {'Can(effort)':>18}  {'Do(breadth)':>18}  {'Do(comp)':>18}")
    L.append(f"  {'':22}  {'':4}  {'Mdn   Mean    SD':>18}  {'Mdn   Mean    SD':>18}  {'Mdn   Mean    SD':>18}")
    L.append("  " + sep("─", 90))

    role_masks = {}
    for role_name, col_idx in ROLE_COLS.items():
        mask = df.iloc[:, col_idx] == "Yes"
        role_masks[role_name] = mask
        n_role = int(mask.sum())
        if n_role == 0:
            continue
        ce = scores.loc[mask, "can_effort"].dropna()
        db = scores.loc[mask, "do_breadth"].dropna()
        dc = scores.loc[mask, "do_composite"].dropna()
        L.append(f"  {role_name:<22}  {n_role:>4}  "
                 f"{ce.median():>4.1f} {ce.mean():>6.2f} {ce.std(ddof=1):>6.2f}  "
                 f"{db.median():>4.1f} {db.mean():>6.2f} {db.std(ddof=1):>6.2f}  "
                 f"{dc.median():>4.1f} {dc.mean():>6.2f} {dc.std(ddof=1):>6.2f}")
    L.append("")

    # Technical vs Management (Mann-Whitney, descriptive, uncorrected)
    if "Technical" in role_masks and "Management" in role_masks:
        L.append("  Technical vs Management – Mann-Whitney (descriptive, uncorrected)")
        L.append(f"  {'Score':<22}  {'Mdn_Tech':>8} {'Mdn_Mgmt':>8}  {'ΔMdn':>6}  {'p':>7}  effect")
        L.append("  " + sep("─", 70))
        for score_key, score_lbl in score_specs:
            r = mw_test(scores[score_key],
                        {"Technical":  role_masks["Technical"],
                         "Management": role_masks["Management"]}, "")
            if r is None:
                L.append(f"  {score_lbl:<22}  – skipped (group too small)")
                continue
            mark = " *" if r["p"] < T["p_sig"] else "  "
            L.append(f"  {score_lbl:<22}  {r['mdn1']:>8.2f} {r['mdn2']:>8.2f}  "
                     f"{r['delta_mdn']:>+6.2f}  {fmt_p(r['p']):>7}  {_r_label(r['r'])}{mark}")
        L.append(f"    n: Technical={int(role_masks['Technical'].sum())}, "
                 f"Management={int(role_masks['Management'].sum())}")
    L.append("")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyse_hypothesis_tests(df, report_lines):
    scores    = build_dimension_scores(df)
    ctx_masks = build_context_masks(df)
    n = len(df)

    L = [sep("="),
         "  GREEN SOFTWARE METRICS SURVEY – HYPOTHESIS TESTS (H1 – H8)",
         f"  n={n}  |  Significance threshold: p<{T['p_sig']}",
         "  Primary pairs defined in config.py; secondary/sensitivity analyses uncorrected.",
         sep("=")]

    # ── Dimension scoring documentation ──────────────────────────────────────
    # Definitions only. The rationale behind each scoring decision belongs in
    # the paper (Section "Operationalization"), not repeated in every report.
    L += ["", "  DIMENSION SCORING",
          "  " + sep("─", 70),
          "  CONTEXT  (grouping variable – not scored)",
          "    size_250         SME (1–249)        vs Enterprise (250+)   [EU SME definition]",
          "    domain_web       Non-Web            vs Web",
          "    domain_breadth   Narrow (≤1)        vs Broad (≥2) domains",
          "    maturity_strict  Low (none/ad hoc)  vs High (KPIs / SDLC)  ['do not know' excluded]",
          "    size_1000        SME (1–999)        vs Enterprise (1000+)  [sensitivity only]",
          "",
          "  WANT  (1–5)",
          "    want_env   mean(CO2_reduction, Energy_reduction)",
          "    want_ops   mean(Cost_reduction, Performance, Compliance)",
          "    want_all   mean of all five objectives",
          "",
          "  CAN",
          "    can_constraint  [0–2]  barrier severity: 0 structural, 1 addressable, 2 clear",
          "                           (primary for H2; skipped item = NaN, not a maximum)",
          "    can_effort      [1–5]  'optimization is worth the required effort' (col 52)",
          "                           (primary for H5)",
          "    can_score       [0–10] 10 − n_barriers_checked  (legacy; H2 sensitivity only)",
          "",
          "  DO",
          "    do_breadth           [0–3]  cumulative tier ladder: tier N requires ≥1 metric",
          "                                from ALL tiers 1..N. 1 proxy/resource,",
          "                                2 + physical energy, 3 + environmental",
          "    do_institutionalized [0–3]  decision frequency (0/1/2) + green_req (0/1)",
          "    do_composite         [0–6]  do_breadth + do_institutionalized",
          "",
          "  OUTLOOK",
          "    outlook_intent    [1–5]  likelihood of adopting structured guidelines",
          "    outlook_awareness [0–9]  count of decision areas recognised as impacted",
          ""]

    # ── Score descriptives ────────────────────────────────────────────────────
    L += ["  SCORE DESCRIPTIVES",
          "  " + sep("─", 74),
          f"  {'Score':<38} {'n':>4} {'Mdn':>6} {'Mean':>6} {'SD':>6} {'Min':>5} {'Max':>5}",
          "  " + sep("─", 74)]
    descriptives_order = [
        ("maturity_ordinal",      "Maturity (ordinal 0–3, excl. DK)"),
        ("want_env",              "Want (environmental obj.)"),
        ("want_ops",              "Want (operational obj.)"),
        ("want_all",              "Want (all objectives)"),
        ("can_constraint",        "Can (constraint 0–2)    *primary H2"),
        ("can_effort",            "Can (effort-worthwhile) *primary H5"),
        ("can_score",             "Can (barrier-inverted)   sensitivity"),
        ("do_breadth",            "Do (metric tier breadth)"),
        ("do_institutionalized",  "Do (institutionalization)"),
        ("do_composite",          "Do (composite 0–6)"),
        ("outlook_intent",        "Outlook (intent 1–5)"),
        ("outlook_awareness",     "Outlook (awareness 0–9)"),
    ]
    for key, lbl in descriptives_order:
        if key not in scores.columns: continue
        s = scores[key].dropna()
        if len(s) == 0: continue
        L.append(f"  {lbl:<38} {len(s):>4} {s.median():>6.2f} {s.mean():>6.2f} "
                 f"{s.std(ddof=1):>6.2f} {s.min():>5.1f} {s.max():>5.1f}")
    L.append("")

    # ── Context group sizes (diagnostic) ─────────────────────────────────────
    L += ["  CONTEXT GROUP SIZES",
          "  " + sep("─", 70)]
    for split_name, pair in ctx_masks.items():
        parts = ", ".join(f"{k}={int(v.sum())}" for k, v in pair.items())
        L.append(f"  {split_name:<18}  {parts}")
    L.append("")

    # ── Run all hypothesis blocks (H6/H7 are assessed in pls_sem.py) ─────────
    h1_verdict, h1_res, h1_padj  = run_h1(scores, ctx_masks, L)
    h2_verdict, h2_res, h2_padj  = run_h2(scores, ctx_masks, L)
    h3_verdict, h3_res, h3_padj  = run_ctx_do(scores, ctx_masks, L)
    h4_verdict, h4_res, h4_key   = run_h3(scores, L)         # paper H4: Want→Do
    h5_verdict, h5_res           = run_h4(scores, L)         # paper H5: Can→Do
    h8_verdict, h8_res           = run_h5(scores, L)         # paper H8: Do→Outlook
    run_breadth_robustness(scores, df, L)
    run_role_descriptives(scores, df, L)

    # ── Summary table ────────────────────────────────────────────────────────
    L += [sep("="), "  HYPOTHESIS SUMMARY  (primary model)", sep("="), ""]

    L.append(f"  H1   Context → Want       {h1_verdict}")
    best_h1 = min((i for i in h1_padj if h1_res[i] is not None),
                  key=lambda i: h1_padj[i], default=None)
    if best_h1 is not None:
        r = h1_res[best_h1]
        L.append(f"       Best:  {r['label']}  ΔMdn={r['delta_mdn']:+.2f}  "
                 f"{fmt_p(r['p'])}  p_adj={h1_padj[best_h1]:.3f}  {_r_label(r['r'])}")
    L.append("")

    L.append(f"  H2   Context → Can        {h2_verdict}")
    best_h2 = min((i for i in h2_padj if h2_res[i] is not None),
                  key=lambda i: h2_padj[i], default=None)
    if best_h2 is not None:
        r = h2_res[best_h2]
        L.append(f"       Best:  {r['label']}  ΔMdn={r['delta_mdn']:+.2f}  "
                 f"{fmt_p(r['p'])}  p_adj={h2_padj[best_h2]:.3f}  {_r_label(r['r'])}")
    L.append("")

    L.append(f"  H3   Context → Do         {h3_verdict}")
    best_h3 = min((i for i in h3_padj if h3_res[i] is not None),
                  key=lambda i: h3_padj[i], default=None)
    if best_h3 is not None:
        r = h3_res[best_h3]
        L.append(f"       Best:  {r['label']}  ΔMdn={r['delta_mdn']:+.2f}  "
                 f"{fmt_p(r['p'])}  p_adj={h3_padj[best_h3]:.3f}  {_r_label(r['r'])}")
    L.append("")

    L.append(f"  H4   Want → Do            {h4_verdict}")
    r_primary_h4 = next(r for r in h4_res if r["lx"] == h4_key[0] and r["ly"] == h4_key[1])
    L.append(f"       Primary pair: {h4_key[0]} × {h4_key[1]}  "
             f"ρ={r_primary_h4['rho']:+.3f}  {fmt_p(r_primary_h4['p'])}")
    L.append("")

    L.append(f"  H5   Can → Do             {h5_verdict}")
    r_primary_h5 = h5_res[0]
    L.append(f"       Primary:     can_effort × do_composite  "
             f"ρ={r_primary_h5['rho']:+.3f}  {fmt_p(r_primary_h5['p'])}")
    r_sens_h5 = h5_res[1]
    L.append(f"       Sensitivity: can_score  × do_composite  "
             f"ρ={r_sens_h5['rho']:+.3f}  {fmt_p(r_sens_h5['p'])}  "
             "(engagement-confound caveat)")
    L.append("")

    L.append("  H6   Want → Outlook       direct effect, verdict from PLS-SEM (run pls_bootstrap.py)")
    L.append("  H7   Can  → Outlook       direct effect, verdict from PLS-SEM (run pls_bootstrap.py)")
    L.append("")

    L.append(f"  H8   Do → Outlook         {h8_verdict}")
    for r in h8_res:
        L.append(f"       do_composite × {r['ly']:<22} "
                 f"ρ={r['rho']:+.3f}  {fmt_p(r['p'])}  "
                 f"{sig_stars(r['p']).strip()}")
    L.append("")

    # ── Methodology notes ────────────────────────────────────────────────────
    k_h1 = len(h1_res)
    k_h2 = len(h2_res)
    k_h3 = len(h3_res)
    L += [sep("─"),
          "  METHODOLOGY NOTES",
          "  " + sep("─", 70),
          f"  Holm-Bonferroni   H1 k={k_h1} theory-driven pairs | H2 k={k_h2} splits (can_constraint)",
          f"                    | H3 k={k_h3} splits (do_composite). Exploratory matrices uncorrected.",
          "  Primary pairs     H4 want_env × do_institutionalized | H5 can_effort × do_composite",
          "  Not tested here   H6/H7 (Want/Can → Outlook) — joint PLS-SEM verdict",
          "  Exploratory only  role descriptives (groups non-exclusive, no hypothesis)",
          f"  Power             Spearman at n={n} detects |ρ|≥0.40 at ≈80%; Mann-Whitney with",
          f"                    two groups of ~{n // 2} detects r≥0.40. Groups under n={MIN_GROUP_N} flagged ⚠",
          "                    as underpowered — p reported, no verdict carried.",
          ""]

    out(L, report_lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    csv_path = config.csv_arg(sys.argv, DEFAULT_CSV)
    print(f"\nLoading: {csv_path}")
    df = load_survey(csv_path)
    print(f"n={len(df)}  |  {df.shape[1]} columns\n")

    report_lines = [
        "GREEN SOFTWARE METRICS SURVEY – HYPOTHESIS TEST REPORT",
        f"File: {config.display_path(csv_path)}  |  n={len(df)}",
        sep("="), "",
    ]

    analyse_hypothesis_tests(df, report_lines)

    report_path = config.report_path("hypothesis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  -> Report saved: {report_path}")
    print()


if __name__ == "__main__":
    main()