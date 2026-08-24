#!/usr/bin/env python3
"""
=============================================================================
Green Software Metrics Survey - LaTeX macro generator
=============================================================================
Computes every data-derived number that appears in the paper and writes them
as LaTeX macros to  ../chapters/generated/macros.tex .  The paper \\input's that
file in the preamble, so all figures, table cells, and in-text statistics are
pulled from the current survey export -- no hand-transcribed numbers.

Single source of truth: this script reuses the scoring/test functions from
hypothesis_test.py and the reliability functions from descriptive_analysis.py.
Change a definition there and the paper updates the next time this generator is
run.

Scope: what the paper prints, and nothing beyond it.  A value the manuscript
does not use is not computed here.  The ones the robustness report needs are
added by experiments/extra_macros.py, which extends this registry; the ones
nothing reads are commented out in place, beside the code that produced them,
so re-enabling one is a single uncomment.  Two exceptions are deliberate.  A
loop over a specification list emits its whole family even where the paper
cites some cells and not others, because the spec list is the analysis and
pruning it to the printed cells would misstate what was run.  A list whose
entries each cost a bootstrap is pruned to the cited entries, with the dropped
ones named in a comment, because there the cost is the reason.  Either way the
run reports what it defined and the paper does not cite, so a value leaving the
prose stays visible.

Usage:
    python gen_latex.py [path/to/export.csv]      # regenerate macros
    python gen_latex.py --list                    # print name/value table only

Run this manually after every new data cut, then recompile the PDF and review
the number changes.  (It is deliberately NOT wired into the LaTeX build so that
figures cannot move silently between data cuts.)
=============================================================================
"""
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd


def _rhu(x, nd):
    """Round half up to nd decimals (matches the paper's rounding convention)."""
    q = Decimal(1).scaleb(-nd) if nd > 0 else Decimal(1)
    return Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP)

import config
import hypothesis_test as H
import descriptive_analysis as D
from config import (
    SURVEY_DIMENSIONS, ROLE_COLS, DOMAIN_COLS,
    CONTEXT_SIZE_GROUPS, CONTEXT_MATURITY_LOW, CONTEXT_MATURITY_HIGH,
    CONTEXT_MATURITY_UNKNOWN, EFFORT_MAP, OUTLOOK_MAP,
)

OUT = os.path.join(config.GENERATED_TEX_DIR, "macros.tex")

# =============================================================================
# FORMATTING HELPERS  (match the paper's existing rendering exactly)
# =============================================================================

def f_star(p, alpha=0.05):
    """Significance marker as a macro, so tables never hard-code one.

    A hand-placed asterisk silently goes stale when the data change: the value
    next to it regenerates, the marker does not.  Emitting it alongside the
    value keeps the two in step.
    """
    return "^{*}" if (p is not None and p < alpha) else ""


def f_star_ci(lo, hi):
    """Marker for interval-based decisions (significant iff the CI excludes 0)."""
    return "^{*}" if (lo > 0 or hi < 0) else ""


def f_pct(x):                       # 41
    return f"{int(_rhu(x, 0)):d}"

def f_int(x):                       # 28
    return f"{int(round(x)):d}"

def f_p(p):                         # .018  /  <.001
    if p < 0.001:
        return "<.001"
    return str(_rhu(p, 3)).lstrip("0")

def f_p4(p):                        # .0047  /  <.0001
    """Raw p at 4 dp.

    Used where the paper prints a raw p next to its Holm-adjusted counterpart:
    at 3 dp the multiplication is not reproducible by the reader (.002 x 2
    does not give .005), which invites a spurious arithmetic complaint.
    """
    if p < 0.0001:
        return "<.0001"
    return str(_rhu(p, 4)).lstrip("0")

def f_rho(r):                       # +0.290  (signed, 3dp, keep leading zero)
    return f"{'+' if r >= 0 else '-'}{_rhu(abs(r), 3):.3f}"

def f_d2(x):                        # +0.50   (signed, 2dp, keep leading zero)
    return f"{'+' if x >= 0 else '-'}{_rhu(abs(x), 2):.2f}"

def f_m2(x):                        # 4.00   (unsigned, 2dp)
    return f"{_rhu(x, 2):.2f}"

def f_mdn_int(x):                   # 1
    return f"{int(round(x)):d}"

def f_beta(x):                      # +0.36   (signed, 2dp, keep leading zero)
    return f"{'+' if x >= 0 else '-'}{_rhu(abs(x), 2):.2f}"

def f_ci(lo, hi):                   # [+0.13, +0.60]
    return f"[{f_beta(lo)}, {f_beta(hi)}]"

def f_a2(x):                        # 0.90  (2dp, no sign)
    return f"{_rhu(x, 2):.2f}"

def f_a3(x):                        # 0.638 (3dp, no sign)
    return f"{_rhu(x, 3):.3f}"

def f_prop(x):                      # .39  (proportion, 2dp, no sign, strip leading zero)
    return f"{_rhu(abs(x), 2):.2f}".lstrip("0")


# =============================================================================
# COLUMN-LEVEL DESCRIPTIVE HELPERS
# =============================================================================

def pct_yes(df, col):
    return (df.iloc[:, col] == "Yes").mean() * 100

def n_yes(df, col):
    return int((df.iloc[:, col] == "Yes").sum())


# =============================================================================
# MACRO REGISTRY
# =============================================================================

class Macros:
    def __init__(self):
        self._m = {}      # name -> formatted string
        self._order = []

    def add(self, name, value):
        if name in self._m:
            raise ValueError(f"duplicate macro name: {name}")
        # TeX control sequences are letters only.  A digit silently terminates
        # the name, so \gsmHThree... works but \gsmH3... parses as \gsmH
        # followed by literal text -- an "Undefined control sequence" at
        # compile time rather than an error here.  Catch it at generation.
        if not name.isalpha():
            raise ValueError(
                f"invalid macro name {name!r}: TeX control sequences must be "
                f"letters only (no digits or punctuation)")
        self._m[name] = value
        self._order.append(name)

    def get(self, name):
        """Formatted value of one macro, without the leading backslash.

        experiments/gen_robustness_report.py renders its tables from these
        values rather than recomputing them, so every number the report shares
        with the paper is by construction the number the PDF prints. Raises
        rather than returning a default: a silently missing value would look
        like a real result.
        """
        try:
            return self._m[name]
        except KeyError:
            raise KeyError(
                f"no such macro: {name!r}. Macro names are defined in "
                f"gen_latex.build(); run 'python gen_latex.py --list' to see "
                f"the {len(self._order)} available.") from None

    def names(self):
        """Macro names in definition order."""
        return list(self._order)

    def write(self, path):
        """Write macros.tex.

        Every macro build() computes is emitted, unconditionally.  The file is
        a pure function of the data and this script, so its diff shows a data
        cut or a generator change and nothing else.  Which macros the paper
        currently references is a property of the prose, not of the data, and
        is reported to stdout by main() instead of encoded here -- otherwise
        editing a sentence rewrites a generated file, and the diff stops
        telling you whether the numbers moved.
        """
        lines = [
            "% =============================================================",
            "% AUTO-GENERATED by scripts/gen_latex.py -- DO NOT EDIT BY HAND.",
            "% Regenerate after each data cut:  python gen_latex.py",
            "% Every value build() computes is defined here whether or not the",
            "% paper currently cites it; gen_latex.py reports unused ones.",
            "% =============================================================",
            "",
        ]
        for name in self._order:
            lines.append(f"\\newcommand{{\\{name}}}{{{self._m[name]}}}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def print_table(self):
        w = max(len(n) for n in self._order)
        for name in self._order:
            print(f"  \\{name:<{w}}  =  {self._m[name]}")
        print(f"\n  {len(self._order)} macros")


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def build(df):
    """Every value the paper prints, as a Macros registry.

    Sections below follow the paper's order.  Nothing here reads a committed
    cache: the resampled and simulated quantities that need one are all
    robustness-report values, and they live in experiments/extra_macros.py.
    """
    M = Macros()
    scores = H.build_dimension_scores(df)
    masks  = H.build_context_masks(df)
    n = len(df)

    # ---- Sample sizes -------------------------------------------------------
    n_want = int(scores["want_env"].notna().sum())
    # want_ops / want_all are available-item means: NaN only when every item of
    # the sub-scale is missing, so their n differs from the environmental n.
    n_want_ops = int(scores["want_ops"].notna().sum())
    mat_col = df.iloc[:, SURVEY_DIMENSIONS["context_maturity"]["items"]["Sustainability_maturity"]].fillna("None")
    n_unknown = int(mat_col.isin(CONTEXT_MATURITY_UNKNOWN).sum())
    # listwise-complete on the structural model's indicators (authoritative)
    n_listwise = len(H.model_frame(df))
    M.add("gsmSampleFull", f_int(n))
    M.add("gsmSampleWantSubscale", f_int(n_want))
    M.add("gsmSampleWantNonResponse", f_int(n - n_want))
    M.add("gsmSampleWantOpsScale", f_int(n_want_ops))
    M.add("gsmSampleListwiseComplete", f_int(n_listwise))
    M.add("gsmSampleMaturityUnknown", f_int(n_unknown))

    # ---- Context: size ------------------------------------------------------
    size = df.iloc[:, SURVEY_DIMENSIONS["context_size"]["items"]["Org_size"]]
    sme = size.isin(CONTEXT_SIZE_GROUPS["SME (1–249)"])
    ent = size.isin(CONTEXT_SIZE_GROUPS["Enterprise (250+)"])
    M.add("gsmPctSme", f_pct(sme.mean() * 100));          M.add("gsmNSme", f_int(sme.sum()))
    M.add("gsmPctEnterprise", f_pct(ent.mean() * 100));   M.add("gsmNEnterprise", f_int(ent.sum()))
    # Not cited: the paper reports the SME/enterprise split, not the top band.
    # big = (size == "5000+")
    # M.add("gsmPctSizeFiveThousandPlus", f_pct(big.mean() * 100))
    # M.add("gsmNSizeFiveThousandPlus", f_int(big.sum()))

    # ---- Context: roles -----------------------------------------------------
    mgmt = df.iloc[:, ROLE_COLS["Management"]] == "Yes"
    tech = df.iloc[:, ROLE_COLS["Technical"]] == "Yes"
    # nont = df.iloc[:, ROLE_COLS["Non-Technical"]] == "Yes"
    # M.add("gsmPctManagement", f_pct(mgmt.mean() * 100));     M.add("gsmNManagement", f_int(mgmt.sum()))
    M.add("gsmNTechnical", f_int(tech.sum()))
    # M.add("gsmPctTechnical", f_pct(tech.mean() * 100))
    # M.add("gsmPctNonTechnical", f_pct(nont.mean() * 100));   M.add("gsmNNonTechnical", f_int(nont.sum()))
    both = (mgmt & tech)
    M.add("gsmPctManagementAndTechnical", f_pct(both.mean() * 100))
    # M.add("gsmNManagementAndTechnical", f_int(both.sum()))

    # ---- Context: domains ---------------------------------------------------
    # Not cited: the domain figure (fig_domain) carries these shares, and the
    # domain-count distribution is not printed at all.
    # M.add("gsmPctDomainBusiness", f_pct(pct_yes(df, DOMAIN_COLS["Business_Enterprise"])))
    # M.add("gsmPctDomainPlatform", f_pct(pct_yes(df, DOMAIN_COLS["Platform_Infra"])))
    # M.add("gsmPctDomainWeb", f_pct(pct_yes(df, DOMAIN_COLS["Web"])))
    # M.add("gsmPctDomainMobile", f_pct(pct_yes(df, DOMAIN_COLS["Mobile"])))
    # M.add("gsmPctDomainIoTEmbedded", f_pct(pct_yes(df, DOMAIN_COLS["IoT_Embedded"])))
    # dom_count = pd.concat([(df.iloc[:, c] == "Yes").astype(int) for c in DOMAIN_COLS.values()], axis=1).sum(axis=1)
    # M.add("gsmPctDomainCountOne", f_pct((dom_count == 1).mean() * 100))
    # M.add("gsmPctDomainCountTwo", f_pct((dom_count == 2).mean() * 100))

    # ---- Context: systems ---------------------------------------------------
    # Workload mix, runtime environments, operating model and architectural
    # style are asked only of Technical-role respondents (survey skip logic),
    # so every share here is over that subsample.  A full-sample denominator
    # would silently count never-asked respondents as "No".
    dfc = df[df.iloc[:, ROLE_COLS["Technical"]] == "Yes"]
    for _key, _tag in [("Transactional", "Transactional"), ("Batch_Pipeline", "BatchPipeline"),
                       ("Analytics", "Analytics"), ("Event_Realtime", "EventRealtime"),
                       ("ML_AI", "MlAi"), ("HPC", "Hpc"), ("Embedded_Edge", "EmbeddedEdge")]:
        M.add(f"gsmPctWorkload{_tag}", f_pct(pct_yes(dfc, config.WORKLOAD_COLS[_key])))
    for _key, _tag in [("Containers", "Containers"), ("VMs", "Vms"),
                       ("Serverless", "Serverless"), ("Bare_metal", "BareMetal")]:
        M.add(f"gsmPctRuntime{_tag}", f_pct(pct_yes(dfc, config.RUNTIME_COLS[_key])))
    for _key, _tag in [("Self_operated", "SelfOperated"), ("Fully_Managed", "FullyManaged"),
                       ("VM_External", "VmExternal"), ("PaaS_External", "PaasExternal"),
                       ("Hosted_Colocated", "HostedColocated")]:
        M.add(f"gsmPctInfra{_tag}", f_pct(pct_yes(dfc, config.INFRASTRUCTURE_COLS[_key])))
    # Architectural style is single-answer and left blank by some of the
    # technical respondents, so it carries its own denominator.
    arch = df.iloc[:, config.ARCH_STYLE_COL].dropna()
    M.add("gsmNArchAnswered", f_int(len(arch)))
    for _label, _tag in [("Microservices", "Microservices"),
                         ("Modular monolith", "ModularMonolith"),
                         ("Monolithic", "Monolithic")]:
        M.add(f"gsmPctArch{_tag}", f_pct((arch == _label).mean() * 100))
    # Decision authority is asked of everyone.
    auth = df.iloc[:, config.AUTHORITY_COL]
    for _label, _tag in [("Company / Organization", "Company"),
                         ("Group / Division / Unit", "Division"),
                         ("Department", "Department"), ("Team", "Team"),
                         ("Individual / Employee", "Individual"),
                         ("I do not know", "Unknown")]:
        M.add(f"gsmPctAuthority{_tag}", f_pct((auth == _label).mean() * 100))

    # ---- Context: maturity --------------------------------------------------
    low = mat_col.isin(CONTEXT_MATURITY_LOW); high = mat_col.isin(CONTEXT_MATURITY_HIGH)
    unk = mat_col.isin(CONTEXT_MATURITY_UNKNOWN)
    M.add("gsmPctMaturityLow", f_pct(low.mean() * 100));    M.add("gsmNMaturityLow", f_int(low.sum()))
    M.add("gsmPctMaturityHigh", f_pct(high.mean() * 100));  M.add("gsmNMaturityHigh", f_int(high.sum()))
    M.add("gsmPctMaturityUnknown", f_pct(unk.mean() * 100))
    # M.add("gsmNMaturityUnknownPct", f_int(unk.sum()))   # gsmSampleMaturityUnknown is the cited count

    # ---- Want ---------------------------------------------------------------
    M.add("gsmMdnWantOperational", f_m2(scores["want_ops"].median()))
    M.add("gsmSdWantOperational", f_m2(scores["want_ops"].std(ddof=1)))
    M.add("gsmMdnWantEnvironmental", f_m2(scores["want_env"].median()))
    M.add("gsmSdWantEnvironmental", f_m2(scores["want_env"].std(ddof=1)))
    infl = df.iloc[:, SURVEY_DIMENSIONS["want_influence"]["items"]["Energy_influences"]].astype(str).str.startswith("Yes")
    M.add("gsmPctEnergyInfluence", f_pct(infl.mean() * 100))

    # ---- Can ----------------------------------------------------------------
    # Primary capability measure: can_constraint (3-level severity ordinal).
    cc = scores["can_constraint"]
    M.add("gsmPctCanStructural",     f_pct((cc == 0).mean() * 100))
    # Not cited: only the structural share is printed as a percentage; the rest
    # of the partition is given as the counts below.
    # M.add("gsmPctCanAddressable",    f_pct((cc == 1).mean() * 100))
    # M.add("gsmPctCanClear",          f_pct((cc == 2).mean() * 100))
    # M.add("gsmPctCanNonInformative", f_pct(cc.isna().mean() * 100))
    M.add("gsmNCanInformative",      f_int(int(cc.notna().sum())))
    M.add("gsmNCanNonInformative",   f_int(int(cc.isna().sum())))
    # Counts alongside the percentages: the four shares are of the full sample
    # and round to 99, so the raw counts are what makes the partition legible.
    M.add("gsmNCanStructural",       f_int(int((cc == 0).sum())))
    M.add("gsmNCanAddressable",      f_int(int((cc == 1).sum())))
    M.add("gsmNCanClear",            f_int(int((cc == 2).sum())))
    # ---- Free-text barrier coding ------------------------------------------
    # How far the nine closed categories cover what respondents volunteer.
    _bcol = SURVEY_DIMENSIONS["can_barriers"]["items"]["No_barriers"] + 1
    _ft = df.iloc[:, _bcol].dropna().astype(str).str.strip()
    _ft = _ft[_ft.ne("")]
    _coded, _uncoded = [], []
    for entry in _ft:
        hit = config.FREETEXT_BARRIER_CODING.get(entry.lower())
        (_coded if hit else _uncoded).append(hit or entry)
    if _uncoded:
        # A new data cut may add entries; fail rather than under-count.
        raise ValueError("uncoded free-text barrier entries (add them to "
                         "config.FREETEXT_BARRIER_CODING): " + "; ".join(_uncoded))
    _real = [c for c in _coded if c[0] != "Not a barrier"]
    _novel = [c for c in _real if c[1] is None]
    M.add("gsmNFreetextBarriers", f_int(len(_ft)))
    M.add("gsmPctRespondentsFreetext", f_pct(len(_ft) / n * 100))
    # The novel share is a fraction of the entries that name a barrier at all,
    # which is fewer than the entries written in: the denominator must be stated
    # in the text or the percentage will not reconcile with the entry count.
    M.add("gsmNFreetextCoded", f_int(len(_real)))
    M.add("gsmNFreetextNovel", f_int(len(_novel)))
    M.add("gsmPctFreetextNovel", f_pct(len(_novel) / len(_real) * 100))
    # Per-category tallies of the coded write-ins.  Retired from the paper: the
    # novel-versus-restating split is what bounds the closed list, and the
    # categories themselves read better named in words than counted one by one.
    # for cat, tag in [("Leadership priority", "Leadership"), ("Motivation", "Motivation"),
    #                  ("Business case", "BusinessCase"), ("Provider data", "ProviderData"),
    #                  ("Tooling", "Tooling")]:
    #     M.add(f"gsmNFreetext{tag}", f_int(sum(1 for c in _real if c[0] == cat)))

    # Write-ins on the enabler and role items. Reported qualitatively rather
    # than coded: at this count a category scheme would over-formalize them.
    _en = df.iloc[:, 70].dropna().astype(str).str.strip(); _en = _en[_en.ne("")]
    M.add("gsmNFreetextEnablers", f_int(len(_en)))
    _rl = df.iloc[:, 33].dropna().astype(str).str.strip(); _rl = _rl[_rl.ne("")]
    _none = _rl[_rl.str.lower().str.contains("none|no one|aren't applicable|not in scope")]
    M.add("gsmNFreetextRoles", f_int(len(_rl)))
    M.add("gsmNFreetextRolesNone", f_int(len(_none)))

    # Legacy barrier-count score, retained only for the H2 sensitivity line.
    M.add("gsmMeanBarriers", f"{scores['barrier_count'].mean():.1f}")
    # Not cited: the barrier figure (fig_barriers) carries the per-category
    # shares, and the maximum of the legacy barrier-count score is not printed.
    # M.add("gsmCanScoreMax", f_int(len([k for k in SURVEY_DIMENSIONS["can_barriers"]["items"]
    #                                    if k != "No_barriers"]) + 1))   # 9 listed + Other
    bcols = SURVEY_DIMENSIONS["can_barriers"]["items"]
    # M.add("gsmPctBarrierGroundTruth", f_pct(pct_yes(df, bcols["Lack_ground_truth"])))
    # M.add("gsmPctBarrierUnclearResp", f_pct(pct_yes(df, bcols["Unclear_resp"])))
    # M.add("gsmPctBarrierRestrictedAccess", f_pct(pct_yes(df, bcols["Restricted_access"])))
    # M.add("gsmPctBarrierSkillsGaps", f_pct(pct_yes(df, bcols["Skills_gaps"])))
    # M.add("gsmPctBarrierLackMethods", f_pct(pct_yes(df, bcols["Lack_methods"])))
    # M.add("gsmPctBarrierCostInstrument", f_pct(pct_yes(df, bcols["Cost_instrument"])))

    # Do the leading technical and the leading organizational barrier travel
    # together?  The conclusion claims they do not, which is a claim about the
    # data and needs a number rather than an impression.  Phi is the correlation
    # between two yes/no items; the bootstrap interval says how much a null this
    # size can actually exclude.
    _gt = (df.iloc[:, bcols["Lack_ground_truth"]] == "Yes").astype(int).values
    _ur = (df.iloc[:, bcols["Unclear_resp"]] == "Yes").astype(int).values
    M.add("gsmNBarrierGroundTruth", f_int(_gt.sum()))
    M.add("gsmNBarrierUnclearResp", f_int(_ur.sum()))
    M.add("gsmNBarrierPairBoth", f_int(int(((_gt == 1) & (_ur == 1)).sum())))
    from scipy import stats as _st
    _phi, _phi_p = _st.pearsonr(_gt, _ur)
    _rng = np.random.default_rng(7)
    _boot = []
    for _ in range(10000):
        _i = _rng.integers(0, len(_gt), len(_gt))
        _s = np.corrcoef(_gt[_i], _ur[_i])[0, 1]
        if np.isfinite(_s):
            _boot.append(_s)
    M.add("gsmBarrierPairPhi", f_rho(_phi))
    M.add("gsmBarrierPairPhiCi", f_ci(np.percentile(_boot, 2.5), np.percentile(_boot, 97.5)))
    M.add("gsmBarrierPairP", f_p(_phi_p))

    M.add("gsmMdnCanEffort", f_m2(scores["can_effort"].median()))
    eff_agree = df.iloc[:, SURVEY_DIMENSIONS["want_effort"]["items"]["Effort_worthwhile"]].map(EFFORT_MAP) >= 4
    M.add("gsmPctEffortAgree", f_pct(eff_agree.mean() * 100))
    # Not cited: the enabler figure (fig_enablers) carries these shares.
    # ecols = SURVEY_DIMENSIONS["can_enablers"]["items"]
    # M.add("gsmPctEnablerStdMetrics", f_pct(pct_yes(df, ecols["Std_metrics"])))
    # M.add("gsmPctEnablerProcedures", f_pct(pct_yes(df, ecols["Procedures"])))
    # M.add("gsmPctEnablerRefDashboards", f_pct(pct_yes(df, ecols["Ref_dashboards"])))
    # M.add("gsmPctEnablerCiCdPatterns", f_pct(pct_yes(df, ecols["CI_CD_patterns"])))
    reg = df.iloc[:, SURVEY_DIMENSIONS["can_political"]["items"]["Regulation_needed"]] == "Yes"
    M.add("gsmPctRegulationNeeded", f_pct(reg.mean() * 100))

    # ---- Do -----------------------------------------------------------------
    # Not cited: the coverage figure (fig_coverage) carries the tier
    # distribution and the metrics figure (fig_metrics) the per-metric shares.
    # for k in range(4):
    #     M.add(f"gsmPctDoTier{['Zero','One','Two','Three'][k]}",
    #           f_pct((scores["do_breadth"] == k).mean() * 100))
    #     M.add(f"gsmNDoTier{['Zero','One','Two','Three'][k]}",
    #           f_int(int((scores["do_breadth"] == k).sum())))
    M.add("gsmMdnDoBreadth", f_mdn_int(scores["do_breadth"].median()))
    mcols = SURVEY_DIMENSIONS["do_metrics"]["items"]
    # M.add("gsmPctMetricCpu", f_pct(pct_yes(df, mcols["CPU_util"])))
    # M.add("gsmPctMetricMemory", f_pct(pct_yes(df, mcols["Memory_util"])))
    # M.add("gsmPctMetricNetwork", f_pct(pct_yes(df, mcols["Network_util"])))
    # M.add("gsmPctMetricEnergyKwh", f_pct(pct_yes(df, mcols["Energy_kWh"])))
    # M.add("gsmPctMetricPowerW", f_pct(pct_yes(df, mcols["Power_W"])))
    # M.add("gsmPctMetricCoTwoE", f_pct(pct_yes(df, mcols["CO2e"])))
    # M.add("gsmPctMetricSci", f_pct(pct_yes(df, mcols["SCI"])))
    # M.add("gsmPctMetricWater", f_pct(pct_yes(df, mcols["Water"])))
    n_co2 = n_yes(df, mcols["CO2e"])
    co2_mask = df.iloc[:, mcols["CO2e"]] == "Yes"
    scope_col = SURVEY_DIMENSIONS["do_co2_scope"]["items"]["CO2_scope"]
    scope_yes = df.iloc[:, scope_col][co2_mask].astype(str).str.startswith("Yes")
    M.add("gsmNCoTwoETracked", f_int(n_co2))
    M.add("gsmPctCoTwoEScopeDifferentiated", f_pct(scope_yes.mean() * 100))
    # The four scope answers partition the CO2e trackers, so the counts are what
    # the text reports: at this subsample size percentages would mislead.
    scope_vals = df.iloc[:, scope_col][co2_mask].astype(str)
    M.add("gsmNCoTwoEScopeFull", f_int(int(scope_yes.sum())))
    M.add("gsmNCoTwoEScopePartial", f_int(int(scope_vals.str.startswith("Partially").sum())))
    M.add("gsmNCoTwoEScopeNone", f_int(int(scope_vals.str.startswith("No,").sum())))
    M.add("gsmNCoTwoEScopeUnsure", f_int(int(scope_vals.str.startswith("Not sure").sum())))
    # Lifecycle / observation-scope / automation items are shown only to
    # Technical-role respondents (survey skip logic), so their percentages are
    # computed over that subsample -- full-sample denominators would silently
    # count never-asked respondents as "No".
    dft = df[df.iloc[:, ROLE_COLS["Technical"]] == "Yes"]
    lc = SURVEY_DIMENSIONS["do_lifecycle"]["items"]
    M.add("gsmPctLifecycleProduction", f_pct(pct_yes(dft, lc["Production"])))
    M.add("gsmPctLifecycleDevelopment", f_pct(pct_yes(dft, lc["Development"])))
    M.add("gsmPctLifecycleTestingQa", f_pct(pct_yes(dft, lc["Testing_QA"])))
    oc = SURVEY_DIMENSIONS["do_observation_scope"]["items"]
    M.add("gsmPctObsInfraPhysical", f_pct(pct_yes(dft, oc["Infra_physical"])))
    M.add("gsmPctObsInfraVm", f_pct(pct_yes(dft, oc["Infra_VM"])))
    M.add("gsmPctObsInfraContainer", f_pct(pct_yes(dft, oc["Infra_container"])))
    M.add("gsmPctObsSwService", f_pct(pct_yes(dft, oc["SW_service"])))
    M.add("gsmPctObsSwTransaction", f_pct(pct_yes(dft, oc["SW_transaction"])))
    M.add("gsmPctObsSwCode", f_pct(pct_yes(dft, oc["SW_code"])))
    mr = SURVEY_DIMENSIONS["do_measurement_runtime"]["items"]
    for _key, _tag in [("Meas_bare_metal", "BareMetal"), ("Meas_VMs", "Vms"),
                       ("Meas_containers", "Containers"), ("Meas_serverless", "Serverless")]:
        M.add(f"gsmPctMeasRuntime{_tag}", f_pct(pct_yes(dft, mr[_key])))
    # Who is involved is asked of everyone, unlike the items above it.
    for _key, _tag in [("Operations_IT", "OperationsIt"), ("Sustainability_ESG", "SustainabilityEsg"),
                       ("Developers", "Developers"), ("Architects", "Architects"),
                       ("Product_Management", "ProductManagement"), ("DevOps_SRE", "DevOpsSre"),
                       ("External_Stakeholders", "ExternalStakeholders")]:
        M.add(f"gsmPctGreenRole{_tag}", f_pct(pct_yes(df, config.GREEN_METRICS_ROLE_COLS[_key])))
    auto = df.iloc[:, SURVEY_DIMENSIONS["do_automation"]["items"]["Automated"]]
    M.add("gsmPctAutomated", f_pct((auto == "Yes").sum() / auto.notna().sum() * 100))
    M.add("gsmMdnDoInstitutionalized", f_mdn_int(scores["do_institutionalized"].median()))
    freq = df.iloc[:, SURVEY_DIMENSIONS["do_decisions"]["items"]["Decision_frequency"]]
    M.add("gsmPctDecisionsRegularly", f_pct((freq == "Regularly").mean() * 100))
    greq = df.iloc[:, SURVEY_DIMENSIONS["do_requirements"]["items"]["Green_requirements"]] == "Yes"
    M.add("gsmPctGreenRequirements", f_pct(greq.mean() * 100))
    M.add("gsmMdnDoComposite", f_mdn_int(scores["do_composite"].median()))
    M.add("gsmPctCompositeAtMostTwo", f_pct((scores["do_composite"] <= 2).mean() * 100))

    # ---- Outlook ------------------------------------------------------------
    M.add("gsmMdnOutlookIntent", f_mdn_int(scores["outlook_intent"].median()))
    M.add("gsmPctIntentUnlikely", f_pct((scores["outlook_intent"] <= 2).mean() * 100))
    M.add("gsmPctIntentLikely", f_pct((scores["outlook_intent"] >= 4).mean() * 100))
    M.add("gsmMdnOutlookAwareness", f_mdn_int(scores["outlook_awareness"].median()))
    # Not cited: the decision-area figure (fig_areas) carries these shares.
    # ic = SURVEY_DIMENSIONS["outlook_impact"]["items"]
    # M.add("gsmPctAreaSustainabilityReporting", f_pct(pct_yes(df, ic["Sustainability_reporting"])))
    # M.add("gsmPctAreaHardware", f_pct(pct_yes(df, ic["Hardware"])))
    # M.add("gsmPctAreaArchitecture", f_pct(pct_yes(df, ic["Architecture"])))
    # M.add("gsmPctAreaRequirements", f_pct(pct_yes(df, ic["Requirements"])))

    # ---- Reliability --------------------------------------------------------
    want_data = pd.concat(
        [pd.to_numeric(df.iloc[:, c], errors="coerce") for c in SURVEY_DIMENSIONS["want_objectives"]["items"].values()],
        axis=1).dropna()
    # Not cited: the paper reports reliability per sub-scale, not for the five
    # items pooled, because the two sub-scales are not one construct.
    # a_all, lo_all, hi_all = D.cronbach_alpha_ci(want_data)
    # M.add("gsmSampleWantFiveItem", f_int(len(want_data)))   # listwise across all 5 Want items
    # M.add("gsmAlphaWantAllItems", f_a3(a_all))
    # M.add("gsmAlphaWantAllItemsCiLow", f_a3(lo_all))
    # M.add("gsmAlphaWantAllItemsCiHigh", f_a3(hi_all))
    # rho on the same listwise want sample used for alpha (matches descriptive_analysis)
    rho_env = want_data.iloc[:, [0, 1]].corr(method="spearman").iloc[0, 1]
    M.add("gsmRhoCoTwoEnergy", f_a2(rho_env))
    # M.add("gsmRhoCoTwoEnergyThreeDp", f_a3(rho_env))          # two decimals are what is printed
    # M.add("gsmAlphaWantEnvTwoDp", f_a2(D.spearman_brown(rho_env)))
    # gsmAlphaWantEnvSpearmanBrown -> experiments/extra_macros.py (robustness report)
    ops_items = SURVEY_DIMENSIONS["want_objectives"]["clusters"]["Operational"]
    oc2 = [SURVEY_DIMENSIONS["want_objectives"]["items"][x] for x in ops_items]
    ops_data = pd.concat([pd.to_numeric(df.iloc[:, c], errors="coerce") for c in oc2], axis=1).dropna()
    a_ops, _, _ = D.cronbach_alpha(ops_data)
    M.add("gsmAlphaWantOperational", f_a3(a_ops))

    # ---- Harman single-factor -----------------------------------------------
    likert = []
    for c in SURVEY_DIMENSIONS["want_objectives"]["items"].values():
        likert.append(pd.to_numeric(df.iloc[:, c], errors="coerce"))
    likert.append(df.iloc[:, SURVEY_DIMENSIONS["want_effort"]["items"]["Effort_worthwhile"]].map(EFFORT_MAP))
    likert.append(df.iloc[:, SURVEY_DIMENSIONS["outlook_likely"]["items"]["Adopt_guidelines"]].map(OUTLOOK_MAP))
    Xh = pd.concat(likert, axis=1).dropna()
    Xc = (Xh - Xh.mean()) / Xh.std(ddof=0)
    s = np.linalg.svd(Xc.values, compute_uv=False)
    M.add("gsmHarmanFirstFactorPct", f"{(s[0]**2 / (s**2).sum()) * 100:.1f}")

    # The early-vs-late responder comparison (Armstrong-Overton) is check 6 of
    # the robustness report and is printed nowhere in the paper:
    # experiments/extra_macros.py.

    # ---- H1: Mann-Whitney 3x3 ----------------------------------------------
    h1_grid = [
        ("size_250", "SizeOps", [("want_env", "Env"), ("want_ops", "Ops"), ("want_all", "All")]),
        ("domain_web", "DomainWeb", [("want_env", "Env"), ("want_ops", "Ops"), ("want_all", "All")]),
        ("maturity_strict", "Maturity", [("want_env", "Env"), ("want_ops", "Ops"), ("want_all", "All")]),
    ]
    # Confirmatory p_adj (Holm k=2).  domain_web is excluded from the family:
    # it was selected after screening all five domain categories, so an adjusted
    # p over it would not price in the selection step.  It is reported
    # uncorrected, with a conservative Bonferroni bound over the 5 screened
    # categories so the reader can see what selection would cost it.
    h1_primary = [("size_250", "want_ops"), ("maturity_strict", "want_env")]
    h1_praw = [H.mw_test(scores[o], masks[s_], "")["p"] for s_, o in h1_primary]
    h1_padj = dict(zip(h1_primary, H.holm_bonferroni(h1_praw)))
    for split, tag, subs in h1_grid:
        for okey, otag in subs:
            r = H.mw_test(scores[okey], masks[split], "")
            M.add(f"gsmCtxWantN{tag}{otag}", f_int(r["n1"] + r["n2"]))
            M.add(f"gsmCtxWantDmdn{tag}{otag}", f_d2(r["delta_mdn"]))
            if (split, okey) in h1_padj:
                M.add(f"gsmCtxWantPadj{tag}{otag}", f_p(h1_padj[(split, okey)]))
            # Not cited: the H1 readout prints the shift and its interval, never
            # the standardized effect size or an uncorrected p.
            # M.add(f"gsmCtxWantR{tag}{otag}", f_a2(r["r"]))   # standardized rank effect size |z|/sqrt(n)
            # M.add(f"gsmCtxWantPraw{tag}{otag}", f_p(r["p"]))
            # M.add(f"gsmCtxWantPrawFour{tag}{otag}", f_p4(r["p"]))
    M.add("gsmCtxWantHolmK", f_int(len(h1_primary)))
    _web_p = H.mw_test(scores["want_env"], masks["domain_web"], "")["p"]
    M.add("gsmCtxWantDomainWebSelBound", f_p(min(1.0, _web_p * len(DOMAIN_COLS))))
    M.add("gsmCtxWantNDomainScreened", f_int(len(DOMAIN_COLS)))
    # Not cited: the mobile contrast was exploratory, and the maturity-ordinal
    # correlations are a second reading of the maturity split reported above.
    # rm = H.mw_test(scores["want_all"], masks["domain_Mobile"], "")
    # M.add("gsmCtxWantMobileDmdn", f_d2(rm["delta_mdn"]))
    # M.add("gsmCtxWantMobileP", f_p(rm["p"]))
    # M.add("gsmCtxWantMobileN", f_int(rm["n2"]))
    # r_me = H.spearman_test(scores["maturity_ordinal"], scores["want_env"], "", "")
    # M.add("gsmCtxWantMaturityOrdEnvRho", f_rho(r_me["rho"]));  M.add("gsmCtxWantMaturityOrdEnvP", f_p(r_me["p"]))
    # r_ma = H.spearman_test(scores["maturity_ordinal"], scores["want_all"], "", "")
    # M.add("gsmCtxWantMaturityOrdAllRho", f_rho(r_ma["rho"]));  M.add("gsmCtxWantMaturityOrdAllP", f_p(r_ma["p"]))

    # ---- H2 -----------------------------------------------------------------
    # Primary: can_constraint (3-level severity ordinal), Holm across 3 splits.
    h2_primary = ["domain_breadth", "size_250", "maturity_strict"]
    h2_praw = [H.mw_test(scores["can_constraint"], masks[s_], "")["p"] for s_ in h2_primary]
    h2_padj = dict(zip(h2_primary, H.holm_bonferroni(h2_praw)))
    M.add("gsmCtxCanSizePadj", f_p(h2_padj["size_250"]))
    M.add("gsmCtxCanMaturityPadj", f_p(h2_padj["maturity_strict"]))
    M.add("gsmCtxCanBreadthPadj", f_p(h2_padj["domain_breadth"]))
    # Not cited: H2 is reported as not decidable on this sample, so the paper
    # prints the three adjusted p-values above and nothing beneath them.
    # rb = H.mw_test(scores["can_constraint"], masks["domain_breadth"], "")
    # M.add("gsmCtxCanBreadthDmdn", f_d2(rb["delta_mdn"]))
    # M.add("gsmCtxCanBreadthR", f_d2(rb["r"] * -1 if rb["delta_mdn"] < 0 else rb["r"]))
    # M.add("gsmCtxCanBreadthNarrowN", f_int(rb["n1"]));  M.add("gsmCtxCanBreadthBroadN", f_int(rb["n2"]))
    # r_dc = H.spearman_test(scores["domain_count"], scores["can_constraint"], "", "")
    # M.add("gsmCtxCanDomainCountRho", f_beta(r_dc["rho"]));  M.add("gsmCtxCanDomainCountP", f_p(r_dc["p"]))
    # M.add("gsmCtxCanSizeR", f_a2(H.mw_test(scores["can_constraint"], masks["size_250"], "")["r"]))
    # M.add("gsmCtxCanMaturityR", f_a2(H.mw_test(scores["can_constraint"], masks["maturity_strict"], "")["r"]))
    # Sensitivity: barrier-count score (10 − barrier count), uncorrected raw p-values.
    # h2_sens = {s_: H.mw_test(scores["can_score"], masks[s_], "")["p"] for s_ in h2_primary}
    # M.add("gsmCtxCanSensBreadthP",  f_p(h2_sens["domain_breadth"]))
    # M.add("gsmCtxCanSensSizeP",     f_p(h2_sens["size_250"]))
    # M.add("gsmCtxCanSensMaturityP", f_p(h2_sens["maturity_strict"]))
    # Complementary Can measure: effort-worthwhile x domain breadth (uncorrected).
    # rb_eff = H.mw_test(scores["can_effort"], masks["domain_breadth"], "")
    # M.add("gsmCtxCanEffortBreadthR", f_a2(rb_eff["r"]))
    # M.add("gsmCtxCanEffortBreadthP", f_p(rb_eff["p"]))

    # ---- Want x Do  (paper H4; emits gsmWantDo*) ----------------------------
    want_keys = [("want_env", "Env"), ("want_ops", "Ops"), ("want_all", "All")]
    do_keys = [("do_breadth", "Tier"), ("do_institutionalized", "Gov"), ("do_composite", "Composite")]
    r_wd_env_gov = None       # H4 primary pair, kept for the Holm family below
    for wk, wt in want_keys:
        for dk, dt in do_keys:
            r = H.spearman_test(scores[wk], scores[dk], "", "")
            M.add(f"gsmWantDo{wt}{dt}Rho", f_rho(r["rho"]))
            # Not cited: the H4 readout prints rho with its interval and the
            # Holm-adjusted p, never the uncorrected one.
            # M.add(f"gsmWantDo{wt}{dt}P", f_p(r["p"]))
            if (wk, dk) == ("want_env", "do_institutionalized"):
                r_wd_env_gov = r

    # ---- Can x Do  (paper H5; emits gsmCanDo*) ------------------------------
    r_cd_primary = None       # H5 primary pair, kept for the Holm family below
    for dk, dt in [("do_composite", "Composite"), ("do_breadth", "Tier"), ("do_institutionalized", "Gov")]:
        r = H.spearman_test(scores["can_effort"], scores[dk], "", "")
        M.add(f"gsmCanDoEffort{dt}Rho", f_rho(r["rho"]))
        # M.add(f"gsmCanDoEffort{dt}P", f_p(r["p"]))   # the Holm-adjusted p is what is printed
        if dk == "do_composite":
            r_cd_primary = r
    r_sens = H.spearman_test(scores["can_score"], scores["do_composite"], "", "")
    M.add("gsmCanDoSensitivityRho", f_rho(r_sens["rho"]));  M.add("gsmCanDoSensitivityP", f_p(r_sens["p"]))

    # ---- Valuation vs. goal-level Want: discriminant + formal comparison -----
    # Reviewer point: the paper compared the H4 and H5 correlations informally.
    # (a) Are the two predictors even distinct?  (b) Is the gap between their
    # correlations with the same outcome statistically distinguishable?
    r_disc = H.spearman_test(scores["can_effort"], scores["want_env"], "", "")
    M.add("gsmValWantDiscriminantRho", f_rho(r_disc["rho"]))
    M.add("gsmSampleValWantListwise", f_int(r_disc["n"]))
    for out_key, out_tag in [("do_composite", "Composite"), ("outlook_intent", "Intent")]:
        dd = H.dependent_rho_diff(scores["can_effort"], scores["want_env"], scores[out_key])
        M.add(f"gsmValWantDiff{out_tag}", f_rho(dd["diff"]))
        M.add(f"gsmValWantDiff{out_tag}Ci", f_ci(dd["ci_low"], dd["ci_high"]))
    # The p-values of both comparisons, and the power simulation that says how
    # much a non-significant one can bear, are report-only:
    # experiments/extra_macros.py.

    # Smallest correlation this sample can detect 80% of the time, two-tailed at
    # alpha = .05, via the Fisher z transform.  The conclusion-validity paragraph
    # quotes it to say which supported associations sit near the detection floor,
    # so it must track n rather than be typed in.
    from scipy import stats as _st2
    _zr = (_st2.norm.ppf(0.975) + _st2.norm.ppf(0.80)) / np.sqrt(len(df) - 3)
    M.add("gsmPowerRhoEighty", f_a2(np.tanh(_zr)))

    # ---- Do x Outlook  (paper H8; emits gsmDoOutlook*) ---------------------
    # Uncorrected p-values are never printed: the Holm-adjusted family below is
    # what the readout and the results figure quote.
    r_i = H.spearman_test(scores["do_composite"], scores["outlook_intent"], "", "")
    M.add("gsmDoOutlookIntentRho", f_rho(r_i["rho"]))
    # M.add("gsmDoOutlookIntentP", f_p(r_i["p"]))
    r_a = H.spearman_test(scores["do_composite"], scores["outlook_awareness"], "", "")
    M.add("gsmDoOutlookAwarenessRho", f_rho(r_a["rho"]))
    # M.add("gsmDoOutlookAwarenessP", f_p(r_a["p"]))

    # ---- Want/Can x Outlook pairwise (H6/H7 isolation view; the H6 pairwise-
    #      vs-joint contrast motivates the structural model section) ----------
    r_wo = H.spearman_test(scores["want_env"], scores["outlook_intent"], "", "")
    M.add("gsmWantOutlookIntentRho", f_rho(r_wo["rho"]))
    # M.add("gsmWantOutlookIntentP", f_p(r_wo["p"]))
    r_wa = H.spearman_test(scores["want_env"], scores["outlook_awareness"], "", "")
    M.add("gsmWantOutlookAwarenessRho", f_rho(r_wa["rho"]))
    # M.add("gsmWantOutlookAwarenessP", f_p(r_wa["p"]))
    r_co = H.spearman_test(scores["can_effort"], scores["outlook_intent"], "", "")
    M.add("gsmCanOutlookIntentRho", f_rho(r_co["rho"]))
    # M.add("gsmCanOutlookIntentP", f_p(r_co["p"]))
    r_ca = H.spearman_test(scores["can_effort"], scores["outlook_awareness"], "", "")
    M.add("gsmCanOutlookAwarenessRho", f_rho(r_ca["rho"]))
    # M.add("gsmCanOutlookAwarenessP", f_p(r_ca["p"]))

    # ---- Holm correction across the H4-H8 association family -----------------
    # The context families H1-H3 were already Holm-corrected above; the Spearman
    # family was not, which left the correction rule uneven across hypotheses.
    # Same holm_bonferroni() used for the Mann-Whitney families, applied to every
    # primary pair in H4-H8 so one rule covers all eight hypotheses.
    assoc_family = [
        ("gsmWantDoEnvGov",         r_wd_env_gov),   # H4 primary
        ("gsmCanDoEffortComposite", r_cd_primary),   # H5 primary
        ("gsmWantOutlookIntent",    r_wo),           # H6 primary pair 1
        ("gsmWantOutlookAwareness", r_wa),           # H6 primary pair 2
        ("gsmCanOutlookIntent",     r_co),           # H7 primary pair 1
        ("gsmCanOutlookAwareness",  r_ca),           # H7 primary pair 2
        ("gsmDoOutlookIntent",      r_i),            # H8 primary pair 1
        ("gsmDoOutlookAwareness",   r_a),            # H8 primary pair 2
    ]
    assoc_padj = H.holm_bonferroni([r["p"] for _, r in assoc_family])
    for (name, _), padj in zip(assoc_family, assoc_padj):
        M.add(f"{name}Padj", f_p(padj))
    # Keep the adjusted values addressable by name so the H6/H7 verdicts below
    # derive from the same numbers the tables print, rather than a second
    # computation that could drift from them on a new data cut.
    assoc_padj_by_name = {name: p for (name, _), p in zip(assoc_family, assoc_padj)}
    M.add("gsmAssocFamilyK", f_int(len(assoc_family)))
    # Not cited: the readout names the family size and prints each adjusted p,
    # so a tally of how many survived would repeat the table.
    # M.add("gsmAssocFamilyMaxPadj", f_p(max(assoc_padj)))
    # M.add("gsmAssocFamilyNSigAfter", f_int(sum(p < .05 for p in assoc_padj)))
    # M.add("gsmAssocFamilyNSigBefore", f_int(sum(r["p"] < .05 for _, r in assoc_family)))

    # The context x Do-component split, which shows the circular predictor and
    # the non-circular one explaining different components, is check 3 of the
    # robustness report: experiments/extra_macros.py.

    # Metric coverage is the pre-specified primary outcome for H3 (see
    # chapters/measures.tex), so the three coverage contrasts get their own Holm
    # family.  Previously only the composite family was corrected, which left the
    # H3 verdict resting on an uncorrected p.
    _cov_raw = [H.mw_test(scores["do_breadth"], masks[s], "")["p"]
                for s in ("maturity_strict", "domain_breadth", "size_250")]
    _cov_padj = H.holm_bonferroni(_cov_raw)
    for stag, padj in zip(("Maturity", "Breadth", "Size"), _cov_padj):
        M.add(f"gsmCtxDo{stag}CoveragePadj", f_p(padj))

    # ---- Sensitivity of the findings to how Do is scored --------------------
    # The tier ladder and the equal weighting of coverage against governance are
    # author decisions.  Each association is therefore recomputed under eight
    # scorings of Do, so a reader can see which findings survive the choice.
    # Variant list and hypothesis list live in config.py so that this generator
    # and experiments/extra_macros.py cannot fall out of step.
    do_variants = config.DO_VARIANT_SPECS
    n_rob = {tag: 0 for _k, tag, _lbl in config.DO_VARIANT_HYPOTHESES}
    for tag, _label, build_variant in do_variants:
        variant = build_variant(scores)
        for score_key, who, _lbl in config.DO_VARIANT_HYPOTHESES:
            # H8 runs Do -> Outlook; the other two run predictor -> Do.
            r = (H.spearman_test(variant, scores[score_key], "", "")
                 if who == "Outlook" else
                 H.spearman_test(scores[score_key], variant, "", ""))
            # The per-variant correlations are the robustness report's check 4
            # table: experiments/extra_macros.py.  The paper prints how many
            # variants each association survives, and that is what is kept.
            if r["p"] < .05:
                n_rob[who] += 1
    M.add("gsmSensNVariants", f_int(len(do_variants)))
    M.add("gsmSensWantSig", f_int(n_rob["Want"]))
    M.add("gsmSensOutlookSig", f_int(n_rob["Outlook"]))
    # gsmSensCanSig and the coverage-rule disagreement counts are report-only:
    # experiments/extra_macros.py.
    # disagree = (scores["do_breadth"] != scores["do_highest"]).sum()
    # M.add("gsmSensTierDisagreeN", f_int(disagree))
    # M.add("gsmSensTierDisagreePct", f_pct(disagree / len(df) * 100))
    # _dis = scores["do_breadth"] != scores["do_highest"]
    # M.add("gsmSensTierDisagreeEnv",
    #       f_int(int((_dis & (scores["do_highest"] == 3)).sum())))
    # M.add("gsmSensTierDisagreeProxy",
    #       f_int(int((_dis & (scores["do_highest"] == 2)).sum())))

    # ---- Confidence intervals for the pairwise statistics -------------------
    # A p-value states whether an association is distinguishable from zero; the
    # interval states how precisely it is pinned down, which at n<100 is the
    # more informative quantity.  One interval costs a 10,000-draw bootstrap, so
    # the list holds the pairs the paper prints an interval for and no others.
    ci_pairs = [
        ("WantDoEnvTier", scores["want_env"], scores["do_breadth"]),
        ("WantDoEnvGov", scores["want_env"], scores["do_institutionalized"]),
        ("WantDoOpsGov", scores["want_ops"], scores["do_institutionalized"]),
        ("CanDoEffortComposite", scores["can_effort"], scores["do_composite"]),
        ("WantOutlookIntent", scores["want_env"], scores["outlook_intent"]),
        ("WantOutlookAwareness", scores["want_env"], scores["outlook_awareness"]),
        ("CanOutlookIntent", scores["can_effort"], scores["outlook_intent"]),
        ("CanOutlookAwareness", scores["can_effort"], scores["outlook_awareness"]),
        ("DoOutlookIntent", scores["do_composite"], scores["outlook_intent"]),
        ("DoOutlookAwareness", scores["do_composite"], scores["outlook_awareness"]),
        # The discriminant pair carries the paper's central claim, so it needs an
        # interval like every other primary estimate: a non-significant rho on its
        # own cannot distinguish "no association" from "too small a sample".
        ("ValWantDiscriminant", scores["want_env"], scores["can_effort"]),
        # The sub-scale x component pairs below were intervalled here too.  None
        # is printed, so none is computed:
        #   WantDoEnvComposite, WantDoOpsTier, WantDoOpsComposite,
        #   WantDoAllTier, WantDoAllGov, WantDoAllComposite,
        #   CanDoEffortTier, CanDoEffortGov
    ]
    # ---- Equivalence and attenuation for the discriminant pair --------------
    # The paper's central claim is that prior work merges two constructs that
    # are not one.  A non-significant rho cannot support that, so we test it
    # the other way round: can a correlation as large as two indicators of a
    # single construct would show be rejected?  The primary bound is Hair et
    # al.'s convergent-validity rule of thumb (indicators of one construct
    # share at least half their variance), the stricter bound is Cohen's
    # moderate effect.
    for tag, delta in [("", 0.50), ("Strict", 0.30)]:
        t = H.spearman_tost(scores["want_env"], scores["can_effort"], delta)
        M.add(f"gsmTost{tag}Delta", f"{delta:.2f}")
        M.add(f"gsmTost{tag}P", f_p(t["p"]))
        # gsmTostCiNinety and gsmTostN are not printed: the equivalence verdict
        # is read off the bound and its p, and the pair's n is already given by
        # gsmSampleValWantListwise.
        # if tag == "":
        #     M.add("gsmTostCiNinety", f_ci(t["lo90"], t["hi90"]))
        #     M.add("gsmTostN", f_int(t["n"]))
    # Could unreliability alone explain the null?  Corrected for measurement
    # error at a deliberately pessimistic reliability for the single item, the
    # discriminant correlation stays negligible, so it cannot.  The same
    # correction applied to the positive finding shows that direction of bias
    # runs the other way: single-item noise understates H5 and H7.  Neither the
    # paper nor the robustness report prints any of it.
    # _rel_env = f_a3(D.spearman_brown(rho_env))
    # _rel_floor = 0.50
    # _rho_disc = H.spearman_test(scores["want_env"], scores["can_effort"], "", "")["rho"]
    # _rho_comp = H.spearman_test(scores["can_effort"], scores["do_composite"], "", "")["rho"]
    # M.add("gsmRelFloor", f"{_rel_floor:.2f}")
    # M.add("gsmDisattenDiscriminant", f_rho(H.disattenuate(_rho_disc, _rel_floor, float(_rel_env))))
    # M.add("gsmDisattenCompositeAtNinety", f_rho(H.disattenuate(_rho_comp, 0.90)))
    # M.add("gsmDisattenCompositeAtSeventy", f_rho(H.disattenuate(_rho_comp, 0.70)))
    for tag, xs, ys in ci_pairs:
        r = H.spearman_ci(xs, ys)
        M.add(f"gsm{tag}Ci", f_ci(r["lo"], r["hi"]))

    # Group contrasts: a bootstrap interval for the standardized rank effect
    # size.  Each costs a 5,000-draw bootstrap, so only the contrasts the paper
    # prints an interval for are computed.  The Hodges-Lehmann shift and its
    # distribution-free interval were computed alongside them for all six
    # contrasts and are printed nowhere.
    for split, stag, outcome, otag in [
        ("domain_web", "DomainWebEnv", scores["want_env"], None),
        ("maturity_strict", "MaturityEnv", scores["want_env"], None),
        ("domain_breadth", "BreadthCan", scores["can_constraint"], None),
        ("domain_breadth", "BreadthCoverage", scores["do_breadth"], None),
        # Not printed, so not computed:
        # ("maturity_strict", "MaturityDo", scores["do_composite"], None),
        # ("domain_breadth", "BreadthDo", scores["do_composite"], None),
    ]:
        pair = masks[split]
        # labels = list(pair)
        # hl = H.hodges_lehmann(outcome[pair[labels[1]]], outcome[pair[labels[0]]])
        # M.add(f"gsmHl{stag}", f_d2(hl["est"]))
        # M.add(f"gsmHl{stag}Ci", f_ci(hl["lo"], hl["hi"]))
        rc = H.mannwhitney_r_ci(outcome, pair)
        M.add(f"gsmRci{stag}", f_a2(rc["r"]))
        M.add(f"gsmRci{stag}Ci", f"[{f_a2(rc['lo'])}, {f_a2(rc['hi'])}]")

    # ---- Ordinal logistic regression (the paper's joint estimator) ----------
    # Every model is fitted on the SAME listwise-complete cases. Letting each
    # keep whatever cases its own predictors allow would make the rows
    # incomparable across outcomes.
    ord_common = pd.DataFrame({
        "intent": scores["outlook_intent"], "awareness": scores["outlook_awareness"],
        "composite": scores["do_composite"], "coverage": scores["do_breadth"],
        "governance": scores["do_institutionalized"], "want": scores["want_env"],
        "valuation": scores["can_effort"], "maturity": scores["maturity_ordinal"],
    }).dropna()
    # Specifications live in config.py, shared with the robustness report.
    ord_specs = config.ORD_SPECS
    label = {col: term for col, (term, _lbl) in config.ORD_TERMS.items()}
    for tag, ycol, pcols, _outcome_label in ord_specs:
        res = H.ordinal_logit(ord_common[ycol],
                              {label[c]: ord_common[c] for c in pcols})
        # Per-model n, pseudo R^2, term intervals and significance markers are
        # the robustness report's check 2 table: experiments/extra_macros.py.
        # M.add(f"gsmSample{tag}", f_int(res["n"]))
        # M.add(f"gsm{tag}PseudoRsq", f_prop(res["prsq"]))
        for term, v in res["terms"].items():
            M.add(f"gsm{tag}{term}Or", f_a2(v["or"]))
            M.add(f"gsm{tag}{term}P", f_p(v["p"]))
            # M.add(f"gsm{tag}{term}Ci", f"[{f_a2(v['lo'])}, {f_a2(v['hi'])}]")
            # M.add(f"gsm{tag}{term}Star", f_star(v["p"]))
    # The same model on every case its own predictors allow, as the contrast:
    # goal-level motivation does reach intent there, which is exactly the
    # pairwise result that the joint model dissolves (H6).
    wide = H.ordinal_logit(scores["outlook_intent"],
                           {"Want": scores["want_env"],
                            "Valuation": scores["can_effort"],
                            "Practice": scores["do_composite"]})
    M.add("gsmSampleOrdIntentWide", f_int(wide["n"]))
    M.add("gsmOrdIntentWideWantP", f_p(wide["terms"]["Want"]["p"]))
    # gsmOrdIntentWideWantOr -> experiments/extra_macros.py (robustness report)

    # ---- Role descriptive (Technical vs Management) -------------------------
    tmask = df.iloc[:, ROLE_COLS["Technical"]] == "Yes"
    mmask = df.iloc[:, ROLE_COLS["Management"]] == "Yes"
    r_tm = H.mw_test(scores["do_composite"], {"Technical": tmask, "Management": mmask}, "")
    M.add("gsmRoleTechCompositeMdn", f_mdn_int(r_tm["mdn1"]))
    M.add("gsmRoleMgmtCompositeMdn", f_mdn_int(r_tm["mdn2"]))
    M.add("gsmRoleTechVsMgmtP", f_p(r_tm["p"]))
    M.add("gsmRoleTechVsMgmtR", f_d2(r_tm["r"]))

    # The PLS-SEM path estimates, their bootstrap intervals, the R^2 and
    # reliability figures and the formative VIF are reported in the replication
    # package, not in the paper: experiments/extra_macros.py and pls_sem.py.

    # ---- Verdicts (with consistency-driving labels) -------------------------
    L_dummy = []
    h1_v, _, _ = H.run_h1(scores, masks, L_dummy[:])
    h2_v, _, _ = H.run_h2(scores, masks, L_dummy[:])
    h3_v, _, _ = H.run_h3(scores, L_dummy[:])
    h4_v, _ = H.run_h4(scores, L_dummy[:])
    h5_v, _ = H.run_h5(scores, L_dummy[:])
    def verdict_word(v):
        if v.startswith("SUPPORTED"):  return "supported"
        if v.startswith("PARTIALLY"):  return "partially supported"
        if v.startswith("DIRECTIONAL"): return "not supported"   # directional but n.s. at corrected threshold
        return "not supported"
    M.add("gsmVerdictCtxWant", verdict_word(h1_v))
    # H2 is a special case.  Constraint severity is on its floor for two thirds
    # of the informative sample, so a null result cannot distinguish "context
    # does not matter" from "this sample has too little variance to tell".  A
    # hypothesis that cannot be tested has not failed, so it gets its own word
    # rather than being counted among the rejections.
    _cc = scores["can_constraint"].dropna()
    _floor = (_cc == 0).mean() >= (2 / 3)
    M.add("gsmVerdictCtxCan",
          "not decidable" if (_floor and not verdict_word(h2_v).startswith("supported"))
          else verdict_word(h2_v))
    M.add("gsmPctCanFloorInformative", f_pct((_cc == 0).mean() * 100))
    M.add("gsmVerdictWantDo", verdict_word(h3_v))
    M.add("gsmVerdictCanDo", verdict_word(h4_v))
    M.add("gsmVerdictDoOutlook", verdict_word(h5_v))
    # ---- H3 (flow order): Context -> Do. Defensive: a failure here must not
    #      abort macro generation, so it degrades to a NOT SUPPORTED verdict
    #      and a warning.
    try:
        ctxdo_v, ctxdo_res, ctxdo_padj = H.run_ctx_do(scores, masks, L_dummy[:])
        # The composite contrasts behind the verdict are not printed: the H3
        # readout quotes the coverage family (gsmCtxDo*CoveragePadj) instead.
        # for idx, key in [(0, "Size"), (1, "Maturity"), (2, "Breadth")]:
        #     r = ctxdo_res[idx]
        #     M.add(f"gsmCtxDo{key}Dmdn", f_d2(r["delta_mdn"]))
        #     M.add(f"gsmCtxDo{key}R", f_a2(r["r"]))
        #     M.add(f"gsmCtxDo{key}Padj", f_p(ctxdo_padj[idx]))
    except Exception as e:
        sys.stderr.write(f"[gen_latex] WARNING: Context->Do (H3) test failed: {e}\n")
        ctxdo_v = "NOT SUPPORTED"
    M.add("gsmVerdictCtxDo", verdict_word(ctxdo_v))

    # ---- H6 / H7: Want -> Outlook and Can(Val) -> Outlook -------------------
    # Both are decided on the association family above, like H4, H5 and H8, so
    # their verdicts read straight off the Holm-adjusted p-values rather than
    # being asserted in the text. A hypothesis counts as supported when at
    # least one of its two Outlook pairs survives correction; the scope column
    # of the verdict table names which one.
    def _assoc_verdict(*names):
        return "supported" if any(assoc_padj_by_name[n] < config.T["p_sig"] for n in names) \
               else "not supported"
    h6_word = _assoc_verdict("gsmWantOutlookIntent", "gsmWantOutlookAwareness")
    h7_word = _assoc_verdict("gsmCanOutlookIntent", "gsmCanOutlookAwareness")
    M.add("gsmVerdictWantOutlook", h6_word)
    M.add("gsmVerdictCanOutlook", h7_word)
    # Which component carries each verdict, so the prose cannot claim a reach
    # the data do not show.
    for tag, base in [("WantOutlook", "gsmWantOutlook"), ("CanOutlook", "gsmCanOutlook"),
                      ("DoOutlook", "gsmDoOutlook")]:
        hits = [lbl for lbl, key in [("adoption intent", base + "Intent"),
                                     ("decision-area awareness", base + "Awareness")]
                if assoc_padj_by_name[key] < config.T["p_sig"]]
        M.add(f"gsmScope{tag}", " and ".join(hits) if hits else "neither component")

    # The maturity-exclusion sensitivity (legacy split, "I do not know" grouped
    # with Low) is check 7 of the robustness report: experiments/extra_macros.py.

    # Verdict counts over ALL EIGHT hypotheses, in paper numbering:
    #   H1 h1_v | H2 h2_v | H3 ctxdo_v | H4 h3_v | H5 h4_v
    #   H6 h6_word | H7 h7_word | H8 h5_v
    # (run_h3/h4/h5 are named for their dimension pair, not the paper's H-number.)
    # The paper prints these counts rather than stating them, so a data cut that
    # flips a verdict updates the prose instead of silently contradicting it.
    # H2 carries the floor-effect override set above, so the tally separates
    # "tested and failed" from "not testable on this sample".
    _h2_word = M.get("gsmVerdictCtxCan")
    all_verdicts = [verdict_word(h1_v), _h2_word] \
                   + [verdict_word(v) for v in (ctxdo_v, h3_v, h4_v)] \
                   + [h6_word, h7_word, verdict_word(h5_v)]
    n_supported = sum(v == "supported" for v in all_verdicts)
    # n_undecidable = sum(v == "not decidable" for v in all_verdicts)
    M.add("gsmNHypotheses", f_int(len(all_verdicts)))
    # Spelled-out forms, so the sentence reads naturally without hard-coding.
    # Only the supported count is printed; the sentence names the rest.
    _words = {0: "None", 1: "One", 2: "Two", 3: "Three", 4: "Four",
              5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
    M.add("gsmNHypothesesSupportedWord", _words[n_supported])
    # Lower-case total, for sentences that already spell the supported count:
    # "Seven of the 8" mixes a word and a digit inside one comparison.
    M.add("gsmNHypothesesWord", _words[len(all_verdicts)].lower())
    # M.add("gsmNHypothesesSupported", f_int(n_supported))
    # M.add("gsmNHypothesesNotSupported",
    #       f_int(len(all_verdicts) - n_supported - n_undecidable))
    # M.add("gsmNHypothesesNotDecidable", f_int(n_undecidable))
    # M.add("gsmNHypothesesNotSupportedWord",
    #       _words[len(all_verdicts) - n_supported - n_undecidable].lower())
    # M.add("gsmNHypothesesNotDecidableWord", _words[n_undecidable].lower())

    return M


def paper_macro_usage(root_tex):
    """Macro names referenced by the compiled paper.

    Follows the \\input chain from the main .tex rather than globbing
    chapters/, because the two differ: chapters/survey_charts.tex and the
    retired archive/ are on disk but not in the paper.  Deriving the set from
    the actual chain means re-enabling an appendix automatically brings its
    macros back, with no list to maintain here.

    Comment-stripping is line-based (% to end of line, respecting \\%), which
    matches how the paper is written and is enough to keep a commented-out
    \\input or citation from counting as a use.
    """
    root_tex = os.path.abspath(root_tex)
    base = os.path.dirname(root_tex)
    # The generated file is \input by the paper but must not be scanned: every
    # \newcommand{\gsmX} in it matches the usage pattern, so including it would
    # mark all macros used and defeat the check.
    generated = os.path.abspath(os.path.join(config.GENERATED_TEX_DIR, "macros.tex"))
    seen, pending, used = set(), [root_tex], set()
    while pending:
        path = pending.pop()
        if path in seen or path == generated or not os.path.isfile(path):
            continue
        seen.add(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = re.sub(r"(?<!\\)%.*", "", raw)
                used.update(re.findall(r"\\(gsm[A-Za-z]+)", line))
                for inc in re.findall(r"\\input\{([^}]+)\}", line):
                    cand = os.path.join(base, inc)
                    pending.append(cand if inc.endswith(".tex") else cand + ".tex")
    return used


def main():
    csv = config.csv_arg(sys.argv, H.DEFAULT_CSV)
    df = config.load_survey(csv)
    M = build(df)
    if "--list" in sys.argv:
        M.print_table()
        return
    os.makedirs(config.GENERATED_TEX_DIR, exist_ok=True)
    root_tex = os.path.join(config.REPO_DIR, "gsm_enterprise-survey.tex")
    used = paper_macro_usage(root_tex)
    M.write(OUT)
    unknown = sorted(used - set(M.names()))
    unused = [n for n in M.names() if n not in used]
    print(f"wrote {OUT}  ({len(M._order)} macros, n={len(df)})")
    if unused:
        # Not an error: build() is the single source of truth and the
        # replication reports render values the paper does not cite.  Printed
        # so that a value silently leaving the prose is still visible.
        print(f"[gen_latex] {len(unused)} macros not cited by the paper: "
              + ", ".join(unused[:8]) + (" ..." if len(unused) > 8 else ""))
    if unknown:
        # The paper references a macro build() does not define: a guaranteed
        # "Undefined control sequence" at compile time, so fail loudly here.
        sys.stderr.write("[gen_latex] ERROR: paper references undefined macros: "
                         + ", ".join(unknown) + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
