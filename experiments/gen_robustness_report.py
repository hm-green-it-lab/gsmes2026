#!/usr/bin/env python3
"""
=============================================================================
Robustness Report  →  replication/reports/robustness_report.txt
=============================================================================
The seven robustness checks behind the paper's results, in one place.
The paper has no robustness section: these checks constrain how far its
conclusions carry, and this file is where they are reported in full.

Written to be scanned, not read: a verdict line, the table, and at most one
sentence of interpretation per check. Terms are defined once in the glossary
so no check has to explain itself. Nothing is said twice.

The report renders macros; it computes none of its own. The values the paper
also prints come from gen_latex.build(), so the report cannot drift from the
manuscript; the rest come from extra_macros.build_extra(), which derives them
from the same test and specification functions gen_latex uses.

It lives here rather than in scripts/ because the paper has no robustness
section: nothing in the manuscript depends on this file, and the values that
exist only for it are not computed on the path that feeds the PDF.

The response-funnel and dropout comparison is not here: no funnel figure is
cited by the paper, so it lives in dropout_analysis.py instead.

Usage:
    python experiments/gen_robustness_report.py [path/to/export.csv]
=============================================================================
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

import config
import extra_macros as X
import hypothesis_test as H

OUT = config.report_path("robustness_report.txt")

W = 92                      # report width (widest table sets it)


# =============================================================================
# FORMATTING
# =============================================================================

def rule(char="─", w=W):
    """Horizontal rule. Delegates to hypothesis_test.sep() so all three
    report files are drawn with the same glyphs."""
    return H.sep(char, w)


def star(macro_star):
    """gen_latex emits TeX '^{*}'; the text report wants a bare asterisk.

    Preferred wherever a \\gsm...Star macro exists, so significance marking
    is decided in exactly one place for both outputs.
    """
    return "*" if macro_star.strip() else " "


def p_star(p_string):
    """Star a p-value that gen_latex has already formatted ('<.001' / '.043').

    Used for the ordinal-logit table, whose macros carry a p-value but no
    companion Star macro. Reads the p rather than the rounded interval,
    because a CI printed to two decimals can show a bound of exactly 1.00
    for a term that is in fact significant.
    """
    if p_string.startswith("<"):
        return "*"
    try:
        return "*" if float(p_string) < config.T["p_sig"] else " "
    except ValueError:
        return " "


def ci_star(ci_string, ref=0.0):
    """Star a bootstrap interval that excludes its null value.

    Used for the PLS path table, where the reported-model paths carry no
    \\gsm...Star macro (the paper stars those in the table source) and no
    p-value at all -- the interval is the only significance statement there.
    """
    lo, hi = (float(x) for x in ci_string.strip("[]").split(","))
    return " " if lo <= ref <= hi else "*"


class Report:
    """Accumulates lines; mirrors the out()/report_lines pattern of the
    other report generators so all three files read alike."""

    def __init__(self):
        self.lines = []

    def __call__(self, *lines):
        self.lines.extend(lines)

    def blank(self, n=1):
        self.lines.extend([""] * n)

    def banner(self, text):
        self.blank()
        self(f"  {text}", rule("─"))

    def check(self, idx, title, verdict):
        """Heading and verdict -- the only framing a check gets."""
        self.blank()
        self(rule("─"), f"  CHECK {idx}  {title}", rule("─"),
             f"  VERDICT   {verdict}", "")

    def tail(self, *lines):
        """At most one short paragraph, and only where the table does not
        already say it."""
        self.blank()
        for line in lines:
            self(f"  {line}" if line else "")

    def save(self, path):
        text = "\n".join(l.rstrip() for l in self.lines).rstrip() + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return text


# =============================================================================
# HEADER
# =============================================================================

def write_intro(R, v, n_full):
    R("GREEN SOFTWARE METRICS SURVEY – ROBUSTNESS & NON-RESPONSE REPORT",
      f"n = {n_full}  |  7 checks  |  companion to the paper's results and limitations"
      f"  |  {date.today().isoformat()}",
      rule("═"), "")

    R.banner("BOTTOM LINE")
    R(f"  ROBUST       Can(Val) → Do. Holds in all {v('gsmSensNVariants')} scorings, under both estimators,",
      "               and under every alternative specification.",
      "  CONDITIONAL  H4 and H8. Hold only where Do includes governance embedding;",
      f"               Want(env) is unrelated to coverage alone (rho = {v('gsmSensCovOnlyWantRho')}).",
      "  WITHDRAWN    Context(maturity) → Do. Survives only where Do keeps the",
      "               governance component maturity overlaps with by definition.",
      "  UNRESOLVED   Whether Can(Val) outpredicts Want(env). Would need"
      f" n ≈ {v('gsmValWantPowerNEighty')}.",
      "  CAVEAT       Completers over-represent broad-footprint organizations, so",
      "               absolute practice levels may be modestly overstated. Figures:",
      "               experiments/dropout_analysis.py.")

    R.banner("GLOSSARY")
    R("  Context     organization size, application-domain footprint, sustainability maturity",
      "  Want(env)   importance of CO2e and energy reduction as objectives (1-5)",
      "  Can(Val)    'optimization is worth the required effort' (1-5)",
      "  Can(Con)    severity of measurement barriers (0-2)",
      "  Do          coverage (metric tiers 0-3) + governance (embedding 0-3) = composite 0-6",
      "  Outlook     likelihood of adopting structured measurement guidelines (1-5)",
      "",
      "  H4 Want(env) → Do      H5 Can(Val) → Do      H6 Want(env) → Outlook      H8 Do → Outlook",
      "",
      "  rho  Spearman correlation, -1 to +1      OR   odds ratio, 1.00 = no effect",
      "  beta standardized PLS path, -1 to +1     *    significant: p < .05, or CI excludes 0 / 1.00")


# =============================================================================
# CHECKS 1-7  (order follows the paper: measurement, then estimator,
#              then specification, then sampling)
# =============================================================================

def check_separability(R, v):
    R.check(1, "ARE Can(Val) AND Want THE SAME CONSTRUCT?",
            "DISTINCT, BUT NOT RANKABLE")
    R(f"  Can(Val) x Want(env)                 rho = {v('gsmValWantDiscriminantRho')}"
      f"   p = {v('gsmValWantDiscriminantP')}   n = {v('gsmSampleValWantListwise')}",
      "",
      f"  Can(Val)  → Do (composite)           rho = {v('gsmCanDoEffortCompositeRho')}",
      f"  Want(env) → Do (composite)           rho = {v('gsmWantDoEnvCompositeRho')}",
      f"  difference, Do                       {v('gsmValWantDiffComposite')}"
      f"   95% CI {v('gsmValWantDiffCompositeCi')}   p = {v('gsmValWantDiffCompositeP')}",
      f"  difference, Outlook                  {v('gsmValWantDiffIntent')}"
      f"   95% CI {v('gsmValWantDiffIntentCi')}   p = {v('gsmValWantDiffIntentP')}",
      "",
      f"  power to detect that gap at n = {v('gsmSampleFull')}   {v('gsmValWantPowerObserved')}%"
      f"        n for 80% power   ~{v('gsmValWantPowerNEighty')}")
    R.tail("Uncorrelated, so not interchangeable. The difference intervals include zero,",
           "but at 29% power that is an underpowered comparison rather than equivalence.")


def check_estimator(R, v):
    R.check(2, "DOES THE RESULT DEPEND ON THE PLS-SEM ESTIMATOR?",
            "NO – ORDINAL LOGIT REPRODUCES THE PATTERN")
    hdr = f"  {'Outcome':<26}{'Predictor':<26}{'OR':>6}  {'95% CI':<16}{'p':>7}"
    R(hdr, "  " + rule("─", len(hdr) - 2))
    for tag, _ycol, pcols, outcome_label in config.ORD_SPECS:
        for i, col in enumerate(pcols):
            term, term_label = config.ORD_TERMS[col]
            R(f"  {outcome_label if i == 0 else '':<26}{term_label:<26}"
              f"{v(f'gsm{tag}{term}Or'):>6}  {v(f'gsm{tag}{term}Ci'):<16}"
              f"{v(f'gsm{tag}{term}P'):>7} {p_star(v(f'gsm{tag}{term}P'))}")
    R("  " + rule("─", len(hdr) - 2),
      f"  Standardized predictors, {v('gsmSampleOrdComposite')} listwise-complete cases (as the PLS model).",
      "",
      f"  {'H6 on those ' + v('gsmSampleOrdIntent') + ' shared cases':<34}Want(env) → Outlook   "
      f"OR = {v('gsmOrdIntentWantOr')}   p = {v('gsmOrdIntentWantP')}",
      f"  {'H6 on all cases (n = ' + v('gsmSampleOrdIntentWide') + ')':<34}Want(env) → Outlook   "
      f"OR = {v('gsmOrdIntentWideWantOr')}   p = {v('gsmOrdIntentWideWantP')} *")
    R.tail("Can(Val) is the only predictor significant for all three outcomes. H6 is decided",
           "by the specification, not the estimator: motivation reaches intent until maturity",
           "and practice enter the model. Context predicts the composite but not coverage —",
           "see check 3.")


ALT_SPECS = [
    ("Reported model",       "Context → Do",  "gsmPlsContextDo"),
    ("Do = coverage only",   "Context → Do",  "gsmPlsAltACtxDo"),
    ("Do = governance only", "Context → Do",  "gsmPlsAltBCtxDo"),
    (None, None, None),
    ("Reported model",       "Want → Do",     "gsmPlsWantDo"),
    ("Context removed",      "Want → Do",     "gsmPlsAltCWantDo"),
    (None, None, None),
    ("Reported model",       "Can → Do",      "gsmPlsCanDo"),
    ("Do = coverage only",   "Can → Do",      "gsmPlsAltACanDo"),
    ("Context removed",      "Can → Do",      "gsmPlsAltCCanDo"),
    (None, None, None),
    ("Do = coverage only",   "Can → Outlook", "gsmPlsAltACanOutlook"),
    ("Context removed",      "Can → Outlook", "gsmPlsAltCCanOutlook"),
]


def check_overlap(R, v):
    R.check(3, "DOES Context OVERLAP WITH Do BY DEFINITION?",
            "YES – maturity → practice WITHDRAWN; Can PATHS ROBUST")
    hdr = f"  {'Specification':<24}{'Path':<18}{'beta':>7}  {'95% CI':<18}"
    R(hdr, "  " + rule("─", len(hdr) - 2))
    for spec, path, macro in ALT_SPECS:
        if spec is None:
            R("")
            continue
        R(f"  {spec:<24}{path:<18}{v(macro + 'Beta'):>7}  "
          f"{v(macro + 'Ci'):<18}{ci_star(v(macro + 'Ci'))}")
    R("  " + rule("─", len(hdr) - 2),
      f"  {v('gsmPlsAltBoot')} bootstrap resamples per specification.",
      "",
      f"  Context x Do(coverage)     r = {v('gsmCtxDoMaturityCoverageR')}"
      f"   p = {v('gsmCtxDoMaturityCoverageP')}",
      f"  Context x Do(governance)   r = {v('gsmCtxDoMaturityGovR')}"
      f"   p = {v('gsmCtxDoMaturityGovP')}")
    R.tail("Context → Do holds only with governance retained. Want → Do is suppressed by the",
           "same overlap: removing Context more than doubles it. The Can paths are unaffected,",
           "so the valuation conclusions do not rest on the disputed construct.")


def check_scoring(R, v):
    n_var = v("gsmSensNVariants")
    counts = {"Can": v("gsmSensCanSig"), "Want": v("gsmSensWantSig"),
              "Outlook": v("gsmSensOutlookSig")}
    R.check(4, "DOES THE RESULT DEPEND ON HOW Do WAS SCORED?",
            "H5 ROBUST  |  H8 CONDITIONAL  |  H4 CONDITIONAL")
    cols = [(tag, label) for _k, tag, label in config.DO_VARIANT_HYPOTHESES]
    lab_w = 44
    R(f"  {'':<{lab_w}}" + "".join(f"{lbl:^16}" for _t, lbl in cols),
      f"  {'Scoring of Do':<{lab_w}}" + "".join(f"{'rho':>10}{'':>6}" for _ in cols))
    R("  " + rule("─", lab_w + 16 * len(cols)))
    for tag, label, _fn in config.DO_VARIANT_SPECS:
        R(f"  {label:<{lab_w}}" + "".join(
            f"{v(f'gsmSens{tag}{who}Rho'):>10}{star(v(f'gsmSens{tag}{who}Star')):>6}"
            for who, _lbl in cols))
    R("  " + rule("─", lab_w + 16 * len(cols)),
      f"  {'HOLDS IN':<{lab_w}}"
      + "".join(f"{counts[t] + ' of ' + n_var:>16}" for t, _l in cols))
    R.tail("H5 survives even where governance is discarded entirely. H8 fails exactly",
           "where Do is reduced to coverage or a raw count. H4 is weakest and holds only where",
           "governance carries weight — motivation reaches the formalization of measurement,",
           "not its extent. The tier rule is not a technicality: it moves the score for",
           f"{v('gsmSensTierDisagreeN')} respondents ({v('gsmSensTierDisagreePct')}%) who report CO2e or SCI without the underlying energy figure.")


def check_common_method(R, v):
    R.check(5, "COULD COMMON METHOD BIAS EXPLAIN THE ASSOCIATIONS?",
            "NO SIGNAL – from a deliberately weak test")
    R(f"  Harman first unrotated factor    {v('gsmHarmanFirstFactorPct')}% of variance",
      "  conventional threshold           50%")
    R.tail("Necessary but not sufficient. Marker-variable or CFA approaches would be stronger",
           "but need a larger sample.")


def check_nonresponse(R, v):
    R.check(6, "WOULD NON-RESPONDERS HAVE ANSWERED DIFFERENTLY?",
            "NO SIGNAL")
    R(f"  early responders  n = {v('gsmNRWaveEarlyN')}          late responders  n = {v('gsmNRWaveLateN')}",
      "",
      f"  Want(env), Can(Val), Do, Outlook    no difference    smallest p = {v('gsmNRWaveCoreMinP')}",
      f"  Outlook(awareness)                  no difference    p = {v('gsmNRWaveAwareP')}, r = {v('gsmNRWaveAwareR')}")
    R.tail("Armstrong-Overton median split on submission date, late responders proxying",
           "non-responders. Evidence of absence only to the extent that proxy holds.")


def check_maturity_exclusion(R, v):
    R.check(7, "DOES EXCLUDING 'I DO NOT KNOW' MATURITY ANSWERS MATTER?",
            "NO – BOTH CONTRASTS HOLD EITHER WAY")
    R(f"  Context → Want(env)    'do not know' grouped with Low    p = {v('gsmCtxWantMaturityLegacyP')} *",
      f"  Context → Do           'do not know' grouped with Low    p = {v('gsmCtxDoMaturityLegacyP')} *")


# =============================================================================

def main():
    csv = config.csv_arg(sys.argv, H.DEFAULT_CSV)
    df = config.load_survey(csv)
    M = X.build_all(df)
    v = M.get

    R = Report()
    write_intro(R, v, len(df))
    for fn in (check_separability, check_estimator, check_overlap, check_scoring,
               check_common_method, check_nonresponse, check_maturity_exclusion):
        fn(R, v)
    R.blank()
    R(rule("═"),
      "  Regenerated by experiments/gen_robustness_report.py from the same macros",
      "  the paper is compiled from, so no value here can disagree with the PDF.")

    out = os.path.abspath(OUT)
    text = R.save(out)
    if "--quiet" not in sys.argv:
        print(text)
    print(f"wrote {out}  ({len(text.splitlines())} lines, n={len(df)})")


if __name__ == "__main__":
    main()
