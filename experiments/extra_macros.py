#!/usr/bin/env python3
"""
=============================================================================
Robustness-report macros  ->  extends scripts/gen_latex.py's registry
=============================================================================
gen_latex.py computes what the paper prints and stops there.  The robustness
report needs more: the per-variant sensitivity table, the ordinal-logit
intervals, the PLS path estimates and alternative specifications, the
early-versus-late responder comparison, and the power figure the discriminant
comparison is read against.  None of those reaches the manuscript, so none of
them belongs in the generator that feeds it.

build_extra(df, M) adds them to a registry gen_latex.build(df) has produced.
It computes nothing gen_latex already emits, and it derives every value from
the same functions gen_latex does -- hypothesis_test.py for the tests,
config.py for the specification lists, pls_bootstrap.py for the model -- so a
value that appears in both the paper and the report cannot differ between them.

Three of these values are read from committed .npz caches rather than
recomputed: the PLS bootstrap, the alternative specifications, and the power
simulation.  Each is checked against one fingerprint of this export, so none
can be a leftover from another data cut.

Usage:
    python extra_macros.py [path/to/export.csv]     # print name/value table

    from extra_macros import build_extra            # as a library
=============================================================================
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

import config
import hypothesis_test as H
import descriptive_analysis as D
import pls_bootstrap as PB
import gen_latex as G
from config import SURVEY_DIMENSIONS, CONTEXT_MATURITY_HIGH
from gen_latex import (
    f_a2, f_a3, f_beta, f_ci, f_d2, f_int, f_p, f_pct, f_prop, f_rho, f_star,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")


# =============================================================================
# COMMITTED CACHES
# =============================================================================

def _open_cache(filename, fingerprint, regenerate_cmd):
    """Open a committed .npz cache, refusing anything not built from this data.

    The caches hold resamples and simulations that take about a quarter of an
    hour to rebuild, so they ship with the package. That only stays honest if a
    cache from a different survey export cannot be picked up silently: every
    writer stamps the fingerprint of its input frame, and a cache without one
    predates the check and is not trusted either.
    """
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        raise SystemExit(f"missing {filename} -- run: {regenerate_cmd}")
    z = np.load(path, allow_pickle=True)
    cached = str(z["fingerprint"]) if "fingerprint" in z.files else ""
    if cached != fingerprint:
        why = "carries no data fingerprint" if not cached else \
              "was built from a different survey export"
        raise SystemExit(f"{filename} {why} -- run: {regenerate_cmd}")
    return z


def _load_power_cache(fingerprint):
    """Read the power simulation for the Valuation-vs-Want correlation contrast.

    Produced by validate_rho_diff.py (which also runs the calibration checks for
    the procedure).  Regenerate with:  python scripts/validate_rho_diff.py
    Returns power at the study's own sample size and the sample size that would
    be needed for 80% power, linearly interpolated between simulated points.
    """
    z = _open_cache("rho_diff_power_cache.npz", fingerprint,
                    "python scripts/validate_rho_diff.py")
    ns, power = z["ns"].astype(float), z["power"].astype(float)
    at_obs = float(power[0])                      # first simulated n is the study n
    if power.max() < 0.80:
        n80 = float("nan")
    else:
        i = int(np.argmax(power >= 0.80))
        if i == 0:
            n80 = ns[0]
        else:                                     # interpolate between bracket
            n80 = ns[i-1] + (ns[i] - ns[i-1]) * (0.80 - power[i-1]) / (power[i] - power[i-1])
    return {"at_observed_n": at_obs, "n_for_eighty": int(round(n80 / 50.0) * 50)}


def _load_pls_alt_cache(fingerprint):
    """Bootstrap results for the alternative PLS specifications.

    Produced by pls_alternatives.py, which refits the model with Do reduced to
    each of its two components and with Context removed, to test whether the
    Context -> Do path is carried by the component that overlaps with the
    maturity definition.  Regenerate with:  python scripts/pls_alternatives.py
    """
    z = _open_cache("pls_alt_cache.npz", fingerprint,
                    "python scripts/pls_alternatives.py")
    store = {k: z[k] for k in z.files}

    def get(spec, path_name):
        pt = float(store[f"{spec}|point|{path_name}"][0])
        b = store[f"{spec}|boot|{path_name}"]
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {"beta": pt, "lo": float(lo), "hi": float(hi), "n_boot": len(b)}

    return get, {s: int(store[f"{s}|n"][0]) for s in ("BASE", "ALT_A", "ALT_B", "ALT_C")}


# =============================================================================
# THE EXTRA MACROS
# =============================================================================

def build_extra(df, M):
    """Add the report-only values to M, in the order the report reads them."""
    scores = H.build_dimension_scores(df)
    masks = H.build_context_masks(df)
    fingerprint = config.survey_fingerprint(df)

    # ---- Check 1: separability of Can(Val) from Want(env) --------------------
    # The paper prints the correlation and its interval; the report also prints
    # the p-value of each, and the power the difference test actually had.
    r_disc = H.spearman_test(scores["can_effort"], scores["want_env"], "", "")
    M.add("gsmValWantDiscriminantP", f_p(r_disc["p"]))
    for out_key, out_tag in [("do_composite", "Composite"), ("outlook_intent", "Intent")]:
        dd = H.dependent_rho_diff(scores["can_effort"], scores["want_env"], scores[out_key])
        M.add(f"gsmValWantDiff{out_tag}P", f_p(dd["p"]))
    pw = _load_power_cache(fingerprint)
    M.add("gsmValWantPowerObserved", f_pct(pw["at_observed_n"] * 100))
    M.add("gsmValWantPowerNEighty", f_int(pw["n_for_eighty"]))

    # ---- Check 2: ordinal logit as the alternative estimator -----------------
    # Same models gen_latex fits, on the same listwise-complete cases; the
    # report prints the interval and the per-model n alongside the odds ratio.
    ord_common = pd.DataFrame({
        "intent": scores["outlook_intent"], "awareness": scores["outlook_awareness"],
        "composite": scores["do_composite"], "coverage": scores["do_breadth"],
        "governance": scores["do_institutionalized"], "want": scores["want_env"],
        "valuation": scores["can_effort"], "maturity": scores["maturity_ordinal"],
    }).dropna()
    label = {col: term for col, (term, _lbl) in config.ORD_TERMS.items()}
    for tag, ycol, pcols, _outcome_label in config.ORD_SPECS:
        res = H.ordinal_logit(ord_common[ycol],
                              {label[c]: ord_common[c] for c in pcols})
        M.add(f"gsmSample{tag}", f_int(res["n"]))
        M.add(f"gsm{tag}PseudoRsq", f_prop(res["prsq"]))
        for term, v in res["terms"].items():
            M.add(f"gsm{tag}{term}Ci", f"[{f_a2(v['lo'])}, {f_a2(v['hi'])}]")
    wide = H.ordinal_logit(scores["outlook_intent"],
                           {"Want": scores["want_env"],
                            "Valuation": scores["can_effort"],
                            "Practice": scores["do_composite"]})
    M.add("gsmOrdIntentWideWantOr", f_a2(wide["terms"]["Want"]["or"]))

    # ---- Check 3: construct overlap between Context and Do -------------------
    # The alternative specifications, and the contrasts that show the circular
    # predictor and the non-circular one explaining different components.
    alt, alt_n = _load_pls_alt_cache(fingerprint)
    M.add("gsmPlsAltBoot", f_int(alt_n["ALT_A"]))
    for spec, stag, pname, ptag in [
        ("ALT_A", "AltA", "Context->Do", "CtxDo"),     # Do = coverage only
        ("ALT_B", "AltB", "Context->Do", "CtxDo"),     # Do = governance only
        ("ALT_C", "AltC", "Want->Do",    "WantDo"),    # Context removed
        ("ALT_C", "AltC", "Can->Do",     "CanDo"),
        ("ALT_A", "AltA", "Can->Do",     "CanDo"),
        ("ALT_A", "AltA", "Can->Outlook", "CanOutlook"),
        ("ALT_C", "AltC", "Can->Outlook", "CanOutlook"),
    ]:
        v = alt(spec, pname)
        M.add(f"gsmPls{stag}{ptag}Beta", f_beta(v["beta"]))
        M.add(f"gsmPls{stag}{ptag}Ci", f_ci(v["lo"], v["hi"]))
    for split, stag in [("maturity_strict", "Maturity"), ("domain_breadth", "Breadth"),
                        ("size_250", "Size")]:
        for key, ktag in [("do_breadth", "Coverage"), ("do_institutionalized", "Gov")]:
            r = H.mw_test(scores[key], masks[split], "")
            M.add(f"gsmCtxDo{stag}{ktag}R", f_a2(r["r"]))
            M.add(f"gsmCtxDo{stag}{ktag}P", f_p(r["p"]))

    # ---- Check 4: sensitivity to how Do is scored ---------------------------
    # gen_latex counts how many variants each association survives; the report
    # prints the correlation behind every cell of that count.
    n_rob = {tag: 0 for _k, tag, _lbl in config.DO_VARIANT_HYPOTHESES}
    for tag, _label, build_variant in config.DO_VARIANT_SPECS:
        variant = build_variant(scores)
        for score_key, who, _lbl in config.DO_VARIANT_HYPOTHESES:
            # H8 runs Do -> Outlook; the other two run predictor -> Do.
            r = (H.spearman_test(variant, scores[score_key], "", "")
                 if who == "Outlook" else
                 H.spearman_test(scores[score_key], variant, "", ""))
            M.add(f"gsmSens{tag}{who}Rho", f_rho(r["rho"]))
            M.add(f"gsmSens{tag}{who}P", f_p(r["p"]))
            M.add(f"gsmSens{tag}{who}Star", f_star(r["p"]))
            if r["p"] < .05:
                n_rob[who] += 1
    M.add("gsmSensCanSig", f_int(n_rob["Can"]))
    # How often do the two coverage rules disagree?  If rarely, the cumulative
    # rule is a detail; if often, it is a substantive modelling choice.
    disagree = (scores["do_breadth"] != scores["do_highest"]).sum()
    M.add("gsmSensTierDisagreeN", f_int(disagree))
    M.add("gsmSensTierDisagreePct", f_pct(disagree / len(df) * 100))

    # ---- Check 6: non-response proxy (Armstrong-Overton) --------------------
    # Late responders proxy non-responders; a median split on submission date
    # tests whether the realized sample differs systematically from one with
    # more reluctant participants.
    from scipy.stats import mannwhitneyu as _mwu
    subdate = pd.to_datetime(df["Date submitted"], errors="coerce")
    order = subdate.rank(method="first")
    late = order > order.median()
    M.add("gsmNRWaveEarlyN", f_int((~late).sum()))
    M.add("gsmNRWaveLateN", f_int(late.sum()))

    def _wave(score_name):
        x = pd.to_numeric(scores[score_name], errors="coerce")
        a = x[~late].dropna(); b = x[late].dropna()
        u, p = _mwu(a, b, alternative="two-sided")
        r = 1 - 2 * u / (len(a) * len(b))   # rank-biserial; <0 => late lower
        return p, r

    # core constructs expected to be unaffected; report the least-reassuring p
    core_ps = [_wave(s_)[0] for s_ in
               ("want_env", "can_effort", "do_composite", "outlook_intent")]
    M.add("gsmNRWaveCoreMinP", f_p(min(core_ps)))
    p_aw, r_aw = _wave("outlook_awareness")
    M.add("gsmNRWaveAwareP", f_p(p_aw))
    M.add("gsmNRWaveAwareR", f_d2(r_aw))

    # ---- Check 7: maturity-exclusion sensitivity ----------------------------
    # The legacy split groups "I do not know" with Low instead of dropping it.
    _mat = df.iloc[:, SURVEY_DIMENSIONS["context_maturity"]["items"]["Sustainability_maturity"]].fillna("None")
    _leg = {"Low": _mat.isin(config.CONTEXT_MATURITY_LOW_LEGACY),
            "High": _mat.isin(CONTEXT_MATURITY_HIGH)}
    M.add("gsmCtxWantMaturityLegacyP", f_p(H.mw_test(scores["want_env"], _leg, "")["p"]))
    M.add("gsmCtxDoMaturityLegacyP", f_p(H.mw_test(scores["do_composite"], _leg, "")["p"]))

    # ---- Reported PLS-SEM ---------------------------------------------------
    # Point estimates from the model in pls_bootstrap.py, intervals from its
    # seeded, disk-cached bootstrap.  Fill the cache first with:
    #   python scripts/pls_bootstrap.py --seconds 35   (repeat until 5000/5000)
    data, cfg = H.model_frame(df), PB.make_cfg()
    point = PB.fit_paths(data, cfg)
    boot, done = PB.run_boot_cached(data, cfg, PB.N_BOOT)
    if done < PB.N_BOOT:
        raise SystemExit(
            f"bootstrap cache incomplete ({done}/{PB.N_BOOT}); "
            f"run `python scripts/pls_bootstrap.py --seconds 35` until complete.")

    def ci(k):
        arr = np.array([x for x in boot[k] if np.isfinite(x)])
        return np.percentile(arr, [2.5, 97.5])

    for k, tag in {
        "Context->Want": "ContextWant", "Context->Can": "ContextCan",
        "Context->Do": "ContextDo",
        "Want->Do": "WantDo", "Can->Do": "CanDo",
        "Want->Outlook": "WantOutlook", "Can->Outlook": "CanOutlook",
        "Do->Outlook": "DoOutlook",
    }.items():
        lo, hi = ci(k)
        M.add(f"gsmPls{tag}Beta", f_beta(point[k]))
        M.add(f"gsmPls{tag}Ci", f_ci(lo, hi))

    # ---- Reliability the report quotes but the paper does not ---------------
    # Spearman-Brown on the two environmental Want items, on the same listwise
    # sample gen_latex takes their correlation from.
    want_data = pd.concat(
        [pd.to_numeric(df.iloc[:, c], errors="coerce")
         for c in SURVEY_DIMENSIONS["want_objectives"]["items"].values()],
        axis=1).dropna()
    rho_env = want_data.iloc[:, [0, 1]].corr(method="spearman").iloc[0, 1]
    M.add("gsmAlphaWantEnvSpearmanBrown", f_a3(D.spearman_brown(rho_env)))

    return M


def build_all(df):
    """The paper's macros plus the report's, in one registry."""
    return build_extra(df, G.build(df))


def main():
    csv = config.csv_arg(sys.argv, H.DEFAULT_CSV)
    df = config.load_survey(csv)
    M = G.Macros()
    build_extra(df, M)
    M.print_table()
    print(f"{len(M._order)} report-only macros, n={len(df)}")


if __name__ == "__main__":
    main()
