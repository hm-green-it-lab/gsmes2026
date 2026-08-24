#!/usr/bin/env python3
"""Seeded, resumable non-parametric bootstrap for the reported PLS-SEM.

Defines the model once -- `build_data()` and `make_cfg()` here are the single
specification; pls_sem.py, pls_alternatives.py and experiments/extra_macros.py
all import or mirror this one.  Refits it on each resample and collects the
direct structural path coefficients.

plspm's own `bootstrap=True` is not used: it forks worker processes and returns
no seed control, so its intervals are not reproducible run to run.  This loop is
single-process and draws every resample from a named seed instead."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import hypothesis_test as H
from config import load_survey
from plspm.plspm import Plspm
from plspm.config import Config, Structure, MV
from plspm.mode import Mode
from plspm.scheme import Scheme

N_BOOT = 5000
BOOT_SEED = 7
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"boot_cache_b{N_BOOT}.npz")


def build_data():
    """The listwise-complete frame the model is fitted on.

    Defined by hypothesis_test.model_frame(), which gen_latex.py also reads the
    paper's listwise-complete n from, so the analysis sample is specified once.
    """
    return H.model_frame(load_survey(H.DEFAULT_CSV))


def make_cfg():
    """Measurement and structural model.

      Context  single indicator    : sustainability maturity
      Want     reflective (Mode A) : CO2 reduction, energy reduction
      Can      single indicator    : effort-worthwhile Likert
      Do       formative  (Mode B) : metric coverage, governance embedding
      Outlook  formative  (Mode B) : adoption intent, decision-area awareness
    """
    st = Structure()
    st.add_path(["Context"], ["Want"]); st.add_path(["Context"], ["Can"])
    st.add_path(["Want"], ["Do"]);  st.add_path(["Can"], ["Do"])
    st.add_path(["Do"], ["Outlook"])
    # direct paths from motivation and capability to Outlook (H6, H7)
    st.add_path(["Want"], ["Outlook"]); st.add_path(["Can"], ["Outlook"])
    # direct Context -> Do path (H3): the full conceptual model is tested jointly
    st.add_path(["Context"], ["Do"])
    cfg = Config(st.path(), scaled=True)
    cfg.add_lv("Context", Mode.A, MV("maturity"))
    cfg.add_lv("Want", Mode.A, MV("co2"), MV("energy"))
    cfg.add_lv("Can", Mode.A, MV("can_effort"))
    cfg.add_lv("Do", Mode.B, MV("do_breadth"), MV("do_inst"))
    cfg.add_lv("Outlook", Mode.B, MV("ol_intent"), MV("ol_aware"))
    return cfg


def fit_paths(data, cfg):
    """The eight direct structural paths, as {"Src->Tgt": beta}."""
    p = Plspm(data, cfg, Scheme.PATH, iterations=300, tolerance=1e-7).path_coefficients()
    ctx_w = p.loc["Want", "Context"]; ctx_can = p.loc["Can", "Context"]
    want_d = p.loc["Do", "Want"]; can_d = p.loc["Do", "Can"]
    do_o = p.loc["Outlook", "Do"]
    want_o = p.loc["Outlook", "Want"]; can_o = p.loc["Outlook", "Can"]
    ctx_do = p.loc["Do", "Context"]
    return {
        "Context->Want": ctx_w, "Context->Can": ctx_can, "Context->Do": ctx_do,
        "Want->Do": want_d, "Can->Do": can_d,
        "Want->Outlook": want_o, "Can->Outlook": can_o,
        "Do->Outlook": do_o,
    }


def run_boot_cached(data, cfg, n_boot=N_BOOT, cache_path=CACHE_PATH,
                    max_seconds=None, verbose=False):
    """Resumable, disk-cached non-parametric bootstrap.

    Each resample j draws its indices from a deterministic, independent RNG
    (np.random.SeedSequence(BOOT_SEED).spawn(n_boot)[j]), so the full set of
    n_boot resamples is reproducible and order-stable regardless of how many
    invocations it is split across.  Results are stored row-per-seed (NaN row
    for a failed PLS refit) in an .npz cache; a run resumes from the cached
    count and may stop early after max_seconds, allowing the bootstrap to be
    filled incrementally.  Returns (boot_dict_of_lists, done_count).
    """
    n = len(data)
    keys = list(fit_paths(data, cfg).keys())
    seeds = np.random.SeedSequence(BOOT_SEED).spawn(n_boot)
    idx = np.arange(n)
    # Keyed to the survey export, so a new data cut invalidates the cache
    # instead of silently reusing resamples drawn from different respondents.
    fp = config.survey_fingerprint(load_survey(H.DEFAULT_CSV))

    effects = np.full((n_boot, len(keys)), np.nan)
    done = 0
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if (list(z["keys"]) == keys and int(z["n_boot"]) == n_boot
                and "fingerprint" in z.files and str(z["fingerprint"]) == fp):
            cached = z["effects"]
            effects[:cached.shape[0]] = cached
            done = int(z["done"])

    def _save(d):
        np.savez(cache_path, keys=np.array(keys), effects=effects[:d],
                 done=d, n_boot=n_boot, fingerprint=fp)

    t0 = time.time()
    while done < n_boot:
        rng = np.random.default_rng(seeds[done])
        s = rng.choice(idx, n, replace=True)
        try:
            r = fit_paths(data.iloc[s].reset_index(drop=True), cfg)
            effects[done] = [r[k] for k in keys]
        except Exception:
            pass  # leave NaN row; keeps seed alignment
        done += 1
        if done % 100 == 0:
            _save(done)  # checkpoint so a killed run never loses progress
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (n_boot - done) / rate if rate > 0 else float("inf")
            eta = f"~{remaining:.0f}s remaining" if np.isfinite(remaining) else "estimating..."
            print(f"  {done}/{n_boot} ({100*done/n_boot:.0f}%)  {eta}   ", end="\r", flush=True)
        if max_seconds is not None and (done % 20 == 0) and (time.time() - t0) > max_seconds:
            break

    _save(done)
    print()  # end the \r progress line
    if verbose:
        ok = int(np.isfinite(effects[:done, 0]).sum())
        print(f"  cache {cache_path}: {done}/{n_boot} seeds processed "
              f"({ok} successful refits)")
    boot = {k: list(effects[:done, i]) for i, k in enumerate(keys)}
    return boot, done


def main():
    data = build_data()
    cfg = make_cfg()
    n = len(data)
    point = fit_paths(data, cfg)
    keys = list(point.keys())
    # honour an optional wall-clock budget so the bootstrap can be filled in
    # chunks: `python pls_bootstrap.py --seconds 35` (repeat until complete).
    budget = None
    if "--seconds" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--seconds") + 1])
    boot, done = run_boot_cached(data, cfg, N_BOOT, max_seconds=budget, verbose=True)

    # Readable console labels for the direct structural paths, tagged with the
    # hypothesis each edge corresponds to.
    DISPLAY = {
        "Context->Want": "Context -> Want (H1)",
        "Context->Can":  "Context -> Can  (H2)",
        "Context->Do":   "Context -> Do   (H3)",
        "Want->Do":      "Want -> Do      (H4)",
        "Can->Do":       "Can -> Do       (H5)",
        "Want->Outlook": "Want -> Outlook (H6)",
        "Can->Outlook":  "Can -> Outlook  (H7)",
        "Do->Outlook":   "Do -> Outlook   (H8)",
    }

    def ci(k):
        arr = np.array([x for x in boot[k] if np.isfinite(x)])
        return np.percentile(arr, [2.5, 97.5])

    def show(k):
        lo, hi = ci(k)
        sig = "*" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {DISPLAY.get(k, k):<22}{point[k]:>+9.3f}  [{lo:>+6.3f}, {hi:>+6.3f}]  {sig}")

    print(f"PLS-SEM bootstrap  n={n}  seeds_processed={done}/{N_BOOT}\n")
    print(f"  {'direct path':<22}{'estimate':>9}  {'95% CI':>20}  {'sig'}")
    print("  " + "-" * 60)
    for k in keys:
        show(k)

    # H6 and H7 are evaluated only here (direct effects on Outlook). Print their
    # verdicts explicitly so they are visible without reading the CI column.
    def verdict(k):
        lo, hi = ci(k)
        return "SUPPORTED" if (lo > 0 or hi < 0) else "NOT SUPPORTED"
    print("\n  Hypothesis verdicts assessed in this model:")
    print(f"    H6  Want -> Outlook :  {verdict('Want->Outlook')}")
    print(f"    H7  Can  -> Outlook :  {verdict('Can->Outlook')}")


if __name__ == "__main__":
    main()
