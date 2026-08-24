#!/usr/bin/env python3
"""
=============================================================================
Alternative PLS-SEM specifications (circularity and robustness checks)
=============================================================================
Sustainability maturity, which operationalizes Context, is defined partly by
whether an organization has sustainability KPIs or SDLC integration.  The Do
construct contains a governance-embedding component built from decision
frequency and formal green requirements in the design process.  Those overlap
in content, so the Context -> Do path is at risk of being partly definitional
rather than empirical.

This script refits the model under specifications that isolate the overlap:

  BASE   Do = metric coverage + governance embedding, Context included
         (the paper's model, refit here as a specification check)
  ALT_A  Do = metric coverage ONLY
         Removes the component that overlaps with the maturity definition.
         If Context -> Do survives here, the path is not merely definitional.
  ALT_B  Do = governance embedding ONLY
         The complement of ALT_A; isolates the overlapping component.
  ALT_C  Context removed entirely
         Shows what the remaining paths look like when the overlapping
         predictor cannot absorb their variance -- the check that revealed
         Want -> Do to be suppressed by Context rather than genuinely null.

Resumable by design: bootstrap draws are appended to pls_alt_cache.npz on every
run, so the target resample count can be reached across several invocations
without losing work, and draw j is drawn from its own spawned seed so the result
does not depend on how the run was split.  The cache carries a fingerprint of
the survey export and is rejected if that changes.
experiments/extra_macros.py reads it; regenerate after every data cut.

NOTE ON THE COMMITTED CACHE: the shipped pls_alt_cache.npz was accumulated
before the seeding above was made split-independent, so rebuilding it from
scratch draws a different, equally valid set of resamples and shifts the
interval bounds in the third decimal.  It is kept as shipped because the
intervals in the paper were read from it.  Delete it to rebuild, and expect
\gsmPlsAlt...Ci to move by about 0.02; no point estimate and no significance
verdict changes.

Usage:
    python pls_alternatives.py                # top all specs up to TARGET_B
    python pls_alternatives.py --chunk 400    # add at most 400 draws per spec
    python pls_alternatives.py --status       # report progress, compute nothing
    python pls_alternatives.py --force        # discard cache and restart
=============================================================================
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import hypothesis_test as H
from config import SURVEY_DIMENSIONS, load_survey
from plspm.plspm import Plspm
from plspm.config import Config, Structure, MV
from plspm.mode import Mode
from plspm.scheme import Scheme

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pls_alt_cache.npz")
TARGET_B = 1000
SEED = 7

# spec name -> (Do indicators, include Context)
SPECS = {
    "BASE":  (["do_breadth", "do_inst"], True),
    "ALT_A": (["do_breadth"],            True),
    "ALT_B": (["do_inst"],               True),
    "ALT_C": (["do_breadth", "do_inst"], False),
}

# BASE is the paper's own model and is already bootstrapped at 5,000 resamples
# by pls_bootstrap.py; it is refitted here only to confirm this script's
# specification matches, so a small resample count is sufficient for it.
TARGETS = {"BASE": 200, "ALT_A": TARGET_B, "ALT_B": TARGET_B, "ALT_C": TARGET_B}


def build_data():
    """Same listwise-complete frame the main model is fitted on."""
    df = load_survey(H.DEFAULT_CSV)
    S = H.build_dimension_scores(df)
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


def make_cfg(do_mvs, with_context):
    """The model of pls_bootstrap.make_cfg(), with Do's indicators and the
    presence of Context parameterized."""
    st = Structure()
    if with_context:
        st.add_path(["Context"], ["Want"])
        st.add_path(["Context"], ["Can"])
        st.add_path(["Context"], ["Do"])
    st.add_path(["Want"], ["Do"])
    st.add_path(["Can"], ["Do"])
    st.add_path(["Do"], ["Outlook"])
    st.add_path(["Want"], ["Outlook"])
    st.add_path(["Can"], ["Outlook"])
    cfg = Config(st.path(), scaled=True)
    if with_context:
        cfg.add_lv("Context", Mode.A, MV("maturity"))
    cfg.add_lv("Want", Mode.A, MV("co2"), MV("energy"))
    cfg.add_lv("Can", Mode.A, MV("can_effort"))
    cfg.add_lv("Do", Mode.B, *[MV(m) for m in do_mvs])
    cfg.add_lv("Outlook", Mode.B, MV("ol_intent"), MV("ol_aware"))
    return cfg


def fit_paths(cfg, d):
    """Non-zero direct path coefficients as {'Src->Tgt': beta}."""
    pc = Plspm(d, cfg, Scheme.PATH).path_coefficients()
    return {f"{src}->{tgt}": pc.loc[tgt, src]
            for tgt in pc.index for src in pc.columns if pc.loc[tgt, src] != 0}


def load_cache(fingerprint):
    """Cached draws, or an empty store if the cache is absent or stale.

    A cache without a fingerprint predates the check and is discarded rather
    than trusted: silently reusing resamples from a different data cut is the
    one failure mode this file exists to prevent.
    """
    if not os.path.exists(CACHE_PATH):
        return {}
    z = np.load(CACHE_PATH, allow_pickle=True)
    store = {k: z[k] for k in z.files}
    cached_fp = str(store.get("fingerprint", ""))
    if cached_fp != fingerprint:
        why = "no fingerprint" if not cached_fp else "different survey export"
        print(f"  discarding {os.path.basename(CACHE_PATH)} ({why}); "
              f"recomputing from scratch")
        return {}
    return store


def save_cache(store, fingerprint):
    store["fingerprint"] = np.array(fingerprint)
    np.savez_compressed(CACHE_PATH, **store)


def run(chunk, target=TARGET_B):
    data = build_data()
    fp = config.survey_fingerprint(load_survey(H.DEFAULT_CSV))
    store = load_cache(fp)
    print(f"listwise-complete n = {len(data)};  target = {target} resamples/spec")

    for spec, (mvs, ctx) in SPECS.items():
        cfg = make_cfg(mvs, ctx)
        point = fit_paths(cfg, data)
        for k, v in point.items():
            store[f"{spec}|point|{k}"] = np.array([v])

        spec_target = TARGETS.get(spec, target)
        have = int(store.get(f"{spec}|n", np.array([0]))[0])
        todo = max(0, min(chunk, spec_target - have))
        if todo == 0:
            print(f"  {spec}: {have}/{spec_target} -- complete")
            continue

        # Draw j comes from its own spawned seed, so the sequence depends only
        # on j -- not on how many invocations it was filled across. Every spec
        # spawns from the same SEED and therefore sees the same resamples, which
        # is what makes the specifications comparable rather than merely similar.
        seeds = np.random.SeedSequence(SEED).spawn(spec_target)
        acc = {k: list(store.get(f"{spec}|boot|{k}", np.array([]))) for k in point}
        done = 0
        for j in range(have, have + todo):
            rng = np.random.default_rng(seeds[j])
            s = data.iloc[rng.integers(0, len(data), len(data))].reset_index(drop=True)
            try:
                for k, v in fit_paths(cfg, s).items():
                    if k in acc:
                        acc[k].append(v)
                done += 1
            except Exception:
                pass                     # non-converging resample, discarded
        for k, v in acc.items():
            store[f"{spec}|boot|{k}"] = np.array(v)
        store[f"{spec}|n"] = np.array([have + done])
        save_cache(store, fp)
        print(f"  {spec}: {have + done}/{spec_target} (+{done} this run)")

    save_cache(store, fp)


def summarize():
    store = load_cache(config.survey_fingerprint(load_survey(H.DEFAULT_CSV)))
    if not store:
        print("cache empty or stale -- run without --status first")
        return
    for spec in SPECS:
        n = int(store.get(f"{spec}|n", np.array([0]))[0])
        print(f"\n{spec}  ({n} resamples)")
        keys = sorted(k.split("|")[-1] for k in store if k.startswith(f"{spec}|point|"))
        for k in keys:
            pt = float(store[f"{spec}|point|{k}"][0])
            b = store.get(f"{spec}|boot|{k}")
            if b is None or len(b) < 50:
                print(f"   {k:16s} beta={pt:+.3f}   (CI pending)")
                continue
            lo, hi = np.percentile(b, [2.5, 97.5])
            flag = "sig " if (lo > 0 or hi < 0) else "n.s."
            print(f"   {k:16s} beta={pt:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  {flag}")


def main():
    if "--force" in sys.argv and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    if "--status" in sys.argv:
        summarize()
        return
    chunk = TARGET_B
    if "--chunk" in sys.argv:
        chunk = int(sys.argv[sys.argv.index("--chunk") + 1])
    run(chunk)
    summarize()


if __name__ == "__main__":
    main()
