#!/usr/bin/env python3
"""
=============================================================================
Validation and power analysis for H.dependent_rho_diff()
=============================================================================
The paper compares two dependent, overlapping Spearman correlations: the
association of Valuation (can_effort) with an outcome against the association
of goal-level motivation (want_env) with the SAME outcome.  Because both
correlations share the outcome variable they are statistically dependent and
cannot be compared with an independent-samples test.

hypothesis_test.dependent_rho_diff() implements a non-parametric percentile
bootstrap over cases.  This script establishes that the procedure is sound
before its output is reported:

  A  Seed stability      -- the estimate must not depend on the RNG seed.
  B  Cross-validation    -- agreement with Williams' t, the parametric
                            alternative (which assumes bivariate normality
                            that the ordinal scores do not satisfy).
  C  Null calibration    -- false-positive rate at a true null difference
                            must sit at or below the nominal .05.
  D  Power at the observed effect -- the number that decides whether a
                            non-significant result is evidence of equality
                            or merely an underpowered comparison.

(D) is the substantive one.  It is cached to  rho_diff_power_cache.npz  and read
by experiments/extra_macros.py, so the power figures in the robustness report
are generated rather than hand-transcribed.  Like every cache in the package it carries a fingerprint of
the input data: the simulation is calibrated to the observed correlations, so a
new export invalidates it.

Usage:
    python validate_rho_diff.py              # run checks, refresh cache
    python validate_rho_diff.py --power-only # refresh the cache only
=============================================================================
"""
import os
import sys
import numpy as np
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import hypothesis_test as H
from config import load_survey

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "rho_diff_power_cache.npz")

# Simulation settings.  N_SIM x B_SIM is a deliberate compromise: large enough
# that the power estimate is stable to about +/-.04, small enough to rerun.
N_SIM = 400
B_SIM = 500
POWER_NS = (82, 150, 250, 400)
SEED = 11


# -----------------------------------------------------------------------------
# Fast rank correlation for the simulation loops only.  Identical in value to
# scipy.stats.spearmanr but without the per-call overhead, which dominates when
# it is invoked several hundred thousand times.  Verified against scipy below.
# -----------------------------------------------------------------------------
def _rho(a, b):
    ra = rankdata(a); rb = rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return (ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb))


def _rho_rows(A, B):
    """Tie-aware Spearman for each row pair of two (m, n) arrays at once.

    rankdata(..., axis=1) resolves ties by average rank exactly as
    scipy.stats.spearmanr does, so this is the same statistic computed for m
    bootstrap resamples in one pass.  Vectorizing matters here: the power
    simulation evaluates the statistic several hundred thousand times.
    """
    RA = rankdata(A, axis=1); RB = rankdata(B, axis=1)
    RA = RA - RA.mean(axis=1, keepdims=True)
    RB = RB - RB.mean(axis=1, keepdims=True)
    num = (RA * RB).sum(axis=1)
    den = np.sqrt((RA ** 2).sum(axis=1) * (RB ** 2).sum(axis=1))
    return num / den


def _ordinalize(v, k=5):
    """Bin a continuous variate into k ordered levels, matching the Likert-type
    granularity of the real scores (ties matter for a rank statistic)."""
    q = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(v, q).astype(float)


def _diff_p(x1, x2, y, n_boot, seed):
    """Bootstrap two-sided p for rho(x1,y) - rho(x2,y); mirrors the shipped
    implementation but evaluates all resamples in one vectorized pass."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    d = _rho_rows(x1[idx], y[idx]) - _rho_rows(x2[idx], y[idx])
    return 2 * min((d <= 0).mean(), (d >= 0).mean())


def _calibrate(target_rho, rng, n=60000):
    """Find the loading a such that ordinalized  a*y + noise  has Spearman
    rho ~= target_rho with ordinalized y."""
    def emp(a):
        y = rng.normal(size=n)
        x = a * y + rng.normal(size=n)
        return _rho(_ordinalize(x), _ordinalize(y))
    lo, hi = 0.01, 3.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if emp(mid) < target_rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# =============================================================================
# Checks
# =============================================================================
def check_fast_rho_matches_scipy():
    """The vectorized statistic must equal scipy's, ties included, or every
    downstream calibration figure is measuring the wrong thing."""
    rng = np.random.default_rng(0)
    worst_scalar = worst_rows = 0.0
    A = np.empty((50, 90)); B = np.empty((50, 90))
    for i in range(50):
        # deliberately coarse (3 levels) to force heavy ties
        a = _ordinalize(rng.normal(size=90), k=3)
        b = _ordinalize(rng.normal(size=90), k=3)
        A[i], B[i] = a, b
        worst_scalar = max(worst_scalar,
                           abs(_rho(a, b) - stats.spearmanr(a, b).statistic))
    rows = _rho_rows(A, B)
    for i in range(50):
        worst_rows = max(worst_rows,
                         abs(rows[i] - stats.spearmanr(A[i], B[i]).statistic))
    print(f"  [0] vs scipy.spearmanr (heavy ties): scalar dev={worst_scalar:.2e}, "
          f"vectorized dev={worst_rows:.2e}")
    assert worst_scalar < 1e-10 and worst_rows < 1e-10, \
        "rank correlation diverges from scipy"


def check_seed_stability(S):
    print("  [A] Seed stability (real data, Valuation vs Want on do_composite)")
    out = []
    for seed in (7, 42, 123, 2024, 99999):
        d = H.dependent_rho_diff(S["can_effort"], S["want_env"],
                                 S["do_composite"], n_boot=5000, seed=seed)
        out.append(d["p"])
        print(f"      seed={seed:<6d} diff={d['diff']:+.3f} "
              f"CI[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] p={d['p']:.3f}")
    print(f"      spread of p across seeds: {max(out) - min(out):.3f}")


def check_against_williams(S):
    print("  [B] Cross-validation against Williams' t")

    def williams(rjk, rjh, rkh, n):
        R = 1 - rjk**2 - rjh**2 - rkh**2 + 2 * rjk * rjh * rkh
        t = ((rjk - rjh) * np.sqrt((n - 1) * (1 + rkh))
             / np.sqrt(2 * ((n - 1) / (n - 3)) * R
                       + (((rjk + rjh) / 2) ** 2) * ((1 - rkh) ** 3)))
        return t, 2 * (1 - stats.t.cdf(abs(t), n - 3))

    for out in ("do_composite", "outlook_intent", "do_institutionalized"):
        d = H.dependent_rho_diff(S["can_effort"], S["want_env"], S[out], n_boot=5000)
        _, pw = williams(d["rho1"], d["rho2"], d["rho_predictors"], d["n"])
        print(f"      {out:22s} bootstrap p={d['p']:.3f}  Williams p={pw:.3f}  "
              f"|delta|={abs(d['p'] - pw):.3f}")


def check_null_calibration(n_sim=N_SIM):
    print(f"  [C] Null calibration ({n_sim} simulations, n=82, equal true strength)")
    rng = np.random.default_rng(1)
    rej = 0
    for i in range(n_sim):
        y = rng.normal(size=82)
        x1 = 0.45 * y + rng.normal(size=82)
        x2 = 0.45 * y + rng.normal(size=82)
        if _diff_p(_ordinalize(x1), _ordinalize(x2), _ordinalize(y), B_SIM, i) < .05:
            rej += 1
    rate = rej / n_sim
    print(f"      false-positive rate: {rej}/{n_sim} = {rate:.3f}  (nominal .05)")
    return rate


def compute_power(rho_high, rho_low, ns=POWER_NS, n_sim=N_SIM):
    """Power to detect a difference of the observed magnitude, by sample size."""
    cal = np.random.default_rng(3)
    a1 = _calibrate(rho_high, cal)
    a2 = _calibrate(rho_low, cal)
    print(f"  [D] Power at the observed effect (rho {rho_high:.3f} vs {rho_low:.3f}; "
          f"loadings {a1:.3f} / {a2:.3f})")
    res = {}
    for n in ns:
        rng = np.random.default_rng(SEED)
        det = 0
        for i in range(n_sim):
            y = rng.normal(size=n)
            x1 = a1 * y + rng.normal(size=n)
            x2 = a2 * y + rng.normal(size=n)
            if _diff_p(_ordinalize(x1), _ordinalize(x2), _ordinalize(y), B_SIM, i) < .05:
                det += 1
        res[n] = det / n_sim
        print(f"      n={n:<5d} power = {res[n]:.2f}")
    return res


# =============================================================================
def main():
    power_only = "--power-only" in sys.argv
    df = load_survey(H.DEFAULT_CSV)
    S = H.build_dimension_scores(df)

    obs = H.dependent_rho_diff(S["can_effort"], S["want_env"], S["do_composite"])
    print(f"\nObserved (listwise n={obs['n']}): "
          f"rho_valuation={obs['rho1']:+.3f}, rho_want={obs['rho2']:+.3f}, "
          f"diff={obs['diff']:+.3f}, p={obs['p']:.3f}\n")

    if not power_only:
        check_fast_rho_matches_scipy()
        check_seed_stability(S)
        check_against_williams(S)
        check_null_calibration()

    power = compute_power(round(obs["rho1"], 3), round(obs["rho2"], 3))

    fp = config.survey_fingerprint(df)
    np.savez(CACHE_PATH,
             ns=np.array(list(power.keys())),
             power=np.array(list(power.values())),
             rho_high=obs["rho1"], rho_low=obs["rho2"],
             n_sim=N_SIM, b_sim=B_SIM, seed=SEED, fingerprint=fp)
    print(f"\nwrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
