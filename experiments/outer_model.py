#!/usr/bin/env python3
"""
=============================================================================
Outer (measurement) model for the reported PLS-SEM, with bootstrap intervals
=============================================================================
For formative measurement the diagnostic quantities are the outer weights (each
indicator's contribution to the composite) together with their significance, and
the outer loadings (each indicator's simple association with the composite),
which are what justifies retaining an indicator whose weight is not significant.
This script produces all of them.

The paper reports VIF for its formative blocks and has no measurement-model
table, so none of these figures are cited and the analysis is not part of the
macro pipeline.  It is kept because it was run: a reviewer who asks for outer
weights should get them from the package rather than from a rerun.

Standardized outer weights are obtained by regressing the standardized latent
variable score on its standardized indicators, not by rescaling the `weight`
column of plspm's outer_model().  That column is an intermediate quantity and
does not exactly reproduce the composite for Mode B blocks; the regression does
(verified at R^2 = 1.0, see check_exact() below).

Loadings are the correlation of each indicator with its own composite.

Bootstrap resamples are sign-corrected before aggregation: PLS composites are
identified only up to sign, so a resample whose block flips sign would
otherwise inflate the interval.  Each resample's block is flipped back to agree
with the point estimate.

Resumable in the same way as scripts/pls_alternatives.py.  Writes its own
cache beside this file; nothing else reads it.

Usage:
    python outer_model.py                # top up to TARGET_B
    python outer_model.py --chunk 200    # add at most 200 resamples
    python outer_model.py --status       # report only
=============================================================================
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

import pls_alternatives as A
from plspm.plspm import Plspm
from plspm.scheme import Scheme

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "outer_model_cache.npz")
TARGET_B = 1000
SEED = 11

# Only the multi-indicator blocks are diagnostic; single-indicator blocks have
# weight and loading equal to one by construction.
BLOCKS = {
    "Want":    ["co2", "energy"],           # reflective (Mode A)
    "Do":      ["do_breadth", "do_inst"],   # formative  (Mode B)
    "Outlook": ["ol_intent", "ol_aware"],   # formative  (Mode B)
}


def _z(frame):
    return (frame - frame.mean()) / frame.std(ddof=0)


def outer_stats(d):
    """Standardized outer weights and loadings for every multi-indicator block."""
    cfg = A.make_cfg(["do_breadth", "do_inst"], True)
    p = Plspm(d, cfg, Scheme.PATH)
    scores = p.scores()
    out = {}
    for lv, items in BLOCKS.items():
        z = _z(d[items])
        y = _z(scores[lv])
        w = np.linalg.lstsq(z.values, y.values, rcond=None)[0]
        for k, item in enumerate(items):
            out[f"{lv}|{item}|w"] = float(w[k])
            out[f"{lv}|{item}|l"] = float(np.corrcoef(z[item], y)[0, 1])
    return out


def check_exact(d):
    """The regression must reproduce the composite exactly, or the weights it
    returns are not the composite's weights."""
    cfg = A.make_cfg(["do_breadth", "do_inst"], True)
    scores = Plspm(d, cfg, Scheme.PATH).scores()
    worst = 1.0
    for lv, items in BLOCKS.items():
        z = _z(d[items]); y = _z(scores[lv])
        b = np.linalg.lstsq(z.values, y.values, rcond=None)[0]
        r2 = np.corrcoef(z.values @ b, y.values)[0, 1] ** 2
        worst = min(worst, r2)
    return worst


def _sign_correct(sample, point):
    """Flip a resample's block if PLS resolved its composite with opposite sign."""
    fixed = dict(sample)
    for lv, items in BLOCKS.items():
        ref = sum(point[f"{lv}|{i}|w"] for i in items)
        cur = sum(sample.get(f"{lv}|{i}|w", 0.0) for i in items)
        if ref * cur < 0:
            for i in items:
                for suffix in ("w", "l"):
                    k = f"{lv}|{i}|{suffix}"
                    if k in fixed:
                        fixed[k] = -fixed[k]
    return fixed


def run(chunk):
    d = A.build_data()
    r2 = check_exact(d)
    print(f"listwise-complete n = {len(d)};  weight recovery R^2 = {r2:.6f}")
    if r2 < 0.999:
        raise SystemExit("regression does not reproduce the composite; weights invalid")

    point = outer_stats(d)
    store = {}
    if os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH, allow_pickle=True)
        store = {k: z[k] for k in z.files}
    for k, v in point.items():
        store[f"point|{k}"] = np.array([v])

    have = int(store.get("n", np.array([0]))[0])
    todo = max(0, min(chunk, TARGET_B - have))
    if todo == 0:
        print(f"  {have}/{TARGET_B} -- complete")
        return

    rng = np.random.default_rng(SEED + have)
    acc = {k: list(store.get(f"boot|{k}", np.array([]))) for k in point}
    done = 0
    for _ in range(todo):
        s = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
        try:
            for k, v in _sign_correct(outer_stats(s), point).items():
                acc[k].append(v)
            done += 1
        except Exception:
            pass
    for k, v in acc.items():
        store[f"boot|{k}"] = np.array(v)
    store["n"] = np.array([have + done])
    np.savez_compressed(CACHE_PATH, **store)
    print(f"  {have + done}/{TARGET_B} (+{done} this run)")


def summarize():
    if not os.path.exists(CACHE_PATH):
        print("cache empty")
        return
    z = np.load(CACHE_PATH, allow_pickle=True)
    store = {k: z[k] for k in z.files}
    n = int(store.get("n", np.array([0]))[0])
    print(f"\nOuter model ({n} resamples)")
    print(f"{'Block':9s} {'Indicator':12s} {'weight':>8s} {'95% CI':>18s} {'':4s} "
          f"{'loading':>8s} {'95% CI':>18s}")
    for lv, items in BLOCKS.items():
        for i in items:
            row = f"{lv:9s} {i:12s} "
            for suffix in ("w", "l"):
                pt = float(store[f"point|{lv}|{i}|{suffix}"][0])
                b = store.get(f"boot|{lv}|{i}|{suffix}")
                if b is None or len(b) < 50:
                    row += f"{pt:8.3f} {'(pending)':>18s} {'':4s}"
                    continue
                lo, hi = np.percentile(b, [2.5, 97.5])
                flag = "*" if (lo > 0 or hi < 0) else "n.s."
                row += f"{pt:8.3f} {f'[{lo:+.3f},{hi:+.3f}]':>18s} {flag:4s}"
            print(row)


def main():
    if "--status" in sys.argv:
        summarize(); return
    chunk = TARGET_B
    if "--chunk" in sys.argv:
        chunk = int(sys.argv[sys.argv.index("--chunk") + 1])
    run(chunk)
    summarize()


if __name__ == "__main__":
    main()
