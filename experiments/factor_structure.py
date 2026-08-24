#!/usr/bin/env python3
"""
Exploratory factor analysis of the five Want objective items.

The split into an environmental group (CO2, energy) and an operational group
(cost, performance, compliance) was an author decision.  This asks whether the
data carry that structure.  It was run during analysis and changed no verdict:
the grouping is carried by prior work and by the reliability figures reported
in Section IV-B, and the EFA added only a KMO caveat, so the paper reports the
reliability evidence instead.  Kept here because the decision was checked, and
a reader who wants that check should be able to rerun it.

Not part of the macro pipeline.  numpy/scipy only -- no extra dependency.

Usage:  python factor_structure.py [path/to/export.csv]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

import config
from config import SURVEY_DIMENSIONS, load_survey

DEFAULT_CSV = config.data_path("results-survey668719.csv")


def efa_two_factor(X, n_factors=2):
    """Principal-component extraction with varimax rotation.

    Returns Bartlett's test of sphericity (is there enough common variance to
    factor at all), the Kaiser-Meyer-Olkin index (how much of the correlation
    between items is shared rather than pairwise), the eigenvalues, and the
    rotated loadings.
    """
    X = X.dropna()
    n, p = X.shape
    R = np.corrcoef(X.values, rowvar=False)
    Rinv = np.linalg.inv(R)

    # Bartlett: chi-square against the identity matrix.
    chi2 = -((n - 1) - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    dfree = p * (p - 1) / 2
    bart_p = float(stats.chi2.sf(chi2, dfree))

    # KMO: squared correlations against squared partial correlations.
    D = np.sqrt(np.diag(np.diag(Rinv)))
    P = -np.linalg.inv(D) @ Rinv @ np.linalg.inv(D)
    np.fill_diagonal(P, 1.0)
    off = ~np.eye(p, dtype=bool)
    kmo = float((R[off] ** 2).sum() / ((R[off] ** 2).sum() + (P[off] ** 2).sum()))

    eigenvalues = np.sort(np.linalg.eigvalsh(R))[::-1]

    # Principal axis factoring was tried first and produced a Heywood case -- a
    # standardized loading above 1 -- because the environmental factor rests on
    # only two items and its communalities are not identified from so few.
    # Components are bounded by construction and report the structure without
    # that artifact.
    w, v = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1][:n_factors]
    load = v[:, idx] * np.sqrt(np.maximum(w[idx], 0))

    # Varimax rotation.
    L = load.copy()
    Rot = np.eye(n_factors)
    d_old = 0
    for _ in range(200):
        Lr = L @ Rot
        u, s, vt = np.linalg.svd(
            L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(axis=0)) / L.shape[0]))
        Rot = u @ vt
        d = s.sum()
        if d_old != 0 and d / d_old < 1 + 1e-9:
            break
        d_old = d
    L = L @ Rot

    # Sign and column order are arbitrary in EFA: fix them so output is stable.
    for k in range(n_factors):
        if L[:, k].sum() < 0:
            L[:, k] *= -1
    order = np.argsort(-(L ** 2).sum(axis=0))
    L = L[:, order]
    return {"n": int(n), "bartlett_chi2": float(chi2), "bartlett_p": bart_p,
            "kmo": kmo, "eigenvalues": eigenvalues,
            "loadings": pd.DataFrame(L, index=X.columns,
                                     columns=[f"F{i + 1}" for i in range(n_factors)])}


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    df = load_survey(csv)

    items = SURVEY_DIMENSIONS["want_objectives"]["items"]
    X = pd.DataFrame({k: pd.to_numeric(df.iloc[:, v], errors="coerce")
                      for k, v in items.items()}).dropna()
    efa = efa_two_factor(X)

    clusters = SURVEY_DIMENSIONS["want_objectives"]["clusters"]
    L = efa["loadings"]
    env, ops = L.loc[clusters["Environmental"]], L.loc[clusters["Operational"]]
    # Which extracted factor the environmental items landed on.
    f_env = 0 if abs(env.iloc[0, 0]) > abs(env.iloc[0, 1]) else 1

    print(f"\nEFA of the five Want objective items   (listwise n = {efa['n']})")
    print(f"  Bartlett sphericity   chi2 = {efa['bartlett_chi2']:.1f}, "
          f"p = {efa['bartlett_p']:.4f}")
    print(f"  KMO                   {efa['kmo']:.3f}"
          "   (below .60 is the caveat that kept this out of the paper)")
    print("  eigenvalues           "
          + ", ".join(f"{e:.2f}" for e in efa["eigenvalues"]))
    print(f"  factors with eigenvalue > 1   {int((efa['eigenvalues'] > 1).sum())}")

    print("\n  varimax-rotated loadings")
    print("  " + L.round(3).to_string().replace("\n", "\n  "))

    print("\n  Environmental items on their factor   "
          f"{env.iloc[:, f_env].min():.3f} to {env.iloc[:, f_env].max():.3f}")
    print("  Operational items on their factor     "
          f"{ops.iloc[:, 1 - f_env].min():.3f} to {ops.iloc[:, 1 - f_env].max():.3f}")
    print("  Largest cross-loading                 "
          f"{max(abs(env.iloc[:, 1 - f_env]).max(), abs(ops.iloc[:, f_env]).max()):.3f}\n")


if __name__ == "__main__":
    main()
