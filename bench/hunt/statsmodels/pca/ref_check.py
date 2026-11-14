"""Compare clean.py against the installed statsmodels (0.15.0) PCA.

Runs ``harness.run`` on both ``clean`` and ``statsmodels.multivariate.pca``
(the upstream class has the same attribute names), then compares every
output grouped by case (the key prefix before the first dot). Also checks a
few extra cases directly: a pandas-free tall/wide pair with every method,
and ``project`` on a fitted object. Prints PASS/FAIL per case; exits
non-zero on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
from statsmodels.multivariate import pca as upstream

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, list) and a and isinstance(a[0], str):
        return a == b
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True))


def main() -> int:
    clean = _load("clean")
    harness = _load("harness")
    ours = harness.run(clean)
    theirs = harness.run(upstream)
    cases: dict[str, list[str]] = {}
    for key in sorted(set(ours) | set(theirs)):
        case = key.split(".")[0]
        if key not in ours or key not in theirs:
            cases.setdefault(case, []).append(f"{key}: missing")
            continue
        if not _same(ours[key], theirs[key]):
            cases.setdefault(case, []).append(
                f"{key}: ours {np.shape(ours[key])} vs upstream {np.shape(theirs[key])}"
            )
        else:
            cases.setdefault(case, [])

    # Extra direct cases: every method on a tall and a wide matrix, with and
    # without weights, comparing the main attributes and project() variants.
    rng = np.random.default_rng(99)
    for nobs, nvar, ncomp in ((25, 6, 3), (5, 8, 4), (40, 7, 7)):
        x = rng.normal(size=(nobs, nvar)) @ rng.normal(size=(nvar, nvar))
        wts = rng.uniform(0.3, 3.0, size=nvar)
        for method in ("svd", "eig", "nipals"):
            for wt in (None, wts):
                tag = f"direct_{nobs}x{nvar}_{method}_{'w' if wt is not None else 'nw'}"
                bad: list[str] = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a = clean.PCA(x, ncomp=ncomp, method=method, weights=wt)
                    b = upstream.PCA(x, ncomp=ncomp, method=method, weights=wt)
                for attr in (
                    "factors", "loadings", "coeff", "projection", "eigenvals",
                    "eigenvecs", "rsquare", "ic", "weights", "transformed_data",
                ):
                    if not _same(getattr(a, attr), getattr(b, attr)):
                        bad.append(attr)
                for kw in (
                    {}, {"ncomp": 1}, {"transform": False},
                    {"transform": False, "unweight": False},
                ):
                    if not _same(a.project(**kw), b.project(**kw)):
                        bad.append(f"project{kw}")
                cases[tag] = bad

    failed = 0
    for case, problems in sorted(cases.items()):
        if problems:
            failed += 1
            print(f"FAIL {case}")
            for p in problems:
                print(f"    {p}")
        else:
            print(f"PASS {case}")
    print(f"{len(cases) - failed}/{len(cases)} cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
