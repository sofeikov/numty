"""Benchmark: math-library style code, written blind to checker support.

PCA, least squares, FFT denoising, ODE integration, k-means, signal
processing, geometry, polynomials, rankings. Seeded, runnable, with
runtime shape asserts as the oracle.
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(1)


def pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    total = (singular**2).sum()
    return components[:n_components] @ centered.T, (singular**2)[:n_components] / total


def least_squares(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    sol, residuals, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return sol, float(residuals[0] if residuals.size else 0.0)


def polyfit_predict(xs: np.ndarray, ys: np.ndarray, degree: int, grid: np.ndarray) -> np.ndarray:
    coef = np.polyfit(xs, ys, degree)
    return np.polyval(coef, grid)


def fft_denoise(signal: np.ndarray, keep: int) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    spectrum[keep:] = 0.0
    return np.fft.irfft(spectrum, n=signal.shape[0])


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones((window,)) / window
    return np.convolve(x, kernel, mode="valid")


def lorenz_step(state: np.ndarray, dt: float) -> np.ndarray:
    x, y, z = state
    dx = 10.0 * (y - x)
    dy = x * (28.0 - z) - y
    dz = x * y - (8.0 / 3.0) * z
    return state + dt * np.array([dx, dy, dz])


def integrate_lorenz(n_steps: int, dt: float) -> np.ndarray:
    traj = np.zeros((n_steps, 3))
    traj[0] = np.array([1.0, 1.0, 1.0])
    for i in range(1, n_steps):
        traj[i] = lorenz_step(traj[i - 1], dt)
    return traj


def kmeans(x: np.ndarray, k: int, iters: int) -> tuple[np.ndarray, np.ndarray]:
    centers = x[rng.choice(x.shape[0], size=k, replace=False)]
    for _ in range(iters):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = dists.argmin(axis=1)
        for c in range(k):
            members = x[labels == c]
            if members.shape[0]:
                centers[c] = members.mean(axis=0)
    return centers, labels


def rotation_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_points(points: np.ndarray, mats: np.ndarray) -> np.ndarray:
    return np.einsum("bij,bj->bi", mats, points)


def describe(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def histogram_features(x: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(x, bins=bins)
    return counts / counts.sum(), (edges[1:] + edges[:-1]) / 2.0


def rank_features(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, axis=0)
    ranks = np.empty_like(order)
    ranks[order, np.arange(x.shape[1])] = np.arange(x.shape[0])[:, None]
    return ranks / x.shape[0]


def zscore_clip(x: np.ndarray, width: float = 3.0) -> np.ndarray:
    z = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    return np.clip(z, -width, width)


def pad_reflect_tile(x: np.ndarray, pad: int) -> np.ndarray:
    padded = np.pad(x, pad, mode="reflect")
    tiled = np.tile(padded, (2, 1))
    return np.flip(tiled, axis=0)


def solve_circuit() -> np.ndarray:
    conductance = np.array([[4.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 4.0]])
    currents = np.array([5.0, 0.0, -5.0])
    voltages = np.linalg.solve(conductance, currents)
    assert np.allclose(conductance @ voltages, currents)
    return voltages


def cov_eig(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = np.cov(x, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    return values[::-1], vectors[:, ::-1]


def main() -> None:
    data = rng.normal(size=(200, 6))
    proj, ratio = pca(data, 2)
    assert proj.shape == (2, 200)
    assert ratio.shape == (2,)
    a = rng.normal(size=(50, 4))
    b = a @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(scale=0.1, size=50)
    sol, res = least_squares(a, b)
    assert sol.shape == (4,)
    grid = np.linspace(0.0, 1.0, 25)
    assert polyfit_predict(np.linspace(0, 1, 20), np.sin(np.linspace(0, 1, 20)), 3, grid).shape == (25,)
    t = np.linspace(0.0, 1.0, 256)
    noisy = np.sin(2 * np.pi * 5 * t) + 0.3 * rng.normal(size=256)
    assert fft_denoise(noisy, 12).shape == (256,)
    assert moving_average(noisy, 8).shape == (249,)
    traj = integrate_lorenz(500, 0.01)
    assert traj.shape == (500, 3)
    centers, labels = kmeans(rng.normal(size=(120, 2)), 3, 10)
    assert centers.shape == (3, 2) and labels.shape == (120,)
    mats = np.stack([rotation_2d(t) for t in (0.0, 0.5, 1.0)], axis=0)
    pts = rng.normal(size=(3, 2))
    assert transform_points(pts, mats).shape == (3, 2)
    stats = describe(noisy)
    assert set(stats) == {"mean", "std", "median", "p95", "min", "max"}
    counts, mids = histogram_features(noisy, 16)
    assert counts.shape == (16,) and mids.shape == (16,)
    assert rank_features(data).shape == data.shape
    assert zscore_clip(data).shape == data.shape
    assert pad_reflect_tile(np.ones((4, 5)), 2).shape == (16, 9)
    assert solve_circuit().shape == (3,)
    values, vectors = cov_eig(data)
    assert values.shape == (6,) and vectors.shape == (6, 6)
    uniq, counts_u = np.unique(labels, return_counts=True)
    assert uniq.shape == counts_u.shape
    print("math bench ok")


if __name__ == "__main__":
    main()
