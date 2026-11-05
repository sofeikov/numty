"""Benchmark: realistic ML code, written blind to checker support.

A small MLP with manual backprop, a 1-D conv classifier head, batch-norm
style normalization, dropout, embeddings, and a train/eval loop. Seeded,
runnable, with runtime shape asserts as the oracle.
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


def make_blobs(n: int = 300, dim: int = 4, n_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    centers = rng.normal(size=(n_classes, dim)) * 3.0
    labels = rng.integers(0, n_classes, size=n)
    points = centers[labels] + rng.normal(scale=0.8, size=(n, dim))
    return points, labels


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], n_classes))
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def train_test_split(
    x: np.ndarray, y: np.ndarray, frac: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = rng.permutation(x.shape[0])
    cut = int(x.shape[0] * frac)
    tr, te = idx[:cut], idx[cut:]
    return x[tr], y[tr], x[te], y[te]


def init_params(d_in: int, d_hidden: int, d_out: int) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(scale=np.sqrt(2.0 / d_in), size=(d_in, d_hidden)),
        "b1": np.zeros((d_hidden,)),
        "w2": rng.normal(scale=np.sqrt(2.0 / d_hidden), size=(d_hidden, d_out)),
        "b2": np.zeros((d_out,)),
    }


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h = relu(x @ params["w1"] + params["b1"])
    probs = softmax(h @ params["w2"] + params["b2"])
    return probs, {"h": h}


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(targets * np.log(np.clip(probs, eps, 1.0))) / probs.shape[0])


def backward(
    params: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    x: np.ndarray,
    probs: np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    n = x.shape[0]
    dlogits = (probs - targets) / n
    h = cache["h"]
    dw2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ params["w2"].T
    dh[h <= 0.0] = 0.0
    dw1 = x.T @ dh
    db1 = dh.sum(axis=0)
    return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}


def sgd_step(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> None:
    for key in params:
        params[key] -= lr * grads[key]


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def dropout(x: np.ndarray, p: float) -> np.ndarray:
    mask = (rng.random(size=x.shape) >= p).astype(x.dtype)
    return x * mask / (1.0 - p)


def embed(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    return table[ids]


def conv1d_valid(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Cross-correlation over the time axis, channels last."""
    t, c_in = signal.shape
    k, _, c_out = kernel.shape
    out_t = t - k + 1
    windows = np.stack([signal[i : i + k] for i in range(out_t)], axis=0)
    return np.einsum("tko,koc->tc", windows, kernel)


def top_k(probs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    part = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(probs, part, axis=1), axis=1)
    best = np.take_along_axis(part, order, axis=1)
    return best, np.take_along_axis(probs, best, axis=1)


def main() -> None:
    x, labels = make_blobs()
    assert x.shape == (300, 4)
    y = one_hot(labels, 3)
    assert y.shape == (300, 3)
    xtr, ytr, xte, yte = train_test_split(x, labels)
    params = init_params(4, 16, 3)
    for _ in range(5):
        probs, cache = forward(params, xtr)
        loss = cross_entropy(probs, one_hot(ytr, 3))
        grads = backward(params, cache, xtr, probs, one_hot(ytr, 3))
        sgd_step(params, grads, 0.1)
    print(f"train loss {loss:.4f}")
    test_probs, _ = forward(params, xte)
    print(f"test acc {accuracy(test_probs, yte):.4f}")

    normed = layer_norm(xtr)
    assert normed.shape == xtr.shape
    dropped = dropout(xtr, 0.1)
    assert dropped.shape == xtr.shape
    table = rng.normal(size=(10, 8))
    assert embed(np.array([1, 4, 7]), table).shape == (3, 8)
    sig = rng.normal(size=(64, 2))
    ker = rng.normal(size=(5, 2, 3))
    assert conv1d_valid(sig, ker).shape == (60, 3)
    best, vals = top_k(test_probs, 2)
    assert best.shape == (test_probs.shape[0], 2)
    assert vals.shape == best.shape
    print("ml bench ok")


if __name__ == "__main__":
    main()
