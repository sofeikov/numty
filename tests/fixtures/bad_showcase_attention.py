"""README showcase: attention that returns the weights, not the values."""
import numpy as np

from numty import F32


def attention(
    q: F32["batch heads seq dim"],
    k: F32["batch heads seq dim"],
    v: F32["batch heads seq dim"],
) -> F32["batch heads seq dim"]:
    scores = q @ k.swapaxes(-1, -2)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights  # E: (batch, heads, seq, seq), forgot `@ v`
