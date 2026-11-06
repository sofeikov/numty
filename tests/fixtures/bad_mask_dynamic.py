"""Correct code hitting a solver gap (see #13): boolean-mask row
filtering selects a data-dependent count, so the hook fails loudly at
the indexing site instead of guessing."""
import numpy as np

from numty import F64, batch


def select(y: F64[batch, 16]) -> None:
    mask = y.mean(axis=1) > 0.5
    filtered = y[mask]
