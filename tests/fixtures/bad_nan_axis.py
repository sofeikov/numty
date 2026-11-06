"""nan reductions check axes like the base ops."""
import numpy as np


def f() -> None:
    np.nanmean(np.zeros((4, 3)), axis=2)
