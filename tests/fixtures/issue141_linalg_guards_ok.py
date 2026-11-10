"""Issue #141: det/slogdet/qr/matrix_power/cond/tensorsolve/cross solve shapes."""
import numpy as np


def f() -> None:
    reveal_type(np.linalg.det(np.zeros((2, 2))))
    reveal_type(np.linalg.slogdet(np.zeros((2, 2))))
    reveal_type(np.linalg.qr(np.zeros((3, 4))))
    reveal_type(np.linalg.matrix_power(np.zeros((2, 2)), 2))
    reveal_type(np.linalg.cond(np.zeros((2, 2))))
    reveal_type(np.linalg.tensorsolve(np.eye(4).reshape(2, 2, 2, 2), np.zeros((2, 2))))
    reveal_type(np.cross(np.zeros(3), np.zeros(3)))
