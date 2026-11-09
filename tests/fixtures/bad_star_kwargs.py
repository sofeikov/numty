"""Star/keyword misuse fails loudly (issue #52).

Each row is a runtime ValueError the checker used to miss (wrong shape
with no error) or misreport: clip bounds never broadcast, reshape
shape= changes the total, and out= cannot hold the computed shape.
"""
import numpy as np

a = np.ones((2, 3))
w = np.ones((4,))
c = np.zeros((4, 5, 6))
small = np.empty((3,))

reveal_type(np.clip(a, min=w))  # (2, 3) vs (4,): broadcast clash
reveal_type(np.reshape(a, shape=(4, 2)))  # 6 elements into (4, 2)
reveal_type(np.multiply(a, a, out=small))  # (2, 3) does not fit (3,)
reveal_type(np.sum(c[0], axis=0, out=np.zeros(2)))  # (6,) does not fit (2,)
