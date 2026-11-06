"""Same-named numpy twins keep their own stub types: none of these may error.

np.ma, np.emath / np.lib.scimath and numpy.matlib share function names
with the ndarray API (array, zeros, concatenate, sqrt, ...) but return
MaskedArray, complex results and matrix. numty dispatches on exact
fullnames, so it must not rewrite them into plain float64 ndarrays.
Run as a script, each result is checked against numpy itself.
"""
from typing import Any

import numpy as np
import numpy.lib.scimath
import numpy.matlib

m = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
print(m.mask.tolist(), m.filled(0.0).tolist(), m.count())
assert isinstance(m, np.ma.MaskedArray)

x: np.ma.MaskedArray[Any, Any] = np.ma.zeros((2, 3))
cat: np.ma.MaskedArray[Any, Any] = np.ma.concatenate([x, x])
print(type(cat).__name__, cat.shape, cat.mask)
assert isinstance(cat, np.ma.MaskedArray) and cat.shape == (4, 3)
print(np.ma.concatenate([np.ma.zeros(2), np.ma.zeros(2)]).mask, np.ma.zeros((2, 3)).count())

a = np.ones((2, 3))
picked: np.ma.MaskedArray[Any, Any] = np.ma.where(a > 0, a, 0.0)
folded: np.ma.MaskedArray[Any, Any] = np.ma.reshape(x, (3, 2))
wrapped: np.ma.MaskedArray[Any, Any] = np.ma.asarray(a)
for y in (picked, folded, wrapped):
    assert isinstance(y, np.ma.MaskedArray)
print(folded.shape)

# emath is not the np.sqrt ufunc: real input can give complex output.
r: np.ndarray[tuple[int], np.dtype[np.complex128]] = np.emath.sqrt(
    np.array([-1.0, 4.0])
)
lg: np.ndarray[tuple[int], np.dtype[np.complex128]] = np.lib.scimath.log(
    np.array([-1.0])
)
print(r.dtype, lg.dtype)
assert r.dtype.kind == "c" and lg.dtype.kind == "c"

mat: np.matrix[Any, Any] = numpy.matlib.zeros(3)
print(type(mat).__name__, mat.shape)
assert isinstance(mat, np.matrix) and mat.shape == (1, 3)
