"""Recursive type aliases reach numpy calls without crashing mypy.

The mixed-shape guard walks operand and result types for ndarrays; a
recursive alias expanded without a cycle check raised RecursionError
(an INTERNAL ERROR) on any numpy call it reached.
"""
import numpy as np

JSON = dict[str, "JSON"] | list["JSON"] | int
Nested = list["Nested"] | float


def scalar(v: JSON) -> bool:
    return bool(np.isscalar(v))


def build(v: Nested) -> None:
    np.array(v)
    np.asarray(v)


def method(v: Nested, n: int) -> None:
    a = np.zeros((n, 3))
    a.fill(0.0)
    a = a * 2
    np.zeros(n).searchsorted(v)
    np.clip(a, v, v)
    np.full((n, 3), v)


if __name__ == "__main__":
    print(scalar({"a": [1, 2]}), scalar(3))
    build([1.0, 2.0])
    method(0.5, 4)
