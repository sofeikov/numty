"""Broadcast mismatches fail for the newly hooked operators (issue #54)."""
import numpy as np

from numty import F64, I64


def floordiv_bad(x: F64[2, 3], c: F64[5]) -> None:
    x // c


def mod_bad(x: F64[2, 3], c: F64[5]) -> None:
    x % c


def xor_bad(k: I64[2, 3], c: I64[5]) -> None:
    k ^ c


def and_bad(k: I64[2, 3], c: I64[5]) -> None:
    k & c
