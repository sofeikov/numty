"""All good: mypy must report zero errors here."""
from typing import Any, Final, Literal, cast

import numpy as np

from numty import Array2D, FloatArray, Matrix, ident, matmul, narrow

A2 = FloatArray[tuple[Literal[2]]]
A3 = FloatArray[tuple[Literal[3]]]
A1x3 = FloatArray[tuple[Literal[1], Literal[3]]]
A1x4 = FloatArray[tuple[Literal[4]]]
A2x1 = FloatArray[tuple[Literal[2], Literal[1]]]
A2x2 = FloatArray[tuple[Literal[2], Literal[2]]]
A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]
A2x4 = FloatArray[tuple[Literal[2], Literal[4]]]
A2x5 = FloatArray[tuple[Literal[2], Literal[5]]]
A3x2 = FloatArray[tuple[Literal[3], Literal[2]]]
A3x3 = FloatArray[tuple[Literal[3], Literal[3]]]
A4x4 = FloatArray[tuple[Literal[4], Literal[4]]]
A6int = np.ndarray[tuple[Literal[6]], np.dtype[np.signedinteger[Any]]]
A2x7 = FloatArray[tuple[Literal[2], Literal[7]]]
A2x2x3 = FloatArray[tuple[Literal[2], Literal[2], Literal[3]]]
B2x3 = Array2D[Literal[2], Literal[3]]

H: Final = 2
W: Final = 3


def use_2(m: A2) -> None: ...


def use_3(m: A3) -> None: ...


def use_1x3(m: A1x3) -> None: ...


def use_1x4(m: A1x4) -> None: ...


def use_2x1(m: A2x1) -> None: ...


def use_2x2(m: A2x2) -> None: ...


def use_2x3(m: A2x3) -> None: ...


def use_b(m: B2x3) -> None: ...


def use_2x4(m: A2x4) -> None: ...


def use_2x5(m: A2x5) -> None: ...


def use_3x2(m: A3x2) -> None: ...


def use_4x4(m: A4x4) -> None: ...


def use_3x3(m: A3x3) -> None: ...


def use_6int(m: A6int) -> None: ...


A3int = np.ndarray[tuple[Literal[3]], np.dtype[np.signedinteger[Any]]]


def use_3int(m: A3int) -> None: ...


def use_2x7(m: A2x7) -> None: ...


def use_2x2x3(m: A2x2x3) -> None: ...


def chain() -> None:
    c = matmul(np.zeros((2, 3)), np.zeros((3, 4)))
    use_2x4(c)
    use_2x4(ident(c))


A5x2x4 = FloatArray[tuple[Literal[5], Literal[2], Literal[4]]]


def use_5x2x4(m: A5x2x4) -> None: ...


def matmul_batch_broadcasts() -> None:
    use_5x2x4(matmul(np.zeros((1, 2, 3)), np.zeros((5, 3, 4))))
    use_5x2x4(matmul(np.zeros((5, 2, 3)), np.zeros((3, 4))))


def unblessed_shapes_solve() -> None:
    use_2x2(matmul(np.zeros((2, 3)), np.zeros((3, 2))))
    use_2x5(matmul(np.zeros((2, 3)), np.zeros((3, 5))))


def matmul_operator() -> None:
    # @ is accepted but unchecked (numpy's default); use matmul() to solve.
    use_2x4(np.zeros((2, 3)) @ np.zeros((3, 4)))


def precision_visible() -> None:
    reveal_type(matmul(np.zeros((2, 3)), np.zeros((3, 4))))
    reveal_type(np.zeros((2, 4)) + np.zeros((2, 1)))
    reveal_type(np.zeros((2, 3)))
    reveal_type(np.zeros((2, 3)).reshape((3, -1)))
    reveal_type(np.zeros((H * 2, W + 1)))
    reveal_type(np.zeros((2, 3)).transpose())
    reveal_type(np.zeros((2, 3)).sum(axis=1, keepdims=True))
    reveal_type(np.zeros_like(np.zeros((2, 3))))
    reveal_type(np.arange(6))


def narrowed(m: Matrix) -> None:
    use_2x4(matmul(narrow(m, np.zeros((2, 3))), np.zeros((3, 4))))


def stored_constructor_stays_precise() -> None:
    w = np.zeros((2, 3))
    use_2x3(w)
    use_b(w)
    v: A2x3 = np.zeros((2, 3))
    use_2x3(v)


def use_matrix(m: Matrix) -> None: ...


def dynamic_dims_bail_out_safely(n: int) -> None:
    # Unknown sizes cannot be proven wrong, so the hook leaves them alone.
    w = np.zeros((n, 3))
    use_matrix(w)


def end_to_end_stays_precise() -> None:
    use_2x4(matmul(np.zeros((2, 3)), np.zeros((3, 4))) + np.zeros((2, 4)))
    use_2x4(np.zeros((2, 4)) * np.zeros((2, 1)))
    use_2x4(np.zeros((2, 4)) + 1.0)


def reshape_stays_precise() -> None:
    use_3x2(np.zeros((2, 3)).reshape((3, -1)))
    use_2x4(np.zeros((8,)).reshape(2, 4))
    use_2x4(np.reshape(np.zeros((8,)), (2, 4)))
    use_1x4(np.zeros((2, 2)).reshape(4))
    use_2x4(matmul(np.zeros((2, 3)), np.zeros((3, 4))).reshape((2, -1)))


def folded_shapes() -> None:
    use_2x3(np.zeros((H, W)))
    use_1x4(np.zeros((H * W - 2,)))
    use_4x4(np.zeros((H * 2, W + 1)))
    use_3x2(np.zeros((H, W)).reshape((W, H)))


def rank_ops_stay_precise() -> None:
    use_3x2(np.zeros((2, 3)).transpose())
    use_3x2(np.zeros((2, 3)).transpose(1, 0))
    use_3x2(np.transpose(np.zeros((2, 3)), (1, 0)))
    use_3(np.zeros((1, 3)).squeeze())
    use_2x3(np.zeros((1, 2, 3)).squeeze(0))
    use_2x3(np.squeeze(np.zeros((2, 1, 3)), axis=1))
    use_1x3(np.expand_dims(np.zeros((3,)), 0))
    use_2(np.zeros((2, 3)).sum(axis=1))
    use_2x1(np.zeros((2, 3)).sum(axis=1, keepdims=True))
    use_2x1(np.zeros((2, 3)).mean(axis=-1, keepdims=True))
    use_2x3(np.zeros((2, 3)).sum(axis=()))


Af32_2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]
Ac64_2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.complex64]]
Ai32_2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.int32]]


def use_f32_2x3(m: Af32_2x3) -> None: ...


def use_c64_2x3(m: Ac64_2x3) -> None: ...


def promotion_stays_precise() -> None:
    f32 = cast(Af32_2x3, np.zeros((2, 3)))
    use_2x3(f32 + np.zeros((2, 3)))  # float32 + float64 -> float64
    use_f32_2x3(f32 + 1.5)  # weak float keeps float32
    use_f32_2x3(f32 + True)  # weak bool keeps the array dtype
    use_f32_2x3(f32 * np.float32(2.0))  # strong numpy scalar promotes
    use_c64_2x3(f32 * 1j)  # weak complex + float32 -> complex64
    i32 = cast(Ai32_2x3, np.zeros((2, 3)))
    use_2x3(i32 + 2.5)  # int32 + float -> float64
    use_2x4(matmul(f32, np.zeros((3, 4))))  # matmul promotes too


def builders_stay_precise() -> None:
    use_2x3(np.zeros_like(np.zeros((2, 3))))
    use_2x3(np.zeros_like(np.zeros((2, 4)), shape=(2, 3)))
    use_3x2(np.ones_like(np.zeros((2, 3)), shape=(3, 2)))
    use_2x3(np.full((2, 3), 5.0))
    use_3x3(np.eye(3))
    use_3x2(np.eye(3, 2))
    use_6int(np.arange(6))
    use_3int(np.arange(1, 10, 3))
    use_2x7(np.concatenate((np.zeros((2, 3)), np.zeros((2, 4))), axis=1))
    use_2x2x3(np.stack((np.zeros((2, 3)), np.zeros((2, 3))), axis=0))


def subscript_stays_precise() -> None:
    use_3(np.zeros((2, 3))[0])
    use_2(np.zeros((2, 3))[:, 1])
    use_1x3(np.zeros((2, 3))[1:2])
    use_2x3(np.zeros((4, 3))[1:3])
    use_2x3(np.zeros((4, 3))[::2])
    use_1x3(np.zeros((3,))[None, :])
    use_2(np.zeros((2, 3))[..., 1])
    use_3(np.zeros((2, 3))[-1])
    use_2x3(np.zeros((1, 2, 3))[0])
    use_2x3(np.zeros((2, 3))[np.newaxis, :, :][0])
    use_1x3(np.zeros((2, 3))[1:99])


Ab2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.bool_]]
A4x2 = FloatArray[tuple[Literal[4], Literal[2]]]
Ai4 = np.ndarray[tuple[Literal[4]], np.dtype[np.signedinteger[Any]]]
Ai2 = np.ndarray[tuple[Literal[2]], np.dtype[np.signedinteger[Any]]]
Ai3 = np.ndarray[tuple[Literal[3]], np.dtype[np.signedinteger[Any]]]
Ai2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.signedinteger[Any]]]


def use_i2(m: Ai2) -> None: ...


def use_i3(m: Ai3) -> None: ...


def use_i2x3(m: Ai2x3) -> None: ...


def use_b2x3(m: Ab2x3) -> None: ...


def use_4x2(m: A4x2) -> None: ...


def use_i4(m: Ai4) -> None: ...


def comparisons_stay_precise() -> None:
    use_b2x3(np.zeros((2, 3)) == np.zeros((2, 3)))
    use_b2x3(np.zeros((2, 3)) != 1.0)
    use_b2x3(np.zeros((2, 3)) < np.zeros((2, 1)))


def arg_family_stays_precise() -> None:
    use_i2(np.zeros((2, 3)).argmax(axis=1))
    use_i3(np.argmax(np.zeros((2, 3)), axis=0))
    use_i2x3(np.zeros((2, 3)).argsort(axis=1))
    use_2x3(np.sort(np.zeros((2, 3))))
    use_3(np.partition(np.zeros((5,)), 2)[:3])


def list_stacking_stays_precise() -> None:
    use_2x2x3(np.stack([np.zeros((2, 3)), np.zeros((2, 3))], axis=0))
    use_2x7(np.concatenate([np.zeros((2, 3)), np.zeros((2, 4))], axis=1))


def rng_stays_precise() -> None:
    g = np.random.default_rng(0)
    use_2x3(g.normal(size=(2, 3)))
    use_i4(g.integers(0, 10, size=(4,)))
    use_3(g.choice(np.zeros((5,)), size=(3,)))
    use_4x2(g.permutation(np.zeros((4, 2))))


def scalar_reduction_uses_stub_default() -> None:
    reveal_type(np.zeros((2, 3)).sum())


def reshape_dynamic_bails_out_safely(n: int) -> None:
    # Unknown dims cannot be proven wrong, so the hook leaves them alone.
    w = np.zeros((n, 3)).reshape((n, 3))
    use_matrix(w)
