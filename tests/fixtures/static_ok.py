"""All good: mypy must report zero errors here."""
from enum import IntEnum
from typing import Any, Final, Literal, TypeVar, cast

import numpy as np
from typing_extensions import TypeVarTuple, Unpack

from numty import Array2D, FloatArray, FlexArray, Matrix, ident, matmul, narrow

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
M = TypeVar("M", bound=int)
C = TypeVar("C", bound=int)
Ts = TypeVarTuple("Ts")


class Slots(IntEnum):
    X = 0
    Y = 1
    VX = 2

A2 = FloatArray[tuple[Literal[2]]]
A3 = FloatArray[tuple[Literal[3]]]
A1x3 = FloatArray[tuple[Literal[1], Literal[3]]]
A1x4 = FloatArray[tuple[Literal[4]]]
A2x1 = FloatArray[tuple[Literal[2], Literal[1]]]
A2x2 = FloatArray[tuple[Literal[2], Literal[2]]]
A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]
A2x3x2 = FloatArray[tuple[Literal[2], Literal[3], Literal[2]]]
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
COL: Final = slice(Slots.X, Slots.Y + 1)


def use_2(m: A2) -> None: ...


def use_3(m: A3) -> None: ...


def use_1x3(m: A1x3) -> None: ...


def use_1x4(m: A1x4) -> None: ...


def use_2x1(m: A2x1) -> None: ...


def use_2x2(m: A2x2) -> None: ...


def use_2x3(m: A2x3) -> None: ...


def use_2x3x2(m: A2x3x2) -> None: ...


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
Af2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.floating[Any]]]
Ai5 = np.ndarray[tuple[Literal[5]], np.dtype[np.signedinteger[Any]]]
Bb1x3 = np.ndarray[tuple[Literal[1], Literal[3]], np.dtype[np.bool_]]
Bb3 = np.ndarray[tuple[Literal[3]], np.dtype[np.bool_]]
SN102 = np.ndarray[tuple[N, Literal[10], Literal[2]], np.dtype[np.float64]]
SN20 = np.ndarray[tuple[N, Literal[20]], np.dtype[np.float64]]
A5x20 = FloatArray[tuple[Literal[5], Literal[20]]]
Ai5x10 = np.ndarray[tuple[Literal[5], Literal[10]], np.dtype[np.signedinteger[Any]]]
A5 = FloatArray[tuple[Literal[5]]]
A5x10 = FloatArray[tuple[Literal[5], Literal[10]]]
A5x3x2 = FloatArray[tuple[Literal[5], Literal[3], Literal[2]]]
A5x10x2 = FloatArray[tuple[Literal[5], Literal[10], Literal[2]]]
A5x3x10x2 = FloatArray[tuple[Literal[5], Literal[3], Literal[10], Literal[2]]]
A5x2 = FloatArray[tuple[Literal[5], Literal[2]]]
A5x4 = FloatArray[tuple[Literal[5], Literal[4]]]
A4 = FloatArray[tuple[Literal[4]]]
A5x3 = FloatArray[tuple[Literal[5], Literal[3]]]
A5x5 = FloatArray[tuple[Literal[5], Literal[5]]]
A7x2 = FloatArray[tuple[Literal[7], Literal[2]]]
Ac4 = np.ndarray[tuple[Literal[4]], np.dtype[np.complexfloating[Any]]]
Ac4x4 = np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.complexfloating[Any]]]


def use_i2(m: Ai2) -> None: ...


def use_i3(m: Ai3) -> None: ...


def use_i2x3(m: Ai2x3) -> None: ...


def use_f2x3(m: Af2x3) -> None: ...


def use_i5(m: Ai5) -> None: ...


def use_b1x3(m: Bb1x3) -> None: ...


def use_b3(m: Bb3) -> None: ...


def use_5x20(m: A5x20) -> None: ...


A2x3x3 = FloatArray[tuple[Literal[2], Literal[3], Literal[3]]]


def use_2x3x3(m: A2x3x3) -> None: ...


def use_i5x10(m: Ai5x10) -> None: ...


def use_5(m: A5) -> None: ...


def use_5x10(m: A5x10) -> None: ...


def use_5x3x2(m: A5x3x2) -> None: ...


def use_5x10x2(m: A5x10x2) -> None: ...


def use_5x3x10x2(m: A5x3x10x2) -> None: ...


def use_5x2(m: A5x2) -> None: ...


def use_5x4(m: A5x4) -> None: ...


def use_4(m: A4) -> None: ...


def use_5x3(m: A5x3) -> None: ...


def use_5x5(m: A5x5) -> None: ...


def use_7x2(m: A7x2) -> None: ...


def use_c4(m: Ac4) -> None: ...


def use_c4x4(m: Ac4x4) -> None: ...


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


def einsum_stays_precise() -> None:
    a = np.zeros((2, 3))
    b = np.zeros((3, 4))
    use_2x4(np.einsum("ij,jh->ih", a, b))
    use_4x2(np.einsum("ij,jh", a, b))
    use_2(np.einsum("ij,j->i", a, np.zeros(3)))
    use_3(np.einsum("ii->i", np.zeros((3, 3))))
    use_7x2(np.einsum("bij,bj->bi", np.zeros((7, 2, 3)), np.zeros((7, 3))))
    use_7x2(np.einsum("...ij,j->...i", np.zeros((7, 2, 3)), np.zeros(3)))
    use_7x2(np.einsum("...ij,...j", np.zeros((7, 2, 3)), np.zeros((7, 3))))


def linalg_stays_precise() -> None:
    a4 = np.zeros((4, 4))
    use_4(np.linalg.solve(a4, np.zeros(4)))
    use_4x2(np.linalg.solve(a4, np.zeros((4, 2))))
    use_4x4(np.linalg.inv(a4))
    use_c4(np.linalg.eig(a4)[0])
    use_c4x4(np.linalg.eig(a4)[1])
    w, v = np.linalg.eigh(a4)
    use_4(w)
    use_4x4(v)
    u, s, vh = np.linalg.svd(np.zeros((5, 3)))
    use_5x5(u)
    use_3(s)
    use_3x3(vh)
    u2, s2, vh2 = np.linalg.svd(np.zeros((5, 3)), full_matrices=False)
    use_5x3(u2)
    use_3(s2)
    use_3x3(vh2)
    use_3(np.linalg.svd(np.zeros((5, 3)), compute_uv=False))
    sol, _, _, _ = np.linalg.lstsq(np.zeros((5, 3)), np.zeros(5))
    use_3(sol)
    use_2(np.linalg.norm(np.zeros((2, 3)), axis=1))


# Symbolic shapes: TypeVars in the shape tuple, applied explicitly at each
# use (a bare SNK would silently mean tuple[Any, Any]).
SNK = np.ndarray[tuple[N, K], np.dtype[np.float64]]
SNKC = np.ndarray[tuple[N, K, C], np.dtype[np.float64]]
SNKb = np.ndarray[tuple[N, K], np.dtype[np.bool_]]


def pack_like(values: SNKC[N, K, C], present: SNKb[N, K]) -> SNKC[N, K, C]:
    order = np.argsort(~present, axis=1, kind="stable")
    return np.take_along_axis(values, order[..., None], axis=1)


def symbolic_matmul(
    a: np.ndarray[tuple[N, K], np.dtype[np.float64]],
    b: np.ndarray[tuple[K, M], np.dtype[np.float64]],
) -> np.ndarray[tuple[N, M], np.dtype[np.float64]]:
    return np.matmul(a, b)


def symbolic_broadcast(
    a: SNK[N, K], v: np.ndarray[tuple[K], np.dtype[np.float64]]
) -> SNK[N, K]:
    return a + v


def symbolic_einsum(
    a: np.ndarray[tuple[N, K], np.dtype[np.float64]],
    b: np.ndarray[tuple[K, M], np.dtype[np.float64]],
) -> np.ndarray[tuple[N, M], np.dtype[np.float64]]:
    return np.einsum("ij,jh->ih", a, b)


def symbolic_solve(
    a: np.ndarray[tuple[N, N], np.dtype[np.float64]],
    b: np.ndarray[tuple[N], np.dtype[np.float64]],
) -> np.ndarray[tuple[N], np.dtype[np.float64]]:
    return np.linalg.solve(a, b)


def sym_slice_keep(a: SNK[N, K]) -> SNK[N, K]:
    return a[:, :]


def sym_slice_fancy(
    a: SNK[N, K], idx: np.ndarray[tuple[N], np.dtype[np.int64]]
) -> SNK[N, K]:
    return a[idx, :]


def sym_take_broadcast(
    a: SNKC[N, K, C], idx: np.ndarray[tuple[N, K], np.dtype[np.signedinteger[Any]]]
) -> SNKC[N, K, C]:
    return np.take_along_axis(a, idx[..., None], axis=1)


def sym_callsite_ok(a: SNK[N, K], b: SNK[N, K]) -> SNK[N, K]:
    return a


def sym_and(a: SNKb[N, K], b: SNKb[N, K]) -> SNKb[N, K]:
    return a & b


def sym_or(a: SNKb[N, K], b: SNKb[N, K]) -> SNKb[N, K]:
    return a | b


def sym_reshape(a: SN102[N]) -> SN20[N]:
    return a.reshape(a.shape[0], 20)


def sym_reshape_tup(a: SN102[N]) -> SN20[N]:
    return a.reshape((a.shape[0], 20))


def sym_reshape_m1(a: SN102[N]) -> SN20[N]:
    return a.reshape(a.shape[0], -1)


def sym_len(a: SNK[N, K]) -> np.ndarray[tuple[N], np.dtype[np.signedinteger[Any]]]:
    return np.arange(len(a))


def sym_stack_enum(d: dict[Slots, SNK[N, K]]) -> np.ndarray[tuple[N, K, Literal[3]], np.dtype[np.float64]]:
    return np.stack([d[c] for c in Slots], axis=-1)


def symbolic_calls_stay_precise() -> None:
    use_5x10x2(pack_like(np.zeros((5, 10, 2)), np.zeros((5, 10)) > 1.0))
    use_2x3(sym_slice_keep(np.zeros((2, 3))))
    use_2x3(sym_slice_fancy(np.zeros((2, 3)), np.arange(2)))
    use_5x10x2(sym_take_broadcast(np.zeros((5, 10, 2)), np.arange(50).reshape(5, 10)))
    use_2x3(sym_callsite_ok(np.zeros((2, 3)), np.zeros((2, 3))))
    use_2x3(sym_callsite_ok(b=np.zeros((2, 3)), a=np.zeros((2, 3))))
    use_2x3(sym_callsite_ok(*[np.zeros((2, 3)), np.zeros((2, 3))]))
    use_b2x3(sym_and(np.zeros((2, 3), dtype=bool), np.zeros((2, 3), dtype=bool)))
    use_b2x3(sym_or(np.zeros((2, 3), dtype=bool), np.zeros((2, 3), dtype=bool)))
    use_b3(np.zeros((2, 3), dtype=bool).any(axis=0))
    use_b1x3(np.zeros((2, 3), dtype=bool).all(axis=0, keepdims=True))
    use_5x20(sym_reshape(np.zeros((5, 10, 2))))
    use_5x20(sym_reshape_tup(np.zeros((5, 10, 2))))
    use_5x20(sym_reshape_m1(np.zeros((5, 10, 2))))
    use_i2(sym_len(np.zeros((2, 3))))
    use_3(np.zeros(len(Slots)))
    use_2x3(np.zeros((2, len(Slots))))
    use_2x3(np.zeros(len(Slots) * 2).reshape(2, 3))
    use_2x3x3(sym_stack_enum({c: np.zeros((2, 3)) for c in Slots}))
    use_i3(np.arange(len(Slots)))
    use_2x4(symbolic_matmul(np.zeros((2, 3)), np.zeros((3, 4))))
    use_2x3(symbolic_broadcast(np.zeros((2, 3)), np.zeros(3)))
    use_2x4(symbolic_einsum(np.zeros((2, 3)), np.zeros((3, 4))))
    use_4(symbolic_solve(np.zeros((4, 4)), np.zeros(4)))


def indexing_stays_precise() -> None:
    a = np.zeros((5, 10, 2))
    # In-bounds ids: 0..14 would raise on axis 1 (size 10) at runtime.
    ids = np.arange(15).reshape(5, 3) % 10
    use_5x3x2(np.take_along_axis(a, ids[..., None], axis=1))
    use_5(np.take_along_axis(np.zeros(10), np.arange(5), axis=None))
    order = np.argsort(~(np.zeros((5, 10)) > 1.0), axis=1)
    use_i5x10(order)
    b = np.zeros((5, 22))
    actor = np.arange(5)
    use_5(b[np.arange(5), actor])
    use_5x2(b[actor, 1:3])
    use_5x5(b[:, actor])
    table = np.zeros((7, 4))
    use_5x4(table[np.arange(5)])
    use_5x10x2(a[ids[:, 0]])
    use_5x3x10x2(a[ids])


def enum_index_stays_precise() -> None:
    table = np.zeros((5, 4))
    reveal_type(table[..., Slots.X])
    reveal_type(table[:, COL])
    use_5(table[..., Slots.X])
    use_5x2(table[:, COL])
    use_4(table[Slots.VX - 1, :])
    use_5(np.zeros((5, 4))[np.arange(5), Slots.Y - 1])


def tail2_sq(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts]], np.dtype[np.float64]]:  # type: ignore[type-var]
    """Sum of squares over a trailing pair, at any rank."""
    return v[..., 0] ** 2 + v[..., 1] ** 2


def tail2_present(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts]], np.dtype[np.bool_]]:  # type: ignore[type-var]
    """Both trailing-pair entries finite, at any rank."""
    return np.asarray(np.isfinite(v).all(axis=-1), dtype=np.bool_)


def ts_poly_stays_precise() -> None:
    reveal_type(tail2_sq(np.zeros((2, 3, 2))))
    reveal_type(tail2_sq(np.zeros((3, 2))))
    use_2x3(tail2_sq(np.zeros((2, 3, 2))))
    use_3(tail2_sq(np.zeros((3, 2))))
    use_2x3x2(tail2_sq(np.zeros((2, 3, 2, 2))) + tail2_sq(np.zeros((2, 3, 2, 2))))
    use_b2x3(tail2_present(np.zeros((2, 3, 2))))


def ts_bail_stays_silent(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    _ = v + np.zeros((2, 3))  # unprovable against *Ts: silent, not an error
    _ = v.sum(axis=0)  # empty prefix: silent
    _ = v[0]  # no ellipsis to align with: silent


Bool2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.bool_]]
Bool2x3x2 = np.ndarray[tuple[Literal[2], Literal[3], Literal[2]], np.dtype[np.bool_]]


def union_body_stays_precise(
    cond: Bool2x2 | Bool2x3x2,
    v: A2x2 | A2x3x2,
) -> None:
    reveal_type(np.where(cond, v, 0.0))
    reveal_type(np.isfinite(v))
    reveal_type(np.sqrt(v))
    reveal_type(v == 0.0)
    reveal_type(np.clip(v, 0.0, 1.0))
    reveal_type(v.shape)


def union_genexp_stays_precise(a: A2x2, b: A2x3x2) -> None:
    out = [np.where(np.isfinite(x), x, 0.0) for x in (a, b)]
    reveal_type(out)


def unpack_display_stays_precise() -> None:
    a, b = (np.zeros((2, 2)), np.zeros((2, 3, 2)))
    reveal_type(a)
    reveal_type(b)
    use_2x2(a)
    use_2x3x2(b)


def unpack_genexp_stays_union(a: A2x2, b: A2x3x2) -> None:
    x, y = (np.where(np.isfinite(v), v, 0.0) for v in (a, b))
    reveal_type(x)


def named_dtype_stays_precise() -> None:
    use_f2x3(np.zeros((2, 3), dtype=np.float32))
    use_i2x3(np.zeros((2, 3), dtype=np.int64))
    use_f2x3(np.asarray(np.zeros((2, 3)), dtype=np.float32))
    use_f2x3(np.zeros((2, 3)).astype(np.float32))
    use_2(np.array([105.0, 68.0]))
    use_i3(np.array([1, 2, 3]))
    use_f2x3(-np.zeros((2, 3)))
    use_i5(np.arange(len(np.zeros(5))))
    use_2x3(np.broadcast_to(np.zeros(3), np.zeros((2, 3)).shape))


def ufunc_stays_precise() -> None:
    a = np.zeros((2, 3))
    b = np.zeros(3)
    use_2x3(np.sqrt(a))
    use_2x3(np.exp(a))
    use_2x3(np.sin(a))
    use_2x3(np.abs(a))
    use_2x3(np.floor(a))
    use_2x3(np.add(a, b))
    use_2x3(np.subtract(a, b))
    use_2x3(a - b)
    use_2x3(a / b)
    use_2x3(a**2)
    use_2x3(np.power(a, 2))
    use_2x3(np.where(a > 0, a, 0.0))
    use_2x3(np.clip(a, 0.0, 1.0))
    use_2x3(a.clip(0.0, 1.0))
    use_2x3(np.broadcast_to(b, (2, 3)))
    use_2x3(np.asarray(a))
    use_b2x3(np.isfinite(a))
    use_b2x3(np.equal(a, b))
    use_2x4(np.matmul(np.zeros((2, 3)), np.zeros((3, 4))))
    ints = np.arange(6).reshape(2, 3)
    use_2x3(np.sqrt(ints))
    use_2x3(np.true_divide(ints, ints))
    use_i2x3(ints - ints)


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


A2x4x6 = FloatArray[tuple[Literal[2], Literal[4], Literal[6]]]


def use_2x4x6(m: A2x4x6) -> None: ...


def ts_matmul(
    a: FlexArray[tuple[Unpack[Ts], Literal[4], Literal[5]], np.dtype[np.float64]],  # type: ignore[type-var]
    b: FlexArray[tuple[Unpack[Ts], Literal[5], Literal[6]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts], Literal[4], Literal[6]], np.dtype[np.float64]]:  # type: ignore[type-var]
    return np.matmul(a, b)


def ts_matmul_mixed(
    a: FlexArray[tuple[Unpack[Ts], Literal[4], Literal[5]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts], Literal[4], Literal[6]], np.dtype[np.float64]]:  # type: ignore[type-var]
    return np.matmul(a, np.zeros((5, 6)))


def ts_matmul_calls_stay_precise() -> None:
    reveal_type(ts_matmul(np.zeros((2, 4, 5)), np.zeros((2, 5, 6))))
    use_2x4x6(ts_matmul(np.zeros((2, 4, 5)), np.zeros((2, 5, 6))))
    reveal_type(ts_matmul_mixed(np.zeros((2, 4, 5))))
    use_2x4x6(ts_matmul_mixed(np.zeros((2, 4, 5))))


def ts_transpose_bails_stay_silent(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    # Reversing would also reverse *Ts itself, which no annotation can
    # spell, so every *Ts transpose bails out instead of guessing
    # (issue #48): the default full reversal, np.transpose, .T, an
    # explicit axis list, and transpose(None) all stay on the stubs.
    _ = v.transpose()
    _ = np.transpose(v)
    _ = v.T
    _ = v.transpose(None)


def ts_transpose_bails_explicit_axes(
    v: FlexArray[tuple[Unpack[Ts], Literal[2], Literal[3]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    # Explicit axes would pin the rank but cannot order the unknown
    # block, so the hook leaves them alone.
    reveal_type(v.transpose(1, 0))


def ts_transpose_prefix_bails_stay_silent(
    v: FlexArray[tuple[Literal[5], Unpack[Ts], Literal[4], Literal[5]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    _ = v.transpose()
    _ = v.T


def ts_reshape_stays_precise(
    v: FlexArray[tuple[Unpack[Ts], Literal[2], Literal[3]], np.dtype[np.float64]],  # type: ignore[type-var]
    w: FlexArray[tuple[Literal[5], Unpack[Ts], Literal[2], Literal[3]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    # Without star args the target is the whole new shape, whatever the
    # input: *Ts is never put back in (issue #48).
    reveal_type(v.reshape(6))
    reveal_type(v.reshape((2, 3)))
    reveal_type(v.reshape(5, 6))
    reveal_type(w.reshape((w.shape[0], 6)))
    reveal_type(w.reshape((5, 6)))


def ts_reshape_bails_stay_silent(
    v: FlexArray[tuple[Unpack[Ts], Literal[2], Literal[3]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    # A -1 would absorb the unknown *Ts product, so the hook leaves it
    # alone. (A `*v.shape[:-2]` star splice would carry *Ts through, but
    # the numpy stubs reject variadic star args before hooks run, so
    # that spelling stays a stub overload error.)
    _ = v.reshape(-1, 6)


def ts_head(
    v: FlexArray[tuple[Literal[5], Unpack[Ts], Literal[7]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts], Literal[7]], np.dtype[np.float64]]:  # type: ignore[type-var]
    return v[0]


def ts_pairwise_stays_precise(
    teammates: FlexArray[tuple[Literal[5], Unpack[Ts], Literal[7]], np.dtype[np.float64]],  # type: ignore[type-var]
    opponents: FlexArray[tuple[Literal[5], Unpack[Ts], Literal[7]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    # Rank-generic pairwise differences spell the ellipsis; without one
    # the middle indices would reach into *Ts, which stays silent.
    reveal_type(teammates[:, ..., None, :] - opponents[:, ..., None, :])
    reveal_type(teammates[0])
    use_2x7(ts_head(np.zeros((5, 2, 7))))


def ts_pairwise_bails_stay_silent(
    teammates: FlexArray[tuple[Literal[5], Unpack[Ts], Literal[7]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    _ = teammates[:, 0]
    _ = teammates[:, :, None, :]


def ts_prefix_pairwise_stays_precise(
    teammates: FlexArray[tuple[Literal[5], Literal[6], Literal[7], Unpack[Ts], Literal[8]], np.dtype[np.float64]],  # type: ignore[type-var]
    opponents: FlexArray[tuple[Literal[5], Literal[6], Literal[7], Unpack[Ts], Literal[8]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    # With the leading dims known, the non-ellipsis spelling resolves
    # purely in the prefix — the classic pairwise-difference body.
    reveal_type(teammates[:, :, None, :] - opponents[:, None, :, :])
