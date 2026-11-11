"""Issue #244: methods of a generic NamedTuple keep receiver TypeVars."""
from typing import Generic, NamedTuple, TypeVar

from numty import F64, I64

P = TypeVar("P", bound=int)
F = TypeVar("F", bound=int)
Q = TypeVar("Q", bound=int)
K = TypeVar("K", bound=int)


class Block(NamedTuple, Generic[P, F]):
    seconds: F64[F]
    player_x: F64[P, F]

    def same(self) -> "Block[P, F]":
        return self

    def take(self, idx: I64[K]) -> "Block[P, K]":
        raise NotImplementedError


def use(b: Block[P, F]) -> None: ...


def caller(a: Block[int, F], b: Block[Q, F], c: Block[int, int], i: I64[K]) -> None:
    reveal_type(a.same())
    use(a.same())
    use(b.same())
    use(c.same())
    reveal_type(b.take(i))
    use(b.take(i))


def keep_p(game: Block[P, F], i: I64[K]) -> Block[P, K]:
    return game.take(i)
