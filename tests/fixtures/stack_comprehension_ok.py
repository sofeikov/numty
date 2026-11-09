"""List comprehensions whose items share one shape solve (#30).

A comprehension has no statically known length, but every item has the
element expression's shape: the joined axis is a fresh unknown and
every other dim and the dtype ride the item (filters, nested loops,
element transforms and attribute reads included). Runtime prints
numpy's shape and dtype for each revealed line.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from numty import F32

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape, x.dtype)
        return x


@dataclass
class Transition:
    agent_valid: F32["n_agents", 2]


def assemble(
    items: list[F32["T n_agents", 2]],
    transitions: list[Transition],
    seq: Sequence[F32[3, 2]],
    rows: list[list[F32[3]]],
    maybe: list[Optional[F32[3]]],
) -> None:
    reveal_type(np.stack([a for a in items]))
    reveal_type(np.stack([a for a in items], axis=-1))
    reveal_type(np.stack([a * 2 for a in items], axis=1))
    reveal_type(np.stack([a for a in seq]))
    reveal_type(np.stack([t.agent_valid for t in transitions]))
    reveal_type(np.stack([np.stack([a for a in r]) for r in rows]))
    reveal_type(np.stack([a for r in rows for a in r]))
    reveal_type(np.stack([a for a in maybe if a is not None]))
    reveal_type(np.stack([a.astype(np.float64) for a in seq]))
    reveal_type(np.concatenate([a for a in items], axis=1))
    reveal_type(np.vstack([a for r in rows for a in r]))
    reveal_type(np.hstack([a for a in seq]))
    reveal_type(np.column_stack([a for r in rows for a in r]))
    reveal_type(np.dstack([a for a in seq]))


def demo():  # unannotated: the runtime data is not what is checked
    f32 = np.float32
    assemble(
        [np.zeros((5, 4, 2), f32) for _ in range(3)],
        [Transition(np.zeros((4, 2), f32)) for _ in range(6)],
        [np.zeros((3, 2), f32) for _ in range(7)],
        [[np.zeros(3, f32) for _ in range(2)] for _ in range(8)],
        [np.zeros(3, f32), None, np.zeros(3, f32)],
    )


if __name__ == "__main__":
    demo()
