"""Legal neighbours of the argument checks: none of these may error.

Run as a script, each solved shape is printed so the test can compare
it with numpy's own result.
"""
import numpy as np

from numty import F64


def solved(
    rot: F64[4, 3],
    rot_neg: F64[2, 4, 3],
    rolled: F64[2, 3, 4],
    flipped: F64[2, 3, 4],
    squeezed: F64[2, 3],
    rep_axis: F64[3, 3],
    rep_flat: F64[6],
    part: F64[3, 4],
    part_flat: F64[12],
    taken: F64[3],
    taken_list: F64[3, 2],
    conv: F64[3],
    corr: F64[1],
    big: F64[1099511627776, 0],
) -> None:
    print(
        *(
            x.shape
            for x in (
                rot,
                rot_neg,
                rolled,
                flipped,
                squeezed,
                rep_axis,
                rep_flat,
                part,
                part_flat,
                taken,
                taken_list,
                conv,
                corr,
                big,
            )
        ),
        sep="\n",
    )


def main() -> None:
    a = np.zeros((2, 3, 4))
    m = np.zeros((3, 4))
    v = np.zeros(3)
    solved(
        np.rot90(m, k=-1, axes=(0, 1)),
        np.rot90(a, 1, (1, -1)),
        # roll may name an axis twice, and a length-1 shift broadcasts.
        np.roll(a, (1,), axis=(0, 0, 2)),
        np.flip(a, (0, -1)),
        np.squeeze(np.zeros((2, 1, 3, 1)), axis=(1, -1)),
        np.repeat(np.zeros((2, 3)), [1, 2], axis=0),
        np.repeat(np.zeros((2, 3)), [1, 1, 1, 1, 1, 1]),
        np.partition(m, (0, 3, -4), axis=1),
        np.partition(m, 11, axis=None),
        np.take(m, -4, axis=1),
        np.take(m, [0, 3], axis=1),
        np.convolve(v, np.ones(1)),
        np.correlate(v, np.ones(3)),
        np.zeros((2**40, 0)),
    )
    # Unsolved results still type-check: only the arguments are checked.
    np.gradient(m, axis=(0, 1))
    np.repeat(np.zeros((2, 3)), [2], axis=0)
    np.argpartition(m, 3)
    m.argpartition(-4, axis=1)
    np.linalg.norm(a)
    np.linalg.norm(m, ord=2)
    np.linalg.norm(m, ord="fro")
    np.linalg.norm(a, ord=1, axis=0)
    np.linalg.norm(a, ord="nuc", axis=(0, 2))
    np.take(m, 12, mode="clip")
    m.take(9, axis=1, mode="wrap")
    np.matmul(m, np.zeros(4))
    np.sum(m, axis=(0, -1))


if __name__ == "__main__":
    main()
