"""Incremental-cache parity: a warm re-check matches a cold run (issue #42)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LIBMOD = '''\
from typing import Generic, TypeVar

import numpy as np

from numty import F64, batch, hid, inp

n = TypeVar("n", bound=int)
m = TypeVar("m", bound=int)
N = TypeVar("N", bound=int)


def tr(x: F64["n m"]) -> F64["m n"]:
    return x.T


def lin(x: F64["batch inp"], w: F64["inp hid"]) -> F64["batch hid"]:
    return x @ w


def mm(x: F64["a b"], y: F64["b c"]) -> F64["a c"]:
    return np.matmul(x, y)


class Padded(Generic[N]):
    def __init__(self, v: F64[N]):
        self.p = np.pad(v, 1)
        self.q = np.pad(v, 1)
'''

USEMOD = '''\
from typing import TYPE_CHECKING

import numpy as np

from libmod import Padded, lin, mm, tr

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print("runtime:", getattr(x, "shape", type(x).__name__))
        return x

reveal_type(tr(np.zeros((2, 5))))
reveal_type(lin(np.zeros((5, 4)), np.zeros((4, 3))))
reveal_type(mm(np.zeros((2, 3)), np.zeros((3, 4))))
try:
    reveal_type(mm(np.zeros((2, 4)), np.zeros((5, 3))))
except ValueError:
    print("runtime: ValueError")
c = Padded(np.zeros(4))
reveal_type(c.p + c.q)
'''


def _run_mypy(cwd: Path, cache: Path) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MYPYPATH": str(ROOT)}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(ROOT / "mypy.ini"),
            "--cache-dir",
            str(cache),
            "usemod.py",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_warm_matches_cold() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "libmod.py").write_text(LIBMOD)
        (work / "usemod.py").write_text(USEMOD)
        cache = work / "cache"

        cold = _run_mypy(work, cache)
        # Touch only the caller: the callee is served from the cache.
        with open(work / "usemod.py", "a") as fh:
            fh.write("# edit\n")
        warm = _run_mypy(work, cache)

        assert cold.stdout == warm.stdout, (
            f"--- cold ---\n{cold.stdout}\n--- warm ---\n{warm.stdout}"
        )
        # The real shape bug is still reported (not lost), with no
        # cache-state arg-type noise.
        assert "call-site shape conflict for 'b'" in cold.stdout, cold.stdout
        assert "arg-type" not in cold.stdout, cold.stdout
        for shape in (
            "tuple[Literal[5], Literal[2]]",
            "tuple[Literal[5], Literal[3]]",
            "tuple[Literal[2], Literal[4]]",
            "tuple[Literal[2], Literal[3]]",
            "tuple[N+2]",
        ):
            assert shape in cold.stdout, cold.stdout

        run = subprocess.run(
            [sys.executable, "usemod.py"],
            cwd=work,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stderr
        assert run.stdout.strip().splitlines() == [
            "runtime: (5, 2)",
            "runtime: (5, 3)",
            "runtime: (2, 4)",
            "runtime: ValueError",
            "runtime: (6,)",
        ]
