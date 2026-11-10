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
            "tuple[Literal[6]]",
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


LIBMOD146 = '''\
from numty import F32


def dup(x: F32["myhid myrest"]) -> F32["2*myhid myrest"]:
    return x  # type: ignore[return-value]


def inc(x: F32["N M"]) -> F32["N+1 M"]:
    return x  # type: ignore[return-value]
'''

USEMOD146 = '''\
from libmod import dup, inc
from numty import F32


def caller(a: F32["myhid myrest"]):
    y = dup(a)
    reveal_type(y)


def caller2(b: F32["N M"]):
    z = inc(b)
    reveal_type(z)
'''

SAMEFILE146 = '''\
from numty import F32


def f1(x: F32["myhid myrest"]) -> F32["2*myhid myrest"]:
    return x  # type: ignore[return-value]


def f2(x: F32["myhid myrest"]) -> F32["2*myhid myrest"]:
    return x  # type: ignore[return-value]


def use(a: F32["myhid myrest"]):
    y = f1(a)
    z = f2(a)
    reveal_type(y)
    reveal_type(z)
'''


def _run_mypy_files(
    cwd: Path, cache: Path, *files: str
) -> "subprocess.CompletedProcess[str]":
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
            *files,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_implicit_expr_cold_warm_parity() -> None:
    """Cold/warm/dmypy parity for implicit expr dims (issue #146)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "libmod.py").write_text(LIBMOD146)
        (work / "usemod.py").write_text(USEMOD146)
        cache = work / "cache"

        cold = _run_mypy_files(work, cache, "usemod.py", "libmod.py")
        with open(work / "usemod.py", "a") as fh:
            fh.write("# edit\n")
        warm = _run_mypy_files(work, cache, "usemod.py", "libmod.py")

        assert cold.stdout == warm.stdout, (
            f"--- cold ---\n{cold.stdout}\n--- warm ---\n{warm.stdout}"
        )
        # Cold used to bail to var-annotated + Any; both must now be
        # precise with no annotation error.
        assert 'Need type annotation for "y"' not in cold.stdout, cold.stdout
        assert 'Need type annotation for "z"' not in cold.stdout, cold.stdout
        assert "tuple[Any, Any]" not in cold.stdout, cold.stdout
        for shape in ("tuple[2*myhid, myrest]", "tuple[N+1, M]"):
            assert shape in cold.stdout, cold.stdout


def test_implicit_two_func_same_file_parity() -> None:
    """Two funcs sharing one implicit spec agree cold and warm (issue #146)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "both.py").write_text(SAMEFILE146)
        cache = work / "cache"

        cold = _run_mypy_files(work, cache, "both.py")
        with open(work / "both.py", "a") as fh:
            fh.write("# edit\n")
        warm = _run_mypy_files(work, cache, "both.py")

        assert cold.stdout == warm.stdout, (
            f"--- cold ---\n{cold.stdout}\n--- warm ---\n{warm.stdout}"
        )
        assert 'Need type annotation for "y"' not in cold.stdout, cold.stdout
        assert 'Need type annotation for "z"' not in cold.stdout, cold.stdout
        assert "tuple[Any, Any]" not in cold.stdout, cold.stdout
        assert cold.stdout.count("tuple[2*myhid, myrest]") == 2, cold.stdout
