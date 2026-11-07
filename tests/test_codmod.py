"""Tests for numty_codmod: opt-in pre-pass rewrites.

Unit tests pin exact rewritten source (rewrites are spliced into the
original text, so everything else comes back byte for byte); end-to-end
tests prove the precision gain by running mypy on the rewritten output.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from numty_codmod import codemod_source

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".mypy-codmod-cache"



def _non_reports(warnings: list[str]) -> list[str]:
    """Warnings other than the per-site 'self-ops rewrote' reports."""
    return [w for w in warnings if "self-ops rewrote" not in w]

def codemod(src: str, **flags: bool) -> tuple[str, list[str]]:
    out, warnings = codemod_source(src, **flags)
    return out.rstrip("\n"), warnings


def test_self_op_add() -> None:
    out, warnings = codemod(
        "import numpy as np\nx = np.zeros(3)\ny = x + x\n", self_ops=True
    )
    assert _non_reports(warnings) == []
    assert out == "\n".join(
        [
            "import numpy as np",
            "x = np.zeros(3)",
            "y = np.add(x, x)",
        ]
    )


def test_self_op_family() -> None:
    out, warnings = codemod(
        "import numpy as np\n"
        "x = np.ones(3)\n"
        "a = x - x\n"
        "b = x * x\n"
        "c = x / x\n"
        "d = x ** x\n",
        self_ops=True,
    )
    assert _non_reports(warnings) == []
    assert out == "\n".join(
        [
            "import numpy as np",
            "x = np.ones(3)",
            "a = np.subtract(x, x)",
            "b = np.multiply(x, x)",
            "c = np.divide(x, x)",
            "d = np.power(x, x)",
        ]
    )


def test_self_op_respects_numpy_alias() -> None:
    out, warnings = codemod(
        "import numpy as npy\nx = npy.zeros(3)\ny = x + x\n", self_ops=True
    )
    assert _non_reports(warnings) == []
    assert "y = npy.add(x, x)" in out.splitlines()


def test_self_op_leaves_others_alone() -> None:
    src = (
        "import numpy as np\n"
        "a = x + y\n"
        "x += x\n"
        "b = -x\n"
        "c = x @ x\n"
    )
    out, warnings = codemod(src, self_ops=True)
    assert warnings == []
    assert out == "\n".join(
        [
            "import numpy as np",
            "a = x + y",
            "x += x",
            "b = -x",
            "c = x @ x",
        ]
    )


def test_self_op_without_numpy_warns() -> None:
    out, warnings = codemod("y = x + x\n", self_ops=True)
    assert warnings == ["self-ops skipped: no `import numpy [as ...]` found"]
    assert out == "y = x + x"


# Every operand here is a plain Python object: the operator means
# concatenation, unbounded ints, bool -> int, or ZeroDivisionError, none
# of which the ufunc reproduces.
NON_ARRAYS = """import numpy as np


def cells(n: int) -> int:
    return n * n  # grid size, plain Python int


lst = [1, 2]
doubled = lst + lst  # list concatenation
t = (1, 2)
words = ["a", "b"]
s = "ab"
flag = True
z = 0
k = -2
m = 20
p = 3
print("lst + lst:", doubled)
print("t + t:", t + t, "words:", words + words, "s + s:", repr(s + s))
print("flag + flag:", repr(flag + flag))
print("cells(4_000_000_000):", cells(4_000_000_000))
print("k ** k:", k ** k, "m ** m:", m ** m, "p ** p:", repr(p ** p))
print("lambda:", (lambda: lst + lst)())
for op in ("lst * lst", "lst - lst", "z / z"):
    try:
        print(op, "->", eval(op))
    except (TypeError, ZeroDivisionError) as e:
        print(op, "->", type(e).__name__)
try:
    print("z / z:", z / z)
except ZeroDivisionError as e:
    print("z / z: ZeroDivisionError:", e)
"""


def _run_python(src: str, tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.write_text(src)
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_self_op_skips_non_arrays(tmp_path: Path) -> None:
    out, warnings = codemod_source(NON_ARRAYS, self_ops=True)
    assert out == NON_ARRAYS
    skipped = [w.split(": ", 1)[1] for w in warnings]
    assert skipped == [
        f"self-ops skipped: {name} is not known to be an array"
        for name in ["n", "lst", "t", "words", "s", "flag", "k", "m", "p", "lst", "z"]
    ]
    assert _run_python(out, tmp_path, "after.py") == _run_python(
        NON_ARRAYS, tmp_path, "before.py"
    )


def test_self_op_array_bindings_still_rewrite(tmp_path: Path) -> None:
    src = (
        "from typing import Literal, TypeAlias\n"
        "import numpy as np\n"
        "import numpy.typing as npt\n"
        "from numty import F64, FlexArray\n"
        "\n"
        "Vec = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]\n"
        "Alias: TypeAlias = Vec\n"
        "a = np.zeros(3)\n"
        "b = np.array([1, 2], dtype=np.int64)\n"
        "b += 1\n"
        "\n"
        "def f(x: F64[3]) -> F64[3]:\n"
        "    return x + x\n"
        "\n"
        "def g(x: npt.NDArray[np.float64], y: 'np.ndarray', z: Alias | Vec):\n"
        "    return x + x, y - y, z + z, [a - a for _ in range(1)]\n"
        "\n"
        "def h(v: FlexArray[tuple[int], np.dtype[np.float64]]):\n"
        "    w: np.ndarray = v\n"
        "    return v + v, w + w, b + b\n"
        "\n"
        "print(f(a), g(a, a, a), h(a))\n"
    )
    out, warnings = codemod_source(src, self_ops=True)
    assert _non_reports(warnings) == []
    for rewritten in [
        "return np.add(x, x)",
        "return np.add(x, x), np.subtract(y, y), np.add(z, z),"
        " [np.subtract(a, a) for _ in range(1)]",
        "return np.add(v, v), np.add(w, w), np.add(b, b)",
    ]:
        assert rewritten in out
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    runs = []
    for name, text in (("before.py", src), ("after.py", out)):
        (tmp_path / name).write_text(text)
        runs.append(
            subprocess.run(
                [sys.executable, str(tmp_path / name)],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        )
    assert runs[0].returncode == 0, runs[0].stderr
    assert runs[1].stdout == runs[0].stdout


@pytest.mark.parametrize(
    "src",
    [
        # Rebound to a non-array in the same scope (flow-insensitive).
        "x = np.zeros(3)\nx = [1, 2]\ny = x + x\n",
        # Local parameter shadows the module array.
        "x = np.zeros(3)\ndef f(x):\n    return x + x\n",
        "x = np.zeros(3)\ndef f(x: int):\n    return x + x\n",
        # A `global` rebinding elsewhere counts against the module name.
        "x = np.zeros(3)\ndef f():\n    global x\n    x = 4\n"
        "def g():\n    return x + x\n",
        # `nonlocal` rebinding from a nested function.
        "def f():\n    x = np.zeros(3)\n    def g():\n        nonlocal x\n"
        "        x = 4\n    g()\n    return x + x\n",
        # Comprehension / lambda / loop / unpacking / with targets.
        "xs = [np.zeros(3)]\ny = [x + x for x in xs]\n",
        "f = lambda x: x + x\n",
        "for x in [np.zeros(3)]:\n    y = x + x\n",
        "x, y = np.zeros(3), np.zeros(3)\nz = x + x\n",
        "x = np.zeros(3)\nwith open('f') as x:\n    pass\ny = x + x\n",
        # Class-body names are invisible to methods.
        "class C:\n    x = np.zeros(3)\n    def m(self):\n        return x + x\n",
        # Constructors that can return a subclass or a tuple.
        "m = np.matrix([[1]])\nx = np.zeros_like(m)\ny = x * x\n",
        "x = np.linspace(0, 1, 5, retstep=True)\ny = x + x\n",
        "x = np.asarray([1], like=None)\ny = x + x\n",
        # Builtins, unbound names, imports, plain calls.
        "y = x + x\n",
        "from mod import x\ny = x + x\n",
        "x = make()\ny = x + x\n",
        "x = np.sum(np.zeros(3))\ny = x + x\n",
        # Annotations that admit non-arrays.
        "def f(x: np.ndarray | int):\n    return x + x\n",
        "def f(*x: np.ndarray):\n    return x + x\n",
        # linspace's 5th positional is retstep: a tuple comes back.
        "j = np.linspace(0, 1, 3, True, True)\nprint(j + j)\n",
        # Splatted positionals may carry retstep too.
        "args = (0, 1, 3, True, True)\nm = np.linspace(*args)\nprint(m + m)\n",
        # A star import may rebind an earlier array name.
        "a = np.zeros(2)\nfrom starmod import *\nprint(a + a)\n",
    ],
)
def test_self_op_unsure_sites_left_alone(src: str) -> None:
    src = "import numpy as np\n" + src
    out, warnings = codemod_source(src, self_ops=True)
    assert out == src
    assert len(warnings) == 1
    assert warnings[0].endswith("is not known to be an array")


def test_self_op_numpy_alias_shadowed() -> None:
    src = (
        "import numpy as np\n"
        "def f(x: np.ndarray, np: object):\n"
        "    return x + x\n"
    )
    out, warnings = codemod_source(src, self_ops=True)
    assert out == src
    assert warnings == ["line 3: self-ops skipped: np does not mean numpy here"]


def test_unpack_genexp_to_display() -> None:
    out, warnings = codemod(
        "a, b = (x for x in (p1, p2))\n", unpack_displays=True
    )
    assert warnings == []
    assert out == "a, b = (p1, p2)"


def test_unpack_listcomp_keeps_list() -> None:
    out, warnings = codemod(
        "a, b = [x for x in [p1, p2]]\n", unpack_displays=True
    )
    assert warnings == []
    assert out == "a, b = [p1, p2]"


def test_unpack_tuple_call_and_starred() -> None:
    out, warnings = codemod(
        "a, b = tuple(x for x in (p1, p2))\n"
        "c, d = tuple([p1, p2])\n"
        "e, f = (*[p1, p2],)\n",
        unpack_displays=True,
    )
    assert warnings == []
    assert out == "\n".join(
        [
            "a, b = (p1, p2)",
            "c, d = (p1, p2)",
            "e, f = (p1, p2)",
        ]
    )


def test_unpack_leaves_risky_forms_alone() -> None:
    src = (
        "a, b = (f(x) for x in (p1, p2))\n"
        "c, d = [x for x in t if c]\n"
        "e, f = (x for x in t)\n"
        "g, h = map(f, (p1, p2))\n"
        "i = (x for x in (p1, p2))\n"
    )
    out, warnings = codemod(src, unpack_displays=True)
    assert warnings == []
    assert out == "\n".join(
        [
            "a, b = (f(x) for x in (p1, p2))",
            "c, d = [x for x in t if c]",
            "e, f = (x for x in t)",
            "g, h = map(f, (p1, p2))",
            "i = (x for x in (p1, p2))",
        ]
    )


def _run_mypy(path: Path) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(CACHE), str(path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_self_op_end_to_end_pairs_union(tmp_path: Path) -> None:
    src = (
        "from typing import Literal\n"
        "import numpy as np\n"
        "A2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float64]]\n"
        "A2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float64]]\n"
        "def f(x: A2x2 | A2x3) -> None:\n"
        "    reveal_type(x + x)\n"
    )
    out, _ = codemod_source(src, self_ops=True)
    assert "np.add(x, x)" in out
    probe = tmp_path / "probe_selfop.py"
    probe.write_text(out)
    result = _run_mypy(probe)
    assert result.returncode == 0, result.stdout
    assert "tuple[Literal[2], Literal[2]]" in result.stdout
    assert "tuple[Literal[2], Literal[3]]" in result.stdout
    assert "tuple[Any, ...]" not in result.stdout


def test_unpack_end_to_end_distributes(tmp_path: Path) -> None:
    src = (
        "from typing import Literal\n"
        "import numpy as np\n"
        "A2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float64]]\n"
        "A2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float64]]\n"
        "def f(p1: A2x3, p2: A2x2) -> None:\n"
        "    a, b = (x for x in (p1, p2))\n"
        "    reveal_type(a)\n"
        "    reveal_type(b)\n"
    )
    out, _ = codemod_source(src, unpack_displays=True)
    assert "a, b = (p1, p2)" in out
    probe = tmp_path / "probe_unpack.py"
    probe.write_text(out)
    result = _run_mypy(probe)
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 2, result.stdout
    assert "tuple[Literal[2], Literal[3]]" in notes[0], result.stdout
    assert "tuple[Literal[2], Literal[2]]" in notes[1], result.stdout


README_NORM = '''# SPDX-License-Identifier: MIT
from typing import Literal
from typing_extensions import TypeVarTuple, Unpack
import numpy as np
from numty import FlexArray

Ts = TypeVarTuple("Ts")


def squared_norm(  # README "One signature, every rank", verbatim
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts]], np.dtype[np.float64]]:  # type: ignore[type-var]
    return v[..., 0] ** 2 + v[..., 1] ** 2


print(squared_norm(np.zeros((5, 7, 2))).shape)
'''


@pytest.mark.parametrize(
    "src",
    [
        README_NORM,
        README_NORM.replace("\n", "\r\n"),
        README_NORM.rstrip("\n"),
    ],
    ids=["readme", "crlf", "no-final-newline"],
)
@pytest.mark.parametrize(
    "flags",
    [{}, {"self_ops": True, "unpack_displays": True}],
    ids=["no-flags", "all-flags"],
)
def test_nothing_to_rewrite_is_identity(src: str, flags: dict[str, bool]) -> None:
    out, warnings = codemod_source(src, **flags)
    assert warnings == []
    assert out == src


def test_rewrites_keep_surrounding_text() -> None:
    src = (
        "# license header\n"
        "import numpy as np\n"
        "\n"
        "\n"
        "x = np.zeros(3)\n"
        "y = x + x  # keep me\n"
        "a, b = (  # pair\n"
        "    v for v in (x, y)\n"
        ")  # type: ignore[misc]\n"
        "c, = tuple([x  # only\n"
        "])\n"
    )
    out, warnings = codemod_source(src, self_ops=True, unpack_displays=True)
    assert _non_reports(warnings) == []
    assert out == (
        "# license header\n"
        "import numpy as np\n"
        "\n"
        "\n"
        "x = np.zeros(3)\n"
        "y = np.add(x, x)  # keep me\n"
        "a, b = (x, y)  # type: ignore[misc]\n"
        "c, = (x,  # only\n"
        ")\n"
    )


def test_readme_norm_still_checks_after_codmod(tmp_path: Path) -> None:
    out, _ = codemod_source(README_NORM, self_ops=True, unpack_displays=True)
    probe = tmp_path / "probe_norm.py"
    probe.write_text(out)
    result = _run_mypy(probe)
    assert result.returncode == 0, result.stdout


def _cli(*args: str, cwd: Path) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, str(ROOT / "numty_codmod.py"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("args", [["-h"], ["--help"], ["--help", "ok.py"]])
def test_cli_help(args: list[str], tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("y = 1  # keep me\n")
    result = _cli(*args, cwd=tmp_path)
    assert result.returncode == 0
    assert result.stdout.startswith("usage: numty_codmod.py")


def test_cli_errors_are_one_line(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def f(:\n    pass\n")
    missing = _cli("--self-ops", "missing.py", cwd=tmp_path)
    assert missing.returncode == 2
    assert missing.stderr == (
        "numty_codmod.py: error: missing.py: No such file or directory\n"
    )
    broken = _cli("broken.py", cwd=tmp_path)
    assert broken.returncode == 2
    assert broken.stderr == (
        "numty_codmod.py: error: broken.py:1: syntax error: invalid syntax\n"
    )
    bogus = _cli("--bogus", "broken.py", cwd=tmp_path)
    assert bogus.returncode == 2
    assert "unrecognized arguments: --bogus" in bogus.stderr


def test_cli_no_flags_writes_bytes_unchanged(tmp_path: Path) -> None:
    src = README_NORM.replace("\n", "\r\n").encode()
    (tmp_path / "norm.py").write_bytes(src)
    result = _cli("norm.py", "out.py", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out.py").read_bytes() == src


def test_self_op_skips_fstrings() -> None:
    # f"{x * x=}" prints its own source text, so it must stay as written.
    src = 'import numpy as np\nx = np.zeros(2)\nprint(f"{x * x=}")\n'
    out, warnings = codemod_source(src, self_ops=True)
    assert out == src
    assert warnings == []


def test_self_op_reports_rewritten_sites() -> None:
    src = "import numpy as np\nx = np.zeros(2)\ny = x * x\n"
    out, warnings = codemod_source(src, self_ops=True)
    assert out == "import numpy as np\nx = np.zeros(2)\ny = np.multiply(x, x)\n"
    assert warnings == ["line 3: self-ops rewrote to np.multiply(x, x)"]


def test_self_op_star_import_blocks_numpy_alias() -> None:
    src = (
        "import numpy as np\n"
        "from config import *\n"
        "def f(x: np.ndarray):\n"
        "    return x + x\n"
    )
    out, _ = codemod_source(src, self_ops=True)
    assert out == src


# Issue #79: an annotation admits ndarray subclasses, and np.matrix makes
# `*` a matrix product. The `*` site must stay; `+` (identical on numpy's
# own subclasses) is still rewritten, and its report says it trusted the
# annotation.
MATRIX_REPRO = """import numpy as np


def square(x: np.ndarray) -> np.ndarray:
    return x * x


def double(x: np.ndarray) -> np.ndarray:
    return x + x


mat = np.matrix([[1, 2], [3, 4]])
masked = np.ma.array([1.0, np.nan, 3.0], mask=[0, 0, 1])
print(square(mat).tolist(), type(square(mat)).__name__)
for arr in (mat, masked):
    d = double(arr)
    print(type(d).__name__, d.tolist(), np.ma.getmaskarray(d).tolist())
"""


def test_self_op_matrix_annotation_keeps_matmul(tmp_path: Path) -> None:
    out, warnings = codemod_source(MATRIX_REPRO, self_ops=True)
    assert out == MATRIX_REPRO.replace("return x + x", "return np.add(x, x)")
    assert warnings == [
        "line 5: self-ops skipped: x is only known to be an array from an"
        " annotation, which admits ndarray subclasses (np.matrix makes `*`"
        " a matrix product)",
        "line 9: self-ops rewrote to np.add(x, x) (trusting the annotation of x)",
    ]
    before = _run_python(MATRIX_REPRO, tmp_path, "before.py")
    assert before.splitlines()[0] == "[[7, 10], [15, 22]] matrix"
    assert _run_python(out, tmp_path, "after.py") == before


_ANNOTATION_HEADER = (
    "import numpy as np\n"
    "import numpy.typing as npt\n"
    "from numty import F64, FlexArray\n"
)
_OVERRIDES = {
    "*": "np.matrix makes `*` a matrix product",
    "**": "np.matrix makes `**` a matrix power",
    "/": "np.ma.MaskedArray masks nan results of `/` that np.divide keeps",
}


@pytest.mark.parametrize("op", list(_OVERRIDES))
@pytest.mark.parametrize(
    "body",
    [
        "def f(x: np.ndarray):\n    return x {op} x\n",
        "def f(x: npt.NDArray[np.float64]):\n    return x {op} x\n",
        # numty heads check only ndarray + dtype: a float64 np.matrix
        # passes isinstance and narrow, so they are annotations too.
        "def f(x: F64[2, 2]):\n    return x {op} x\n",
        "def f(x: FlexArray[tuple[int], np.dtype[np.float64]]):\n"
        "    return x {op} x\n",
        "def f(v):\n    x: np.ndarray = v\n    return x {op} x\n",
        "Arr = np.ndarray[tuple[int], np.dtype[np.float64]]\n"
        "def f(x: Arr):\n    return x {op} x\n",
        # One annotation-only binding is enough to keep the site.
        "def f(x: np.ndarray, reset: bool):\n"
        "    if reset:\n        x = np.zeros(3)\n    return x {op} x\n",
    ],
    ids=["ndarray", "NDArray", "F64", "FlexArray", "annassign", "alias", "mixed"],
)
def test_self_op_annotation_only_skips_subclass_ops(body: str, op: str) -> None:
    src = _ANNOTATION_HEADER + body.format(op=op)
    out, warnings = codemod_source(src, self_ops=True)
    assert out == src
    line = len(src.splitlines())
    assert warnings == [
        f"line {line}: self-ops skipped: x is only known to be an array from"
        f" an annotation, which admits ndarray subclasses ({_OVERRIDES[op]})"
    ]


def test_self_op_subclass_evidence(tmp_path: Path) -> None:
    # Why _OVERRIDES holds `*`, `**` and `/` but not `+` and `-`: each of
    # the three differs from its ufunc on a subclass numpy ships, while
    # `+` and `-` agree (values, mask, type) on all of them.
    src = """import warnings
import numpy as np
warnings.simplefilter("ignore")
mat = np.matrix([[1.0, 2.0], [3.0, 4.0]])
masked = np.ma.array([-1.5, np.nan, 2.0], mask=[0, 0, 1])
def same(a, b):
    return (type(a) is type(b)
        and np.ma.getmaskarray(a).tolist() == np.ma.getmaskarray(b).tolist()
        and np.array_equal(np.ma.filled(a, 0), np.ma.filled(b, 0),
                           equal_nan=a.dtype.kind == "f"))
def raises(f, x):
    try:
        f(x, x)
    except TypeError:
        return True
    return False
print(same(mat * mat, np.multiply(mat, mat)))
print(raises(lambda a, b: a ** b, mat), raises(np.power, mat))
print(same(masked / masked, np.divide(masked, masked)))
print(same(masked ** masked, np.power(masked, masked)))
for x in (mat, masked, np.char.array(["a", "bc"]),
          np.arange(3.0).view(np.recarray)):
    print(same(x + x, np.add(x, x)), end=" ")
    if x.dtype.kind == "f":
        print(same(x - x, np.subtract(x, x)), end=" ")
print()
"""
    assert _run_python(src, tmp_path, "evidence.py").splitlines() == [
        "False",
        "True False",
        "False",
        "False",
        "True True True True True True True ",
    ]


def test_self_op_constructor_bindings_rewrite_every_op(tmp_path: Path) -> None:
    # np.<constructor>(...) builds a base-class ndarray even from a
    # subclass, so every operator is the ufunc there.
    src = (
        "import numpy as np\n"
        "mat = np.matrix([[1.0, 2.0], [3.0, 4.0]])\n"
        "a = np.asarray(mat)\n"
        "b = np.array(np.ma.array([1.0, 0.0], mask=[0, 1]))\n"
        "\n"
        "def f():\n"
        "    c: np.ndarray = np.ones((2, 2))\n"
        "    c += 1\n"
        "    return (c * c).tolist(), (c ** c).tolist(), (c / c).tolist()\n"
        "\n"
        "print((a * a).tolist(), (a ** a).tolist(), (a / a).tolist())\n"
        "print((b / b).tolist(), type(b * b).__name__, f())\n"
    )
    out, warnings = codemod_source(src, self_ops=True)
    assert _non_reports(warnings) == []
    assert not any("trusting" in w for w in warnings)
    for rewritten in [
        "return (np.multiply(c, c)).tolist(), (np.power(c, c)).tolist(),"
        " (np.divide(c, c)).tolist()",
        "print((np.multiply(a, a)).tolist(), (np.power(a, a)).tolist(),"
        " (np.divide(a, a)).tolist())",
        "print((np.divide(b, b)).tolist(), type(np.multiply(b, b)).__name__",
    ]:
        assert rewritten in out
    before = _run_python(src, tmp_path, "before.py")
    assert before.startswith("[[1.0, 4.0], [9.0, 16.0]]")
    assert _run_python(out, tmp_path, "after.py") == before
