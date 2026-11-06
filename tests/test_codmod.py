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


def codemod(src: str, **flags: bool) -> tuple[str, list[str]]:
    out, warnings = codemod_source(src, **flags)
    return out.rstrip("\n"), warnings


def test_self_op_add() -> None:
    out, warnings = codemod(
        "import numpy as np\nx = np.zeros(3)\ny = x + x\n", self_ops=True
    )
    assert warnings == []
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
        "a = x - x\n"
        "b = x * x\n"
        "c = x / x\n"
        "d = x ** x\n",
        self_ops=True,
    )
    assert warnings == []
    assert out == "\n".join(
        [
            "import numpy as np",
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
    assert warnings == []
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
    assert warnings == []
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
