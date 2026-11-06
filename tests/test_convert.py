"""Golden tests for numty_convert: jaxtyping specs -> static shapes.

Each test feeds a small source through convert_source and compares the
full normalized output plus warnings. The normalizer strips blank lines
so formatting churn in ast.unparse does not churn the goldens.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from numty_convert import convert_source

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def convert(src: str) -> tuple[str, list[str]]:
    out, warnings = convert_source(src)
    lines = [line for line in out.splitlines() if line.strip()]
    return "\n".join(lines), warnings


def test_named_dims_share_typevars() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def f(x: Float[np.ndarray, "batch features"])'
        ' -> Float[np.ndarray, "batch"]: ...\n'
    )
    assert warnings == []
    assert out == "\n".join(
        [
            "import numpy as np",
            "from typing import Any, TypeVar",
            "from numty import FlexArray",
            "BATCH = TypeVar('BATCH', bound=int)",
            "FEATURES = TypeVar('FEATURES', bound=int)",
            "def f(x: FlexArray[tuple[BATCH, FEATURES], np.dtype[Any]])"
            " -> FlexArray[tuple[BATCH], np.dtype[Any]]:",
            "    ...",
        ]
    )


def test_exact_dtype_prefers_float_array() -> None:
    out, warnings = convert(
        "from jaxtyping import Float64\n"
        "import numpy as np\n"
        'def g(a: Float64[np.ndarray, "n m"]): ...\n'
    )
    assert warnings == []
    assert out == "\n".join(
        [
            "import numpy as np",
            "from typing import TypeVar",
            "from numty import FloatArray",
            "M = TypeVar('M', bound=int)",
            "N = TypeVar('N', bound=int)",
            "def g(a: FloatArray[tuple[N, M]]):",
            "    ...",
        ]
    )


def test_exact_dtype_with_unpack_stays_flex() -> None:
    out, warnings = convert(
        "from jaxtyping import Int32\n"
        "import numpy as np\n"
        'def h(x: Int32[np.ndarray, "batch *rest"]): ...\n'
    )
    assert warnings == []
    assert out == "\n".join(
        [
            "import numpy as np",
            "from typing import TypeVar, TypeVarTuple",
            "from typing_extensions import Unpack",
            "from numty import FlexArray",
            "BATCH = TypeVar('BATCH', bound=int)",
            "Ts_REST = TypeVarTuple('Ts_REST')",
            "def h(x: FlexArray[tuple[BATCH, Unpack[Ts_REST]],"
            " np.dtype[np.int32]]):",
            "    ...",
        ]
    )


def test_fixed_dims_become_literals() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def m(x: Float[np.ndarray, "3 4"]): ...\n'
    )
    assert warnings == []
    assert "def m(x: FlexArray[tuple[Literal[3], Literal[4]]," in out
    assert "np.dtype[Any]]):" in out
    assert "from typing import Any, Literal" in out


def test_anonymous_variadic_gets_fresh_name() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def k(x: Float[np.ndarray, "... features"]): ...\n'
    )
    assert warnings == []
    assert "Ts_ANON0 = TypeVarTuple('Ts_ANON0')" in out
    assert "tuple[Unpack[Ts_ANON0], FEATURES]" in out


def test_underscore_dim_gets_fresh_name() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def u(x: Float[np.ndarray, "_ features"]): ...\n'
    )
    assert warnings == []
    assert "_ANON0 = TypeVar('_ANON0', bound=int)" in out
    assert "tuple[_ANON0, FEATURES]" in out


def test_symbolic_dims_left_with_warning() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def s(x: Float[np.ndarray, "n+1 m"]): ...\n'
    )
    assert warnings == ["line 3: symbolic or stacked dims; left as is"]
    assert "def s(x: Float[np.ndarray, 'n+1 m']):" in out
    assert "from jaxtyping import Float" in out
    assert "numty" not in out


def test_stacked_base_left_without_leaking_typevars() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def t(x: Float[Float[np.ndarray, "a b"], "c"]): ...\n'
    )
    assert warnings == ["line 3: non-numpy array base; left as is"]
    assert "TypeVar" not in out
    assert "from jaxtyping import Float" in out


def test_collision_bail_leaves_no_trace() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        "BATCH = 5\n"
        'def c(x: Float[np.ndarray, "ok BATCH"]): ...\n'
    )
    assert warnings == ["line 4: name BATCH already assigned; left as is"]
    assert "def c(x: Float[np.ndarray, 'ok BATCH']):" in out
    # The dims converted before the collision must not leak decls.
    assert "OK = TypeVar" not in out
    assert "from numty import" not in out


def test_shadowed_head_left_verbatim() -> None:
    out, warnings = convert(
        "import numpy as np\n"
        "Float = 1\n"
        'def v(x: Float[np.ndarray, "n"]): ...\n'
    )
    assert warnings == ["line 3: name Float shadows a jaxtyping head; left as is"]
    assert "def v(x: Float[np.ndarray, 'n']):" in out
    assert "numty" not in out


def test_function_local_binding_does_not_block() -> None:
    # `N = 5` inside a body binds at call time, after annotations have
    # evaluated, so the module-level TypeVar decl is safe.
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def w(x: Float[np.ndarray, "n"]):\n'
        "    N = 5\n"
        "    return x\n"
    )
    assert warnings == []
    assert "N = TypeVar('N', bound=int)" in out
    assert "FlexArray[tuple[N], np.dtype[Any]]" in out


def test_load_only_name_still_converts() -> None:
    # A bare `Float` mention that is only loaded (never bound) must not
    # count as shadowing: the annotation still converts.
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def w(x: Float[np.ndarray, "n"]) -> Float: ...\n'
    )
    assert warnings == []
    assert "FlexArray[tuple[N]" in out


def test_aliased_head_warns_and_keeps_import() -> None:
    out, warnings = convert(
        "from jaxtyping import Float as F\n"
        "import numpy as np\n"
        'def a(x: F[np.ndarray, "n"]): ...\n'
    )
    assert warnings == ["line 1: aliased jaxtyping import Float as F; not converted"]
    assert "from jaxtyping import Float as F" in out


def test_plain_file_untouched() -> None:
    out, warnings = convert("import numpy as np\ndef u(x: np.ndarray): ...\n")
    assert warnings == []
    assert out == "import numpy as np\ndef u(x: np.ndarray):\n    ..."


def test_broadcast_and_treepath_markers_fold() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def b(x: Float[np.ndarray, "#batch ?features"]): ...\n'
    )
    assert warnings == []
    assert "tuple[BATCH, FEATURES]" in out


def _check_converted(name: str, tmp_path: Path) -> tuple[Path, list[str]]:
    """Convert a jaxtyping fixture; write the output beside it in tmp."""
    src = (FIXTURES / name).read_text()
    out, warnings = convert_source(src)
    converted = tmp_path / name.replace("jaxtyping_", "converted_")
    converted.write_text(out)
    return converted, warnings


def _run_mypy(path: Path, tmp_path: Path) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-incremental",
            f"--cache-dir={tmp_path / 'mypy-cache'}",
            str(path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_e2e_converted_mlp_is_clean_and_runs(tmp_path: Path) -> None:
    converted, warnings = _check_converted("jaxtyping_mlp.py", tmp_path)
    assert warnings == []
    result = _run_mypy(converted, tmp_path)
    assert result.returncode == 0, result.stdout
    # The converted file executes unchanged: annotations evaluate.
    run = subprocess.run(
        [sys.executable, str(converted)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "(5, 2)"


def test_e2e_seeded_bias_bug_is_caught(tmp_path: Path) -> None:
    converted, warnings = _check_converted("jaxtyping_mlp_bug.py", tmp_path)
    assert warnings == []
    result = _run_mypy(converted, tmp_path)
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout
    assert "call-site shape conflict for 'HID'" in errors[0], result.stdout
    assert "8 (argument 'w1', axis 1) vs 2 (argument 'b2', axis 0)" in errors[0], result.stdout
