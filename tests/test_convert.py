"""Golden tests for numty_convert: jaxtyping specs -> static shapes.

Each test feeds a small source through convert_source and compares the
exact output plus warnings: the converter splices only the converted
annotations into the original text, so everything else (comments,
pragmas, blank lines) must come back byte for byte.
"""
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from numty_convert import convert_source

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def _convert_cache_dir() -> Path:
    _worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    return ROOT / (".mypy-convert-cache" if not _worker else f".mypy-convert-cache-{_worker}")


@pytest.fixture(scope="session", autouse=True)
def _fresh_convert_cache() -> Iterator[None]:
    # Fresh per session (like test_static): mypy does not reliably
    # invalidate plugin results, but within one run the plugin is fixed
    # so incremental reuse across the 3 e2e checks stays fast.
    import shutil

    shutil.rmtree(_convert_cache_dir(), ignore_errors=True)
    yield
    # Keep the cache for the next run's warmth? No: clear at start is
    # enough; leave the warmed cache on disk for the next invocation.
    pass


def convert(src: str) -> tuple[str, list[str]]:
    return convert_source(src)


def test_named_dims_share_typevars() -> None:
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def f(x: Float[np.ndarray, "batch features"])'
        ' -> Float[np.ndarray, "batch"]: ...\n'
    )
    assert warnings == []
    assert out == (
        "import numpy as np\n"
        "from typing import Any, TypeVar\n"
        "from numty import FlexArray\n"
        "BATCH = TypeVar('BATCH', bound=int)\n"
        "FEATURES = TypeVar('FEATURES', bound=int)\n"
        "def f(x: FlexArray[tuple[BATCH, FEATURES], np.dtype[Any]])"
        " -> FlexArray[tuple[BATCH], np.dtype[Any]]: ...\n"
    )


def test_exact_dtype_prefers_float_array() -> None:
    out, warnings = convert(
        "from jaxtyping import Float64\n"
        "import numpy as np\n"
        'def g(a: Float64[np.ndarray, "n m"]): ...\n'
    )
    assert warnings == []
    assert out == (
        "import numpy as np\n"
        "from typing import TypeVar\n"
        "from numty import FloatArray\n"
        "M = TypeVar('M', bound=int)\n"
        "N = TypeVar('N', bound=int)\n"
        "def g(a: FloatArray[tuple[N, M]]): ...\n"
    )


def test_exact_dtype_with_unpack_stays_flex() -> None:
    out, warnings = convert(
        "from jaxtyping import Int32\n"
        "import numpy as np\n"
        'def h(x: Int32[np.ndarray, "batch *rest"]): ...\n'
    )
    assert warnings == []
    # Unpack comes from typing (numty needs Python >= 3.13, never
    # typing_extensions), and a *Ts shape carries the ignore numty needs.
    assert out == (
        "import numpy as np\n"
        "from typing import TypeVar, TypeVarTuple, Unpack\n"
        "from numty import FlexArray\n"
        "BATCH = TypeVar('BATCH', bound=int)\n"
        "Ts_REST = TypeVarTuple('Ts_REST')\n"
        "def h(x: FlexArray[tuple[BATCH, Unpack[Ts_REST]],"
        " np.dtype[np.int32]]): ...  # type: ignore[type-var]\n"
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
    assert 'def s(x: Float[np.ndarray, "n+1 m"]): ...' in out
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
    assert 'def c(x: Float[np.ndarray, "ok BATCH"]): ...' in out
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
    assert 'def v(x: Float[np.ndarray, "n"]): ...' in out
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
    src = "import numpy as np\ndef u(x: np.ndarray): ...\n"
    out, warnings = convert(src)
    assert warnings == []
    assert out == src


def test_treepath_marker_folds() -> None:
    # `?name` only relaxes the dim across PyTree leaves; a bare array
    # annotation has one leaf, so it is the plain name.
    out, warnings = convert(
        "from jaxtyping import Float\n"
        "import numpy as np\n"
        'def b(x: Float[np.ndarray, "?features"]): ...\n'
    )
    assert warnings == []
    assert "tuple[FEATURES]" in out


def test_broadcast_dim_left_verbatim_with_warning() -> None:
    # `#n` admits size 1 *or* n: no static spelling, and the strict `N`
    # would reject valid calls, so that annotation stays jaxtyping.
    src = (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        'def add(x: Float64[np.ndarray, "n"], y: Float64[np.ndarray, "#n"])'
        ' -> Float64[np.ndarray, "n"]:\n'
        "    return x + y\n"
    )
    out, warnings = convert(src)
    assert warnings == [
        "line 3: broadcastable dim #n has no static equivalent; left as is"
    ]
    assert 'y: Float64[np.ndarray, "#n"]' in out
    assert "def add(x: FloatArray[tuple[N]]," in out
    # The verbatim annotation still needs jaxtyping.
    assert "from jaxtyping import Float64\n" in out


def test_unchecked_dims_are_fresh() -> None:
    # `_n` is never checked: each occurrence is its own size.
    out, warnings = convert(
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        'def f(x: Float64[np.ndarray, "_n"], y: Float64[np.ndarray, "_n *_"])'
        " -> None: ...\n"
    )
    assert warnings == []
    assert (
        "def f(x: FloatArray[tuple[_ANON0]], y: FlexArray[tuple[_ANON1,"
        " Unpack[Ts_ANON2]], np.dtype[np.float64]]) -> None: ..."
    ) in out
    assert " N = " not in out and "\nN = " not in out


@pytest.mark.parametrize(
    ("dims", "decls", "shape"),
    [
        ("B b", ["B", "b"], "tuple[B, b]"),
        ("ß ss", ["ss", "ß"], "tuple[ß, ss]"),
        ("batch", ["BATCH"], "tuple[BATCH]"),
    ],
)
def test_names_stay_distinct(dims: str, decls: list[str], shape: str) -> None:
    # jaxtyping names are case-sensitive; upper-casing is cosmetic and
    # only applies while it keeps distinct names distinct.
    out, warnings = convert(
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        f'def f(x: Float64[np.ndarray, "{dims}"]) -> None: ...\n'
    )
    assert warnings == []
    for name in decls:
        assert f"{name} = TypeVar({name!r}, bound=int)\n" in out
    assert f"def f(x: FloatArray[{shape}]) -> None: ..." in out


def test_unspellable_names_left_verbatim() -> None:
    # `len` would rebind the builtin; `ﬁ` and `fi` are one identifier
    # after NFKC folding, though jaxtyping keeps them apart.
    src = (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        'def f(x: Float64[np.ndarray, "len LEN"]) -> None: ...\n'
        'def g(x: Float64[np.ndarray, "ﬁ FI fi"]) -> None: ...\n'
    )
    out, warnings = convert(src)
    assert out == src
    assert warnings == [
        "line 3: name len would shadow a builtin or a name the converter"
        " emits; left as is",
        "line 4: dim ﬁ has no identifier distinct from another dim name;"
        " left as is",
    ]


def test_two_variadics_left_verbatim() -> None:
    src = (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        'def f(x: Float64[np.ndarray, "*a *b"]) -> None: ...\n'
        'def g(x: Float64[np.ndarray, "... *b"]) -> None: ...\n'
    )
    out, warnings = convert(src)
    assert out == src
    assert warnings == [
        "line 3: more than one variadic dim; left as is",
        "line 4: more than one variadic dim; left as is",
    ]


def test_only_def_signatures_convert() -> None:
    # Module/class variables and runtime uses (isinstance) have no def to
    # bind a TypeVar: rewriting them gives unbound TypeVars or crashes.
    src = (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        'W: Float64[np.ndarray, "n n"] = np.eye(3)\n'
        "class C:\n"
        '    v: Float64[np.ndarray, "n"]\n'
        "def check(x: object) -> bool:\n"
        '    return isinstance(x, Float64[np.ndarray, "b c"])\n'
    )
    out, warnings = convert(src)
    assert out == src
    assert warnings == [
        "line 3: jaxtyping subscript outside a def signature; left as is",
        "line 5: jaxtyping subscript outside a def signature; left as is",
        "line 7: jaxtyping subscript outside a def signature; left as is",
    ]


def test_return_only_dims() -> None:
    # A return dim no parameter binds is unknown to the caller: `int`
    # (rank: `*tuple[int, ...]`) instead of an unsolvable TypeVar. A name
    # the return repeats ties those axes, so it stays verbatim.
    src = (
        "import numpy as np\n"
        "from jaxtyping import Float\n"
        'def f(a: Float[np.ndarray, "n"]) -> Float[np.ndarray, "_ m"]: ...\n'
        'def g(a: Float[np.ndarray, "n"]) -> Float[np.ndarray, "... n"]: ...\n'
        'def h(a: Float[np.ndarray, "n"]) -> Float[np.ndarray, "m m"]: ...\n'
    )
    out, warnings = convert(src)
    assert warnings == [
        "line 5: dim m repeats in the return but no parameter binds it;"
        " left as is"
    ]
    assert "-> FlexArray[tuple[int, int], np.dtype[Any]]: ..." in out
    assert (
        "-> FlexArray[tuple[Unpack[tuple[int, ...]], N], np.dtype[Any]]: ..."
    ) in out
    assert '-> Float[np.ndarray, "m m"]: ...' in out


def test_imported_name_blocks_typevar() -> None:
    # `from config import BATCH` binds BATCH: a TypeVar decl would rebind it.
    src = (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        "from config import BATCH\n"
        'def f(x: Float64[np.ndarray, "batch"]) -> None: ...\n'
    )
    out, warnings = convert(src)
    assert out == src
    assert warnings == ["line 4: name BATCH already assigned; left as is"]


def test_header_goes_after_late_imports() -> None:
    # Environment setup between imports runs first, and the decls follow
    # the import that supplies TypeVar.
    out, warnings = convert(
        "import os\n"
        'os.environ.setdefault("OMP_NUM_THREADS", "1")\n'
        "from typing import Optional\n"
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        "\n"
        'def f(x: Float64[np.ndarray, "n"], y: Optional[int] = None) -> None:\n'
        "    pass\n"
    )
    assert warnings == []
    assert out == (
        "import os\n"
        'os.environ.setdefault("OMP_NUM_THREADS", "1")\n'
        "from typing import Optional\n"
        "import numpy as np\n"
        "from typing import TypeVar\n"
        "from numty import FloatArray\n"
        "N = TypeVar('N', bound=int)\n"
        "\n"
        "def f(x: FloatArray[tuple[N]], y: Optional[int] = None) -> None:\n"
        "    pass\n"
    )


def test_string_and_dunder_all_uses_keep_import() -> None:
    src = (
        "import numpy as np\n"
        "from jaxtyping import Float, Float64\n"
        '__all__ = ["Float", "f"]\n'
        "def f(x: \"Float64[np.ndarray, 'n']\") -> None: ...\n"
        'def g(x: Float64[np.ndarray, "m"]) -> None: ...\n'
    )
    out, warnings = convert(src)
    assert warnings == ["line 4: string annotation not converted; left as is"]
    assert "from jaxtyping import Float, Float64\n" in out
    assert "def g(x: FloatArray[tuple[M]]) -> None: ..." in out


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
        "# -*- coding: utf-8 -*-\ns = 'ß'  # é\n\n\n\nx = 1;  y = (  # c\n  2)\n",
    ],
    ids=["readme", "crlf", "no-final-newline", "unicode"],
)
def test_nothing_to_convert_round_trips(src: str) -> None:
    out, warnings = convert(src)
    assert warnings == []
    assert out == src


def test_comments_and_pragmas_survive_conversion() -> None:
    src = (
        "# header comment\n"
        "import numpy as np\n"
        "from jaxtyping import Float64, Int32  # tidy\n"
        "\n"
        "\n"
        "def f(  # leading comment\n"
        '    x: Float64[np.ndarray, "*batch c"],  # the input\n'
        '    y: Float64[np.ndarray, "c"],\n'
        ') -> Float64[np.ndarray, "*batch c"]:\n'
        "    return x + y  # broadcast\n"
        "\n"
        "\n"
        "class K:\n"
        '    def m(self, a: Int32[np.ndarray, "... n"])'
        ' -> Int32[np.ndarray, "n"]:  # type: ignore[misc]\n'
        "        return a[0]\n"
    )
    out, warnings = convert(src)
    assert warnings == []
    assert out == (
        "# header comment\n"
        "import numpy as np\n"
        "from typing import TypeVar, TypeVarTuple, Unpack\n"
        "from numty import FlexArray, FloatArray\n"
        "C = TypeVar('C', bound=int)\n"
        "N = TypeVar('N', bound=int)\n"
        "Ts_ANON0 = TypeVarTuple('Ts_ANON0')\n"
        "Ts_BATCH = TypeVarTuple('Ts_BATCH')\n"
        "\n"
        "\n"
        "def f(  # leading comment\n"
        "    x: FlexArray[tuple[Unpack[Ts_BATCH], C], np.dtype[np.float64]],"
        "  # type: ignore[type-var]  # the input\n"
        "    y: FloatArray[tuple[C]],\n"
        ") -> FlexArray[tuple[Unpack[Ts_BATCH], C], np.dtype[np.float64]]:"
        "  # type: ignore[type-var]\n"
        "    return x + y  # broadcast\n"
        "\n"
        "\n"
        "class K:\n"
        "    def m(self, a: FlexArray[tuple[Unpack[Ts_ANON0], N],"
        " np.dtype[np.int32]]) -> FlexArray[tuple[N], np.dtype[np.int32]]:"
        "  # type: ignore[misc, type-var]\n"
        "        return a[0]\n"
    )


def test_crlf_conversion_keeps_line_endings() -> None:
    src = (
        "import numpy as np\r\n"
        "from jaxtyping import Float\r\n"
        "\r\n"
        'def f(a: Float[np.ndarray, "n"]) -> None:  # c\r\n'
        "    pass\r\n"
    )
    out, warnings = convert(src)
    assert warnings == []
    assert out == (
        "import numpy as np\r\n"
        "from typing import Any, TypeVar\r\n"
        "from numty import FlexArray\r\n"
        "N = TypeVar('N', bound=int)\r\n"
        "\r\n"
        "def f(a: FlexArray[tuple[N], np.dtype[Any]]) -> None:  # c\r\n"
        "    pass\r\n"
    )


def _cli(*args: str, cwd: Path) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-m", "numty_convert", *args],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_cli_help(flag: str, tmp_path: Path) -> None:
    result = _cli(flag, cwd=tmp_path)
    assert result.returncode == 0
    assert result.stdout.startswith("usage: numty_convert.py")


def test_cli_errors_are_one_line(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def f(:\n    pass\n")
    missing = _cli("missing.py", cwd=tmp_path)
    assert missing.returncode == 2
    assert missing.stderr == (
        "numty_convert.py: error: missing.py: No such file or directory\n"
    )
    broken = _cli("broken.py", cwd=tmp_path)
    assert broken.returncode == 2
    assert broken.stderr == (
        "numty_convert.py: error: broken.py:1: syntax error: invalid syntax\n"
    )


def test_cli_writes_bytes_unchanged(tmp_path: Path) -> None:
    src = README_NORM.replace("\n", "\r\n")
    (tmp_path / "norm.py").write_bytes(src.encode())
    result = _cli("norm.py", "out.py", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert (tmp_path / "out.py").read_bytes() == src.encode()
    stdout = _cli("norm.py", cwd=tmp_path)
    assert stdout.stdout == README_NORM


def _check_converted(name: str, tmp_path: Path) -> tuple[Path, list[str]]:
    """Convert a jaxtyping fixture; write the output beside it in tmp."""
    src = (FIXTURES / name).read_text()
    out, warnings = convert_source(src)
    converted = tmp_path / name.replace("jaxtyping_", "converted_")
    converted.write_text(out)
    return converted, warnings


def _run_mypy(*paths: Path, tmp_path: Path) -> "subprocess.CompletedProcess[str]":
    # Shared incremental cache (per xdist worker): converted files differ
    # per test but numpy/plugin parsing is cached, so the 2nd/3rd e2e run
    # is warm instead of cold with --no-incremental.
    cache_dir = _convert_cache_dir()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            f"--cache-dir={cache_dir}",
            *map(str, paths),
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
    result = _run_mypy(converted, tmp_path=tmp_path)
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
    result = _run_mypy(converted, tmp_path=tmp_path)
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    # The HID/OUT mismatch surfaces twice: the unprovable (BATCH, OUT) +
    # (HID) broadcast mints ?U, which then mismatches the (BATCH, OUT)
    # return, plus the call-site HID conflict (8 vs 2).
    assert len(errors) == 2, result.stdout
    assert "tuple[BATCH, ?U" in errors[0] and "[return-value]" in errors[0], result.stdout
    assert "call-site shape conflict for 'HID'" in errors[1], result.stdout
    assert "8 (argument 'w1', axis 1) vs 2 (argument 'b2', axis 0)" in errors[1], result.stdout


# A stand-in for jaxtyping at runtime. Without `@jaxtyped` real jaxtyping
# checks nothing either: `Float[np.ndarray, "n"]` just evaluates to an
# annotation, so the originals print exactly what they print with it.
JAXTYPING_STUB = """from typing import Annotated, Any


class _Head:
    def __getitem__(self, item: Any) -> Any:
        return Annotated[item[0], item[1]]


Float = Float64 = Int32 = _Head()
"""

E2E = {
    "namedvar": (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        "\n"
        'def f(x: Float64[np.ndarray, "*batch c"])'
        ' -> Float64[np.ndarray, "*batch c"]:\n'
        "    return x * 2\n"
        "\n"
        "print(f(np.zeros((4, 5, 3))).shape)\n"
    ),
    "unchecked": (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        "\n"
        'def f(x: Float64[np.ndarray, "_n"], y: Float64[np.ndarray, "_n"])'
        " -> None:\n"
        "    print(x.shape, y.shape)\n"
        "\n"
        "f(np.zeros(3), np.zeros(4))\n"
    ),
    "case": (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        "\n"
        'def f(x: Float64[np.ndarray, "B b"]) -> Float64[np.ndarray, "B b"]:\n'
        "    return x\n"
        "\n"
        "print(f(np.zeros((2, 3))).shape)\n"
    ),
    "eszett": (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        "\n"
        'def f(x: Float64[np.ndarray, "ß ss"]) -> None:\n'
        "    print(x.shape)\n"
        "\n"
        "f(np.zeros((2, 3)))\n"
    ),
    "retanon": (
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        "\n"
        'def f(x: Float64[np.ndarray, "n"]) -> Float64[np.ndarray, "_"]:\n'
        "    return x[:2]\n"
        "\n"
        "r = f(np.zeros(5))\n"
        "print(r.shape)\n"
    ),
    "late": (
        "import os\n"
        'os.environ.setdefault("OMP_NUM_THREADS", "1")\n'
        "from typing import Optional\n"
        "import numpy as np\n"
        "from jaxtyping import Float64\n"
        "\n"
        'def f(x: Float64[np.ndarray, "n"], y: Optional[int] = None)'
        ' -> Float64[np.ndarray, "n"]:\n'
        "    return x\n"
        "\n"
        "print(f(np.zeros(3)).shape)\n"
    ),
    "norm": README_NORM,
}


def test_e2e_converted_runs_and_checks(tmp_path: Path) -> None:
    """Converted files check clean and print what the originals print.

    Covers the lost-comment cases (README `squared_norm`; `*batch c`,
    which needs its ignore), `_n`, case, return-only `_` and late imports.
    """
    import concurrent.futures

    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "jaxtyping.py").write_text(JAXTYPING_STUB)
    converted = []
    for name, src in E2E.items():
        out, warnings = convert_source(src)
        assert warnings == [], name
        assert "jaxtyping" not in out, name
        (tmp_path / f"orig_{name}.py").write_text(src)
        path = tmp_path / f"conv_{name}.py"
        path.write_text(out)
        converted.append(path)
    result = _run_mypy(*converted, tmp_path=tmp_path)
    assert result.returncode == 0, result.stdout

    def _run_pair(name: str) -> tuple[str, int, str, int, str, str, str]:
        before = subprocess.run(
            [sys.executable, str(tmp_path / f"orig_{name}.py")],
            env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{stub}"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        # No stub here: a fully converted file needs no jaxtyping.
        after = subprocess.run(
            [sys.executable, str(tmp_path / f"conv_{name}.py")],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        return (name, before.returncode, before.stderr, after.returncode, after.stderr,
                before.stdout, after.stdout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(E2E))) as pool:
        for name, bc, berr, ac, aerr, bout, aout in pool.map(_run_pair, list(E2E)):
            assert bc == 0, (name, berr)
            assert ac == 0, (name, aerr)
            assert aout == bout, name
