"""Exceptions inside the plugin are loud, never a silent pass (issue #164).

A hook that raises must not abort the run, but it must not look like a
clean result either: production reports [numty-internal] at the call
site and records a `crash` trace event, and NUMTY_DEBUG=1 (which
conftest.py sets for the whole suite) lets the exception propagate.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mypy.nodes import ARG_NAMED, ARG_POS, Block, ClassDef, NameExpr, SymbolTable, TypeInfo
from mypy.types import Instance, LiteralType, UnionType

import numty_codmod
import numty_plugin
from numty_convert import numpy_alias
from numty_plugin._builders import _scalar_ret_name, _scalar_spec
from numty_plugin._dtypes import _kind_bits_name, _nbit_tag_bits
from numty_plugin._errors import NUMTY_AXIS
from numty_plugin._reshape import (
    _bound_args, _check_axis, _dtype_kind_size, _fail_axis, _fail_duplicate_axis, _nbit_bits,
    _nbit_bytes, _validate_axes,
)

ROOT = Path(__file__).resolve().parent.parent

# A plugin that is numty with one solver helper broken, standing in for
# any latent bug: every `a + b` on arrays now raises inside its hook.
BROKEN_PLUGIN = """\
import numty_plugin
from numty_plugin import plugin


def _boom(*args, **kwargs):
    raise RuntimeError("seeded failure")


numty_plugin._broadcast_result = _boom
"""

SOURCE = """\
import numpy as np

a = np.zeros((2, 3)) + np.zeros((2, 3))
b = np.matmul(np.zeros((2, 3)), np.zeros((4, 5)))
"""


def _run(
    tmp_path: Path, debug: bool, config: str = ""
) -> tuple["subprocess.CompletedProcess[str]", Path]:
    (tmp_path / "broken_numty.py").write_text(BROKEN_PLUGIN, encoding="utf-8")
    (tmp_path / "mypy.ini").write_text(
        "[mypy]\nplugins = broken_numty\n" + config, encoding="utf-8"
    )
    (tmp_path / "repro.py").write_text(SOURCE, encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}",
        "NUMTY_TRACE": str(trace),
    }
    env.pop("NUMTY_DEBUG", None)
    if debug:
        env["NUMTY_DEBUG"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "cache"), "repro.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result, trace


def test_hook_exception_is_reported_not_silent(tmp_path: Path) -> None:
    result, trace = _run(tmp_path, debug=False)
    out = result.stdout + result.stderr
    assert "INTERNAL ERROR" not in out, out
    internal = [line for line in result.stdout.splitlines() if "[numty-internal]" in line]
    # Both hooks on the `+` (signature and method) reach the broken helper.
    assert internal, out
    for line in internal:
        assert line.startswith("repro.py:3: error: numty internal error in "), out
        assert "RuntimeError: seeded failure" in line, out
        assert "result is unchecked" in line, out
    assert any("numpy.ndarray.__add__" in line for line in internal), out
    # The run carried on: the unrelated bug on the next line is still found.
    assert "repro.py:4: error: matmul inner dimensions disagree: 3 vs 4" in result.stdout, out
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    crashes = [e for e in events if e["outcome"] == "crash"]
    assert crashes and all(e["line"] == 3 for e in crashes), events
    assert "RuntimeError: seeded failure" in crashes[0]["reason"], events


def test_hook_exception_is_silenceable_by_code(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, debug=False, config="disable_error_code = numty-internal\n")
    assert "numty internal error" not in result.stdout, result.stdout
    assert "matmul inner dimensions disagree" in result.stdout, result.stdout


def test_debug_mode_propagates_the_exception(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, debug=True)
    out = result.stdout + result.stderr
    assert "INTERNAL ERROR" in out, out
    assert "[numty-internal]" not in out, out


def test_suite_runs_in_debug_mode() -> None:
    # conftest.py turns this on; without it a swallowed exception in any
    # fixture run would degrade to a passing test.
    assert os.environ.get("NUMTY_DEBUG") == "1"


def test_every_broad_except_goes_through_the_policy() -> None:
    # Every `except Exception:` must route through _swallowed() on the
    # very next line, except the reporting helpers themselves (those
    # lines carry "reporting must never"). Scans every .py under
    # numty_plugin/ so Phase 2 leaves (_*.py) are covered.
    # NOTE: the floor is 75 (was 76 on the monolith). Monolith line-scan
    # counted 76; Phase 2 removed a dead _swallowed-guarded try in
    # _forms_provably_ne (pure after the move) as part of the leaf
    # extraction, so 75 is the correct package floor. Do not raise to
    # 77 (grep -c also matches docstring prose).
    plugin_dir = ROOT / "numty_plugin"
    files = sorted(
        p for p in plugin_dir.rglob("*.py") if "__pycache__" not in p.parts
    )
    assert files, f"no .py files found under {plugin_dir} - glob broken?"
    assert plugin_dir / "__init__.py" in files, "scan must include __init__.py"
    stray = []
    total = 0
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(plugin_dir)
        for i, line in enumerate(lines):
            if line.strip().startswith("except Exception"):
                total += 1
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else "<EOF>"
                inside_helper = "reporting must never" in line
                if not nxt.startswith("_swallowed(") and not inside_helper:
                    stray.append(f"{rel}:{i + 1}: {line.strip()} -> {nxt}")
    assert not stray, "broad except without _swallowed():\n" + "\n".join(stray)
    assert total >= 75, f"broad-except total dropped: {total} < 75"


def test_swallowed_reraises_only_in_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMTY_DEBUG", "1")
    with pytest.raises(KeyError):
        try:
            raise KeyError("x")
        except Exception:
            numty_plugin._swallowed(None, "unit")
    monkeypatch.delenv("NUMTY_DEBUG")
    try:
        raise KeyError("x")
    except Exception:
        numty_plugin._swallowed(None, "unit")  # no context: trace-only, no raise


SEEDED_TUPLE_PLUGIN = """\
import sys

from mypy.types import TupleType
from numty_plugin import plugin

_original = TupleType.copy_modified


def _copy_modified(self, *args, **kwargs):
    if sys._getframe(1).f_code.co_name == "_substitute_return_exprs":
        raise RuntimeError("seeded failure")
    return _original(self, *args, **kwargs)


TupleType.copy_modified = _copy_modified
"""

FORMULA_RETURN_SOURCE = """\
from typing import TypeVar, reveal_type

import numpy as np

from numty import I64

M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)


def make_offsets(lengths: I64[M]) -> I64["M+1"]:
    return np.concatenate([[0], np.cumsum(lengths)])


def caller(a: I64[K]) -> None:
    reveal_type(make_offsets(a))
"""


def test_return_substitution_survives_a_failed_tuple_copy(tmp_path: Path) -> None:
    # `_substitute_return_exprs` rebuilds the shape tuple by hand when
    # TupleType.copy_modified raises. That path used `shape.fallback`,
    # an attribute TupleType does not have (it is `partial_fallback`),
    # so the rebuild raised too and the call site lost its K+1 shape.
    # Found by ty (#213).
    (tmp_path / "seeded_numty.py").write_text(SEEDED_TUPLE_PLUGIN, encoding="utf-8")
    (tmp_path / "mypy.ini").write_text("[mypy]\nplugins = seeded_numty\n", encoding="utf-8")
    (tmp_path / "repro.py").write_text(FORMULA_RETURN_SOURCE, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"}
    env.pop("NUMTY_DEBUG", None)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "cache"), "repro.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = result.stdout + result.stderr
    assert "INTERNAL ERROR" not in out, out
    # The seeded failure is reported, and the hand-built tuple is used.
    assert "RuntimeError: seeded failure" in result.stdout, out
    assert "AttributeError" not in out, out
    assert 'Revealed type is "numpy.ndarray[tuple[K+1], ' in result.stdout, out


# Shared kernel helpers (issue #323): argument plumbing, axis checks and
# dtype resolution, exercised directly on hand-built contexts and types.


def _call(formals: list[tuple[str | None, list[tuple[object, str]]]]) -> Any:
    """A stand-in call context: per formal, its name and its (kind, value) actuals."""
    return SimpleNamespace(
        callee_arg_names=[name for name, _ in formals],
        arg_kinds=[[kind for kind, _ in acts] for _, acts in formals],
        arg_types=[[value for _, value in acts] for _, acts in formals],
        args=[[NameExpr(value) for _, value in acts] for _, acts in formals],
    )


def _values(found: list | None) -> list | None:
    return None if found is None else [None if p is None else p[0] for p in found]


def test_bound_args_reads_formals_then_positional_slots() -> None:
    positional = _call([("a", [(ARG_POS, "x")]), ("n", [(ARG_POS, "8")]), ("axis", [])])
    assert _values(_bound_args(positional, "a", ("n", "axis"), (1, 2))) == ["8", None]
    # Erased formals (overload fallback): only the slots are readable.
    erased = _call([(None, [(ARG_POS, "x")]), (None, [(ARG_POS, "8")]), (None, [(ARG_POS, "-1")])])
    assert _values(_bound_args(erased, "a", ("n", "axis"), (1, 2))) == ["8", "-1"]
    assert _values(_bound_args(erased, "a", ("n", "axis"), (1, None))) == ["8", None]
    assert _values(_bound_args(erased, "a", ("n", "axis"), (1, 2), slots_first=True)) == ["8", "-1"]


def test_bound_args_keywords_need_the_first_formal() -> None:
    keyword = _call([("a", [(ARG_POS, "x")]), ("n", []), ("axis", [(ARG_NAMED, "0")])])
    assert _values(_bound_args(keyword, "a", ("n", "axis"), (1, 2))) == [None, "0"]
    assert _values(_bound_args(keyword, "a", ("n", "axis"), (1, 2), slots_first=True)) == [None, "0"]
    # A keyword behind erased formals hides where it binds: bail.
    erased = _call([(None, [(ARG_POS, "x")]), (None, [(ARG_NAMED, "0")])])
    assert _bound_args(erased, "a", ("axis",), (1,)) is None
    # first=None skips that check (a method, whose self is not an actual).
    assert _values(_bound_args(erased, None, ("axis",), (1,))) == [None]


class _Recorder:
    def __init__(self) -> None:
        self.errors: list[tuple[str, object]] = []

    def fail(self, msg: str, context: object, code: object = None) -> None:
        self.errors.append((msg, code))


def _ctx() -> Any:
    return SimpleNamespace(api=_Recorder(), context=None)


def test_fail_axis_message_and_code() -> None:
    ctx = _ctx()
    _fail_axis(ctx, "swapaxes axis2", 5, [2, 3])
    _fail_axis(ctx, "stack axis", -4, [2], 2)
    assert ctx.api.errors == [
        ("swapaxes axis2 5 is out of range for 2 dimensions (shape (2, 3))", NUMTY_AXIS),
        ("stack axis -4 is out of range for 2 dimensions (shape (2,))", NUMTY_AXIS),
    ]


def test_check_axis_normalizes_or_reports() -> None:
    ctx = _ctx()
    assert _check_axis(ctx, "fft axis", -1, [4, 5]) == 1
    assert _check_axis(ctx, "repeat axis", -1, [], zero_d=True) == 0
    assert ctx.api.errors == []
    assert _check_axis(ctx, "repeat axis", 1, [], zero_d=True) is None
    assert _check_axis(ctx, "fft axis", 0, []) is None
    assert ctx.api.errors == [
        ("repeat axis 1 is out of range for 0 dimensions (shape ())", NUMTY_AXIS),
        ("fft axis 0 is out of range for 0 dimensions (shape ())", NUMTY_AXIS),
    ]


def test_validate_axes_range_then_duplicates() -> None:
    ctx = _ctx()
    assert _validate_axes(ctx, "sum", [0, -1], [2, 3, 4]) == [0, 2]
    assert _validate_axes(ctx, "transpose", [1, -1], [2, 3], unique=False) == [1, 1]
    assert ctx.api.errors == []
    assert _validate_axes(ctx, "sum", [0, -3], [2, 3, 4]) is None
    assert _validate_axes(ctx, "sum", [3, 0, 0], [2, 3, 4]) is None
    assert _validate_axes(ctx, "max", [0], [], zero_d=True) == [0]
    assert ctx.api.errors == [
        ("sum has duplicate axis -3 in (0, -3) (shape (2, 3, 4))", NUMTY_AXIS),
        ("sum axis 3 is out of range for 3 dimensions (shape (2, 3, 4))", NUMTY_AXIS),
    ]
    _fail_duplicate_axis(ctx, "expand_dims", 1, [1, 1], [5])
    assert ctx.api.errors[-1] == (
        "expand_dims has duplicate axis 1 in (1, 1) (shape (5,))",
        NUMTY_AXIS,
    )


def _inst(fullname: str, *args) -> Instance:
    module, name = fullname.rsplit(".", 1)
    defn = ClassDef(name, Block([]))
    defn.fullname = fullname
    return Instance(TypeInfo(SymbolTable(), defn, module), list(args))


_NBIT = "numpy._typing._nbit_base."


def test_nbit_tag_bits_and_kind_names() -> None:
    assert _nbit_tag_bits(_NBIT + "_32Bit") == 32
    assert _nbit_tag_bits(_NBIT + "_96Bit") == 96
    assert _nbit_tag_bits(_NBIT + "NBitBase") is None
    assert _nbit_tag_bits("numpy.float64") is None
    assert _nbit_tag_bits("numpy.int_") is None
    assert _kind_bits_name("f", 32) == "numpy.float32"
    assert _kind_bits_name("c", 64) == "numpy.complex64"
    assert _kind_bits_name("i", 128) is None
    assert _kind_bits_name("f", 12) is None


def test_nbit_bits_reads_tags_and_unions() -> None:
    b32, b64, b96 = (_inst(_NBIT + f"_{n}Bit") for n in (32, 64, 96))
    assert _nbit_bits(b64) == 64
    assert _nbit_bits(UnionType([b32, b64])) is None
    assert _nbit_bits(UnionType([b32, b64]), union=True) == 64
    assert _nbit_bits(UnionType([b32, _inst("builtins.int")]), union=True) is None
    # Bytes: a literal width counts, widths numpy has no dtype for do not.
    assert _nbit_bytes(LiteralType(16, _inst("builtins.int"))) == 2
    assert _nbit_bytes(b96) is None


def test_scalar_dtype_resolution() -> None:
    b32, b64 = _inst(_NBIT + "_32Bit"), _inst(_NBIT + "_64Bit")
    intp = _inst("numpy.signedinteger", UnionType([b32, b64]))
    assert _scalar_ret_name(_inst("numpy.floating", b32)) == "numpy.float32"
    assert _scalar_ret_name(_inst("numpy.bool")) == "numpy.bool"
    assert _scalar_ret_name(intp) == "numpy.int64"
    assert _scalar_ret_name(_inst("numpy.complexfloating", b32, b32)) is None
    assert _dtype_kind_size(_inst("numpy.dtype", _inst("numpy.complexfloating", b32, b32))) == ("c", 8)
    assert _dtype_kind_size(_inst("numpy.dtype", _inst("numpy.unsignedinteger", b32))) == ("u", 4)
    assert _dtype_kind_size(_inst("numpy.dtype", _inst("numpy.float64"))) == ("f", 8)
    assert _dtype_kind_size(_inst("numpy.dtype", intp)) is None
    assert _scalar_spec(_inst("builtins.float")) == ("weak", "builtins.float")
    assert _scalar_spec(_inst("numpy.int16")) == ("strong", "numpy.int16")
    assert _scalar_spec(intp) == ("strong", "numpy.int64")
    assert _scalar_spec(_inst("builtins.str")) is None


def test_numpy_alias_is_shared_by_convert_and_codmod() -> None:
    assert numpy_alias(ast.parse("import numpy as np\n")) == "np"
    assert numpy_alias(ast.parse("import os, numpy\n")) == "numpy"
    assert numpy_alias(ast.parse("from numpy import zeros\n")) is None
    assert numty_codmod.numpy_alias is numpy_alias


def test_code_enabled_follows_mypys_rule() -> None:
    # Issue 327: numty-names is on by default, and a hook that reports
    # it while mypy has it disabled fails the overload mypy is checking
    # (an [operator] error out of nowhere). `_code_enabled` must agree
    # with mypy's own `Errors.is_error_code_enabled` for every mix of
    # enabled and disabled codes, parent codes included.
    import itertools

    from mypy.errorcodes import MISC
    from mypy.errors import Errors
    from mypy.options import Options

    from numty_plugin._errors import NUMTY_CAST, NUMTY_NAMES, NUMTY_SHAPE, _code_enabled

    assert NUMTY_NAMES.default_enabled and not NUMTY_CAST.default_enabled
    codes = (NUMTY_NAMES, NUMTY_CAST, NUMTY_SHAPE)
    for code in codes:
        others = (code, MISC)
        for disabled, enabled in itertools.product(
            itertools.product((False, True), repeat=len(others)), repeat=2
        ):
            options = Options()
            options.disabled_error_codes = {c for c, on in zip(others, disabled) if on}
            options.enabled_error_codes = {c for c, on in zip(others, enabled) if on}
            expected = Errors(options).is_error_code_enabled(code)
            assert _code_enabled(options, code) == expected, (code, disabled, enabled)
    assert not _code_enabled(None, NUMTY_NAMES)
