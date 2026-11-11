"""Error codes + exception policy + startup gates (Phase 2, #196).

Depends only on ._trace (+ mypy.errorcodes at runtime). Facade seam:
_installed_numpy_version stays in __init__.py (tests patch it via the
facade; plugin() must see the patch). Pure helpers (_parse_version_tuple,
_mypy_version_warning, _numpy_version_warning) live here.
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from mypy.errorcodes import MISC, ErrorCode

from ._trace import _numty_strict, _trace_event

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, MethodContext
    from mypy.types import Type

# Error codes, one per kind of diagnostic a user may want to silence or
# select on its own (`# type: ignore[numty-bail]`,
# `disable_error_code = numty-bail`). All are sub-codes of misc, so an
# existing `ignore[misc]` or `disable_error_code = misc` still covers them.
NUMTY_SHAPE = ErrorCode(
    "numty-shape",
    "Array dimensions, ranks or size arguments do not fit the operation",
    "numty",
    sub_code_of=MISC,
)
NUMTY_BROADCAST = ErrorCode(
    "numty-broadcast", "Array shapes do not broadcast together", "numty", sub_code_of=MISC
)
NUMTY_CALLSITE = ErrorCode(
    "numty-callsite",
    "Shape variable bound to conflicting sizes at a call",
    "numty",
    sub_code_of=MISC,
)
NUMTY_AXIS = ErrorCode(
    "numty-axis", "Axis or index argument out of range or invalid", "numty", sub_code_of=MISC
)
NUMTY_BAIL = ErrorCode(
    "numty-bail", "numty cannot check this expression", "numty", sub_code_of=MISC
)
NUMTY_ANNOTATION = ErrorCode(
    "numty-annotation", "Invalid numty shape annotation", "numty", sub_code_of=MISC
)
NUMTY_UNSOLVED = ErrorCode(
    "numty-unsolved",
    "numty could not prove this shape; the result is unchecked",
    "numty",
    sub_code_of=MISC,
)
NUMTY_INTERNAL = ErrorCode(
    "numty-internal",
    "numty itself failed on this expression; the result is unchecked",
    "numty",
    sub_code_of=MISC,
)


def _numty_debug() -> bool:
    """Whether NUMTY_DEBUG=1 asks plugin exceptions to propagate."""
    return os.environ.get("NUMTY_DEBUG") == "1"


def _swallowed(ctx: object, where: str) -> None:
    """The one policy for an exception caught inside the plugin (issue #164).

    Call it first in every `except Exception:` block. A plugin bug must
    never abort the mypy run, but it must not pass for a clean result
    either:

    - NUMTY_DEBUG=1 re-raises the exception being handled, so the test
      suite (which sets it) fails on any swallowed exception and
      developers get the traceback;
    - otherwise the failure is recorded in NUMTY_TRACE with outcome
      `crash` and reported at the call site as [numty-internal]. mypy
      drops repeated identical messages on a line, so no dedup state is
      kept here (process-lifetime dedup hid errors under dmypy, #44).

    `ctx` is any hook context (or None where none is in reach: those
    sites are trace-only outside debug mode). Conditions that are
    expected in normal operation should be caught by a narrower
    `except` instead of coming through here.
    """
    exc = sys.exception()
    if _numty_debug() and exc is not None:
        raise exc
    detail = f"{type(exc).__name__}: {exc}" if exc is not None else "unknown error"
    api = getattr(ctx, "api", None)
    context = getattr(ctx, "context", None)
    if ctx is not None:
        _trace_event(ctx, where, "crash", detail)
    fail = getattr(api, "fail", None)
    if fail is None or context is None:
        return
    try:
        fail(
            f"numty internal error in {where}: {detail} "
            "(result is unchecked; please report this)",
            context,
            code=NUMTY_INTERNAL,
        )
    except Exception:  # reporting must never be the thing that crashes
        if _numty_debug():
            raise


# Startup gates (issue #164 step 2): hooks key on private numpy module
# paths and on mypy internals, so a version drift can silently unhook
# everything (#67: constructor hooks never fired on numpy 2.2-2.4).
# Tested versions are pinned in pyproject.toml (mypy==2.4.1,
# numpy>=2.5); warn loudly outside them.
_TESTED_MYPY_VERSION = "2.4.1"
_TESTED_MYPY_MAJOR = 2
_TESTED_MYPY_MINOR = 4
_MIN_NUMPY_VERSION: tuple[int, ...] = (2, 5)

# Hooked fullnames that must resolve in the loaded stubs. If any is
# missing, hooks for that module silently never fire, so the first
# hook lookup warns naming them.
_CANARY_FULLNAMES: tuple[str, ...] = (
    "numpy._core.multiarray.zeros",
    "numpy._core.multiarray.asarray",
    "numpy._core.fromnumeric.reshape",
    "numpy._core.einsumfunc.einsum",
    "numpy.linalg._linalg.norm",
    "numpy.ndarray.reshape",
    "numpy.ndarray.__matmul__",
    "numpy.ufunc.__call__",
)

_CANARY_BY_MODULE: dict[str, tuple[str, ...]] = {}
for _canary in _CANARY_FULLNAMES:
    _module, _, _ = _canary.rpartition(".")
    _CANARY_BY_MODULE[_module] = _CANARY_BY_MODULE.get(_module, ()) + (_canary,)
del _canary, _module


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Leading numeric components of a version string ("2.4.1" -> (2, 4, 1))."""
    parts: list[int] = []
    for piece in version.strip().split("."):
        num = ""
        for ch in piece:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        parts.append(int(num))
    return tuple(parts)


def _mypy_version_warning(version: str) -> str | None:
    """Loud warning when `version` leaves the tested mypy range, else None."""
    got = _parse_version_tuple(version)
    if len(got) < 2 or got[0] != _TESTED_MYPY_MAJOR or got[1] != _TESTED_MYPY_MINOR:
        return (
            f"numty: warning: untested mypy version {version} "
            f"(tested with {_TESTED_MYPY_VERSION}); "
            "hooks may silently not fire (see issue #164)"
        )
    return None


def _numpy_version_warning(version: str) -> str | None:
    """Loud warning when `version` is below the minimum numpy, else None."""
    got = _parse_version_tuple(version)
    if not got:
        return None
    if got < _MIN_NUMPY_VERSION:
        want = ".".join(str(p) for p in _MIN_NUMPY_VERSION)
        return (
            f"numty: warning: untested numpy version {version} "
            f"(minimum supported is {want}); "
            "hooks may silently not fire (see issue #164)"
        )
    return None


def _bail(
    ctx: FunctionContext | MethodContext,
    op: str,
    reason: str,
) -> Type:
    """Give up on one call: trace it, optionally fail loudly, keep stubs.

    Central helper for the 506 `return ctx.default_return_type` sites:
    it records {op, bail, reason} to NUMTY_TRACE and, under
    NUMTY_STRICT=1, reports [numty-unsolved]. Strict fails for every
    bail, even scalar-only ones like np.sqrt(8): the Any born there
    poisons the known-shape truediv whose hook mypy then skips, so a
    known-shape gate would leave that loss silent. New hooks should
    call this instead of returning the default directly so the reason
    stays specific; _guard_mixed already applies the same policy
    generically to every hook that returns the default.
    """
    _trace_event(ctx, op, "bail", reason)
    if _numty_strict():
        ctx.api.fail(
            f"numty could not prove {op}: {reason} (result is unchecked)",
            ctx.context,
            code=NUMTY_UNSOLVED,
        )
    return ctx.default_return_type


def _warn_startup(message: str) -> None:
    try:
        print(message, file=sys.stderr)
    except Exception:  # reporting must never be the thing that crashes
        pass
