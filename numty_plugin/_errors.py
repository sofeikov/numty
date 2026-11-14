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
    from mypy.nodes import Expression
    from mypy.options import Options
    from mypy.plugin import FunctionContext, FunctionSigContext, MethodContext, MethodSigContext
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
# On by default since issue 327 (opt-in before, issue 280).
# `F64[N] + F64[M]` runs when N == M, so it is not provably wrong, but a
# function generic in N and M has to work when they differ, and a body
# that pairs two different names is reported. On 3,600 lines of annotated
# scikit-learn code it caught 41 more of 149 seeded bugs (66 -> 107) for
# 3 false positives on the correct code, so it is worth having on.
# `disable_error_code = numty-names` turns it off: two different names
# then meet silently, and a broadcast of them is an unknown size (`?U`).
NUMTY_NAMES = ErrorCode(
    "numty-names",
    "Two different named sizes are used where one size is needed",
    "numty",
)
# Off by default (issue 308). `a[idx] = value` casts with
# casting="unsafe", so floats written into an integer array are
# truncated silently; that is right code when the truncation is meant,
# and `.astype(...)` on the value says so.
NUMTY_CAST = ErrorCode(
    "numty-cast",
    "An indexed assignment casts its value to a kind that loses information",
    "numty",
    default_enabled=False,
)
NUMTY_INTERNAL = ErrorCode(
    "numty-internal",
    "numty itself failed on this expression; the result is unchecked",
    "numty",
    sub_code_of=MISC,
)


def _code_enabled(options: Options | None, code: ErrorCode) -> bool:
    """Whether mypy reports `code` under these (per-module) options.

    The same rule as mypy's `Errors.is_error_code_enabled`: a disabled
    code is off, then an enabled one is on, then a disabled parent code
    turns it off, else the code's default. A check that emits a code only
    when it is on must agree with mypy: a message of a disabled code still
    fails the overload mypy is checking (see `_types._report_names`).
    """
    if options is None:
        return False
    if code in options.disabled_error_codes:
        return False
    if code in options.enabled_error_codes:
        return True
    if code.sub_code_of is not None and code.sub_code_of in options.disabled_error_codes:
        return False
    return code.default_enabled


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
    known-shape gate would leave that loss silent (see _report_unsolved
    for why the report cannot change any type). New hooks should
    call this instead of returning the default directly so the reason
    stays specific; _guard_mixed already applies the same policy
    generically to every hook that returns the default.
    """
    _trace_event(ctx, op, "bail", reason)
    _report_unsolved(ctx, op, reason)
    return ctx.default_return_type


def _trial_failed(ctx: object) -> bool:
    """Whether mypy's innermost speculative check has already failed.

    mypy checks an overload (or a union member, or an operator) with its
    errors filtered and drops it when any error was raised; a hook that
    runs after its arguments failed to fit is in a call mypy discards.
    """
    api: object = getattr(ctx, "api", None)
    errors: object = getattr(getattr(api, "msg", None), "errors", None)
    watchers = getattr(errors, "_watchers", None)
    if not isinstance(watchers, list) or not watchers:
        return False
    return bool(watchers[-1].has_new_errors())


def _report_unsolved(
    ctx: object, op: str, reason: str, failed_before: bool | None = None
) -> None:
    """Under NUMTY_STRICT=1, report one give-up as [numty-unsolved].

    The report must not change what mypy infers (issue 334). mypy
    matches overloads, splits unions and retries operators by checking
    a call with its errors filtered and then asking whether any error
    was raised: a plain `ctx.api.fail` from a hook makes that check
    fail, so a correct call gets `[call-overload]` / `[operator]` and
    another overload's (or member's) type. The report is therefore
    recorded past mypy's error watchers, which never see it. A check
    that already failed before the hook ran (an overload whose
    arguments do not fit) is a trial mypy throws away, so it reports
    nothing (see _trial_failed). `failed_before` is that state taken
    before the hook ran: an error the hook itself reported (a broadcast
    mismatch) does not make the call one mypy drops. mypy deduplicates
    the same message on one line when a call is checked more than once.
    """
    if not _numty_strict():
        return
    api: object = getattr(ctx, "api", None)
    msg: object = getattr(api, "msg", None)
    errors: object = getattr(msg, "errors", None)
    fail = getattr(api, "fail", None)
    if errors is None or fail is None:
        return
    watchers = getattr(errors, "_watchers", None)
    if not isinstance(watchers, list):
        return
    if failed_before is None:
        failed_before = _trial_failed(ctx)
    if failed_before:
        return
    message = f"numty could not prove {op}: {reason} (result is unchecked)"
    context = getattr(ctx, "context", None)
    if not watchers:
        fail(message, context, code=NUMTY_UNSOLVED)
        return
    try:
        setattr(errors, "_watchers", [])
        fail(message, context, code=NUMTY_UNSOLVED)
    finally:
        setattr(errors, "_watchers", watchers)


def _warn_startup(message: str) -> None:
    try:
        print(message, file=sys.stderr)
    except Exception:  # reporting must never be the thing that crashes
        pass


def _quiet_type(
    ctx: FunctionContext | MethodContext | FunctionSigContext | MethodSigContext, expr: Expression
) -> Type:
    """The type of an operand expression, with its own errors held back.

    For hooks that read an operand while mypy tries the items of an
    overload: an error the operand replays there (an `[arg-type]` of a
    call in it) fails the item, and the operator gets an `[operator]` or
    `[arg-type]` follow-on (issues 356, 364). mypy reports the operand's
    errors when it checks the operand itself.
    """
    with ctx.api.msg.filter_errors():
        return ctx.api.get_expression_type(expr)


def _checked_operand(ctx: MethodSigContext, expr: Expression) -> tuple[Type, bool]:
    """The type of an operand expression, as `_quiet_type`, and whether checking it reports an error (issue 363).

    `v - np.dot(A, x)` with a names report inside `np.dot`: mypy checks
    the right operand once per item of `__sub__`, the report fails every
    item, and mypy adds `Unsupported operand types` with two identical
    types. The operand's own report stands for the operator.
    """
    with ctx.api.msg.filter_errors() as watcher:
        tp = ctx.api.get_expression_type(expr)
    return tp, watcher.has_new_errors()
