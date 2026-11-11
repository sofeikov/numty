"""Trace sink (Phase 2, #196). Pure env/JSONL, stdlib + mypy types only.

No imports from the package. Leaf of the module DAG:
_trace (leaf) <- _errors <- __init__.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _numty_trace_path() -> str | None:
    """JSONL trace sink from NUMTY_TRACE, or None when tracing is off."""
    path = os.environ.get("NUMTY_TRACE")
    return path if path else None


def _numty_strict() -> bool:
    """Whether NUMTY_STRICT=1 asks for a loud error on every lost shape."""
    return os.environ.get("NUMTY_STRICT") == "1"


def _trace_event(
    ctx: object,
    op: str,
    outcome: str,
    reason: str,
) -> None:
    """Append one {file, line, col, op, outcome, reason} record to NUMTY_TRACE.

    Only modules that are actually re-checked emit events, so run mypy
    with a fresh --cache-dir when collecting a trace. Failures to write
    never break checking.
    """
    path = _numty_trace_path()
    if path is None:
        return
    try:
        api: object = getattr(ctx, "api", None)
        file = str(getattr(api, "path", "") or "")
        context = getattr(ctx, "context", None)
        line = int(getattr(context, "line", 0) or 0)
        col = int(getattr(context, "column", 0) or 0)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "file": file,
                        "line": line,
                        "col": col,
                        "op": op,
                        "outcome": outcome,
                        "reason": reason,
                    }
                )
                + "\n"
            )
    except OSError:
        return
