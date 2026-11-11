"""Per-file give-up report over the NUMTY_TRACE stream (issue #164 step 3).

`python -m numty_report` runs mypy with tracing on a fresh cache and
prints per-file counts of solved / bailed / unhooked / crashed calls,
plus the top bail reasons -- the same idea as mypy's
`--any-exprs-report`. Bails stay silent by default (erroring on them
punishes correct code; that is what `NUMTY_STRICT=1` is for); the
report is how you measure them instead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

OUTCOMES = ("solved", "bail", "unhooked", "crash")


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="numty_report",
        description=(
            "Run mypy with NUMTY_TRACE on a fresh cache and report "
            "per-file solved/bailed/unhooked/crashed hook counts plus "
            "the top bail reasons. Extra arguments are forwarded to "
            "mypy (e.g. `python -m numty_report --top 5 repro.py`). "
            "The child mypy runs without NUMTY_DEBUG so crashes are "
            "recorded as trace events instead of aborting the run."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="how many top bail reasons to show (default: 10)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="mypy cache dir (default: a fresh temporary dir, removed afterwards)",
    )
    parser.add_argument(
        "--trace-file",
        default=None,
        help="where to write the NUMTY_TRACE stream (default: a temporary file)",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="include third-party stub and site-packages files "
        "(default: project .py files only)",
    )
    args, mypy_args = parser.parse_known_args(argv[1:])
    return args, mypy_args


def _load_events(trace_file: Path) -> list[dict]:
    events: list[dict] = []
    try:
        text = trace_file.read_text(encoding="utf-8")
    except OSError:
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _display_path(file: str) -> str:
    try:
        return str(Path(file).relative_to(Path.cwd()))
    except ValueError:
        return file


def _is_project_file(file: str) -> bool:
    """Whether an event source is project code rather than stub noise.

    A fresh-cache mypy run also evaluates hooks inside third-party
    stubs (attribute defaults, overload candidates), which would
    drown the per-file table. Those live in site-packages, typeshed,
    or `.pyi` files; project code is `.py` everywhere else. The numty
    package itself (issue 235: 76 events for numty/__init__.py from a
    2-op file) is library noise like stubs, hidden unless asked.
    """
    lowered = file.replace("\\", "/").lower()
    if lowered.endswith(".pyi"):
        return False
    for marker in ("site-packages/", "dist-packages/", "/typeshed/", "typeshed/"):
        if marker in lowered:
            return False
    # Issue 235: numty's own package is not user code.
    if "/numty/__init__.py" in lowered or "/numty_plugin/" in lowered:
        return False
    return True


def _is_numpy_op(op: str) -> bool:
    """Whether an op is a numpy operation (issue 235).

    Trace includes non-numpy ops (SupportsIndex.__index__, builtins.len,
    polars methods) with bail outcome; those are not numpy calls and
    drown the report (1,741 non-numpy sites on omnisync). Only numpy.*
    counts in the main table; the rest are hidden unless --all-files.
    """
    return op.startswith("numpy.")


def _print_report(events: list[dict], top: int, show_all: bool) -> None:
    hidden = sum(1 for e in events if not _is_project_file(str(e.get("file", ""))))
    if not show_all:
        events = [e for e in events if _is_project_file(str(e.get("file", "")))]
    # Issue 235: dedup by (file, line, col, op); a site is solved if any
    # event there is solved (mypy visits each site several times: overload
    # candidates, re-inference). Non-numpy ops are hidden unless asked.
    # Crash beats solved (plugin bug must show), solved beats bail (trial
    # bails from non-selected overloads hide), bail beats unhooked.
    if not show_all:
        events = [e for e in events if _is_numpy_op(str(e.get("op", "")))]
    # Priority: crash first, then solved, bail, unhooked.
    rank = {"crash": 0, "solved": 1, "bail": 2, "unhooked": 3}
    best: dict[tuple[str, int, int, str], dict] = {}
    for event in events:
        outcome = event.get("outcome")
        if outcome not in OUTCOMES:
            continue
        try:
            line = int(event.get("line", 0))
        except (TypeError, ValueError):
            line = 0
        try:
            col = int(event.get("col", event.get("column", 0)))
        except (TypeError, ValueError):
            col = 0
        key = (str(event.get("file", "")), line, col, str(event.get("op", "")))
        prev = best.get(key)
        if prev is None or rank.get(str(outcome), 9) < rank.get(str(prev.get("outcome")), 9):
            best[key] = event
    events = list(best.values())
    per_file: dict[str, Counter] = {}
    bail_reasons: Counter[tuple[str, str]] = Counter()
    unhooked_ops: Counter[str] = Counter()
    crashes: list[dict] = []
    for event in events:
        outcome = event.get("outcome")
        if outcome not in OUTCOMES:
            continue
        shown = _display_path(str(event.get("file", "")))
        counts = per_file.setdefault(shown, Counter())
        counts[outcome] += 1
        if outcome == "bail":
            bail_reasons[(str(event.get("op", "")), str(event.get("reason", "")))] += 1
        elif outcome == "unhooked":
            unhooked_ops[str(event.get("op", ""))] += 1
        elif outcome == "crash":
            crashes.append(event)
    total = sum(sum(counts.values()) for counts in per_file.values())
    print(f"numty report: {total} traced call(s) in {len(per_file)} file(s)", end="")
    if hidden and not show_all:
        print(f" ({hidden} stub/site-packages event(s) hidden; --all-files to show)")
    else:
        print()
    if not events:
        print("no numty trace events (no hooked calls re-checked in this run)")
        return
    width = max([len(name) for name in per_file] + [len("file")])
    header = (
        f"{'file':<{width}}  {'solved':>6}  {'bailed':>6}  "
        f"{'unhooked':>8}  {'crashed':>7}  {'total':>5}"
    )
    print(header)
    for name in sorted(per_file):
        counts = per_file[name]
        row_total = sum(counts[o] for o in OUTCOMES)
        print(
            f"{name:<{width}}  {counts['solved']:>6}  {counts['bail']:>6}  "
            f"{counts['unhooked']:>8}  {counts['crash']:>7}  {row_total:>5}"
        )
    if bail_reasons:
        print(f"top bail reasons (top {top}):")
        for (op, reason), count in bail_reasons.most_common(top):
            print(f"  {count}x {op}: {reason}")
    if unhooked_ops:
        print(f"top unhooked ops (top {top}):")
        for op, count in unhooked_ops.most_common(top):
            print(f"  {count}x {op}")
    for event in crashes:
        shown = _display_path(str(event.get("file", "")))
        print(
            f"crash {shown}:{event.get('line')}: "
            f"{event.get('op')}: {event.get('reason')}"
        )


def main(argv: list[str]) -> int:
    args, mypy_args = _parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="numty-report-cache-") as tmp_cache:
        cache_dir = args.cache_dir or tmp_cache
        if args.trace_file is not None:
            trace_path = Path(args.trace_file)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            result = _run_mypy(cache_dir, args.trace_file, mypy_args)
            events = _load_events(trace_path)
        else:
            with tempfile.TemporaryDirectory(prefix="numty-report-trace-") as tmp_trace:
                trace_file = str(Path(tmp_trace) / "trace.jsonl")
                result = _run_mypy(cache_dir, trace_file, mypy_args)
                events = _load_events(Path(trace_file))
    _print_report(events, args.top, args.all_files)
    return result.returncode


def _run_mypy(
    cache_dir: str, trace_file: str, mypy_args: list[str]
) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "NUMTY_TRACE": trace_file}
    # Production behavior: crashes are recorded as `crash` trace events.
    # With NUMTY_DEBUG=1 (which the test suite sets) they would raise
    # instead and abort the mypy run, leaving no trace to report on.
    env.pop("NUMTY_DEBUG", None)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", cache_dir, *mypy_args],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")
    return result
