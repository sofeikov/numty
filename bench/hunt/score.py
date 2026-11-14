"""Score bug-hunt areas: check clean files and mutants, run mutants.

Layout of one area (e.g. ``bench/hunt/filterpy/kalman``)::

    clean.py        annotated, standalone, numpy-only extract (must check clean)
    harness.py      ``run(mod) -> dict[str, object]``: calls into ``mod`` with
                    fixed seeds and sizes that never coincide
    mutants/*.py    full copies of clean.py with one seeded bug each; the first
                    line is ``# MUTANT: <one-line description>``
    ref_check.py    (optional) compares clean.py against the upstream library

Usage::

    python bench/hunt/score.py bench/hunt/filterpy/kalman [more areas...]
    python bench/hunt/score.py --static-only AREA      # skip the runtime part

For each area it writes ``score.json`` and prints a summary. A mutant is
*caught* in a mode when mypy reports an error on it in that mode that the
clean file does not have (matched by message and code); the clean file
should report none. At runtime a mutant is ``crash`` (raises),
``silent`` (runs, outputs differ from clean), ``same`` (outputs equal clean:
not a real bug for this harness) or ``timeout``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = {"default": HERE / "mypy.ini", "names": HERE / "mypy-names.ini"}
ERR = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+): error: (?P<msg>.*?)(?:\s+\[(?P<code>[\w-]+)\])?$")

RUNNER = r"""
import importlib.util, json, sys, traceback
import numpy as np

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def arr(a):
    a = np.asarray(a)
    try:
        c = a.astype(complex).ravel()
    except (TypeError, ValueError):
        return {"repr": repr(a.tolist())}
    return {"shape": list(a.shape), "re": c.real.tolist(), "im": c.imag.tolist()}

def norm(x):
    if isinstance(x, (np.ndarray, np.generic, int, float, complex, bool)):
        return arr(x)
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    if isinstance(x, dict):
        return {str(k): norm(v) for k, v in x.items()}
    return {"repr": repr(x)}

harness, target, out = sys.argv[1:4]
try:
    h = load(harness, "_harness")
    m = load(target, "_target")
    res = {"ok": True, "out": norm(h.run(m))}
except BaseException as e:
    res = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:300]}
with open(out, "w") as f:
    json.dump(res, f)
"""


def norm_msg(msg: str) -> str:
    """Drop the line numbers numty puts in unknown names (``?U@946b``)."""
    return re.sub(r"@\d+[a-z]*", "@", msg)


def match(reported: str, errs: dict) -> str:
    """Map a path as mypy printed it to one of the input files."""
    rep = Path(reported)
    for k in errs:
        kp = Path(k)
        if rep.is_absolute() and rep == kp:
            return k
        if not rep.is_absolute() and kp.parts[-len(rep.parts):] == rep.parts:
            return k
    return reported


def mypy(files: list[Path], mode: str, cache: Path) -> dict[str, list[dict]]:
    cmd = [sys.executable, "-m", "mypy", "--config-file", str(MODES[mode]),
           "--cache-dir", str(cache / mode), "--no-error-summary", "--show-error-codes",
           "--hide-error-context", *map(str, files)]
    # numty must be importable (installed in the environment); the repo root is
    # not put on PYTHONPATH, because then mypy prints paths relative to it.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    errs: dict[str, list[dict]] = {str(f): [] for f in files}
    for line in proc.stdout.splitlines():
        m = ERR.match(line)
        if not m:
            continue
        errs.setdefault(match(m["file"], errs), []).append({"line": int(m["line"]), "msg": m["msg"], "code": m["code"]})
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"mypy failed ({proc.returncode}): {proc.stderr[-2000:]}")
    stray = [k for k in errs if k not in {str(f) for f in files}]
    if stray:
        raise RuntimeError(f"could not match mypy paths to input files: {stray[:3]}")
    return errs


def close(a, b) -> bool:
    import numpy as np
    if isinstance(a, dict) and isinstance(b, dict):
        if "shape" in a and "shape" in b:
            if a["shape"] != b["shape"]:
                return False
            return bool(np.allclose(a["re"], b["re"], rtol=1e-7, atol=1e-10, equal_nan=True)
                        and np.allclose(a["im"], b["im"], rtol=1e-7, atol=1e-10, equal_nan=True))
        if set(a) != set(b):
            return False
        return all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def run(harness: Path, target: Path, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.json"
        try:
            subprocess.run([sys.executable, "-c", RUNNER, str(harness), str(target), str(out)],
                           capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        if not out.exists():
            return {"ok": False, "error": "runner died"}
        return json.loads(out.read_text())


def plugin_digest() -> str:
    """A hash of the numty_plugin sources this interpreter imports.

    mypy's cache does not notice edits to the plugin's submodules, so the
    cache directory is keyed on them: a plugin edit gets a fresh cache.
    """
    import numty_plugin

    digest = hashlib.sha256()
    for path in sorted(Path(numty_plugin.__file__).parent.rglob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def score(area: Path, static_only: bool, timeout: float) -> dict:
    clean = area / "clean.py"
    mutants = sorted((area / "mutants").glob("*.py"))
    cache = Path(tempfile.gettempdir()) / "numty-hunt-cache" / plugin_digest() / area.as_posix().replace("/", "_")
    result: dict = {"area": str(area), "clean": {}, "mutants": []}
    per_mode = {mode: mypy([clean, *mutants], mode, cache) for mode in MODES}
    for mode in MODES:
        result["clean"][mode] = per_mode[mode].get(str(clean), [])
    clean_run = None
    if not static_only and (area / "harness.py").exists():
        clean_run = run(area / "harness.py", clean, timeout)
        result["clean"]["runtime"] = "ok" if clean_run.get("ok") else clean_run.get("error")
    for m in mutants:
        first = m.read_text().splitlines()[0] if m.read_text() else ""
        rec = {"file": m.name, "desc": first.removeprefix("# MUTANT:").strip()}
        for mode in MODES:
            errs = per_mode[mode].get(str(m), [])
            new_errs = list(errs)
            for e in result["clean"][mode]:  # the mutant's first line shifts lines, so match by text
                same = next((x for x in new_errs if norm_msg(x["msg"]) == norm_msg(e["msg"]) and x["code"] == e["code"]),
                            None)
                if same is not None:
                    new_errs.remove(same)
            rec[mode] = {"caught": bool(new_errs), "errors": errs, "new_errors": new_errs}
        if clean_run is not None and clean_run.get("ok"):
            r = run(area / "harness.py", m, timeout)
            if not r.get("ok"):
                rec["runtime"] = "timeout" if r.get("error") == "timeout" else "crash"
                rec["runtime_error"] = r.get("error")
            else:
                rec["runtime"] = "same" if close(r["out"], clean_run["out"]) else "silent"
        result["mutants"].append(rec)
    real = [r for r in result["mutants"] if r.get("runtime") != "same"]
    result["summary"] = {
        "clean_errors": {mode: len(result["clean"][mode]) for mode in MODES},
        "mutants": len(result["mutants"]),
        "real_bugs": len(real),
        "caught_default": sum(r["default"]["caught"] for r in real),
        "caught_names_extra": sum(r["names"]["caught"] and not r["default"]["caught"] for r in real),
        "silent": sum(r.get("runtime") == "silent" for r in real),
        "silent_caught_any": sum(r.get("runtime") == "silent" and (r["default"]["caught"] or r["names"]["caught"])
                                 for r in real),
    }
    (area / "score.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("areas", nargs="+", type=Path)
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    for area in a.areas:
        r = score(area.resolve(), a.static_only, a.timeout)
        s = r["summary"]
        print(f"{area}: clean errors {s['clean_errors']}; real bugs {s['real_bugs']}/{s['mutants']} mutants; "
              f"caught default {s['caught_default']}, +names {s['caught_names_extra']}; "
              f"silent {s['silent']} (caught {s['silent_caught_any']})")
        for rec in r["mutants"]:
            flag = "D" if rec["default"]["caught"] else ("N" if rec["names"]["caught"] else "-")
            print(f"  [{flag}] {rec.get('runtime', '?'):7} {rec['file']}: {rec['desc']}")


if __name__ == "__main__":
    main()
