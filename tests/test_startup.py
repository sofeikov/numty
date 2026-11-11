"""Startup gates: untested mypy/numpy warn, missing hooks warn (issue #164).

Step 2 of #164: `plugin(version)` compares the mypy version with the
tested range, the installed numpy version is checked on first use,
and a canary resolves a short list of hooked fullnames in the loaded
stubs, warning (once) naming any that is missing.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numty_plugin
from numty_plugin import NumtyPlugin

ROOT = Path(__file__).resolve().parent.parent


def _plug_options():
    from mypy.options import Options

    return Options()


def _run_mypy(path: Path, cache: Path) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(cache), str(path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_parse_version_tuple() -> None:
    assert numty_plugin._parse_version_tuple("2.4.1") == (2, 4, 1)
    assert numty_plugin._parse_version_tuple("2.5") == (2, 5)
    assert numty_plugin._parse_version_tuple("1.11.0") == (1, 11, 0)
    assert numty_plugin._parse_version_tuple("") == ()
    assert numty_plugin._parse_version_tuple("dev") == ()


def test_mypy_version_warning_only_outside_tested_range() -> None:
    assert numty_plugin._mypy_version_warning("2.4.1") is None
    assert numty_plugin._mypy_version_warning("2.4.0") is None
    old = numty_plugin._mypy_version_warning("1.11")
    assert old is not None and "1.11" in old and "2.4.1" in old
    new = numty_plugin._mypy_version_warning("2.5.0")
    assert new is not None and "2.5.0" in new
    assert numty_plugin._mypy_version_warning("") is not None


def test_numpy_version_warning_only_below_minimum() -> None:
    assert numty_plugin._numpy_version_warning("2.5.4") is None
    assert numty_plugin._numpy_version_warning("2.5") is None
    assert numty_plugin._numpy_version_warning("2.9") is None
    old = numty_plugin._numpy_version_warning("2.4.3")
    assert old is not None and "2.4.3" in old and "2.5" in old
    assert numty_plugin._numpy_version_warning("") is None


def test_plugin_returns_plugin_and_warns_on_untested_mypy(capsys) -> None:
    assert numty_plugin.plugin("2.4.1") is NumtyPlugin
    assert capsys.readouterr().err == ""
    assert numty_plugin.plugin("9.9-fake") is NumtyPlugin
    err = capsys.readouterr().err
    assert "untested mypy version 9.9-fake" in err
    assert "issue #164" in err


def test_plugin_warns_on_old_numpy(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        numty_plugin, "_installed_numpy_version", lambda: "2.4.3"
    )
    assert numty_plugin.plugin("2.4.1") is NumtyPlugin
    err = capsys.readouterr().err
    assert "untested numpy version 2.4.3" in err


def _plug_with_lookup(mapping: dict[str, bool]) -> NumtyPlugin:
    plug = NumtyPlugin(_plug_options())

    def fake_lookup(fullname: str):
        if mapping.get(fullname, False):
            return SimpleNamespace(node=object())
        return None

    setattr(plug, "lookup_fully_qualified", fake_lookup)
    return plug


def test_canary_silent_when_all_hooks_resolve(capsys) -> None:
    mapping = {name: True for name in numty_plugin._CANARY_FULLNAMES}
    plug = _plug_with_lookup(mapping)
    for name in numty_plugin._CANARY_FULLNAMES:
        plug._ensure_startup_check(name)
    assert capsys.readouterr().err == ""


def test_canary_warns_naming_missing_hook(capsys) -> None:
    missing = "numpy._core.multiarray.asarray"
    mapping = {name: True for name in numty_plugin._CANARY_FULLNAMES}
    mapping[missing] = False
    plug = _plug_with_lookup(mapping)
    plug._ensure_startup_check("numpy._core.multiarray.zeros")
    err = capsys.readouterr().err
    assert "missing from the loaded numpy stubs" in err
    assert missing in err
    assert "issue #164" in err
    # Once per module: a second lookup for the same module stays silent.
    plug._ensure_startup_check("numpy._core.multiarray.asarray")
    assert capsys.readouterr().err == ""


def test_canary_ignores_non_numpy_and_unloaded_modules(capsys) -> None:
    plug = _plug_with_lookup({})
    plug._ensure_startup_check("builtins.len")
    assert capsys.readouterr().err == ""
    # Trigger itself unresolvable: module not loaded yet, retry later.
    plug._ensure_startup_check("numpy._core.multiarray.zeros")
    assert capsys.readouterr().err == ""
    assert plug._startup_canary_checked == set()


def test_canary_only_judges_the_loaded_module(capsys) -> None:
    # Only the multiarray module is "loaded"; linalg is never touched,
    # so its canary entries must not warn.
    mapping = {
        "numpy._core.multiarray.zeros": True,
        "numpy._core.multiarray.asarray": True,
    }
    plug = _plug_with_lookup(mapping)
    plug._ensure_startup_check("numpy._core.multiarray.zeros")
    assert capsys.readouterr().err == ""


def test_supported_env_warns_nothing(tmp_path: Path) -> None:
    src = tmp_path / "startup_repro.py"
    src.write_text(
        "import numpy as np\n"
        "from numpy.linalg import norm\n"
        "a = np.zeros((2, 3))\n"
        "b = np.asarray([[1.0, 2.0], [3.0, 4.0]])\n"
        "c = np.reshape(a, (3, 2))\n"
        "d = np.einsum('ij,jk->ik', a, c)\n"
        "e = norm(b)\n"
        "f = a.reshape(3, 2)\n"
        "g = a @ c\n"
        "h = np.sqrt(a)\n",
        encoding="utf-8",
    )
    result = _run_mypy(src, tmp_path / "cache")
    out = result.stdout + result.stderr
    assert "untested mypy version" not in out, out
    assert "untested numpy version" not in out, out
    assert "missing from the loaded numpy stubs" not in out, out
