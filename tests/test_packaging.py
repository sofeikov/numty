"""Out-of-tree adoption packaging (issue #31).

Editable installs used to resolve `from numty import F32` to `Any`
(silent, everything green), and wheel installs reported
`Skipping analyzing "numty": module is installed, but missing library
stubs or py.typed marker` (top-level `py.typed` is not honored for
single-file modules). The fix is a real package layout
(`numty/__init__.py` + `numty/py.typed`).

These tests build a wheel, install it into a fresh venv, and run mypy
from a directory that cannot see the checkout (scrubbed `PYTHONPATH` /
`MYPYPATH`), asserting the out-of-tree experience is clean.
"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

uv = shutil.which("uv")
pytestmark = pytest.mark.skipif(uv is None, reason="uv is required (see README quickstart)")


def _run(cmd: list[str], **kwargs) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, **kwargs
    )


def _build_wheel(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run(
        [uv, "build", "--wheel", "-o", str(out_dir), "--no-build-logs"],
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    wheels = sorted(out_dir.glob("numty-*.whl"))
    assert len(wheels) == 1, [p.name for p in out_dir.iterdir()]
    return wheels[0]


def test_wheel_has_package_marker(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path / "dist")
    names = set(zipfile.ZipFile(wheel).namelist())
    # Package layout: mypy honors numty/py.typed, not top-level py.typed.
    assert "numty/__init__.py" in names
    assert "numty/py.typed" in names
    # The mypy plugin and tools still ship as top-level modules.
    assert "numty_plugin.py" in names


def test_wheel_install_checks_out_of_tree(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path / "dist")
    venv = tmp_path / "isoenv"
    result = _run([uv, "venv", str(venv)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    python = venv / "bin" / "python"
    result = _run(
        [uv, "pip", "install", "--python", str(python),
         str(wheel), "mypy==2.4.1", "numpy>=2"],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "check.py").write_text(
        "from numty import F32\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from typing import reveal_type\n"
        "x: F32[3]\n"
        "reveal_type(x)\n"
    )
    (proj / "mypy.ini").write_text("[mypy]\nplugins = numty_plugin\n")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "MYPYPATH")
    }
    result = _run(
        [str(venv / "bin" / "mypy"), "--cache-dir", str(tmp_path / "cache"),
         "check.py"],
        cwd=proj,
        env=env,
    )
    assert result.returncode == 0, result.stdout
    assert "Skipping analyzing" not in result.stdout
    assert "tuple[Literal[3]]" in result.stdout, result.stdout
    assert "Revealed type is \"Any\"" not in result.stdout
