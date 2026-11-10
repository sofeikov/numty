"""Out-of-tree adoption packaging (issues #31, #147).

Wheel installs used to report `Skipping analyzing "numty"` (#31: fixed
by the `numty/` package + `numty/py.typed` marker). The mypy plugin and
the two tools then shipped as top-level single-file modules, which mypy
can never mark typed: `import numty_plugin` reported `module is
installed, but missing library stubs or py.typed marker` (neither a
top-level `py.typed` next to them nor a co-located `.pyi` is honored for
an installed top-level `.py`). Editable installs additionally
split-brained: `numty` imported live from the checkout while the tools
resolved to stale site-packages copies. The fix is one regular package
per tool (`numty_plugin/`, `numty_convert/`, `numty_codmod/`,
`numty_report/`), each with
its own `py.typed`, keeping the same top-level import names — so the
editable install is a single `.pth` and everything imports live.

These tests build a wheel, install it (and an editable checkout) into
fresh venvs, and run mypy from a directory that cannot see the checkout
(scrubbed `PYTHONPATH` / `MYPYPATH`), asserting the out-of-tree
experience is clean and single-sourced.
"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

TOOLS = ("numty_plugin", "numty_convert", "numty_codmod", "numty_report")

uv = shutil.which("uv")
pytestmark = pytest.mark.skipif(uv is None, reason="uv is required (see README quickstart)")


def _run(cmd: list[str], **kwargs) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, **kwargs
    )


def _scrubbed_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "MYPYPATH")
    }


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


def _fresh_venv(path: Path) -> Path:
    result = _run([uv, "venv", str(path)], cwd=path.parent)
    assert result.returncode == 0, result.stderr
    return path / "bin" / "python"


def test_wheel_has_package_markers(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path / "dist")
    names = set(zipfile.ZipFile(wheel).namelist())
    # Every importable top-level name is a package with its own marker.
    assert "numty/__init__.py" in names
    assert "numty/py.typed" in names
    for tool in TOOLS:
        assert f"{tool}/__init__.py" in names
        assert f"{tool}/py.typed" in names
        # No stale single-file copies shadowing the packages.
        assert f"{tool}.py" not in names


def test_wheel_install_checks_out_of_tree(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path / "dist")
    python = _fresh_venv(tmp_path / "isoenv")
    result = _run(
        [uv, "pip", "install", "--python", str(python),
         str(wheel), "mypy==2.4.1", "numpy>=2.5"],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "check.py").write_text(
        "import numty_convert\n"
        "import numty_codmod\n"
        "import numty_plugin\n"
        "from numty import F32\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from typing import reveal_type\n"
        "x: F32[3]\n"
        "reveal_type(x)\n"
    )
    (proj / "mypy.ini").write_text("[mypy]\nplugins = numty_plugin\n")
    result = _run(
        [str(python.parent / "mypy"), "--cache-dir", str(tmp_path / "cache"),
         "check.py"],
        cwd=proj,
        env=_scrubbed_env(),
    )
    assert result.returncode == 0, result.stdout
    assert "Skipping analyzing" not in result.stdout
    assert "tuple[Literal[3]]" in result.stdout, result.stdout
    assert "Revealed type is \"Any\"" not in result.stdout


def test_editable_install_imports_live_from_one_place(tmp_path: Path) -> None:
    python = _fresh_venv(tmp_path / "editenv")
    result = _run(
        [uv, "pip", "install", "--python", str(python),
         "-e", str(ROOT), "numpy>=2.5"],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    probe = (
        "import importlib.util\n"
        "for name in ['numty', 'numty_plugin', 'numty_convert', 'numty_codmod', 'numty_report']:\n"
        "    spec = importlib.util.find_spec(name)\n"
        "    print(name, spec.origin)\n"
    )
    result = _run(
        [str(python), "-c", probe],
        cwd=tmp_path,
        env=_scrubbed_env(),
    )
    assert result.returncode == 0, result.stderr
    origins = dict(
        line.split(" ", 1) for line in result.stdout.strip().splitlines()
    )
    assert set(origins) == {
        "numty", "numty_plugin", "numty_convert", "numty_codmod", "numty_report",
    }, result.stdout
    root = str(ROOT)
    stale = {
        name: origin
        for name, origin in origins.items()
        if not origin.startswith(root)
    }
    assert not stale, f"editable install split-brains: {stale}"
