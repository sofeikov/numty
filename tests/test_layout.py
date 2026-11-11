"""Layout guardrails for the numty_plugin package split (Phase 0, #192/#194).

The monolith ``numty_plugin/__init__.py`` will be split into submodules
with ``__init__`` kept as a facade. This file pins what the split must
not break, without knowing submodule names in advance:

- every name tests import from ``numty_plugin`` stays reachable as
  ``numty_plugin.<name>`` (the facade re-export contract);
- the process-lifetime singletons stay one object across the facade
  and any submodule that defines them (today: facade only; later: facade
  + submodule must agree);
- plugin entry, hook-table identity, cache round trip, head table
  (numty / plugin / codmod) and monkeypatch seams keep working.

No subprocess, no hypothesis: importing ``numty_plugin`` itself
transitively needs the pinned mypy dev dep which the whole suite
already requires.
"""
import ast
import importlib
import importlib.util
import pkgutil
import tomllib
from pathlib import Path

import pytest

import numty_plugin

ROOT = Path(__file__).resolve().parent.parent

# Every name the tests read/write through ``numty_plugin`` today.
# Sources: test_internal.py, test_startup.py, test_runtime.py,
# test_dim_algebra.py (``p.<name>`` + ``patch.setattr``),
# test_report.py (same seams as test_internal).
REQUIRED_EXPORTS: list[str] = [
    "plugin",
    "NumtyPlugin",
    "_DivForm",
    "_LinForm",
    "_MonoForm",
    "_parse_version_tuple",
    "_mypy_version_warning",
    "_numpy_version_warning",
    "_installed_numpy_version",
    "_CANARY_FULLNAMES",
    "_swallowed",
    "_broadcast_result",
    "_FAM_HEAD_DTYPES",
    "_promote_strong",
    "_promote_weak",
    "_NUMPY_FUNCTION_HOOKS",
    "_numpy_function_hooks",
    "_recheck_depth",
    "_expr_ids",
    "_implicit_scope_vars",
    "_seen_vars",
    "_expr_constituents",
    "_unknown_ids",
    "_unknown_names",
    "_unknown_per_line",
    "_lin_add",
    "_lin_scale",
    "_form_mul",
    "_form_to_coeff_vars",
    "_coeff_vars_to_form",
    "_canon_div",
    "_form_has_unknown",
    "_forms_provably_ne",
    "_form_never_size",
    "_UNKNOWN_PREFIX",
    "_eval_form",
    "_EXPR_PREFIX",
    "_form_from_fullname",
    "_expr_forms",
    "_expr_str",
    "_dim_add_const",
    "_dim_add2",
    "_dim_mul_const",
    "_dim_div",
    "_mul_dims",
    "_linear_product",
    "_resolve_dim",
    "_materialize_dim",
    "_fresh_unknown",
]

_SINGLETON_DICTS = (
    "_expr_forms",
    "_expr_ids",
    "_seen_vars",
    "_implicit_scope_vars",
    "_expr_constituents",
    "_unknown_ids",
    "_unknown_names",
    "_unknown_per_line",
)


def _known_modules() -> dict[str, object]:
    mods: dict[str, object] = {"numty_plugin": numty_plugin}
    try:
        paths = list(numty_plugin.__path__)  # type: ignore[attr-defined]
    except AttributeError:
        return mods
    for info in pkgutil.walk_packages(paths, prefix="numty_plugin."):
        mods[info.name] = importlib.import_module(info.name)
    return mods


def _owners(name: str) -> dict[str, object]:
    return {
        mod_name: getattr(mod, name)
        for mod_name, mod in _known_modules().items()
        if hasattr(mod, name)
    }


def _package_py_files() -> list[Path]:
    return sorted(
        p
        for p in (ROOT / "numty_plugin").rglob("*.py")
        if "__pycache__" not in p.parts
    )


@pytest.mark.parametrize("name", REQUIRED_EXPORTS)
def test_required_export_present(name: str) -> None:
    assert hasattr(numty_plugin, name), (
        f"numty_plugin.{name} missing: facade must re-export {name}"
    )


def test_fam_import_identity() -> None:
    from numty_plugin import _DivForm, _LinForm, _MonoForm

    assert _LinForm is numty_plugin._LinForm
    assert _DivForm is numty_plugin._DivForm
    assert _MonoForm is numty_plugin._MonoForm


def test_plugin_entry_returns_numty_plugin() -> None:
    assert callable(numty_plugin.plugin)
    assert numty_plugin.plugin("2.4.1") is numty_plugin.NumtyPlugin
    assert numty_plugin.plugin("") is numty_plugin.NumtyPlugin


def test_hook_table_shape_and_resolvable() -> None:
    table = numty_plugin._NUMPY_FUNCTION_HOOKS
    assert type(table) is dict
    assert len(table) > 100
    assert all(isinstance(k, str) and k.startswith("numpy.") for k in table)
    assert all(callable(v) for v in table.values())
    missing = []
    for fullname in table:
        module, _, attr = fullname.rpartition(".")
        try:
            found = hasattr(importlib.import_module(module), attr)
        except ImportError:
            found = False
        if not found:
            missing.append(fullname)
    assert missing == []


def test_hook_table_regeneration_identity() -> None:
    # Phase-2 pure moves must preserve the exact key set. Regeneration
    # (not a frozen snapshot) so adding a legit hook updates with code.
    regenerated = numty_plugin._numpy_function_hooks()
    assert set(regenerated) == set(numty_plugin._NUMPY_FUNCTION_HOOKS)


def test_hook_table_excludes_stub_twins() -> None:
    keys = set(numty_plugin._NUMPY_FUNCTION_HOOKS)
    assert not any(
        k.startswith(
            ("numpy.ma.", "numpy.emath.", "numpy.matlib.", "numpy.char.", "numpy.rec.")
        )
        for k in keys
    ), "exact-fullname dispatch must leave masked/emath/matlib stubs alone"


def test_canary_fullnames_present() -> None:
    canary = numty_plugin._CANARY_FULLNAMES
    assert isinstance(canary, tuple) and canary
    assert all(isinstance(n, str) and "." in n for n in canary)
    assert "numpy._core.multiarray.asarray" in canary


def test_singleton_dicts_exist_and_are_one_object() -> None:
    for name in _SINGLETON_DICTS:
        assert hasattr(numty_plugin, name), name
        facade = getattr(numty_plugin, name)
        assert type(facade) is dict, (name, type(facade))
        for owner, obj in _owners(name).items():
            if owner == "numty_plugin":
                continue
            assert obj is facade, (
                f"{name}: {owner} has a different object than the facade; "
                "facade must re-export the same dict, not a copy"
            )


def test_singleton_mutation_shares_object() -> None:
    for name in _SINGLETON_DICTS:
        sub_owners = [
            (owner, mod)
            for owner, mod in _known_modules().items()
            if owner != "numty_plugin" and hasattr(mod, name)
        ]
        if not sub_owners:
            continue
        facade_dict = getattr(numty_plugin, name)
        key = "__layout_probe__"
        try:
            facade_dict[key] = 1
            for owner, mod in sub_owners:
                assert getattr(mod, name).get(key) == 1, (name, owner)
        finally:
            facade_dict.pop(key, None)


def test_recheck_depth_is_int_and_agrees_across_modules() -> None:
    assert hasattr(numty_plugin, "_recheck_depth")
    facade = numty_plugin._recheck_depth
    assert type(facade) is int
    for owner, obj in _owners("_recheck_depth").items():
        if owner == "numty_plugin":
            continue
        assert type(obj) is int and obj == facade, owner


def _fixed_forms():
    from numty_plugin import _DivForm, _LinForm, _MonoForm

    return [
        _LinForm((), 0),
        _LinForm((), 7),
        _LinForm((("t.N", 1),), 0),
        _LinForm((("t.K", 2), ("t.N", -1)), 3),
        _MonoForm(1, ("t.K", "t.N")),
        _MonoForm(2, ("t.M", "t.M", "t.N")),
        _DivForm(_LinForm((("t.N", 1),), 0), 2, 1),
        _DivForm(_LinForm((("t.K", 3),), 2), 4, -1),
    ]


@pytest.mark.parametrize("form", _fixed_forms())
def test_form_cache_round_trip(form) -> None:
    assert numty_plugin._eval_form(ast.parse(repr(form), mode="eval").body) == form
    fullname = numty_plugin._EXPR_PREFIX + repr(form)
    try:
        assert numty_plugin._form_from_fullname(fullname) == form
    finally:
        numty_plugin._expr_forms.pop(fullname, None)


def test_head_triple_sync() -> None:
    import numty
    import numty_codmod

    heads = {
        name: obj._numty_dtype
        for name, obj in vars(numty).items()
        if isinstance(obj, type) and getattr(obj, "_numty_dtype", None) is not None
    }
    assert set(heads) == {"F64", "F32", "F16", "I64", "I32", "I16", "I8", "U8", "Bool"}
    plugin = {
        full.removeprefix("numty."): scalar
        for full, scalar in numty_plugin._FAM_HEAD_DTYPES.items()
    }
    assert set(plugin) == set(heads)
    import numpy as np

    for name, dtype in heads.items():
        scalar = plugin[name].removeprefix("numpy.")
        assert np.dtype(getattr(np, scalar)) == dtype, name
    assert set(heads) <= numty_codmod._NUMTY_ARRAY_TYPES


_SEAMS = (
    "_broadcast_result",
    "_swallowed",
    "_installed_numpy_version",
    "_resolve_dim",
    "_materialize_dim",
    "_fresh_unknown",
)


@pytest.mark.parametrize("name", _SEAMS)
def test_monkeypatch_seam_present_and_settable(name: str, monkeypatch) -> None:
    assert callable(getattr(numty_plugin, name)), name
    sentinel = lambda *a, **k: None  # noqa: E731
    monkeypatch.setattr(numty_plugin, name, sentinel)
    assert getattr(numty_plugin, name) is sentinel


def test_installed_numpy_version_seam_drives_plugin_warning(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(numty_plugin, "_installed_numpy_version", lambda: "2.4.3")
    assert numty_plugin.plugin("2.4.1") is numty_plugin.NumtyPlugin
    assert "untested numpy version 2.4.3" in capsys.readouterr().err


def test_package_layout_and_manifest() -> None:
    pkg_dir = ROOT / "numty_plugin"
    assert pkg_dir.is_dir()
    assert (pkg_dir / "__init__.py").exists()
    assert (pkg_dir / "py.typed").exists()
    assert not (ROOT / "numty_plugin.py").exists()
    assert Path(numty_plugin.__file__).resolve() == (pkg_dir / "__init__.py").resolve()
    assert Path(numty_plugin.__path__[0]).resolve() == pkg_dir.resolve()
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    wheels = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "numty_plugin" in wheels
    for py in _package_py_files():
        rel = py.relative_to(ROOT).with_suffix("")
        dotted = ".".join(rel.parts)
        if dotted.endswith(".__init__"):
            dotted = dotted[: -len(".__init__")]
        assert importlib.util.find_spec(dotted) is not None, (
            f"{py} not importable as {dotted}"
        )


def test_every_fail_site_has_code() -> None:
    plugin_dir = ROOT / "numty_plugin"
    files = _package_py_files()
    assert files, f"no .py files found under {plugin_dir} - glob broken?"
    assert plugin_dir / "__init__.py" in files, "scan must include __init__.py"
    total = 0
    missing: list[str] = []
    for py in files:
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fail"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "api"
            ):
                total += 1
                if not any(k.arg == "code" for k in node.keywords):
                    missing.append(f"{py.relative_to(plugin_dir)}:{node.lineno}")
    assert total > 100, total
    assert total >= 233, f"fail-site total dropped: {total} < 233"
    assert not missing, f"ctx.api.fail without code= at {missing}"


def test_every_broad_except_goes_through_policy_across_package() -> None:
    plugin_dir = ROOT / "numty_plugin"
    files = _package_py_files()
    assert files, f"no .py files found under {plugin_dir} - glob broken?"
    assert plugin_dir / "__init__.py" in files, "scan must include __init__.py"
    stray: list[str] = []
    total = 0
    for py in files:
        lines = py.read_text(encoding="utf-8").splitlines()
        rel = py.relative_to(plugin_dir)
        for i, line in enumerate(lines):
            if line.strip().startswith("except Exception"):
                total += 1
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else "<EOF>"
                inside_helper = "reporting must never" in line
                if not nxt.startswith("_swallowed(") and not inside_helper:
                    stray.append(f"{rel}:{i + 1}: {line.strip()} -> {nxt}")
    assert not stray, "broad except without _swallowed():\n" + "\n".join(stray)
    assert total >= 76, f"broad-except total dropped: {total} < 76"
