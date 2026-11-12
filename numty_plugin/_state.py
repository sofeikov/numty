"""Process-lifetime state bundle (Phase 4 Phase 0, #198; part of #192).

Module-level singleton STATE; NumtyPlugin.__init__ does
self.state = STATE (never NumtyState()). No clearing in hooks.
Region B is keyed (ns, fullname) since #209.

Region C is not pruned (Phase 2 of #198), on purpose:
- An unknown's TypeVarId is minted once per source site. A type that
  another module already holds can contain that id. If the entry is
  dropped and minted again, the two ids differ, and overload
  resolution reads that as a disagreement and falls back to Any.
- The mypy plugin API has no hook for "a re-check of module M starts",
  so there is no point in a daemon run where a module's entries are
  known to be unused.
- The cost of no pruning is one small dict entry per unknown site per
  (line, column) seen in the life of a dmypy process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy.types import Type, TypeVarId, TypeVarLikeType, TypeVarType

    from ._forms import _DivForm, _LinForm, _MonoForm


@dataclass
class NumtyState:
    # Region A - content-addressed, process-stable, never cleared.
    expr_forms: dict[str, _LinForm | _DivForm | _MonoForm] = field(default_factory=dict)
    expr_ids: dict[str, TypeVarId] = field(default_factory=dict)
    # Region B (Phase 1 #198): scope-bound, keyed (ns, fullname).
    # Outer expr key stays global (expr vars are namespace="").
    seen_vars: dict[tuple[str, str], TypeVarType] = field(default_factory=dict)
    expr_constituents: dict[str, dict[tuple[str, str], TypeVarType]] = field(default_factory=dict)
    implicit_scope_vars: dict[tuple[str, str], TypeVarLikeType] = field(default_factory=dict)
    # Region C - ephemeral display/debug (never cleared in Phase 0).
    unknown_ids: dict[str, TypeVarId] = field(default_factory=dict)
    unknown_names: dict[str, str] = field(default_factory=dict)
    unknown_per_line: dict[tuple[str, int], int] = field(default_factory=dict)
    recheck_depth: int = 0
    # Instance-mirrored sets (per-Plugin lifetime; STATE fields alias the
    # owning plugin's objects - see NumtyPlugin.__init__).
    fam_late_bound: set[str] = field(default_factory=set)
    startup_canary_checked: set[str] = field(default_factory=set)
    # Region D - index positions (issues #285-287): producer calls and
    # Index[...] annotations record the size an int array's values index
    # into ("positions on P"). Keys are "module@line@column" for exact
    # producer-call sites and "module@line" for line entries; values are
    # (shape dims or None, recorded size) lists. Entries dedupe by content
    # so rechecks and overload candidates stay stable; lookups with zero
    # or several matches stay silent.
    index_on: dict[str, list[tuple[list[int | Type] | None, int | Type]]] = field(
        default_factory=dict
    )


STATE: NumtyState = NumtyState()
