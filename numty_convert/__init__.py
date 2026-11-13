"""Convert jaxtyping annotations to numty static shapes.

jaxtyping checks shapes at runtime
(``Float[np.ndarray, "batch features"]``); numty checks them statically.
This one-time migration rewrites the annotations so the existing solver
can verify them without changing any runtime behavior:

    Float[np.ndarray, "batch features"]    ->  FlexArray[tuple[BATCH, FEATURES], np.dtype[Any]]
    Float64[np.ndarray, "batch features"]  ->  FloatArray[tuple[BATCH, FEATURES]]
    Int32[np.ndarray, "batch *rest"]       ->  FlexArray[tuple[BATCH, Unpack[Ts_REST]], np.dtype[np.int32]]

Names become shared TypeVars (same name unifies, exactly like
jaxtyping's runtime memo); fixed sizes become literals; ``...``, ``_``
and unchecked ``_name`` become fresh rank/dim variables; ``*name``
becomes a named ``*Ts``. Names are case-sensitive, as in jaxtyping:
TypeVars are upper-cased for readability unless that would merge two
distinct names (``B`` and ``b``), in which case both keep their exact
spelling. A dim that only the return annotation mentions (once) is
unknown to the caller, so it becomes ``int`` (``*tuple[int, ...]`` for
a rank).

Only parameter and return annotations of ``def`` signatures convert.
Anything the static solver cannot express is left untouched with a
warning: symbolic dims, broadcastable ``#name`` dims (size 1 *or* the
name -- no static equivalent), more than one variadic, non-numpy array
bases, stacked subscriptions, string annotations, jaxtyping subscripts
outside a signature (module/class variables, ``isinstance``), return
dims repeated without a binding parameter, and names whose TypeVar
would rebind something (a module binding or import, a builtin).

The output is the input with only the converted annotations spliced in,
one inserted import/TypeVar block, the ``# type: ignore[type-var]`` a
``*Ts`` signature needs, and jaxtyping imports nothing uses any more:
comments, pragmas, blank lines and formatting elsewhere stay
byte-identical, and a file with nothing to convert comes back unchanged.

Usage: ``python numty_convert.py src.py [dst.py]`` (stdout when dst is
omitted; ``-h`` for help). API: ``convert_source(src)`` returns
``(out, warnings)``.
"""

import argparse
import ast
import builtins
import io
import keyword
import re
import sys
import tokenize
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Exact dtype heads: jaxtyping name -> numpy type understood by numty.
# Heads in neither table (Key, BFloat16, ...) have no numpy equivalent
# and are left as written.
DTYPE_EXACT = {
    "Bool": "bool_",
    "UInt8": "uint8",
    "UInt16": "uint16",
    "UInt32": "uint32",
    "UInt64": "uint64",
    "Int8": "int8",
    "Int16": "int16",
    "Int32": "int32",
    "Int64": "int64",
    "Float16": "float16",
    "Float32": "float32",
    "Float64": "float64",
    "Complex64": "complex64",
    "Complex128": "complex128",
}
# Family heads: shapes convert precisely, dtypes stay loose.
DTYPE_LOOSE = {
    "Float",
    "Int",
    "UInt",
    "Integer",
    "Real",
    "Num",
    "Inexact",
    "Complex",
    "Shaped",
}
HEADS = set(DTYPE_EXACT) | DTYPE_LOOSE

_TYPE_VAR_IGNORE = "# type: ignore[type-var]"

# Spellings a dim TypeVar must never take: rebinding a builtin (`len`) or
# a name the converted text itself uses would change runtime behavior.
_RESERVED = (
    set(dir(builtins))
    | set(keyword.kwlist)
    | {"Any", "FlexArray", "FloatArray", "Literal", "TypeVar", "TypeVarTuple", "Unpack"}
)


# -- source splicing (shared with numty_codmod) ----------------------------
class Source:
    """Source text with ast positions mapped to string offsets.

    ast columns are UTF-8 byte offsets into a line; lines split on
    ``\\n``, ``\\r\\n`` and ``\\r`` only, exactly like the tokenizer.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.lines = io.StringIO(text, newline="").readlines()
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, lineno: int, col: int) -> int:
        if lineno > len(self.lines):
            return len(self.text)
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(
            line.encode("utf-8")[:col].decode("utf-8")
        )

    def span(self, node: ast.AST) -> tuple[int, int]:
        end_lineno = getattr(node, "end_lineno")
        end_col = getattr(node, "end_col_offset")
        return (
            self.offset(getattr(node, "lineno"), getattr(node, "col_offset")),
            self.offset(end_lineno, end_col),
        )

    def segment(self, node: ast.AST) -> str:
        start, end = self.span(node)
        return self.text[start:end]

    def newline(self) -> str:
        """The file's own line terminator (first one seen)."""
        for line in self.lines:
            stripped = line.rstrip("\r\n")
            if stripped != line:
                return line[len(stripped):]
        return "\n"


Edit = tuple[int, int, str]


def splice(text: str, edits: list[Edit]) -> str:
    """Replace each ``text[start:end]`` with its replacement.

    Edits must not overlap; insertions (``start == end``) at the same
    offset land in the order given.
    """
    out: list[str] = []
    pos = 0
    for start, end, repl in sorted(edits, key=lambda e: (e[0], e[1])):
        if start < pos:
            raise ValueError(f"overlapping edits at offset {start}")
        out.append(text[pos:start])
        out.append(repl)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def read_source(path: str) -> tuple[str, str]:
    """(text, encoding) of a Python file; newlines stay untranslated."""
    data = Path(path).read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    return data.decode(encoding), encoding


def run_cli(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    transform: Callable[[str], tuple[str, list[str]]],
) -> int:
    """Read ``args.src``, transform, write ``args.dst`` or stdout.

    I/O, decoding and syntax errors print one ``prog: error: ...`` line
    and exit 2 instead of a traceback.
    """
    src_path: str = args.src
    try:
        src, encoding = read_source(src_path)
        out, warnings = transform(src)
    except OSError as exc:
        parser.exit(2, f"{parser.prog}: error: {src_path}: {exc.strerror or exc}\n")
    except UnicodeDecodeError as exc:
        parser.exit(2, f"{parser.prog}: error: {src_path}: cannot decode: {exc.reason}\n")
    except SyntaxError as exc:
        where = f"{src_path}:{exc.lineno}" if exc.lineno else src_path
        parser.exit(2, f"{parser.prog}: error: {where}: syntax error: {exc.msg}\n")
    for warning in warnings:
        print(f"{src_path}: {warning}", file=sys.stderr)
    if args.dst is None:
        sys.stdout.write(out)
        return 0
    try:
        Path(args.dst).write_bytes(out.encode(encoding))
    except OSError as exc:
        parser.exit(2, f"{parser.prog}: error: {args.dst}: {exc.strerror or exc}\n")
    return 0


# -- dim strings -------------------------------------------------------------
def _parse_dims(dim_str: str) -> list[tuple[str, str]] | str:
    """Split a jaxtyping dim string into (kind, value) items.

    Kinds: fixed, named, anonymous (``_``, unchecked ``_name``),
    variadic (``...``, ``*_``), named_variadic (``*name``). A string
    return is the reason the spec is out of scope (symbolic dims, two
    variadics, broadcastable ``#`` dims). The treepath marker ``?``
    folds into the named dim it decorates (no PyTree here).
    """
    items: list[tuple[str, str]] = []
    seen_variadic = False
    for elem in dim_str.strip().split():
        if "," in elem and "(" not in elem:
            return "symbolic or stacked dims"
        if elem == "...":
            if seen_variadic:
                return "more than one variadic dim"
            seen_variadic = True
            items.append(("variadic", ""))
            continue
        markers = ""
        while elem[:1] in ("#", "*", "_", "?"):
            if elem[0] == "*" and "*" in markers:
                return "symbolic or stacked dims"
            markers += elem[0]
            elem = elem[1:]
        if "#" in markers:
            return f"broadcastable dim {markers}{elem!s} has no static equivalent"
        if "=" in elem and "(" not in elem:
            _, _, elem = elem.partition("=")
        variadic = "*" in markers
        if variadic:
            if seen_variadic:
                return "more than one variadic dim"
            seen_variadic = True
        if "_" in markers or not elem:
            # Unchecked (`_name`) or anonymous (`_`): any size, fresh.
            if elem and not elem.isidentifier():
                return "symbolic or stacked dims"
            items.append(("variadic" if variadic else "anonymous", ""))
        elif elem.isidentifier():
            items.append(("named_variadic" if variadic else "named", elem))
        else:
            try:
                size = int(elem)
            except ValueError:
                return "symbolic or stacked dims"
            if variadic:
                return "symbolic or stacked dims"
            items.append(("fixed", str(size)))
    return items


# -- conversion ----------------------------------------------------------------
@dataclass
class _Spec:
    """A jaxtyping subscript in a def signature."""

    node: ast.Subscript
    head: str
    items: list[tuple[str, str]] | None  # None: left as is (warned)
    np_prefix: str | None


@dataclass
class _Converted:
    node: ast.Subscript
    text: str
    typevars: set[str]
    tuplevartuples: set[str]
    singles: set[str]  # jaxtyping names bound (params only)
    variadics: set[str]
    anon_count: int
    need_literal: bool
    need_unpack: bool
    loose: bool
    flex: bool
    needs_ignore: bool


def _is_candidate(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in HEADS
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
    )


def _candidates(node: ast.AST) -> Iterator[ast.Subscript]:
    """Outermost jaxtyping-shaped subscripts under node."""
    if _is_candidate(node):
        assert isinstance(node, ast.Subscript)
        yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from _candidates(child)


def _mentioned_names(node: ast.Subscript) -> set[str]:
    """Every dim name a jaxtyping subscript's string spells, markers off.

    Loose on purpose (works on specs ``_parse_dims`` rejects): it only
    decides whether a return dim is tied to some parameter.
    """
    assert isinstance(node.slice, ast.Tuple)
    shape = node.slice.elts[1]
    if not isinstance(shape, ast.Constant) or not isinstance(shape.value, str):
        return set()
    names: set[str] = set()
    for elem in shape.value.split():
        name = elem.lstrip("#*_?")
        if "_" in elem[: len(elem) - len(name)]:
            continue  # unchecked: binds nothing
        name = name.partition("=")[0]
        if name.isidentifier():
            names.add(name)
    return names


def _string_mentions_head(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            try:
                inner = ast.parse(sub.value.strip(), mode="eval")
            except SyntaxError:
                continue
            if any(
                isinstance(n, ast.Name) and n.id in HEADS for n in ast.walk(inner)
            ):
                return True
    return False


def numpy_alias(tree: ast.Module) -> str | None:
    """The `np` in `import numpy as np` (or `numpy` for plain import)."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy":
                    return a.asname or "numpy"
    return None


class _Converter:
    def __init__(self, tree: ast.Module, source: Source) -> None:
        self.tree = tree
        self.source = source
        self.typevars: set[str] = set()
        self.tuplevartuples: set[str] = set()
        self.anon_count = 0
        self.need_literal = False
        self.need_unpack = False
        self.need_float_array = False
        self.need_flex_array = False
        self.used_loose = False
        self.warnings: list[str] = []
        # Pre-pass: every name bound at module/class level. A binding of
        # an emitted TypeVar name would be rebound by our decl (or vice
        # versa), and a binding of a jaxtyping head means the annotation
        # never was jaxtyping at runtime -- both bail out verbatim.
        self.assigned = _module_shadows(tree)
        self.numpy_alias = numpy_alias(tree)
        self.single_ids: dict[str, str | None] = {}
        self.variadic_ids: dict[str, str | None] = {}
        self.converted: list[_Converted] = []

    # -- names ---------------------------------------------------------
    @staticmethod
    def _ids(names: set[str]) -> dict[str, str | None]:
        """Upper-case each name unless that merges distinct names.

        jaxtyping names are case-sensitive (and ``"ß".upper() == "SS"``):
        a group of names that would fold together keeps exact spelling.
        Python folds identifiers by NFKC, so names that still coincide
        after that (``ﬁ`` and ``fi``) map to None: no faithful spelling.
        """
        groups: dict[str, set[str]] = {}
        for name in names:
            upper = name.upper()
            key = unicodedata.normalize("NFKC", upper)
            if not upper.isidentifier():
                key = unicodedata.normalize("NFKC", name)
            groups.setdefault(key, set()).add(name)
        ids: dict[str, str | None] = {}
        for members in groups.values():
            if len(members) == 1:
                (name,) = members
                upper = name.upper()
                ids[name] = upper if upper.isidentifier() else name
                continue
            exact = {unicodedata.normalize("NFKC", n) for n in members}
            for name in members:
                ids[name] = name if len(exact) == len(members) else None
        return ids

    def _name_id(self, name: str, variadic: bool, lineno: int) -> str | None:
        ids = self.variadic_ids if variadic else self.single_ids
        base = ids[name]
        if base is None:
            self.warnings.append(
                f"line {lineno}: dim {name} has no identifier distinct from"
                " another dim name; left as is"
            )
            return None
        ident = "Ts_" + base if variadic else base
        if ident in _RESERVED:
            self.warnings.append(
                f"line {lineno}: name {ident} would shadow a builtin or a"
                " name the converter emits; left as is"
            )
            return None
        # Registration requires passing this check, so registered names
        # never hold an assigned one: the membership test is the rule.
        if ident in self.assigned:
            self.warnings.append(
                f"line {lineno}: name {ident} already assigned; left as is"
            )
            return None
        return ident

    def _fresh(self, prefix: str, count: int, taken: set[str]) -> tuple[str, int]:
        planned = (
            {v for v in self.single_ids.values() if v is not None}
            | {"Ts_" + v for v in self.variadic_ids.values() if v is not None}
            | self.assigned
            | taken
        )
        while f"{prefix}{count}" in planned:
            count += 1
        return f"{prefix}{count}", count + 1

    # -- planning --------------------------------------------------------
    def _spec(self, node: ast.Subscript) -> _Spec:
        assert isinstance(node.value, ast.Name)
        assert isinstance(node.slice, ast.Tuple)
        head = node.value.id
        spec = _Spec(node, head, None, None)
        if head in self.assigned:
            # The head is rebound at module/class level: at runtime this
            # was never jaxtyping, so rewriting would change behavior.
            self.warnings.append(
                f"line {node.lineno}: name {head} shadows a jaxtyping head;"
                " left as is"
            )
            return spec
        array_node, shape_node = node.slice.elts
        if not (
            isinstance(array_node, ast.Attribute)
            and array_node.attr in ("ndarray", "NDArray")
            and isinstance(array_node.value, ast.Name)
            and array_node.value.id in ("np", "npt", "numpy")
        ):
            # Stacked or foreign base: leave the whole subtree verbatim so
            # no unused TypeVars leak from heads nested inside.
            self.warnings.append(
                f"line {node.lineno}: non-numpy array base; left as is"
            )
            return spec
        if not isinstance(shape_node, ast.Constant) or not isinstance(
            shape_node.value, str
        ):
            return spec
        parsed = _parse_dims(shape_node.value)
        if isinstance(parsed, str):
            self.warnings.append(f"line {node.lineno}: {parsed}; left as is")
            return spec
        if array_node.attr == "ndarray" and array_node.value.id != "npt":
            spec.np_prefix = array_node.value.id
        else:
            spec.np_prefix = self.numpy_alias
        spec.items = parsed
        return spec

    def _convert(
        self,
        spec: _Spec,
        returns_of: tuple[set[str], set[str]] | None,
        mentioned: set[str] | None = None,
        return_uses: Counter[tuple[str, str]] | None = None,
    ) -> _Converted | None:
        """Build the replacement text; None (warned) leaves it verbatim.

        ``returns_of`` is the (single, variadic) names the def's converted
        parameters bind, when spec sits in the return annotation: any
        other dim there is unconstrained for the caller, so it becomes
        ``int`` (an unsolvable TypeVar would only make mypy ask for an
        annotation at every call) -- unless a parameter left verbatim
        ``mentioned`` it, whose tie the return cannot express, or the
        return spells it more than once (``return_uses``): ``"m m"``
        promises a square, which ``int, int`` would silently drop.
        """
        node = spec.node
        assert spec.items is not None
        if returns_of is not None:
            for kind, value in spec.items:
                if kind not in ("named", "named_variadic"):
                    continue
                bound = returns_of[0] if kind == "named" else returns_of[1]
                if value in bound:
                    continue
                if value in (mentioned or set()):
                    self.warnings.append(
                        f"line {node.lineno}: dim {value} is tied to a"
                        " parameter left as is; left as is"
                    )
                    return None
                if (return_uses or Counter())[(kind, value)] > 1:
                    self.warnings.append(
                        f"line {node.lineno}: dim {value} repeats in the"
                        " return but no parameter binds it; left as is"
                    )
                    return None
        typevars: set[str] = set()
        tvts: set[str] = set()
        singles: set[str] = set()
        variadics: set[str] = set()
        count = self.anon_count
        need_literal = need_unpack = needs_ignore = False
        dims: list[str] = []
        for kind, value in spec.items:
            if kind == "fixed":
                need_literal = True
                dims.append(f"Literal[{value}]")
            elif kind in ("named", "anonymous"):
                if returns_of is not None and (
                    kind == "anonymous" or value not in returns_of[0]
                ):
                    dims.append("int")
                    continue
                if kind == "named":
                    tvar = self._name_id(value, False, node.lineno)
                    if tvar is None:
                        return None
                    singles.add(value)
                else:
                    tvar, count = self._fresh(
                        "_ANON", count, self.typevars | typevars
                    )
                typevars.add(tvar)
                dims.append(tvar)
            else:
                need_unpack = True
                if returns_of is not None and (
                    kind == "variadic" or value not in returns_of[1]
                ):
                    dims.append("Unpack[tuple[int, ...]]")
                    continue
                if kind == "named_variadic":
                    tvar = self._name_id(value, True, node.lineno)
                    if tvar is None:
                        return None
                    variadics.add(value)
                else:
                    tvar, count = self._fresh(
                        "Ts_ANON", count, self.tuplevartuples | tvts
                    )
                tvts.add(tvar)
                needs_ignore = True
                dims.append(f"Unpack[{tvar}]")
        if not dims:
            shape = "tuple[()]"
        else:
            shape = f"tuple[{', '.join(dims)}]"
        loose = spec.head not in DTYPE_EXACT
        if spec.head == "Float64" and not need_unpack:
            text = f"FloatArray[{shape}]"
            flex = False
        else:
            np_ = spec.np_prefix
            if np_ is None:
                self.warnings.append(
                    f"line {node.lineno}: no `import numpy` to spell the"
                    " dtype; left as is"
                )
                return None
            dtype_arg = "Any" if loose else f"{np_}.{DTYPE_EXACT[spec.head]}"
            text = f"FlexArray[{shape}, {np_}.dtype[{dtype_arg}]]"
            flex = True
        return _Converted(
            node=node,
            text=text,
            typevars=typevars,
            tuplevartuples=tvts,
            singles=singles,
            variadics=variadics,
            anon_count=count,
            need_literal=need_literal,
            need_unpack=need_unpack,
            loose=loose,
            flex=flex,
            needs_ignore=needs_ignore,
        )

    def _commit(self, conv: _Converted) -> None:
        self.typevars |= conv.typevars
        self.tuplevartuples |= conv.tuplevartuples
        self.anon_count = conv.anon_count
        self.need_literal |= conv.need_literal
        self.need_unpack |= conv.need_unpack
        self.used_loose |= conv.loose
        if conv.flex:
            self.need_flex_array = True
        else:
            self.need_float_array = True
        self.converted.append(conv)

    # -- driver ------------------------------------------------------------
    def run(self) -> list[Edit]:
        self._alias_warnings()
        defs = sorted(
            (
                n
                for n in ast.walk(self.tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            key=lambda d: (d.lineno, d.col_offset),
        )
        in_signature: set[int] = set()
        plans: list[tuple[list[_Spec], list[_Spec]]] = []
        for fn in defs:
            a = fn.args
            params = [
                p.annotation
                for p in (
                    *a.posonlyargs,
                    *a.args,
                    *([a.vararg] if a.vararg else []),
                    *a.kwonlyargs,
                    *([a.kwarg] if a.kwarg else []),
                )
                if p.annotation is not None
            ]
            anns: list[tuple[ast.expr, bool]] = [(p, False) for p in params]
            if fn.returns is not None:
                anns.append((fn.returns, True))
            param_specs: list[_Spec] = []
            return_specs: list[_Spec] = []
            for ann, is_return in anns:
                if _string_mentions_head(ann):
                    self.warnings.append(
                        f"line {ann.lineno}: string annotation not converted;"
                        " left as is"
                    )
                for cand in _candidates(ann):
                    in_signature.add(id(cand))
                    spec = self._spec(cand)
                    (return_specs if is_return else param_specs).append(spec)
            plans.append((param_specs, return_specs))
        for cand in _candidates(self.tree):
            assert isinstance(cand.value, ast.Name)
            if id(cand) not in in_signature and cand.value.id not in self.assigned:
                self.warnings.append(
                    f"line {cand.lineno}: jaxtyping subscript outside a def"
                    " signature; left as is"
                )
        singles: set[str] = set()
        variadics: set[str] = set()
        for param_specs, return_specs in plans:
            for spec in (*param_specs, *return_specs):
                for kind, value in spec.items or []:
                    if kind == "named":
                        singles.add(value)
                    elif kind == "named_variadic":
                        variadics.add(value)
        self.single_ids = self._ids(singles)
        self.variadic_ids = self._ids(variadics)
        for param_specs, return_specs in plans:
            bound_singles: set[str] = set()
            bound_variadics: set[str] = set()
            mentioned: set[str] = set()
            for spec in param_specs:
                mentioned |= _mentioned_names(spec.node)
                if spec.items is None:
                    continue
                conv = self._convert(spec, None)
                if conv is not None:
                    self._commit(conv)
                    bound_singles |= conv.singles
                    bound_variadics |= conv.variadics
            # A return-only name is free only if the return spells it
            # once; one a verbatim return spec mentions is tied as well.
            return_uses: Counter[tuple[str, str]] = Counter()
            for spec in return_specs:
                if spec.items is not None:
                    return_uses.update(
                        item
                        for item in spec.items
                        if item[0] in ("named", "named_variadic")
                    )
                    continue
                for name in _mentioned_names(spec.node):
                    return_uses[("named", name)] += 2
                    return_uses[("named_variadic", name)] += 2
            for spec in return_specs:
                if spec.items is None:
                    continue
                conv = self._convert(
                    spec,
                    (bound_singles, bound_variadics),
                    mentioned,
                    return_uses,
                )
                if conv is not None:
                    self._commit(conv)
        if not self.converted:
            return []
        edits: list[Edit] = []
        for conv in self.converted:
            start, end = self.source.span(conv.node)
            edits.append((start, end, conv.text))
        edits.extend(self._ignore_edits())
        edits.extend(self._header_edits())
        return edits

    def _alias_warnings(self) -> None:
        for stmt in self.tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "jaxtyping":
                for alias in stmt.names:
                    if alias.asname is not None and alias.name in HEADS:
                        self.warnings.append(
                            f"line {stmt.lineno}: aliased jaxtyping import "
                            f"{alias.name} as {alias.asname}; not converted"
                        )

    def _ignore_edits(self) -> list[Edit]:
        """``# type: ignore[type-var]`` on every line with a ``*Ts`` shape.

        numty needs it wherever a TypeVarTuple meets ndarray's
        ``tuple[int, ...]`` bound (see README). A converted span that
        crosses lines collapses onto one, so the comment goes on the last
        line of each group of touching spans.
        """
        groups: list[list[int]] = []
        for start, end in sorted(
            (c.node.lineno, c.node.end_lineno or c.node.lineno)
            for c in self.converted
        ):
            if groups and start <= groups[-1][1]:
                groups[-1][1] = max(groups[-1][1], end)
            else:
                groups.append([start, end])
        lines: set[int] = set()
        for conv in self.converted:
            if conv.needs_ignore:
                for start, end in groups:
                    if start <= conv.node.lineno <= end:
                        lines.add(end)
        if not lines:
            return []
        comments: dict[int, tuple[int, str]] = {}
        line_ends: set[int] = set()
        try:
            for tok in tokenize.generate_tokens(
                io.StringIO(self.source.text, newline="").readline
            ):
                if tok.type == tokenize.COMMENT:
                    comments[tok.start[0]] = (tok.start[1], tok.string)
                elif tok.type in (tokenize.NEWLINE, tokenize.NL):
                    line_ends.add(tok.start[0])
        except (tokenize.TokenError, SyntaxError):
            pass
        edits: list[Edit] = []
        for line in sorted(lines):
            base = self.source.starts[line - 1]
            if line in comments:
                col, text = comments[line]
                m = re.match(r"#\s*type:\s*ignore\b(\[([^\]]*)\])?", text)
                if m is None:
                    edits.append((base + col, base + col, _TYPE_VAR_IGNORE + "  "))
                elif m.group(1) is not None:
                    codes = [c.strip() for c in m.group(2).split(",") if c.strip()]
                    if "type-var" not in codes:
                        new = f"[{', '.join([*codes, 'type-var'])}]"
                        s, e = base + col + m.start(1), base + col + m.end(1)
                        edits.append((s, e, new))
                # A bare type-ignore comment already covers it.
            elif line in line_ends:
                content = self.source.lines[line - 1].rstrip("\r\n").rstrip(" \t")
                pos = base + len(content)
                edits.append((pos, pos, "  " + _TYPE_VAR_IGNORE))
            else:
                self.warnings.append(
                    f"line {line}: could not place `{_TYPE_VAR_IGNORE}`"
                    " (line continues); add it by hand"
                )
        return edits

    def _header_edits(self) -> list[Edit]:
        """One inserted block: missing imports, then TypeVar decls.

        It goes right after the last top-level import that precedes the
        first converted statement (so every name it needs, and any
        environment setup between imports, already ran), or just above
        that statement when no import precedes it.
        """
        body = self.tree.body
        first = min(
            i
            for i, stmt in enumerate(body)
            for conv in self.converted
            if stmt.lineno <= conv.node.lineno <= (stmt.end_lineno or stmt.lineno)
        )
        before = body[:first]
        imports = [s for s in before if isinstance(s, (ast.Import, ast.ImportFrom))]
        if imports:
            last = imports[-1]
            pos = self.source.starts[last.end_lineno or last.lineno]
        else:
            stmt = body[first]
            top = min(
                [stmt.lineno]
                + [d.lineno for d in getattr(stmt, "decorator_list", [])]
            )
            pos = self.source.starts[top - 1]

        def imported(module: str) -> set[str]:
            return {
                a.name
                for s in imports
                if isinstance(s, ast.ImportFrom) and s.module == module and s.level == 0
                for a in s.names
                if a.asname is None or a.asname == a.name
            }

        typing_names = set()
        if self.typevars:
            typing_names.add("TypeVar")
        if self.tuplevartuples:
            typing_names.add("TypeVarTuple")
        if self.need_literal:
            typing_names.add("Literal")
        if self.used_loose:
            typing_names.add("Any")
        if self.need_unpack:
            typing_names.add("Unpack")
        typing_names -= imported("typing")
        numty_names = set()
        if self.need_float_array:
            numty_names.add("FloatArray")
        if self.need_flex_array:
            numty_names.add("FlexArray")
        numty_names -= imported("numty")
        lines: list[str] = []
        if typing_names:
            lines.append(f"from typing import {', '.join(sorted(typing_names))}")
        if numty_names:
            lines.append(f"from numty import {', '.join(sorted(numty_names))}")
        for name in sorted(self.typevars):
            lines.append(f"{name} = TypeVar({name!r}, bound=int)")
        for name in sorted(self.tuplevartuples):
            lines.append(f"{name} = TypeVarTuple({name!r})")
        nl = self.source.newline()
        return [(pos, pos, "".join(line + nl for line in lines))]


def convert_source(src: str) -> tuple[str, list[str]]:
    """Convert jaxtyping annotations in src; return (out, warnings)."""
    tree = ast.parse(src)
    conv = _Converter(tree, Source(src))
    edits = conv.run()
    warnings = list(dict.fromkeys(conv.warnings))
    if not edits:
        return src, warnings
    return _prune_jaxtyping(splice(src, edits)), warnings


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="numty_convert.py",
        description="Rewrite jaxtyping shape annotations into numty static"
        " shapes. Only the converted annotations change; warnings go to"
        " stderr.",
    )
    parser.add_argument("src", help="Python file to convert")
    parser.add_argument(
        "dst", nargs="?", help="where to write the result (default: stdout)"
    )
    args = parser.parse_args(argv[1:])
    return run_cli(parser, args, convert_source)


def _module_shadows(tree: ast.Module) -> set[str]:
    """Names bound at module/class level (def-time scope).

    Function/lambda bodies are skipped: their bindings happen at call
    time, after annotations have already evaluated. Defaults, decorators,
    and class bodies execute at def time, so they are visited. Imports
    bind too (``from config import BATCH``), except the jaxtyping heads
    themselves.
    """
    shadows: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                shadows.add(node.name)
            exprs: list[ast.expr] = [
                *getattr(node, "decorator_list", []),
                *node.args.defaults,
                *[d for d in node.args.kw_defaults if d is not None],
            ]
            returns = getattr(node, "returns", None)
            if returns is not None:
                exprs.append(returns)
            for expr in exprs:
                visit(expr)
            return
        if isinstance(node, ast.ClassDef):
            shadows.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            shadows.add(node.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                shadows.add(a.asname or a.name.split(".")[0])
            return
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                if node.module == "jaxtyping" and a.asname is None:
                    continue
                shadows.add(a.asname or a.name)
            return
        elif isinstance(node, ast.ExceptHandler) and node.name:
            shadows.add(node.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return shadows


def _used_names(tree: ast.Module) -> set[str]:
    """Loaded names, plus names inside strings that parse as expressions.

    Strings cover forward-reference annotations (``"Float[...]"``),
    ``__all__`` entries and ``cast("...")``; over-counting only keeps an
    import alive.
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                inner = ast.parse(node.value.strip(), mode="eval")
            except SyntaxError:
                continue
            used.update(n.id for n in ast.walk(inner) if isinstance(n, ast.Name))
    return used


def _prune_jaxtyping(text: str) -> str:
    """Drop jaxtyping imports whose names have no remaining uses.

    A fully converted file no longer references the heads, so keeping
    ``from jaxtyping import Float`` would drag a runtime dependency into a
    file that no longer needs it. Names still used anywhere (unconverted
    annotations, value uses, ``TYPE_CHECKING`` blocks, string annotations,
    ``__all__``) keep their import. An import sharing its line with other
    statements is kept as is.
    """
    tree = ast.parse(text)
    source = Source(text)
    used = _used_names(tree)
    edits: list[Edit] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "jaxtyping":
            kept = [
                a
                for a in stmt.names
                if (a.asname or a.name) in used or a.name not in HEADS
            ]
            prefix = f"from {'.' * stmt.level}jaxtyping import "
        elif isinstance(stmt, ast.Import) and "jaxtyping" not in used:
            # Keep `import jaxtyping[.x]` only while `jaxtyping` is used;
            # otherwise strip just those names (`import jaxtyping, os`
            # keeps `os`).
            kept = [
                a
                for a in stmt.names
                if (a.asname or a.name).split(".")[0] != "jaxtyping"
            ]
            prefix = "import "
        else:
            continue
        if len(kept) == len(stmt.names):
            continue
        start, end = source.span(stmt)
        last_line = stmt.end_lineno or stmt.lineno
        last = source.lines[last_line - 1]
        before = text[source.starts[stmt.lineno - 1]:start]
        after = text[end:source.starts[last_line - 1] + len(last.rstrip("\r\n"))]
        own_lines = not before.strip() and (
            not after.strip() or after.strip().startswith("#")
        )
        if kept:
            names = ", ".join(
                a.name if a.asname is None else f"{a.name} as {a.asname}"
                for a in kept
            )
            edits.append((start, end, prefix + names))
        elif own_lines:
            edits.append((source.starts[stmt.lineno - 1], source.starts[last_line], ""))
    return splice(text, edits)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
