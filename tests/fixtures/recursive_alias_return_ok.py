"""Issue #243: a call returning a recursive alias stays clean."""
from pathlib import Path

JSON = dict[str, "JSON"] | list["JSON"] | None | bool | int | float | str
type JSON2 = dict[str, JSON2] | list[JSON2] | str | None


def read_json(uri: str | Path) -> JSON:
    raise NotImplementedError


def read_json2(uri: str | Path) -> JSON2:
    raise NotImplementedError


def caller(p: Path) -> None:
    a = read_json(p)
    reveal_type(a)
    b = read_json2(p)
    reveal_type(b)
