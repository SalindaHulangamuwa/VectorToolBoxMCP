"""Validate and normalise Chroma ``where`` / ``where_document`` filters.

Chroma's filter grammar is strict in ways that trip up callers (human or
model): exactly one operator per dict, ``$and``/``$or`` need at least two
children, ``{}`` is an error, and an array field matched with a bare list is
silently wrong. This module accepts the forgiving spellings, rewrites them to
the strict form, and rejects the genuinely invalid ones with a message that
says how to fix them - before anything reaches the database.

Rewrites (always safe):

* ``{}`` -> no filter.
* ``{"a": 1, "b": 2}`` -> ``{"$and": [{"a": 1}, {"b": 2}]}``.
* ``{"year": {"$gte": 2020, "$lte": 2024}}`` -> two clauses under ``$and``.
* ``{"$and": [x]}`` -> ``x`` (a single-child logical operator).
"""

from __future__ import annotations

import re
from typing import Any

from ...errors import ToolboxError


class FilterError(ToolboxError):
    """A where / where_document filter Chroma would reject or misread."""


COMPARISON_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}
INCLUSION_OPS = {"$in", "$nin"}
ARRAY_OPS = {"$contains", "$not_contains"}
METADATA_OPS = COMPARISON_OPS | INCLUSION_OPS | ARRAY_OPS
LOGICAL_OPS = {"$and", "$or"}
DOCUMENT_OPS = {"$contains", "$not_contains", "$regex", "$not_regex"}

_SCALAR = (str, int, float, bool)
_UNSUPPORTED_REGEX = [
    (re.compile(r"\(\?<?[=!]"), "look-around ((?=...), (?!...), (?<=...), (?<!...))"),
    (re.compile(r"\\[1-9]"), "backreferences (\\1, \\2, ...)"),
]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


# --------------------------------------------------------------------------
# where (metadata)
# --------------------------------------------------------------------------
def normalize_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a strict Chroma ``where`` dict, or ``None`` for no filter."""
    if where is None or where == {}:
        return None
    return _norm_where(where, "where")


def _norm_where(node: Any, path: str) -> dict[str, Any]:
    if not isinstance(node, dict) or not node:
        raise FilterError(f"{path} must be a non-empty object, got {node!r}.")

    if len(node) > 1:
        logical = [k for k in node if k in LOGICAL_OPS]
        if logical:
            raise FilterError(
                f"{path} mixes {', '.join(logical)} with other keys. Put every condition "
                "inside the $and / $or list."
            )
        return _norm_where({"$and": [{k: v} for k, v in node.items()]}, path)

    key, value = next(iter(node.items()))

    if key in LOGICAL_OPS:
        if not isinstance(value, list) or not value:
            raise FilterError(f"{path}.{key} must be a non-empty list of conditions.")
        children = [_norm_where(child, f"{path}.{key}[{i}]") for i, child in enumerate(value)]
        if len(children) == 1:
            return children[0]
        return {key: children}

    if key.startswith("$"):
        hint = ""
        if key == "$not":
            hint = " Chroma has no $not; negate the inner operator instead ($ne, $nin, $not_contains)."
        elif key in METADATA_OPS:
            hint = f" {key} applies to a field: {{'<field>': {{'{key}': ...}}}}."
        raise FilterError(
            f"{path} uses {key!r} at the top level. Allowed there: a field name, $and, $or.{hint}"
        )

    return _norm_field(key, value, f"{path}.{key}")


def _norm_field(field_name: str, value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, list):
        raise FilterError(
            f"{path} compares against a list. Use {{'$in': [...]}} to match any of several "
            "values, or {'$contains': x} to test whether an array field holds x."
        )
    if value is None:
        raise FilterError(
            f"{path} is null. Chroma cannot filter on null; it has no $exists operator either."
        )
    if isinstance(value, _SCALAR):
        return {field_name: {"$eq": value}}
    if not isinstance(value, dict) or not value:
        raise FilterError(f"{path} must be a scalar or an operator object, got {value!r}.")

    if len(value) > 1:
        return {"$and": [_norm_field(field_name, {op: v}, path) for op, v in value.items()]}

    op, operand = next(iter(value.items()))
    if op not in METADATA_OPS:
        hint = ""
        if op in {"$regex", "$not_regex"}:
            hint = " Regex works on documents only - use where_document."
        elif op == "$exists":
            hint = " Chroma has no $exists."
        raise FilterError(
            f"{path} uses unknown operator {op!r}. Metadata operators: "
            f"{', '.join(sorted(METADATA_OPS))}.{hint}"
        )

    if op in {"$gt", "$gte", "$lt", "$lte"}:
        if not _is_number(operand):
            raise FilterError(f"{path}.{op} needs a number, got {operand!r}.")
    elif op in {"$eq", "$ne"}:
        if not isinstance(operand, _SCALAR):
            raise FilterError(f"{path}.{op} needs a string, number or boolean, got {operand!r}.")
    elif op in INCLUSION_OPS:
        if not isinstance(operand, list) or not operand:
            raise FilterError(f"{path}.{op} needs a non-empty list, got {operand!r}.")
        kinds = {_scalar_type(v) for v in operand}
        same_type = len(kinds) == 1 or kinds <= {"int", "float"}
        if not all(isinstance(v, _SCALAR) for v in operand) or not same_type:
            raise FilterError(
                f"{path}.{op} needs a list of values of one type, got types {sorted(kinds)}."
            )
    else:  # $contains / $not_contains on an array metadata field
        if not isinstance(operand, _SCALAR):
            raise FilterError(
                f"{path}.{op} needs a single value matching the array's element type "
                f"(str, int, float or bool), got {operand!r}."
            )
    return {field_name: {op: operand}}


# --------------------------------------------------------------------------
# where_document (full-text)
# --------------------------------------------------------------------------
def normalize_where_document(where_document: dict[str, Any] | None) -> dict[str, Any] | None:
    if where_document is None or where_document == {}:
        return None
    return _norm_doc(where_document, "where_document")


def _norm_doc(node: Any, path: str) -> dict[str, Any]:
    if not isinstance(node, dict) or not node:
        raise FilterError(f"{path} must be a non-empty object, got {node!r}.")
    if len(node) > 1:
        if any(k in LOGICAL_OPS for k in node):
            raise FilterError(f"{path} mixes $and/$or with other keys.")
        return _norm_doc({"$and": [{k: v} for k, v in node.items()]}, path)

    key, value = next(iter(node.items()))
    if key in LOGICAL_OPS:
        if not isinstance(value, list) or not value:
            raise FilterError(f"{path}.{key} must be a non-empty list.")
        children = [_norm_doc(child, f"{path}.{key}[{i}]") for i, child in enumerate(value)]
        return children[0] if len(children) == 1 else {key: children}

    if key not in DOCUMENT_OPS:
        raise FilterError(
            f"{path} uses {key!r}. Document operators: {', '.join(sorted(DOCUMENT_OPS))}, "
            "combined with $and / $or. Metadata conditions belong in `where`."
        )
    if not isinstance(value, str) or not value:
        raise FilterError(f"{path}.{key} needs a non-empty string, got {value!r}.")
    if key in {"$regex", "$not_regex"}:
        validate_regex(value, f"{path}.{key}")
    return {key: value}


def validate_regex(pattern: str, path: str = "regex") -> None:
    for probe, what in _UNSUPPORTED_REGEX:
        if probe.search(pattern):
            raise FilterError(
                f"{path} uses {what}, which Chroma's regex engine does not support. "
                "Rewrite with plain alternation/classes, or combine $regex with $not_regex."
            )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise FilterError(f"{path} is not a valid regular expression: {exc}.") from exc


def build_text_filter(
    *,
    contains: list[str] | None = None,
    not_contains: list[str] | None = None,
    regex: list[str] | None = None,
    not_regex: list[str] | None = None,
    match: str = "all",
) -> dict[str, Any] | None:
    """Turn simple term lists into a ``where_document`` expression.

    ``match="all"`` ANDs the positive terms; ``match="any"`` ORs them. Negative
    terms (``not_contains`` / ``not_regex``) are always ANDed on, because
    "exclude X" only makes sense as a hard constraint.
    """
    if match not in {"all", "any"}:
        raise FilterError("match must be 'all' or 'any'.")
    positive = [{"$contains": t} for t in contains or []] + [{"$regex": r} for r in regex or []]
    negative = [{"$not_contains": t} for t in not_contains or []] + [
        {"$not_regex": r} for r in not_regex or []
    ]
    clauses: list[dict[str, Any]] = []
    if positive:
        if match == "any" and len(positive) > 1:
            clauses.append({"$or": positive})
        else:
            clauses.extend(positive)
    clauses.extend(negative)
    if not clauses:
        return None
    return normalize_where_document(clauses[0] if len(clauses) == 1 else {"$and": clauses})
