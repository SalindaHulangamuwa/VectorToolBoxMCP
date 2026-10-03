"""Weaviate filters from JSON - two accepted spellings, one strict output.

1. **Weaviate's native ``where`` format** - the one the REST/GraphQL API, the
   docs and Weaviate's own MCP server use::

       {"operator": "And", "operands": [
           {"path": ["year"], "operator": "GreaterThanEqual", "valueInt": 2020},
           {"path": ["tags"], "operator": "ContainsAny", "valueTextArray": ["ai", "ml"]}]}

   A plain ``"value"`` works in place of ``valueText``/``valueInt``/... .

2. **The Mongo-style operators the Chroma and Pinecone tools use**, so one
   filter vocabulary works across backends::

       {"year": {"$gte": 2020}, "tags": {"$in": ["ai", "ml"]}}

Both are normalised to the native format (``normalize``) and compiled to the
v4 client's ``Filter`` objects (``compile_filter``). With the collection's
property types supplied, unknown properties and type mismatches are caught
before the request is sent.

Special paths: ``id`` / ``_id`` (object UUID), ``_creationTimeUnix``,
``_lastUpdateTimeUnix``, ``len(prop)`` (string/array length, needs
``indexPropertyLength``), ``count(refProp)`` (number of references), and
``["refProp", "TargetClass", "prop"]`` to filter through a cross-reference.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ...errors import ToolboxError


class FilterError(ToolboxError):
    """A filter Weaviate would reject or misread."""


LEAF_OPERATORS = {
    "Equal": "equal",
    "NotEqual": "not_equal",
    "GreaterThan": "greater_than",
    "GreaterThanEqual": "greater_or_equal",
    "LessThan": "less_than",
    "LessThanEqual": "less_or_equal",
    "Like": "like",
    "ContainsAny": "contains_any",
    "ContainsAll": "contains_all",
    "ContainsNone": "contains_none",
    "IsNull": "is_none",
    "WithinGeoRange": "within_geo_range",
}
LOGICAL_OPERATORS = {"And", "Or", "Not"}
VALUE_KEYS = (
    "value", "valueText", "valueString", "valueInt", "valueNumber", "valueBoolean", "valueDate",
    "valueTextArray", "valueStringArray", "valueIntArray", "valueNumberArray", "valueBooleanArray",
    "valueDateArray", "valueGeoRange",
)
MONGO = {
    "$eq": "Equal", "$ne": "NotEqual", "$gt": "GreaterThan", "$gte": "GreaterThanEqual",
    "$lt": "LessThan", "$lte": "LessThanEqual", "$like": "Like", "$in": "ContainsAny",
    "$nin": "ContainsNone", "$all": "ContainsAll", "$contains": "ContainsAny",
    "$not_contains": "ContainsNone", "$is_null": "IsNull", "$geo": "WithinGeoRange",
}
ID_PATHS = {"id", "_id", "uuid"}
CREATED_PATHS = {"_creationTimeUnix", "creation_time", "_creation_time"}
UPDATED_PATHS = {"_lastUpdateTimeUnix", "last_update_time", "update_time", "_last_update_time"}
NUMERIC_OPS = {"GreaterThan", "GreaterThanEqual", "LessThan", "LessThanEqual"}
LIST_OPS = {"ContainsAny", "ContainsAll", "ContainsNone"}


# --------------------------------------------------------------------------
# normalise
# --------------------------------------------------------------------------
def normalize(where: Any) -> dict[str, Any] | None:
    """Return the canonical native-format filter, or None for no filter."""
    if where is None or where == {}:
        return None
    return _norm(where, "filters")


def _is_native(node: dict[str, Any]) -> bool:
    return "operator" in node or "path" in node or "operands" in node


def _norm(node: Any, at: str) -> dict[str, Any]:
    if not isinstance(node, dict) or not node:
        raise FilterError(f"{at} must be a non-empty object, got {node!r}.")
    if _is_native(node):
        return _norm_native(node, at)
    return _norm_mongo(node, at)


def _norm_native(node: dict[str, Any], at: str) -> dict[str, Any]:
    op = node.get("operator")
    if op in LOGICAL_OPERATORS:
        operands = node.get("operands")
        if not isinstance(operands, list) or not operands:
            raise FilterError(f"{at}: {op} needs a non-empty 'operands' list.")
        if op == "Not" and len(operands) != 1:
            raise FilterError(f"{at}: Not takes exactly one operand, got {len(operands)}.")
        children = [_norm(child, f"{at}.operands[{i}]") for i, child in enumerate(operands)]
        if op != "Not" and len(children) == 1:
            return children[0]
        return {"operator": op, "operands": children}
    if op not in LEAF_OPERATORS:
        raise FilterError(
            f"{at}: unknown operator {op!r}. Use one of: {', '.join(sorted(LEAF_OPERATORS))}, "
            "or And / Or / Not with 'operands'."
        )
    path = node.get("path")
    if isinstance(path, str):
        path = [path]
    if not isinstance(path, list) or not path or not all(isinstance(p, str) and p for p in path):
        raise FilterError(f"{at}: 'path' must be a property name or a list like ['ref', 'Target', 'prop'].")
    if len(path) > 1 and len(path) % 2 == 0:
        raise FilterError(
            f"{at}: a reference path alternates property and target collection and ends on a property, "
            f"e.g. ['author', 'Person', 'name']; got {path}."
        )
    given = [k for k in VALUE_KEYS if k in node]
    if len(given) > 1:
        raise FilterError(f"{at}: give one value key, got {given}.")
    if not given:
        raise FilterError(f"{at}: missing a value - 'value' or a typed key such as valueText / valueInt.")
    value = node[given[0]]
    _check_value(op, value, given[0], at)
    return {"path": path, "operator": op, "value": value}


def _norm_mongo(node: dict[str, Any], at: str) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = []
    for key, value in node.items():
        if key in ("$and", "$or"):
            if not isinstance(value, list) or not value:
                raise FilterError(f"{at}.{key} must be a non-empty list.")
            children = [_norm(c, f"{at}.{key}[{i}]") for i, c in enumerate(value)]
            clauses.append(children[0] if len(children) == 1 else
                           {"operator": "And" if key == "$and" else "Or", "operands": children})
        elif key == "$not":
            clauses.append({"operator": "Not", "operands": [_norm(value, f"{at}.$not")]})
        elif key.startswith("$"):
            raise FilterError(
                f"{at}: {key!r} is not a top-level operator. Use $and / $or / $not, or a property name."
            )
        elif isinstance(value, dict) and value and all(k.startswith("$") for k in value):
            for mop, operand in value.items():
                if mop not in MONGO:
                    raise FilterError(
                        f"{at}.{key}: unknown operator {mop!r}. Use one of: {', '.join(sorted(MONGO))}."
                    )
                op = MONGO[mop]
                if mop in ("$contains", "$not_contains"):
                    operand = operand if isinstance(operand, list) else [operand]
                if mop == "$is_null" and not isinstance(operand, bool):
                    raise FilterError(f"{at}.{key}.$is_null needs true or false.")
                _check_value(op, operand, mop, f"{at}.{key}")
                clauses.append({"path": [key], "operator": op, "value": operand})
        elif isinstance(value, list):
            raise FilterError(
                f"{at}.{key} compares against a list. Use {{'$in': [...]}} for any-of, or "
                "{'$all': [...]} for an array property holding every value."
            )
        elif value is None:
            clauses.append({"path": [key], "operator": "IsNull", "value": True})
        else:
            _check_value("Equal", value, "value", f"{at}.{key}")
            clauses.append({"path": [key], "operator": "Equal", "value": value})
    if len(clauses) == 1:
        return clauses[0]
    return {"operator": "And", "operands": clauses}


def _check_value(op: str, value: Any, key: str, at: str) -> None:
    if op == "IsNull":
        if not isinstance(value, bool):
            raise FilterError(f"{at}: IsNull needs true or false.")
    elif op == "WithinGeoRange":
        try:
            _geo(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise FilterError(
                f"{at}: WithinGeoRange needs {{'geoCoordinates': {{'latitude': .., 'longitude': ..}}, "
                "'distance': {'max': <metres>}} (or {'latitude', 'longitude', 'distance'})."
            ) from exc
    elif op in LIST_OPS:
        if not isinstance(value, list) or not value:
            raise FilterError(f"{at}: {op} needs a non-empty list.")
    elif op in NUMERIC_OPS:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise FilterError(f"{at}: {op} needs a number or an RFC 3339 date string, got {value!r}.")
    elif op == "Like":
        if not isinstance(value, str):
            raise FilterError(f"{at}: Like needs a string pattern ('*' and '?' wildcards).")
    elif isinstance(value, (dict, list)) or value is None:
        raise FilterError(f"{at}: {op} needs a single value, got {value!r}.")


def _geo(value: dict[str, Any]) -> tuple[float, float, float]:
    if "geoCoordinates" in value:
        coords, dist = value["geoCoordinates"], value["distance"]
        return float(coords["latitude"]), float(coords["longitude"]), float(
            dist["max"] if isinstance(dist, dict) else dist
        )
    return float(value["latitude"]), float(value["longitude"]), float(value["distance"])


# --------------------------------------------------------------------------
# compile to weaviate.classes.query.Filter
# --------------------------------------------------------------------------
def _datetime(value: Any) -> Any:
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value


def compile_filter(where: Any, properties: dict[str, str] | None = None) -> Any:
    """Normalise then build a v4 ``Filter``. ``properties`` maps name -> dataType."""
    canonical = normalize(where)
    if canonical is None:
        return None
    return _compile(canonical, properties, "filters")


def _compile(node: dict[str, Any], props: dict[str, str] | None, at: str) -> Any:
    from weaviate.classes.query import Filter, GeoCoordinate

    op = node["operator"]
    if op in LOGICAL_OPERATORS:
        children = [_compile(c, props, f"{at}.operands[{i}]") for i, c in enumerate(node["operands"])]
        if op == "Not":
            return Filter.not_(children[0])
        return Filter.all_of(children) if op == "And" else Filter.any_of(children)

    path, value = node["path"], node["value"]
    target = _target(path, props, op, value, at)
    method = getattr(target, LEAF_OPERATORS[op], None)
    if method is None:
        raise FilterError(f"{at}: {op} is not available on {'/'.join(path)}.")
    if op == "WithinGeoRange":
        lat, lon, dist = _geo(value)
        return method(coordinate=GeoCoordinate(latitude=lat, longitude=lon), distance=dist)
    if isinstance(value, list):
        value = [_datetime(v) for v in value]
    else:
        value = _datetime(value)
    return method(value)


def _target(path: list[str], props: dict[str, str] | None, op: str, value: Any, at: str) -> Any:
    from weaviate.classes.query import Filter

    if len(path) > 1:  # through cross-references: ref, Target, ref, Target, ..., prop
        f: Any = Filter.by_ref(link_on=path[0])
        for i in range(2, len(path) - 1, 2):
            f = f.by_ref(link_on=path[i])
        return f.by_property(path[-1])

    name = path[0]
    if name in ID_PATHS:
        if op not in {"Equal", "NotEqual", "ContainsAny", "ContainsNone"}:
            raise FilterError(f"{at}: the object id supports Equal, NotEqual, ContainsAny, ContainsNone.")
        return Filter.by_id()
    if name in CREATED_PATHS:
        return Filter.by_creation_time()
    if name in UPDATED_PATHS:
        return Filter.by_update_time()
    if name.startswith("len(") and name.endswith(")"):
        inner = name[4:-1]
        _check_property(inner, props, at)
        return Filter.by_property(inner, length=True)
    if name.startswith("count(") and name.endswith(")"):
        return Filter.by_ref_count(link_on=name[6:-1])

    dtype = _check_property(name, props, at)
    if dtype:
        _check_type(name, dtype, op, value, at)
    return Filter.by_property(name)


def _check_property(name: str, props: dict[str, str] | None, at: str) -> str | None:
    if props is None:
        return None
    if name not in props:
        raise FilterError(
            f"{at}: the collection has no property {name!r}. Properties: {', '.join(sorted(props)) or 'none'}."
        )
    return props[name]


def _check_type(name: str, dtype: str, op: str, value: Any, at: str) -> None:
    base = dtype.rstrip("[]")
    is_array = dtype.endswith("[]")
    if op == "IsNull":
        return
    if op == "WithinGeoRange" and base != "geoCoordinates":
        raise FilterError(f"{at}: WithinGeoRange needs a geoCoordinates property; {name!r} is {dtype}.")
    if base == "geoCoordinates" and op != "WithinGeoRange":
        raise FilterError(f"{at}: {name!r} is geoCoordinates - filter it with WithinGeoRange.")
    if op == "Like" and base not in {"text", "string"}:
        raise FilterError(f"{at}: Like works on text properties; {name!r} is {dtype}.")
    if op in NUMERIC_OPS and base in {"text", "string", "boolean", "uuid"}:
        raise FilterError(f"{at}: {op} compares numbers and dates; {name!r} is {dtype}.")
    if op in LIST_OPS and not is_array and base not in {"text", "string"}:
        # ContainsAny on a scalar works as "equals any of" - allowed, Weaviate accepts it.
        return
    samples = value if isinstance(value, list) else [value]
    for v in samples:
        if base in {"int", "number"} and op not in {"Like"} and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise FilterError(f"{at}: {name!r} is {dtype}; {v!r} is not a number.")
        if base == "boolean" and not isinstance(v, bool):
            raise FilterError(f"{at}: {name!r} is {dtype}; use true/false, not {v!r}.")
        if base in {"text", "string", "uuid"} and not isinstance(v, str):
            raise FilterError(f"{at}: {name!r} is {dtype}; {v!r} is not a string.")
        if base == "date" and isinstance(_datetime(v), str):
            raise FilterError(f"{at}: {name!r} is a date; give an RFC 3339 string like 2026-01-31T00:00:00Z.")
