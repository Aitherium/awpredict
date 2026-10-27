"""The wire schema for the generative world-stream contract.

The record types live in :mod:`awpredict.contracts`; this module turns them
into JSON Schema so the Node engine can validate what it receives without
importing Python, and so the two sides cannot drift in silence. It is
dependency-free on purpose — the package ships with no runtime deps, and a
schema generator that needs pydantic would undo that.

Two consumers:

  * ``python -m awpredict.wire`` prints the schema document (used to write
    ``docs/world-stream.schema.json``).
  * ``problems(kind, payload)`` answers "what is wrong with this message",
    naming the field — never a bare False.

Rules:
  * A required field is one with no default. Nothing is inferred.
  * An unknown key is a problem, not a forward-compatible extra: a typo that
    silently passes is a message that silently means something else.
  * ``Dict[str, Any]`` is an open object and is not descended into. ``meta``
    is where a backend may say anything it likes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import MISSING, fields
from typing import (
    Any,
    Dict,
    List,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)

from awpredict.contracts import Action, Frame, StateDigest, Steer

#: Message kind -> the record that defines it. The key is what goes on the wire.
KINDS = {
    "frame": Frame,
    "action": Action,
    "steer": Steer,
    "state": StateDigest,
}

_SCALARS = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(tp: Any) -> Dict[str, Any]:
    """The JSON Schema fragment for one annotation. Unknown shapes stay open."""
    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any or tp is None:
        return {}
    if tp in _SCALARS:
        return {"type": _SCALARS[tp]}

    if origin is not None and origin.__name__ in ("Dict", "dict", "Mapping"):
        return {"type": "object"}
    if origin is not None and origin.__name__ in ("Tuple", "tuple"):
        if args and args[-1] is not Ellipsis:
            same = len(set(args)) == 1
            return {
                "type": "array",
                "minItems": len(args),
                "maxItems": len(args),
                "items": _json_type(args[0]) if same else {},
            }
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin is not None and origin.__name__ in ("List", "list", "Sequence"):
        return {"type": "array", "items": _json_type(args[0]) if args else {}}

    # Optional[X] / Union[X, None]
    union = get_origin(tp) is not None and str(origin).endswith("Union")
    if union or type(tp).__name__ == "UnionType":
        real = [a for a in args if a is not type(None)]
        if len(real) == 1:
            frag = dict(_json_type(real[0]))
            frag["nullable"] = True
            return frag
        return {}

    return {}


def _required(cls: type) -> List[str]:
    return [f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING]  # type: ignore[misc]


def schema_for(kind: str) -> Dict[str, Any]:
    """The JSON Schema object for one message kind."""
    cls = KINDS[kind]
    hints = get_type_hints(cls)
    props = {f.name: _json_type(hints.get(f.name, Any)) for f in fields(cls)}
    required = _required(cls)
    out: Dict[str, Any] = {
        "type": "object",
        "title": cls.__name__,
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        out["required"] = required
    return out


def document() -> Dict[str, Any]:
    """The whole wire contract, one ``$defs`` entry per message kind."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aitherium.com/schemas/world-stream.json",
        "title": "Aither world stream",
        "description": (
            "Control-plane records for a generative world model. Frame payloads "
            "travel on a side channel, never inside these objects."
        ),
        "$defs": {kind: schema_for(kind) for kind in KINDS},
    }


def problems(kind: str, payload: Any) -> List[str]:
    """What is wrong with ``payload`` as a ``kind`` message (empty == fine)."""
    if kind not in KINDS:
        return [f"unknown message kind {kind!r}; expected one of {sorted(KINDS)}"]
    if not isinstance(payload, dict):
        return [f"{kind} must be a JSON object, got {type(payload).__name__}"]

    cls = KINDS[kind]
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    out: List[str] = []

    for name in _required(cls):
        if name not in payload:
            out.append(f"{kind}.{name} is required")
    for name in payload:
        if name not in names:
            out.append(f"{kind}.{name} is not a field of {cls.__name__}")

    for name, value in payload.items():
        if name not in names or value is None:
            continue
        expected = hints.get(name, Any)
        for scalar, label in ((str, "a string"), (int, "an integer"),
                              (float, "a number"), (bool, "a boolean")):
            if expected is scalar:
                if not isinstance(value, scalar) or (scalar is int and isinstance(value, bool)):
                    out.append(f"{kind}.{name} must be {label}")
                break
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """Print the schema document. ``--kind frame`` narrows to one definition."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--kind":
        if len(argv) < 2 or argv[1] not in KINDS:
            print(f"unknown kind; expected one of {sorted(KINDS)}", file=sys.stderr)
            return 2
        print(json.dumps(schema_for(argv[1]), indent=2, sort_keys=True))
        return 0
    print(json.dumps(document(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
