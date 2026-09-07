"""Command line over awpredict: world-model engines and the contracts they satisfy.

  engines   which engines can actually construct here, and why not if they cannot
  conforms  does a class satisfy WorldModel or EnvironmentAdapter, and what is missing

--json for machine-readable output.
"""

from __future__ import annotations

import argparse
import importlib
import json
from typing import List

from . import contracts

_PROTOCOLS = {"WorldModel": contracts.WorldModel,
              "EnvironmentAdapter": contracts.EnvironmentAdapter}


def cmd_engines(args) -> int:
    """Report per engine, and say WHICH kind of unusable.

    Import failure, a raising gate, and a feature switch that is simply off need
    three different fixes. lewm's enabled() is the last of those -- an
    ARC_WORLD_ENGINE env gate, default off -- so calling it 'dependencies
    missing' sends someone to install what they already have.
    """
    rows = []
    for name, mod in (("lewm", "awpredict.core.lewm"), ("mlp", "awpredict.core.mlp")):
        try:
            m = importlib.import_module(mod)
        except Exception as exc:
            rows.append({"engine": name, "usable": False,
                         "reason": f"import failed: {type(exc).__name__}: {exc}"})
            continue
        enabled = getattr(m, "enabled", None)
        if enabled is None:
            rows.append({"engine": name, "usable": True, "reason": "imports"})
            continue
        try:
            ok = bool(enabled())
        except Exception as exc:
            rows.append({"engine": name, "usable": False,
                         "reason": f"enabled() raised: {type(exc).__name__}"})
            continue
        gate = getattr(m, "_ENGINE_ENV", "ARC_WORLD_ENGINE")
        rows.append({"engine": name, "usable": ok,
                     "reason": "ready" if ok else f"gated off: set {gate}=1"})

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"  {'ok  ' if r['usable'] else 'FAIL'} {r['engine']:<6} {r['reason']}")
    return 0 if any(r["usable"] for r in rows) else 1


def cmd_conforms(args) -> int:
    proto = _PROTOCOLS[args.protocol]
    try:
        mod = importlib.import_module(args.module)
    except Exception as exc:
        print(f"cannot import {args.module}: {type(exc).__name__}: {exc}")
        return 2
    obj = getattr(mod, args.name, None)
    if obj is None:
        print(f"{args.module} has no {args.name!r}")
        return 2

    missing: List[str] = contracts.conforms(obj, proto)
    if args.json:
        print(json.dumps({"module": args.module, "name": args.name,
                          "protocol": args.protocol, "missing": missing,
                          "conforms": not missing}, indent=2))
    elif missing:
        print(f"{args.name} does NOT satisfy {args.protocol}; missing: {', '.join(missing)}")
    else:
        print(f"{args.name} satisfies {args.protocol}")
    return 0 if not missing else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="awpredict",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("engines", help="which world-model engines are usable here")

    p_c = sub.add_parser("conforms", help="does a class satisfy a protocol")
    p_c.add_argument("--module", required=True, help="importable module path")
    p_c.add_argument("--name", required=True, help="class or object in that module")
    p_c.add_argument("--protocol", choices=sorted(_PROTOCOLS), default="WorldModel")
    return ap


def main(argv: "list[str] | None" = None) -> int:
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 2
    return {"engines": cmd_engines, "conforms": cmd_conforms}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
