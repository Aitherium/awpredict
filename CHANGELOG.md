# Changelog

## Unreleased

- `awpredict.adapters.routing_env.RoutingEnv`: mixture-of-experts routing as an
  EnvironmentAdapter - observe the last ubatch's per-layer experts, act with a
  candidate warm set, reward = the next ubatch's recall. Reads AXRL routing logs
  (`read_axrl`, with rotation), dependency-free.

## 0.1.4 — 2026-10-20

- `awpredict.viz`: a dream visualizer for any WorldModel engine — open-loop
  imagination and one-step predictions scored against the copy-last baseline,
  a 2D PCA of the latent trajectory, and a decoded filmstrip (LeWorldModel) or
  latent heatmap, rendered to one self-contained HTML page.
- `awpredict viz --engine lewm|mlp|concept --demo`: trains on a synthetic world
  and writes the page. Exit 0 written, 1 no rollout, 2 could not run.
- `MLPWorldModel.predict_latent(z, action)`: the vector-space step `predict`
  (which answers in state hashes) could not provide.

## 0.1.1 — 2026-10-04

- The Problem contract: `ProblemSpec`, a `Scorer` that can refuse, and one record
  shape per role (`Attempt` among them) — what awgym 0.1.1 imports.
- A command-line interface.
- The doctor's own measured NO now reaches its exit code.

## 0.1.0 — 2026-08-18

Initial public release.

- Extracted `world_model.core.lewm.LeWorldModel` (LeWM-style JEPA, value head
  included), `world_model.core.mlp.MLPWorldModel`, `world_model.contracts`,
- Extracted `awpredict.core.lewm.LeWorldModel` (LeWM-style JEPA, value head
  included), `awpredict.core.mlp.MLPWorldModel`, `awpredict.contracts`,
  the `code_world` example adapter, and the optional homeostatic online-LR
  controller, boundary-scrubbed of internal identifiers.
- Added `tests/test_value_head.py` — the value head shipped with zero test
  coverage; this asserts it's structurally inert when off and actually fits
  a target when on, not just plumbing-correct.
- `RESULTS.md`: two negative findings from running this against ARC-AGI-3
  (a two-line lookup beats the trained model on 98.9% of rows; an
  aggregate-accuracy ranking picked the worst individual predictor as
  champion), plus the current ARC-AGI-3 solve rate.
