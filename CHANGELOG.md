# Changelog

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
