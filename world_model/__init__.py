"""world_model — a small, dependency-light world-model package (JEPA + MLP engines).

One package, two engines, N environment adapters:

  * ``world_model.core.lewm.LeWorldModel`` — the LeWM-style JEPA
    (two-term loss: next-latent MSE + SIGReg; CEM planner), used as the
    world model behind an ARC-AGI-3 solving agent. Includes an optional
    value head (``_ValueHead``/``_FsAdapter``/``value()``/``train_value_step``)
    for value-guided CEM planning — a frozen-latent predictor head trained
    on returns, off by default, on by constructing with a value config. This
    is the same engine the solving agent runs, not a cut-down demo of it —
    the goal of this package is to give you everything needed to bootstrap
    your own world model, not a subset of it.
  * ``world_model.core.mlp.MLPWorldModel`` — an embedding-MLP transition
    model (tabular → hybrid → neural).

Contracts live in ``world_model.contracts`` (WorldModel, EnvironmentAdapter).
Torch and numpy are OPTIONAL at import time — engines degrade loudly
(ok == False), never raise into a caller.
"""

from world_model.contracts import EnvironmentAdapter, WorldModel, conforms

__version__ = "0.1.0"

__all__ = ["EnvironmentAdapter", "WorldModel", "conforms", "__version__"]
