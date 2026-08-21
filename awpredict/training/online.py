"""Surprise-modulated online LR + homeostatic SIGReg pressure (flag-gated).

CLEAN-ROOM implementation: the *concepts* — surprise-gated plasticity (learn
faster when the world violates prediction) and a homeostatic set-point (a
slow negative-feedback loop holding an activity statistic inside a band) —
come from a restricted-license research project kept deliberately sandboxed
upstream; concept provenance and the license boundary are documented in
PROVENANCE.md next to this file (the project is deliberately not named here
— the .md carries the credit, this file carries none of its identifiers).
The mechanism is written against LeWM's own quantities: train_step's latent
MSE as the surprise signal, per-dim latent variance as the homeostatic
statistic, and sig_lambda (SIGReg weight) as the effector.

OFF by default (WM_ONLINE_LR_ENABLED=0): the controller constructs inert and
after_step() returns None without touching the model — asserted by tests.

Usage:
    ctl = OnlineLRController(model)          # reads env; inert unless enabled
    for _ in range(steps):
        m = model.train_step()
        ctl.after_step(m)                    # adjusts lr (+ sig_lambda)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("awpredict.online")


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class OnlineLRController:
    """Modulates optimizer LR by relative surprise; holds latent variance in
    a band by adjusting SIGReg pressure.

    LR law:   lr = base_lr * clamp(1 + alpha * (s_rel - 1), lr_min_x, lr_max_x)
              where s_rel = mse / EMA(mse). A step that surprises more than
              the recent average learns faster; a routine step cools back
              toward base. EMA beta is high (slow) so a single outlier batch
              cannot whipsaw the LR.
    Homeostat: mean per-dim variance of the latent batch below var_lo means
              the representation is contracting toward collapse -> raise
              sig_lambda (more isotropy pressure); above var_hi -> lower it.
              Multiplicative 5%/step, clamped to [0.1x, 10x] of the model's
              original sig_lambda: a slow set-point loop, not a bang-bang.
    """

    def __init__(self, model: Any, enabled: Optional[bool] = None) -> None:
        self.model = model
        self.enabled = (
            enabled if enabled is not None
            else os.environ.get("WM_ONLINE_LR_ENABLED", "0").lower()
            in ("1", "true", "yes"))
        self.alpha = _envf("WM_ONLINE_LR_ALPHA", 0.5)
        self.lr_min_x = _envf("WM_ONLINE_LR_MIN_X", 0.25)
        self.lr_max_x = _envf("WM_ONLINE_LR_MAX_X", 4.0)
        self.ema_beta = _envf("WM_ONLINE_MSE_BETA", 0.98)
        self.var_lo = _envf("WM_ONLINE_VAR_LO", 0.5)
        self.var_hi = _envf("WM_ONLINE_VAR_HI", 2.0)
        self.sig_step = _envf("WM_ONLINE_SIG_STEP", 0.05)
        self._mse_ema: Optional[float] = None
        self._base_lr = float(getattr(model, "lr", 0.0) or 0.0)
        self._base_sig = float(getattr(model, "sig_lambda", 0.0) or 0.0)
        self._steps = 0
        if self.enabled and not self._base_lr:
            logger.warning("online-lr: model has no base lr; controller inert")
            self.enabled = False

    def after_step(self, metrics: Optional[Dict[str, Any]],
                   z_batch: Any = None) -> Optional[Dict[str, float]]:
        """Call once after each model.train_step(). Returns what was applied,
        or None when disabled / nothing to act on (never raises into the
        training loop)."""
        if not self.enabled or not metrics:
            return None
        mse = metrics.get("mse")
        if not isinstance(mse, (int, float)) or mse != mse:  # None or NaN
            return None

        # -- surprise-modulated LR ------------------------------------------
        if self._mse_ema is None:
            self._mse_ema = float(mse)
        s_rel = float(mse) / max(self._mse_ema, 1e-9)
        self._mse_ema = self.ema_beta * self._mse_ema + (1 - self.ema_beta) * float(mse)
        factor = 1.0 + self.alpha * (s_rel - 1.0)
        factor = min(max(factor, self.lr_min_x), self.lr_max_x)
        new_lr = self._base_lr * factor
        opt = getattr(self.model, "_opt", None)
        if opt is not None:
            for group in opt.param_groups:
                group["lr"] = new_lr

        # -- homeostatic SIGReg pressure ------------------------------------
        new_sig = None
        if z_batch is not None and self._base_sig > 0:
            try:
                var = float(z_batch.detach().float().var(dim=0).mean())
            except Exception:
                var = None
            if var is not None:
                sig = float(getattr(self.model, "sig_lambda", self._base_sig))
                if var < self.var_lo:
                    sig *= (1.0 + self.sig_step)   # contracting -> push back
                elif var > self.var_hi:
                    sig *= (1.0 - self.sig_step)   # over-dispersed -> relax
                sig = min(max(sig, 0.1 * self._base_sig), 10.0 * self._base_sig)
                self.model.sig_lambda = sig
                new_sig = sig

        self._steps += 1
        out = {"lr": new_lr, "lr_factor": factor, "surprise_rel": s_rel,
               "mse_ema": self._mse_ema}
        if new_sig is not None:
            out["sig_lambda"] = new_sig
        return out

    def restore(self) -> None:
        """Put base lr / sig_lambda back (e.g. before a gated eval, so the
        controller can never leak state into a fairness-sensitive run)."""
        opt = getattr(self.model, "_opt", None)
        if opt is not None and self._base_lr:
            for group in opt.param_groups:
                group["lr"] = self._base_lr
        if self._base_sig:
            self.model.sig_lambda = self._base_sig
