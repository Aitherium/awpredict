# Homeostatic surprise-modulated online learning — concept provenance

`online.py` (WM_ONLINE_LR_ENABLED, default off) is a CLEAN-ROOM
implementation. Two concepts are inspired by the **temporal-neuron** research
project (wizzense/temporal-neuron, TNRL v1.0 — non-commercial, share-alike):

1. **Surprise-gated plasticity** — a learner should change its weights most
   when its prediction was most violated, and consolidate (learn slowly)
   when the world is behaving as expected. temporal-neuron expresses this as
   surprise-modulated STDP; here it is an LR multiplier driven by the ratio
   of the current latent-MSE to its own slow EMA.
2. **Homeostatic set-point regulation** — a slow negative-feedback loop that
   holds an activity statistic inside a band rather than optimizing it.
   temporal-neuron applies this to firing rates; here the statistic is mean
   per-dim latent variance and the effector is LeWM's SIGReg weight
   (`sig_lambda`): variance under the band means the representation is
   contracting toward collapse → raise the isotropy pressure; over the band
   → relax it.

**License boundary:** no code from temporal-neuron was read into, copied
into, linked into, or derived into this file — an automated boundary scan
asserts that mechanically on every change upstream. This document exists so
the concept credit is explicit while the code stays clean-room.

Other inputs: LeWM's SIGReg (Epps–Pulley isotropy test) is the collapse
sensor/effector pair the homeostat drives; the surprise signal is the same
latent MSE the LeWM gate scores.
