"""Budgeted exploration over real environments: the cartographer core.

The world model gives you ONE latent per observation; these modules add the
accumulating pieces that turn it into a map and a policy: a landmark memory
(``awpredict.memory``), a cold-start encoder (``histogram_encoder``), and the
filesystem cartographer itself (``fs_cartographer``) with the curiosity /
predictive / prospector / planner policies and the blind-search baselines they
are gated against.
"""

from awpredict.explorer.fs_cartographer import FsCartographer, observe_dir, summarize
from awpredict.explorer.histogram_encoder import HistogramEncoder

__all__ = ["FsCartographer", "HistogramEncoder", "observe_dir", "summarize"]
