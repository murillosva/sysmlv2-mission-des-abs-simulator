"""Metrics subsystem: records, aggregator, replanning signal.

Three layers of output land in the run directory:

1. ``engagements.parquet`` — one row per :class:`EngagementOutcome`.
2. ``metrics.parquet`` — long-format aggregate MOS/MOE/MOP, one row per
   (config, AoR, metric).
3. ``events_<config>.jsonl.gz`` — chronological log of state changes and
   notable events.

Plus, when MOS-1 < 0.80 for the modernized configuration:

4. ``replanning_signal.json`` — distribution of leaks across AoRs/sectors
   for the OPT loop.
"""

from cuas_sim.metrics.aggregator import MetricsAggregator
from cuas_sim.metrics.records import EngagementRecord, RunLogger, StateChangeEvent
from cuas_sim.metrics.replanning import ReplanningSignal, build_replanning_signal

__all__ = [
    "EngagementRecord",
    "MetricsAggregator",
    "ReplanningSignal",
    "RunLogger",
    "StateChangeEvent",
    "build_replanning_signal",
]
