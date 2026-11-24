"""Engagement subsystem.

Couples the A-29 (state machine + kinematics), the tracking policy
(MOE-2 sampling), and the weapon policy (firing + Bernoulli outcome). The
:class:`EngagementController` is the single source of truth for the
per-engagement metric record consumed by the aggregator.
"""

from cuas_sim.engagement.controller import EngagementController, EngagementOutcome

__all__ = ["EngagementController", "EngagementOutcome"]
