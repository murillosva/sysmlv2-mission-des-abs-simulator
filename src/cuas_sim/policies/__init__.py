"""Tracking and weapon policies (decoupled from agent code).

Each ``Policy`` is a pure function family that returns the *next decision*
(e.g. next tracking update interval, P_track value, firing distance, P_hit).
The policy depends on the configuration name (``baseline``|``modernized``),
the UAS archetype, the current range, and a dedicated decision RNG. The
agents in :mod:`cuas_sim.agents` call into these policies without knowing
the underlying formulas.
"""

from cuas_sim.policies.tracking import (
    BaselineTrackingPolicy,
    ModernizedTrackingPolicy,
    TrackingPolicy,
    make_tracking_policy,
)
from cuas_sim.policies.weapon import (
    BaselineWeaponPolicy,
    ModernizedWeaponPolicy,
    WeaponPolicy,
    make_weapon_policy,
)

__all__ = [
    "BaselineTrackingPolicy",
    "BaselineWeaponPolicy",
    "ModernizedTrackingPolicy",
    "ModernizedWeaponPolicy",
    "TrackingPolicy",
    "WeaponPolicy",
    "make_tracking_policy",
    "make_weapon_policy",
]
