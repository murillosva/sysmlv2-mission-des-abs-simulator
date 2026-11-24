"""Apply an OptimizerDirective to produce the next iteration's MissionConfig.

First-iteration behaviour: the directive recommends which FARPs
to activate. This module translates that recommendation into a MissionConfig update.
Current implementation is conservative: config is returned unchanged if applying
the directive would remove all FARPs (safety guard) or the directive's FARPs are
not a subset of config.farps (directive references candidates not in config).

CAP positions are fixed in this
version of the simulator. cap_adjustments in the directive is always silently
ignored. CAP repositioning is future work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuas_sim.config import MissionConfig
    from cuas_sim.optimization.scp_solver import OptimizerDirective


def apply_directive(
    config: MissionConfig,
    directive: OptimizerDirective,
) -> MissionConfig:
    """Produce next-iteration config by applying an OptimizerDirective.

    The directive specifies which FARPs to activate. This filters config.farps
    to only the selected ones. If the result would be empty (safety guard),
    the original config is returned unchanged.

    Note: A29Schedule entries are NOT modified — FARPs excluded from activate_farps
    but still referenced by an A29Schedule remain in config.farps to avoid breaking
    MetricsAggregator._t_pea_for_aor(). Full FARP schedule switching is a known
    limitation; a future spec may address this once the A29Schedule is decoupled.
    """
    requested = set(directive.activate_farps)
    available = set(config.farps.keys())

    # FARPs referenced by A29Schedule that we must keep (to avoid KeyError in sim)
    required_by_schedule = {
        sched.farp_name for sched in config.a29_schedules.values()
    }

    # Safe set: requested ∩ available, but always keep schedule-required FARPs
    safe = (requested & available) | required_by_schedule

    if not safe or safe == available:
        return config

    new_farps = {k: v for k, v in config.farps.items() if k in safe}
    return config.replaced(farps=new_farps)
