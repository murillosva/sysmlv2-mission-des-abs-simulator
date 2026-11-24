"""A-29 lifecycle state enum.

Each name maps 1:1 to a snapshot in the SysML v2 ``MissionExecutionPackage``
(see ``sysml/Execution/MissionExecutionPackage.sysml``). Events emitted to
``events_<config>.jsonl.gz`` carry these names verbatim so each entry of the
event stream is traceable back to a SysML snapshot.
"""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class A29State(StrEnum):
    """Sequential states of an A-29 across the six mission phases (15 states).

    Pre-PEA (symbolic, no kinematics):

    * :attr:`AT_BASE` — on the ground at SBPV before takeoff.
    * :attr:`TRANSIT_TO_FARP` — Phase 1 (SBPV → FARP).
    * :attr:`AT_FARP_PRE_PEA` — refueling complete, awaiting departure to CAP.
    * :attr:`REPOSITIONING_TO_CAP` — Phase 2 (FARP → CAP).
    * :attr:`HOLDING_ORBIT` — Phase 3 (waiting for T+153 min).

    PEA (active simulation):

    * :attr:`AT_CAP_PEA` — on station, available for engagement.
    * :attr:`PURSUING` — flying intercept toward a UAS.
    * :attr:`ENGAGING` — inside the weapon envelope, executing the TDE.
    * :attr:`RETURNING_TO_CAP` — return to the CAP from where the engagement ended.

    Recovery (geometric):

    * :attr:`RTB_BINGO` — bingo-fuel abort: flying from the current position to
      the designated FARP (``DepartCAPForFARP``). **Legacy only** under
      ``uas_spawn_window_enabled=False``.
    * :attr:`DEPARTING_CAP` — Phase 5 (CAP → FARP) at end of a normal PEA.
      **Legacy only** under ``uas_spawn_window_enabled=False``.
    * :attr:`RTB_FUEL_LIMIT` — unified fuel-limit-driven terminal state
      when ``uas_spawn_window_enabled=True``; replaces both ``RTB_BINGO`` and
      ``DEPARTING_CAP`` in that path (maps 1:1 to SysML ``DepartCAPForFARP``).
    * :attr:`AT_FARP_POST_PEA` — refuel for the return leg (``RefuelAtFARP``).
    * :attr:`RECOVERING_TO_BASE` — Phase 6 (FARP → SBPV, ``RecoverToMainBase``).
    * :attr:`MISSION_COMPLETE` — terminal.
    """

    AT_BASE = "AT_BASE"
    TRANSIT_TO_FARP = "TRANSIT_TO_FARP"
    AT_FARP_PRE_PEA = "AT_FARP_PRE_PEA"
    REPOSITIONING_TO_CAP = "REPOSITIONING_TO_CAP"
    HOLDING_ORBIT = "HOLDING_ORBIT"
    AT_CAP_PEA = "AT_CAP_PEA"
    PURSUING = "PURSUING"
    ENGAGING = "ENGAGING"
    RETURNING_TO_CAP = "RETURNING_TO_CAP"
    RTB_BINGO = "RTB_BINGO"
    DEPARTING_CAP = "DEPARTING_CAP"
    RTB_FUEL_LIMIT = "RTB_FUEL_LIMIT"
    AT_FARP_POST_PEA = "AT_FARP_POST_PEA"
    RECOVERING_TO_BASE = "RECOVERING_TO_BASE"
    MISSION_COMPLETE = "MISSION_COMPLETE"


#: Pre-PEA phase sequence used by the symbolic state machine in
#: :meth:`cuas_sim.agents.a29.A29._pre_pea_phase`.
PRE_PEA_SEQUENCE: tuple[A29State, ...] = (
    A29State.TRANSIT_TO_FARP,
    A29State.AT_FARP_PRE_PEA,
    A29State.REPOSITIONING_TO_CAP,
    A29State.HOLDING_ORBIT,
)

#: Post-PEA phase sequence.
POST_PEA_SEQUENCE: tuple[A29State, ...] = (
    A29State.DEPARTING_CAP,
    A29State.AT_FARP_POST_PEA,
    A29State.RECOVERING_TO_BASE,
)
