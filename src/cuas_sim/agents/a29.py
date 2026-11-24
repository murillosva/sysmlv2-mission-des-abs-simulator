"""A-29 agent — full 14-state lifecycle.

Each A-29 is one SimPy process per configuration (so a paired run has 12
processes: 6 baseline + 6 modernized). The process walks the state machine
of :class:`cuas_sim.states.A29State`:

#. Pre-PEA (symbolic timeouts): TRANSIT_TO_FARP → AT_FARP_PRE_PEA →
   REPOSITIONING_TO_CAP → HOLDING_ORBIT.
#. PEA (active): AT_CAP_PEA → (PURSUING → ENGAGING → RETURNING_TO_CAP)\\* …
#. Recovery (geometric, ``t_to_farp``):
   * normal end of PEA: DEPARTING_CAP → AT_FARP_POST_PEA →
     RECOVERING_TO_BASE → MISSION_COMPLETE;
   * bingo-fuel abort: RTB_BINGO → AT_FARP_POST_PEA → RECOVERING_TO_BASE →
     MISSION_COMPLETE.

The engagement substates are driven by :class:`EngagementController`, which also
truncates the engagement on bingo fuel. Phase 4 (PEA) terminates
at ``min(t_pea_start_s + phase4_pea_s, t_bingo_cap)``; any UAS still in the inbox
after that is left untouched (recorded as a "leak" by the metrics aggregator).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from cuas_sim.engagement.controller import EngagementController
from cuas_sim.geometry.projection import XY, LocalProjection, t_to_farp
from cuas_sim.states import A29State

if TYPE_CHECKING:
    import simpy

    from cuas_sim.agents.e99m_dispatcher import PendingPool
    from cuas_sim.config import A29Schedule, MissionConfig
    from cuas_sim.metrics.records import RunLogger
    from cuas_sim.policies.tracking import TrackingPolicy
    from cuas_sim.policies.weapon import WeaponPolicy
    from cuas_sim.rng import _PerConfigRNG
    from cuas_sim.world.aor import AoRMap
    from cuas_sim.world.arrivals import UASBirth

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class A29:
    """One A-29 in one configuration."""

    a29_id: str
    schedule: A29Schedule
    aor_name: str
    cfg: MissionConfig
    aor_map: AoRMap
    projection: LocalProjection
    tracking: TrackingPolicy
    weapon: WeaponPolicy
    rng: _PerConfigRNG
    run_logger: RunLogger
    config_name: str
    state: A29State = A29State.AT_BASE
    engagements: int = field(default=0)
    # Rockets remaining this sortie (set at Phase 4 start;
    # depletes across all engagements, no rearming in PEA).
    rockets_left: int = field(default=0)
    # Shared pending pool (injected by mission.py; ``None``
    # leaves behaviour unchanged for callers that do not wire it).
    pending_pool: PendingPool | None = field(default=None)
    # This config's fleet (injected by mission.py) so a
    # bingo-aborting A-29 can tell whether a teammate is free to take over the
    # released UAS; ``None`` ⇒ no teammate ⇒ the UAS leaks ``bingo-fuel``.
    fleet: dict[str, A29] | None = field(default=None)
    # ``True`` once Phase 4 has elapsed (PEA closed).
    _pea_closed: bool = False
    # Mission time the tank was last filled (FARP departure).
    _t_farp_departure: float = field(default=0.0)
    # Set when an engagement (or the CAP idle) hits bingo.
    _bingo_triggered: bool = field(default=False)
    _bingo_xy: XY | None = field(default=None)
    # Shared first-bingo event for this configuration
    # (injected by ``MissionSimulator.run_paired``). When the A-29 enters
    # ``RTB_FUEL_LIMIT`` it ``.succeed(value={"a29_id", "t_s"})`` the event
    # idempotently (``if not first_bingo_event.triggered``). ``None`` keeps the
    # legacy callers (unit tests built without a paired sim) intact.
    first_bingo_event: simpy.Event | None = field(default=None)

    @property
    def is_free(self) -> bool:
        """True when the A-29 can accept a new UAS assignment."""
        return self.state not in (A29State.PURSUING, A29State.ENGAGING)

    def run(self, env: simpy.Environment, inbox: simpy.Store) -> Generator[Any, Any, None]:
        """Main SimPy process."""
        # ─── Pre-PEA (symbolic timeouts) ─────────────────────────────────
        if not self.cfg.skip_pre_pea:
            for state, duration in self._pre_pea_sequence():
                self._set_state(env, state)
                # Full tank on FARP departure for the CAP.
                if state == A29State.REPOSITIONING_TO_CAP:
                    self._t_farp_departure = float(env.now)
                if duration > 0:
                    yield env.timeout(duration)

        # ─── Wait until the synchronized PEA start ───────────────────────
        wait = self.cfg.t_pea_start_s - env.now
        if wait > 0:
            yield env.timeout(wait)
        self._set_state(env, A29State.AT_CAP_PEA)
        # Load the launcher for this sortie (PEA start).
        # Per-config capacity (modernized ← APKWS, baseline ←
        # SBAT-70); falls back to the shared default when unset.
        self.rockets_left = self.cfg.rocket_budget_for(self.config_name)
        # When pre-PEA is skipped the A-29 starts on station with
        # a full tank; anchor the fuel clock at the PEA start.
        if self.cfg.skip_pre_pea:
            self._t_farp_departure = float(env.now)

        cap_xy = self._cap_xy()
        farp_xy = self._farp_xy()
        v = self.cfg.a29_cruise_speed_mps
        # Fuel-driven bingo while idling at the CAP.
        t_bingo_cap = self._fuel_exhaust_t - t_to_farp(cap_xy, farp_xy, v)

        # ─── PEA loop ────────────────────────────────────────────────────
        bingo_at_cap = False
        # Under ``uas_spawn_window_enabled=True`` the PEA has
        # no static end (duration is emergent from ``bingo_evt``); only bingo
        # or an engagement-bingo break out of the loop.
        pea_end_s = (
            float("inf") if self.cfg.uas_spawn_window_enabled
            else env.now + self.schedule.phase4_pea_s
        )
        while env.now < pea_end_s:
            # Before blocking on the inbox, take over any
            # claimable pending UAS (and prune the ones that already evaded).
            if self.pending_pool is not None:
                yield from self._drain_pending_pool(env, pea_end_s)
                if self._bingo_triggered or env.now >= pea_end_s:
                    break
            # Wait for a UAS notification, the PEA end (legacy only), or the
            # CAP bingo, whichever comes first.
            # NOTE: the legacy branch preserves the *exact* original creation
            # order ``timeout_evt → bingo_evt`` to keep
            # SimPy's per-event id monotone and avoid CRN drift.
            get_evt = inbox.get()
            if self.cfg.uas_spawn_window_enabled:
                bingo_evt = env.timeout(max(0.0, t_bingo_cap - env.now))
                yield env.any_of([get_evt, bingo_evt])
            else:
                timeout_evt = env.timeout(max(0.0, pea_end_s - env.now))
                bingo_evt = env.timeout(max(0.0, t_bingo_cap - env.now))
                yield env.any_of([get_evt, timeout_evt, bingo_evt])

            if get_evt.triggered:
                birth = cast('UASBirth', get_evt.value)
                # Skip any UAS that arrived before PEA started, or after end
                if birth.spawn_time_s < self.cfg.t_pea_start_s:
                    continue
                if env.now >= pea_end_s:
                    # PEA closed while waiting — store back? No: easier to
                    # just drop (will be counted as a leak by aggregator).
                    self.run_logger.log_uas_leak(self.config_name, birth, reason="pea-closed")
                    break
                yield env.process(self._engage(env, birth))
                if self._bingo_triggered:
                    break
            elif bingo_evt.triggered and env.now < pea_end_s:
                # Out of fuel while idling on station (no UAS in hand → no leak).
                bingo_at_cap = True
                break
            else:
                # PEA ended without a new notification (legacy timeout path).
                break

        self._pea_closed = True

        # ─── Recovery (geometric) ─────────────────────────────
        if self._bingo_triggered or bingo_at_cap:
            origin = self._bingo_xy if self._bingo_xy is not None else cap_xy
            yield from self._recover(env, origin=origin, bingo=True)
        else:
            yield from self._recover(env, origin=cap_xy, bingo=False)

        self._set_state(env, A29State.MISSION_COMPLETE)

    # ─── Engagement substate ─────────────────────────────────────────────

    def _engage(self, env: simpy.Environment, birth: UASBirth) -> Generator[Any, Any, None]:
        """Handle a single engagement (PURSUING → ENGAGING → AT_CAP_PEA)."""
        from cuas_sim.agents.uas import UAS

        # Winchester: no rockets left ⇒ this UAS leaks.
        if self.rockets_left <= 0:
            self.run_logger.log_uas_leak(self.config_name, birth, reason="winchester")
            return

        uas = UAS.from_birth(birth, self.projection)
        a29_xy = self._cap_xy()
        t_notify = env.now

        self._set_state(env, A29State.PURSUING)
        controller = EngagementController(
            cfg=self.cfg,
            tracking_policy=self.tracking,
            weapon_policy=self.weapon,
            projection=self.projection,
            config_name=self.config_name,
        )
        outcome = controller.run(
            a29_id=self.a29_id,
            aor_name=self.aor_name,
            uas=uas,
            a29_xy_at_notify=a29_xy,
            t_notify_s=t_notify,
            rng_tracking=self.rng.tracking_updates,
            rng_firing=self.rng.firing_phil,
            rng_p_hit=self.rng.p_hit_outcome,
            rng_reengage=self.rng.reengagement_interval,
            rockets_available=self.rockets_left,
            fuel_exhaust_t=self._fuel_exhaust_t,
            farp_xy=self._farp_xy(),
        )
        # Every weapon release (incl. P_hit=0 long shots)
        # consumes one rocket; the budget depletes across the sortie.
        self.rockets_left = max(0, self.rockets_left - outcome.n_shots)

        # Bingo abort: advance to the abort instant,
        # release the UAS (pool if a teammate is free, else ``bingo-fuel`` leak),
        # flag the RTB and return (the PEA loop then routes to RTB_BINGO).
        if outcome.bingo_aborted:
            bingo_dt = max(0.0, outcome.bingo_t_s - t_notify)
            if bingo_dt > 0:
                yield env.timeout(bingo_dt)
            if outcome.is_neutralized:
                uas.mark_killed(outcome.t_kill_s)
            else:
                uas.mark_leak(outcome.bingo_t_s)
                if self.pending_pool is not None and self._teammate_free():
                    self.pending_pool.add(birth, t_added_s=float(env.now))
                else:
                    self.run_logger.log_uas_leak(
                        self.config_name, birth, reason="bingo-fuel"
                    )
            self.engagements += 1
            self.run_logger.log_engagement(outcome)
            self._bingo_triggered = True
            self._bingo_xy = outcome.bingo_xy
            return

        # Advance SimPy clock to the actual kill / no-kill time so concurrent
        # arrivals during the engagement back up in the inbox (modelling the
        # case study's serial-engagement assumption: a single A-29 cannot
        # engage two UAS simultaneously).
        if outcome.feasible and not math.isnan(outcome.t_envelope_entry_s):
            # PURSUING: clock advances from notification to weapon-envelope entry.
            pursuing_dt = max(0.0, outcome.t_envelope_entry_s - t_notify)
            if pursuing_dt > 0:
                yield env.timeout(pursuing_dt)
            # ENGAGING: clock advances from envelope entry to weapon release.
            self._set_state(env, A29State.ENGAGING)
            firing_dt = max(0.0, outcome.t_fire_s - float(env.now)) if not math.isnan(outcome.t_fire_s) else 0.0
            if firing_dt > 0:
                yield env.timeout(firing_dt)

        # Apply UAS outcome
        if outcome.is_neutralized:
            uas.mark_killed(outcome.t_kill_s)
        else:
            uas.mark_leak(outcome.t_fire_s if outcome.feasible else t_notify)

        self.engagements += 1
        self.run_logger.log_engagement(outcome)

        # Return from where the engagement actually
        # ended (kill point, or the A-29's position at the last shot/floor),
        # not the symmetric outbound duration.
        if outcome.end_xy is not None:
            return_time = math.hypot(
                outcome.end_xy.x - a29_xy.x, outcome.end_xy.y - a29_xy.y
            ) / self.cfg.a29_cruise_speed_mps
        else:
            return_time = 0.0
        if return_time > 0:
            self._set_state(env, A29State.RETURNING_TO_CAP)
            yield env.timeout(return_time)
        self._set_state(env, A29State.AT_CAP_PEA)

    # ─── Pending-pool consultation ──────────────────────

    def _cap_xy(self) -> XY:
        """Projected position of this A-29's CAP."""
        return self.projection.project(self.cfg.caps[f"CAP-{self.schedule.cap_index}"])

    def _farp_xy(self) -> XY:
        """Projected position of this A-29's designated FARP."""
        return self.projection.project(self.cfg.farps[self.schedule.farp_name])

    @property
    def _fuel_exhaust_t(self) -> float:
        """Absolute mission time the tank empties."""
        return self._t_farp_departure + self.cfg.a29_endurance_s

    def _teammate_free(self) -> bool:
        """True if another A-29 of this config is free to take a released UAS."""
        if self.fleet is None:
            return False
        return any(
            agent.is_free for aid, agent in self.fleet.items() if aid != self.a29_id
        )

    def _drain_pending_pool(
        self, env: simpy.Environment, pea_end_s: float,
    ) -> Generator[Any, Any, None]:
        """Claim and engage pending UAS while any remain claimable.

        Each :meth:`PendingPool.claim` call also prunes pendings that crossed
        their 80 nmi ZIDA depth (logged ``reason="pending-evaded"``).
        """
        pool = self.pending_pool
        if pool is None:
            return
        while env.now < pea_end_s:
            birth = pool.claim(
                cap_xy=self._cap_xy(),
                now_s=float(env.now),
                a29_speed_mps=self.cfg.a29_cruise_speed_mps,
                envelope_max_m=self.weapon.envelope_max_m,
            )
            if birth is None:
                return
            yield env.process(self._engage(env, birth))
            if self._bingo_triggered:
                return

    # ─── State-machine helpers ───────────────────────────────────────────

    def _pre_pea_sequence(self) -> list[tuple[A29State, float]]:
        s = self.schedule
        return [
            (A29State.TRANSIT_TO_FARP, s.phase1_sbpv_to_farp_s),
            (A29State.AT_FARP_PRE_PEA, 0.0),  # refueling already in phase1
            (A29State.REPOSITIONING_TO_CAP, s.phase2_farp_to_cap_s),
            (A29State.HOLDING_ORBIT, max(0.0, s.phase3_holding_s)),
        ]

    def _recover(
        self, env: simpy.Environment, *, origin: XY, bingo: bool,
    ) -> Generator[Any, Any, None]:
        """Geometric recovery to the FARP then the SBPV.

        ``bingo`` selects the entry state (``RTB_BINGO`` from the abort position
        vs ``DEPARTING_CAP`` at end of a normal PEA). With
        ``geometric_recovery_phases`` (default) both legs fly at cruise speed via
        :func:`t_to_farp`; otherwise the legacy fixed ``phase5/6`` durations are
        used. The FARP refuel resets the fuel clock.
        """
        if self.cfg.skip_pre_pea and not bingo:
            # Back-compat: skip-pre-PEA callers also skip the symbolic recovery.
            # Under ``uas_spawn_window_enabled=True`` this branch is
            # unreachable (PEA only ends via bingo), but the unified state name
            # is still used if it ever fires.
            if self.cfg.uas_spawn_window_enabled:
                self._set_state(env, A29State.RTB_FUEL_LIMIT)
            else:
                self._set_state(env, A29State.DEPARTING_CAP)
            return

        farp_xy = self._farp_xy()
        v = self.cfg.a29_cruise_speed_mps
        s = self.schedule

        if self.cfg.uas_spawn_window_enabled:
            # Unified ``RTB_FUEL_LIMIT`` terminal
            # state; log the abort metadata for ``window.json`` and trigger the
            # config's first-bingo event idempotently.
            self._set_state(env, A29State.RTB_FUEL_LIMIT)
            self.run_logger.log_rtb_fuel_limit(
                config_name=self.config_name, a29_id=self.a29_id,
                t_s=float(env.now), pos_xy=(origin.x, origin.y),
                t_pea_actual_s=float(env.now) - self.cfg.t_pea_start_s,
            )
            if (
                self.first_bingo_event is not None
                and not self.first_bingo_event.triggered
            ):
                self.first_bingo_event.succeed(
                    value={"a29_id": self.a29_id, "t_s": float(env.now)}
                )
        else:
            self._set_state(env, A29State.RTB_BINGO if bingo else A29State.DEPARTING_CAP)
        if self.cfg.geometric_recovery_phases:
            leg1 = t_to_farp(origin, farp_xy, v)
            sbpv_xy = self.projection.project(self.cfg.main_base)
            leg2 = t_to_farp(farp_xy, sbpv_xy, v)
        else:
            leg1 = s.phase5_cap_to_farp_s
            leg2 = s.phase6_farp_to_sbpv_s
        if leg1 > 0:
            yield env.timeout(leg1)

        # FARP refuel — reset the fuel clock; default 0 s delay.
        self._set_state(env, A29State.AT_FARP_POST_PEA)
        self._t_farp_departure = float(env.now)
        if self.cfg.farp_refuel_time_s > 0:
            yield env.timeout(self.cfg.farp_refuel_time_s)

        self._set_state(env, A29State.RECOVERING_TO_BASE)
        if leg2 > 0:
            yield env.timeout(leg2)

    def _set_state(self, env: simpy.Environment, new_state: A29State) -> None:
        if new_state == self.state:
            return
        self.run_logger.log_state(
            config=self.config_name, a29=self.a29_id,
            t_s=float(env.now), old=self.state.value, new=new_state.value,
        )
        self.state = new_state
