"""Engagement controller — pursuit, MOE-2 sampling, firing, outcome.

The controller models a single engagement of a single A-29 against a single
UAS, starting from the moment the E-99M dispatcher hands the target off. Its
responsibility is to produce **one** :class:`EngagementOutcome` record (and a
list of MOE-2 raw samples in the pursuit annulus) that feeds the metrics
aggregator. It does **not** mutate the A-29's lifecycle state machine — that
remains the responsibility of :class:`cuas_sim.agents.a29.A29`.

The pursuit is integrated in discrete steps of pure pursuit
(:func:`cuas_sim.geometry.intercept.pure_pursuit_step`). Time is not stepped at
1 s resolution: each step lasts one heading update drawn from the tracking policy,
until the separation reaches the weapon envelope and the engagement transitions
to ``ENGAGING``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cuas_sim.geometry.intercept import (
    InterceptGeometry,
    pure_pursuit_step,
    time_to_separation,
)
from cuas_sim.geometry.projection import XY, LocalProjection, t_to_farp

if TYPE_CHECKING:
    from cuas_sim.agents.uas import UAS
    from cuas_sim.config import MissionConfig
    from cuas_sim.policies.tracking import TrackingPolicy
    from cuas_sim.policies.weapon import WeaponPolicy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EngagementOutcome:
    """All measurements collected during one engagement.

    Conventions
    -----------
    * All times in seconds.
    * All distances in meters.
    * ``moe2_samples`` are the raw :math:`P_{\\text{track}}` evaluations in the
      pursuit annulus; the per-engagement MOE-2 is their arithmetic mean.
    """

    config_name: str
    a29_id: str
    uas_id: int
    aor_name: str
    sector_index: int
    archetype_name: str
    # Timestamps (mission-clock seconds, NaN if not reached)
    t_pursuit_start_s: float
    t_envelope_entry_s: float
    t_fire_s: float
    t_kill_s: float
    # Geometry
    initial_distance_m: float
    intercept_distance_m: float
    firing_distance_m: float
    # MOS
    is_neutralized: bool          # bernoulli realisation
    is_neutralized_before_zida: bool  # MOS-1 numerator
    # MOE
    moe1_engagement_time_s: float   # t_end - t_not; t_ef of Eq. (3.6) if neutralized
    moe2_samples: list[float] = field(default_factory=list)
    moe3_p_hit_expected: float = 0.0  # P_neut,i of Eq. (3.15)
    # MOP
    mop1_heading_update_count: int = 0
    mop2_acquisition_range_m: float = 0.0
    mop3_envelope_max_m: float = 0.0
    # Multi-shot re-engagement diagnostics
    n_shots: int = 0
    rockets_remaining: int = 0
    end_xy: XY | None = None
    # Bingo-fuel truncation
    bingo_aborted: bool = False
    bingo_xy: XY | None = None
    bingo_t_s: float = float("nan")
    # Evasion instant (projected 80 nmi ZIDA crossing) for the
    # MOE-1 Eq. (3.6) time of evaded engagements. ``NaN`` if neutralized / N/A.
    t_ev_s: float = float("nan")
    # Diagnostics
    feasible: bool = True
    note: str = ""


class EngagementController:
    """Drives one engagement from notification to outcome."""

    def __init__(
        self,
        cfg: MissionConfig,
        tracking_policy: TrackingPolicy,
        weapon_policy: WeaponPolicy,
        projection: LocalProjection,
        config_name: str,
    ) -> None:
        self.cfg = cfg
        self.tracking = tracking_policy
        self.weapon = weapon_policy
        self.projection = projection
        self.config_name = config_name

    def run(
        self,
        *,
        a29_id: str,
        aor_name: str,
        uas: UAS,
        a29_xy_at_notify: XY,
        t_notify_s: float,
        rng_tracking: np.random.Generator,
        rng_firing: np.random.Generator,
        rng_p_hit: np.random.Generator,
        rng_reengage: np.random.Generator,
        rockets_available: int | None = None,
        fuel_exhaust_t: float = float("inf"),
        farp_xy: XY | None = None,
    ) -> EngagementOutcome:
        """Compute the full engagement outcome (closed-form fast-forward).

        Parameters
        ----------
        a29_id
            Designator for logging (e.g. ``"A-29-3"``).
        aor_name
            AoR label for logging and the leak distribution.
        uas
            The target UAS (kinematics are read; ``is_alive`` is mutated by
            the caller, not here).
        a29_xy_at_notify
            Projected position of the A-29 at the moment of notification.
        t_notify_s
            Mission-clock time of notification by the E-99M.
        rng_tracking, rng_firing, rng_p_hit
            Per-config decision RNGs (already split by ``RNGRegistry``).
        fuel_exhaust_t
            Absolute mission time at which the A-29's tank empties
            (``t_farp_departure + a29_endurance_s``). Default ``inf`` disables
            the bingo trigger (callers that do not model fuel).
        farp_xy
            Projected position of the A-29's designated FARP. Required for the
            bingo trigger; ``None`` disables it.
        """
        a_speed = self.cfg.a29_cruise_speed_mps
        env_max = self.weapon.envelope_max_m
        budget = (self.cfg.rocket_budget_per_sortie
                  if rockets_available is None else int(rockets_available))

        # Projected instant this UAS crosses the 80 nmi
        # ZIDA depth (the Eq. 3.6 time of evaded engagements). Pure function
        # of the UAS path, independent of the outcome; computed once and set on
        # every non-neutralized return (NaN on neutralization).
        from cuas_sim.world.boundary import ZIDABoundary

        t_ev = ZIDABoundary(radius_nmi=self.cfg.zida_boundary_nmi).crossing_time_s(
            spawn_point=uas.spawn_point, spawn_xy=uas.spawn_xy,
            heading_rad=uas.heading_rad, speed_mps=uas.speed_mps,
            spawn_time_s=uas.spawn_time_s, projection=self.projection,
        )

        # 1) PURSUING — discrete-integration pure-pursuit loop.
        #    The A-29 freezes its heading toward the current UAS position once
        #    per tracking update and flies that heading for ``dt`` seconds. A
        #    single loop drives both the kinematics and MOE-2 sampling at the
        #    tracking-update cadence (Table 3.3).
        u_xy_notify = uas.xy_at(t_notify_s)
        initial_distance_m = math.hypot(
            u_xy_notify.x - a29_xy_at_notify.x,
            u_xy_notify.y - a29_xy_at_notify.y,
        )

        a29_xy: tuple[float, float] = (a29_xy_at_notify.x, a29_xy_at_notify.y)
        uas_xy: tuple[float, float] = (u_xy_notify.x, u_xy_notify.y)
        uas_vxy: tuple[float, float] = (
            uas.speed_mps * math.cos(uas.heading_rad),
            uas.speed_mps * math.sin(uas.heading_rad),
        )
        t = t_notify_s
        moe2_samples: list[float] = []
        mop1_count = 0
        prev_sep = initial_distance_m
        n_non_decreasing = 0

        # First step uses ``first_update_interval`` (baseline 12.5 s floor,
        # modernized U(1,10) — Table 3.3).
        dt = self.tracking.first_update_interval(rng_tracking)

        while True:
            a29_xy, uas_xy = pure_pursuit_step(a29_xy, uas_xy, uas_vxy, a_speed, dt)
            t += dt
            sep = math.hypot(uas_xy[0] - a29_xy[0], uas_xy[1] - a29_xy[1])
            mop1_count += 1
            if self.tracking.in_sampling_annulus(sep):
                moe2_samples.append(
                    self.tracking.p_track(sep, uas.archetype, rng_tracking)
                )
            # Bingo-fuel truncation: evaluated every
            # step. Reserve zero — the A-29 aborts the pursuit the instant the
            # remaining endurance no longer covers the flight back to the FARP.
            # Bingo takes precedence over envelope entry (out of fuel ⇒ cannot
            # engage).
            if farp_xy is not None:
                a29_xy_now = XY(a29_xy[0], a29_xy[1])
                if (fuel_exhaust_t - t) <= t_to_farp(a29_xy_now, farp_xy, a_speed):
                    return EngagementOutcome(
                        config_name=self.config_name,
                        a29_id=a29_id,
                        uas_id=uas.uas_id,
                        aor_name=aor_name,
                        sector_index=uas.sector_index,
                        archetype_name=uas.archetype.name,
                        t_pursuit_start_s=t_notify_s,
                        t_envelope_entry_s=float("nan"),
                        t_fire_s=float("nan"),
                        t_kill_s=float("nan"),
                        initial_distance_m=initial_distance_m,
                        intercept_distance_m=float("nan"),
                        firing_distance_m=float("nan"),
                        is_neutralized=False,
                        is_neutralized_before_zida=False,
                        moe1_engagement_time_s=0.0,
                        moe2_samples=moe2_samples,
                        mop1_heading_update_count=mop1_count,
                        mop2_acquisition_range_m=self._mop2(),
                        mop3_envelope_max_m=env_max,
                        n_shots=0,
                        rockets_remaining=budget,
                        end_xy=a29_xy_now,
                        bingo_aborted=True,
                        bingo_xy=a29_xy_now,
                        bingo_t_s=t,
                        t_ev_s=t_ev,
                        feasible=True,
                        note="bingo-fuel — aborted during pursuit",
                    )
            # Envelope-entry → ENGAGING.
            if sep <= env_max:
                t_envelope_entry = t
                break
            # Infeasible: separation non-decreasing for N steps.
            if sep >= prev_sep:
                n_non_decreasing += 1
                if n_non_decreasing >= self.cfg.n_infeasible_updates:
                    return EngagementOutcome(
                        config_name=self.config_name,
                        a29_id=a29_id,
                        uas_id=uas.uas_id,
                        aor_name=aor_name,
                        sector_index=uas.sector_index,
                        archetype_name=uas.archetype.name,
                        t_pursuit_start_s=t_notify_s,
                        t_envelope_entry_s=float("nan"),
                        t_fire_s=float("nan"),
                        t_kill_s=float("nan"),
                        initial_distance_m=initial_distance_m,
                        intercept_distance_m=float("nan"),
                        firing_distance_m=float("nan"),
                        is_neutralized=False,
                        is_neutralized_before_zida=False,
                        moe1_engagement_time_s=0.0,
                        moe2_samples=moe2_samples,
                        mop1_heading_update_count=mop1_count,
                        t_ev_s=t_ev,
                        feasible=False,
                        note="infeasible — non-decreasing separation",
                        mop2_acquisition_range_m=self._mop2(),
                        mop3_envelope_max_m=env_max,
                        n_shots=0,
                        rockets_remaining=budget,
                    )
            else:
                n_non_decreasing = 0
            prev_sep = sep
            dt = self.tracking.next_update_interval(rng_tracking)

        # 2) Build the post-envelope geometry passed to ENGAGING. The A-29
        #    heading is frozen at the bearing to the UAS at envelope-entry
        #    ``t_intercept`` is computed from this frozen heading via
        #    ``time_to_separation``.
        a29_xy_at_envelope = XY(a29_xy[0], a29_xy[1])
        u_xy_at_envelope = XY(uas_xy[0], uas_xy[1])
        heading_at_envelope = math.atan2(
            uas_xy[1] - a29_xy[1], uas_xy[0] - a29_xy[0]
        )
        tr_intercept = time_to_separation(
            a29_xy_at_envelope, a_speed, u_xy_at_envelope, uas.heading_rad,
            uas.speed_mps, a29_heading_rad=heading_at_envelope, target_sep_m=0.0,
        )
        if tr_intercept is None or tr_intercept <= 0.0:
            # Fallback: closing rate estimate; only used to clamp ``_t_at_sep``.
            closing_rate = max(a_speed - uas.speed_mps, 1.0)
            tr_intercept = sep / closing_rate
        t_intercept = t_envelope_entry + tr_intercept

        post_envelope_geom = InterceptGeometry(
            feasible=True,
            t_intercept=tr_intercept,
            intercept_xy=XY(float("nan"), float("nan")),
            a29_heading_rad=heading_at_envelope,
            t_envelope_entry=0.0,
            initial_distance_m=sep,
        )

        # 3) ENGAGING — multi-shot re-engagement, in
        #    ``_engage_after_envelope``.
        return self._engage_after_envelope(
            a29_id=a29_id,
            aor_name=aor_name,
            uas=uas,
            t_pursuit_start_s=t_notify_s,
            t_envelope_entry=t_envelope_entry,
            t_intercept=t_intercept,
            a29_xy_at_envelope=a29_xy_at_envelope,
            u_xy_at_envelope=u_xy_at_envelope,
            geom=post_envelope_geom,
            a_speed=a_speed,
            env_max=env_max,
            budget=budget,
            moe2_samples=moe2_samples,
            mop1_count=mop1_count,
            rng_firing=rng_firing,
            rng_p_hit=rng_p_hit,
            rng_reengage=rng_reengage,
            fuel_exhaust_t=fuel_exhaust_t,
            farp_xy=farp_xy,
            t_ev=t_ev,
        )

    def _engage_after_envelope(
        self,
        *,
        a29_id: str,
        aor_name: str,
        uas: UAS,
        t_pursuit_start_s: float,
        t_envelope_entry: float,
        t_intercept: float,
        a29_xy_at_envelope: XY,
        u_xy_at_envelope: XY,
        geom: InterceptGeometry,
        a_speed: float,
        env_max: float,
        budget: int,
        moe2_samples: list[float],
        mop1_count: int,
        rng_firing: np.random.Generator,
        rng_p_hit: np.random.Generator,
        rng_reengage: np.random.Generator,
        fuel_exhaust_t: float = float("inf"),
        farp_xy: XY | None = None,
        t_ev: float = float("nan"),
    ) -> EngagementOutcome:
        """Multi-shot re-engagement.

        The philosophy sets the FIRST shot distance; medium/long then re-fire
        every ``Triangular(*reengagement_triangular_s)`` as the A-29 closes on the fixed
        collision heading, until hit / floor / rocket budget.

        Geometry base: envelope-entry instant. The A-29 heading
        is frozen at the bearing to the UAS at envelope-entry (``geom.a29_heading_rad``)
        and the closed-form ``time_to_separation`` primitive is reused to map
        firing distances to mission times.

        ``t_pursuit_start_s`` is the notify mission time, preserved here only
        to populate :attr:`EngagementOutcome.t_pursuit_start_s` and the MOE-1
        numerator. All kinematic calculations use ``t_envelope_entry`` as base.
        """
        d1, philosophy_idx = self.weapon.draw_firing_distance(rng_firing)
        floor = self.weapon.floor_m
        # Short = single shot, keyed off the philosophy
        # index (0), not the old ``d1 <= envelope_min`` test (the short sub-band
        # is now ``]min, min+R/3]``, strictly above ``envelope_min``).
        is_short = philosophy_idx == 0

        def _t_at_sep(target: float) -> float:
            if target >= geom.initial_distance_m:
                return t_envelope_entry
            tr = time_to_separation(
                a29_xy_at_envelope, a_speed, u_xy_at_envelope, uas.heading_rad,
                uas.speed_mps, a29_heading_rad=geom.a29_heading_rad,
                target_sep_m=target,
            )
            if tr is None or (t_envelope_entry + tr) > t_intercept:
                return t_intercept
            return min(max(t_envelope_entry + tr, t_envelope_entry), t_intercept)

        def _bingo_at(t_k: float) -> bool:
            # Bingo on the frozen-heading position at the
            # planned shot time. Reserve zero. ``inf`` endurance ⇒ never fires.
            if farp_xy is None:
                return False
            pos = _a29_xy_at(geom, a29_xy_at_envelope, t_envelope_entry, a_speed, t_k)
            return (fuel_exhaust_t - t_k) <= t_to_farp(pos, farp_xy, a_speed)

        bingo_aborted = False
        bingo_xy: XY | None = None
        bingo_t_s = float("nan")

        t1 = _t_at_sep(d1)
        # The very first shot may itself be pre-empted by bingo (the shot does
        # not resolve its Bernoulli — abort immediately).
        shots: list[tuple[float, float]] = []
        if _bingo_at(t1):
            bingo_aborted = True
            bingo_t_s = t1
            bingo_xy = _a29_xy_at(geom, a29_xy_at_envelope, t_envelope_entry, a_speed, t1)
        else:
            shots.append((t1, max(d1, floor)))
            if not is_short:
                tk = t1
                tri_l, tri_m, tri_r = self.cfg.reengagement_triangular_s
                while len(shots) < budget:
                    # Degenerate Triangular (left == right) is a valid config but
                    # ``numpy.triangular`` rejects it — treat as a constant.
                    dt = (tri_l if tri_l == tri_r
                          else float(rng_reengage.triangular(tri_l, tri_m, tri_r)))
                    tk += dt
                    if tk > t_intercept:
                        break
                    if _bingo_at(tk):
                        bingo_aborted = True
                        bingo_t_s = tk
                        bingo_xy = _a29_xy_at(
                            geom, a29_xy_at_envelope, t_envelope_entry, a_speed, tk
                        )
                        break
                    sep = _distance(self.projection, uas, geom, a29_xy_at_envelope,
                                    t_envelope_entry, a_speed, tk)
                    if sep <= floor:
                        break
                    shots.append((tk, sep))
        shots = shots[:max(0, budget)]

        # Per-shot P_hit at the real separation / active regime.
        p_seq = [self.weapon.p_hit_at(sep, uas.archetype) for _, sep in shots]
        p_neut = cumulative_neutralization(p_seq)          # MOE-3

        # Realized Bernoulli — stop at the first hit (MOS-1).
        is_neut = False
        t_kill = float("nan")
        kill_sep = float("nan")
        n_shots = len(shots)
        for k, ((t_k, sep_k), p_k) in enumerate(zip(shots, p_seq, strict=True), start=1):
            if float(rng_p_hit.random()) < p_k:
                is_neut = True
                t_kill = t_k
                kill_sep = sep_k
                n_shots = k
                break

        # End instant / position. A kill returns from the kill point; a bingo
        # abort with no kill returns from the bingo position;
        # otherwise the last fired shot.
        t_fire = t1 if shots else float("nan")
        last_sep = shots[-1][1] if shots else float("nan")
        if is_neut:
            t_end = t_kill
        elif bingo_aborted:
            t_end = bingo_t_s
        else:
            t_end = shots[-1][0]
        if bingo_aborted and not is_neut:
            end_xy = bingo_xy
        else:
            end_xy = _a29_xy_at(geom, a29_xy_at_envelope, t_envelope_entry, a_speed, t_end)
        rockets_remaining = max(0, budget - n_shots)

        # 4) MOS-1 — neutralized before ZIDA? (80 nmi = scoring only.)
        from cuas_sim.world.boundary import ZIDABoundary

        zida = ZIDABoundary(radius_nmi=self.cfg.zida_boundary_nmi)
        is_before_zida = False
        if is_neut:
            kill_point = self.projection.unproject(uas.xy_at(t_kill))
            is_before_zida = zida.is_inside(uas.spawn_point, kill_point)

        # 5) Engagement time, notification → engagement end: the Eq. (3.6) time of a
        #    neutralized engagement (the aggregator uses t_ev for evaded ones).
        moe1 = max(0.0, t_end - t_pursuit_start_s)

        return EngagementOutcome(
            config_name=self.config_name,
            a29_id=a29_id,
            uas_id=uas.uas_id,
            aor_name=aor_name,
            sector_index=uas.sector_index,
            archetype_name=uas.archetype.name,
            t_pursuit_start_s=t_pursuit_start_s,
            t_envelope_entry_s=t_envelope_entry,
            t_fire_s=t_fire,
            t_kill_s=t_kill,
            initial_distance_m=geom.initial_distance_m,
            intercept_distance_m=(kill_sep if is_neut else last_sep),
            firing_distance_m=d1,
            is_neutralized=is_neut,
            is_neutralized_before_zida=is_before_zida,
            moe1_engagement_time_s=moe1,
            moe2_samples=moe2_samples,
            moe3_p_hit_expected=p_neut,
            mop1_heading_update_count=mop1_count,
            mop2_acquisition_range_m=self._mop2(),
            mop3_envelope_max_m=env_max,
            n_shots=n_shots,
            rockets_remaining=rockets_remaining,
            end_xy=end_xy,
            bingo_aborted=bingo_aborted,
            bingo_xy=bingo_xy,
            bingo_t_s=bingo_t_s,
            # t_ev only for evaded (non-neutralized) engagements.
            t_ev_s=(float("nan") if is_neut else t_ev),
            feasible=True,
        )

    # ─── MOP helpers ─────────────────────────────────────────────────────

    def _mop2(self) -> float:
        """MOP-2: nominal acquisition range for the configuration."""
        if self.config_name == "baseline":
            return self.cfg.baseline_max_acquisition_m
        return self.cfg.modernized_eoir_max_m


def cumulative_neutralization(p_hits: list[float]) -> float:
    """Eq. (3.15): probability of at least one hit over the
    planned shot sequence — ``1 − ∏_k (1 − p_k)``. Empty ⇒ 0.0."""
    prod = 1.0
    for p in p_hits:
        prod *= 1.0 - p
    return float(1.0 - prod)


def _a29_xy_at(
    geom: InterceptGeometry,
    a29_xy_at_notify: XY,
    t_notify_s: float,
    a_speed: float,
    t: float,
) -> XY:
    """A-29 projected position at time ``t`` on the fixed collision heading."""
    head = geom.a29_heading_rad
    dt = t - t_notify_s
    return XY(
        a29_xy_at_notify.x + a_speed * math.cos(head) * dt,
        a29_xy_at_notify.y + a_speed * math.sin(head) * dt,
    )


def _distance(
    projection: LocalProjection,
    uas: UAS,
    geom: InterceptGeometry,
    a29_xy_at_notify: XY,
    t_notify_s: float,
    a_speed: float,
    t: float,
) -> float:
    """Distance between A-29 and UAS at time ``t`` (s), in meters."""
    u_xy = uas.xy_at(t)
    head = geom.a29_heading_rad
    dt = t - t_notify_s
    ax = a29_xy_at_notify.x + a_speed * math.cos(head) * dt
    ay = a29_xy_at_notify.y + a_speed * math.sin(head) * dt
    return math.hypot(u_xy.x - ax, u_xy.y - ay)
