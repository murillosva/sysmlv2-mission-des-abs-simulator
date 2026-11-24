"""Geometric replay of A-29 trajectories for the matplotlib animator.

Reconstructs the *type* of manoeuvre executed by the
:class:`~cuas_sim.engagement.controller.EngagementController` from the data
already persisted in the run directory — ``engagements.parquet``,
``spawns.parquet``, plus the resolved :class:`~cuas_sim.config.MissionConfig`
and :class:`~cuas_sim.policies.weapon.WeaponPolicy`, plus the per-A-29
state-transition log. Output is consumed by ``viz.animate``
for fidelity-correct A-29 markers and heading quivers
across the full mission — deployment (SBPV→FARP→CAP), PURSUING, ENGAGING,
RTB-to-CAP, and recovery (CAP→FARP→SBPV).

Design constraints:

* Reproducibility — every function here is **pure**: no RNG,
  no I/O, no global side-effects. The replay is deterministic per run-dir.
* Formula integrity — no simulation formula is recomputed;
  the trajectory is geometric replay only, never a metric source.
* Scope — the first of the three replay modules of ``viz/``.

SysML traceability:

* ``v_a29_mps`` ↦ ``TechnicalComponentsPackage.A29SuperTucano.a29CruiseSpeed``
* ``envelope_max_m`` ↦ ``TechnicalComponentsPackage`` weapon envelopes
  (baseline / A-29N).
* CAP / FARP / Base coordinates ↦ ``ContextPackage`` area-of-responsibility.
* Phase ordering (PURSUING → ENGAGING → RTB → recovery) ↦
  ``MissionExecutionPackage.airspacePolicing`` / ``recoverToBase``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cuas_sim.geometry.coordinates import GeoPoint
from cuas_sim.geometry.intercept import pure_pursuit_step
from cuas_sim.geometry.projection import XY, LocalProjection
from cuas_sim.states import A29State

if TYPE_CHECKING:
    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.policies.weapon import WeaponPolicy


@dataclass(frozen=True, slots=True)
class Waypoint:
    """One sample point on a replayed trajectory."""

    t_s: float
    x_m: float
    y_m: float
    heading_rad: float


_PURSUIT_TOLERANCE_S: float = 1.0
"""Safety tolerance for the PURSUING stop condition.

The loop halts when separation ≤ ``envelope_max_m`` OR when ``t`` exceeds
``t_envelope_entry_s + tolerance`` — the tolerance absorbs the discrete
integration drift relative to the closed-form analytical reference.
"""


def replay_pursuit(
    cap_xy: tuple[float, float],
    uas_spawn_xy: tuple[float, float],
    uas_heading_rad: float,
    uas_speed_mps: float,
    t_pursuit_start_s: float,
    t_envelope_entry_s: float,
    v_a29_mps: float,
    envelope_max_m: float,
    dt: float = 2.0,
) -> list[Waypoint]:
    """Replay the PURSUING phase as a list of waypoints.

    Iterates :func:`cuas_sim.geometry.intercept.pure_pursuit_step` starting
    from ``cap_xy`` at ``t = t_pursuit_start_s``, with the UAS projected
    forward by constant velocity from ``uas_spawn_xy``. The loop terminates
    when **either** the A-29↔UAS separation falls to ``envelope_max_m`` or
    below, **or** simulated time exceeds ``t_envelope_entry_s +
    _PURSUIT_TOLERANCE_S`` — whichever happens first.

    The semantics match the discrete pure-pursuit integration
    (``EngagementController.run`` PURSUING loop) up to the tracking-policy
    update interval, which is *not* persisted in the run-dir. Replay is
    therefore geometrically faithful to the manoeuvre type but not
    bit-exact with the simulator's RNG (documented limitation).

    Parameters
    ----------
    cap_xy
        A-29 position at ``t_pursuit_start_s`` in projected meters
        (= CAP centroid for normal pre-engagement state).
    uas_spawn_xy
        UAS position at ``t_pursuit_start_s`` in projected meters. The
        caller (e.g. :func:`build_a29_trajectory`) is responsible for
        projecting from literal spawn → pursuit-start when needed.
    uas_heading_rad, uas_speed_mps
        UAS constant-velocity course (atan2 convention; m/s).
    t_pursuit_start_s
        Mission-time origin of the PURSUING phase (seconds).
    t_envelope_entry_s
        Analytical reference for the envelope-entry time, used to bound
        the loop's runtime ceiling (see ``_PURSUIT_TOLERANCE_S``).
    v_a29_mps
        A-29 cruise speed in m/s. SysML:
        ``TechnicalComponentsPackage.A29SuperTucano.a29CruiseSpeed``.
    envelope_max_m
        Stop condition: pursuit halts when separation ≤ this value.
        SysML: ``TechnicalComponentsPackage`` weapon envelopes.
    dt
        Discrete integration step (seconds, > 0). Default 2.0 s matches
        the typical baseline-tracking update interval order of magnitude.

    Returns
    -------
    list[Waypoint]
        Waypoints in strictly-increasing ``t_s`` order. The first sample is
        the initial CAP anchor; the last sample either satisfies the
        envelope condition or the time ceiling.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if v_a29_mps <= 0.0:
        raise ValueError(f"v_a29_mps must be > 0, got {v_a29_mps}")

    uas_vxy = (
        uas_speed_mps * math.cos(uas_heading_rad),
        uas_speed_mps * math.sin(uas_heading_rad),
    )

    a29 = (float(cap_xy[0]), float(cap_xy[1]))
    uas = (float(uas_spawn_xy[0]), float(uas_spawn_xy[1]))
    t = float(t_pursuit_start_s)

    initial_heading = math.atan2(uas[1] - a29[1], uas[0] - a29[0])
    waypoints: list[Waypoint] = [Waypoint(t, a29[0], a29[1], initial_heading)]

    sep = math.hypot(uas[0] - a29[0], uas[1] - a29[1])
    if sep <= envelope_max_m:
        return waypoints

    t_ceiling = float(t_envelope_entry_s) + _PURSUIT_TOLERANCE_S
    while t < t_ceiling:
        heading = math.atan2(uas[1] - a29[1], uas[0] - a29[0])
        a29, uas = pure_pursuit_step(a29, uas, uas_vxy, v_a29_mps, dt)
        t += dt
        waypoints.append(Waypoint(t, a29[0], a29[1], heading))
        sep = math.hypot(uas[0] - a29[0], uas[1] - a29[1])
        if sep <= envelope_max_m:
            break

    return waypoints


def replay_engaging(
    env_entry_xy: tuple[float, float],
    env_entry_heading_rad: float,
    t_envelope_entry_s: float,
    t_end_s: float,
    v_a29_mps: float,
    dt: float = 1.0,
) -> list[Waypoint]:
    """Replay the ENGAGING phase with frozen heading.

    Mirrors the simulator's semantics: at envelope entry the A-29
    locks heading to the bearing-to-UAS at that instant and flies straight
    at ``v_a29_mps`` until the engagement ends (kill or budget exhaustion).
    Pure linear advance — no further pursuit corrections.

    The final waypoint is anchored exactly at ``t_end_s`` even if the last
    fixed-``dt`` step would overshoot, so downstream interpolation lands on
    the correct mission-time mark.

    Parameters
    ----------
    env_entry_xy
        A-29 position at the moment of envelope entry (projected meters).
        This is the seamless handoff point from :func:`replay_pursuit`.
    env_entry_heading_rad
        Bearing to UAS at envelope entry. Frozen for the whole phase.
    t_envelope_entry_s, t_end_s
        Phase boundaries (seconds). ``t_end_s`` is derived by the caller —
        ``t_kill_s`` for neutralised engagements, otherwise the next state
        transition logged in ``events.jsonl.gz``.
    v_a29_mps
        A-29 cruise speed. SysML:
        ``A29SuperTucano.a29CruiseSpeed``.
    dt
        Sample interval (seconds, > 0). Default 1.0 s gives sub-frame
        granularity for typical 5–60 s engagements.

    Returns
    -------
    list[Waypoint]
        Waypoints sampled at ``dt`` spacing from envelope entry to
        ``t_end_s``. Each successive displacement equals ``v_a29 * Δt``
        along ``(cos h, sin h)`` exactly (oracle T2).
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if t_end_s < t_envelope_entry_s:
        raise ValueError(
            f"t_end_s ({t_end_s}) must be ≥ t_envelope_entry_s ({t_envelope_entry_s})"
        )

    cos_h = math.cos(env_entry_heading_rad)
    sin_h = math.sin(env_entry_heading_rad)
    x0, y0 = float(env_entry_xy[0]), float(env_entry_xy[1])
    t0 = float(t_envelope_entry_s)
    t_end = float(t_end_s)

    waypoints: list[Waypoint] = [Waypoint(t0, x0, y0, env_entry_heading_rad)]
    if t_end == t0:
        return waypoints

    # Advance in strict v·Δt increments; the final point lands exactly at t_end.
    t = t0 + dt
    while t < t_end - 1e-9:
        dx = v_a29_mps * cos_h * (t - t0)
        dy = v_a29_mps * sin_h * (t - t0)
        waypoints.append(Waypoint(t, x0 + dx, y0 + dy, env_entry_heading_rad))
        t += dt
    # Final anchor at exactly t_end_s.
    dx_end = v_a29_mps * cos_h * (t_end - t0)
    dy_end = v_a29_mps * sin_h * (t_end - t0)
    waypoints.append(Waypoint(t_end, x0 + dx_end, y0 + dy_end, env_entry_heading_rad))
    return waypoints


def replay_rtb(
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
    t_start_s: float,
    t_end_s: float,
) -> list[Waypoint]:
    """Replay any straight-line phase (RTB to CAP, CAP→FARP, FARP→Base).

    Linear interpolation from ``start_xy`` to
    ``target_xy`` between ``t_start_s`` and ``t_end_s``. Heading is constant
    and equal to the segment bearing.

    Two-waypoint output (start and target) is sufficient because the
    matplotlib animator's downstream interpolation is linear in both space
    and time — additional intermediate samples on a straight segment would
    be redundant.
    """
    if t_end_s < t_start_s:
        raise ValueError(
            f"t_end_s ({t_end_s}) must be ≥ t_start_s ({t_start_s})"
        )

    x0, y0 = float(start_xy[0]), float(start_xy[1])
    x1, y1 = float(target_xy[0]), float(target_xy[1])
    bearing = math.atan2(y1 - y0, x1 - x0)
    return [
        Waypoint(float(t_start_s), x0, y0, bearing),
        Waypoint(float(t_end_s), x1, y1, bearing),
    ]


# ─── Higher-level orchestration ───────────────────────────────────────────────


def build_a29_trajectory(
    a29_id: str,
    config_name: str,
    engagements_df: pd.DataFrame,
    events: list[tuple[float, str]],
    spawns_df: pd.DataFrame,
    mission_cfg: MissionConfig,
    weapon_policy: WeaponPolicy,
) -> np.ndarray:
    """Concatenate every mission phase for one A-29 into one trajectory.

    The full sequence, including the deployment and recovery legs, is:

    #. **Deployment** (SBPV → FARP → CAP) — reconstructed from the
       state-transition ``events`` of the symbolic pre-PEA phases
       (TRANSIT_TO_FARP, REPOSITIONING_TO_CAP, HOLDING_ORBIT). Without an event
       log the trajectory falls back to a single CAP anchor at ``t = 0``
       (legacy behaviour, preserved for the empty-events oracle T4 and the
       parquet-only animator smoke).
    #. **PEA engagements** — the per-A-29 engagement records walked in
       chronological order; each engagement's PURSUING → ENGAGING → RTB-to-CAP
       triple is stitched together, padding inter-engagement intervals with the
       CAP anchor so the trajectory is strictly-monotonic in time with no
       spatial teleports (oracle T4).
    #. **Recovery** (CAP → FARP → SBPV) — reconstructed from the recovery
       events (RTB_FUEL_LIMIT / DEPARTING_CAP / RTB_BINGO → AT_FARP_POST_PEA →
       RECOVERING_TO_BASE → MISSION_COMPLETE).
       Without recovery events the trajectory simply ends at the last CAP anchor.

    The symbolic legs are driven by the *event timestamps* (authoritative) with
    the geographic anchors (base / FARP / CAP) for positions, so the
    event-derived state colours in the animator always match the leg the marker
    is visually traversing.

    Parameters
    ----------
    a29_id
        Aircraft designator (e.g. ``"A-29-1"``).
    config_name
        Configuration filter — typically ``"baseline"`` or ``"modernized"``.
    engagements_df
        Frame matching the ``engagements.parquet`` schema
        (see :class:`cuas_sim.metrics.records.EngagementRecord`).
    events
        Decoded ``(t_s, state_value)`` transitions from
        ``events_<config>.jsonl.gz`` for this A-29 (sorted ascending). Drives
        the deployment and recovery legs; an empty list yields the legacy
        CAP-anchored trajectory.
    spawns_df
        Frame matching the ``spawns.parquet`` schema (UAS birth records).
    mission_cfg
        Resolved configuration — supplies AOO centroid, base / FARP / CAP
        coordinates, and the A-29 cruise speed (SysML ``a29CruiseSpeed``).
    weapon_policy
        Unused by this function — the per-engagement ``mop3_envelope_max_m``
        field already carries the relevant bound. Kept in the signature
        to allow future policies with non-uniform envelopes
        without changing the public API.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 4)``; columns ``(t_s, lon_deg, lat_deg, heading_rad)``.
        Sorted by ``t_s`` strict-monotonically.
    """
    del weapon_policy   # accepted for API stability

    proj = LocalProjection(
        lat0_deg=mission_cfg.aoo_centroid_lat,
        lon0_deg=mission_cfg.aoo_centroid_lon,
    )

    schedule = mission_cfg.a29_schedules[a29_id]
    cap_geo = mission_cfg.caps[f"CAP-{schedule.cap_index}"]
    cap_xy = proj.project(cap_geo)
    cap_tuple = (cap_xy.x, cap_xy.y)
    farp_xy = proj.project(mission_cfg.farps[schedule.farp_name])
    farp_tuple = (farp_xy.x, farp_xy.y)
    base_xy = proj.project(mission_cfg.main_base)
    base_tuple = (base_xy.x, base_xy.y)
    v_a29 = float(mission_cfg.a29_cruise_speed_mps)

    # Deployment lead-in (SBPV → FARP → CAP) from the event log. Falls back to
    # a single CAP anchor when no pre-PEA transitions are present.
    waypoints: list[Waypoint] = _deployment_waypoints(
        events, base_tuple, farp_tuple, cap_tuple
    )
    if not waypoints:
        waypoints = [Waypoint(0.0, cap_xy.x, cap_xy.y, 0.0)]

    df = (
        engagements_df[
            (engagements_df["a29_id"] == a29_id)
            & (engagements_df["config_name"] == config_name)
        ]
        .sort_values("t_pursuit_start_s")
        .reset_index(drop=True)
        if engagements_df is not None and not engagements_df.empty
        else None
    )

    rows = list(df.iterrows()) if df is not None else []
    # Next engagement's pursuit start bounds this one's recovery: a single A-29
    # engages serially, so its RTB-to-CAP cannot extend past the moment it
    # begins chasing the next UAS. The geometric RTB (and a long miss) can
    # otherwise overshoot it and break the strictly-monotonic time order the
    # animator's binary lookup depends on. ``math.inf`` for the last engagement.
    t_ps_all = [float(r["t_pursuit_start_s"]) for _, r in rows]
    for i, (_, eng) in enumerate(rows):
        next_t_ps = t_ps_all[i + 1] if i + 1 < len(t_ps_all) else math.inf
        t_ps = float(eng["t_pursuit_start_s"])
        _append_cap_hold_if_needed(waypoints, t_ps, cap_tuple, v_a29)

        if not bool(eng.get("feasible", True)):
            continue

        spawn_rows = spawns_df[spawns_df["uas_id"] == eng["uas_id"]]
        if spawn_rows.empty:
            continue
        spawn = spawn_rows.iloc[0]
        uas_h = float(spawn["heading_rad"])
        uas_v = float(spawn["speed_mps"])
        t_spawn = float(spawn["spawn_time_s"])
        spawn_xy = proj.project(GeoPoint(lat=float(spawn["lat"]), lon=float(spawn["lon"])))

        # Project UAS position to pursuit-start time.
        dt_to_ps = t_ps - t_spawn
        uas_xy_at_ps = (
            spawn_xy.x + uas_v * math.cos(uas_h) * dt_to_ps,
            spawn_xy.y + uas_v * math.sin(uas_h) * dt_to_ps,
        )

        env_max = float(eng["mop3_envelope_max_m"])
        t_env = float(eng["t_envelope_entry_s"])

        pursuit_wps = replay_pursuit(
            cap_xy=cap_tuple,
            uas_spawn_xy=uas_xy_at_ps,
            uas_heading_rad=uas_h,
            uas_speed_mps=uas_v,
            t_pursuit_start_s=t_ps,
            t_envelope_entry_s=t_env,
            v_a29_mps=v_a29,
            envelope_max_m=env_max,
            dt=2.0,
        )
        # Skip duplicate first waypoint when it coincides with the prior anchor.
        _extend_skip_duplicate(waypoints, pursuit_wps)

        last = waypoints[-1]
        # UAS position at the (replay-time) envelope entry instant.
        dt_to_env = last.t_s - t_spawn
        ux_env = spawn_xy.x + uas_v * math.cos(uas_h) * dt_to_env
        uy_env = spawn_xy.y + uas_v * math.sin(uas_h) * dt_to_env
        bearing_env = math.atan2(uy_env - last.y_m, ux_env - last.x_m)

        n_shots = int(eng.get("n_shots", 1) or 1)
        if bool(eng.get("is_neutralized", False)):
            t_end = float(eng["t_kill_s"])
        else:
            # Approximation: each subsequent shot ~13 s apart (mean of
            # Triangular(8,12,18)). Not bit-exact; this
            # parameter is visual only (see ``missile_flyout``).
            t_end = last.t_s + max(1.0, n_shots * 13.0)
        # Clamp to the serial-engagement bound (never past the next pursuit).
        t_end = max(last.t_s + 1e-3, min(t_end, next_t_ps))

        engaging_wps = replay_engaging(
            env_entry_xy=(last.x_m, last.y_m),
            env_entry_heading_rad=bearing_env,
            t_envelope_entry_s=last.t_s,
            t_end_s=t_end,
            v_a29_mps=v_a29,
            dt=1.0,
        )
        _extend_skip_duplicate(waypoints, engaging_wps)

        # RTB to CAP — straight-line at cruise speed, but never overshooting the
        # next engagement's pursuit start (serial-engagement invariant). When the
        # geometric arrival is later, the leg lands at the CAP exactly in time for
        # the next pursuit (a slightly faster apparent return — visual only).
        last = waypoints[-1]
        d_to_cap = math.hypot(cap_tuple[0] - last.x_m, cap_tuple[1] - last.y_m)
        t_arrive = min(last.t_s + d_to_cap / v_a29, next_t_ps)
        if d_to_cap > 1.0 and t_arrive > last.t_s + 1e-9:
            rtb_wps = replay_rtb(
                start_xy=(last.x_m, last.y_m),
                target_xy=cap_tuple,
                t_start_s=last.t_s,
                t_end_s=t_arrive,
            )
            _extend_skip_duplicate(waypoints, rtb_wps)

    # Recovery lead-out (CAP → FARP → SBPV) from the event log.
    _append_recovery(waypoints, events, farp_tuple, base_tuple)

    return _waypoints_to_lonlat_array(waypoints, proj)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _first_state_time(
    events: list[tuple[float, str]], *states: A29State,
) -> float | None:
    """Earliest event timestamp whose state matches any of ``states``.

    ``events`` is the ``(t_s, state_value)`` transition list for one A-29
    (sorted ascending); returns ``None`` when no listed state occurs.
    """
    wanted = {s.value for s in states}
    for t_ev, st in events:
        if str(st) in wanted:
            return float(t_ev)
    return None


def _advance_to(
    waypoints: list[Waypoint],
    target_xy: tuple[float, float],
    t_arrive: float,
) -> None:
    """Append a straight leg ending at ``target_xy`` at ``t_arrive``.

    The new waypoint carries the segment travel bearing (so the animator's
    destination-waypoint heading convention shows the correct course); a
    target coinciding with the current tail preserves the prior heading (a
    stationary hold). Non-advancing legs (``t_arrive`` ≤ tail) are skipped so
    the trajectory stays strictly monotonic in time even when an engagement
    replay drifted past an event timestamp.
    """
    last = waypoints[-1]
    if t_arrive <= last.t_s + 1e-9:
        return
    dx = target_xy[0] - last.x_m
    dy = target_xy[1] - last.y_m
    heading = last.heading_rad if math.hypot(dx, dy) <= 1.0 else math.atan2(dy, dx)
    waypoints.append(Waypoint(float(t_arrive), target_xy[0], target_xy[1], heading))


def _deployment_waypoints(
    events: list[tuple[float, str]],
    base_xy: tuple[float, float],
    farp_xy: tuple[float, float],
    cap_xy: tuple[float, float],
) -> list[Waypoint]:
    """SBPV → FARP → CAP lead-in waypoints from the pre-PEA transition log.

    Returns ``[]`` when there is no ``TRANSIT_TO_FARP`` event (e.g. an
    event-less run-dir or a ``skip_pre_pea`` config), letting the caller fall
    back to the legacy single CAP anchor. The arrival-at-FARP time is the
    ``REPOSITIONING_TO_CAP`` transition (departure for the CAP, which the agent
    logs at the same instant it reaches the FARP since ``AT_FARP_PRE_PEA`` has
    zero dwell); the arrival-at-CAP time is ``HOLDING_ORBIT`` (or ``AT_CAP_PEA``
    when holding is skipped).
    """
    t_transit = _first_state_time(events, A29State.TRANSIT_TO_FARP)
    if t_transit is None:
        return []
    t_farp = _first_state_time(
        events, A29State.REPOSITIONING_TO_CAP, A29State.AT_FARP_PRE_PEA
    )
    t_cap = _first_state_time(events, A29State.HOLDING_ORBIT, A29State.AT_CAP_PEA)

    b2f = math.atan2(farp_xy[1] - base_xy[1], farp_xy[0] - base_xy[0])
    waypoints: list[Waypoint] = [Waypoint(float(t_transit), base_xy[0], base_xy[1], b2f)]
    if t_farp is not None:
        _advance_to(waypoints, farp_xy, t_farp)
    if t_cap is not None:
        _advance_to(waypoints, cap_xy, t_cap)
    return waypoints


def _append_recovery(
    waypoints: list[Waypoint],
    events: list[tuple[float, str]],
    farp_xy: tuple[float, float],
    base_xy: tuple[float, float],
) -> None:
    """Append the CAP → FARP → SBPV recovery legs from the recovery event log.

    No-op when there is no terminal RTB transition (RTB_FUEL_LIMIT, or the
    legacy DEPARTING_CAP / RTB_BINGO). The A-29 first holds at its current
    position (the CAP after the final RTB-to-CAP) until the RTB instant, then
    flies to the FARP (arriving at ``AT_FARP_POST_PEA``), refuels in place until
    ``RECOVERING_TO_BASE``, then flies to SBPV (arriving at ``MISSION_COMPLETE``).
    Each leg is anchored on its event timestamp; positions use the geographic
    anchors, so the magenta RTB_FUEL_LIMIT / grey recovery state colours always
    coincide with the visible leg.
    """
    t_rtb = _first_state_time(
        events,
        A29State.RTB_FUEL_LIMIT,
        A29State.DEPARTING_CAP,
        A29State.RTB_BINGO,
    )
    if t_rtb is None:
        return
    t_at_farp = _first_state_time(events, A29State.AT_FARP_POST_PEA)
    t_recover = _first_state_time(events, A29State.RECOVERING_TO_BASE)
    t_complete = _first_state_time(events, A29State.MISSION_COMPLETE)

    # Hold at the current (CAP) position until the RTB departure instant.
    last = waypoints[-1]
    if t_rtb > last.t_s + 1e-9:
        waypoints.append(Waypoint(float(t_rtb), last.x_m, last.y_m, last.heading_rad))

    if t_at_farp is not None:
        _advance_to(waypoints, farp_xy, t_at_farp)
    # Refuel dwell at the FARP (default 0 s → no extra sample).
    if t_recover is not None and t_recover > waypoints[-1].t_s + 1e-9:
        tail = waypoints[-1]
        waypoints.append(Waypoint(float(t_recover), tail.x_m, tail.y_m, tail.heading_rad))
    if t_complete is not None:
        _advance_to(waypoints, base_xy, t_complete)


def _append_cap_hold_if_needed(
    waypoints: list[Waypoint],
    t_target: float,
    cap_xy: tuple[float, float],
    v_a29: float,
) -> None:
    """Ensure the trajectory holds at the CAP up to ``t_target`` (no teleports).

    If the current tail is not at the CAP, inserts a fly-back leg first so
    the spatial-gap bound ``Δd ≤ v_a29·Δt + ε`` (oracle T4) stays satisfied.
    Then anchors a stationary CAP waypoint at exactly ``t_target``.
    """
    last = waypoints[-1]
    if t_target <= last.t_s + 1e-9:
        return

    d_to_cap = math.hypot(cap_xy[0] - last.x_m, cap_xy[1] - last.y_m)
    if d_to_cap > 1.0:
        t_arrive = last.t_s + d_to_cap / v_a29
        if t_arrive < t_target - 1e-6:
            bearing = math.atan2(cap_xy[1] - last.y_m, cap_xy[0] - last.x_m)
            waypoints.append(Waypoint(t_arrive, cap_xy[0], cap_xy[1], bearing))
        else:
            # Not enough time to reach CAP — drop a colinear sample at t_target.
            frac = (t_target - last.t_s) / max(1e-9, t_arrive - last.t_s)
            x_mid = last.x_m + (cap_xy[0] - last.x_m) * frac
            y_mid = last.y_m + (cap_xy[1] - last.y_m) * frac
            bearing = math.atan2(cap_xy[1] - last.y_m, cap_xy[0] - last.x_m)
            waypoints.append(Waypoint(t_target, x_mid, y_mid, bearing))
            return

    last = waypoints[-1]
    if t_target > last.t_s + 1e-9:
        waypoints.append(Waypoint(t_target, cap_xy[0], cap_xy[1], last.heading_rad))


def _extend_skip_duplicate(
    waypoints: list[Waypoint],
    new_wps: list[Waypoint],
) -> None:
    """Append ``new_wps`` to ``waypoints`` skipping a leading duplicate sample.

    The phase-replay helpers emit their own opening anchor; when the caller
    has already placed an identical anchor (CAP hold, prior RTB landing,
    etc.) we elide it to keep ``t_s`` strictly monotonic.
    """
    if not new_wps:
        return
    if not waypoints:
        waypoints.extend(new_wps)
        return

    last = waypoints[-1]
    first = new_wps[0]
    if (
        abs(first.t_s - last.t_s) <= 1e-9
        and abs(first.x_m - last.x_m) <= 1e-6
        and abs(first.y_m - last.y_m) <= 1e-6
    ):
        waypoints.extend(new_wps[1:])
    else:
        waypoints.extend(new_wps)


def _waypoints_to_lonlat_array(
    waypoints: list[Waypoint], proj: LocalProjection,
) -> np.ndarray:
    """Project the (t, x_m, y_m, heading) sequence to (t, lon, lat, heading)."""
    arr = np.zeros((len(waypoints), 4), dtype=np.float64)
    for i, wp in enumerate(waypoints):
        geo = proj.unproject(XY(wp.x_m, wp.y_m))
        arr[i, 0] = wp.t_s
        arr[i, 1] = geo.lon
        arr[i, 2] = geo.lat
        arr[i, 3] = wp.heading_rad
    return arr
