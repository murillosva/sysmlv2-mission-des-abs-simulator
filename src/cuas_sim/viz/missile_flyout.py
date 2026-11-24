"""Visual rocket flyout events for the matplotlib animator.

Turns the per-engagement records already persisted in the run directory
(``engagements.parquet`` + ``spawns.parquet``) plus the pre-computed A-29
replay trajectories (see :mod:`cuas_sim.viz.replay`) into a flat list of
:class:`MissileEvent` describing each projectile in flight. ``viz.animate``
consumes the list to draw a small marker travelling from shooter to target
during ``[t_fire_s, t_impact_s]``, after which the pre-existing
neutralisation / miss flash takes over.

Units
-----
The design first named the position fields ``shooter_xy_m`` / ``target_xy_m``
"in meters", but the Phase-2 trajectory cache stores ``(t_s, lon_deg, lat_deg,
heading_rad)`` in **degrees** and the animator plots in lon/lat. To avoid a
projection round-trip on the visual path, :class:`MissileEvent` stores both
endpoints as ``(lon_deg, lat_deg)`` — the animator's native frame — under the
unit-agnostic names ``shooter_xy`` / ``target_xy``. The time-of-flight, which
genuinely needs meters, is taken from the engagement record's
``firing_distance_m`` field (already in meters), so no projection is performed
inside this module and it stays pure. Tests that need a metric distance
project the lon/lat back through :class:`LocalProjection` themselves.

Design constraints:

* Reproducibility — every function here is **pure**: no
  RNG, no I/O, no global side-effects. Output is deterministic per run-dir.
* Formula integrity — ``v_munition_mps`` is a purely
  **visual** parameter (default 800 m/s, APKWS ≈ Mach 2.3). It times the
  marker only and MUST NOT enter Eq. 3.14 (P_hit) or any metric aggregation.
* Scope — the second of the three replay modules of ``viz/``.

SysML traceability:

* ``v_munition_mps`` is **intentionally not** SysML-rooted — it affects no
  system requirement or metric, only the visual marker timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cuas_sim.geometry.projection import EARTH_RADIUS_M

if TYPE_CHECKING:
    import pandas as pd

_DEG_PER_RAD: float = 180.0 / math.pi

_SHOT_SPACING_S: float = 13.0
"""Visual spacing between successive shots of a multi-shot engagement.

Approximates the mean of the re-engagement interval
``Triangular(8, 12, 18)`` (mean ≈ 12.7 s, rounded to 13 s, matching
:mod:`cuas_sim.viz.replay`). The simulator persists only the first shot's
``t_fire_s``, so per-shot timing here is a **visual approximation, not
bit-exact** with the engagement's RNG samples.
"""


@dataclass(frozen=True, slots=True)
class MissileEvent:
    """One projectile in flight, for the animator's missile layer.

    ``shooter_xy`` / ``target_xy`` are ``(lon_deg, lat_deg)`` — see the
    module-level unit-reconciliation note. ``shot_index`` is 1-based.
    """

    t_fire_s: float
    t_impact_s: float
    shooter_xy: tuple[float, float]
    target_xy: tuple[float, float]
    is_neutralized: bool
    shot_index: int


def missile_events_from_engagements(
    engagements_df: pd.DataFrame,
    a29_trajectory_cache: dict[str, dict[str, np.ndarray]],
    spawns_df: pd.DataFrame,
    v_munition_mps: float = 800.0,
) -> list[MissileEvent]:
    """Build visual missile-in-flight events from engagement records.

    For every feasible engagement, emits ``n_shots`` :class:`MissileEvent`
    objects (each missile flies the APKWS time-of-flight approximation
    ``tof = firing_distance_m / v_munition_mps``). The fire/impact schedule
    depends on the outcome:

    * **Neutralised** — the controller stops at the first hit
      ([controller.py] ``_engage_after_envelope``), so the logged ``n_shots``-th
      shot *is* the kill at ``t_kill_s``. The ``n_shots`` impacts are spread
      evenly with the last landing exactly on ``t_kill_s`` (fire = impact −
      ``tof``). This makes the killing missile coincide with the kill flash,
      launches nothing after the kill, and gives even a single-shot instant kill
      (run-dir logs ``t_kill_s == t_fire_s``) a real ``tof``-long flight instead
      of a zero-span marker.
    * **Miss / no kill** — shots keep the fixed ~13 s cadence from ``t_fire_s``
      with full time-of-flight.

    The shooter endpoint is looked up in ``a29_trajectory_cache`` at the shot's
    fire time; the target endpoint is the UAS position projected to the impact
    time by constant velocity from its spawn (same equirectangular kinematics as
    the animator's ``_uas_frame``).

    Multi-shot timing is a **visual approximation, not bit-exact** with the
    simulator's ``Triangular(8, 12, 18)`` re-engagement RNG — the run-dir
    persists only ``t_fire_s`` (first shot) and ``t_kill_s``.
    Only the last shot of a neutralised engagement carries
    ``is_neutralized=True`` (the killing round); earlier shots are misses.

    ``v_munition_mps`` is a **visual-only** parameter (default 800 m/s, APKWS
    ≈ Mach 2.3). It MUST NOT feed Eq. 3.14 (P_hit) or any metric.

    Parameters
    ----------
    engagements_df
        Frame matching the ``engagements.parquet`` schema
        (:class:`cuas_sim.metrics.records.EngagementRecord`). May be a
        config-filtered view; rows are dispatched by their own ``config_name``.
    a29_trajectory_cache
        ``{config: {a29_id: (N, 4) ndarray}}`` of ``(t_s, lon, lat, heading)``
        replay trajectories. Used only for the shooter lookup.
    spawns_df
        Frame matching the ``spawns.parquet`` schema (UAS birth records).
    v_munition_mps
        Visual munition speed (m/s, > 0). Default 800.0.

    Returns
    -------
    list[MissileEvent]
        One entry per shot, across all feasible engagements in
        ``engagements_df``. Empty when there is nothing to draw.
    """
    if v_munition_mps <= 0.0:
        raise ValueError(f"v_munition_mps must be > 0, got {v_munition_mps}")

    if engagements_df is None or engagements_df.empty or spawns_df is None or spawns_df.empty:
        return []

    required = {"config_name", "a29_id", "uas_id", "t_fire_s", "firing_distance_m"}
    if not required.issubset(engagements_df.columns):
        return []

    spawns_idx = spawns_df.set_index("uas_id")

    events: list[MissileEvent] = []
    for _, eng in engagements_df.iterrows():
        if not bool(eng.get("feasible", True)):
            continue

        config = str(eng["config_name"])
        a29_id = str(eng["a29_id"])
        traj = a29_trajectory_cache.get(config, {}).get(a29_id)
        if traj is None or traj.shape[0] == 0:
            continue

        try:
            t_fire = float(eng["t_fire_s"])
            firing_distance = float(eng["firing_distance_m"])
        except (TypeError, ValueError):
            continue
        if math.isnan(t_fire) or math.isnan(firing_distance):
            continue

        uid = eng["uas_id"]
        if uid not in spawns_idx.index:
            continue
        spawn = spawns_idx.loc[uid]
        spawn_t = float(spawn["spawn_time_s"])
        spawn_lat = float(spawn["lat"])
        spawn_lon = float(spawn["lon"])
        uas_heading = float(spawn["heading_rad"])
        uas_speed = float(spawn["speed_mps"])

        t_kill = _opt_float(eng.get("t_kill_s"))
        eng_neutralized = bool(eng.get("is_neutralized", False))
        n_shots = max(1, int(eng.get("n_shots", 1) or 1))

        tof = firing_distance / v_munition_mps
        # Fire/impact schedule per engagement:
        #   • Neutralised — the controller stops at the first hit, so the logged
        #     n_shots-th shot IS the kill at t_kill_s. Spread the n_shots impacts
        #     evenly, the last landing exactly on t_kill, each missile flying
        #     `tof` (fire = impact − tof). The killing missile thus coincides
        #     with the kill flash, no shot is launched after the kill, and even a
        #     single-shot instant kill (run-dir logs t_kill == t_fire) shows a
        #     real `tof`-long flight rather than collapsing to zero span.
        #   • Miss / no kill — fixed ~13 s cadence from t_fire, full ToF.
        if eng_neutralized and t_kill is not None:
            if n_shots <= 1:
                impacts = [t_kill]
            else:
                first_impact = min(t_fire + tof, t_kill)
                step = (t_kill - first_impact) / (n_shots - 1)
                impacts = [first_impact + i * step for i in range(n_shots)]
            schedule = [
                (k, impacts[k - 1] - tof, impacts[k - 1])
                for k in range(1, n_shots + 1)
            ]
        else:
            schedule = [
                (
                    k,
                    t_fire + (k - 1) * _SHOT_SPACING_S,
                    t_fire + (k - 1) * _SHOT_SPACING_S + tof,
                )
                for k in range(1, n_shots + 1)
            ]

        for idx, (k, t_fire_k, t_impact_k) in enumerate(schedule):
            # Only the last shot of a neutralised engagement is the killing round.
            shot_neutralized = eng_neutralized and (idx == len(schedule) - 1)
            shooter = _lonlat_at(traj, t_fire_k)
            target = _uas_lonlat_at(
                spawn_lon, spawn_lat, uas_heading, uas_speed, spawn_t, t_impact_k
            )
            events.append(MissileEvent(
                t_fire_s=t_fire_k,
                t_impact_s=t_impact_k,
                shooter_xy=shooter,
                target_xy=target,
                is_neutralized=shot_neutralized,
                shot_index=k,
            ))

    return events


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _opt_float(value: object) -> float | None:
    """Coerce ``value`` to float, returning ``None`` for missing / NaN."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _lonlat_at(arr: np.ndarray, t_s: float) -> tuple[float, float]:
    """(lon, lat) of an A-29 replay trajectory at ``t_s`` (binary lookup + lerp).

    Mirrors the animator's :meth:`MissionAnimator._a29_traj_at` position lookup
    on a ``(N, 4)`` ``(t_s, lon, lat, heading)`` array; ``t_s`` outside the span
    clamps to the first / last sample.
    """
    ts = arr[:, 0]
    if t_s <= ts[0]:
        return float(arr[0, 1]), float(arr[0, 2])
    if t_s >= ts[-1]:
        return float(arr[-1, 1]), float(arr[-1, 2])
    j = int(np.searchsorted(ts, t_s, side="left"))
    t0, lon0, lat0, _ = arr[j - 1]
    t1, lon1, lat1, _ = arr[j]
    span = t1 - t0
    frac = (t_s - t0) / span if span > 0.0 else 0.0
    return float(lon0 + frac * (lon1 - lon0)), float(lat0 + frac * (lat1 - lat0))


def _uas_lonlat_at(
    spawn_lon: float,
    spawn_lat: float,
    heading_rad: float,
    speed_mps: float,
    spawn_t_s: float,
    t_s: float,
) -> tuple[float, float]:
    """UAS (lon, lat) at ``t_s`` by constant-velocity projection from spawn.

    Same equirectangular kinematics the animator uses for live UAS markers
    (``MissionAnimator._uas_frame``) so the missile target tracks the same
    on-screen point the UAS marker occupies.
    """
    dt = t_s - spawn_t_s
    lat_rad = math.radians(spawn_lat)
    dx = speed_mps * math.cos(heading_rad) * dt
    dy = speed_mps * math.sin(heading_rad) * dt
    lon = spawn_lon + dx / (EARTH_RADIUS_M * math.cos(lat_rad)) * _DEG_PER_RAD
    lat = spawn_lat + dy / EARTH_RADIUS_M * _DEG_PER_RAD
    return lon, lat
