"""Post-run mission replay animation.

Reads a paired-run output directory and renders an interactive side-by-side
matplotlib animation — Baseline (A-29) left, Modernizado (A-29N) right —
with A-29 positions interpolated from state-transition events and UAS
positions computed from spawns.parquet via the same lazy kinematic formula
used by ``UAS.xy_at()``.

Usage via CLI::

    py -m cuas_sim.cli animate outputs/runs/<timestamp>_paired/ \\
        --config configs/zida_shield_default.yaml
"""

from __future__ import annotations

import contextlib
import gzip
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod

from cuas_sim.config import MissionConfig
from cuas_sim.policies.weapon import make_weapon_policy
from cuas_sim.states import A29State
from cuas_sim.viz.hud import (
    a29_state_at,
    active_engagement_at,
    fuel_anchor_s,
    fuel_fraction,
    is_in_flight,
    render_a29_hud,
    rockets_remaining_at,
    shot_fire_times,
    should_render_hud,
    state_short_label,
)
from cuas_sim.viz.missile_flyout import (
    _SHOT_SPACING_S,
    MissileEvent,
    missile_events_from_engagements,
)
from cuas_sim.viz.replay import build_a29_trajectory
from cuas_sim.world.aor import AoRMap
from cuas_sim.world.border_polygon import BorderPolygon, default_border_polygon

# Earth equatorial radius — matches LocalProjection in geometry/projection.py
_R_M = 6_378_137.0

# Geodesic engine for the ZIDA-band perpendicular offset (mirrors
# world/border_polygon.py so the band uses the same azimuth convention).
_GEOD = Geod(ellps="WGS84")

# A-29 state → marker fill colour (single source for the markers, the HUD
# border and the A-29 STATE legend). The legend lists, in lifecycle order (the
# SysML MissionPhasesPackage order), the ten states that actually hold the
# screen. Each mission segment is one colour family, so the segment reads at a
# glance and the shade gives the state:
#   pre-PEA  — one blue ramp, light → dark as the legs go by (OKLCH h 250, C .12);
#   PEA      — the combat colours: green / amber / red / purple;
#   post-PEA — the fuel-limit magenta, then a grey ramp that fades the sortie out.
# The magenta is reserved for RTB_FUEL_LIMIT alone. The
# RETURNING_TO_CAP purple sits darker than it (#BB88FF before 2026-11-18 was
# indistinguishable from the magenta under deuteranopia, OKLab ΔE 1.8 against a
# target of 8; #A06CC4 clears 9.5 against every colour of the PEA window).
_STATE_LEGEND: tuple[tuple[A29State, str], ...] = (
    (A29State.TRANSIT_TO_FARP,      "#93D7FF"),
    (A29State.REPOSITIONING_TO_CAP, "#61A3E6"),
    (A29State.HOLDING_ORBIT,        "#3275B4"),
    (A29State.AT_CAP_PEA,           "#00CC44"),
    (A29State.PURSUING,             "#FFAA00"),
    (A29State.ENGAGING,             "#FF4400"),
    (A29State.RETURNING_TO_CAP,     "#A06CC4"),
    (A29State.RTB_FUEL_LIMIT,       "#FF00FF"),
    (A29State.RECOVERING_TO_BASE,   "#9AA7B5"),
    (A29State.MISSION_COMPLETE,     "#5E6A78"),
)

# The other five states never hold the screen on the default path, so they get
# no legend entry and wear the colour of the leg they open or close:
#   AT_BASE          — every A-29 takes off at t = 0 (opens the transit leg);
#   AT_FARP_PRE_PEA  — 0 s, its refuel is booked inside phase 1 (closes it);
#   AT_FARP_POST_PEA — 0 s by default (farp_refuel_time_s); the refuel ends the
#                      fuel emergency and opens the leg home, so it does not
#                      wear the RTB_FUEL_LIMIT magenta;
#   DEPARTING_CAP / RTB_BINGO — legacy egress (uas_spawn_window_enabled=False
#                      only), folded into the leg home for the same reason.
_STATE_COLOUR_ALIAS: dict[A29State, A29State] = {
    A29State.AT_BASE:          A29State.TRANSIT_TO_FARP,
    A29State.AT_FARP_PRE_PEA:  A29State.TRANSIT_TO_FARP,
    A29State.AT_FARP_POST_PEA: A29State.RECOVERING_TO_BASE,
    A29State.DEPARTING_CAP:    A29State.RECOVERING_TO_BASE,
    A29State.RTB_BINGO:        A29State.RECOVERING_TO_BASE,
}

_STATE_COLORS: dict[str, str] = dict(_STATE_LEGEND)
_STATE_COLORS.update(
    {alias: _STATE_COLORS[shown] for alias, shown in _STATE_COLOUR_ALIAS.items()}
)

_UAS_COLORS: dict[str, str] = {
    "multi-rotor": "#FF8C00",
    "fixed-wing":  "#FF2222",
}


def _opt_float(value: object) -> float | None:
    """Coerce ``value`` to float, returning ``None`` for missing / NaN."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _edge_side(px: float, py: float, qx: float, qy: float,
               x: float, y: float) -> float:
    """Sign of which side of directed edge ``p→q`` point ``(x, y)`` is on."""
    return (qx - px) * (y - py) - (qy - py) * (x - px)


def _shoelace(poly: list[tuple[float, float]]) -> float:
    """Signed area (shoelace) of a closed polygon — sign indicates winding."""
    n = len(poly)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return s


def _inside_both_edges(
    ax: float, ay: float, px: float, py: float, bx: float, by: float,
    ex: float, ey: float, rx: float, ry: float,
) -> bool:
    """True iff offset ``(ex, ey)`` lies on the reference (interior) side of
    both border edges ``a→p`` and ``p→b`` (``(rx, ry)`` is the interior ref)."""
    s1 = math.copysign(1.0, _edge_side(ax, ay, px, py, ex, ey))
    r1 = math.copysign(1.0, _edge_side(ax, ay, px, py, rx, ry))
    s2 = math.copysign(1.0, _edge_side(px, py, bx, by, ex, ey))
    r2 = math.copysign(1.0, _edge_side(px, py, bx, by, rx, ry))
    return s1 == r1 and s2 == r2


def _zida_band_lonlat(
    cfg: MissionConfig, polygon: BorderPolygon
) -> list[tuple[float, float]]:
    """Closed (lon, lat) ring of the ZIDA continental band.

    Western edge = the PE/CO border polyline the simulator already uses;
    eastern edge = each vertex offset 80 NM (``cfg.zida_boundary_nmi``) inward.
    For an interior vertex the inward offset is the chord-normal (perpendicular
    to the skip-vertex tangent ``a→b``) that lies on the AoO-centroid side of
    BOTH adjacent edges. On a sharp "quina" neither chord-normal passes that
    test; there we fall back to the simple eastward-perpendicular heuristic
    (``border_polygon``-style), which tracks the border cleanly without
    folding the band shut. The N/S endpoints use a pure constant-latitude
    eastward offset so the band closes with horizontal caps. DECEA eAIP
    ENR 5.2: 80 NM measured into national territory.
    """
    segs = [polygon.segment(i) for i in polygon.indices]
    # Border vertices in sector-index order; the two extremes (first/last)
    # are the N/S tripoints that receive the horizontal caps below.
    border = [segs[0].start] + [s.end for s in segs]
    d_m = cfg.zida_boundary_nmi * 1_852.0
    n = len(border) - 1
    ref_lon, ref_lat = cfg.aoo_centroid_lon, cfg.aoo_centroid_lat

    east: list[tuple[float, float]] = []
    for k, p in enumerate(border):
        if k == 0 or k == n:
            # Endpoint: due-east at constant latitude (horizontal cap).
            dlon = math.degrees(d_m / (_R_M * math.cos(math.radians(p.lat))))
            east.append((p.lon + dlon, p.lat))
        else:
            a, b = border[k - 1], border[k + 1]
            az, _, _ = _GEOD.inv(a.lon, a.lat, b.lon, b.lat)
            cm = _GEOD.fwd(p.lon, p.lat, az - 90.0, d_m)[:2]
            cp = _GEOD.fwd(p.lon, p.lat, az + 90.0, d_m)[:2]
            okm = _inside_both_edges(a.lon, a.lat, p.lon, p.lat, b.lon,
                                     b.lat, cm[0], cm[1], ref_lon, ref_lat)
            okp = _inside_both_edges(a.lon, a.lat, p.lon, p.lat, b.lon,
                                     b.lat, cp[0], cp[1], ref_lon, ref_lat)
            if okm and not okp:
                ex, ey = cm
            elif okp and not okm:
                ex, ey = cp
            else:
                # Sharp quina: neither chord-normal sits inside both edges.
                # Eastward-perpendicular fallback (border_polygon style) —
                # tracks the border cleanly without folding the band.
                perp = (az - 90.0
                        if math.sin(math.radians(az - 90.0)) > 0
                        else az + 90.0)
                ex, ey, _ = _GEOD.fwd(p.lon, p.lat, perp, d_m)
            east.append((ex, ey))

    west = [(p.lon, p.lat) for p in border]
    return west + east[::-1]


def _zida_quina_quads(
    cfg: MissionConfig, polygon: BorderPolygon
) -> list[list[tuple[float, float]]]:
    """Per-segment trapezoids covering the sharp-quina run of the border.

    The single-polygon band returned by :func:`_zida_band_lonlat` becomes a
    self-intersecting figure-eight at a sharp "quina" (e.g., south of CAP-6),
    so matplotlib's nonzero fill cancels the oppositely-wound loop and leaves
    an unpainted wedge. Drawn together with the band ring in a *single*
    compound ``PathPatch``, these per-segment trapezoids — each convex and
    oriented to match the band's winding — paint that wedge while preserving
    uniform translucency (single draw → no dark overlap). The quina run is
    detected programmatically (no hard-coded latitudes): an interior vertex
    is in the run when **neither** chord-normal sits inside both adjacent
    edges (the same fallback condition used in :func:`_zida_band_lonlat`).
    """
    segs = [polygon.segment(i) for i in polygon.indices]
    border = [segs[0].start] + [s.end for s in segs]
    d_m = cfg.zida_boundary_nmi * 1_852.0
    n = len(border) - 1
    ref_lon, ref_lat = cfg.aoo_centroid_lon, cfg.aoo_centroid_lat

    quina_verts: set[int] = set()
    for k in range(1, n):
        a, p, b = border[k - 1], border[k], border[k + 1]
        az, _, _ = _GEOD.inv(a.lon, a.lat, b.lon, b.lat)
        cm = _GEOD.fwd(p.lon, p.lat, az - 90.0, d_m)[:2]
        cp = _GEOD.fwd(p.lon, p.lat, az + 90.0, d_m)[:2]
        okm = _inside_both_edges(a.lon, a.lat, p.lon, p.lat, b.lon, b.lat,
                                 cm[0], cm[1], ref_lon, ref_lat)
        okp = _inside_both_edges(a.lon, a.lat, p.lon, p.lat, b.lon, b.lat,
                                 cp[0], cp[1], ref_lon, ref_lat)
        if not ((okm and not okp) or (okp and not okm)):
            quina_verts.add(k)

    if not quina_verts:
        return []

    # Cover segments incident to any quina vertex, plus one neighbour each
    # side so the cover blends seamlessly into the good band.
    seg_idx: set[int] = set()
    for k in quina_verts:
        for i in (k - 2, k - 1, k):
            if 0 <= i < n:
                seg_idx.add(i)

    seg_az: dict[int, float] = {}
    covers: list[list[tuple[float, float]]] = []
    for i in sorted(seg_idx):
        P, Q = border[i], border[i + 1]
        az_edge, _, _ = _GEOD.inv(P.lon, P.lat, Q.lon, Q.lat)
        # Interior normal of THIS segment: the eastward-component perpendicular
        # (``border_polygon``-style sin rule). The centroid-distance pick is
        # unreliable at the northern segments because the AoO centroid lies
        # far south — it biases the normal toward SW and the cover misses the
        # disk-fan area to the east of the border (e.g., CAP-6). The quina
        # run is entirely on the PE/CO border where "east = into Brazil", so
        # the sin rule gives the geographically-correct interior side here.
        cand_minus = az_edge - 90.0
        best_az = (cand_minus if math.sin(math.radians(cand_minus)) > 0
                   else az_edge + 90.0)
        seg_az[i] = float(best_az)
        Po_lon, Po_lat, _ = _GEOD.fwd(P.lon, P.lat, best_az, d_m)
        Qo_lon, Qo_lat, _ = _GEOD.fwd(Q.lon, Q.lat, best_az, d_m)
        covers.append([
            (float(P.lon),  float(P.lat)),
            (float(Q.lon),  float(Q.lat)),
            (float(Qo_lon), float(Qo_lat)),
            (float(Po_lon), float(Po_lat)),
        ])

    # Corner-fan KITE at each quina vertex closes the disk-fan gap that
    # per-segment trapezoids leave (a point within 80 NM of the vertex but
    # outside both perpendicular strips — e.g., CAP-6 lies in border[4]'s
    # fan, between the two adjacent normals). A 4-vertex kite [vertex,
    # offset(n_prev), offset(bisector), offset(n_next)] widens the cover
    # over the full wedge while staying ≤80 NM from the vertex (convex hull
    # of points each ≤d_m away ⇒ all interior points ≤d_m away).
    for k in sorted(quina_verts):
        i_prev, i_next = k - 1, k
        if i_prev in seg_az and i_next in seg_az:
            p = border[k]
            ap, an = seg_az[i_prev], seg_az[i_next]
            # Circular mean of the two normal azimuths → bisector direction.
            ve = math.sin(math.radians(ap)) + math.sin(math.radians(an))
            vn = math.cos(math.radians(ap)) + math.cos(math.radians(an))
            bis_az = (ap if math.hypot(ve, vn) < 1e-9
                      else math.degrees(math.atan2(ve, vn)))
            Op_lon, Op_lat, _ = _GEOD.fwd(p.lon, p.lat, ap, d_m)
            Ob_lon, Ob_lat, _ = _GEOD.fwd(p.lon, p.lat, bis_az, d_m)
            On_lon, On_lat, _ = _GEOD.fwd(p.lon, p.lat, an, d_m)
            covers.append([
                (float(p.lon),  float(p.lat)),
                (float(Op_lon), float(Op_lat)),
                (float(Ob_lon), float(Ob_lat)),
                (float(On_lon), float(On_lat)),
            ])
    return covers


@dataclass
class _A29Info:
    """Pre-computed geometry and event timeline for one A-29."""

    a29_id: str
    farp_lon: float
    farp_lat: float
    cap_lon: float
    cap_lat: float
    events: list[tuple[float, str]]   # sorted (t_s, new_state_value)
    aor_sectors: list[int]            # sector indices assigned to this AoR


class MissionAnimator:
    """Loads a run output directory and animates the mission replay."""

    def __init__(self, run_dir: Path, config_path: Path) -> None:
        self._run_dir = Path(run_dir)
        self._cfg = MissionConfig.from_yaml(config_path)
        self._aor_map = AoRMap.from_config(self._cfg)
        self._polygon = default_border_polygon(self._cfg)
        self._load_data()

    # ── data loading ─────────────────────────────────────────────────────

    def _load_data(self) -> None:
        spawns_path = self._run_dir / "spawns.parquet"
        self._spawns: pd.DataFrame = (
            pd.read_parquet(spawns_path) if spawns_path.exists()
            else pd.DataFrame(columns=[
                "uas_id", "spawn_time_s", "lat", "lon",
                "heading_rad", "speed_mps", "archetype_name", "sector_index",
            ])
        )

        eng_path = self._run_dir / "engagements.parquet"
        self._engagements: pd.DataFrame = (
            pd.read_parquet(eng_path) if eng_path.exists() else pd.DataFrame()
        )

        self._a29_info: dict[str, dict[str, _A29Info]] = {}
        for config in ("baseline", "modernized"):
            events_path = self._run_dir / f"events_{config}.jsonl.gz"
            raw = self._parse_events(events_path) if events_path.exists() else {}
            self._a29_info[config] = self._build_a29_info(raw)

        all_times: list[float] = [0.0]
        for info_map in self._a29_info.values():
            for info in info_map.values():
                all_times.extend(t for t, _ in info.events)
        if not self._spawns.empty:
            all_times.append(float(self._spawns["spawn_time_s"].max()))
        self._t_end = float(max(all_times))

        # {config: {uas_id: kill_time_s}} for neutralised UAS — per-config so a UAS
        # killed in one A-29 variant disappears only from that panel, at its own
        # t_kill (a kill outside the ZIDA still removes the marker: physical
        # destruction, distinct from the MOS-1 "evaded" scoring).
        self._kill_times: dict[str, dict[int, float]] = {"baseline": {}, "modernized": {}}
        if not self._engagements.empty and "is_neutralized" in self._engagements.columns:
            killed = self._engagements[self._engagements["is_neutralized"] == True]  # noqa: E712
            for _, row in killed.iterrows():
                kt = row.get("t_kill_s", float("nan"))
                try:
                    kt_f = float(kt)
                except (TypeError, ValueError):
                    continue
                cfg = str(row.get("config_name", ""))
                if cfg in self._kill_times and not math.isnan(kt_f):
                    self._kill_times[cfg][int(row["uas_id"])] = kt_f

        # Precompute fire/kill-flash positions per config from engagements +
        # spawn kinematics. Each entry: (flash_lon, flash_lat, t_flash, is_neut).
        # The flash is anchored at the moment the engagement *resolves* — at
        # t_kill_s for a neutralisation (the killing shot, since the controller
        # stops at the first hit) and at the last shot's fire time for a miss —
        # NOT at the first shot's t_fire_s. This keeps the green/red marker in
        # sync with the missile-flyout layer: the flash now
        # coincides with the last missile reaching the target instead of
        # appearing while shots are still en route.
        self._fire_events: dict[str, list[tuple[float, float, float, bool]]] = {
            "baseline": [], "modernized": []
        }
        if (
            not self._engagements.empty
            and not self._spawns.empty
            and "t_fire_s" in self._engagements.columns
            and "feasible" in self._engagements.columns
        ):
            spawns_idx = self._spawns.set_index("uas_id")
            feasible = self._engagements[self._engagements["feasible"] == True]  # noqa: E712
            for _, row in feasible.iterrows():
                t_fire = row.get("t_fire_s", float("nan"))
                cfg_name = str(row.get("config_name", ""))
                if cfg_name not in self._fire_events:
                    continue
                try:
                    t_fire = float(t_fire)
                except (TypeError, ValueError):
                    continue
                if math.isnan(t_fire):
                    continue
                uid = int(row["uas_id"])
                if uid not in spawns_idx.index:
                    continue
                is_neut = bool(row.get("is_neutralized", False))
                n_shots = max(1, int(row.get("n_shots", 1) or 1))
                t_kill = float(row.get("t_kill_s", float("nan")))
                if is_neut and not math.isnan(t_kill):
                    t_flash = t_kill
                else:
                    # Miss: anchor when the A-29 fires its final shot (matches the
                    # last missile's launch in the flyout layer).
                    t_flash = t_fire + (n_shots - 1) * _SHOT_SPACING_S
                sr = spawns_idx.loc[uid]
                spawn_t = float(sr["spawn_time_s"])
                lat = float(sr["lat"])
                lon = float(sr["lon"])
                heading = float(sr["heading_rad"])
                speed = float(sr["speed_mps"])
                dt = t_flash - spawn_t
                lat_rad = math.radians(lat)
                dx = speed * math.cos(heading) * dt
                dy = speed * math.sin(heading) * dt
                flash_lon = lon + dx / (_R_M * math.cos(lat_rad)) * (180.0 / math.pi)
                flash_lat = lat + dy / _R_M * (180.0 / math.pi)
                self._fire_events[cfg_name].append((flash_lon, flash_lat, t_flash, is_neut))

        # Pre-compute each A-29's geometric replay trajectory
        # (deployment SBPV→FARP→CAP + PURSUING + ENGAGING + RTB + recovery
        # CAP→FARP→SBPV) once, cached as a (N, 4) ndarray with columns
        # (t_s, lon, lat, heading_rad) sorted by t_s. This drives the
        # O(log n)-per-frame lookup in _a29_traj_at / _a29_lonlat_at, replacing
        # the former hardcoded parabolic arc. The per-A-29 state-transition log
        # (_a29_info[config][a29_id].events) drives the symbolic deployment and
        # recovery legs; without it the trajectory falls back to a CAP anchor.
        # weapon_policy is supplied for API completeness; build_a29_trajectory
        # reads the envelope from each engagement record's mop3_envelope_max_m.
        self._a29_trajectory: dict[str, dict[str, np.ndarray]] = {}
        for config in ("baseline", "modernized"):
            weapon = make_weapon_policy(self._cfg, config)
            self._a29_trajectory[config] = {
                a29_id: build_a29_trajectory(
                    a29_id=a29_id,
                    config_name=config,
                    engagements_df=self._engagements,
                    events=self._a29_info[config][a29_id].events,
                    spawns_df=self._spawns,
                    mission_cfg=self._cfg,
                    weapon_policy=weapon,
                )
                for a29_id in self._cfg.a29_schedules
            }

        # Pre-compute visual missile-flyout events per
        # config (shooter→target marker drawn in [t_fire, t_impact]). Built once
        # here; the per-frame draw in update() is a cheap window filter. Filtered
        # by config_name so each panel only sees its own engagements. v_munition
        # is the module default (visual only — NOT a metric input).
        self._missile_events: dict[str, list[MissileEvent]] = {
            "baseline": [], "modernized": []
        }
        if not self._engagements.empty and "config_name" in self._engagements.columns:
            for config in ("baseline", "modernized"):
                df_cfg = self._engagements[self._engagements["config_name"] == config]
                self._missile_events[config] = missile_events_from_engagements(
                    engagements_df=df_cfg,
                    a29_trajectory_cache=self._a29_trajectory,
                    spawns_df=self._spawns,
                )

        # per-A-29 HUD inputs reconstructed from the data
        # already persisted (engagements + state log); consumed per frame by
        # _a29_hud_fields. All derivations are pure (viz/hud.py). Three caches:
        #   • _a29_eng_windows[cfg][id] — (t_start, t_end, uas_id) per engagement,
        #     for the active-target lookup. t_start = t_pursuit_start (target shows
        #     while pursuing); t_end matches the fire/missile layers (t_kill_s for a
        #     neutralisation, else the last-shot instant).
        #   • _a29_shot_times[cfg][id] — flat, sorted list of individual shot-fire
        #     instants (from t_fire_s on, NOT the pursuit start) so the rocket count
        #     ticks down as the A-29 actually shoots, in step with the missile layer.
        #   • _a29_refuel_times[cfg][id] — AT_FARP_POST_PEA instants that re-arm the
        #     launcher and re-anchor the fuel clock.
        self._a29_eng_windows: dict[str, dict[str, list[tuple[float, float, str]]]] = {}
        self._a29_shot_times: dict[str, dict[str, list[float]]] = {}
        self._a29_refuel_times: dict[str, dict[str, list[float]]] = {}
        for config in ("baseline", "modernized"):
            self._a29_eng_windows[config] = {}
            self._a29_shot_times[config] = {}
            self._a29_refuel_times[config] = {}
            for a29_id in self._cfg.a29_schedules:
                self._a29_eng_windows[config][a29_id] = []
                self._a29_shot_times[config][a29_id] = []
                self._a29_refuel_times[config][a29_id] = [
                    t_ev for t_ev, st in self._a29_info[config][a29_id].events
                    if st == A29State.AT_FARP_POST_PEA.value
                ]
            if self._engagements.empty or "config_name" not in self._engagements.columns:
                continue
            df_cfg = self._engagements[self._engagements["config_name"] == config]
            for _, eng in df_cfg.iterrows():
                aid = str(eng["a29_id"])
                if aid not in self._a29_shot_times[config]:
                    continue
                t_start = _opt_float(eng.get("t_pursuit_start_s"))
                if t_start is None:
                    continue
                n_shots = max(0, int(eng.get("n_shots", 0) or 0))
                # Rockets leave the count at the actual firing instants (t_fire on).
                t_fire = _opt_float(eng.get("t_fire_s"))
                if n_shots > 0 and t_fire is not None:
                    t_last_fire = self._engagement_end_s(eng, n_shots, t_fire)
                    self._a29_shot_times[config][aid].extend(
                        shot_fire_times(t_fire, t_last_fire, n_shots)
                    )
                if not bool(eng.get("feasible", True)):
                    continue
                t_end = self._engagement_end_s(eng, max(1, n_shots), t_start)
                self._a29_eng_windows[config][aid].append(
                    (t_start, t_end, str(int(eng["uas_id"])))
                )
            for aid in self._cfg.a29_schedules:
                self._a29_eng_windows[config][aid].sort()
                self._a29_shot_times[config][aid].sort()

    @staticmethod
    def _engagement_end_s(eng: pd.Series, n_shots: int, t_start: float) -> float:
        """End of an engagement window for the HUD target lookup.

        Consistent with the fire-flash / missile-flyout layers: ``t_kill_s`` for
        a neutralisation, else the last-shot instant
        ``t_fire_s + (n_shots-1)·_SHOT_SPACING_S``; falls back to ``t_start`` when
        neither time is logged.
        """
        if bool(eng.get("is_neutralized", False)):
            t_kill = _opt_float(eng.get("t_kill_s"))
            if t_kill is not None:
                return t_kill
        t_fire = _opt_float(eng.get("t_fire_s"))
        if t_fire is not None:
            return t_fire + (n_shots - 1) * _SHOT_SPACING_S
        return t_start

    def _parse_events(self, path: Path) -> dict[str, list[tuple[float, str]]]:
        result: dict[str, list[tuple[float, str]]] = {}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Skip diagnostic / info-only entries (old == new), e.g. the
                # RTB_FUEL_LIMIT_INFO record emitted alongside the real
                # RTB_FUEL_LIMIT transition at the same t_s (records.py
                # log_rtb_fuel_limit). They are not state transitions and would
                # otherwise shadow the real state — turning the magenta
                # RTB_FUEL_LIMIT marker (and its HUD line) into the unknown-state
                # fallback during the bingo→FARP leg.
                if str(obj["old"]) == str(obj["new"]):
                    continue
                result.setdefault(str(obj["a29"]), []).append(
                    (float(obj["t_s"]), str(obj["new"]))
                )
        for a29_id in result:
            result[a29_id].sort(key=lambda x: x[0])
        return result

    def _build_a29_info(
        self, raw: dict[str, list[tuple[float, str]]]
    ) -> dict[str, _A29Info]:
        info: dict[str, _A29Info] = {}
        for a29_id, schedule in self._cfg.a29_schedules.items():
            farp = self._cfg.farps[schedule.farp_name]
            cap = self._cfg.caps[f"CAP-{schedule.cap_index}"]
            aor_name = self._aor_map.aor_of_a29(a29_id)
            sectors = list(self._aor_map.aor_to_sectors.get(aor_name, []))
            info[a29_id] = _A29Info(
                a29_id=a29_id,
                farp_lon=farp.lon, farp_lat=farp.lat,
                cap_lon=cap.lon,  cap_lat=cap.lat,
                events=raw.get(a29_id, []),
                aor_sectors=sectors,
            )
        return info

    # ── kinematic helpers ─────────────────────────────────────────────────

    def _a29_traj_at(
        self, config: str, a29_id: str, t_s: float
    ) -> tuple[float, float, float]:
        """(lon, lat, heading_rad) for one A-29 at time ``t_s``.

        Pure O(log n) binary lookup + linear interpolation over the replay
        trajectory pre-computed in ``_a29_trajectory``. Positions are
        interpolated linearly between the bracketing waypoints; a ``t_s``
        outside the trajectory span clamps to the first/last sample. Heading is
        taken from the destination waypoint of the bracketing segment (the
        replay's per-step travel bearing) rather than interpolated, to avoid
        angle-wrap artefacts across the ±π branch cut.
        """
        arr = self._a29_trajectory[config][a29_id]
        ts = arr[:, 0]
        if t_s <= ts[0]:
            return float(arr[0, 1]), float(arr[0, 2]), float(arr[0, 3])
        if t_s >= ts[-1]:
            return float(arr[-1, 1]), float(arr[-1, 2]), float(arr[-1, 3])
        j = int(np.searchsorted(ts, t_s, side="left"))
        t0, lon0, lat0, _ = arr[j - 1]
        t1, lon1, lat1, head1 = arr[j]
        span = t1 - t0
        frac = (t_s - t0) / span if span > 0.0 else 0.0
        lon = lon0 + frac * (lon1 - lon0)
        lat = lat0 + frac * (lat1 - lat0)
        return float(lon), float(lat), float(head1)

    def _a29_lonlat_at(
        self, config: str, a29_id: str, t_s: float
    ) -> tuple[float, float]:
        """Interpolated (lon, lat) for one A-29 at time ``t_s``.

        Position-only wrapper over :meth:`_a29_traj_at`; used by the A-29 marker
        scatter in :meth:`animate`. Replaces the former hardcoded parabolic-arc
        interpolation with a faithful lookup over the geometric replay cache.
        """
        lon, lat, _ = self._a29_traj_at(config, a29_id, t_s)
        return lon, lat

    def _a29_hud_fields(
        self, config: str, a29_id: str, t_s: float
    ) -> tuple[str, int, float, str | None, tuple[float, float]]:
        """HUD inputs for one A-29 at ``t_s``.

        Returns ``(state, rockets_remaining, fuel_pct, target_uas_id,
        marker_lonlat)`` reconstructed via the pure ``viz/hud.py`` helpers from
        the precomputed caches and the marker position. The caller gates the
        actual draw on :func:`should_render_hud`.
        """
        events = self._a29_info[config][a29_id].events
        state = a29_state_at(events, t_s)
        rockets = rockets_remaining_at(
            self._a29_shot_times[config][a29_id], t_s,
            self._cfg.rocket_budget_per_sortie,
            self._a29_refuel_times[config][a29_id],
        )
        anchor = fuel_anchor_s(events, t_s, fallback_s=self._cfg.t_pea_start_s)
        fuel_pct = fuel_fraction(
            t_s, anchor, self._cfg.a29_endurance_s, is_in_flight(state)
        )
        target = active_engagement_at(self._a29_eng_windows[config][a29_id], t_s)
        marker = self._a29_lonlat_at(config, a29_id, t_s)
        return state, rockets, fuel_pct, target, marker

    def _uas_frame(
        self, config: str, t_s: float
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """(lons, lats, archetype_names, heading_rads) for live UAS at t_s, for one
        config panel (a UAS killed in ``config`` disappears at its t_kill)."""
        _empty = np.array([])
        if self._spawns.empty:
            return _empty, _empty, [], _empty
        df = self._spawns[self._spawns["spawn_time_s"] <= t_s].copy()
        if df.empty:
            return _empty, _empty, [], _empty

        # Remove UAS neutralised in this config
        kills = self._kill_times.get(config, {})
        dead_ids = {uid for uid, kt in kills.items() if kt <= t_s}
        df = df[~df["uas_id"].isin(dead_ids)]
        if df.empty:
            return _empty, _empty, [], _empty

        dt = t_s - df["spawn_time_s"].values
        lat_rad = np.radians(df["lat"].values)
        speed = df["speed_mps"].values
        heading = df["heading_rad"].values

        # Equirectangular displacement (same approximation as projection.py)
        dx = speed * np.cos(heading) * dt
        dy = speed * np.sin(heading) * dt
        cur_lons = df["lon"].values + dx / (_R_M * np.cos(lat_rad)) * (180.0 / math.pi)
        cur_lats = df["lat"].values + dy / _R_M * (180.0 / math.pi)

        return cur_lons, cur_lats, list(df["archetype_name"].values), heading

    # ── map drawing ───────────────────────────────────────────────────────

    def _draw_static_map(self, ax: plt.Axes) -> None:  # type: ignore[name-defined]  # noqa: F821
        """Draw fixed geographic elements on a subplot axes."""
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        _segs = [self._polygon.segment(i) for i in self._polygon.indices]

        # ZIDA continental band (DECEA eAIP ENR 5.2): 80 NM inward from the
        # land border. Drawn as ONE compound PathPatch (band ring + per-
        # segment cover trapezoids over sharp quinas), so the union is filled
        # once with uniform translucency — the figure-eight notch at the
        # CAP-6 quina is covered without darker overlap seams. Low zorder
        # keeps agents/markers clearly on top.
        _band = _zida_band_lonlat(self._cfg, self._polygon)
        _quads = _zida_quina_quads(self._cfg, self._polygon)

        def _mk(poly: list[tuple[float, float]]) -> Path:
            return Path(
                [*poly, poly[0]],
                [Path.MOVETO] + [Path.LINETO] * (len(poly) - 1) + [Path.CLOSEPOLY],
            )

        # Align each cover quad's winding with the band ring's so the nonzero
        # fill rule yields the union (filled wedge at CAP-6 instead of cancel).
        _band_sign = 1.0 if _shoelace(_band) >= 0 else -1.0
        _sub_paths = [_mk(_band)]
        for _q in _quads:
            _q_oriented = (_q if ((1.0 if _shoelace(_q) >= 0 else -1.0)
                                  == _band_sign) else list(reversed(_q)))
            _sub_paths.append(_mk(_q_oriented))
        _band_patch = ax.add_patch(PathPatch(
            Path.make_compound_path(*_sub_paths),
            facecolor="#2ecc71", alpha=0.18, edgecolor="none", zorder=0.5,
        ))

        # Clip the band to the interior side of the border (everything east
        # of the dashed red frontier). Trims thin "rebarba" projecting west
        # into Colombia at sharp concave corners (e.g., "cabeça do cachorro"
        # near the N tripoint) without altering anything east of the border.
        _border_pts = ([(_segs[0].start.lon, _segs[0].start.lat)]
                       + [(s.end.lon, s.end.lat) for s in _segs])
        _N_lon, _N_lat = _border_pts[0]
        _S_lon, _S_lat = _border_pts[-1]
        _FAR_LON = _N_lon + 50.0   # well east of any visible map extent
        _FAR_N = _N_lat + 20.0
        _FAR_S = _S_lat - 20.0
        _clip_verts = [
            *_border_pts,
            (_S_lon, _FAR_S),
            (_FAR_LON, _FAR_S),
            (_FAR_LON, _FAR_N),
            (_N_lon, _FAR_N),
        ]
        _clip = Path(
            [*_clip_verts, _clip_verts[0]],
            [Path.MOVETO] + [Path.LINETO] * (len(_clip_verts) - 1)
            + [Path.CLOSEPOLY],
        )
        _band_patch.set_clip_path(_clip, transform=ax.transData)

        for seg in _segs:
            ax.plot(
                [seg.start.lon, seg.end.lon],
                [seg.start.lat, seg.end.lat],
                color="#CC2222", linewidth=1.1, linestyle="--", alpha=0.65,
            )

        base = self._cfg.main_base
        ax.plot(base.lon, base.lat, marker="*", color="#FFD700",
                markersize=13, zorder=8)
        ax.text(base.lon, base.lat - 0.25,
                base.name if base.name else "Base",
                color="#FFD700", fontsize=6.5, ha="center", va="top")

        for name, farp in self._cfg.farps.items():
            ax.plot(farp.lon, farp.lat, marker="^", color="#00FFFF",
                    markersize=9, zorder=7)
            ax.text(farp.lon, farp.lat + 0.18, name,
                    color="#00FFFF", fontsize=6, ha="center", va="bottom")

        for name, cap in self._cfg.caps.items():
            ax.plot(cap.lon, cap.lat, marker="D", color="#44FF88",
                    markersize=7, zorder=7)
            ax.text(cap.lon, cap.lat + 0.18, name,
                    color="#44FF88", fontsize=6, ha="center", va="bottom")

        ax.set_xlabel("Longitude", color="#888888", fontsize=7)
        ax.set_ylabel("Latitude",  color="#888888", fontsize=7)
        ax.tick_params(colors="#666666", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(alpha=0.12, color="white", linewidth=0.3)

        # Auto-fit extent
        all_lons = (
            [base.lon]
            + [f.lon for f in self._cfg.farps.values()]
            + [c.lon for c in self._cfg.caps.values()]
            + [s.start.lon for s in _segs]
            + [s.end.lon   for s in _segs]
            + [p[0] for p in _band]
            + [v[0] for q in _quads for v in q]
        )
        all_lats = (
            [base.lat]
            + [f.lat for f in self._cfg.farps.values()]
            + [c.lat for c in self._cfg.caps.values()]
            + [s.start.lat for s in _segs]
            + [s.end.lat   for s in _segs]
            + [p[1] for p in _band]
            + [v[1] for q in _quads for v in q]
        )
        mlon = (max(all_lons) - min(all_lons)) * 0.06
        mlat = (max(all_lats) - min(all_lats)) * 0.06
        ax.set_xlim(min(all_lons) - mlon, max(all_lons) + mlon)
        ax.set_ylim(min(all_lats) - mlat, max(all_lats) + mlat)

    # ── animation ─────────────────────────────────────────────────────────

    def animate(self, fps: int = 20, speed_factor: float = 60.0) -> None:
        """Launch interactive side-by-side animation window.

        Parameters
        ----------
        fps
            Target frames per second (smoothness only — playback speed does
            not depend on it).
        speed_factor
            Simulated seconds that advance per real second of animation.
            Default 60 → 1 real second = 1 simulated minute. Honoured against
            the wall clock, so it holds even when frames render slower than
            ``fps``. The Speed slider spans 10–3600 on a log scale.
        """
        import time

        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.artist import Artist
        from matplotlib.axes import Axes
        from matplotlib.axis import Axis
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import Collection, PathCollection
        from matplotlib.patches import Patch
        from matplotlib.quiver import Quiver
        from matplotlib.transforms import Bbox
        from matplotlib.widgets import Button, Slider

        # plots.py sets Agg for headless CI — switch if needed
        if matplotlib.get_backend().lower() == "agg":
            try:  # noqa: SIM105 — best-effort interactive backend switch
                matplotlib.use("TkAgg")
            except Exception:
                pass

        fig, (ax_b, ax_m) = plt.subplots(
            1, 2, figsize=(16, 7), gridspec_kw={"wspace": 0.06}
        )
        fig.patch.set_facecolor("#0d0d1a")

        for ax, title, subtitle in [
            (ax_b, "Baseline A-29", "unguided SBAT-70 and naked eye"),
            (ax_m, "Modernized A-29N",
             "C-UAS digital avionics system: APKWS laser guidance kit, "
             "FLIR Star SAFIRE II and datalink"),
        ]:
            ax.set_facecolor("#0d1b2a")
            ax.set_title(title, color="white", fontsize=11, pad=20)
            ax.text(
                0.5, 1.012, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", color="#9FB3C8", fontsize=7.5,
            )
            self._draw_static_map(ax)

        # Static spawn-location markers — small × on the border for every UAS.
        # These never move: spawn lat/lon is on the polygon by construction.
        if not self._spawns.empty:
            for ax in (ax_b, ax_m):
                ax.scatter(
                    self._spawns["lon"].values,
                    self._spawns["lat"].values,
                    s=8, marker="x", color="#FF8C00", alpha=0.35,
                    linewidths=0.6, zorder=6,
                )

        # Mutable quiver references — replaced every frame in update().
        self._quiv_b = None    # UAS heading quiver for baseline
        self._quiv_m = None    # UAS heading quiver for modernized
        self._aquiv_b = None   # A-29 heading quiver for baseline
        self._aquiv_m = None   # A-29 heading quiver for modernized
        self._fscat_b = None   # fire-event scatter for baseline
        self._fscat_m = None   # fire-event scatter for modernized
        self._mscat_b = None   # missile-in-flight scatter for baseline
        self._mscat_m = None   # missile-in-flight scatter for modernized
        self._hud_b = None     # A-29 HUD annotations for baseline
        self._hud_m = None     # A-29 HUD annotations for modernized

        a29_ids = sorted(self._cfg.a29_schedules.keys())

        # Every per-frame artist is flagged ``animated`` so a full redraw
        # (window shown, resized, slider moved) paints only the static scene,
        # which is cached; each frame then restores that background and draws
        # just these artists (blitting). Any new per-frame artist MUST be
        # created with ``animated=use_blit`` or it gets baked into the cached
        # background and leaves a ghost trail.
        use_blit = bool(fig.canvas.supports_blit)

        scat_b  = ax_b.scatter([], [], s=75, zorder=10, marker=">", animated=use_blit)
        scat_m  = ax_m.scatter([], [], s=75, zorder=10, marker=">", animated=use_blit)
        uscat_b = ax_b.scatter([], [], s=20, zorder=9,  marker="o", animated=use_blit)
        uscat_m = ax_m.scatter([], [], s=20, zorder=9,  marker="o", animated=use_blit)

        # The clock sits in its own invisible axes rather than on the figure:
        # savefig drops figure-level animated artists, axes-level ones it keeps.
        ax_time = fig.add_axes((0.72, 0.0, 0.27, 0.06))
        ax_time.set_axis_off()
        ax_time.set_navigate(False)
        time_text = ax_time.text(
            0.72, 0.027, "T+000 min  (0 s)", transform=fig.transFigure,
            ha="left", color="white", fontsize=10, fontweight="bold",
            clip_on=False, animated=use_blit,
        )

        _paused: list[bool] = [False]
        _speed: list[float] = [speed_factor]
        _sim_time: list[float] = [0.0]

        def _state_colors(config: str, t_s: float) -> list[str]:
            # Same derivation as hud.a29_state_at (single source of truth).
            return [
                _STATE_COLORS.get(
                    a29_state_at(self._a29_info[config][a29_id].events, t_s), "#FFFFFF"
                )
                for a29_id in a29_ids
            ]

        def update(step_s: float) -> None:
            # step_s = simulated seconds advanced since the previous frame.
            t_s = _sim_time[0]

            for config, scat, aquiv_attr, ax in (
                ("baseline", scat_b, "_aquiv_b", ax_b),
                ("modernized", scat_m, "_aquiv_m", ax_m),
            ):
                triples = [self._a29_traj_at(config, aid, t_s) for aid in a29_ids]
                a_lons = [p[0] for p in triples]
                a_lats = [p[1] for p in triples]
                a_head = [p[2] for p in triples]
                scat.set_offsets(np.column_stack([a_lons, a_lats]))
                scat.set_color(_state_colors(config, t_s))

                # A-29 heading quiver (light cyan), driven by
                # the cached trajectory and distinct from the yellow UAS quiver.
                prev_aq = getattr(self, aquiv_attr)
                if prev_aq is not None:
                    prev_aq.remove()
                if a_lons:
                    aq = ax.quiver(
                        a_lons, a_lats,
                        np.cos(a_head), np.sin(a_head),
                        scale=28, width=0.0024, headwidth=4,
                        color="#66FFFF", alpha=0.75, zorder=12,
                        animated=use_blit,
                    )
                    setattr(self, aquiv_attr, aq)
                else:
                    setattr(self, aquiv_attr, None)

            # UAS markers + heading arrows, per config (a UAS killed in one
            # variant is removed from that panel only, at its own t_kill).
            for config, uscat, quiv_attr, ax in (
                ("baseline", uscat_b, "_quiv_b", ax_b),
                ("modernized", uscat_m, "_quiv_m", ax_m),
            ):
                u_lons, u_lats, archetypes, headings = self._uas_frame(config, t_s)
                if len(u_lons) == 0:
                    uscat.set_offsets(np.empty((0, 2)))
                    uscat.set_color([])
                else:
                    uscat.set_offsets(np.column_stack([u_lons, u_lats]))
                    uscat.set_color(
                        [_UAS_COLORS.get(a, "#FFFFFF") for a in archetypes]
                    )

                prev = getattr(self, quiv_attr)
                if prev is not None:
                    prev.remove()
                if len(u_lons) > 0:
                    q = ax.quiver(
                        u_lons, u_lats,
                        np.cos(headings), np.sin(headings),
                        scale=30, width=0.002, headwidth=4,
                        color="#FFFF66", alpha=0.65, zorder=11,
                        animated=use_blit,
                    )
                    setattr(self, quiv_attr, q)
                else:
                    setattr(self, quiv_attr, None)

            # Fire / kill event markers. A flash lasts 45 sim-s, or one frame
            # when a frame spans more than that (high speed), so no hit or miss
            # can fall between two frames unseen.
            _FIRE_WINDOW_S = 45.0
            fire_window = max(_FIRE_WINDOW_S, step_s)
            for fscat_attr, ax, cfg_name in (
                ("_fscat_b", ax_b, "baseline"),
                ("_fscat_m", ax_m, "modernized"),
            ):
                prev = getattr(self, fscat_attr)
                if prev is not None:
                    for a in prev:
                        a.remove()
                events = self._fire_events.get(cfg_name, [])
                flash_neut = [(lo, la) for lo, la, tf, neut in events if neut and 0 <= t_s - tf <= fire_window]
                flash_miss = [(lo, la) for lo, la, tf, neut in events if not neut and 0 <= t_s - tf <= fire_window]
                perm_kill  = [(lo, la) for lo, la, tf, neut in events if neut and t_s - tf > fire_window]
                artists = []
                if flash_neut:
                    lons_f, lats_f = zip(*flash_neut, strict=True)
                    artists.append(ax.scatter(lons_f, lats_f, s=120, marker="*",
                                              color="#00FF88", alpha=0.9, zorder=14, linewidths=0,
                                              animated=use_blit))
                if flash_miss:
                    lons_f, lats_f = zip(*flash_miss, strict=True)
                    artists.append(ax.scatter(lons_f, lats_f, s=80, marker="x",
                                              color="#FF4444", alpha=0.85, zorder=14, linewidths=1.2,
                                              animated=use_blit))
                if perm_kill:
                    lons_f, lats_f = zip(*perm_kill, strict=True)
                    artists.append(ax.scatter(lons_f, lats_f, s=30, marker="x",
                                              color="#88FF88", alpha=0.55, zorder=8, linewidths=0.8,
                                              animated=use_blit))
                setattr(self, fscat_attr, artists if artists else None)

            # Missiles in flight. Small orange marker
            # interpolated shooter→target over [t_fire, t_impact); it vanishes
            # at impact, where the neutralisation/miss flash above takes over.
            for mscat_attr, ax, cfg_name in (
                ("_mscat_b", ax_b, "baseline"),
                ("_mscat_m", ax_m, "modernized"),
            ):
                prev = getattr(self, mscat_attr)
                if prev is not None:
                    prev.remove()
                live: list[tuple[float, float]] = []
                for ev in self._missile_events.get(cfg_name, []):
                    span = ev.t_impact_s - ev.t_fire_s
                    if span <= 0.0 or not (ev.t_fire_s <= t_s < ev.t_impact_s):
                        continue
                    frac = (t_s - ev.t_fire_s) / span
                    lon = ev.shooter_xy[0] + frac * (ev.target_xy[0] - ev.shooter_xy[0])
                    lat = ev.shooter_xy[1] + frac * (ev.target_xy[1] - ev.shooter_xy[1])
                    live.append((lon, lat))
                if live:
                    offs = np.array(live, dtype=float)
                    setattr(self, mscat_attr, ax.scatter(
                        offs[:, 0], offs[:, 1], s=10, marker=".",
                        color="#FF8800", alpha=0.95, zorder=13, linewidths=0,
                        animated=use_blit,
                    ))
                else:
                    setattr(self, mscat_attr, None)

            # per-A-29 HUD textboxes (state/rockets/fuel/
            # target). Rendered only for active A-29s (not AT_BASE/MISSION_COMPLETE)
            # at low opacity so the map stays legible with many simultaneous
            # engagements. Same per-frame remove/redraw pattern as the layers above.
            for hud_attr, ax, cfg_name in (
                ("_hud_b", ax_b, "baseline"),
                ("_hud_m", ax_m, "modernized"),
            ):
                prev = getattr(self, hud_attr)
                if prev is not None:
                    for ann in prev:
                        ann.remove()
                annotations = []
                for aid in a29_ids:
                    state, rockets, fuel_pct, target, marker = self._a29_hud_fields(
                        cfg_name, aid, t_s
                    )
                    if not should_render_hud(state):
                        continue
                    hud = render_a29_hud(
                        ax=ax, a29_id=aid, state=state, rockets_remaining=rockets,
                        fuel_pct=fuel_pct, target_uas_id=target, marker_lonlat=marker,
                        accent_color=_STATE_COLORS.get(state, "#66FFFF"),
                    )
                    hud.set_animated(use_blit)
                    annotations.append(hud)
                setattr(self, hud_attr, annotations if annotations else None)

            time_text.set_text(f"T+{int(t_s / 60):03d} min  ({t_s:.0f} s)")

        from matplotlib.lines import Line2D

        # Three framed, identically-styled legend boxes (STATE / CONTACTS /
        # ENGAGEMENT) sit in the reserved bottom strip and read as one cohesive
        # legend bar. fig.legend appends to fig.legends, so the three coexist.
        # A-29 STATE shows one swatch per state that holds the screen, named by
        # the same short label the HUD prints next to the marker. Legends fill
        # column by column, so two rows read as the mission timeline left to
        # right (two rows is all the strip under the maps can hold).
        status_handles = [
            Patch(color=colour, label=state_short_label(state))
            for state, colour in _STATE_LEGEND
        ]
        contact_handles = [
            Line2D([0], [0], marker="o", color="#FF8C00", markersize=7,
                   linestyle="None", label="UAS multi-rotor"),
            Line2D([0], [0], marker="o", color="#FF2222", markersize=7,
                   linestyle="None", label="UAS fixed-wing"),
            Line2D([0], [0], marker="x", color="#FF8C00", markersize=7,
                   alpha=0.5, linestyle="None", label="Spawn point"),
        ]
        engage_handles = [
            Line2D([0], [0], marker=".", color="#FF8800", markersize=10,
                   linestyle="None", label="Rocket"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#00FF88",
                   markersize=11, linestyle="None", label="Neutralized"),
            Line2D([0], [0], marker="x", color="#FF4444", markersize=8,
                   linestyle="None", label="Miss"),
            Line2D([0], [0], marker="x", color="#88FF88", markersize=7,
                   linestyle="None", label="Destroyed"),
        ]

        # The three boxes are anchored by their lower-left corner and clustered
        # as one centred group with uniform gaps just above the control bar, so
        # they read as a single cohesive HUD legend — wargame style. The x
        # anchors come from the boxes' measured widths, so relabelling a box
        # never needs hand-retuned coordinates. _LEGEND_Y sits clear of the
        # playback controls (y≤0.065).
        _LEGEND_Y = 0.080
        _LEGEND_GAP = 0.024  # figure fraction between neighbouring boxes

        def _section(
            handles: Sequence[Patch | Line2D], title: str, ncol: int,
        ) -> Any:
            leg = fig.legend(
                handles=handles, loc="lower left", bbox_to_anchor=(0.0, _LEGEND_Y),
                ncol=ncol, title=title,
                frameon=True, fancybox=True, framealpha=0.9,
                facecolor="#13243a", edgecolor="#33506e",
                labelcolor="white", fontsize=7,
                borderpad=0.7, handletextpad=0.5, columnspacing=1.2,
            )
            ttl = leg.get_title()
            ttl.set_color("#9FB3C8")
            ttl.set_fontsize(7.5)
            ttl.set_fontweight("bold")
            return leg

        sections = [
            _section(status_handles,  "A-29 STATE", math.ceil(len(status_handles) / 2)),
            _section(contact_handles, "CONTACTS",   2),
            _section(engage_handles,  "ENGAGEMENT", 2),
        ]
        widths = [leg.get_window_extent().width / fig.bbox.width for leg in sections]
        x0 = 0.5 - (sum(widths) + _LEGEND_GAP * (len(sections) - 1)) / 2
        for leg, width in zip(sections, widths, strict=True):
            leg.set_bbox_to_anchor((x0, _LEGEND_Y), transform=fig.transFigure)
            x0 += width + _LEGEND_GAP

        # Reserve a clean bottom strip for the legend cluster + controls. We set
        # the subplot box explicitly instead of tight_layout(rect=…), which is a
        # no-op here (it bails on figure legends, leaving the default bottom=0.11
        # so the maps would otherwise overlap the legends).
        fig.subplots_adjust(top=0.90, bottom=0.215)

        # ── playback controls ─────────────────────────────────────────────
        ax_play  = fig.add_axes((0.22, 0.015, 0.040, 0.048))
        ax_pause = fig.add_axes((0.27, 0.015, 0.040, 0.048))
        ax_stop  = fig.add_axes((0.32, 0.015, 0.040, 0.048))
        ax_fs    = fig.add_axes((0.365, 0.015, 0.038, 0.048))
        ax_speed = fig.add_axes((0.47, 0.027, 0.200, 0.026))

        btn_play  = Button(ax_play,  "▶",  color="#1a2a3a", hovercolor="#2a4a6a")
        btn_pause = Button(ax_pause, "||", color="#1a2a3a", hovercolor="#2a4a6a")
        btn_stop  = Button(ax_stop,  "■",  color="#1a2a3a", hovercolor="#2a4a6a")
        btn_fs    = Button(ax_fs,    "",   color="#1a2a3a", hovercolor="#2a4a6a")

        # Fullscreen icon: four L-shaped corner brackets (no edge centres),
        # drawn in axes coords so it is font-independent.
        ax_fs.set_xlim(0, 1)
        ax_fs.set_ylim(0, 1)
        _arm, _gap = 0.30, 0.22
        for _x0, _y0, _sx, _sy in (
            (_gap, _gap, 1, 1),
            (1 - _gap, _gap, -1, 1),
            (_gap, 1 - _gap, 1, -1),
            (1 - _gap, 1 - _gap, -1, -1),
        ):
            ax_fs.plot(
                [_x0 + _sx * _arm, _x0, _x0],
                [_y0, _y0, _y0 + _sy * _arm],
                color="white", lw=1.4, solid_capstyle="round", zorder=5,
            )

        # Log-scale speed, 10×–3600×: the everyday 10–600 range keeps most of
        # the bar's travel and the top end reaches 1 simulated hour per real
        # second. The speed used and shown is rounded to 2 significant digits.
        _SPEED_MIN, _SPEED_MAX = 10.0, 3600.0
        sld_speed = Slider(
            ax_speed, "Speed ", math.log10(_SPEED_MIN), math.log10(_SPEED_MAX),
            valinit=math.log10(min(max(speed_factor, _SPEED_MIN), _SPEED_MAX)),
            color="#2a6a4a",
        )
        sld_speed.valtext.set_text(f"{speed_factor:.0f}")

        for btn in (btn_play, btn_pause, btn_stop):
            btn.label.set_color("white")
            btn.label.set_fontsize(13)
        sld_speed.label.set_color("white")
        sld_speed.valtext.set_color("#88FF88")

        def on_play(event: object) -> None:
            if _sim_time[0] >= self._t_end:
                _sim_time[0] = 0.0
            _paused[0] = False

        def on_pause(event: object) -> None:
            _paused[0] = True

        def on_stop(event: object) -> None:
            _sim_time[0] = 0.0
            _paused[0] = True

        def on_speed(val: float) -> None:
            speed = 10.0 ** val
            digit = 10.0 ** (math.floor(math.log10(speed)) - 1)
            _speed[0] = round(speed / digit) * digit
            sld_speed.valtext.set_text(f"{_speed[0]:.0f}")

        def on_fullscreen(event: object) -> None:
            mgr = getattr(fig.canvas, "manager", None)
            if mgr is not None:
                with contextlib.suppress(Exception):
                    mgr.full_screen_toggle()

        btn_play.on_clicked(on_play)
        btn_pause.on_clicked(on_pause)
        btn_stop.on_clicked(on_stop)
        btn_fs.on_clicked(on_fullscreen)
        sld_speed.on_changed(on_speed)

        # ── frame loop: wall-clock playback + blitting ────────────────────
        # Sim time advances by speed × real elapsed time, so the Speed value
        # holds even when a frame renders slower than 1/fps. The real time one
        # frame may claim is capped, so a stall (window drag, OS hiccup) cannot
        # fling the replay forward.
        _MAX_FRAME_DT_S = 0.25
        frame_s = 1.0 / fps
        canvas: Any = fig.canvas  # copy_from_bbox/restore_region: blit backends only
        widget_axes = (ax_play, ax_pause, ax_stop, ax_fs, ax_speed)
        _wall: list[float | None] = [None]
        _bg: list[tuple[Any, Any] | None] = [None]
        _bg_under_m: list[Any] = [None]  # static scene minus the right panel
        _legend_bgs: list[Any] = []
        _layout: dict[str, Any] = {}
        _layout_sig: list[tuple[Any, ...] | None] = [None]
        _exact_drawing: list[bool] = [False]
        _force_full: list[bool] = [False]
        timer = canvas.new_timer(interval=max(1, int(1000.0 * frame_s)))

        def _dynamic(ax: Axes) -> list[Artist]:
            # Animated artists in the order a full redraw paints them: a stable
            # zorder sort over get_children(), exactly as Axes.draw does.
            return sorted(
                (a for a in ax.get_children() if a.get_animated() and a.get_visible()),
                key=lambda a: a.get_zorder(),
            )

        def _region(x0: float, y0: float, x1: float, y1: float) -> Bbox:
            # Whole-pixel box clipped to the canvas (copy/blit work on pixels).
            w, h = fig.bbox.width, fig.bbox.height
            return Bbox.from_extents(
                max(0.0, math.floor(x0)), max(0.0, math.floor(y0)),
                min(w, math.ceil(x1)), min(h, math.ceil(y1)),
            )

        def _inside(inner: Bbox, outer: Bbox) -> bool:
            return (inner.x0 >= outer.x0 and inner.y0 >= outer.y0
                    and inner.x1 <= outer.x1 and inner.y1 <= outer.y1)

        def _static_boxes(ax: Axes, r: Any) -> list[Bbox]:
            # Everything static a full redraw paints for ``ax``: its face,
            # frame and tick marks (padded box), then each label/artist alone
            # (+2 px for antialiased edges).
            boxes = [ax.bbox.padded(8.0 * fig.dpi / 72.0)]
            for c in ax.get_children():
                if c is ax.patch or c.get_animated() or not c.get_visible():
                    continue
                if isinstance(c, Axis):
                    texts = [*c.get_ticklabels(), c.label, c.offsetText]
                    boxes += [t.get_window_extent(r) for t in texts
                              if t.get_visible() and t.get_text()]
                elif (bb := c.get_tightbbox(r)) is not None:
                    boxes.append(bb)
            return [bb.padded(2) for bb in boxes if bb.width > 0 and bb.height > 0]

        def _cache_scene() -> None:
            # Called right after a full redraw painted the static scene: cache
            # the blit regions and work out which layers can meet which.
            r = canvas.get_renderer()
            w, h = fig.bbox.width, fig.bbox.height
            t_ext = time_text.get_window_extent(r)
            time_reg = _region(t_ext.x0 - 4, 0, w, t_ext.y1 + 4)
            widgets = [bb for wax in widget_axes if (bb := wax.get_tightbbox(r)) is not None]
            # Everything above the control row (legends included) is restored
            # each frame; the controls are left alone so button hover persists.
            y_cut = max([bb.y1 for bb in widgets] + [time_reg.y1])
            maps_reg = _region(0, y_cut + 1, w, h)
            legends = [leg.get_window_extent(r).padded(2) for leg in fig.legends]
            later_m = _static_boxes(ax_m, r)
            pb, pm = ax_b.bbox.padded(2), ax_m.bbox.padded(2)
            _layout.update(
                maps=maps_reg, time=time_reg, legends=legends,
                # Static layers a full redraw paints after the left panel's
                # artists: the whole right panel ...
                later_m=later_m,
                # ... and, per panel, the few such boxes reaching INTO it (its
                # markers/arrows are clipped to it, so only those can meet them).
                coll_b=[(bb, "spill") for bb in later_m if bb.overlaps(pb)]
                + [(bb, "full") for bb in legends if bb.overlaps(pb)],
                coll_m=[(bb, "full") for bb in legends if bb.overlaps(pm)],
                panels_inside=_inside(pb, maps_reg) and _inside(pm, maps_reg),
                time_under_legend=any(time_reg.overlaps(bb) for bb in legends),
                # The right panel can be repainted on its own only where all of
                # it lies in the restored region (and on an Agg canvas).
                partial_ok=(isinstance(canvas, FigureCanvasAgg)
                            and all(_inside(bb, maps_reg) for bb in later_m)),
            )
            _bg[0] = (canvas.copy_from_bbox(maps_reg), canvas.copy_from_bbox(time_reg))
            # Legends as a full redraw leaves them (they are painted last).
            _legend_bgs[:] = [canvas.copy_from_bbox(_region(bb.x0, bb.y0, bb.x1, bb.y1))
                              for bb in legends]
            # The scene-minus-right-panel only changes with size or view (a
            # slider drag redraws too, but must not force a re-render of it).
            sig = (w, h, fig.dpi, ax_b.bbox.bounds, ax_m.bbox.bounds,
                   tuple(ax_b.viewLim.bounds), tuple(ax_m.viewLim.bounds))
            if sig != _layout_sig[0]:
                _layout_sig[0] = sig
                _bg_under_m[0] = None

        def _capture_under_m() -> Any:
            # The static scene WITHOUT the right panel = the canvas state a full
            # redraw has just before painting that panel. Rendered off-screen
            # (Agg only, no GUI blit), so the missing panel is never shown.
            _exact_drawing[0] = True
            ax_m.set_visible(False)
            try:
                FigureCanvasAgg.draw(canvas)  # type: ignore[no-untyped-call]
                return canvas.copy_from_bbox(_layout["maps"])
            finally:
                ax_m.set_visible(True)
                _exact_drawing[0] = False

        def _marker_boxes(a: Collection, ax: Axes) -> np.ndarray:
            # One (x0, y0, x1, y1) box per marker/arrow, clipped to its axes as
            # the painting is. Arrow reach: 1.3 × axes width / scale (measured
            # ~1.0 at 100 and 125 dpi); marker reach: radius + edge.
            offs = np.asarray(a.get_offsets(), dtype=float).reshape(-1, 2)
            pts = a.get_offset_transform().transform(offs)
            if isinstance(a, Quiver):
                m = (1.3 * ax.bbox.width / a.scale if a.scale else 0.1 * ax.bbox.width) + 2.0
            elif isinstance(a, PathCollection):
                radius_pt = math.sqrt(float(np.max(a.get_sizes(), initial=0.0))) / 2.0
                edge_pt = float(np.max(a.get_linewidth(), initial=0.0))
                m = (radius_pt + edge_pt) * fig.dpi / 72.0 + 2.0
            else:
                m = 0.1 * ax.bbox.width
            c = ax.bbox.padded(1)
            boxes = np.column_stack([
                np.maximum(pts[:, 0] - m, c.x0), np.maximum(pts[:, 1] - m, c.y0),
                np.minimum(pts[:, 0] + m, c.x1), np.minimum(pts[:, 1] + m, c.y1),
            ])
            return boxes[(boxes[:, 0] <= boxes[:, 2]) & (boxes[:, 1] <= boxes[:, 3])]

        def _boxes_hit(boxes: np.ndarray, bb: Bbox) -> bool:
            return bool(np.any((boxes[:, 2] >= bb.x0) & (boxes[:, 0] <= bb.x1)
                               & (boxes[:, 3] >= bb.y0) & (boxes[:, 1] <= bb.y1)))

        def _frame_mode() -> str:
            # Blitting paints the dynamic artists over the WHOLE cached scene,
            # but a full redraw paints some static layers after them: the right
            # panel (opaque face, labels) after every left-panel artist, and the
            # bottom legends/controls after both panels. So each frame picks:
            #   "blit"  — no dynamic artist reaches a later layer;
            #   "spill" — a left-panel artist reaches the right panel (a HUD box
            #             near SBPV at take-off/landing): repaint that panel
            #             over it, exactly as a full redraw orders them;
            #   "full" / "stray" — anything else: redraw the figure in full
            #             ("stray": paint left the blit regions; refresh the
            #             cached background next frame).
            # Every mode yields exactly the pixels of a full redraw.
            r = canvas.get_renderer()
            maps = _layout["maps"]
            full = _layout["time_under_legend"]
            spill = False
            for ax, coll_checks in ((ax_b, _layout["coll_b"]), (ax_m, _layout["coll_m"])):
                for a in _dynamic(ax):
                    if isinstance(a, Collection):
                        if _layout["panels_inside"] and not coll_checks:
                            continue
                        boxes = _marker_boxes(a, ax)
                        if not _layout["panels_inside"] and not (
                            np.all(boxes[:, 0] >= maps.x0) and np.all(boxes[:, 1] >= maps.y0)
                            and np.all(boxes[:, 2] <= maps.x1) and np.all(boxes[:, 3] <= maps.y1)
                        ):
                            return "stray"
                        for bb, kind in coll_checks:
                            if _boxes_hit(boxes, bb):
                                full = full or kind == "full"
                                spill = spill or kind == "spill"
                        continue
                    bb = a.get_window_extent(r)
                    if bb.width <= 1 and bb.height <= 1:  # annotation clipped away: not drawn
                        continue
                    bb = bb.padded(8.0 * fig.dpi / 72.0)  # HUD box padding + edge
                    if not _inside(bb, maps):
                        return "stray"
                    full = full or any(bb.overlaps(L) for L in _layout["legends"])
                    if ax is ax_b and any(bb.overlaps(L) for L in _layout["later_m"]):
                        spill = True
            if full or (spill and not _layout["partial_ok"]):
                return "full"
            return "spill" if spill else "blit"

        def _paint(mode: str) -> None:
            # Paint the frame into the canvas buffer ("blit"/"spill"); the
            # caller updates the screen.
            assert _bg[0] is not None
            maps_bg, time_bg = _bg[0]
            if mode == "spill":
                if _bg_under_m[0] is None:
                    _bg_under_m[0] = _capture_under_m()
                canvas.restore_region(_bg_under_m[0])
            else:
                canvas.restore_region(maps_bg)
            canvas.restore_region(time_bg)
            for a in _dynamic(ax_b):
                ax_b.draw_artist(a)
            if mode == "spill":
                ax_m.draw(canvas.get_renderer())  # static only: dynamics are animated
            for a in _dynamic(ax_m):
                ax_m.draw_artist(a)
            if mode == "spill":
                # The legends go over the right panel's labels, as painted last.
                for leg_bg in _legend_bgs:
                    canvas.restore_region(leg_bg)
            for a in _dynamic(ax_time):
                ax_time.draw_artist(a)

        def _on_draw(event: object) -> None:
            # A full redraw (window shown or resized, slider moved, toolbar)
            # just painted the static scene: cache it, then complete this very
            # frame with the dynamic artists, in the exact order.
            if _exact_drawing[0]:
                return
            if canvas.is_saving():
                # savefig (e.g. toolbar Save) paints the animated artists too:
                # never cache that; re-cache from a fresh redraw next frame.
                _bg[0] = None
                return
            _cache_scene()
            mode = _frame_mode()
            # "full"/"stray" need a whole redraw, impossible from inside one:
            # paint in blit order now; the next frame (~1/fps later) redoes it.
            _paint(mode if mode == "spill" else "blit")

        def _compose() -> None:
            if _bg[0] is None or _force_full[0]:
                _force_full[0] = False
                canvas.draw_idle()  # full redraw; _on_draw re-caches the scene
                return
            mode = _frame_mode()
            if mode in ("full", "stray"):
                dyn = [a for ax in (ax_b, ax_m, ax_time) for a in _dynamic(ax)]
                _exact_drawing[0] = True
                try:
                    for a in dyn:
                        a.set_animated(False)
                    canvas.draw()
                finally:
                    for a in dyn:
                        a.set_animated(True)
                    _exact_drawing[0] = False
                # Paint outside the blit regions is not undone by the next
                # restore, so the next frame refreshes the whole background.
                _force_full[0] = mode == "stray"
                return
            _paint(mode)
            canvas.blit(_layout["maps"])
            canvas.blit(_layout["time"])

        def _tick() -> None:
            t_wall = time.perf_counter()
            dt_real = 0.0 if _wall[0] is None else min(t_wall - _wall[0], _MAX_FRAME_DT_S)
            _wall[0] = t_wall
            t_prev = _sim_time[0]
            if not _paused[0]:
                _sim_time[0] = min(_sim_time[0] + _speed[0] * dt_real, self._t_end)
                if _sim_time[0] >= self._t_end:
                    _paused[0] = True
            update(max(0.0, _sim_time[0] - t_prev))
            if use_blit:
                _compose()
            else:
                canvas.draw_idle()
            # Aim at one frame per 1/fps: wait only what this frame left over.
            timer.interval = max(1, int(1000.0 * (frame_s - (time.perf_counter() - t_wall))))

        def _on_resize(event: object) -> None:
            _bg[0] = None  # cached scene has the old size; re-cache on next redraw

        if use_blit:
            canvas.mpl_connect("draw_event", _on_draw)
            canvas.mpl_connect("resize_event", _on_resize)
        canvas.mpl_connect("close_event", lambda event: timer.stop())
        timer.add_callback(_tick)
        timer.start()

        plt.show()


def animate_run(
    run_dir: Path,
    config_path: Path,
    fps: int = 20,
    speed_factor: float = 60.0,
) -> None:
    """Entry point called by the CLI ``animate`` command."""
    MissionAnimator(run_dir, config_path).animate(fps=fps, speed_factor=speed_factor)


def plot_border_check(config_path: Path) -> None:
    """Static geographic check — border polygon + CAPs + FARPs, no run needed.

    Plots the border sectors (red dashed), sector indices, CAP positions (green
    diamonds), FARP positions (cyan triangles), and the two confirmed tripoints
    (white crosses) so the researcher can verify the polygon is correctly placed
    before running a full simulation.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() == "agg":
        try:  # noqa: SIM105 — best-effort interactive backend switch
            matplotlib.use("TkAgg")
        except Exception:
            pass

    cfg = MissionConfig.from_yaml(config_path)
    polygon = default_border_polygon(cfg)

    fig, ax = plt.subplots(figsize=(10, 14))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d1b2a")
    ax.set_title(
        "Border polygon check — setores 1–27 (N→S)\n"
        "Vermelho = segmentos do polígono  |  Verde = CAPs  |  Ciano = FARPs  |  × = tripontos",
        color="white", fontsize=9, pad=8,
    )

    _segs = [polygon.segment(i) for i in polygon.indices]

    # Sector lines + index labels
    for seg in _segs:
        ax.plot(
            [seg.start.lon, seg.end.lon],
            [seg.start.lat, seg.end.lat],
            color="#CC2222", linewidth=1.3, linestyle="--", alpha=0.8,
        )
        mid_lon = (seg.start.lon + seg.end.lon) / 2.0
        mid_lat = (seg.start.lat + seg.end.lat) / 2.0
        ax.text(
            mid_lon - 0.35, mid_lat,
            str(seg.index),
            color="#FF8888", fontsize=6.5, ha="right", va="center", fontweight="bold",
        )

    # Sector endpoint dots
    for seg in _segs:
        ax.plot(seg.start.lon, seg.start.lat, "o",
                color="#CC2222", markersize=3, alpha=0.7)
    # Add the final endpoint (last segment's end)
    last = _segs[-1]
    ax.plot(last.end.lon, last.end.lat, "o", color="#CC2222", markersize=3, alpha=0.7)

    # Two confirmed tripoints
    _tripoints = [
        (1.229, -66.853, "Triponto CO-VE-BR\n(Pedra do Cucuí)"),
        (-4.226, -69.951, "Triponto CO-PE-BR\n(Tabatinga)"),
    ]
    for lat, lon, label in _tripoints:
        ax.plot(lon, lat, "x", color="white", markersize=12, markeredgewidth=2, zorder=9)
        ax.text(lon + 0.25, lat, label, color="white", fontsize=6.5,
                va="center", bbox={"facecolor": "#000000", "alpha": 0.5, "pad": 2})

    # Main base
    base = cfg.main_base
    ax.plot(base.lon, base.lat, "*", color="#FFD700", markersize=14, zorder=8)
    ax.text(base.lon, base.lat - 0.30, base.name or "Base",
            color="#FFD700", fontsize=7, ha="center", va="top")

    # FARPs
    for name, farp in cfg.farps.items():
        ax.plot(farp.lon, farp.lat, "^", color="#00FFFF", markersize=9, zorder=7)
        ax.text(farp.lon, farp.lat + 0.20, name,
                color="#00FFFF", fontsize=6.5, ha="center", va="bottom")

    # CAPs
    for name, cap in cfg.caps.items():
        ax.plot(cap.lon, cap.lat, "D", color="#44FF88", markersize=8, zorder=7)
        ax.text(cap.lon, cap.lat + 0.20, name,
                color="#44FF88", fontsize=6.5, ha="center", va="bottom")

    ax.set_xlabel("Longitude (°E)", color="#AAAAAA", fontsize=8)
    ax.set_ylabel("Latitude (°N)", color="#AAAAAA", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(alpha=0.15, color="white", linewidth=0.4)

    all_lons = (
        [base.lon]
        + [f.lon for f in cfg.farps.values()]
        + [c.lon for c in cfg.caps.values()]
        + [s.start.lon for s in _segs] + [s.end.lon for s in _segs]
        + [-66.853, -69.951]
    )
    all_lats = (
        [base.lat]
        + [f.lat for f in cfg.farps.values()]
        + [c.lat for c in cfg.caps.values()]
        + [s.start.lat for s in _segs] + [s.end.lat for s in _segs]
        + [1.229, -4.226]
    )
    mlon = (max(all_lons) - min(all_lons)) * 0.04
    mlat = (max(all_lats) - min(all_lats)) * 0.04
    ax.set_xlim(min(all_lons) - mlon, max(all_lons) + mlon)
    ax.set_ylim(min(all_lats) - mlat, max(all_lats) + mlat)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
