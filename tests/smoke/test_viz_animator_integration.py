"""Integration smoke for the Spec 019 Phase 2 animator wiring.

Verifies: Spec 019 §3.1 FR-7, FR-8

Drives :class:`cuas_sim.viz.animate.MissionAnimator` against a minimal,
schema-correct run directory (spawns + engagements parquet only — ``_load_data``
tolerates absent event logs). Asserts that:

* FR-7 — ``_load_data`` pre-computes ``_a29_trajectory[config][a29_id]`` as a
  ``(N, 4)`` ``np.ndarray`` for every (config, A-29) pair.
* FR-8 — ``_a29_lonlat_at`` is a pure binary-lookup + linear interpolation over
  that cache: exact-waypoint queries reproduce the cached sample, midpoint
  queries land between neighbours, and out-of-range queries clamp to the ends.
* Frame loop (2026-10-15) — ``animate()`` caches the static scene and repaints
  only the per-frame artists over it (blitting): every frame must equal a plain
  full redraw pixel for pixel (take-off HUD spill and ``savefig`` included), and
  sim time must advance by speed × real elapsed time (fake clock).

This is a smoke (not a unit) test because it exercises the real parquet I/O and
the full ``MissionAnimator`` construction path. Matplotlib runs on the headless
``Agg`` backend; the animation window is never shown (``plt.show`` is stubbed),
so no GUI or SimPy is involved (Article VIII is about SimPy mocks — not
relevant here).

Red phase (Article III): FAILS until Phase 2 lands ``_a29_trajectory`` +
the rewritten ``_a29_lonlat_at(config, a29_id, t_s)``.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.text import Annotation, Text

from cuas_sim.config import MissionConfig
from cuas_sim.geometry.projection import XY, LocalProjection
from cuas_sim.policies.weapon import make_weapon_policy
from cuas_sim.states import A29State
from cuas_sim.viz.animate import MissionAnimator

# Headless backend, selected before any figure exists. ``animate()`` would
# switch Agg to TkAgg, so the frame-loop smokes below stub ``matplotlib.use``
# (and ``plt.show``): no GUI window is ever opened, locally or under CI.
matplotlib.use("Agg")

pytestmark = pytest.mark.viz

# Reference config that the CLI ``animate`` command loads. Anchored to the repo
# root so the test is cwd-independent.
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "zida_shield_default.yaml"

# Engagement timeline (seconds). A generous envelope-entry ceiling lets
# ``replay_pursuit`` stop on the geometric separation condition (head-on
# approach converges well inside the ceiling).
_T_SPAWN = 9_200.0
_T_PURSUIT = 9_300.0
_T_ENV_CEILING = _T_PURSUIT + 600.0
_T_KILL = _T_ENV_CEILING + 30.0


def _build_minimal_run_dir(run_dir: Path, cfg: MissionConfig) -> tuple[str, np.ndarray]:
    """Write spawns + engagements parquet for one neutralised baseline engagement.

    Returns ``(a29_id, cap_lonlat)`` — the engaged A-29 and its CAP centroid in
    (lon, lat) for downstream reasonableness checks.
    """
    proj = LocalProjection(cfg.aoo_centroid_lat, cfg.aoo_centroid_lon)
    a29_id = sorted(cfg.a29_schedules)[0]
    schedule = cfg.a29_schedules[a29_id]
    cap_geo = cfg.caps[f"CAP-{schedule.cap_index}"]
    cap_xy = proj.project(cap_geo)

    # UAS spawned 25 km north of the CAP, heading due south at 50 m/s — a
    # head-on closure that the faster A-29 resolves quickly.
    spawn_xy = XY(cap_xy.x, cap_xy.y + 25_000.0)
    spawn_geo = proj.unproject(spawn_xy)
    uas_heading = -math.pi / 2.0
    uas_speed = 50.0

    spawns_df = pd.DataFrame([{
        "uas_id": 1,
        "spawn_time_s": _T_SPAWN,
        "lat": spawn_geo.lat,
        "lon": spawn_geo.lon,
        "heading_rad": uas_heading,
        "speed_mps": uas_speed,
        "archetype_name": "fixed-wing",
        "sector_index": 22,
    }])
    spawns_df.to_parquet(run_dir / "spawns.parquet", index=False)

    env_max = make_weapon_policy(cfg, "baseline").envelope_max_m
    engagements_df = pd.DataFrame([{
        "config_name": "baseline",
        "a29_id": a29_id,
        "uas_id": 1,
        "t_pursuit_start_s": _T_PURSUIT,
        "t_envelope_entry_s": _T_ENV_CEILING,
        "t_fire_s": _T_ENV_CEILING + 1.0,
        "t_kill_s": _T_KILL,
        "firing_distance_m": 1_500.0,
        "mop3_envelope_max_m": env_max,
        "is_neutralized": True,
        "n_shots": 1,
        "feasible": True,
    }])
    engagements_df.to_parquet(run_dir / "engagements.parquet", index=False)

    return a29_id, np.array([cap_geo.lon, cap_geo.lat])


def test_load_data_precomputes_a29_trajectory(tmp_path: Path) -> None:
    """Smoke 1 — every (config, A-29) trajectory is cached as a (N, 4) array."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id, _ = _build_minimal_run_dir(tmp_path, cfg)

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)

    assert hasattr(animator, "_a29_trajectory"), (
        "FR-7: _load_data must populate the _a29_trajectory cache."
    )
    for config in ("baseline", "modernized"):
        assert config in animator._a29_trajectory
        for aid in cfg.a29_schedules:
            arr = animator._a29_trajectory[config][aid]
            assert isinstance(arr, np.ndarray), f"{config}/{aid} not an ndarray"
            assert arr.ndim == 2 and arr.shape[1] == 4, (
                f"{config}/{aid} must be (N, 4); got {arr.shape}"
            )
            assert arr.shape[0] >= 1
            # t_s column must be strictly non-decreasing for binary lookup.
            assert np.all(np.diff(arr[:, 0]) >= -1e-9), (
                f"{config}/{aid} trajectory not sorted by t_s"
            )

    # The engaged A-29 in baseline must have a real multi-phase trajectory.
    engaged = animator._a29_trajectory["baseline"][a29_id]
    assert engaged.shape[0] >= 4, (
        "Engaged A-29 trajectory should span anchor + PURSUING + ENGAGING + RTB."
    )


def test_a29_lonlat_at_reads_cache(tmp_path: Path) -> None:
    """Smoke 2 — ``_a29_lonlat_at`` is exact lookup + interp over the cache."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id, cap_lonlat = _build_minimal_run_dir(tmp_path, cfg)

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)
    arr = animator._a29_trajectory["baseline"][a29_id]
    assert arr.shape[0] >= 4

    # (a) Exact-waypoint queries reproduce the cached sample (proves the lookup
    #     uses the precomputed array, not the removed parabolic hack).
    for i in (0, arr.shape[0] // 2, arr.shape[0] - 1):
        t_i, lon_i, lat_i, _ = arr[i]
        lon, lat = animator._a29_lonlat_at("baseline", a29_id, float(t_i))
        assert lon == pytest.approx(lon_i, abs=1e-6), f"lon mismatch at wp {i}"
        assert lat == pytest.approx(lat_i, abs=1e-6), f"lat mismatch at wp {i}"

    # (b) Midpoint between the first two waypoints interpolates between them.
    t_mid = float((arr[0, 0] + arr[1, 0]) / 2.0)
    lon_m, lat_m = animator._a29_lonlat_at("baseline", a29_id, t_mid)
    lo_lon, hi_lon = sorted((arr[0, 1], arr[1, 1]))
    lo_lat, hi_lat = sorted((arr[0, 2], arr[1, 2]))
    assert lo_lon - 1e-6 <= lon_m <= hi_lon + 1e-6
    assert lo_lat - 1e-6 <= lat_m <= hi_lat + 1e-6

    # (c) Out-of-range queries clamp to the trajectory endpoints.
    lon_lo, lat_lo = animator._a29_lonlat_at("baseline", a29_id, float(arr[0, 0]) - 100.0)
    assert (lon_lo, lat_lo) == pytest.approx((arr[0, 1], arr[0, 2]), abs=1e-9)
    lon_hi, lat_hi = animator._a29_lonlat_at("baseline", a29_id, float(arr[-1, 0]) + 10_000.0)
    assert (lon_hi, lat_hi) == pytest.approx((arr[-1, 1], arr[-1, 2]), abs=1e-9)

    # (d) Reasonableness: every sampled position is finite and the aircraft
    #     actually manoeuvres away from the CAP during the engagement (the old
    #     parabolic arc returned to the CAP; real pursuit does not).
    sampled = [
        animator._a29_lonlat_at("baseline", a29_id, t)
        for t in (5_000.0, _T_PURSUIT, _T_KILL - 1.0, _T_KILL)
    ]
    for lon, lat in sampled:
        assert math.isfinite(lon) and math.isfinite(lat)
        assert -90.0 <= lon <= 0.0 and -30.0 <= lat <= 15.0, "position off-map"

    # Pre-mission hold sits at the CAP; mid-engagement is displaced from it.
    hold_lon, hold_lat = animator._a29_lonlat_at("baseline", a29_id, 5_000.0)
    assert (hold_lon, hold_lat) == pytest.approx(tuple(cap_lonlat), abs=1e-6)
    kill_lon, kill_lat = animator._a29_lonlat_at("baseline", a29_id, _T_KILL)
    assert math.hypot(kill_lon - cap_lonlat[0], kill_lat - cap_lonlat[1]) > 1e-3, (
        "A-29 should be displaced from the CAP at kill time (real pursuit)."
    )


# ── frame loop: blitting fidelity + wall-clock speed ─────────────────────────


class _FakeClock:
    """Stand-in for ``time.perf_counter``: advances only when the test says."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _start_playback(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path, speed: float,
) -> tuple[Figure, Callable[[], None], _FakeClock]:
    """Run ``animate()`` headless; return its figure, frame callback and clock."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    monkeypatch.setattr(matplotlib, "use", lambda *a, **k: None)  # stay on Agg
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    timers: list[Any] = []
    real_new_timer = FigureCanvasBase.new_timer

    def _recording_new_timer(self: FigureCanvasBase, *args: Any, **kwargs: Any) -> Any:
        timers.append(real_new_timer(self, *args, **kwargs))
        return timers[-1]

    monkeypatch.setattr(FigureCanvasBase, "new_timer", _recording_new_timer)
    MissionAnimator(run_dir, _CONFIG_PATH).animate(fps=20, speed_factor=speed)
    fig = plt.gcf()
    ((frame_callback, _, _),) = timers[-1].callbacks
    fig.canvas.draw()  # the window appears: its first full redraw is cached
    return fig, frame_callback, clock


def _full_redraw_pixels(fig: Figure) -> np.ndarray:
    """Reference: a plain full redraw with every artist painted in place."""
    animated = [a for a in fig.findobj() if a.get_animated()]
    for a in animated:
        a.set_animated(False)
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
    for a in animated:
        a.set_animated(True)
    fig.canvas.draw()  # re-cache a clean scene for the next blitted frame
    return pixels


def _write_take_off_events(run_dir: Path, cfg: MissionConfig, a29_id: str) -> None:
    """SBPV → FARP → CAP legs for the engaged A-29, plus a late transition of a
    second A-29 so the replay runs past the scripted kill."""
    other = sorted(cfg.a29_schedules)[1]
    rows = [
        (60.0, a29_id, A29State.AT_BASE, A29State.TRANSIT_TO_FARP),
        (4_000.0, a29_id, A29State.TRANSIT_TO_FARP, A29State.REPOSITIONING_TO_CAP),
        (6_000.0, a29_id, A29State.REPOSITIONING_TO_CAP, A29State.AT_CAP_PEA),
        (12_000.0, other, A29State.AT_BASE, A29State.TRANSIT_TO_FARP),
    ]
    with gzip.open(run_dir / "events_baseline.jsonl.gz", "wt", encoding="utf-8") as fh:
        for t_s, aid, old, new in rows:
            fh.write(json.dumps({"t_s": t_s, "a29": aid, "old": old.value, "new": new.value}))
            fh.write("\n")


def _shown_time_s(fig: Figure) -> int:
    clock_text = next(t.get_text() for t in fig.findobj(Text) if t.get_text().startswith("T+"))
    return int(clock_text.rsplit("(", 1)[1].split()[0])


def test_blitted_frames_match_a_full_redraw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke 3 — every frame the loop paints equals a plain full redraw.

    Walks take-off (a baseline HUD box spilling under the modernized panel),
    pursuit, the kill flash and the permanent kill mark, then ``savefig``
    (which must keep the clock and every per-frame artist). A per-frame artist
    created without ``animated=...`` would be baked into the cached scene and
    leave a ghost: this comparison catches it.
    """
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id, _ = _build_minimal_run_dir(tmp_path, cfg)
    _write_take_off_events(tmp_path, cfg, a29_id)
    fig, frame, clock = _start_playback(monkeypatch, tmp_path, speed=3_600.0)
    ax_b = fig.axes[0]
    spilled = False
    try:
        # 0.05 s frames (180 sim-s) through take-off, then 0.25 s (900 sim-s).
        for dt in [0.0] + [0.05] * 3 + [0.25] * 13:
            clock.now += dt
            frame()
            painted = np.asarray(fig.canvas.buffer_rgba()).copy()
            r = fig.canvas.get_renderer()
            spilled |= any(
                isinstance(a, Annotation) and a.get_window_extent(r).x1 > ax_b.bbox.x1
                for a in ax_b.texts
            )
            np.testing.assert_array_equal(painted, _full_redraw_pixels(fig))
        buf = io.BytesIO()
        fig.savefig(buf, format="rgba", dpi=fig.dpi)
        saved = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(painted.shape)
        np.testing.assert_array_equal(saved, _full_redraw_pixels(fig))
    finally:
        plt.close(fig)
    assert spilled, "take-off frames should push a baseline HUD box past its panel edge"


def test_playback_speed_follows_the_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke 4 — sim time = speed × real elapsed time, whatever each frame took
    to render; a stall counts as at most 0.25 s of playback."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    _build_minimal_run_dir(tmp_path, cfg)
    fig, frame, clock = _start_playback(monkeypatch, tmp_path, speed=60.0)
    try:
        frame()
        assert _shown_time_s(fig) == 0
        for dt in (0.05, 0.20, 0.10):  # uneven frame times, 0.35 s in all
            clock.now += dt
            frame()
        assert _shown_time_s(fig) == 21  # 60 × 0.35
        clock.now += 5.0  # a 5 s stall (window drag, OS hiccup) ...
        frame()
        assert _shown_time_s(fig) == 36  # ... adds only 60 × 0.25
    finally:
        plt.close(fig)
