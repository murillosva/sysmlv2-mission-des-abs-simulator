"""Pure-function oracle tests for ``cuas_sim.viz.hud`` (Spec 019 Phase 4).

Verifies: Spec 019 §3.1 FR-11, FR-12

The HUD (FR-11/FR-12) is a SHOULD with no numbered T-oracle in §5.1, so these
tests are the oracle for its pure helpers — the only part with deterministic,
matplotlib-free answers. The render function itself (which needs an Axes) is
exercised by ``tests/smoke/test_viz_hud_integration.py``.

Group (Red phase): all tests below must FAIL until ``src/cuas_sim/viz/hud.py``
lands. Article III (Test-Oracle First).

Marker ``viz`` keeps these out of the default smoke selection — run with
``pytest tests/unit/test_viz_hud.py -v -m viz``.
"""

from __future__ import annotations

import pytest

from cuas_sim.states import A29State

pytestmark = pytest.mark.viz


# ─── fuel_fraction (FR-11) ─────────────────────────────────────────────────────


def test_fuel_fraction_full_on_ground() -> None:
    """At FARP/Base (in_flight=False) fuel reads 100 % regardless of clock."""
    from cuas_sim.viz.hud import fuel_fraction

    # Even with a long-stale departure clock, on the ground = full tank.
    assert fuel_fraction(50_000.0, 0.0, 18_720.0, in_flight=False) == 1.0


def test_fuel_fraction_full_at_departure_instant() -> None:
    """The instant the tank is filled (t_s == departure) reads 100 %."""
    from cuas_sim.viz.hud import fuel_fraction

    assert fuel_fraction(9_180.0, 9_180.0, 18_720.0, in_flight=True) == pytest.approx(1.0)


def test_fuel_fraction_decays_linearly() -> None:
    """In flight the fraction decays linearly with elapsed time."""
    from cuas_sim.viz.hud import fuel_fraction

    endurance = 18_720.0  # 5.2 h
    dep = 9_180.0
    # Quarter, half, three-quarter of the way through the endurance window.
    assert fuel_fraction(dep + endurance * 0.25, dep, endurance, True) == pytest.approx(0.75)
    assert fuel_fraction(dep + endurance * 0.50, dep, endurance, True) == pytest.approx(0.50)
    assert fuel_fraction(dep + endurance * 0.75, dep, endurance, True) == pytest.approx(0.25)


def test_fuel_fraction_clamps_to_zero_when_exhausted() -> None:
    """Past the endurance the fraction clamps at 0 (never negative)."""
    from cuas_sim.viz.hud import fuel_fraction

    endurance = 18_720.0
    dep = 9_180.0
    assert fuel_fraction(dep + endurance + 5_000.0, dep, endurance, True) == 0.0


# ─── rockets_remaining_at (FR-11) ──────────────────────────────────────────────


def test_rockets_full_before_any_shot() -> None:
    """Before the first shot fires the launcher is full (= budget)."""
    from cuas_sim.viz.hud import rockets_remaining_at

    fire_times = [10_000.0, 10_013.0, 10_500.0]
    assert rockets_remaining_at(fire_times, 9_999.0, 49) == 49


def test_rockets_deplete_per_shot_at_fire_time() -> None:
    """Each rocket leaves the count at its own fire instant (not at pursuit start)."""
    from cuas_sim.viz.hud import rockets_remaining_at

    fire_times = [10_000.0, 10_013.0, 10_026.0]
    assert rockets_remaining_at(fire_times, 10_000.0, 49) == 48   # first away
    assert rockets_remaining_at(fire_times, 10_013.0, 49) == 47   # second away
    assert rockets_remaining_at(fire_times, 10_026.0, 49) == 46   # third away
    assert rockets_remaining_at(fire_times, 10_005.0, 49) == 48   # mid-burst: only 1 fired


def test_rockets_never_negative() -> None:
    """An over-budget tally floors at zero (Winchester), never negative."""
    from cuas_sim.viz.hud import rockets_remaining_at

    fire_times = [10_000.0 + i for i in range(60)]
    assert rockets_remaining_at(fire_times, 11_000.0, 49) == 0


def test_rockets_reset_after_refuel() -> None:
    """A FARP rearm (refuel_times) reloads the launcher: only post-rearm shots count."""
    from cuas_sim.viz.hud import rockets_remaining_at

    fire_times = [10_000.0, 10_001.0, 10_002.0, 10_500.0, 10_501.0, 10_502.0, 10_503.0,
                  20_000.0, 20_001.0, 20_002.0, 20_003.0, 20_004.0]
    refuel = [15_000.0]
    # Before refuel: budget - 7 shots = 42.
    assert rockets_remaining_at(fire_times, 11_000.0, 49, refuel) == 42
    # Right after refuel, before the next burst: back to full.
    assert rockets_remaining_at(fire_times, 16_000.0, 49, refuel) == 49
    # After the post-refuel burst (5 shots): only those count.
    assert rockets_remaining_at(fire_times, 21_000.0, 49, refuel) == 44


# ─── shot_fire_times (FR-11) ───────────────────────────────────────────────────


def test_shot_fire_times_single_shot_at_fire() -> None:
    """A single shot fires once, exactly at t_fire."""
    from cuas_sim.viz.hud import shot_fire_times

    assert shot_fire_times(10_000.0, 10_050.0, 1) == [10_000.0]


def test_shot_fire_times_no_shots_empty() -> None:
    """An engagement that never fired yields no fire instants."""
    from cuas_sim.viz.hud import shot_fire_times

    assert shot_fire_times(10_000.0, 10_000.0, 0) == []


def test_shot_fire_times_spread_uniformly_over_window() -> None:
    """Multi-shot fire instants spread uniformly across [t_first, t_last]."""
    from cuas_sim.viz.hud import shot_fire_times

    times = shot_fire_times(10_000.0, 10_039.0, 4)
    assert times == pytest.approx([10_000.0, 10_013.0, 10_026.0, 10_039.0])


def test_shot_fire_times_zero_span_burst() -> None:
    """A zero-span window (instant kill) bursts all shots at t_first."""
    from cuas_sim.viz.hud import shot_fire_times

    assert shot_fire_times(10_000.0, 10_000.0, 3) == [10_000.0, 10_000.0, 10_000.0]


# ─── active_engagement_at (FR-11) ──────────────────────────────────────────────


def test_active_engagement_inside_window() -> None:
    """Within [t_start, t_end] the helper returns that engagement's target."""
    from cuas_sim.viz.hud import active_engagement_at

    windows = [(9_300.0, 9_400.0, "17"), (9_800.0, 9_950.0, "23")]
    assert active_engagement_at(windows, 9_350.0) == "17"
    assert active_engagement_at(windows, 9_900.0) == "23"


def test_active_engagement_outside_window_is_none() -> None:
    """Between or beyond engagement windows there is no active target."""
    from cuas_sim.viz.hud import active_engagement_at

    windows = [(9_300.0, 9_400.0, "17"), (9_800.0, 9_950.0, "23")]
    assert active_engagement_at(windows, 9_000.0) is None   # before all
    assert active_engagement_at(windows, 9_600.0) is None   # gap
    assert active_engagement_at(windows, 10_000.0) is None  # after all


def test_active_engagement_inclusive_bounds() -> None:
    """The window bounds are inclusive (kill instant still shows the target)."""
    from cuas_sim.viz.hud import active_engagement_at

    windows = [(9_300.0, 9_400.0, "17")]
    assert active_engagement_at(windows, 9_300.0) == "17"
    assert active_engagement_at(windows, 9_400.0) == "17"


# ─── a29_state_at (FR-11/FR-12) ────────────────────────────────────────────────


def test_a29_state_at_defaults_before_first_event() -> None:
    """No event yet (or t before first) → the AT_BASE default."""
    from cuas_sim.viz.hud import a29_state_at

    assert a29_state_at([], 5_000.0) == A29State.AT_BASE.value
    events = [(9_180.0, A29State.AT_CAP_PEA.value)]
    assert a29_state_at(events, 9_000.0) == A29State.AT_BASE.value


def test_a29_state_at_returns_latest_past_transition() -> None:
    """The state is the newest transition whose t_ev ≤ t_s (matches _state_colors)."""
    from cuas_sim.viz.hud import a29_state_at

    events = [
        (9_180.0, A29State.AT_CAP_PEA.value),
        (9_300.0, A29State.PURSUING.value),
        (9_400.0, A29State.ENGAGING.value),
        (9_460.0, A29State.AT_CAP_PEA.value),
    ]
    assert a29_state_at(events, 9_250.0) == A29State.AT_CAP_PEA.value
    assert a29_state_at(events, 9_350.0) == A29State.PURSUING.value
    assert a29_state_at(events, 9_420.0) == A29State.ENGAGING.value
    assert a29_state_at(events, 99_999.0) == A29State.AT_CAP_PEA.value


# ─── is_in_flight / should_render_hud (FR-11/FR-12) ────────────────────────────


def test_in_flight_classification() -> None:
    """Grounded/refuelled states burn no tracked fuel; airborne states do.

    The SBPV→FARP transit (``TRANSIT_TO_FARP``) is airborne and burns fuel from
    the takeoff anchor; the aircraft tops back up at the FARP
    (``AT_FARP_PRE_PEA`` reads full) before the CAP departure.
    """
    from cuas_sim.viz.hud import is_in_flight

    for grounded in (
        A29State.AT_BASE, A29State.AT_FARP_PRE_PEA,
        A29State.AT_FARP_POST_PEA, A29State.MISSION_COMPLETE,
    ):
        assert is_in_flight(grounded.value) is False, grounded
    for airborne in (
        A29State.TRANSIT_TO_FARP, A29State.REPOSITIONING_TO_CAP,
        A29State.HOLDING_ORBIT, A29State.AT_CAP_PEA,
        A29State.PURSUING, A29State.ENGAGING, A29State.RETURNING_TO_CAP,
        A29State.RTB_FUEL_LIMIT, A29State.RECOVERING_TO_BASE,
    ):
        assert is_in_flight(airborne.value) is True, airborne


def test_should_render_hud_skips_base_and_complete() -> None:
    """FR-12 — HUD hidden for AT_BASE / MISSION_COMPLETE, shown otherwise."""
    from cuas_sim.viz.hud import should_render_hud

    assert should_render_hud(A29State.AT_BASE.value) is False
    assert should_render_hud(A29State.MISSION_COMPLETE.value) is False
    for shown in (
        A29State.AT_CAP_PEA, A29State.PURSUING, A29State.ENGAGING,
        A29State.RTB_FUEL_LIMIT, A29State.TRANSIT_TO_FARP,
    ):
        assert should_render_hud(shown.value) is True, shown


# ─── fuel_anchor_s (FR-11) ─────────────────────────────────────────────────────


def test_fuel_anchor_uses_latest_repositioning() -> None:
    """The fuel clock anchors on the most recent FARP departure (CAP reposition)."""
    from cuas_sim.viz.hud import fuel_anchor_s

    events = [
        (100.0, A29State.TRANSIT_TO_FARP.value),
        (3_000.0, A29State.REPOSITIONING_TO_CAP.value),
        (9_180.0, A29State.AT_CAP_PEA.value),
    ]
    assert fuel_anchor_s(events, 9_300.0, fallback_s=9_180.0) == 3_000.0


def test_fuel_anchor_resets_on_post_pea_refuel() -> None:
    """A post-PEA FARP refuel re-anchors the clock (newer than the CAP departure)."""
    from cuas_sim.viz.hud import fuel_anchor_s

    events = [
        (3_000.0, A29State.REPOSITIONING_TO_CAP.value),
        (9_180.0, A29State.AT_CAP_PEA.value),
        (20_000.0, A29State.RTB_FUEL_LIMIT.value),
        (21_000.0, A29State.AT_FARP_POST_PEA.value),
        (21_000.0, A29State.RECOVERING_TO_BASE.value),
    ]
    # During recovery, the clock anchors at the refuel, not the CAP departure.
    assert fuel_anchor_s(events, 22_000.0, fallback_s=9_180.0) == 21_000.0


def test_fuel_anchor_on_sbpv_takeoff() -> None:
    """The SBPV→FARP transit burns from the takeoff anchor (TRANSIT_TO_FARP)."""
    from cuas_sim.viz.hud import fuel_anchor_s, fuel_fraction, is_in_flight

    events = [
        (0.0, A29State.TRANSIT_TO_FARP.value),
        (3_180.0, A29State.AT_FARP_PRE_PEA.value),
        (3_180.0, A29State.REPOSITIONING_TO_CAP.value),
    ]
    # Mid-transit: the clock is anchored at the takeoff, not the PEA fallback,
    # so a full tank burns down during the leg (in-flight, fraction < 1).
    assert fuel_anchor_s(events, 1_500.0, fallback_s=9_180.0) == 0.0
    assert is_in_flight(A29State.TRANSIT_TO_FARP.value) is True
    frac = fuel_fraction(1_500.0, 0.0, 18_720.0, in_flight=True)
    assert frac == pytest.approx(1.0 - 1_500.0 / 18_720.0)
    assert frac < 1.0
    # After the FARP top-up the CAP departure re-anchors (newer than takeoff).
    assert fuel_anchor_s(events, 4_000.0, fallback_s=9_180.0) == 3_180.0


def test_fuel_anchor_falls_back_without_departure_event() -> None:
    """skip_pre_pea runs log no REPOSITIONING_TO_CAP → fall back to PEA start."""
    from cuas_sim.viz.hud import fuel_anchor_s

    events = [(9_180.0, A29State.AT_CAP_PEA.value), (9_300.0, A29State.PURSUING.value)]
    assert fuel_anchor_s(events, 9_400.0, fallback_s=9_180.0) == 9_180.0
