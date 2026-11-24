"""Property-based tests for time_to_separation — spec 014."""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from cuas_sim.geometry.intercept import collision_course_intercept
from cuas_sim.geometry.projection import XY


def _sep_at(a29: XY, asp: float, head: float, u: XY, uh: float,
            usp: float, t: float) -> float:
    ax = a29.x + asp * math.cos(head) * t
    ay = a29.y + asp * math.sin(head) * t
    ux = u.x + usp * math.cos(uh) * t
    uy = u.y + usp * math.sin(uh) * t
    return math.hypot(ux - ax, uy - ay)


_UX = st.floats(min_value=-150_000.0, max_value=150_000.0, allow_nan=False)
_UY = st.floats(min_value=-150_000.0, max_value=150_000.0, allow_nan=False)
_UH = st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False)
_USP = st.floats(min_value=10.0, max_value=130.0, allow_nan=False)


@given(ux=_UX, uy=_UY, uh=_UH, usp=_USP,
       frac=st.floats(min_value=0.05, max_value=0.95, allow_nan=False))
@settings(max_examples=400, deadline=None)
def test_separation_reached(ux: float, uy: float, uh: float, usp: float,
                            frac: float) -> None:
    from cuas_sim.geometry.intercept import time_to_separation

    a29 = XY(0.0, 0.0)
    u = XY(ux, uy)
    asp = 144.0
    geom = collision_course_intercept(a29, asp, u, uh, usp, envelope_max_m=2_000.0)
    assume(geom.feasible)
    d0 = geom.initial_distance_m
    assume(d0 > 1_000.0)
    target = frac * d0
    t = time_to_separation(a29, asp, u, uh, usp,
                           a29_heading_rad=geom.a29_heading_rad,
                           target_sep_m=target)
    assume(t is not None and t > 0.0)
    sep = _sep_at(a29, asp, geom.a29_heading_rad, u, uh, usp, t)
    assert abs(sep - target) <= max(1e-3, 1e-6 * target)


@given(ux=_UX, uy=_UY, uh=_UH, usp=_USP,
       s1=st.floats(min_value=0.05, max_value=0.45, allow_nan=False),
       s2=st.floats(min_value=0.50, max_value=0.95, allow_nan=False))
@settings(max_examples=400, deadline=None)
def test_monotonic_in_target(ux: float, uy: float, uh: float, usp: float,
                             s1: float, s2: float) -> None:
    from cuas_sim.geometry.intercept import time_to_separation

    a29 = XY(0.0, 0.0)
    u = XY(ux, uy)
    asp = 144.0
    geom = collision_course_intercept(a29, asp, u, uh, usp, envelope_max_m=2_000.0)
    assume(geom.feasible)
    d0 = geom.initial_distance_m
    assume(d0 > 1_000.0)
    h = geom.a29_heading_rad
    t_small = time_to_separation(a29, asp, u, uh, usp, a29_heading_rad=h,
                                 target_sep_m=s1 * d0)
    t_large = time_to_separation(a29, asp, u, uh, usp, a29_heading_rad=h,
                                 target_sep_m=s2 * d0)
    assume(t_small is not None and t_large is not None)
    # Larger separation is reached no later than the smaller one.
    assert t_large <= t_small + 1e-6


@given(ux=_UX, uy=_UY, uh=_UH, usp=_USP,
       frac=st.floats(min_value=0.05, max_value=0.95, allow_nan=False))
@settings(max_examples=400, deadline=None)
def test_t_fire_within_invariant(ux: float, uy: float, uh: float, usp: float,
                                 frac: float) -> None:
    from cuas_sim.geometry.intercept import time_to_separation

    a29 = XY(0.0, 0.0)
    u = XY(ux, uy)
    asp = 144.0
    env_max = 2_000.0
    geom = collision_course_intercept(a29, asp, u, uh, usp,
                                      envelope_max_m=env_max)
    assume(geom.feasible)
    d0 = geom.initial_distance_m
    assume(d0 > env_max)
    d_fire = env_max * frac  # in (0, env_max] ⊂ (0, d0)
    t = time_to_separation(a29, asp, u, uh, usp,
                           a29_heading_rad=geom.a29_heading_rad,
                           target_sep_m=d_fire)
    assume(t is not None)
    # Reaching a smaller separation than env_max takes ≥ t_envelope_entry,
    # and ≤ t_intercept (collision at separation 0).
    assert geom.t_envelope_entry - 1e-6 <= t <= geom.t_intercept + 1e-6


@given(ux=_UX, uy=_UY, uh=_UH, usp=_USP)
@settings(max_examples=400, deadline=None)
def test_envelope_entry_separation_is_env_max(ux: float, uy: float, uh: float,
                                              usp: float) -> None:
    a29 = XY(0.0, 0.0)
    u = XY(ux, uy)
    asp = 144.0
    env_max = 2_000.0
    geom = collision_course_intercept(a29, asp, u, uh, usp,
                                      envelope_max_m=env_max)
    assume(geom.feasible)
    t_env = geom.t_envelope_entry
    # Skip the clamped boundary cases (t_env == 0 already inside, or == t_int).
    assume(1e-6 < t_env < geom.t_intercept - 1e-6)
    sep = _sep_at(a29, asp, geom.a29_heading_rad, u, uh, usp, t_env)
    assert sep == pytest.approx(env_max, rel=1e-6, abs=1e-3)
