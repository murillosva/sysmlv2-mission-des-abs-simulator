from __future__ import annotations

"""Property-based oracles for the ``t_to_farp`` primitive (FR-7, FR-8).

Group B (Red phase): must FAIL until C-1 lands ``t_to_farp``.
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cuas_sim.geometry.projection import XY

_coord = st.floats(min_value=-3.0e5, max_value=3.0e5, allow_nan=False, allow_infinity=False)


@given(px=_coord, py=_coord, fx=_coord, fy=_coord)
def test_t_to_farp_non_negative_zero_at_farp(px: float, py: float, fx: float, fy: float) -> None:
    from cuas_sim.geometry.projection import t_to_farp

    t = t_to_farp(XY(px, py), XY(fx, fy), 144.0)
    assert t >= 0.0
    if px == fx and py == fy:
        assert t == 0.0


@given(
    fx=_coord, fy=_coord,
    near=st.floats(min_value=0.0, max_value=1.0e5, allow_nan=False),
    extra=st.floats(min_value=1.0, max_value=1.0e5, allow_nan=False),
)
def test_t_to_farp_monotonic_in_distance(fx: float, fy: float, near: float, extra: float) -> None:
    from cuas_sim.geometry.projection import t_to_farp

    farp = XY(fx, fy)
    # Two points along +x from the FARP: one nearer, one farther.
    p_near = XY(fx + near, fy)
    p_far = XY(fx + near + extra, fy)
    assert t_to_farp(p_far, farp, 144.0) >= t_to_farp(p_near, farp, 144.0)
    # Sanity: t equals euclidean distance / speed.
    assert math.isclose(
        t_to_farp(p_near, farp, 144.0), near / 144.0, rel_tol=1e-9, abs_tol=1e-6
    )


@given(
    endurance_s=st.floats(min_value=60.0, max_value=18_720.0, allow_nan=False),
    d0_m=st.floats(min_value=0.0, max_value=2.0e5, allow_nan=False),
    away_speed_mps=st.floats(min_value=1.0, max_value=144.0, allow_nan=False),
)
def test_bingo_commit_monotonic(endurance_s: float, d0_m: float, away_speed_mps: float) -> None:
    """Once ``remaining ≤ t_to_farp`` becomes true on an outbound trajectory it
    stays true (no "un-bingo"): the A-29 commits to the return (FR-8, FR-12)."""
    from cuas_sim.geometry.projection import t_to_farp

    v_cruise = 144.0
    farp = XY(0.0, 0.0)

    def predicate(t: float) -> bool:
        # A-29 flies straight away from the FARP: distance d0 + away_speed·t.
        pos = XY(d0_m + away_speed_mps * t, 0.0)
        remaining = endurance_s - t
        return remaining <= t_to_farp(pos, farp, v_cruise)

    # Scan time; once the predicate trips it must remain tripped.
    tripped = False
    for k in range(0, 400):
        t = float(k) * (endurance_s / 400.0)
        p = predicate(t)
        if tripped:
            assert p, "bingo predicate flipped back to False after committing"
        if p:
            tripped = True


@given(
    endurance_s=st.floats(min_value=60.0, max_value=18_720.0, allow_nan=False),
    d0_m=st.floats(min_value=0.0, max_value=2.0e5, allow_nan=False),
    away_speed_mps=st.floats(min_value=1.0, max_value=143.0, allow_nan=False),
)
def test_endurance_non_negative_at_bingo(
    endurance_s: float, d0_m: float, away_speed_mps: float
) -> None:
    """At the exact (continuous) crossing ``remaining = t_to_farp``, the A-29
    still has non-negative fuel — the ``≤`` gate guarantees a non-negative
    margin at commit (zero-reserve, Q-B1)."""
    from cuas_sim.geometry.projection import t_to_farp

    v_cruise = 144.0
    farp = XY(0.0, 0.0)
    # remaining(t) = endurance − t ; t_to_farp(t) = (d0 + away·t)/v.
    # Solve endurance − t* = (d0 + away·t*)/v  ⇒  t* = (endurance − d0/v)/(1 + away/v).
    t_star = (endurance_s - d0_m / v_cruise) / (1.0 + away_speed_mps / v_cruise)
    if t_star < 0.0:
        # Already past the point of no return at t=0 (immediate bingo); skip.
        return
    remaining = endurance_s - t_star
    pos = XY(d0_m + away_speed_mps * t_star, 0.0)
    assert remaining >= -1e-6
    assert remaining == pytest.approx(t_to_farp(pos, farp, v_cruise), abs=1e-3)
