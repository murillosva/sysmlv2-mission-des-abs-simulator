from __future__ import annotations

"""Property-based tests for ``pure_pursuit_step`` (P1, P2, P6 — spec §5.2).

Hypothesis with ``max_examples=500`` (spec §5.2 requirement). Tests below must
FAIL until C-1 lands ``pure_pursuit_step``.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_finite = st.floats(
    min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False
)
_dt = st.floats(min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False)
_v_a29 = st.floats(min_value=50.0, max_value=300.0, allow_nan=False, allow_infinity=False)
_uas_v = st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False)


@given(
    ax=_finite, ay=_finite,
    ux=_finite, uy=_finite,
    vux=_uas_v, vuy=_uas_v,
    v_a29=_v_a29, dt=_dt,
)
@settings(max_examples=500, deadline=None)
def test_step_preserves_velocity_magnitudes(
    ax: float, ay: float, ux: float, uy: float,
    vux: float, vuy: float, v_a29: float, dt: float,
) -> None:
    """||a29_new − a29|| == v_a29·dt and ||uas_new − uas|| == ||uas_v||·dt."""
    from cuas_sim.geometry.intercept import pure_pursuit_step

    # Skip degenerate case: A-29 already coincident with UAS (atan2 undefined).
    if math.isclose(ax, ux, abs_tol=1e-9) and math.isclose(ay, uy, abs_tol=1e-9):
        return
    a29 = (ax, ay)
    uas = (ux, uy)
    uas_vxy = (vux, vuy)
    a29_new, uas_new = pure_pursuit_step(a29, uas, uas_vxy, v_a29=v_a29, dt=dt)
    a29_step = math.hypot(a29_new[0] - ax, a29_new[1] - ay)
    uas_step = math.hypot(uas_new[0] - ux, uas_new[1] - uy)
    assert a29_step == pytest.approx(v_a29 * dt, rel=1e-9, abs=1e-6)
    assert uas_step == pytest.approx(math.hypot(vux, vuy) * dt, rel=1e-9, abs=1e-6)


@given(
    ax=st.floats(-5_000, 5_000), ay=st.floats(-5_000, 5_000),
    ux=st.floats(8_000, 12_000), uy=st.floats(-2_000, 2_000),
    uas_speed=st.floats(0.0, 100.0),
    uas_dir=st.floats(0.0, 2 * math.pi),
)
@settings(max_examples=500, deadline=None)
def test_convergence_when_a29_faster(
    ax: float, ay: float, ux: float, uy: float,
    uas_speed: float, uas_dir: float,
) -> None:
    """If v_a29 > ||uas_v|| and UAS within tracking range, loop converges
    (separation reaches envelope) within a bounded number of iterations."""
    from cuas_sim.geometry.intercept import pure_pursuit_step

    v_a29 = 144.0
    if uas_speed >= v_a29 - 1e-6:
        return  # property only applies when A-29 is faster
    uas_v = (uas_speed * math.cos(uas_dir), uas_speed * math.sin(uas_dir))
    a29 = (ax, ay)
    uas = (ux, uy)
    dt = 2.0
    max_iters = 1_000
    envelope_max_m = 14_000.0
    sep = math.hypot(ux - ax, uy - ay)
    if sep > 20_000.0:
        return  # outside initial tracking — spec scope
    for _ in range(max_iters):
        a29, uas = pure_pursuit_step(a29, uas, uas_v, v_a29=v_a29, dt=dt)
        sep = math.hypot(uas[0] - a29[0], uas[1] - a29[1])
        if sep <= envelope_max_m:
            return
    pytest.fail(f"loop did not converge in {max_iters} iters; final sep={sep:.1f}")


@given(
    ax=_finite, ay=_finite,
    ux=_finite, uy=_finite,
    vux=_uas_v, vuy=_uas_v,
    v_a29=_v_a29, dt=_dt,
)
@settings(max_examples=500, deadline=None)
def test_pure_pursuit_step_is_deterministic(
    ax: float, ay: float, ux: float, uy: float,
    vux: float, vuy: float, v_a29: float, dt: float,
) -> None:
    """pure_pursuit_step is a pure function — same inputs give bit-exact same outputs."""
    from cuas_sim.geometry.intercept import pure_pursuit_step

    if math.isclose(ax, ux, abs_tol=1e-9) and math.isclose(ay, uy, abs_tol=1e-9):
        return
    a = (ax, ay)
    u = (ux, uy)
    v = (vux, vuy)
    r1_a, r1_u = pure_pursuit_step(a, u, v, v_a29=v_a29, dt=dt)
    r2_a, r2_u = pure_pursuit_step(a, u, v, v_a29=v_a29, dt=dt)
    assert r1_a == r2_a
    assert r1_u == r2_u
