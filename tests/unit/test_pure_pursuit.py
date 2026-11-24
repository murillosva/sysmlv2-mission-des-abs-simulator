from __future__ import annotations

"""Known-answer tests for ``pure_pursuit_step`` (FR-6).

Group B (Red phase): all tests below must FAIL until C-1 lands ``pure_pursuit_step``.
"""

import math

import pytest


def test_head_on_correct_initial_heading() -> None:
    """Head-on: A-29 aims at UAS along +x; advance v_a29·dt = 1440 m."""
    from cuas_sim.geometry.intercept import pure_pursuit_step

    a29 = (0.0, 0.0)
    uas = (10_000.0, 0.0)
    uas_v = (-30.0, 0.0)
    a29_new, uas_new = pure_pursuit_step(a29, uas, uas_v, v_a29=144.0, dt=10.0)
    assert a29_new[0] == pytest.approx(1440.0, abs=1e-6)
    assert a29_new[1] == pytest.approx(0.0, abs=1e-6)
    assert uas_new[0] == pytest.approx(9700.0, abs=1e-6)
    assert uas_new[1] == pytest.approx(0.0, abs=1e-6)


def test_tail_chase_lag_emerges() -> None:
    """Crossing UAS: lag pursuit aims at the *current* UAS pos; lag emerges."""
    from cuas_sim.geometry.intercept import pure_pursuit_step

    a29 = (0.0, 0.0)
    uas = (0.0, 10_000.0)
    uas_v = (50.0, 0.0)
    a29_new, uas_new = pure_pursuit_step(a29, uas, uas_v, v_a29=144.0, dt=10.0)
    # A-29 heading = atan2(10_000 - 0, 0 - 0) = π/2 → advance 1440 in +y
    assert a29_new[0] == pytest.approx(0.0, abs=1e-6)
    assert a29_new[1] == pytest.approx(1440.0, abs=1e-6)
    # UAS advances 500 m in +x (heading fixed during the step)
    assert uas_new[0] == pytest.approx(500.0, abs=1e-6)
    assert uas_new[1] == pytest.approx(10_000.0, abs=1e-6)


def test_infeasible_when_uas_faster_in_adverse_geometry() -> None:
    """Iterating pure_pursuit_step with UAS fleeing faster: separation does not decrease.

    The leak-reason ``infeasible`` is asserted at controller level (B-3.1); this
    test isolates the *geometric primitive* property — three consecutive steps
    in adverse geometry produce non-decreasing separation.
    """
    from cuas_sim.geometry.intercept import pure_pursuit_step

    a29 = (0.0, 0.0)
    uas = (100.0, 0.0)
    uas_v = (200.0, 0.0)
    dt = 10.0
    seps = [math.hypot(uas[0] - a29[0], uas[1] - a29[1])]
    for _ in range(3):
        a29, uas = pure_pursuit_step(a29, uas, uas_v, v_a29=144.0, dt=dt)
        seps.append(math.hypot(uas[0] - a29[0], uas[1] - a29[1]))
    # Separation must be non-decreasing across all 3 steps.
    for s_prev, s_next in zip(seps, seps[1:], strict=False):  # noqa: RUF007 — intentional successive-pair scan
        assert s_next >= s_prev - 1e-9
    # And the final separation must exceed the initial (UAS faster).
    assert seps[-1] > seps[0]
