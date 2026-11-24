from __future__ import annotations

"""Integration test for the PURSUING discrete-integration loop (FR-1, FR-2, FR-7).

T3 from §5.1: perpendicular UAS — pure pursuit converges to envelope entry
(<=14_000 m) in fewer than 30 iterations at dt=5 s.

Group B (Red phase): must FAIL until C-1 lands ``pure_pursuit_step``.
"""

import math


def test_perpendicular_uas_converges_under_30_iterations() -> None:
    """Iterating pure_pursuit_step in perpendicular geometry → envelope in < 30 steps."""
    from cuas_sim.geometry.intercept import pure_pursuit_step

    a29 = (0.0, 0.0)
    uas = (20_000.0, 0.0)
    uas_v = (0.0, 80.0)
    v_a29 = 144.0
    dt = 5.0
    envelope_max_m = 14_000.0

    steps = 0
    sep = math.hypot(uas[0] - a29[0], uas[1] - a29[1])
    for steps in range(1, 31):
        a29, uas = pure_pursuit_step(a29, uas, uas_v, v_a29=v_a29, dt=dt)
        sep = math.hypot(uas[0] - a29[0], uas[1] - a29[1])
        if sep <= envelope_max_m:
            break

    assert sep <= envelope_max_m, (
        f"perpendicular pure pursuit failed to converge in 30 iters; final sep={sep:.1f}"
    )
    assert steps <= 30
