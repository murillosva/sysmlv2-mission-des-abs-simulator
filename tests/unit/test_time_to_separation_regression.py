from __future__ import annotations

"""Regression: ``time_to_separation`` must be bit-exact vs Spec 014 (FR-4, AC-5).
This test PASSES from the start and stays green throughout Spec 016.

5 hardcoded scenarios derived from Spec 014 §5.1
on the canonical head-on geometry:
    a29_xy=(0,0), a29_speed=144, uas_xy=(100_000, 0), uas_heading=π,
    uas_speed=29, a29_heading=0 (head-on collision course),
    closing rate = 173 m/s, |r(t)| = 100_000 − 173·t.
"""

import math

import pytest

from cuas_sim.geometry.intercept import time_to_separation
from cuas_sim.geometry.projection import XY

_CANONICAL = dict(
    a29_xy=XY(0.0, 0.0),
    a29_speed_mps=144.0,
    uas_xy=XY(100_000.0, 0.0),
    uas_heading_rad=math.pi,
    uas_speed_mps=29.0,
)


def test_time_to_separation_bit_exact_vs_spec014() -> None:
    """5 known-answer scenarios bit-exact vs Spec 014 §5.1."""
    # 1) target_sep = 2_000 → t = 98_000/173 (Spec 014 T1)
    t = time_to_separation(**_CANONICAL, a29_heading_rad=0.0, target_sep_m=2_000.0)
    assert t == pytest.approx(98_000.0 / 173.0, rel=1e-12)

    # 2) target_sep = 500 → t = 99_500/173 (Spec 014 T2)
    t = time_to_separation(**_CANONICAL, a29_heading_rad=0.0, target_sep_m=500.0)
    assert t == pytest.approx(99_500.0 / 173.0, rel=1e-12)

    # 3) target_sep = 14_000 → t = 86_000/173 (modernized envelope max)
    t = time_to_separation(**_CANONICAL, a29_heading_rad=0.0, target_sep_m=14_000.0)
    assert t == pytest.approx(86_000.0 / 173.0, rel=1e-12)

    # 4) target_sep = 8_000 (mid-range)
    t = time_to_separation(**_CANONICAL, a29_heading_rad=0.0, target_sep_m=8_000.0)
    assert t == pytest.approx(92_000.0 / 173.0, rel=1e-12)

    # 5) target_sep = 50_000 (well above current — but still positive root)
    t = time_to_separation(**_CANONICAL, a29_heading_rad=0.0, target_sep_m=50_000.0)
    assert t == pytest.approx(50_000.0 / 173.0, rel=1e-12)
