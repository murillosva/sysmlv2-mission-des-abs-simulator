from __future__ import annotations

"""Bit-exact regressions for the formulas Spec 017 must NOT touch (R1–R4, INV-3).

These tests PASS from the start and MUST stay green throughout Spec 017:
``time_to_separation`` (Spec 014), modernized Eq. (3.13) and baseline angular
Eq. (3.12) ``p_track``, and ``cumulative_neutralization`` (Eq. 3.15).
"""

import math

import numpy as np
import pytest

from cuas_sim.config import MissionConfig, UASArchetypeSpec
from cuas_sim.engagement.controller import cumulative_neutralization
from cuas_sim.geometry.intercept import time_to_separation
from cuas_sim.geometry.projection import XY
from cuas_sim.policies.tracking import make_tracking_policy

_FW = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130.0, dimension_m=14.0, ceiling_m=18000.0)
_MR = UASArchetypeSpec(name="multi-rotor", max_speed_mps=29.0, dimension_m=1.0, ceiling_m=6000.0)

_CANONICAL = dict(
    a29_xy=XY(0.0, 0.0),
    a29_speed_mps=144.0,
    uas_xy=XY(100_000.0, 0.0),
    uas_heading_rad=math.pi,
    uas_speed_mps=29.0,
)

#: 1 arcmin naked-eye acuity — SysML PilotVisualAcquisitionSystemNakedEye
#: .visualAcuityThreshold (hardcoded so a formula change breaks this test).
_ACUITY_RAD = 2.9088820866572160e-4


def test_time_to_separation_bit_exact() -> None:
    t = time_to_separation(**_CANONICAL, a29_heading_rad=0.0, target_sep_m=2_000.0)
    assert t == pytest.approx(98_000.0 / 173.0, rel=1e-12)
    t = time_to_separation(**_CANONICAL, a29_heading_rad=0.0, target_sep_m=14_000.0)
    assert t == pytest.approx(86_000.0 / 173.0, rel=1e-12)


def test_eq15_modernized_bit_exact() -> None:
    pol = make_tracking_policy(MissionConfig(), "modernized")
    rng = np.random.default_rng(0)
    assert pol.p_track(17_000.0, _FW, rng) == 1.0
    assert pol.p_track(16_000.0, _MR, rng) == pytest.approx(0.2, abs=1e-10)
    assert pol.p_track(14_000.0, _MR, rng) == 0.0  # at lower bound
    assert pol.p_track(21_000.0, _FW, rng) == 0.0  # outside annulus


def test_eq11_13_baseline_angular_bit_exact() -> None:
    pol = make_tracking_policy(MissionConfig(), "baseline")
    rng = np.random.default_rng(0)
    # Angular acuity formula: min(1, L_U / (d · θ_acuity)).
    for d, arch in ((13_000.0, _MR), (10_000.0, _MR), (13_000.0, _FW)):
        expected = min(1.0, arch.dimension_m / (d * _ACUITY_RAD))
        assert pol.p_track(d, arch, rng) == pytest.approx(expected, rel=1e-12)
    assert pol.p_track(13_001.0, _MR, rng) == 0.0   # beyond 13 km max
    assert pol.p_track(0.0, _MR, rng) == 0.0


def test_cumulative_neutralization_bit_exact() -> None:
    assert cumulative_neutralization([]) == 0.0
    assert cumulative_neutralization([0.8]) == pytest.approx(0.8, rel=1e-12)
    assert cumulative_neutralization([0.5, 0.5]) == pytest.approx(0.75, rel=1e-12)
    assert cumulative_neutralization([1.0, 0.3]) == pytest.approx(1.0, rel=1e-12)
