from __future__ import annotations

"""Known-answer tests for ``BaselineTrackingPolicy.p_track`` angular formula (FR-9).

Group B (Red phase): all tests below must FAIL until C-2.1 adds
``visual_acuity_rad`` to ``BaselineTrackingPolicy`` and rewrites ``p_track``.
"""

import numpy as np
import pytest

from cuas_sim.config import UASArchetypeSpec
from cuas_sim.policies.tracking import BaselineTrackingPolicy


def _make_policy(visual_acuity_rad: float = 2.9088e-4) -> BaselineTrackingPolicy:
    return BaselineTrackingPolicy(
        update_interval_min_s=5.0,
        update_interval_max_s=60.0,
        max_acquisition_m=13_000.0,
        dim_ref_m=14.0,
        sampling_annulus_min_m=2_000.0,
        sampling_annulus_max_m=13_000.0,
        visual_acuity_rad=visual_acuity_rad,  # ← Spec 016 FR-9 / FR-10
    )


@pytest.fixture
def mr() -> UASArchetypeSpec:
    return UASArchetypeSpec(
        name="multi-rotor", max_speed_mps=20.0, dimension_m=1.0, ceiling_m=4000
    )


@pytest.fixture
def fw() -> UASArchetypeSpec:
    return UASArchetypeSpec(
        name="fixed-wing", max_speed_mps=130.0, dimension_m=14.0, ceiling_m=18000
    )


def test_multi_rotor_at_13km_returns_0_2645(mr: UASArchetypeSpec) -> None:
    """At d=12_999, L=1, θ=2.9088e-4 → min(1, 1/3.781) ≈ 0.2645."""
    pol = _make_policy()
    rng = np.random.default_rng(0)
    expected = min(1.0, 1.0 / (12_999.0 * 2.9088e-4))
    assert pol.p_track(12_999.0, mr, rng) == pytest.approx(expected, abs=1e-4)
    assert pol.p_track(12_999.0, mr, rng) == pytest.approx(0.2645, abs=1e-3)


def test_fixed_wing_at_13km_returns_1_0(fw: UASArchetypeSpec) -> None:
    """At d=12_999, L=14 → min(1, 14/3.781) = 1.0."""
    pol = _make_policy()
    rng = np.random.default_rng(0)
    assert pol.p_track(12_999.0, fw, rng) == pytest.approx(1.0)


def test_multi_rotor_at_1km_returns_1_0(mr: UASArchetypeSpec) -> None:
    """At d=1_000, L=1 → min(1, 1/0.2909) = 1.0 (target resolvable up close)."""
    pol = _make_policy()
    rng = np.random.default_rng(0)
    assert pol.p_track(1_000.0, mr, rng) == pytest.approx(1.0)


def test_outside_annulus_returns_zero(fw: UASArchetypeSpec) -> None:
    """d > max_acquisition_m → 0.0."""
    pol = _make_policy()
    rng = np.random.default_rng(0)
    assert pol.p_track(13_001.0, fw, rng) == 0.0


def test_negative_distance_returns_zero(fw: UASArchetypeSpec) -> None:
    """d ≤ 0 → 0.0 (sanity)."""
    pol = _make_policy()
    rng = np.random.default_rng(0)
    assert pol.p_track(-1.0, fw, rng) == 0.0
    assert pol.p_track(0.0, fw, rng) == 0.0
