from __future__ import annotations

"""Regression: ``ModernizedTrackingPolicy.p_track`` must be bit-exact vs Spec 013
(FR-11, AC-4). This test PASSES from the start and stays green throughout Spec 016.

Hardcoded T1-T5 from Spec 013 §5.1.
"""

import numpy as np
import pytest

from cuas_sim.config import UASArchetypeSpec
from cuas_sim.policies.tracking import ModernizedTrackingPolicy


def _make_policy() -> ModernizedTrackingPolicy:
    return ModernizedTrackingPolicy(
        update_interval_min_s=1.0,
        update_interval_max_s=10.0,
        eoir_min_m=14_000.0,
        eoir_max_m=20_000.0,
        eoir_accuracy_m=5.0,
        p_fw=1.0,
        p_mr=0.2,
        sampling_annulus_min_m=14_000.0,
        sampling_annulus_max_m=20_000.0,
    )


def _fw() -> UASArchetypeSpec:
    return UASArchetypeSpec(name="fixed-wing", max_speed_mps=130.0, dimension_m=14.0, ceiling_m=18000)


def _mr() -> UASArchetypeSpec:
    return UASArchetypeSpec(name="multi-rotor", max_speed_mps=20.0, dimension_m=1.0, ceiling_m=4000)


def test_modernized_p_track_bit_exact_vs_spec013() -> None:
    """Spec 013 §5.1 T1-T5: bit-exact regression."""
    pol = _make_policy()
    rng = np.random.default_rng(0)

    # T1: fixed-wing, 17 km → 1.0
    assert pol.p_track(17_000.0, _fw(), rng) == 1.0
    # T2: multi-rotor, 16 km → 0.2
    assert pol.p_track(16_000.0, _mr(), rng) == pytest.approx(0.2, abs=1e-10)
    # T3: multi-rotor, exactly at lower bound 14_000 → 0.0
    assert pol.p_track(14_000.0, _mr(), rng) == 0.0
    # T4: fixed-wing, outside annulus (21 km) → 0.0
    assert pol.p_track(21_000.0, _fw(), rng) == 0.0
    # T5: fixed-wing constant in annulus (no range factor)
    for d in (14_001.0, 17_000.0, 20_000.0):
        assert pol.p_track(d, _fw(), rng) == 1.0, f"failed at d={d}"
