from __future__ import annotations

"""Sub-band partition + Triangular firing distance (FR-1, FR-2, FR-4).

Group B (Red phase): must FAIL until C-2 lands ``firing_subband`` and the
Triangular ``draw_firing_distance`` returning ``(distance, philosophy_idx)``.
"""

import numpy as np
import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.policies.weapon import make_weapon_policy


def test_apkws_subband_partition() -> None:
    """APKWS [2,14] km split in range/3; mode = SysML philosophy point."""
    from cuas_sim.policies.weapon import firing_subband

    # idx 0 short ]2,6], idx 1 medium ]6,10], idx 2 long ]10,14] (km → m)
    assert firing_subband(2_000.0, 14_000.0, 0) == pytest.approx((2_000.0, 2_000.0, 6_000.0))
    assert firing_subband(2_000.0, 14_000.0, 1) == pytest.approx((6_000.0, 8_000.0, 10_000.0))
    assert firing_subband(2_000.0, 14_000.0, 2) == pytest.approx((10_000.0, 14_000.0, 14_000.0))


def test_sbat70_subband_partition() -> None:
    """SBAT-70 [0.5,2.0] km split in range/3."""
    from cuas_sim.policies.weapon import firing_subband

    assert firing_subband(500.0, 2_000.0, 0) == pytest.approx((500.0, 500.0, 1_000.0))
    assert firing_subband(500.0, 2_000.0, 1) == pytest.approx((1_000.0, 1_250.0, 1_500.0))
    assert firing_subband(500.0, 2_000.0, 2) == pytest.approx((1_500.0, 2_000.0, 2_000.0))


def test_draw_within_apkws_subband() -> None:
    """``draw_firing_distance`` returns ``(distance, idx)`` with the distance
    strictly inside the sub-band of the drawn philosophy."""
    from cuas_sim.policies.weapon import firing_subband

    cfg = MissionConfig()
    policy = make_weapon_policy(cfg, "modernized")
    rng = np.random.default_rng(42)
    seen_idx: set[int] = set()
    for _ in range(300):
        dist, idx = policy.draw_firing_distance(rng)
        low, _mode, high = firing_subband(policy.envelope_min_m, policy.envelope_max_m, idx)
        assert low - 1e-6 <= dist <= high + 1e-6
        assert idx in (0, 1, 2)
        seen_idx.add(idx)
    # 300 draws should exercise all three philosophies (1/3 each).
    assert seen_idx == {0, 1, 2}


def test_short_philosophy_index_is_zero_for_min_subband() -> None:
    """The short philosophy (idx 0) maps to the lowest sub-band; the controller
    keys single-shot off ``idx == 0`` (FR-4)."""
    from cuas_sim.policies.weapon import firing_subband

    low0, mode0, high0 = firing_subband(2_000.0, 14_000.0, 0)
    low1, _m1, _h1 = firing_subband(2_000.0, 14_000.0, 1)
    # short sub-band is below medium; mode anchored at the envelope minimum.
    assert mode0 == low0 == 2_000.0
    assert high0 == low1  # strict partition, no gap
