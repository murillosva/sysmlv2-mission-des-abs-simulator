from __future__ import annotations

"""Property-based oracles for firing sub-bands + Triangular draws (FR-1,2,3).

Group B (Red phase): must FAIL until C-2/C-3.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cuas_sim.config import MissionConfig
from cuas_sim.policies.weapon import make_weapon_policy


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_distance_within_subband(seed: int) -> None:
    from cuas_sim.policies.weapon import firing_subband

    policy = make_weapon_policy(MissionConfig(), "modernized")
    rng = np.random.default_rng(seed)
    dist, idx = policy.draw_firing_distance(rng)
    low, _mode, high = firing_subband(policy.envelope_min_m, policy.envelope_max_m, idx)
    assert low - 1e-6 <= dist <= high + 1e-6


@given(
    lo=st.floats(min_value=100.0, max_value=5_000.0),
    span=st.floats(min_value=300.0, max_value=20_000.0),
)
def test_subbands_partition_full_range(lo: float, span: float) -> None:
    from cuas_sim.policies.weapon import firing_subband

    hi = lo + span
    s0 = firing_subband(lo, hi, 0)
    s1 = firing_subband(lo, hi, 1)
    s2 = firing_subband(lo, hi, 2)
    # Strict, contiguous partition of ]lo, hi].
    assert s0[0] == pytest.approx(lo)
    assert s2[2] == pytest.approx(hi)
    assert s0[2] == pytest.approx(s1[0])  # no gap short→medium
    assert s1[2] == pytest.approx(s2[0])  # no gap medium→long
    # mode within [low, high] for each sub-band (numpy.triangular precondition).
    for low, mode, high in (s0, s1, s2):
        assert low <= mode <= high


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_reengagement_triangular_in_bounds(seed: int) -> None:
    cfg = MissionConfig()
    rng = np.random.default_rng(seed)
    dt = rng.triangular(*cfg.reengagement_triangular_s)
    assert 8.0 <= dt <= 18.0
