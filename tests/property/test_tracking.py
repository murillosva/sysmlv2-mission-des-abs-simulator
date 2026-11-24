"""Property-based tests for tracking policies — spec 013."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cuas_sim.config import MissionConfig, UASArchetypeSpec
from cuas_sim.policies.tracking import make_tracking_policy


@given(d=st.floats(min_value=14_001.0, max_value=20_000.0, allow_nan=False))
@settings(max_examples=500, deadline=None)
def test_modernized_p_track_fw_constant_property(d: float) -> None:
    cfg = MissionConfig()
    pol = make_tracking_policy(cfg, "modernized")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    rng = np.random.default_rng(0)
    assert pol.p_track(d, fw, rng) == pytest.approx(pol.p_track(15_000.0, fw, rng))


@given(
    d=st.floats(min_value=0.0, max_value=50_000.0, allow_nan=False),
    dim=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
)
@settings(max_examples=500, deadline=None)
def test_modernized_p_track_in_range(d: float, dim: float) -> None:
    cfg = MissionConfig()
    pol = make_tracking_policy(cfg, "modernized")
    archetype = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=dim, ceiling_m=18000)
    p = pol.p_track(d, archetype, np.random.default_rng(0))
    assert 0.0 <= p <= 1.0


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=500, deadline=None)
def test_baseline_first_update_interval_range(seed: int) -> None:
    cfg = MissionConfig()
    pol = make_tracking_policy(cfg, "baseline")
    rng = np.random.default_rng(seed)
    dt = pol.first_update_interval(rng)
    assert dt >= cfg.baseline_reaction_time_min_s
    assert dt <= cfg.baseline_update_interval_s[1]
