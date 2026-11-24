from __future__ import annotations

"""Property-based tests for ``BaselineTrackingPolicy.p_track`` angular formula
(P3, P4, P5 — spec §5.2). Tests below must FAIL until C-2.1 rewrites p_track.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from cuas_sim.config import UASArchetypeSpec
from cuas_sim.policies.tracking import BaselineTrackingPolicy


def _make_policy() -> BaselineTrackingPolicy:
    return BaselineTrackingPolicy(
        update_interval_min_s=5.0,
        update_interval_max_s=60.0,
        max_acquisition_m=13_000.0,
        dim_ref_m=14.0,
        sampling_annulus_min_m=2_000.0,
        sampling_annulus_max_m=13_000.0,
        visual_acuity_rad=2.9088e-4,
    )


def _arch(L: float) -> UASArchetypeSpec:
    return UASArchetypeSpec(name="test", max_speed_mps=50.0, dimension_m=L, ceiling_m=4000)


@given(
    d1=st.floats(1.0, 13_000.0, allow_nan=False),
    d2=st.floats(1.0, 13_000.0, allow_nan=False),
    L=st.floats(0.1, 30.0, allow_nan=False),
)
@settings(max_examples=500, deadline=None)
def test_p_track_monotonic_non_increasing_in_distance(d1: float, d2: float, L: float) -> None:
    """For fixed L and d1 < d2 in annulus: p_track(d1) >= p_track(d2)."""
    if d1 > d2:
        d1, d2 = d2, d1
    if d1 == d2:
        return
    pol = _make_policy()
    arch = _arch(L)
    rng = np.random.default_rng(0)
    assert pol.p_track(d1, arch, rng) + 1e-12 >= pol.p_track(d2, arch, rng)


@given(
    d=st.floats(1.0, 13_000.0, allow_nan=False),
    L1=st.floats(0.1, 30.0, allow_nan=False),
    L2=st.floats(0.1, 30.0, allow_nan=False),
)
@settings(max_examples=500, deadline=None)
def test_p_track_monotonic_non_decreasing_in_dimension(d: float, L1: float, L2: float) -> None:
    """For fixed d, L1 < L2 ⇒ p_track(d, L1) <= p_track(d, L2)."""
    if L1 > L2:
        L1, L2 = L2, L1
    if L1 == L2:
        return
    pol = _make_policy()
    rng = np.random.default_rng(0)
    p1 = pol.p_track(d, _arch(L1), rng)
    p2 = pol.p_track(d, _arch(L2), rng)
    assert p1 <= p2 + 1e-12


@given(
    d=st.floats(-1_000.0, 50_000.0, allow_nan=False),
    L=st.floats(0.1, 30.0, allow_nan=False),
)
@settings(max_examples=500, deadline=None)
def test_p_track_in_unit_interval(d: float, L: float) -> None:
    """p_track is always in [0, 1] regardless of inputs."""
    pol = _make_policy()
    rng = np.random.default_rng(0)
    p = pol.p_track(d, _arch(L), rng)
    assert 0.0 <= p <= 1.0
