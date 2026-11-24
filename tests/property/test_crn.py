"""Property-based tests for the CRN registry invariant.

Pillar: the **world streams** (arrivals, sector, spawn geometry, archetype)
must produce **byte-identical sequences** across paired runs sharing the same
master seed. The **decision streams** must be independent across configs.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from cuas_sim.rng import RNGRegistry


@given(seed=st.integers(min_value=0, max_value=2**63 - 1))
@settings(max_examples=50, deadline=None)
def test_world_streams_are_deterministic(seed: int) -> None:
    """Re-creating the registry with the same seed reproduces world samples."""
    r1 = RNGRegistry.from_master_seed(seed)
    r2 = RNGRegistry.from_master_seed(seed)
    # Pull the same number of draws from each world generator
    n = 100
    for attr in ("arrivals", "sector_choice", "spawn_geometry", "archetype"):
        s1 = getattr(r1, attr).random(n).tolist()
        s2 = getattr(r2, attr).random(n).tolist()
        assert s1 == s2


@given(seed=st.integers(min_value=0, max_value=2**63 - 1))
@settings(max_examples=50, deadline=None)
def test_decision_streams_split_per_config(seed: int) -> None:
    """For one master seed, baseline and modernized decision streams differ."""
    r = RNGRegistry.from_master_seed(seed)
    for attr in ("tracking_updates", "firing_phil", "p_hit_outcome",
                 "p_track_sampling", "reengagement_interval"):
        b = getattr(r, attr)["baseline"].random(50).tolist()
        m = getattr(r, attr)["modernized"].random(50).tolist()
        # Vanishingly small probability they coincide bit-for-bit.
        assert b != m


@given(seed=st.integers(min_value=0, max_value=2**63 - 1))
@settings(max_examples=30, deadline=None)
def test_world_streams_shared_across_configs(seed: int) -> None:
    """The world streams are a single instance — they don't split per config."""
    r = RNGRegistry.from_master_seed(seed)
    s1 = r.arrivals.random(50).tolist()
    # Re-pull from the same generator should yield NEW samples (i.e. not the
    # same as s1), but pulling from a fresh registry with same seed reproduces
    # exactly s1.
    s2 = r.arrivals.random(50).tolist()
    assert s1 != s2
    r2 = RNGRegistry.from_master_seed(seed)
    assert s1 == r2.arrivals.random(50).tolist()


def test_per_config_helper_returns_split_streams() -> None:
    r = RNGRegistry.from_master_seed(123)
    per_b = r.per_config("baseline")
    per_m = r.per_config("modernized")
    assert per_b.tracking_updates is r.tracking_updates["baseline"]
    assert per_m.tracking_updates is r.tracking_updates["modernized"]
