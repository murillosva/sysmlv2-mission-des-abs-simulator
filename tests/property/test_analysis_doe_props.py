"""Property — AN-1 DoE (Spec 026 §5.2). Red phase first (Article III)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from cuas_sim.analysis.doe import Factor, create_design


@given(
    low=st.floats(-100, 100),
    width=st.floats(0.1, 200),
    n=st.integers(1, 30),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=50)
def test_design_in_range_and_typed(low: float, width: float, n: int, seed: int) -> None:
    high = low + width
    factors = [
        Factor("x", "numeric", low, high),
        Factor("k", "integer", 1, 6),
        Factor("c", "categorical", options=("a", "b")),
        Factor("b", "boolean"),
    ]
    df = create_design(factors, n_samples=n, design_seed=seed)
    assert ((df["x"] >= low) & (df["x"] <= high)).all()
    assert df["k"].between(1, 6).all()
    assert df["k"].map(lambda v: float(v).is_integer()).all()
    assert set(df["c"].unique()).issubset({"a", "b"})
    assert df["b"].map(lambda v: isinstance(v, (bool, np.bool_))).all()


@given(n=st.integers(2, 40), seed=st.integers(0, 10_000))
@settings(max_examples=50)
def test_latin_stratification(n: int, seed: int) -> None:
    low, high = 2.0, 18.0
    df = create_design([Factor("x", "numeric", low, high)], n_samples=n, design_seed=seed)
    u = (df["x"].to_numpy() - low) / (high - low)
    bins = np.clip(np.floor(u * n).astype(int), 0, n - 1)
    assert sorted(bins.tolist()) == list(range(n))


@given(n=st.integers(1, 30), seed=st.integers(0, 10_000))
@settings(max_examples=50)
def test_design_deterministic(n: int, seed: int) -> None:
    factors = [
        Factor("lambda_per_h", "numeric", 2.0, 18.0),
        Factor("p", "numeric", 0.0, 1.0),
    ]
    a = create_design(factors, n_samples=n, design_seed=seed)
    b = create_design(factors, n_samples=n, design_seed=seed)
    assert a.equals(b)
