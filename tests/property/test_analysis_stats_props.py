"""Property-based oracle for the AN-2 bootstrap (Spec 027 §5.2 P3)."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from cuas_sim.analysis.stats import bootstrap_effect

_floats = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


@given(st.lists(_floats, min_size=8, max_size=25))
def test_bootstrap_ci_contains_effect(delta: list[float]) -> None:
    assume(float(np.std(delta)) > 1e-6)  # non-trivial sample

    res = bootstrap_effect(delta, bootstrap_seed=42, n_resamples=499)

    assert res.ci_low <= res.effect <= res.ci_high
    # determinism: same bootstrap_seed -> identical result
    res2 = bootstrap_effect(delta, bootstrap_seed=42, n_resamples=499)
    assert res == res2
