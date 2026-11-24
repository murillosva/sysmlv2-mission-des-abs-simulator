"""Property-based oracles for the AN-2 features module (Spec 027 §5.2 P1, P2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import assume, given
from hypothesis import strategies as st

from cuas_sim.analysis.features import feature_scores, pareto_front


def _dominates(q: tuple[float, float], p: tuple[float, float]) -> bool:
    """q dominates p: q >= p on gain AND q <= p on cost, strict somewhere (spec §4.2)."""
    qg, qc = q
    pg, pc = p
    ge = qg >= pg and qc <= pc
    strict = qg > pg or qc < pc
    return ge and strict


_obj = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)


@given(st.lists(st.tuples(_obj, _obj), min_size=1, max_size=15))
def test_pareto_correctness(pts: list[tuple[float, float]]) -> None:
    df = pd.DataFrame(pts, columns=["g", "c"]).reset_index(drop=True)
    front = pareto_front(df, maximize=["g"], minimize=["c"])
    front_idx = set(front.index)

    rows = {i: (float(r["g"]), float(r["c"])) for i, r in df.iterrows()}
    # every front point is non-dominated by any other point
    for i in front_idx:
        assert not any(_dominates(rows[j], rows[i]) for j in rows if j != i)
    # every non-front point is dominated by at least one front point
    for i in rows:
        if i not in front_idx:
            assert any(_dominates(rows[j], rows[i]) for j in front_idx)


@given(st.integers(min_value=0, max_value=10_000))
def test_feature_score_permutation_invariant(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 40
    x1 = rng.uniform(2.0, 18.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = 2.0 * x1 - 0.5 * x2 + rng.normal(0.0, 0.5, n)
    df = pd.DataFrame({"lambda_per_h": x1, "uas_archetype_p_mr": x2, "resp": y})
    factors = ["lambda_per_h", "uas_archetype_p_mr"]
    assume(float(np.std(x1)) > 1e-6 and float(np.std(x2)) > 1e-6)

    perm = rng.permutation(n)
    shuffled = df.iloc[perm].reset_index(drop=True)

    for scoring in ("f_regression", "r_regression"):
        base = feature_scores(df, response="resp", factors=factors, scoring=scoring)
        shuf = feature_scores(shuffled, response="resp", factors=factors, scoring=scoring)
        assert [s.factor for s in base] == [s.factor for s in shuf]
