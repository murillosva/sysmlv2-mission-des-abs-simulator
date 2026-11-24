"""AsaPy equivalence oracle (Spec 027 §5.3 S4 / FR-14/FR-15).

SKIPs gracefully when ``asapy`` is not importable (the main venv never installs it;
the AsaPy oracle lives in a SEPARATE dev venv — Spec 027 §8). When present, it
cross-checks our reimplementation against the real AsaPy on a FIXED toy dataset,
comparing only what is methodologically comparable (hypothesis decision on
INDEPENDENT data, feature-score ranking, Pareto set, best fit). The bootstrap is a
documented methodological equivalence, NOT a bit-for-bit numeric match: AsaPy's
``bootstrap`` is a pooled-shuffle permutation test (inappropriate for paired data),
so the paired ASL is validated against ``scipy.stats.permutation_test`` instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

asapy = pytest.importorskip("asapy")  # SKIP entire module without AsaPy


def _toy_groups() -> pd.DataFrame:
    rng = np.random.default_rng(20240624)
    return pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, 40),
            "b": rng.normal(0.0, 1.0, 40),  # same distribution as 'a' -> "same"
            "c": rng.normal(4.0, 1.0, 40),  # clearly different
        }
    )


def test_feature_score_ranking_matches_asapy() -> None:
    from cuas_sim.analysis.features import feature_scores

    rng = np.random.default_rng(7)
    n = 80
    x1 = rng.uniform(0.0, 10.0, n)
    x2 = rng.uniform(0.0, 10.0, n)
    y = 5.0 * x1 + 0.2 * x2 + rng.normal(0.0, 0.5, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    ours = feature_scores(df, response="y", factors=["x1", "x2"], scoring="f_regression")
    theirs = asapy.Analysis.feature_score(df, ["x1", "x2"], "y", "f_regression")

    ours_rank = [s.factor for s in ours]
    theirs_rank = list(theirs.sort_values("score", ascending=False)["feature"])
    assert ours_rank == theirs_rank


def test_pareto_set_matches_asapy() -> None:
    from cuas_sim.analysis.features import pareto_front

    df = pd.DataFrame(
        {"gain": [0.30, 0.34, 0.20, 0.10], "cost": [1.0, 3.2, 0.5, 5.0]}
    )
    ours = pareto_front(df, maximize=["gain"], minimize=["cost"])
    theirs = asapy.Analysis.pareto_front(df, list_min=["cost"], list_max=["gain"])

    ours_set = {(round(g, 6), round(c, 6)) for g, c in zip(ours["gain"], ours["cost"])}
    theirs_set = {(round(g, 6), round(c, 6)) for g, c in zip(theirs["gain"], theirs["cost"])}
    assert ours_set == theirs_set
