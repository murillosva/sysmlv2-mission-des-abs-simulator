"""Known-answer tests for the AN-2 features module (Spec 027 §5.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cuas_sim.analysis.features import feature_scores, pareto_front


def test_feature_score_deterministic() -> None:
    rng = np.random.default_rng(4)
    n = 60
    x1 = rng.uniform(2.0, 18.0, n)  # lambda_per_h — strong driver
    x2 = rng.uniform(0.0, 1.0, n)  # uas_archetype_p_mr — weak driver
    y = 3.0 * x1 + 0.1 * x2 + rng.normal(0.0, 0.5, n)
    df = pd.DataFrame({"lambda_per_h": x1, "uas_archetype_p_mr": x2, "resp": y})
    factors = ["lambda_per_h", "uas_archetype_p_mr"]

    for scoring in ("f_regression", "r_regression"):
        scores = feature_scores(df, response="resp", factors=factors, scoring=scoring)
        assert scores[0].factor == "lambda_per_h"
        by = {s.factor: abs(s.score) for s in scores}
        assert by["lambda_per_h"] > by["uas_archetype_p_mr"]

    a = feature_scores(df, response="resp", factors=factors, scoring="f_regression")
    b = feature_scores(df, response="resp", factors=factors, scoring="f_regression")
    assert a == b  # determinism


def test_pareto_front_known_set() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "ganho": [0.30, 0.34, 0.20, 0.10],
            "custo": [1.0, 3.2, 0.5, 5.0],
        }
    )

    front = pareto_front(df, maximize=["ganho"], minimize=["custo"])

    assert set(front["name"]) == {"A", "B", "C"}  # D dominated by A and C
