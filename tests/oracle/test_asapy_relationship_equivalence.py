"""AsaPy relationship equivalence oracle (Spec 029 §5.3 S4 / FR-15).

SKIPs gracefully when ``asapy`` is not importable (the main venv never installs it;
the AsaPy oracle lives in a SEPARATE dev venv — Spec 029 §8 / INV-8). AsaPy's
``analyze_relationship`` **returns ``None``** — it is side-effecting (prints +
plots with seaborn, ``asapy-ref/asapy/analysis.py:1237``), so equivalence is by
**construction** against the same primitives it uses internally: Pearson
(``pandas.Series.corr``), Spearman (``scipy.stats.spearmanr``), Kendall
(``scipy.stats.kendalltau``) and OLS (``sklearn.LinearRegression`` -> slope /
intercept / ``r2_score`` / ``mean_squared_error``). Our ``analyze_relationship``
is PURE (returns the coefficients); the figures live in ``report.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("asapy")  # SKIP entire module without AsaPy (Spec 029 INV-8)
pytest.importorskip("sklearn")


def _toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(20260701)
    n = 60
    mix = rng.uniform(0.0, 1.0, n)
    gain = 0.25 * mix + rng.normal(0.0, 0.03, n)
    return pd.DataFrame({"mix": mix, "gain": gain})


def test_relationship_matches_reference_primitives() -> None:
    from scipy.stats import kendalltau, spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    from cuas_sim.analysis.stats import analyze_relationship

    df = _toy_df()
    res = analyze_relationship(df, "mix", "gain")

    # same primitives AsaPy uses internally (analyze_relationship returns None/plots)
    ref_pearson = df["mix"].corr(df["gain"])
    ref_spearman = spearmanr(df["mix"], df["gain"]).statistic
    ref_kendall = kendalltau(df["mix"], df["gain"]).statistic
    x = df[["mix"]].to_numpy(dtype=float)
    y = df["gain"].to_numpy(dtype=float)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    assert res.pearson == pytest.approx(ref_pearson, abs=1e-9)
    assert res.spearman == pytest.approx(ref_spearman, abs=1e-9)
    assert res.kendall == pytest.approx(ref_kendall, abs=1e-9)
    assert res.slope == pytest.approx(float(model.coef_[0]), abs=1e-9)
    assert res.intercept == pytest.approx(float(model.intercept_), abs=1e-9)
    assert res.r2 == pytest.approx(r2_score(y, y_pred), abs=1e-9)
    assert res.mse == pytest.approx(mean_squared_error(y, y_pred), abs=1e-9)
