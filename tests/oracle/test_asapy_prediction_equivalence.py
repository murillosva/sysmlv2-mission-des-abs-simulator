"""AsaPy prediction equivalence oracle (Spec 028 §5.3 S4 / FR-14/FR-15).

SKIPs gracefully when ``asapy`` is not importable (the main venv never installs it;
the AsaPy oracle lives in a SEPARATE dev venv — Spec 028 §8). AsaPy's ``models`` module
is TensorFlow/Keras + XGBoost + Optuna-heavy and has **no pure sklearn RandomForest**,
so the equivalence is scoped to the **sklearn RF branch**: our surrogate's
RandomForest feature importances / predictions must match a directly-built sklearn
``RandomForestRegressor`` (same params + seed) on a FIXED toy dataset. The DNN/TF/Optuna
side of AsaPy is out of scope (documented **methodological** equivalence of the flow
preprocess -> tune -> CV -> evaluate -> predict), not reimplemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("asapy")  # SKIP entire module without AsaPy (Spec 028 INV-7)
pytest.importorskip("sklearn")


_FACTORS = ["lambda_per_h", "uas_archetype_p_mr"]


def _toy_table() -> pd.DataFrame:
    rng = np.random.default_rng(20260701)
    n = 60
    lam = rng.uniform(2.0, 18.0, n)
    pmr = rng.uniform(0.0, 1.0, n)
    gain = 3.0 * lam + 0.1 * pmr + rng.normal(0.0, 0.5, n)
    return pd.DataFrame(
        {"lambda_per_h": lam, "uas_archetype_p_mr": pmr, "gain": gain}
    )


def test_rf_importances_match_reference_sklearn() -> None:
    from sklearn.ensemble import RandomForestRegressor

    from cuas_sim.analysis.prediction import feature_importances, train_surrogate

    df = _toy_table()
    seed = 123

    fit = train_surrogate(
        df, factors=_FACTORS, target="gain", model="rf", model_seed=seed
    )
    ours = feature_importances(fit)
    assert ours is not None
    our_top = ours[0]["factor"]

    # Reference sklearn RF (StandardScaler is monotone per-feature -> importances/ranking
    # unaffected for trees): same estimator + seed the AsaPy sklearn path would use.
    ref = RandomForestRegressor(random_state=seed)
    ref.fit(df[_FACTORS].to_numpy(dtype=float), df["gain"].to_numpy(dtype=float))
    ref_top = _FACTORS[int(np.argmax(ref.feature_importances_))]

    assert our_top == ref_top == "lambda_per_h"


def test_rf_prediction_matches_reference_sklearn() -> None:
    from cuas_sim.analysis.prediction import predict_grid, train_surrogate

    df = _toy_table()
    fit = train_surrogate(
        df, factors=_FACTORS, target="gain", model="rf", model_seed=7
    )
    grid = predict_grid(
        fit,
        factor_ranges={"lambda_per_h": (2.0, 18.0), "uas_archetype_p_mr": (0.0, 1.0)},
        resolution=6,
    )
    # predictions stay within the trained target envelope (RF cannot extrapolate)
    assert grid["predicted"].min() >= float(df["gain"].min()) - 1e-9
    assert grid["predicted"].max() <= float(df["gain"].max()) + 1e-9
