"""Property-based oracles for the AN-3 surrogate (Spec 028 §5.2 P1-P4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from cuas_sim.analysis.prediction import (
    build_surrogate_report,
    predict_grid,
    train_surrogate,
)

_FACTORS = ["lambda_per_h", "uas_archetype_p_mr"]
_RANGES = {"lambda_per_h": (2.0, 18.0), "uas_archetype_p_mr": (0.0, 1.0)}

# Training a RandomForest per example is not free; keep examples modest and drop the
# per-example deadline (fit time varies) — still well above the CI floor.
_ML = settings(max_examples=25, deadline=None)


def _per_dp_table(seed: int, n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lam = rng.uniform(2.0, 18.0, n)
    pmr = rng.uniform(0.0, 1.0, n)
    gain = 2.0 * lam - 0.5 * pmr + rng.normal(0.0, 0.5, n)
    return pd.DataFrame(
        {"lambda_per_h": lam, "uas_archetype_p_mr": pmr, "gain": gain}
    )


@_ML
@given(st.integers(min_value=0, max_value=10_000))
def test_prediction_within_training_envelope(seed: int) -> None:
    train = _per_dp_table(seed)
    fit = train_surrogate(
        train, factors=_FACTORS, target="gain", model="rf", model_seed=seed
    )
    grid = predict_grid(fit, factor_ranges=_RANGES, resolution=8)

    lo = float(train["gain"].min())
    hi = float(train["gain"].max())
    preds = grid["predicted"].to_numpy(dtype=float)
    tol = 1e-9
    # A RandomForest averages leaf means -> predictions stay within the target envelope.
    assert preds.min() >= lo - tol
    assert preds.max() <= hi + tol


@_ML
@given(st.integers(min_value=0, max_value=10_000))
def test_train_determinism(seed: int) -> None:
    train = _per_dp_table(seed)
    a = train_surrogate(train, factors=_FACTORS, target="gain", model="rf", model_seed=99)
    b = train_surrogate(train, factors=_FACTORS, target="gain", model="rf", model_seed=99)

    from cuas_sim.analysis.prediction import feature_importances

    assert feature_importances(a) == feature_importances(b)
    assert a.cv == b.cv


@_ML
@given(st.integers(min_value=0, max_value=10_000))
def test_row_permutation_invariance(seed: int) -> None:
    train = _per_dp_table(seed)
    rng = np.random.default_rng(seed + 1)
    shuffled = train.iloc[rng.permutation(len(train))].reset_index(drop=True)

    base = train_surrogate(train, factors=_FACTORS, target="gain", model="rf", model_seed=3)
    perm = train_surrogate(shuffled, factors=_FACTORS, target="gain", model="rf", model_seed=3)

    from cuas_sim.analysis.prediction import feature_importances

    # train_surrogate canonicalises row order -> identical fit regardless of input order.
    assert feature_importances(base) == feature_importances(perm)
    assert base.cv == perm.cv


def _make_datafarm(seed: int, n_points: int = 8, n_reps: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dp in range(n_points):
        lam = 2.0 + dp * 2.0
        pmr = 0.05 + dp * 0.1
        for rep in range(n_reps):
            bsl = float(rng.uniform(0.05, 0.20))
            mod = bsl + 0.01 * lam + float(rng.uniform(0.0, 0.05))
            base = {"design_point_id": dp, "lambda_per_h": lam,
                    "uas_archetype_p_mr": pmr, "rep": rep, "seed": 1000 * dp + rep,
                    "aor": "ALL"}
            for cfg, val in (("baseline", bsl), ("modernized", mod), ("delta", mod - bsl)):
                rows.append({**base, "config": cfg, "metric": "MOS-1", "value": val})
            rows.append({**base, "config": "modernized",
                         "metric": "rockets_per_engagement", "value": float(rng.uniform(1, 5))})
    return pd.DataFrame(rows)


@_ML
@given(st.integers(min_value=0, max_value=10_000))
def test_artifact_determinism(seed: int) -> None:
    df = _make_datafarm(seed)
    r1, _ = build_surrogate_report(df, response="MOS-1", model="rf", model_seed=21)
    r2, _ = build_surrogate_report(df, response="MOS-1", model="rf", model_seed=21)
    assert r1 == r2
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
