"""Known-answer tests for the AN-3 surrogate prediction module (Spec 028 §5.1)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cuas_sim.analysis.features import feature_scores
from cuas_sim.analysis.prediction import (
    build_surrogate_report,
    cross_validate_surrogate,
    feature_importances,
    train_surrogate,
)

_FACTORS = ["lambda_per_h", "uas_archetype_p_mr"]
_RANGES = {"lambda_per_h": (2.0, 18.0), "uas_archetype_p_mr": (0.0, 1.0)}


def _per_dp_table(
    seed: int = 0, n: int = 40, a: float = 3.0, b: float = 0.05, noise: float = 0.3
) -> pd.DataFrame:
    """Per-design-point training table: gain ~ a*lambda + b*mix + small noise."""
    rng = np.random.default_rng(seed)
    lam = rng.uniform(2.0, 18.0, n)
    pmr = rng.uniform(0.0, 1.0, n)
    gain = a * lam + b * pmr + rng.normal(0.0, noise, n)
    return pd.DataFrame(
        {"lambda_per_h": lam, "uas_archetype_p_mr": pmr, "gain": gain}
    )


def _make_datafarm(seed: int = 0, n_points: int = 8, n_reps: int = 6) -> pd.DataFrame:
    """Minimal synthetic AN-1 datafarm (tidy/long) — no simulation run."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dp in range(n_points):
        lam = 2.0 + dp * 2.0
        pmr = 0.05 + dp * 0.1
        for rep in range(n_reps):
            sd = 1000 * dp + rep
            bsl = float(rng.uniform(0.05, 0.20))
            mod = bsl + 0.01 * lam + float(rng.uniform(0.0, 0.05))
            base = {"design_point_id": dp, "lambda_per_h": lam,
                    "uas_archetype_p_mr": pmr, "rep": rep, "seed": sd, "aor": "ALL"}
            for cfg, val in (("baseline", bsl), ("modernized", mod), ("delta", mod - bsl)):
                rows.append({**base, "config": cfg, "metric": "MOS-1", "value": val})
            rows.append({**base, "config": "modernized",
                         "metric": "rockets_per_engagement", "value": float(rng.uniform(1, 5))})
    return pd.DataFrame(rows)


def test_feature_importance_dominant_factor() -> None:
    train = _per_dp_table(seed=1, n=40)
    fit = train_surrogate(
        train, factors=_FACTORS, target="gain", model="rf", model_seed=42
    )
    imps = feature_importances(fit)
    assert imps is not None
    assert imps[0]["factor"] == "lambda_per_h"
    by = {r["factor"]: r["importance"] for r in imps}
    assert by["lambda_per_h"] > by["uas_archetype_p_mr"]


def test_cv_deterministic() -> None:
    train = _per_dp_table(seed=2, n=40)
    cv1 = cross_validate_surrogate(
        train, factors=_FACTORS, target="gain", model="rf", model_seed=7, n_folds=5
    )
    cv2 = cross_validate_surrogate(
        train, factors=_FACTORS, target="gain", model="rf", model_seed=7, n_folds=5
    )
    assert cv1 == cv2  # same seed -> identical CV
    assert cv1.mean["r2"] > 0.8  # near-linear signal


def test_predict_grid_reproducible_monotonic() -> None:
    train = _per_dp_table(seed=3, n=40)
    fit = train_surrogate(
        train, factors=_FACTORS, target="gain", model="rf", model_seed=11
    )
    from cuas_sim.analysis.prediction import predict_grid

    grid_a = predict_grid(fit, factor_ranges=_RANGES, resolution=10)
    grid_b = predict_grid(fit, factor_ranges=_RANGES, resolution=10)
    assert len(grid_a) == 100
    assert grid_a.equals(grid_b)  # reproducible

    lam_lo = grid_a["lambda_per_h"].min()
    lam_hi = grid_a["lambda_per_h"].max()
    lo_mean = grid_a[grid_a["lambda_per_h"] == lam_lo]["predicted"].mean()
    hi_mean = grid_a[grid_a["lambda_per_h"] == lam_hi]["predicted"].mean()
    assert hi_mean > lo_mean  # response increases with lambda


def test_surrogate_json_byte_deterministic() -> None:
    df = _make_datafarm(seed=0)
    r1, _ = build_surrogate_report(df, response="MOS-1", model="rf", model_seed=42)
    r2, _ = build_surrogate_report(df, response="MOS-1", model="rf", model_seed=42)
    assert r1 == r2
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert r1["field_version"] == "1.0"


def test_nfolds_clamp_small_sample() -> None:
    train3 = _per_dp_table(seed=4, n=3)
    cv = cross_validate_surrogate(
        train3, factors=_FACTORS, target="gain", model="rf", model_seed=5, n_folds=5
    )
    assert cv.n_folds == 3  # clamped to n_train
    assert cv.applicable

    train1 = _per_dp_table(seed=4, n=1)
    cv1 = cross_validate_surrogate(
        train1, factors=_FACTORS, target="gain", model="rf", model_seed=5, n_folds=5
    )
    assert not cv1.applicable  # cannot cross-validate a single point


def test_corroborates_an2_feature_scores() -> None:
    train = _per_dp_table(seed=6, n=40)
    fit = train_surrogate(
        train, factors=_FACTORS, target="gain", model="rf", model_seed=13
    )
    imps = feature_importances(fit)
    assert imps is not None
    surrogate_top = imps[0]["factor"]

    scores = feature_scores(
        train, response="gain", factors=_FACTORS, scoring="f_regression"
    )
    an2_top = scores[0].factor

    assert surrogate_top == an2_top == "lambda_per_h"
