"""Surrogate prediction — sklearn substitute model over the data farm.

Mirrors the 4th AsaPy step (Dantas et al. 2024, §3.4 Prediction) reimplemented on
``scikit-learn`` only (no TensorFlow/Keras/Optuna/XGBoost). Trains a surrogate
regressor mapping the DoE factors (lambda_per_h, uas_archetype_p_mr) to a response
(Delta MOS-1 / MOS-1), reports honest k-fold cross-validation, exposes feature
importances (corroborating ``features.feature_scores``) and predicts a dense lambda x mix
response surface -- the cheap "what-if" for configs never simulated.

Training granularity is **per-design-point**: the classical
simulation-metamodeling unit (Kleijnen 2015) that keeps ordinary k-fold valid by
independence (Arlot & Celisse 2010; Santner, Williams & Notz 2018). The per-design-point
table is **canonically sorted by the factor columns** before fitting so the surrogate is
row-order invariant (a RandomForest's bootstrap indexes into rows).

**Pure**: no I/O, no ``matplotlib``, no global state, no
``RNGRegistry``. The only RNG is the explicit ``model_seed`` (meta-level),
threaded as ``random_state`` into every estimator/splitter. ``scikit-learn`` is imported
**lazily inside the functions** (optional ``[analysis]`` extra) so ``import
cuas_sim.analysis`` works without it. File writing + the surface figure live in
``report.py`` (the layer's sole I/O + matplotlib owner) -- this module only builds the
report dict and the grid DataFrame. Read-only over the metrics:
the datafarm values are ML **targets**, never recomputed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FIELD_VERSION = "1.0"

_FACTORS = ("lambda_per_h", "uas_archetype_p_mr")

# Fixed, documented default hyperparameters per model. Recorded in
# ``surrogate.json`` so the artifact is self-describing.
_DEFAULT_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "rf": {"n_estimators": 100, "max_depth": None},
    "gbm": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    "gp": {"normalize_y": True},
}

# Small, documented tuning grids (no Optuna).
_TUNE_GRID: dict[str, dict[str, list[Any]]] = {
    "rf": {"n_estimators": [100, 200], "max_depth": [None, 4, 8]},
    "gbm": {"n_estimators": [100, 200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
    "gp": {},
}

_INSTALL_HINT = (
    "scikit-learn is required for surrogate prediction; install the analysis extra: "
    'pip install -e ".[analysis]"'
)

__all__ = [
    "FIELD_VERSION",
    "CVScores",
    "SurrogateFit",
    "build_surrogate_report",
    "cross_validate_surrogate",
    "feature_importances",
    "predict_grid",
    "train_surrogate",
]


@dataclass(frozen=True)
class CVScores:
    """K-fold cross-validation scores for a surrogate."""

    n_folds: int
    per_fold: dict[str, list[float | None]]
    mean: dict[str, float | None]
    applicable: bool
    note: str = ""


@dataclass(frozen=True)
class SurrogateFit:
    """A trained surrogate + its metadata."""

    pipeline: Any
    model: str
    hyperparameters: dict[str, Any]
    tuned: bool
    factors: tuple[str, ...]
    target: str
    n_train: int
    cv: CVScores


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover — exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc


def _make_regressor(model: str, model_seed: int, hyperparams: dict[str, Any]) -> Any:
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=model_seed, **hyperparams)
    if model == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=model_seed, **hyperparams)
    if model == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor

        return GaussianProcessRegressor(random_state=model_seed, **hyperparams)
    raise ValueError(f"unknown model {model!r} (use rf/gbm/gp)")


def _make_pipeline(model: str, model_seed: int, hyperparams: dict[str, Any]) -> Any:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    reg = _make_regressor(model, model_seed, hyperparams)
    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])


def _prep_xy(
    train: pd.DataFrame, factors: Sequence[str], target: str
) -> tuple[np.ndarray, np.ndarray]:
    """Canonically-ordered (X, y): drop NaN, sort by factors so fit is row-order invariant."""
    cols = list(factors)
    df = train[[*cols, target]].dropna()
    df = df.sort_values(cols, kind="mergesort").reset_index(drop=True)
    x = df[cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    return x, y


def _cv_scores(
    x: np.ndarray,
    y: np.ndarray,
    make_pipe: Callable[[], Any],
    model_seed: int,
    n_folds: int,
) -> CVScores:
    """Honest k-fold CV (R2/MAE/RMSE per fold + mean); k clamped to n_train."""
    n = len(y)
    empty: dict[str, list[float | None]] = {"r2": [], "mae": [], "rmse": []}
    if n < 2:
        return CVScores(
            n_folds=0, per_fold=empty, mean={"r2": None, "mae": None, "rmse": None},
            applicable=False, note="need >= 2 design points for cross-validation",
        )

    import warnings

    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.model_selection import KFold, cross_validate

    k = min(n_folds, n)
    kf = KFold(n_splits=k, shuffle=True, random_state=model_seed)
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    # R2 is undefined on single-sample test folds (small n) -> recorded as None below;
    # silence the expected sklearn warning to keep the console/report clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        res = cross_validate(make_pipe(), x, y, cv=kf, scoring=scoring, error_score="raise")

    def _clean(vals: np.ndarray, negate: bool) -> list[float | None]:
        out: list[float | None] = []
        for v in vals:
            fv = float(-v if negate else v)
            out.append(fv if math.isfinite(fv) else None)
        return out

    per_fold: dict[str, list[float | None]] = {
        "r2": _clean(res["test_r2"], negate=False),
        "mae": _clean(res["test_mae"], negate=True),
        "rmse": _clean(res["test_rmse"], negate=True),
    }
    mean: dict[str, float | None] = {}
    for metric, vals in per_fold.items():
        finite = [v for v in vals if v is not None]
        mean[metric] = float(np.mean(finite)) if finite else None

    note = "" if k >= 2 else "single fold"
    if k < n_folds:
        note = f"k clamped to n_train={n}"
    return CVScores(n_folds=k, per_fold=per_fold, mean=mean, applicable=True, note=note)


def cross_validate_surrogate(
    train: pd.DataFrame,
    *,
    factors: Sequence[str],
    target: str,
    model: str = "rf",
    model_seed: int,
    n_folds: int = 5,
) -> CVScores:
    """Honest k-fold CV (R2/MAE/RMSE per fold + mean); ``k`` clamped to ``n_train``."""
    _require_sklearn()
    x, y = _prep_xy(train, factors, target)
    hp = _DEFAULT_HYPERPARAMS[model]
    return _cv_scores(x, y, lambda: _make_pipeline(model, model_seed, hp), model_seed, n_folds)


def _tune(
    x: np.ndarray, y: np.ndarray, model: str, model_seed: int, n_folds: int
) -> tuple[Any, dict[str, Any]]:
    """Optional GridSearchCV over a small documented grid (no Optuna)."""
    from sklearn.model_selection import GridSearchCV, KFold

    grid = _TUNE_GRID.get(model, {})
    base = _make_pipeline(model, model_seed, _DEFAULT_HYPERPARAMS[model])
    if not grid or len(y) < 2:
        base.fit(x, y)
        return base, dict(_DEFAULT_HYPERPARAMS[model])
    k = min(n_folds, len(y))
    kf = KFold(n_splits=k, shuffle=True, random_state=model_seed)
    param_grid = {f"reg__{name}": vals for name, vals in grid.items()}
    search = GridSearchCV(base, param_grid, cv=kf, scoring="r2")
    search.fit(x, y)
    best = {k2.replace("reg__", ""): v for k2, v in search.best_params_.items()}
    return search.best_estimator_, best


def train_surrogate(
    train: pd.DataFrame,
    *,
    factors: Sequence[str],
    target: str,
    model: str = "rf",
    model_seed: int,
    n_folds: int = 5,
    tune: bool = False,
) -> SurrogateFit:
    """Fit a StandardScaler+regressor Pipeline on the per-design-point table."""
    _require_sklearn()
    x, y = _prep_xy(train, factors, target)

    if tune:
        pipeline, hyperparams = _tune(x, y, model, model_seed, n_folds)
    else:
        hyperparams = dict(_DEFAULT_HYPERPARAMS[model])
        pipeline = _make_pipeline(model, model_seed, hyperparams)
        pipeline.fit(x, y)

    cv = _cv_scores(
        x, y, lambda: _make_pipeline(model, model_seed, hyperparams), model_seed, n_folds
    )
    return SurrogateFit(
        pipeline=pipeline, model=model, hyperparameters=hyperparams, tuned=tune,
        factors=tuple(factors), target=target, n_train=len(y), cv=cv,
    )


def feature_importances(fit: SurrogateFit) -> list[dict[str, Any]] | None:
    """Ranked factor importances (rf/gbm); ``None`` for gp."""
    reg = fit.pipeline.named_steps["reg"]
    imp = getattr(reg, "feature_importances_", None)
    if imp is None:
        return None
    pairs: list[dict[str, Any]] = [
        {"factor": f, "importance": float(v)}
        for f, v in zip(fit.factors, imp, strict=True)
    ]
    pairs.sort(key=lambda d: float(d["importance"]), reverse=True)
    return pairs


def predict_grid(
    fit: SurrogateFit,
    *,
    factor_ranges: dict[str, tuple[float, float]],
    resolution: int = 25,
) -> pd.DataFrame:
    """Predict the response over a dense lambda x mix grid."""
    axes = [np.linspace(factor_ranges[f][0], factor_ranges[f][1], resolution) for f in fit.factors]
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = np.column_stack([m.ravel() for m in mesh])
    preds = fit.pipeline.predict(flat)
    data: dict[str, np.ndarray] = {f: flat[:, i] for i, f in enumerate(fit.factors)}
    data["predicted"] = np.asarray(preds, dtype=float)
    return pd.DataFrame(data)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy scalars -> python and non-finite floats -> None."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def build_surrogate_report(
    datafarm: pd.DataFrame,
    *,
    response: str = "MOS-1",
    aor: str = "ALL",
    model: str = "rf",
    model_seed: int,
    n_folds: int = 5,
    tune: bool = False,
    grid_resolution: int = 25,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Assemble the ``surrogate.json`` dict + the dense grid DataFrame.

    Pure: read-only over ``datafarm``; deterministic given the
    same ``datafarm`` + ``model_seed``. Returns ``(report_dict, grid_df)``;
    ``report.write_surrogate`` persists both.
    """
    _require_sklearn()
    from cuas_sim.analysis.report import per_design_point

    per_dp = per_design_point(datafarm, response, aor)
    factors = [f for f in _FACTORS if f in per_dp.columns]

    fit = train_surrogate(
        per_dp, factors=factors, target="gain", model=model,
        model_seed=model_seed, n_folds=n_folds, tune=tune,
    )
    ranges = {
        f: (float(datafarm[f].min()), float(datafarm[f].max()))
        for f in factors
        if f in datafarm.columns
    }
    grid = predict_grid(fit, factor_ranges=ranges, resolution=grid_resolution)
    preds = grid["predicted"].to_numpy(dtype=float)

    report: dict[str, Any] = {
        "field_version": FIELD_VERSION,
        "study": {
            "response": response,
            "aor": aor,
            "factors": {f: {"min": ranges[f][0], "max": ranges[f][1]} for f in factors},
            "model_seed": model_seed,
            "n_train": fit.n_train,
        },
        "model": {
            "kind": fit.model,
            "hyperparameters": fit.hyperparameters,
            "tuned": fit.tuned,
        },
        "cv": {
            "n_folds": fit.cv.n_folds,
            "applicable": fit.cv.applicable,
            "note": fit.cv.note,
            "per_fold": fit.cv.per_fold,
            "mean": fit.cv.mean,
        },
        "feature_importances": feature_importances(fit),
        "grid_prediction": {
            "resolution": grid_resolution,
            "axes": {f: {"min": ranges[f][0], "max": ranges[f][1]} for f in factors},
            "summary": {
                "min": float(preds.min()),
                "max": float(preds.max()),
                "mean": float(preds.mean()),
            },
            "pointer": "grid_prediction.csv",
        },
        "n_train": fit.n_train,
    }
    return _json_safe(report), grid
