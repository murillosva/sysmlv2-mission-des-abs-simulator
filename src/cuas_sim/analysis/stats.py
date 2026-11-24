"""Statistical analysis — hypothesis testing + bootstrap + fit + outliers.

Pure module: no ``RNGRegistry``, no I/O, no global state.
The only RNG is the explicit ``bootstrap_seed`` (meta-level), passed
to ``scipy.stats.bootstrap`` / the recentered-ASL via ``np.random.default_rng``.

Mirrors the AsaPy decision flow (Dantas et al. 2024, Fig. 2) over ``scipy.stats``
only (no ``pingouin``/GPL). The bootstrap follows Efron & Tibshirani (1993, *An
Introduction to the Bootstrap*, ch. 16): paired differences, BCa CI (percentile
fallback at small n) + recentered-under-H0 ASL p-value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_BOOTSTRAP_SEED = 12345
DEFAULT_N_RESAMPLES = 9999
# Below this many pairs BCa can undercover (Efron & Tibshirani);
# fall back to the percentile CI.
SMALL_N_THRESHOLD = 30

# Correlation-strength cutoffs (Schober, Boer & Schwarte 2018, Anesthesia &
# Analgesia 126(5)) — |r| bands: negligible/weak/moderate/strong/very-strong.
# Versioned threshold table for the consolidated-report insights.
SCHOBER_CUTOFFS = (0.09, 0.39, 0.69, 0.89)

# Causality caveat appended to every correlation/feature-score (Schober 2018
# recommendation). ASCII-only (cp1252-safe, matches oracle tokens).
CAUSALITY_CAVEAT = "correlacao nao implica causalidade"

__all__ = [
    "CAUSALITY_CAVEAT",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_N_RESAMPLES",
    "SCHOBER_CUTOFFS",
    "SMALL_N_THRESHOLD",
    "BootstrapResult",
    "DistFit",
    "HypothesisResult",
    "OutlierResult",
    "PreCheck",
    "RelationshipResult",
    "analyze_relationship",
    "bin_factor_levels",
    "bootstrap_effect",
    "correlation_strength",
    "detect_outliers",
    "fit_distribution",
    "kgroup_hypothesis_test",
    "paired_hypothesis_test",
]


@dataclass(frozen=True)
class RelationshipResult:
    """Pure relationship analysis between two columns.

    Mirrors ``asapy.analysis.analyze_relationship`` primitives (Pearson via
    ``pandas``, Spearman/Kendall via ``scipy``, OLS via ``sklearn``) but is
    **pure** (returns coefficients; the figures live in ``report.py``).
    """

    col_x: str
    col_y: str
    n: int
    pearson: float | None
    spearman: float | None
    kendall: float | None
    slope: float | None
    intercept: float | None
    r2: float | None
    mse: float | None
    residuals: tuple[float, ...]
    strength: str  # Schober band of |pearson|
    caveat: str  # "correlacao nao implica causalidade"


@dataclass(frozen=True)
class PreCheck:
    """A single pre-check: shapiro (normality) or levene (variance)."""

    name: str
    statistic: float
    p_value: float
    passed: bool  # normality: shapiro p > alpha; homogeneity: levene p > alpha


@dataclass(frozen=True)
class BootstrapResult:
    """Non-parametric bootstrap over paired differences."""

    effect: float
    ci_low: float
    ci_high: float
    ci_method: str  # "BCa" | "percentile"
    p_value: float  # recentered-under-H0 ASL (Efron & Tibshirani ch. 16)
    n_resamples: int
    bootstrap_seed: int


@dataclass(frozen=True)
class HypothesisResult:
    """Auto-selected hypothesis test result."""

    test: str
    statistic: float
    p_value: float
    reject_h0: bool
    pre_checks: tuple[PreCheck, ...]
    effect: float | None = None
    bootstrap: BootstrapResult | None = None
    posthoc: dict[str, Any] | None = None


@dataclass(frozen=True)
class DistFit:
    """One distribution-family fit by Kolmogorov-Smirnov."""

    name: str
    ks_statistic: float
    p_value: float
    params: tuple[float, ...]


@dataclass(frozen=True)
class OutlierResult:
    """Outliers + per-column thresholds."""

    method: str
    indices: tuple[int, ...]
    values: tuple[float, ...]
    lower: float | None
    upper: float | None


def _shapiro_check(sample: np.ndarray, alpha: float) -> PreCheck:
    if sample.size < 3:
        return PreCheck("shapiro", float("nan"), float("nan"), passed=False)
    res = stats.shapiro(sample)
    return PreCheck("shapiro", float(res.statistic), float(res.pvalue),
                    passed=bool(res.pvalue > alpha))


def _levene_check(groups: Sequence[np.ndarray], alpha: float) -> PreCheck:
    res = stats.levene(*groups)
    return PreCheck("levene", float(res.statistic), float(res.pvalue),
                    passed=bool(res.pvalue > alpha))


def paired_hypothesis_test(
    baseline: Sequence[float],
    modernized: Sequence[float],
    *,
    alpha: float = 0.05,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
) -> HypothesisResult:
    """Central paired test on Delta = modernized - baseline."""
    bsl = np.asarray(baseline, dtype=float)
    mod = np.asarray(modernized, dtype=float)
    if bsl.shape != mod.shape or bsl.ndim != 1 or bsl.size == 0:
        raise ValueError("baseline and modernized must be non-empty 1-D arrays of equal length")
    delta = mod - bsl

    shapiro = _shapiro_check(delta, alpha)
    levene = _levene_check((bsl, mod), alpha)
    pre_checks = (shapiro, levene)

    if shapiro.passed:
        res = stats.ttest_rel(mod, bsl)
        test, statistic, p_value = "ttest_rel", float(res.statistic), float(res.pvalue)
    elif np.allclose(delta, 0.0):
        test, statistic, p_value = "wilcoxon", 0.0, 1.0
    else:
        res = stats.wilcoxon(delta)
        test, statistic, p_value = "wilcoxon", float(res.statistic), float(res.pvalue)

    boot = bootstrap_effect(delta, bootstrap_seed=bootstrap_seed, n_resamples=n_resamples)
    return HypothesisResult(
        test=test, statistic=statistic, p_value=p_value,
        reject_h0=bool(p_value <= alpha), pre_checks=pre_checks,
        effect=float(np.mean(delta)), bootstrap=boot,
    )


def _welch_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    """Welch's ANOVA (Welch 1951) F statistic + p-value — scipy has no direct call."""
    k = len(groups)
    n = np.array([g.size for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups], dtype=float)
    var = np.array([g.var(ddof=1) for g in groups], dtype=float)
    w = n / var
    w_sum = w.sum()
    grand = float((w * means).sum() / w_sum)
    num = float((w * (means - grand) ** 2).sum()) / (k - 1)
    tail = float((((1.0 - w / w_sum) ** 2) / (n - 1)).sum())
    denom = 1.0 + (2.0 * (k - 2) / (k**2 - 1)) * tail
    f_stat = num / denom
    df1 = k - 1
    df2 = (k**2 - 1) / (3.0 * tail)
    p_value = float(stats.f.sf(f_stat, df1, df2))
    return f_stat, p_value


def kgroup_hypothesis_test(
    groups: Sequence[Sequence[float]],
    *,
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> HypothesisResult:
    """k-group test: ANOVA+Tukey / Welch+Tamhane T2 / Kruskal+Conover."""
    arrs = [np.asarray(g, dtype=float) for g in groups]
    if len(arrs) < 2:
        raise ValueError("k-group test requires at least 2 groups")

    shapiros = tuple(_shapiro_check(g, alpha) for g in arrs)
    levene = _levene_check(arrs, alpha)
    pre_checks = (*shapiros, levene)
    normal = all(pc.passed for pc in shapiros)
    homogeneous = levene.passed

    posthoc: dict[str, Any] | None = None
    if normal and homogeneous:
        res = stats.f_oneway(*arrs)
        test, statistic, p_value = "f_oneway", float(res.statistic), float(res.pvalue)
        if p_value <= alpha:
            posthoc = {"method": "tukey_hsd",
                       "pvalue": stats.tukey_hsd(*arrs).pvalue.tolist()}
    elif normal and not homogeneous:
        statistic, p_value = _welch_anova(arrs)
        test = "welch_anova"
        if p_value <= alpha:
            import scikit_posthocs as sp

            posthoc = {"method": "tamhane",
                       "pvalue": sp.posthoc_tamhane([g.tolist() for g in arrs]).values.tolist()}
    else:
        res = stats.kruskal(*arrs)
        test, statistic, p_value = "kruskal", float(res.statistic), float(res.pvalue)
        if p_value <= alpha:
            import scikit_posthocs as sp

            posthoc = {"method": "conover",
                       "pvalue": sp.posthoc_conover([g.tolist() for g in arrs]).values.tolist()}

    return HypothesisResult(
        test=test, statistic=statistic, p_value=p_value,
        reject_h0=bool(p_value <= alpha), pre_checks=pre_checks, posthoc=posthoc,
    )


def bootstrap_effect(
    delta: Sequence[float],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = 0.95,
    small_n_threshold: int = SMALL_N_THRESHOLD,
) -> BootstrapResult:
    """Bootstrap effect + BCa/percentile CI + recentered ASL p-value.

    Operates on the paired differences ``delta`` (pairing preserved). BCa CI for
    n >= ``small_n_threshold`` else percentile (BCa undercovers at small n). The
    p-value is the recentered-under-H0 achieved significance level (Efron &
    Tibshirani 1993, ch. 16). Deterministic given ``bootstrap_seed``.
    """
    arr = np.asarray(delta, dtype=float)
    n = arr.size
    if n == 0:
        raise ValueError("delta must be non-empty")
    effect = float(arr.mean())

    if np.allclose(arr, 0.0):  # no difference at all
        return BootstrapResult(0.0, 0.0, 0.0, "percentile", 1.0, n_resamples, bootstrap_seed)

    method = "BCa" if n >= small_n_threshold else "percentile"
    ci = stats.bootstrap(
        (arr,), np.mean, n_resamples=n_resamples, confidence_level=confidence_level,
        method=method, rng=np.random.default_rng(bootstrap_seed),
    ).confidence_interval
    ci_low, ci_high = float(ci.low), float(ci.high)

    # Recentered-under-H0 ASL (Efron & Tibshirani ch. 16): resample the shifted
    # differences (mean 0) and measure how often |mean*| >= |observed effect|.
    centered = arr - effect
    rng_asl = np.random.default_rng(bootstrap_seed + 1)
    idx = rng_asl.integers(0, n, size=(n_resamples, n))
    boot_means = centered[idx].mean(axis=1)
    p_value = float(np.mean(np.abs(boot_means) >= abs(effect)))

    return BootstrapResult(effect, ci_low, ci_high, method, p_value, n_resamples, bootstrap_seed)


def fit_distribution(
    values: Sequence[float],
    *,
    candidates: Sequence[str] = ("norm", "uniform", "expon", "chi2", "beta"),
) -> list[DistFit]:
    """KS best-fit over the 5 common families, best-first."""
    arr = np.asarray(values, dtype=float)
    fits: list[DistFit] = []
    for name in candidates:
        dist = getattr(stats, name)
        try:
            params = dist.fit(arr)
            ks = stats.kstest(arr, name, args=params)
        except Exception:  # a family that cannot fit the data is skipped
            continue
        fits.append(DistFit(name, float(ks.statistic), float(ks.pvalue),
                            tuple(float(p) for p in params)))
    fits.sort(key=lambda f: f.p_value, reverse=True)  # best-first
    return fits


def detect_outliers(
    values: Sequence[float],
    *,
    method: str = "IQR",
    thr: float = 3.0,
) -> OutlierResult:
    """IQR (1.5*IQR fences) or Z-score (|z| > thr) outliers."""
    arr = np.asarray(values, dtype=float)
    if method == "IQR":
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower: float | None = q1 - 1.5 * iqr
        upper: float | None = q3 + 1.5 * iqr
        mask = (arr < lower) | (arr > upper)
    elif method == "zscore":
        std = arr.std()
        z = np.zeros_like(arr) if std == 0 else (arr - arr.mean()) / std
        mask = np.abs(z) > thr
        lower = upper = None
    else:
        raise ValueError("method must be 'IQR' or 'zscore'")

    idx = np.nonzero(mask)[0]
    return OutlierResult(
        method=method,
        indices=tuple(int(i) for i in idx),
        values=tuple(float(v) for v in arr[idx]),
        lower=lower,
        upper=upper,
    )


def correlation_strength(r: float | None) -> str:
    """Schober et al. (2018) |r| band label. Pure; ASCII.

    Bands by ``SCHOBER_CUTOFFS`` (0.09/0.39/0.69/0.89): negligible / weak /
    moderate / strong / very-strong. ``None``/NaN -> "indeterminada".
    """
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "indeterminada"
    a = abs(float(r))
    negligible, weak, moderate, strong = SCHOBER_CUTOFFS
    if a <= negligible:
        return "negligivel"
    if a <= weak:
        return "fraca"
    if a <= moderate:
        return "moderada"
    if a <= strong:
        return "forte"
    return "muito forte"


def analyze_relationship(
    df: pd.DataFrame, col_x: str, col_y: str
) -> RelationshipResult:
    """Correlation + OLS between two columns. Pure (no I/O/figures).

    Mirrors ``asapy.analysis.analyze_relationship`` primitives: Pearson
    (``pandas``), Spearman/Kendall (``scipy``), OLS (``sklearn.LinearRegression``
    -> slope/intercept/r2/mse) + residuals + Schober strength label. Degenerate
    (n<2 or zero variance) -> coefficients ``None`` (edge case, no crash).
    """
    x = np.asarray(df[col_x], dtype=float)
    y = np.asarray(df[col_y], dtype=float)
    n = int(min(x.size, y.size))
    if n < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return RelationshipResult(
            col_x, col_y, n, None, None, None, None, None, None, None, (),
            correlation_strength(None), CAUSALITY_CAVEAT,
        )

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    pearson = float(pd.Series(x).corr(pd.Series(y)))
    spearman = float(stats.spearmanr(x, y).statistic)
    kendall = float(stats.kendalltau(x, y).statistic)

    xm = x.reshape(-1, 1)
    model = LinearRegression().fit(xm, y)
    y_pred = model.predict(xm)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(r2_score(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))
    residuals = tuple(float(v) for v in (y - y_pred))
    return RelationshipResult(
        col_x, col_y, n, pearson, spearman, kendall, slope, intercept, r2, mse,
        residuals, correlation_strength(pearson), CAUSALITY_CAVEAT,
    )


def bin_factor_levels(
    table: pd.DataFrame,
    factor: str,
    *,
    n_levels: int = 3,
    method: str = "quantile",
) -> list[str]:
    """Bin a continuous factor into ordered level labels. Pure.

    ``method="quantile"`` (default) uses equal-frequency terciles (``pd.qcut``);
    ``method="uniform"`` uses equal-width bins (``pd.cut``). For ``n_levels==3``
    the labels are ``low``/``mid``/``high``; falls back to equal-width if the
    quantile edges collapse (ties). Returns one label per row.
    """
    col = np.asarray(table[factor], dtype=float)
    labels = ["low", "mid", "high"] if n_levels == 3 else [f"L{i}" for i in range(n_levels)]
    if method not in ("quantile", "uniform"):
        raise ValueError("method must be 'quantile' or 'uniform'")

    cats: Any
    if method == "quantile":
        try:
            cats = pd.qcut(col, q=n_levels, labels=labels)
        except ValueError:  # collapsed quantile edges (heavy ties) -> equal width
            cats = pd.cut(col, bins=n_levels, labels=labels, include_lowest=True)
    else:
        cats = pd.cut(col, bins=n_levels, labels=labels, include_lowest=True)
    return [str(c) for c in cats]
