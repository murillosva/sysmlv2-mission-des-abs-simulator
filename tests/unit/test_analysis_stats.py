"""Known-answer tests for the AN-2 statistics module (Spec 027 §5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from cuas_sim.analysis.stats import (
    detect_outliers,
    fit_distribution,
    kgroup_hypothesis_test,
    paired_hypothesis_test,
)


def _shapiro(pre_checks) -> object:
    return next(pc for pc in pre_checks if pc.name == "shapiro")


def test_paired_normal_ttest_rel() -> None:
    rng = np.random.default_rng(0)
    n = 40
    baseline = rng.normal(0.10, 0.02, n)
    modernized = baseline + 0.20 + rng.normal(0.0, 0.005, n)  # Delta ~ normal, >> 0

    res = paired_hypothesis_test(baseline, modernized, alpha=0.05, bootstrap_seed=123)

    assert res.test == "ttest_rel"
    assert res.reject_h0 is True
    assert res.effect == pytest.approx(float(np.mean(modernized - baseline)), abs=1e-9)
    assert _shapiro(res.pre_checks).passed  # differences are normal


def test_paired_nonnormal_wilcoxon() -> None:
    rng = np.random.default_rng(1)
    n = 50
    baseline = rng.normal(0.10, 0.02, n)
    delta = rng.exponential(0.05, n)  # strongly skewed, non-normal, all > 0
    modernized = baseline + delta

    res = paired_hypothesis_test(baseline, modernized, alpha=0.05, bootstrap_seed=123)

    assert res.test == "wilcoxon"
    assert not _shapiro(res.pre_checks).passed  # differences are non-normal


def test_kgroup_autoselect() -> None:
    rng = np.random.default_rng(2)
    g1 = rng.normal(0.0, 1.0, 30)
    g2 = rng.normal(3.0, 1.0, 30)
    g3 = rng.normal(6.0, 1.0, 30)

    res = kgroup_hypothesis_test([g1, g2, g3], alpha=0.05)
    assert res.test in ("f_oneway", "welch_anova", "kruskal")
    assert res.reject_h0 is True

    same = rng.normal(0.0, 1.0, 30)
    res2 = kgroup_hypothesis_test([same, same.copy(), same.copy()], alpha=0.05)
    assert res2.reject_h0 is False


def test_fit_distribution_uniform() -> None:
    rng = np.random.default_rng(3)
    data = rng.uniform(0.0, 1.0, 2000)

    fits = fit_distribution(data)

    names = [f.name for f in fits]
    assert "uniform" in names
    # ordered best-first
    assert fits[0].p_value >= fits[-1].p_value
    # beta(1,1) == uniform, so beta (a strict generalization) may tie/beat uniform;
    # the meaningful claim is that uniform is a top-2, not-rejected fit and ranks
    # above the families that cannot fit bounded uniform data (norm/expon/chi2).
    rank = {f.name: i for i, f in enumerate(fits)}  # 0 = best
    unif = next(f for f in fits if f.name == "uniform")
    assert rank["uniform"] <= 1
    assert unif.p_value > 0.05  # uniform is not rejected (it is the true law)
    assert rank["uniform"] < rank["norm"]
    assert rank["uniform"] < rank["expon"]
    assert rank["uniform"] < rank["chi2"]


def test_detect_outliers_iqr() -> None:
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 100.0]
    res = detect_outliers(values, method="IQR")

    assert 100.0 in res.values
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    assert res.lower == pytest.approx(q1 - 1.5 * iqr)
    assert res.upper == pytest.approx(q3 + 1.5 * iqr)
