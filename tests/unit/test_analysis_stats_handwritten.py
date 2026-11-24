"""Known-answer tests for the hand-written numeric procedures of AN-2 (Spec 027).

Everything else in ``analysis/stats.py`` delegates to scipy / scikit-learn /
scikit-posthocs, so there is no transcription to get wrong -- a test there would
be comparing scipy against scipy. Exactly two procedures are written out by hand
and therefore carry real transcription risk:

* ``_welch_anova`` -- Welch (1951) F, coded in numpy because scipy has no direct call;
* the recentered-under-H0 ASL loop inside ``bootstrap_effect``
  (Efron & Tibshirani 1993, *An Introduction to the Bootstrap*, ch. 16).

Neither is covered by the AsaPy oracle (``tests/oracle/``): that oracle compares
feature-score ranking and the Pareto set, and the bootstrap is a *deliberate*
divergence from AsaPy's pooled-shuffle permutation, so an AsaPy gabarito would be
vacuous there by construction. The tests below pin both against independent
references -- ``scipy.stats.ttest_ind(equal_var=False)``, ``scipy.stats.f_oneway``,
``scipy.stats.f.sf`` and ``scipy.stats.permutation_test``.

``_welch_anova`` is imported privately on purpose: the public
``kgroup_hypothesis_test`` only routes to it when shapiro passes and levene fails,
which the exact two-level fixtures used here cannot satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from cuas_sim.analysis.stats import _welch_anova, bootstrap_effect


def _exact_group(n: int, mean: float, var: float) -> np.ndarray:
    """A two-level sample whose sample mean and ddof=1 variance are exact.

    Half the mass at ``mean - d`` and half at ``mean + d``, with
    ``d = sqrt(var * (n - 1) / n)``, reproduces ``(mean, var)`` with no sampling
    error. The expected F is then a closed-form constant derived from Welch's
    formula, not a frozen output of the code under test.
    """
    assert n % 2 == 0
    d = float(np.sqrt(var * (n - 1) / n))
    return np.concatenate([np.full(n // 2, mean - d), np.full(n // 2, mean + d)])


def test_welch_anova_equals_welch_t_squared_at_two_groups() -> None:
    #
    # With k = 2, Welch's ANOVA is mathematically identical to Welch's t-test
    # squared. ``scipy.stats.ttest_ind(equal_var=False)`` is a fully independent
    # implementation of that procedure, so the identity is a genuine known answer:
    # it pins the weights, the weighted grand mean, the numerator, and the
    # Welch-Satterthwaite denominator degrees of freedom that drive the p-value.
    rng = np.random.default_rng(11)
    cases = [(30, 30, 1.0, 1.0), (12, 40, 1.0, 3.0), (8, 25, 2.5, 0.7)]

    for n_a, n_b, sd_a, sd_b in cases:
        a = rng.normal(0.0, sd_a, n_a)
        b = rng.normal(0.6, sd_b, n_b)

        f_stat, p_value = _welch_anova([a, b])
        ref = scipy_stats.ttest_ind(a, b, equal_var=False)

        assert f_stat == pytest.approx(float(ref.statistic) ** 2, rel=1e-12)
        assert p_value == pytest.approx(float(ref.pvalue), rel=1e-12)


def test_welch_anova_known_answer_three_groups() -> None:
    #
    # The k = 2 identity above leaves the small-sample correction untested, because
    # its factor 2(k-2)/(k^2-1) vanishes at k = 2. This case exercises it at k = 3.
    #
    # Groups with exact (n, mean, var) = (10, 0, 1), (10, 1, 4), (10, 2, 25), so
    # Welch (1951) can be evaluated by hand:
    #     w     = n/s^2             = (10, 2.5, 0.4);   W = 12.9
    #     x_w   = sum(w*xbar)/W     = 3.3/12.9          = 0.2558139534883721
    #     A     = sum(w*(xbar-x_w)^2)/(k-1)             = 1.6279069767441861
    #     L     = sum((1 - w/W)^2/(n-1))                = 0.1821600683558122
    #     denom = 1 + 2(k-2)/(k^2-1) * L                = 1.0455400170889530
    #     F     = A/denom                               = 1.5570009483394478
    #     df2   = (k^2-1)/(3*L)                         = 14.639102705080274
    groups = [
        _exact_group(10, 0.0, 1.0),
        _exact_group(10, 1.0, 4.0),
        _exact_group(10, 2.0, 25.0),
    ]
    for g, (mean, var) in zip(groups, [(0.0, 1.0), (1.0, 4.0), (2.0, 25.0)], strict=True):
        assert float(g.mean()) == pytest.approx(mean, abs=1e-12)
        assert float(g.var(ddof=1)) == pytest.approx(var, abs=1e-12)

    f_stat, p_value = _welch_anova(groups)

    assert f_stat == pytest.approx(1.5570009483394478, rel=1e-12)
    assert p_value == pytest.approx(0.24373362384642355, rel=1e-12)

    # The tail must be taken on F(df1 = k-1, df2 = Welch-Satterthwaite), never on the
    # classic F(k-1, N-k) = F(2, 27). scipy supplies the tail independently.
    assert p_value == pytest.approx(
        float(scipy_stats.f.sf(f_stat, 2, 14.639102705080274)), rel=1e-12
    )
    assert p_value != pytest.approx(float(scipy_stats.f.sf(f_stat, 2, 27)), rel=1e-3)


def test_welch_anova_correction_decays_toward_classic_f() -> None:
    #
    # Under equal variances and equal group sizes, Welch's F is the classic F divided
    # by the small-sample correction, which tends to 1 as n grows. Anchoring the ratio
    # on ``scipy.stats.f_oneway`` pins the correction's direction and magnitude: it
    # must shrink F (ratio < 1) and must relax monotonically with n.
    ratios = []
    for n in (10, 100, 1000):
        groups = [
            _exact_group(n, 0.0, 4.0),
            _exact_group(n, 1.0, 4.0),
            _exact_group(n, 2.0, 4.0),
        ]
        f_welch, _ = _welch_anova(groups)
        f_classic = float(scipy_stats.f_oneway(*groups).statistic)
        ratios.append(f_welch / f_classic)

    assert all(r < 1.0 for r in ratios)
    assert ratios == sorted(ratios)  # correction relaxes monotonically with n
    assert ratios[0] == pytest.approx(0.9642857142857143, rel=1e-12)  # 1/(1 + 0.25*4/27)
    assert ratios[-1] == pytest.approx(1.0, abs=1e-3)


def _paired_permutation_p(delta: np.ndarray, seed: int) -> float:
    """Two-sided paired permutation p-value (sign flips), via scipy."""
    res = scipy_stats.permutation_test(
        (delta,),
        lambda d, axis=-1: np.mean(d, axis=axis),
        permutation_type="samples",
        alternative="two-sided",
        n_resamples=9999,
        rng=np.random.default_rng(seed),
    )
    return float(res.pvalue)


def test_bootstrap_asl_matches_paired_permutation_test() -> None:
    #
    # The recentered ASL is hand-written. Its independent reference is the paired
    # permutation test (``permutation_type="samples"`` = sign flips), which attacks
    # the same null by a different mechanism. Measured agreement over this grid is
    # <= 0.008; the tolerance sits well above that so the test is not flaky.
    cases = [(60, 0.30), (60, 0.15), (100, 0.10), (200, 0.05), (60, 0.0)]

    for n, mu in cases:
        rng = np.random.default_rng(2026)
        delta = rng.normal(mu, 1.0, n)

        asl = bootstrap_effect(delta, bootstrap_seed=99, n_resamples=9999).p_value
        assert asl == pytest.approx(_paired_permutation_p(delta, 99), abs=0.02)


def test_bootstrap_asl_is_recentered_under_h0() -> None:
    #
    # Discriminating test for the recentering step. Resampling the *observed*
    # differences instead of the H0-shifted ones makes the bootstrap means cluster
    # around the observed effect, so |mean*| >= |effect| roughly half the time and
    # the ASL collapses to ~0.5 for ANY data -- a plausible-looking number that is
    # not a p-value. A correct ASL is near 0 for a large, unambiguous effect.
    rng = np.random.default_rng(4242)
    delta = rng.normal(0.50, 0.05, 80)  # effect sits ~10 sd away from zero

    res = bootstrap_effect(delta, bootstrap_seed=7, n_resamples=9999)

    assert res.effect == pytest.approx(float(np.mean(delta)), rel=1e-12)
    assert res.p_value < 0.01  # a non-recentered implementation returns ~0.5 here

    # Deterministic given the seed (Spec 027 NFR-1).
    again = bootstrap_effect(delta, bootstrap_seed=7, n_resamples=9999)
    assert again.p_value == res.p_value
    assert (again.ci_low, again.ci_high) == (res.ci_low, res.ci_high)
