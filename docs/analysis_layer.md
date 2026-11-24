# Statistical Analysis Layer

The analysis layer (`src/cuas_sim/analysis/`) designs the experiments, executes the paired
replications and analyses their outputs. This document states what each part does, which
artifacts it writes and which rules it applies. How the four studies of the paper used it is
in [`experimental_campaign.md`](experimental_campaign.md).

---

## 1. Scope

- **Three steps.** The layer reimplements the first three of the four steps proposed by
  Dantas et al. (2024) for the AsaPy library: design of experiments, execution control and
  analysis. A consolidated report gathers the analysis of all responses. The fourth step,
  prediction (`cuas-sim analysis predict`), is an exploratory study that was not concluded;
  it is not part of the paper, and its outputs are not published.
- **Read-only over the measures.** The layer consumes the MOS/MOE/MOP values computed by
  `MetricsAggregator` ([`metrics.md`](metrics.md)) and never recomputes them. It does not
  touch the simulation engine, so the [regression ruler](reproducibility.md#regression-ruler-about-5-seconds)
  is the same with or without it.
- **AsaPy as a reference, not a dependency.** The methodology is reimplemented on `scipy`,
  `scikit-learn` and `scikit-posthocs` (extra `[analysis]` of `pyproject.toml`). AsaPy itself
  cannot be installed next to the simulator: its declared dependencies (for example
  `pandas==1.5.3` and `tensorflow==2.12.0`) conflict with the simulator's.

---

## 2. Modules, commands and artifacts

| Step | Modules | Command | Artifact |
|---|---|---|---|
| 1. Design of experiments | `doe.py` (`Factor`, `create_design`: Latin hypercube from `scipy.stats.qmc`) | `cuas-sim analysis design`; `scripts/make_study_design.py` for fixed factors and one-factor designs | `design.parquet` + `.meta.json` (design seed, sampler, factor ranges) |
| 2. Execution control | `experiment.py` (`run_data_farm`: paired replications under common random numbers, one process per design point, canonical output order); `execution.py` (`ExecutionController`, optional early stop, off by default) | `cuas-sim analysis run` | `datafarm_*.parquet` + `.meta.json` (base seed, replications, design reference) |
| Stopping and off-path treatments | `scripts/campaign_postprocess.py` (seed nesting, stopping rule per design point, MOS-2, cost blocks) | — | `offpath_*.json` |
| 3. Analysis, per response | `stats.py`, `features.py`, `report.py` | `cuas-sim analysis stats` | `stats_*/<response>/report.json` |
| Consolidated report | `report.py` (`build_consolidated_report`), `reporting.py` (HTML rendering) | `cuas-sim analysis report` | `report_*/report.json` |
| Interaction surface (EX-4) | `scripts/surface_model.py` | — | `surface_*.json` |
| 4. Prediction (exploratory) | `prediction.py` | `cuas-sim analysis predict` | not published |

The output of step 2 is a *data farm* (Horne and Seichter, 2014): the table of every measure
of every paired replication at every design point, in long format, with the columns
`design_point_id`, `lambda_per_h`, `uas_archetype_p_mr`, `rep`, `seed`, `config`, `aor`,
`metric` and `value`. `config` is `baseline`, `modernized` or `delta` (alternative minus
baseline in the same replication); `aor` is an area of responsibility or `ALL`.

`analysis run` takes `--config` and `--sysml` for the base configuration; a study to be
compared with the paper passes `--config configs/zida_shield_default.yaml --sysml sysml`.
Without `--max-arrivals`, each replication ends with the spawn window; capping arrivals would
flatten the effect of λ. The exact commands of each study are in its `MANIFEST.md`.

---

## 3. Factors and responses

| Factor | Column in the data farm | Field of `MissionConfig` | Nominal range |
|---|---|---|---|
| Arrival rate of incursions, λ | `lambda_per_h` (UAS/h) | `lambda_per_sec` = λ / 3600 | 2 to 18 UAS/h |
| Multi-rotor fraction, p_mr | `uas_archetype_p_mr` | `uas_archetype_p_mr` | 0 to 1 |

Each design point runs with `cfg.replaced(lambda_per_sec=…, uas_archetype_p_mr=…)` over the
base configuration. The nominal point is λ = 6 UAS/h and p_mr = 0.5, the values of the
scenario file.

**Joint variation.** The layer was designed to vary the two factors together in a Latin
hypercube, because a one-factor-at-a-time design is blind to their interaction: the points
of two one-factor lines that cross at the nominal point cannot identify the coefficient of
the product λ·p_mr, whatever the number of replications. The campaign nevertheless ran two
one-factor studies (EX-2 in λ, EX-3 in p_mr) for clarity of attribution, reporting marginal
effects with the other factor at its nominal value, and recovered the interaction with the
two-dimensional design of EX-4.

**Responses.**

- Paired differences Δ per replication; ΔMOS-1 is the primary response of every study.
- Absolute measures per configuration (MOS-1, MOE-1 to MOE-3, MOP-1 to MOP-3).
- MOS-2, which compares the two configurations, is recorded with `config` =
  `modernized_vs_baseline` and has no `delta` row: `analysis report` and
  `campaign_postprocess.py` treat it; `analysis stats --response MOS-2` does not.
- Cost: rockets per engagement and leaks by reason (`leak_winchester`, `leak_bingo-fuel`,
  `leak_pending-evaded`).

Rows of `leak_*` are written only when a leak occurs, so an absent row means zero. A replica
without engagements has no `rockets_per_engagement` row; its mean is undefined, not zero, and
the number of replications with engagements is reported. MOP-2 and MOP-3 are constants of each
configuration and are excluded from the tests.

---

## 4. Tests, intervals and decision

**Paired comparison** (`stats.paired_hypothesis_test`). A Shapiro–Wilk test on the paired
differences selects the paired t test (`ttest_rel`) or the Wilcoxon signed-rank test
(`wilcoxon`); a Levene test is recorded alongside. The common random numbers make the two
configurations of a replication face the same incursions, so the paired difference isolates
the configuration as the only treatment.

**Several groups** (`stats.kgroup_hypothesis_test`), for factor levels binned into terciles
(`stats.bin_factor_levels`). Shapiro–Wilk per group and Levene select one of three paths:

| Pre-checks | Test | Post hoc |
|---|---|---|
| normal, homogeneous variances | one-way ANOVA (`f_oneway`) | Tukey HSD |
| normal, heterogeneous variances | Welch ANOVA | Tamhane T2 |
| not normal | Kruskal–Wallis | Conover |

**Interval and p-value** (`stats.bootstrap_effect`). The mean paired difference always
accompanies the p-value, with a 95% BCa bootstrap interval from 9,999 resamples of the
differences (percentile interval below 30 pairs) and an achieved significance level (ASL)
computed on the differences recentred under the null hypothesis (Efron and Tibshirani, 1993,
ch. 16). Resampling the paired differences, not the two samples pooled, preserves the pairing.
The bootstrap seed is recorded in the artifact.

**Decision rule.** At α = 0.05, a difference is declared only when the test rejects the null
hypothesis and the interval excludes zero. Non-rejection is a statement of bound, not of
null effect or equivalence.

**Relationships and factor ranking.** `stats.analyze_relationship` gives the Pearson,
Spearman and Kendall coefficients and an ordinary least-squares fit of a response on a
factor, with the strength label of Schober et al. (2018); every correlation carries the
caveat that it does not imply causation. `features.py` ranks the factors (univariate
F test, `f_regression`, by default) and computes the Pareto front of gain against cost, over
the means per design point.

**Hand-written procedures.** Two numerical procedures are written in the layer and not
delegated to a library: the Welch ANOVA and the recentred ASL. Both are checked against
independent references in `tests/unit/test_analysis_stats_handwritten.py` (the identity
F = t² for two groups, a closed-form value for three groups, `f_oneway`, and a paired
permutation test).

---

## 5. Consolidated report

`cuas-sim analysis report` writes `report.json` (`field_version` 1.0, `study.kind` =
`consolidated`) from a single pass over the responses and areas of responsibility:

- the master table (each response × area: mean Δ, interval, p-value, test);
- the hypothesis, relationship and per-level blocks;
- insights produced by fixed rules, with versioned thresholds recorded in `thresholds`:

| Rule | Threshold | Source |
|---|---|---|
| Correlation strength | \|r\| cut-offs 0.09, 0.39, 0.69, 0.89 | Schober et al. (2018) |
| Significance | p ≤ α and interval excluding zero | Bussab and Morettin (2010) |
| Evidence gate: design points | fewer than 10 per varied factor | Loeppky et al. (2009) |
| Evidence gate: pairs | fewer than 34 (power 0.80 at α = 0.05) | Cohen (1988) |
| Evidence gate: surrogate | Q² below 0.5 | Golbraikh and Tropsha (2002) |

- a decision index, which is a presentation composite and not a measure of the paper: it
  ranks the areas of responsibility and pools the design points inside each area;
- a validity block with the sampled factor ranges, the seeds and two caveats: the results
  do not generalize beyond the sampled space, and correlation does not imply causation.

The HTML dashboard and the Markdown version are renderings of `report.json` and add no
number; they are not published. The automatic labels of the published reports must be read
with the caveats of [`experimental_campaign.md`](experimental_campaign.md#5-reading-the-published-reports).

---

## 6. Reproducibility

- Four seeds steer the layer, and each is recorded in the artifacts: the base seed, from
  which the seed banks of the replications are derived, and three seeds outside the
  simulator's seed registry, which change no replication: the design seed (Latin
  hypercube), the bootstrap seed and the model seed (exploratory surrogate).
- Replications of a design point draw from disjoint, nested seed banks: the bank of R
  replications is the first R seeds of the bank of any larger R, so a larger rung extends
  the previous batch.
- The data farm is bit-identical for any number of workers, and the JSON artifacts are
  written with sorted keys and without timestamps: the same seeds give byte-identical files.

---

## 7. AsaPy oracle tests

`tests/oracle/` holds five tests, skipped when `asapy` is not importable; AsaPy can run only in
a separate environment (§1). Two of them compare this layer with AsaPy itself (feature-score
ranking and Pareto set, as does `scripts/asapy_oracle_check.py`); the other three compare the
surrogate and the relationship analysis with reference calls of `scikit-learn`, `scipy` and
`pandas`. No record of a run against AsaPy is published, so the equivalence is verifiable by
these tests, not verified. The hypothesis tests and the bootstrap are not covered by the
oracle; they are covered by the tests cited in §4.

---

## Cross-references

- How the four studies used the layer: [`experimental_campaign.md`](experimental_campaign.md).
- Measures consumed by the layer: [`metrics.md`](metrics.md).
- Commands, byte identity and stopping criteria: [`reproducibility.md`](reproducibility.md).
- Bibliography: [`refs.bib`](refs.bib).
