# Experimental Campaign

Condensed description of the simulation campaign reported in the accompanying paper
(Sections 3 and 4). It states how the four studies were designed, executed, stopped and
analysed, and how to read the published study artifacts.

The study folders under [`../outputs/analysis/`](../outputs/analysis/) hold the record of
each study: `BRIEF.md` and `MANIFEST.md` and, for EX-2 to EX-4, the pre-registered
`STOPPING_CRITERION.md`, published unchanged, in Portuguese. Section numbers cited in those
criterion files and in some recorded fields of the JSON artifacts (for example "sec.7.4" or
"§4.3") refer to the working version of this document used during the studies, which is not
part of this release.

---

## 1. Purpose

The campaign measures the paired gain of the alternative configuration (A-29N) over the
baseline (A-29) in the counter-UAS mission case. The primary response is ΔMOS-1, the paired
difference in the fraction of incursions neutralized before crossing the 80-nautical-mile
boundary of the ZIDA. The studies progress from one operating point to the effect of each
threat factor and then to their interaction:

- the arrival rate of incursions, λ, nominally 2 to 18 UAS/h;
- the multi-rotor fraction, p_mr, nominally 0 to 1.

---

## 2. The four studies

| Study | Folder | Design | Points | Replications per point | Pairs | Realized range | Central result (as in the paper) |
|---|---|---|---|---|---|---|---|
| EX-1, nominal point | `ex1_mc_nominal` | Monte Carlo; λ = 6.0 UAS/h and p_mr = 0.5 fixed | 1 | 200 | 200 | — | ΔMOS-1 = +0.1853 [+0.1683; +0.2031] |
| EX-2, intensity sweep | `ex2_lambda` | one-factor Latin hypercube in λ; p_mr = 0.5 fixed | 16 | 96 | 1,536 | λ ∈ [2.99; 17.54] | slope −0.00638 per UAS/h; gain varies 1.74× |
| EX-3, composition sweep | `ex3_mix` | one-factor Latin hypercube in p_mr; λ = 6.0 fixed | 20 | 192 | 3,840 | p_mr ∈ [0.0159; 0.9825] | slope +0.4393 per unit of p_mr; gain varies 29.4× |
| EX-4, two-dimensional design | `ex4_interaction` | Latin hypercube in λ and p_mr | 32 | 384 | 12,288 | λ ∈ [2.364; 17.571], p_mr ∈ [0.0017; 0.9851] | interaction −0.015027; additivity rejected, F(1,28) = 66.90 |
| Total | | | 69 | | 17,864 | | |

Each reported Δ is a mean over the replications of a design point, with a 95% BCa interval.

- **Configuration.** Every study runs with the scenario file and the model
  (`--config configs/zida_shield_default.yaml --sysml sysml`) and without capping arrivals:
  each replication ends with the spawn window.
- **Realized ranges.** The Latin hypercube samples inside each stratum, so the realized range
  lies inside the nominal one. The realized range, not the nominal one, delimits the domain
  of validity of any claim.
- **Seeds.** Each study has its own base seed (EX-1 20260804, EX-2 20260806, EX-3 20260807,
  EX-4 20260808), so no study reuses another's seed bank. The design seeds are 0 (EX-1, whose
  design is fixed) and equal to the base seed in EX-2 to EX-4. All seeds are recorded in the
  artifacts.
- **Pilot.** A pilot data farm (ME-1), run before the campaign without the scenario file and
  the model, was used only to size the replication ladders of EX-2 and EX-3. Their records
  cite it; it is not part of the paper and is not published.

---

## 3. Execution control and stopping

- **Replications.** Each design point receives paired replications from disjoint seed banks,
  so CRN pairing holds within a replication and independence between replications. Nested
  seeds let each larger rung extend the previous batch.
- **Determinism.** The design seed only selects the design points: given a point and its base
  seed, a replication is binary-identical whatever the design seed. Parallel runs with a
  canonical output order keep the data set binary-identical.
- **Stopping rule.** Sample-size rungs and a precision target were declared before execution.
  The target is the half-width of the 95% CI of the mean ΔMOS-1 at the worst design point:
  0.025, a quarter of a +0.10 reference margin, or half of it, 0.050, in the intensity sweep.
  The margin is an implementation acceptance criterion, not an operational requirement. In
  EX-4, the rung of 192 replications per point gave a worst half-width of 0.0294 and was
  rejected, and that of 384 gave 0.0195.

The rungs, the values at each rung and the commit and blob identifiers of each criterion
file are in [Stopping criteria](reproducibility.md#stopping-criteria).

---

## 4. Analysis

The analysis layer reimplements the first three of the four steps proposed by Dantas et al.
(2024): design of experiments, execution control and analysis.

- **Tests and intervals.** At α = 0.05 throughout, pre-checks select the test: a Shapiro–Wilk
  test on the paired differences chooses the paired t test or the Wilcoxon signed-rank test.
  The mean paired difference always accompanies the p-value, with a 95% BCa interval from
  9,999 seeded resamples of the differences.
- **Decision rule.** A difference is declared only when the test rejects the null hypothesis
  and the interval excludes zero. Non-rejection is a statement of bound, not of null effect or
  equivalence.
- **Reading a multi-point design.** The aggregate block over all points is a mean of the
  sampled space, not an operating point, so results are read per design point
  (`by_design_point` in the `offpath_*.json` files). A single-point study (EX-1) is an
  operating point, a Monte Carlo estimate rather than a DoE.
- **Interaction.** In the two-dimensional design (EX-4), a bilinear model in λ, p_mr and their
  product, fitted to the per-point means (`scripts/surface_model.py`), gives the interaction
  coefficient. Its interval is a percentile interval from resampling design points, and a
  partial F test checks additivity. The bilinear model serves only the interaction
  coefficient, not pointwise prediction; pointwise values come from the per-point measures.
- **Prediction.** The fourth step, prediction (`cuas-sim analysis predict`), is an
  exploratory study that was not concluded; it is not part of the paper, and its outputs are
  not published.

---

## 5. Reading the published reports

The `report.json` files produced by `cuas-sim analysis report` carry automatic labels that
must not be cited without these caveats:

- **EX-1.** The evidence gate counts design points, not replications. With a single point,
  `decision_summary.evidence` reads "evidencia insuficiente … (N=1 < 10; Loeppky 2009)" by
  construction, even with 200 pairs. Likewise, the per-design-point blocks (feature scores,
  EDA, Pareto front, surrogate) are degenerate. The valid inference is in the master table
  and in the hypothesis blocks, which work per replication.
- **EX-2 to EX-4.** The surrogate check reports as "Q2" the mean of the per-fold R² of the
  exploratory surrogate, not the pooled Q² of Golbraikh and
  Tropsha (2002) that its 0.5 threshold refers to. In EX-2 it reads 0.20, while the pooled
  value is about 0.91.
- **Design seed.** `validity.seeds.design` is `null` because the report reads only the
  data-farm sidecar. The design seed is in `design.parquet.meta.json` and
  `design.provenance.json`, next to each design.

---

## 6. Records and reproduction

| File (per study) | Contents |
|---|---|
| `BRIEF.md` | Results and their reading |
| `MANIFEST.md` | Seeds, design, stopping criterion, exact commands and artifacts |
| `STOPPING_CRITERION.md` (EX-2 to EX-4) | Stopping criterion declared before the first batch |

The folders hold the data and the JSON results. The HTML, Markdown and PNG renderings of the
reports and the outputs of the exploratory prediction step are not published; the commands
in each MANIFEST regenerate them.

The commands, the expected byte identity and the reference environment are in
[`reproducibility.md`](reproducibility.md).
