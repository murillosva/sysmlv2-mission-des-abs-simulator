# EX-3 — Composition Sweep: Brief

**Question.** Does the gain of the modernization depend on the type of threat? Is there a
composition below which the gain falls short of the +0.10 reference margin?

**Study.** Latin hypercube in one factor: the multi-rotor fraction p_mr over the nominal
range 0 to 1, with the arrival rate fixed at its nominal value, λ = 6.0 UAS/h. 20 design
points × 192 paired replications = 3,840 pairs. With a single factor, the Latin hypercube is
a stratified sample of the interval (one point inside each of 20 strata of 0.05): the
realized range is **p_mr ∈ [0.0159, 0.9825]**. The composition goes from almost pure
fixed-wing (130 m/s, 14 m, dimensional penalty M = 1.0) to almost pure multi-rotor (29 m/s,
1 m, M = 0.5). The effect measured is **marginal**, with λ fixed; the interaction is the
subject of EX-4. The stopping criterion was declared and committed before the first batch in
[`STOPPING_CRITERION.md`](STOPPING_CRITERION.md); seeds, commands and artifacts are in
[`MANIFEST.md`](MANIFEST.md).

---

## 1. Results

ΔMOS-1 per design point (R = 192 paired replications each; 95% BCa intervals), classified
against the +0.10 margin: *exceeds* when the interval lies above it, *falls short* when below,
*indeterminate* when it contains it.

| Design point | p_mr | ΔMOS-1 | 95% CI | Against +0.10 | MOS-1, baseline → modernized |
|---|---|---|---|---|---|
| 3 | 0.0159 | **+0.0154** | [+0.0039, +0.0266] | falls short | 0.083 → 0.098 |
| 1 | 0.0886 | +0.0573 | [+0.0448, +0.0701] | falls short | 0.072 → 0.129 |
| 5 | 0.1282 | +0.0602 | [+0.0450, +0.0749] | falls short | 0.093 → 0.153 |
| 12 | 0.1833 | +0.0748 | [+0.0611, +0.0903] | falls short | 0.100 → 0.175 |
| 19 | 0.2191 | +0.0921 | [+0.0749, +0.1092] | indeterminate | 0.102 → 0.195 |
| 16 | 0.2699 | +0.1015 | [+0.0853, +0.1195] | indeterminate | 0.098 → 0.200 |
| 4 | 0.3231 | +0.1262 | [+0.1097, +0.1440] | exceeds | 0.110 → 0.236 |
| 17 | 0.3682 | +0.1482 | [+0.1309, +0.1678] | exceeds | 0.116 → 0.264 |
| 0 | 0.4367 | +0.1879 | [+0.1693, +0.2074] | exceeds | 0.118 → 0.306 |
| 10 | 0.4750 | +0.1859 | [+0.1672, +0.2049] | exceeds | 0.109 → 0.295 |
| 2 | 0.5381 | +0.2250 | [+0.2056, +0.2446] | exceeds | 0.124 → 0.349 |
| 18 | 0.5604 | +0.2113 | [+0.1932, +0.2299] | exceeds | 0.137 → 0.349 |
| 8 | 0.6427 | +0.2700 | [+0.2488, +0.2925] | exceeds | 0.144 → 0.414 |
| 13 | 0.6667 | +0.2911 | [+0.2709, +0.3115] | exceeds | 0.132 → 0.424 |
| 11 | 0.7448 | +0.3061 | [+0.2873, +0.3258] | exceeds | 0.143 → 0.450 |
| 6 | 0.7532 | +0.3159 | [+0.2957, +0.3370] | exceeds | 0.143 → 0.459 |
| 7 | 0.8211 | +0.3627 | [+0.3421, +0.3849] | exceeds | 0.163 → 0.526 |
| 15 | 0.8794 | +0.3708 | [+0.3491, +0.3927] | exceeds | 0.165 → 0.535 |
| 9 | 0.9198 | +0.4282 | [+0.4081, +0.4478] | exceeds | 0.166 → 0.594 |
| 14 | 0.9825 | **+0.4510** | [+0.4307, +0.4721] | exceeds | 0.171 → 0.622 |

- **The composition governs the gain.** The gain is significant at all 20 points, including
  the fixed-wing extreme, and varies **29.4 times** from the worst to the best point
  (against 1.74 times over λ in EX-2).
- **Trend:** slope **+0.4393 per unit of p_mr**, 95% CI [+0.4092, +0.4678], from a bootstrap
  over the design points; a total change of +0.4246 ± 0.0283 over the realized range.
  Correlation: Pearson +0.993, Spearman +0.997, Kendall +0.979.
- **Crossing of the margin.** The fitted line crosses +0.10 at **p_mr = 0.241**
  [0.227, 0.259]: 14 points exceed the margin, 4 fall short and 2 are indeterminate, and the
  two indeterminate points (p_mr = 0.219 and 0.270) are the ones that bracket the crossing.
- **Groups of p_mr.** Binning p_mr into terciles, the groups differ for all five measures
  (ANOVA, p of the order of 1e−8).

**Drivers at the two extremes** of the composition (mean Δ per design point; at the
fixed-wing extreme, with the 95% BCa interval):

| Measure | p_mr = 0.0159 (fixed-wing extreme) | p_mr = 0.9825 (multi-rotor extreme) |
|---|---|---|
| ΔMOE-3 | **−0.0082 [−0.0240, +0.0067]**, not distinguishable from zero | +0.468 |
| ΔMOE-1 | +0.0154 [+0.0108, +0.0199] | +0.279 |
| ΔMOE-2 | +0.0103 [−0.0055, +0.0262], not distinguishable from zero | −0.339 |
| ΔMOP-1 | **+429** updates per engagement | **+164** |

The intervals at the fixed-wing extreme come from `stats.bootstrap_effect` applied to the 192
paired differences of design point 3 in `datafarm_r192.parquet` (bootstrap seed 20260807,
9,999 resamples); the same call reproduces the ΔMOS-1 interval of that point in
`offpath_r192.json`.

**Cost of the gain**, at four of the 20 points (all points in `offpath_r192.json`,
`cost_by_design_point`):

| p_mr | Rockets per engagement, baseline → modernized | `winchester` leaks (modernized) | `pending-evaded` leaks, baseline → modernized |
|---|---|---|---|
| 0.016 | 1.309 → 2.478 (+1.17) | 0.682 | 4.052 → 3.260 |
| 0.323 | 1.151 → 2.605 (+1.45) | 0.594 | 3.151 → 2.714 |
| 0.643 | 1.064 → 2.669 (+1.61) | 0.125 | 1.995 → 1.656 |
| 0.982 | 0.946 → 2.814 (+1.87) | 0.052 | 0.557 → 0.417 |

The baseline has no `winchester` leak at any of the 20 points.

---

## 2. Reading

- **At the fixed-wing extreme, the weapon is no better.** ΔMOE-3 is not distinguishable from
  zero there: against fixed-wing targets (M = 1.0) the unguided rocket fired at 0.5–2 km
  already has a high hit probability, while the guided rocket fired at 2–14 km pays the range
  term of the hit-probability curve. Against multi-rotor targets (M = 0.5) the baseline loses
  half of its hit probability and has no range to re-engage, while the A-29N accumulates
  attempts. What remains of the gain at the fixed-wing extreme comes from MOE-1, the
  opportunity to engage.
- **Sensor performance is not mission performance.** ΔMOP-1 is largest exactly where the
  mission gain is smallest (+429 updates against +164). Against fixed-wing targets the
  closing speed is 14 m/s (A-29 at 144 m/s, target at 130 m/s), so the A-29N tracks for a long
  time without converting it into kills. The sign of the effect is opposite to EX-2, where
  ΔMOP-1 grew with λ: reading MOP-1 as a proxy of mission value would invert the conclusion.
- **MOE-2 at the fixed-wing extreme.** A fixed-wing target has tracking probability 1.0 in
  both configurations, so ΔMOE-2 should vanish at p_mr = 0; the measured value at
  p_mr = 0.0159 contains zero.
- **Cost.** Rockets per engagement grow with the multi-rotor fraction (the dimensional
  penalty forces more re-engagements). Running out of rockets is a fixed-wing failure mode: it
  falls 13 times across the range, as the A-29N spends its load in tail chases it cannot
  convert. Evasions (`pending-evaded`) fall 7.3 times, because a fixed-wing UAS crosses the
  80 nmi in about 19 min against about 85 min for a multi-rotor.
- **Agreement with EX-1.** λ is 6.0 exactly in both studies. The point nearest the nominal
  composition, p_mr = 0.475, gives +0.1859 [+0.1672, +0.2049], against +0.1853
  [+0.1683, +0.2031] in EX-1: a difference of +0.0006, with disjoint seed banks.

---

## 3. Precision

Criterion: the **maximum over the 20 design points** of the half-width of the 95% BCa
interval of ΔMOS-1 at most 0.025, a quarter of the +0.10 margin, because the curve crosses
the margin and each point has to be classified against it; ladder R ∈ {96, 192, 384, 768}.

| Rung | Maximum half-width | Points above the target | Verdict |
|---|---|---|---|
| R = 96 | 0.0327 (p_mr = 0.753) | 12 of 20 | rejected |
| **R = 192** | **0.0218** (p_mr = 0.643) | **0 of 20** | **accepted** |

The seed banks are nested: the first 96 replications of each point at R = 192 are the R = 96
batch. The confirmatory checks are in the MANIFEST.

---

## 4. Limitations

- **Marginal effect, with λ fixed at 6.0**; the crossing p_mr = 0.241 holds at that
  intensity only (EX-4 measures how it moves with λ).
- **Realized range p_mr ∈ [0.0159, 0.9825]**; do not extrapolate to pure fixed-wing or pure
  multi-rotor threats.
- **The classification depends on the +0.10 margin**, an implementation acceptance criterion,
  not an operational requirement.
- **Common random numbers pair the configurations inside a replication, not across design
  points.**
- **The negative statements** (ΔMOE-3 and ΔMOE-2 at the fixed-wing extreme) are statements
  of bound, not of null effect.
- **The aggregate over the 20 points** in `report_r192/report.json` is a mean of the sampled
  space; its decision index ranks areas of responsibility, not levels of p_mr.
- **MOP-2 and MOP-3** are constants and are excluded from the tests.

---

## 5. Where the numbers come from

| Result | Artifact |
|---|---|
| ΔMOS-1 per point, trend, Pearson correlation | `offpath_r192.json` (`by_design_point`, `trend`) |
| Classification against the margin | Each point's interval in `by_design_point` compared with +0.10 |
| Spearman and Kendall correlations | `report_r192/report.json` (`relationships`) |
| Drivers at the extremes (ΔMOE-3, ΔMOE-1, ΔMOE-2, ΔMOP-1) | `report_r192/report.json` (per-point means: `responses[].eda.summary` and `pareto_front`); intervals recomputed from `datafarm_r192.parquet` as stated above |
| Tercile tests | `report_r192/report.json` (`anova_by_levels`) |
| Cost per point | `offpath_r192.json` (`cost_by_design_point`) |
| Stopping rule and seed nesting | `offpath_r96.json`, `offpath_r192.json` (`by_design_point`, `nesting_check`) |
