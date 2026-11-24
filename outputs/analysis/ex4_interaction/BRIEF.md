# EX-4 — Two-Dimensional Design: Brief

**Question.** Do the effects of the arrival rate λ and of the multi-rotor fraction p_mr on
ΔMOS-1 add up, or does the effect of one factor depend on the level of the other?

**Study.** Latin hypercube in two factors: λ over 2 to 18 UAS/h and p_mr over 0 to 1, with no
factor fixed. 32 design points × 384 paired replications = 12,288 pairs. Realized ranges:
**λ ∈ [2.364, 17.571]** and **p_mr ∈ [0.0017, 0.9851]**. The stopping criterion was declared
and committed before the first batch in [`STOPPING_CRITERION.md`](STOPPING_CRITERION.md);
seeds, commands and artifacts are in [`MANIFEST.md`](MANIFEST.md).

**Why a two-dimensional design.** One-factor sweeps carry no information on the dependence
between factors. The 37 points of EX-1 to EX-3 lie on two lines that cross at the nominal
point (p_mr = 0.5 for EX-2, λ = 6 for EX-3); on those points the column λ·p_mr of the matrix
[1, λ, p_mr, λ·p_mr] is an exact linear combination of the other three (rank 3, condition
number 4.56e+16), so no reanalysis of them can identify the interaction. The EX-4 design has
rank 4 and condition number 122.

---

## 1. Results

### 1.1 Interaction

Bilinear surface fitted by least squares to the 32 per-point means of ΔMOS-1; intervals from
a percentile bootstrap over the design points:

```
ΔMOS-1 = −0.006581 + 0.000629·λ + 0.521918·p_mr − 0.015027·λ·p_mr      R² = 0.9868 (additive: 0.9552)
```

| Coefficient | Estimate | 95% CI |
|---|---|---|
| Intercept | −0.006581 | [−0.038651, +0.022755] |
| λ | +0.000629 | [−0.002126, +0.002932] |
| p_mr | **+0.521918** | [+0.443164, +0.586786] |
| **λ × p_mr** | **−0.015027** | **[−0.020504, −0.008902]** |

- **Additivity is rejected:** partial F test, F(1,28) = 66.90, p = 6.7e−9.
- **Robustness:** in the quadratic model the interaction coefficient is −0.013437
  [−0.018280, −0.010912]: same sign and order, overlapping interval.
- **The interaction compresses the effect of the composition as traffic grows.** The slope in
  p_mr is b2 + b3·λ:

| λ (UAS/h) | Slope in p_mr | 95% CI |
|---|---|---|
| 2.364 (realized minimum) | +0.4864 | [+0.4212, +0.5394] |
| 6.0 (nominal) | +0.4318 | [+0.3869, +0.4668] |
| 17.571 (realized maximum) | +0.2579 | [+0.2197, +0.2969] |

  It falls by **47%** across the realized range of λ (95% CI [32%, 58%]) and by **40.3%**
  [26.2%, 51.6%] from λ = 6 to the realized maximum. Over the realized ranges, the cross term
  is worth **−0.2247** of ΔMOS-1 [−0.3066, −0.1331], beyond a second-order correction.
- **By terciles of λ** (the lines of the paper's Figure 2, each evaluated at the mean λ of its
  tercile): +0.45 (mean λ 4.75), +0.37 (9.98) and +0.29 (15.28).
- **Agreement with EX-3:** at λ = 6 the slope, +0.4318, lies within the interval
  [+0.4092, +0.4678] of the +0.4393 that EX-3 measured on a disjoint seed bank.

### 1.2 Per design point

The modernization raised MOS-1 at all 32 points, every interval above zero, from +0.0103
(λ = 10.57, p_mr = 0.0017) to +0.4575 (λ = 6.14, p_mr = 0.9851): a ratio of about **44
times**. The mean over the 32 points, +0.1837, depends on the factor ranges of the study and
is not an operating point.

| Design point | λ | p_mr | ΔMOS-1 | 95% CI | MOS-2 |
|---|---|---|---|---|---|
| 19 | 2.364 | 0.1618 | +0.1000 | [+0.0806, +0.1197] | +0.1152 |
| 11 | 2.912 | 0.3329 | +0.1601 | [+0.1410, +0.1794] | +0.1927 |
| 10 | 3.452 | 0.5715 | +0.2642 | [+0.2448, +0.2833] | +0.3067 |
| 1 | 3.687 | 0.3842 | +0.1782 | [+0.1627, +0.1944] | +0.1959 |
| 20 | 4.092 | 0.4865 | +0.2156 | [+0.1997, +0.2321] | +0.2559 |
| 0 | 4.878 | 0.2271 | +0.0975 | [+0.0847, +0.1110] | +0.1044 |
| 5 | 5.186 | 0.1363 | +0.0611 | [+0.0498, +0.0727] | +0.0664 |
| 4 | 5.674 | 0.2844 | +0.1104 | [+0.0981, +0.1232] | +0.1041 |
| 7 | 6.135 | 0.9851 | **+0.4575** | [+0.4427, +0.4721] | +0.5460 |
| 31 | 6.617 | 0.6707 | +0.2656 | [+0.2519, +0.2798] | +0.2891 |
| 21 | 7.294 | 0.9540 | +0.4039 | [+0.3897, +0.4181] | +0.4709 |
| 22 | 7.533 | 0.8949 | +0.3580 | [+0.3432, +0.3733] | +0.4020 |
| 26 | 8.159 | 0.3530 | +0.1248 | [+0.1141, +0.1355] | +0.1130 |
| 27 | 8.586 | 0.4392 | +0.1539 | [+0.1422, +0.1652] | +0.1362 |
| 8 | 9.209 | 0.9126 | +0.3376 | [+0.3247, +0.3506] | +0.3755 |
| 9 | 9.961 | 0.7720 | +0.2759 | [+0.2643, +0.2875] | +0.2871 |
| 29 | 10.329 | 0.2758 | +0.0890 | [+0.0797, +0.0986] | +0.0720 |
| 13 | 10.571 | 0.0017 | **+0.0103** | [+0.0045, +0.0164] | −0.0092 |
| 28 | 11.412 | 0.1984 | +0.0663 | [+0.0574, +0.0750] | +0.0377 |
| 15 | 11.616 | 0.5432 | +0.1728 | [+0.1614, +0.1842] | +0.1654 |
| 2 | 12.471 | 0.6016 | +0.1920 | [+0.1806, +0.2036] | +0.1861 |
| 23 | 12.617 | 0.6493 | +0.2066 | [+0.1958, +0.2180] | +0.1950 |
| 18 | 13.436 | 0.7944 | +0.2451 | [+0.2338, +0.2564] | +0.2512 |
| 12 | 13.979 | 0.5106 | +0.1602 | [+0.1501, +0.1705] | +0.1351 |
| 3 | 14.243 | 0.8271 | +0.2592 | [+0.2478, +0.2702] | +0.2475 |
| 6 | 14.781 | 0.1014 | +0.0424 | [+0.0355, +0.0494] | +0.0276 |
| 14 | 15.249 | 0.7144 | +0.2171 | [+0.2067, +0.2275] | +0.2056 |
| 25 | 15.935 | 0.7427 | +0.2243 | [+0.2140, +0.2348] | +0.2293 |
| 30 | 16.098 | 0.4241 | +0.1217 | [+0.1124, +0.1312] | +0.1096 |
| 17 | 16.999 | 0.0741 | +0.0276 | [+0.0219, +0.0337] | −0.0053 |
| 24 | 17.175 | 0.8610 | +0.2515 | [+0.2415, +0.2618] | +0.2574 |
| 16 | 17.571 | 0.0559 | +0.0278 | [+0.0222, +0.0338] | +0.0012 |

MOS-2 is negative at two points of low p_mr (13 and 17); both intervals contain zero, so the
reduction of evasions there is not distinguishable from zero.

### 1.3 Marginal slopes and rockets

- **Average slopes over the sampled space** (the two passes of `campaign_postprocess.py`):
  −0.00450 per UAS/h for λ alone, 95% CI [−0.01292, +0.00277], an interval that contains
  zero, whereas in EX-2, with p_mr fixed, the slope excluded zero; +0.35634 per unit of p_mr
  [+0.30416, +0.40457]. Each λ comes with a different drawn p_mr, so the interaction turns
  the variation of p_mr into scatter of the λ slope: in a two-dimensional design the marginal
  of one factor is not its effect, and the surface is the reading of λ.
- **Rocket shortage.** Running out of rockets (`winchester`) peaks at high λ and low p_mr. At
  the largest intensity drawn (λ = 17.57, p_mr = 0.056), on average 2.378 UAS per
  replication went unengaged by the A-29N for lack of rockets, against 0.096 at λ = 2.36,
  p_mr = 0.162. The baseline has no `winchester` leak at any of the 32 points.

---

## 2. Reading

- **The two factors do not add up.** The composition sets the level of the gain; the traffic
  sets how much the composition is worth. Traffic eroded the advantage against multi-rotor
  targets, consistent with the saturation hypothesis of EX-2: without a free aircraft, easy
  targets did not become neutralizations. The mechanism is a hypothesis, not isolated by
  experiment.
- **Missions are screened by the intensity–composition pair, never by composition alone.** A
  threat profile that justifies the alternative configuration at low traffic may not justify
  it at high traffic, so the pair is judged against the decision maker's gain threshold.

---

## 3. Cross-checks

The surface, fitted on EX-4 data only, reproduces the two lines measured by EX-2
(p_mr = 0.5, 16 points) and EX-3 (λ = 6, 20 points) on disjoint seed banks, with RMSE 0.0192
and 0.0150, against a target of 0.050. The residuals have a structure, though: on the
p_mr = 0.5 line the surface overestimates by +0.0074 on average (+0.038 at worst), and on the
λ = 6 line it underestimates at high p_mr (−0.034 at p_mr = 0.92), a curvature in p_mr that the
bilinear model does not capture. At (6, 0.5) the surface gives +0.2131, against the +0.1853
measured by EX-1: **the bilinear surface is not a pointwise predictor**. It is the instrument
of the interaction coefficient; pointwise values come from the per-point measures.

---

## 4. Precision

Criterion: the **maximum over the 32 design points** of the half-width of the 95% BCa
interval of ΔMOS-1 at most 0.025, a quarter of the +0.10 margin; ladder
R ∈ {192, 384, 768, 1536}.

| Rung | Maximum half-width | Points above the target | Verdict |
|---|---|---|---|
| R = 192 | 0.0294 | 3 of 32 (the three lowest λ) | rejected |
| **R = 384** | **0.0195** | **0 of 32** | **accepted** |

A second primary criterion, on the precision of the interaction coefficient, was rejected
before the first batch: that precision is limited by lack of fit, not by sampling noise, and
does not improve with replications. Replications buy precision per design point; more design
points would be needed to narrow the coefficients of the surface.

---

## 5. Limitations

- **Realized ranges λ ∈ [2.364, 17.571] and p_mr ∈ [0.0017, 0.9851]**; the surface is not
  interpreted beyond them.
- **The bilinear surface has measurable lack of fit** (§3): valid for the interaction
  coefficient, not for pointwise prediction.
- **The interaction coefficient depends on the model**: −0.0150 bilinear, −0.0134 quadratic.
- **Binned tests in λ** in `report_r384/report.json` do not reject for ΔMOS-1, ΔMOE-1,
  ΔMOE-3 and ΔMOE-2: binning λ pools different compositions inside each tercile, and the
  composition dominates the variance. In a two-dimensional design such tests are informative
  only for the dominant factor.
- **Common random numbers pair the configurations inside a replication, not across design
  points.** The decision index of the report ranks areas of responsibility, not design points.
- **The negative statements** (MOS-2 at points 13 and 17; the λ and intercept coefficients)
  are statements of bound, not of null effect.
- **MOE-2** falls by definition (different acquisition annuli, see EX-1); **MOP-2 and MOP-3**
  are configuration constants and are excluded.
- **Reports.** With more than 5,000 pairs, the Shapiro–Wilk pre-check of
  `analysis stats` warns that its p-value may be inaccurate; it only selects the test, and the
  effect and interval come from the bootstrap.

---

## 6. Where the numbers come from

| Result | Artifact |
|---|---|
| Coefficients, F test, quadratic check, slope at λ = 6, cross-checks | `surface_r384.json` (`surface_bilinear`, `surface_quadratic`, `cross_checks`) |
| Slopes at the realized ends of λ, falls of 47% and 40.3%, cross term, tercile slopes | `outputs/verification/surface_derived_audit.json` |
| ΔMOS-1 and MOS-2 per point, mean over the points, average slopes, rocket shortage | `offpath_r384_lambda.json`, `offpath_r384_pmr.json` (`by_design_point`, `stopping.current`, `trend`, `cost_by_design_point`) |
| Stopping rule and seed nesting | `offpath_r192_lambda.json`, `offpath_r384_*.json` (`by_design_point`, `nesting_check`) |
