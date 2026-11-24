# EX-2 — Intensity Sweep: Brief

**Question.** Does the gain of the modernization hold as the arrival rate of incursions, λ,
varies? Does it grow, fall or stay the same?

**Study.** Latin hypercube in one factor: λ over the nominal range 2 to 18 UAS/h, with the
multi-rotor fraction fixed at its nominal value, p_mr = 0.5. 16 design points × 96 paired
replications = 1,536 pairs. With a single factor, the Latin hypercube is a stratified sample
of the interval (one point drawn inside each of 16 strata of 1.0 UAS/h), so the extremes are
not sampled: the realized range is **λ ∈ [2.99, 17.54]**. The effect measured is
**marginal**, with p_mr fixed; the interaction between the two factors is the subject of
EX-4. The stopping criterion was declared before the first batch in
[`STOPPING_CRITERION.md`](STOPPING_CRITERION.md); seeds, commands and artifacts are in
[`MANIFEST.md`](MANIFEST.md).

---

## 1. Results

ΔMOS-1 per design point (R = 96 paired replications each; 95% BCa intervals):

| Design point | λ (UAS/h) | ΔMOS-1 | 95% CI |
|---|---|---|---|
| 7 | 2.992 | **+0.2425** | [+0.2027, +0.2837] |
| 11 | 3.971 | +0.2519 | [+0.2177, +0.2904] |
| 12 | 4.018 | +0.2282 | [+0.1868, +0.2718] |
| 8 | 5.388 | +0.2062 | [+0.1801, +0.2341] |
| 6 | 6.721 | +0.1823 | [+0.1552, +0.2111] |
| 5 | 7.950 | +0.1664 | [+0.1408, +0.1917] |
| 2 | 8.869 | +0.1707 | [+0.1505, +0.1924] |
| 3 | 9.125 | +0.1532 | [+0.1316, +0.1763] |
| 0 | 10.504 | +0.1540 | [+0.1316, +0.1752] |
| 13 | 11.764 | +0.1621 | [+0.1428, +0.1827] |
| 9 | 12.019 | +0.1659 | [+0.1448, +0.1883] |
| 14 | 13.132 | +0.1611 | [+0.1422, +0.1835] |
| 10 | 14.630 | +0.1587 | [+0.1386, +0.1813] |
| 15 | 15.488 | +0.1487 | [+0.1288, +0.1696] |
| 4 | 16.079 | +0.1449 | [+0.1250, +0.1634] |
| 1 | 17.542 | **+0.1519** | [+0.1335, +0.1707] |

- **The gain falls with λ but stays positive.** It is significant at all 16 points
  (bootstrap p < 1e−4 at each), and every interval lies above the +0.10 reference margin:
  the smallest lower bound is +0.1250, at λ = 16.08.
- **Trend:** slope **−0.00638 per UAS/h**, 95% CI [−0.00858, −0.00367], from a bootstrap
  over the design points; a total change of −0.0928 ± 0.0357 over the realized range.
  Correlation with λ: Pearson −0.85, Spearman −0.91, Kendall −0.77.
- **Groups of λ.** Binning λ into terciles, the groups differ for ΔMOS-1 (Welch ANOVA,
  p = 0.0087, Tamhane T2 post hoc), ΔMOE-1 (ANOVA, p = 0.0020) and ΔMOP-1 (Welch ANOVA,
  p = 0.0044), but not for ΔMOE-3 (ANOVA, p = 0.071) or ΔMOE-2 (ANOVA, p = 0.43).

**Cost of the gain**, at five of the 16 points (all points in `offpath_r96.json`,
`cost_by_design_point`):

| λ | Rockets per engagement, baseline → modernized | `winchester` leaks (modernized) | `pending-evaded` leaks, baseline → modernized |
|---|---|---|---|
| 2.992 | 1.280 → 3.117 (+1.837) | 0.219 | 0.94 → 0.76 |
| 6.721 | 1.030 → 2.608 (+1.578) | 0.552 | 2.99 → 2.45 |
| 10.504 | 0.943 → 2.124 (+1.181) | 0.073 | 4.41 → 4.31 |
| 14.630 | 0.920 → 2.409 (+1.489) | 1.490 | 6.02 → 5.18 |
| 17.542 | 0.949 → 2.237 (+1.287) | 0.896 | 6.54 → 5.71 |

---

## 2. Reading

- **A hypothesis is saturation: the constraint moves from the weapon to the fleet.** Each
  A-29 engages one UAS at a time. As λ grows, the test over the terciles does not reject equal
  gains in ΔMOE-3, the effectiveness per engagement (p = 0.071, a statement of bound, not of
  null effect), while ΔMOE-1 falls (correlation −0.79 with λ). The modernization would still
  act on each engagement, but fewer aircraft are free to engage. The mechanism is not isolated
  by experiment.
- **The gain shrinks by about 40%**, from about +0.24 at low intensity to about +0.15 at high
  intensity, and remains above the margin throughout.
- **Cost.** Rockets per engagement grow most where the gain is largest (low λ), where the
  A-29N has time to re-engage; running out of rockets (`winchester`) becomes more frequent at
  high λ.
- **Agreement with EX-1.** The two points next to the nominal λ = 6 give +0.2062 (λ = 5.39)
  and +0.1823 (λ = 6.72); EX-1 measured +0.1853 at λ = 6.0, between them, with overlapping
  intervals. The seed banks of the two studies are disjoint, so this is an independent
  replication.

---

## 3. Precision

Criterion: the **maximum over the 16 design points** of the half-width of the 95% BCa
interval of ΔMOS-1 at most 0.050, half of the +0.10 reference margin, since every point lies
far from it; ladder R ∈ {48, 96, 192}.

| Rung | Maximum half-width | Points above the target | Verdict |
|---|---|---|---|
| R = 48 | 0.0665 (λ = 4.02) | 2 of 16 | rejected |
| **R = 96** | **0.0425** (λ = 4.02) | **0 of 16** | **accepted** |

The seed banks are nested: the first 48 replications of each point at R = 96 are the R = 48
batch. The confirmatory checks are in the MANIFEST.

---

## 4. Limitations

- **Marginal effect, with p_mr fixed at 0.5**; it says nothing about the interaction (EX-4).
- **Realized range λ ∈ [2.99, 17.54]**, not [2, 18]; do not extrapolate.
- **Common random numbers pair the two configurations inside a replication, not across
  design points**: comparisons between levels of λ do not benefit from the pairing.
- **The aggregate over the 16 points** in `report_r96/report.json` is a mean of the sampled
  space, not an operating point; the result of this study is the curve.
- **MOE-2** falls by definition (different acquisition annuli, see EX-1); **MOP-2 and MOP-3**
  are constants and are excluded from the tests.
- **Automatic labels.** The evidence gate of `report_r96/report.json` reports a surrogate
  "Q2 = 0.20" that is the mean of per-fold R², not the pooled Q² (about 0.91); see
  [`docs/experimental_campaign.md`](../../../docs/experimental_campaign.md#5-reading-the-published-reports).

---

## 5. Where the numbers come from

| Result | Artifact |
|---|---|
| ΔMOS-1 per point, half-widths, trend, Pearson correlation | `offpath_r96.json` (`by_design_point`, `trend`) |
| Tercile tests; Spearman and Kendall correlations; correlations of the other measures | `report_r96/report.json` (`anova_by_levels`, `relationships`) |
| Cost per point | `offpath_r96.json` (`cost_by_design_point`) |
| Stopping rule and seed nesting | `offpath_r48.json`, `offpath_r96.json` (`by_design_point`, `nesting_check`) |
