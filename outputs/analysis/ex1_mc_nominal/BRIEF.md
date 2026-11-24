# EX-1 — Nominal Operating Point: Brief

**Question.** At the nominal operating point, how much does the modernized configuration
(A-29N) change the fraction of incursions neutralized inside the ZIDA, relative to the
baseline (A-29)?

**Study.** Monte Carlo at a single design point: λ = 6 UAS/h and p_mr = 0.5, the nominal
values of the scenario file, fixed before the study. 200 paired replications under common
random numbers, so each replication runs both configurations against the same incursions.
With no factor varied, this study is a Monte Carlo estimate, not a design of experiments.
Each replication has 16 to 18 incursions and about 5.3 h of mission time. Seeds, commands
and artifacts are in [`MANIFEST.md`](MANIFEST.md).

---

## 1. Results

N = 200 paired replications; Δ = modernized − baseline; 95% BCa bootstrap intervals.

| Measure | Baseline | Modernized | Δ | 95% CI | Test | p |
|---|---|---|---|---|---|---|
| MOS-1 | 0.1240 | 0.3093 | **+0.1853** | [+0.1683, +0.2031] | Wilcoxon | 2.2e−32 |
| MOE-1 | 0.6927 | 0.8207 | +0.1280 | [+0.1166, +0.1398] | paired t | 4.5e−54 |
| MOE-3 | 0.3369 | 0.5701 | +0.2331 | [+0.2179, +0.2485] | paired t | 2.6e−75 |
| MOE-2 | 0.5450 | 0.3983 | −0.1467 | [−0.1593, −0.1343] | paired t | 9.8e−58 |
| MOP-1 | 73.67 | 374.93 | +301.25 | [+287.81, +315.78] | Wilcoxon | 1.4e−34 |
| MOS-2 | — | — | +0.2012 | [+0.1736, +0.2275] | one-sample bootstrap | < 1e−4 |

- **MOS-1.** The modernization raises the neutralized fraction from 12.4% to 30.9% of the
  incursions, a factor of 2.5. The whole interval lies above the +0.10 reference margin.
- **By area of responsibility.** The gain is significant in each of the six areas, from
  +0.1158 (AoR-1) to +0.2222 (AoR-2). Each area's value comes from the replications with an
  incursion in that area (165 to 200 of the 200), so the areas are secondary to the total.
- **MOS-2.** The modernization removes about 20% of the baseline's evasions; MOS-2 is
  positive in 83.5% of the replications. MOS-2 is already a comparison of the two
  configurations, so it is tested with a one-sample bootstrap against zero.

**Cost of the gain** (per replication, except rockets, which are per engagement):

| Cost | Baseline | Modernized | Δ | 95% CI |
|---|---|---|---|---|
| Rockets per engagement | 1.083 | 2.475 | **+1.391** | [+1.235, +1.589] |
| Leaks: `pending-evaded` | 2.715 | 2.440 | −0.275 | [−0.480, −0.065] |
| Leaks: `winchester` (out of rockets) | 0.000 | 0.170 | +0.170 | [+0.075, +0.365] |
| Leaks: `bingo-fuel` | 0.075 | 0.085 | +0.010 | [−0.035, +0.060] |

Leak rows are filled with zero before averaging, since a replication without a leak of a
given reason writes no row for it.

---

## 2. Reading

- **Drivers.** The EO/IR sensor updates the pursuit heading about five times as often
  (MOP-1); the guided rocket raises the cumulative neutralization probability per engagement
  by 23 points (MOE-3); and engagements take a smaller share of the policing time (MOE-1).
  The net effect on the mission is the MOS-1 gain.
- **MOE-2 falls by definition, not by a loss of capability.** MOE-2 averages the tracking
  probability sampled in each configuration's own acquisition annulus: (2 km, 13 km] for the
  baseline, (14 km, 20 km] for the modernized configuration. The A-29N tracks at distances
  where the A-29 does not see the target. The tracking probability is a closed-form function
  of the trajectory, so the narrow interval of MOE-2 reflects less modelled noise, not higher
  operational reliability.
- **Cost.** The A-29N uses 1.4 more rockets per engagement because it re-engages until a hit
  inside a much larger envelope, and it engages targets the baseline cannot reach (fewer
  `pending-evaded` leaks). It runs out of rockets in 0.17 events per replication, against none
  for the baseline. The difference in `bingo-fuel` leaks is not significant.
- **Regression ruler.** The 20-seed regression ruler of
  [`docs/reproducibility.md`](../../../docs/reproducibility.md#regression-ruler-about-5-seconds)
  gives a gap of +0.2378. Its seeds are disjoint from these and it has 20 replications
  (standard error about 0.028), so the difference of 0.0525 is about 1.8 standard errors.
  The ruler is a regression fixture of the engine; the estimate to cite is this study's.

---

## 3. Precision

The stopping criterion was declared before the first batch: half-width of the 95% BCa
interval of ΔMOS-1 at most 0.025, over the ladder N ∈ {100, 200, 400}.

| Rung | ΔMOS-1 | 95% BCa | Half-width | Verdict |
|---|---|---|---|---|
| N = 100 | +0.1896 | [+0.1652, +0.2154] | 0.025097 | rejected |
| **N = 200** | **+0.1853** | **[+0.1683, +0.2031]** | **0.017399** | **accepted** |

The seed banks are nested: the first 100 replications of N = 200 are the N = 100 batch, row
by row and seed by seed. The 100 extra replications refine the estimate on the same
sequence; they are not a second, independent sample. Significance was never in doubt (the
effect is about 1.5 standard deviations per replication); the replications buy precision.

---

## 4. Limitations

- **One operating point.** The result is the effect at the nominal point, not an average
  over the scenario space. EX-2 and EX-3 vary each factor; EX-4 varies both.
- **Common random numbers pair the scenario, not the decisions.** The two configurations
  consume different numbers of decision draws, so decision noise remains in Δ.
- **MOP-2 and MOP-3 are excluded from the tests**: they are constants of each configuration.
- **Automatic labels.** The evidence gate of `report_n200/report.json` reads "evidencia
  insuficiente … (N=1 < 10)" because it counts design points, not replications; see
  [`docs/experimental_campaign.md`](../../../docs/experimental_campaign.md#5-reading-the-published-reports).

---

## 5. Where the numbers come from

| Result | Artifact |
|---|---|
| Level of each measure per configuration | `datafarm_n200.parquet` (mean of the 200 replications per configuration) |
| Δ, interval, test and p-value | `report_n200/report.json` (`master_table`); `stats_n200/<measure>/report.json` |
| MOS-2, rockets per engagement, leaks | `offpath_n200.json` (`mos2_one_sample`, `cost`) |
| Stopping rule and seed nesting | `offpath_n200.json` (`stopping`, `nesting_check`) |
