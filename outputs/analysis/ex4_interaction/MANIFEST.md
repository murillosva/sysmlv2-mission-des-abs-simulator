# EX-4 — Two-Dimensional Design: Manifest

Reproduction record of the study described in [`BRIEF.md`](BRIEF.md). The stopping
criterion was written in [`STOPPING_CRITERION.md`](STOPPING_CRITERION.md) (in Portuguese)
before the design was generated and committed, together with `scripts/surface_model.py`,
before the first batch. That file carries a publication note listing its editorial changes.

## Execution

- Executed on 2026-08-07, from development commit `23cea10`; the stopping criterion and
  `scripts/surface_model.py` were committed in `f076450` before the first `analysis run`
  (development repository, not public; the commit of the run is recorded as `git_head` in
  `design.provenance.json`). No change under `src/`.
- Environment: the reference environment of
  [`docs/reproducibility.md`](../../../docs/reproducibility.md#environment-reference).
- The regression ruler (`scripts/batch_seeds.py --seeds 20`) matched its reference values
  before and after the study.

## Seeds

| Role | Value | Note |
|---|---|---|
| Design seed | 20260808 | Draws both factors of the Latin hypercube |
| Base seed | 20260808 | Disjoint from EX-1 to EX-3 (20260804, 20260806, 20260807) |
| Bootstrap seed | 20260808 | BCa intervals, achieved significance levels, bootstrap of the surface |
| Model seed | 20260808 | Exploratory surrogate, whose grid is an input of `surface_r384.json` |

Because the seed bank is disjoint, the surface fitted on EX-4 alone is an independent check
of the points measured by EX-2 and EX-3. The design seed is recorded in
`design.parquet.meta.json` and in the `design` block of the `offpath_*.json` files.

## Design

- Sampler `LatinHypercube` (`scipy.stats.qmc.LatinHypercube`, `scramble=True`, two
  dimensions): `lambda_per_h` over [2, 18] and `uas_archetype_p_mr` over [0, 1], no fixed
  factor.
- 32 design points (listed with their results in the BRIEF); realized ranges
  λ ∈ [2.3644, 17.5709] and p_mr ∈ [0.0017, 0.9851].
- The matrix [1, λ, p_mr, λ·p_mr] of the design has rank 4 and condition number 122.1.
- 192 paired replications per point, extended to 384.
- Base configuration `--config configs/zida_shield_default.yaml --sysml sysml`; no
  `--max-arrivals`; `--workers 6`.

## Stopping criterion

Summary of `STOPPING_CRITERION.md`:

- **Primary, the only triggering one:** maximum over the 32 design points of the half-width
  of the 95% BCa interval of ΔMOS-1 (`--aor ALL`) at most **0.025**, a quarter of the +0.10
  reference margin, since the surface crosses the margin.
- **Ladder:** R ∈ {192, 384, 768, 1536}; stop at the first rung that passes.
- **Confirmatory, not triggering:** (1) interaction detected; (2) classification of the
  points against the margin; (3) frontier resolved (half-width at most 0.05 in p_mr); (4) Q²
  at least 0.50 in both readings; (5) stability between rungs; (6) cross-check against EX-2
  and EX-3 with RMSE at most 0.05.
- A second primary criterion, on the precision of the interaction coefficient, was evaluated
  and rejected before the first batch, because that precision is dominated by lack of fit and
  does not improve with replications.

| Rung | Maximum half-width | Binding point | Points above the target | Verdict |
|---|---|---|---|---|
| R = 192 | 0.029378 | 19 (λ = 2.364) | 3 of 32 | rejected |
| **R = 384** | **0.019518** | 19 (λ = 2.364) | **0 of 32** | **accepted** |

The three points above the target at R = 192 are the three of lowest λ (2.36, 2.91, 3.45).

| Confirmatory at R = 384 | Target | Obtained | Result |
|---|---|---|---|
| (1) interaction | interval of b3 excludes zero, negative, F rejects | b3 = −0.015027 [−0.020504, −0.008902]; F(1,28) = 66.90 | pass |
| (2) classification | indeterminate points only next to the frontier | 22 exceed, 7 fall short, 3 indeterminate, each within 0.05 of the frontier | pass |
| (3) frontier | half-width ≤ 0.05 in p_mr | 0.030 (λ = 2), 0.021 (6), 0.016 (12), 0.037 (18) | pass |
| (4) Q² of the surrogate | ≥ 0.50 | random forest 0.8428 as reported, 0.9095 pooled | pass |
| (5) drift of the mean | ≤ 0.00074 | 0.0010 | **fails** |
| (5) relative change of mean / standard deviation | < 5% | 0.55% / 0.97% | pass |
| (6) cross-check RMSE | ≤ 0.050 | 0.0192 (EX-2 line), 0.0150 (EX-3 line) | pass |

The drift budget is 0.25 × the half-width of the aggregate block, which pools all 12,288
pairs and is therefore very narrow (0.00295); the budget tightens as the study grows. The
confirmatory is non-triggering and the other half of it passes; the failure is recorded as
it came out.

Seed nesting: at every design point, the first 192 replications of the R = 384 batch are
identical, row by row and seed by seed, to the R = 192 batch.

## Commands

Run from the repository root.

```bash
D=outputs/analysis/ex4_interaction

python scripts/make_study_design.py --lhs lambda,p_mr --samples 32 --design-seed 20260808 -o $D/design.parquet

python -m cuas_sim.cli analysis run --design $D/design.parquet --reps 192 --seed 20260808 --config configs/zida_shield_default.yaml --sysml sysml --workers 6 -o $D/datafarm_r192.parquet
python -m cuas_sim.cli analysis run --design $D/design.parquet --reps 384 --seed 20260808 --config configs/zida_shield_default.yaml --sysml sysml --workers 6 -o $D/datafarm_r384.parquet

# Stopping. --factor takes one factor and changes only the per-point label and the trend block;
# the primary criterion is identical in both passes. The accepted rung is processed once per factor.
python scripts/campaign_postprocess.py --current $D/datafarm_r192.parquet --factor lambda_per_h --design $D/design.parquet --half-width 0.025 --ladder 192,384,768,1536 --bootstrap-seed 20260808 -o $D/offpath_r192_lambda.json
python scripts/campaign_postprocess.py --current $D/datafarm_r384.parquet --previous $D/datafarm_r192.parquet --factor lambda_per_h --design $D/design.parquet --half-width 0.025 --ladder 192,384,768,1536 --bootstrap-seed 20260808 -o $D/offpath_r384_lambda.json
python scripts/campaign_postprocess.py --current $D/datafarm_r384.parquet --previous $D/datafarm_r192.parquet --factor uas_archetype_p_mr --design $D/design.parquet --half-width 0.025 --ladder 192,384,768,1536 --bootstrap-seed 20260808 -o $D/offpath_r384_pmr.json

# Exploratory surrogate: not published, but its grid is an input of surface_r384.json.
python -m cuas_sim.cli analysis predict --datafarm $D/datafarm_r384.parquet --response MOS-1 --aor ALL --model rf --seed 20260808 --grid-resolution 25 -o $D/predict_rf

# Surface with interaction, frontier and cross-checks (needs ex2_lambda/offpath_r96.json and ex3_mix/offpath_r192.json).
python scripts/surface_model.py --datafarm $D/datafarm_r384.parquet --design $D/design.parquet --bootstrap-seed 20260808 --compare outputs/analysis/ex2_lambda/offpath_r96.json --compare outputs/analysis/ex3_mix/offpath_r192.json --grid $D/predict_rf/grid_prediction.csv -o $D/surface_r384.json

for R in MOS-1 MOE-1 MOE-3 MOP-1 MOE-2; do
  python -m cuas_sim.cli analysis stats --datafarm $D/datafarm_r384.parquet --response $R --aor ALL --alpha 0.05 --bootstrap-seed 20260808 --n-resamples 9999 -o $D/stats_r384/$R
done

python -m cuas_sim.cli analysis report --datafarm $D/datafarm_r384.parquet --responses MOS-1,MOE-1,MOE-3,MOP-1,MOE-2,MOS-2 --aor ALL --format html,json,md --bootstrap-seed 20260808 --model-seed 20260808 -o $D/report_r384
```

The rejected rung was also processed with `--factor uas_archetype_p_mr`
(`offpath_r192_pmr.json`); that output is not published, since its stopping blocks equal those
of `offpath_r192_lambda.json`. The exploratory surrogate was also run with `--model gbm`; its
outputs are not published, nor are the HTML, Markdown and PNG renderings. R = 768 was not run:
the ladder stops at the first rung that passes.

## Artifacts

| File | Contents | Used for |
|---|---|---|
| `STOPPING_CRITERION.md` | Pre-registered stopping criterion (Portuguese) | Stopping criteria of the paper |
| `design.parquet` + `.meta.json` | 32 design points; design seed and sampler | Input of `analysis run` |
| `design.provenance.json` | Command line, commit and tree state at design time | Provenance only |
| `datafarm_r384.parquet` + `.meta.json` | Accepted rung, long format (1,388,264 rows) | Every result of the study; points of Figure 2 |
| `datafarm_r192.parquet` + `.meta.json` | Rejected rung (694,132 rows) | Stopping rule and seed nesting |
| `offpath_r384_lambda.json`, `offpath_r384_pmr.json` | Stopping rule per point, seed nesting, per-point results, average slope of each factor, cost per point | Per-point results, average slopes, rocket shortage |
| `offpath_r192_lambda.json` | The same blocks at the rejected rung | Rejected rung (worst half-width 0.0294) |
| `surface_r384.json` | Bilinear and quadratic surfaces, F test, slope in p_mr by λ, frontier, cross-checks | Interaction, lines of Figure 2, cross-checks |
| `stats_r384/<measure>/report.json` | Test selected by the pre-checks, p-value, interval, per measure | Test selection |
| `report_r384/report.json` | Consolidated report: master table, tests binned by factor, relationships | Binned tests, correlations |

The numbers of the paper derived from this surface (slopes at the realized ends of λ,
declines, cross term, tercile slopes) are recomputed and checked against `surface_r384.json`
by `scripts/paper_derived_numbers_audit.py`, which writes
`outputs/verification/surface_derived_audit.json`.
