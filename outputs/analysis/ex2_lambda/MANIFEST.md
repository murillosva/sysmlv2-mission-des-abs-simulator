# EX-2 — Intensity Sweep: Manifest

Reproduction record of the study described in [`BRIEF.md`](BRIEF.md). The stopping
criterion was written in [`STOPPING_CRITERION.md`](STOPPING_CRITERION.md) (in Portuguese)
before the design was generated and before the first batch, and is published unchanged.

## Execution

- Executed on 2026-08-06, at development commit `f11346b` (development repository, not
  public; recorded as `git_head` in `design.provenance.json`), with no change under `src/`.
- Environment: the reference environment of
  [`docs/reproducibility.md`](../../../docs/reproducibility.md#environment-reference).
- The regression ruler (`scripts/batch_seeds.py --seeds 20`) matched its reference values
  before and after the study.

## Seeds

| Role | Value | Note |
|---|---|---|
| Design seed | 20260806 | Draws the λ values of the Latin hypercube |
| Base seed | 20260806 | Disjoint from EX-1's (20260804), so no design point reuses the EX-1 seed bank |
| Bootstrap seed | 20260806 | BCa intervals and achieved significance levels |
| Model seed | 20260806 | Exploratory surrogate (outputs not published) |

The design seed is recorded in `design.parquet.meta.json` and in the `design` block of the
`offpath_*.json` files (`validity.seeds.design` of the report is `null`, see
[`docs/experimental_campaign.md`](../../../docs/experimental_campaign.md#5-reading-the-published-reports)).

## Design

- Sampler `LatinHypercube+fixed` (`scipy.stats.qmc.LatinHypercube`, `scramble=True`):
  `lambda_per_h` over [2, 18], `uas_archetype_p_mr` fixed at 0.5.
- 16 design points (listed with their results in the BRIEF); realized range
  λ ∈ [2.9922, 17.5417].
- 48 paired replications per point, extended to 96.
- Base configuration `--config configs/zida_shield_default.yaml --sysml sysml`; no
  `--max-arrivals`; `--workers 6`.

## Stopping criterion

Summary of `STOPPING_CRITERION.md`:

- **Primary:** maximum over the 16 design points of the half-width of the 95% BCa interval
  of ΔMOS-1 (`--aor ALL`) at most **0.050**, half of the +0.10 reference margin.
- **Ladder:** R ∈ {48, 96, 192}; stop at the first rung that passes.
- **Confirmatory, not triggering:** (i) half-width of the interval of the total change of
  ΔMOS-1 over the range at most 0.050; (ii) Q² of the surrogate at least 0.50; (iii) drift of
  the mean at most 0.25 × the half-width, and relative change of mean and standard deviation
  below 5%.

| Rung | Maximum half-width | Binding point | Points above the target | Verdict |
|---|---|---|---|---|
| R = 48 | 0.066524 | 12 (λ = 4.018) | 2 of 16 | rejected |
| **R = 96** | **0.042498** | 12 (λ = 4.018) | **0 of 16** | **accepted** |

| Confirmatory at R = 96 | Target | Obtained | Result |
|---|---|---|---|
| (i) total change | ≤ 0.050 | 0.0357 | pass |
| (ii) Q² of the surrogate | ≥ 0.50 | 0.2014 as reported (mean of per-fold R²); 0.9138 pooled | fails as reported; non-triggering |
| (iii) drift of the mean | ≤ 0.0017 | 0.0011 | pass |
| (iii) relative change of mean / standard deviation | < 5% | 0.62% / 2.96% | pass |

Seed nesting: at every design point, the first 48 replications of the R = 96 batch are
identical, row by row and seed by seed, to the R = 48 batch.

## Commands

Run from the repository root.

```bash
D=outputs/analysis/ex2_lambda

python scripts/make_study_design.py --lhs lambda --samples 16 --design-seed 20260806 --fixed p_mr=0.5 -o $D/design.parquet

python -m cuas_sim.cli analysis run --design $D/design.parquet --reps 48 --seed 20260806 --config configs/zida_shield_default.yaml --sysml sysml --workers 6 -o $D/datafarm_r48.parquet
python -m cuas_sim.cli analysis run --design $D/design.parquet --reps 96 --seed 20260806 --config configs/zida_shield_default.yaml --sysml sysml --workers 6 -o $D/datafarm_r96.parquet

python scripts/campaign_postprocess.py --current $D/datafarm_r48.parquet --factor lambda_per_h --design $D/design.parquet --half-width 0.05 --ladder 48,96,192 --bootstrap-seed 20260806 -o $D/offpath_r48.json
python scripts/campaign_postprocess.py --current $D/datafarm_r96.parquet --previous $D/datafarm_r48.parquet --factor lambda_per_h --design $D/design.parquet --half-width 0.05 --ladder 48,96,192 --bootstrap-seed 20260806 -o $D/offpath_r96.json

for R in MOS-1 MOE-1 MOE-3 MOP-1 MOE-2; do
  python -m cuas_sim.cli analysis stats --datafarm $D/datafarm_r96.parquet --response $R --aor ALL --alpha 0.05 --bootstrap-seed 20260806 --n-resamples 9999 -o $D/stats_r96/$R
done

python -m cuas_sim.cli analysis report --datafarm $D/datafarm_r96.parquet --responses MOS-1,MOE-1,MOE-3,MOP-1,MOE-2,MOS-2 --aor ALL --format html,json,md --bootstrap-seed 20260806 --model-seed 20260806 -o $D/report_r96
```

The exploratory surrogate (`analysis predict --model rf` and `--model gbm`, with
`--seed 20260806 --grid-resolution 25`) was also run; its outputs are not published. The
HTML, Markdown and PNG renderings are not published either.

## Artifacts

| File | Contents | Used for |
|---|---|---|
| `STOPPING_CRITERION.md` | Pre-registered stopping criterion (Portuguese) | Stopping criteria of the paper |
| `design.parquet` + `.meta.json` | 16 design points; design seed and sampler | Input of `analysis run` |
| `design.provenance.json` | Command line, commit and tree state at design time | Provenance only |
| `datafarm_r96.parquet` + `.meta.json` | Accepted rung, long format (174,303 rows) | Every result of the study |
| `datafarm_r48.parquet` + `.meta.json` | Rejected rung (87,190 rows) | Stopping rule and seed nesting |
| `offpath_r96.json` | Stopping rule per point, seed nesting, per-point results, trend, cost per point | Curve, trend, cost |
| `offpath_r48.json` | The same blocks at the rejected rung | Rejected rung |
| `stats_r96/<measure>/report.json` | Test selected by the pre-checks, p-value, interval, per measure | Test selection |
| `report_r96/report.json` | Consolidated report: master table, tercile tests, relationships | Tercile tests, correlations |
