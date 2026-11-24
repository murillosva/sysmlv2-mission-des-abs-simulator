# EX-3 — Composition Sweep: Manifest

Reproduction record of the study described in [`BRIEF.md`](BRIEF.md). The stopping
criterion was written in [`STOPPING_CRITERION.md`](STOPPING_CRITERION.md) (in Portuguese)
before the design was generated, committed on its own before the first batch, and is
published unchanged.

## Execution

- Executed on 2026-08-06, at development commit `e2c5ec1`; the stopping criterion was
  committed in `18b0446` before the first `analysis run` (development repository, not
  public; the commit of the run is recorded as `git_head` in `design.provenance.json`). No
  change under `src/`.
- Environment: the reference environment of
  [`docs/reproducibility.md`](../../../docs/reproducibility.md#environment-reference).
- The regression ruler (`scripts/batch_seeds.py --seeds 20`) matched its reference values
  before and after the study.

## Seeds

| Role | Value | Note |
|---|---|---|
| Design seed | 20260807 | Draws the p_mr values of the Latin hypercube |
| Base seed | 20260807 | Disjoint from EX-1 (20260804) and EX-2 (20260806) |
| Bootstrap seed | 20260807 | BCa intervals and achieved significance levels |
| Model seed | 20260807 | Exploratory surrogate (outputs not published) |

The design seed is recorded in `design.parquet.meta.json` and in the `design` block of the
`offpath_*.json` files.

## Design

- Sampler `LatinHypercube+fixed` (`scipy.stats.qmc.LatinHypercube`, `scramble=True`):
  `uas_archetype_p_mr` over [0, 1], `lambda_per_h` fixed at 6.0.
- 20 design points (listed with their results in the BRIEF); realized range
  p_mr ∈ [0.0159, 0.9825].
- 96 paired replications per point, extended to 192.
- Base configuration `--config configs/zida_shield_default.yaml --sysml sysml`; no
  `--max-arrivals`; `--workers 6`.

## Stopping criterion

Summary of `STOPPING_CRITERION.md`:

- **Primary:** maximum over the 20 design points of the half-width of the 95% BCa interval
  of ΔMOS-1 (`--aor ALL`) at most **0.025**, a quarter of the +0.10 reference margin. The
  target is a quarter, not the half of EX-2, because the curve crosses the margin and each
  point has to be classified against it.
- **Ladder:** R ∈ {96, 192, 384, 768}; stop at the first rung that passes.
- **Confirmatory, not triggering:** (i) classification of each point against the margin,
  with no indeterminate point except next to the crossing; (ii) half-width of the interval of
  the total change at most 0.050; (iii) Q² of the surrogate at least 0.50; (iv) drift and
  relative change between rungs, as in EX-2.

| Rung | Maximum half-width | Binding point | Points above the target | Verdict |
|---|---|---|---|---|
| R = 96 | 0.032724 | 6 (p_mr = 0.7532) | 12 of 20 | rejected |
| **R = 192** | **0.021827** | 8 (p_mr = 0.6427) | **0 of 20** | **accepted** |

| Confirmatory at R = 192 | Target | Obtained | Result |
|---|---|---|---|
| (i) classification | indeterminate points only next to the crossing | 14 exceed, 4 fall short, 2 indeterminate (p_mr = 0.219 and 0.270, around the crossing at 0.241) | pass |
| (ii) total change | ≤ 0.050 | 0.0283 | pass |
| (iii) Q² of the surrogate | ≥ 0.50 | 0.9487 as reported (mean of per-fold R²); 0.9609 pooled | pass |
| (iv) drift of the mean | ≤ 0.0014 | 0.0004 | pass |
| (iv) relative change of mean / standard deviation | < 5% | 0.20% / 1.02% | pass |

Seed nesting: at every design point, the first 96 replications of the R = 192 batch are
identical, row by row and seed by seed, to the R = 96 batch.

## Commands

Run from the repository root.

```bash
D=outputs/analysis/ex3_mix

python scripts/make_study_design.py --lhs p_mr --samples 20 --design-seed 20260807 --fixed lambda=6.0 -o $D/design.parquet

python -m cuas_sim.cli analysis run --design $D/design.parquet --reps 96  --seed 20260807 --config configs/zida_shield_default.yaml --sysml sysml --workers 6 -o $D/datafarm_r96.parquet
python -m cuas_sim.cli analysis run --design $D/design.parquet --reps 192 --seed 20260807 --config configs/zida_shield_default.yaml --sysml sysml --workers 6 -o $D/datafarm_r192.parquet

python scripts/campaign_postprocess.py --current $D/datafarm_r96.parquet --factor uas_archetype_p_mr --design $D/design.parquet --half-width 0.025 --ladder 96,192,384,768 --bootstrap-seed 20260807 -o $D/offpath_r96.json
python scripts/campaign_postprocess.py --current $D/datafarm_r192.parquet --previous $D/datafarm_r96.parquet --factor uas_archetype_p_mr --design $D/design.parquet --half-width 0.025 --ladder 96,192,384,768 --bootstrap-seed 20260807 -o $D/offpath_r192.json

for R in MOS-1 MOE-1 MOE-3 MOP-1 MOE-2; do
  python -m cuas_sim.cli analysis stats --datafarm $D/datafarm_r192.parquet --response $R --aor ALL --alpha 0.05 --bootstrap-seed 20260807 --n-resamples 9999 -o $D/stats_r192/$R
done

python -m cuas_sim.cli analysis report --datafarm $D/datafarm_r192.parquet --responses MOS-1,MOE-1,MOE-3,MOP-1,MOE-2,MOS-2 --aor ALL --format html,json,md --bootstrap-seed 20260807 --model-seed 20260807 -o $D/report_r192
```

The exploratory surrogate (`analysis predict --model rf` and `--model gbm`, with
`--seed 20260807 --grid-resolution 25`) was also run; its outputs are not published, nor are
the HTML, Markdown and PNG renderings. R = 384 was not run: the ladder stops at the first
rung that passes.

## Artifacts

| File | Contents | Used for |
|---|---|---|
| `STOPPING_CRITERION.md` | Pre-registered stopping criterion (Portuguese) | Stopping criteria of the paper |
| `design.parquet` + `.meta.json` | 20 design points; design seed and sampler | Input of `analysis run` |
| `design.provenance.json` | Command line, commit and tree state at design time | Provenance only |
| `datafarm_r192.parquet` + `.meta.json` | Accepted rung, long format (428,146 rows) | Every result of the study |
| `datafarm_r96.parquet` + `.meta.json` | Rejected rung (214,013 rows) | Stopping rule and seed nesting |
| `offpath_r192.json` | Stopping rule per point, seed nesting, per-point results, trend, cost per point | Curve, crossing, trend, cost |
| `offpath_r96.json` | The same blocks at the rejected rung | Rejected rung |
| `stats_r192/<measure>/report.json` | Test selected by the pre-checks, p-value, interval, per measure | Test selection |
| `report_r192/report.json` | Consolidated report: master table, per-point means of each measure, tercile tests, relationships | Drivers, tercile tests, correlations |
