# EX-1 — Nominal Operating Point: Manifest

Reproduction record of the study described in [`BRIEF.md`](BRIEF.md). EX-1 has no separate
`STOPPING_CRITERION.md`: its stopping criterion is stated below.

## Execution

- Executed on 2026-08-06, at development commit `c9623c1` (development repository, not
  public; the same value is recorded as `git_head` in `design.provenance.json`), with no
  change under `src/`.
- Environment: the reference environment of
  [`docs/reproducibility.md`](../../../docs/reproducibility.md#environment-reference)
  (Windows 11, Python 3.14.3, the pinned `requirements.txt`).
- The regression ruler (`scripts/batch_seeds.py --seeds 20`) matched its reference values
  before the study.

## Seeds

| Role | Value | Note |
|---|---|---|
| Design seed | 0 | Inert: the one-point design draws nothing |
| Base seed | 20260804 | Root of the seed bank of the replications |
| Bootstrap seed | 20260804 | BCa intervals and achieved significance levels |
| Model seed | 20260804 | Not used: no surrogate for a single design point |

The first five replication seeds are 4267172748, 3524690371, 79349626, 2560020338 and
705804830.

## Design

- Sampler `fixed`: one design point, λ = 6.0 UAS/h exactly (`lambda_per_sec` = 6.0/3600;
  the scenario file stores the same value rounded, 0.001666667) and p_mr = 0.5.
- 100 paired replications, extended to 200.
- Base configuration `--config configs/zida_shield_default.yaml --sysml sysml`; no
  `--max-arrivals`, so each replication ends with the spawn window; `--workers 1`, since
  parallelism is per design point.

## Stopping criterion

Declared before the first batch:

- **Primary:** half-width of the 95% BCa bootstrap interval of ΔMOS-1 (`--aor ALL`) at most
  **0.025**, a quarter of the +0.10 reference margin (labelled "Spec 015 AC-3" in the JSON
  files).
- **Ladder:** N ∈ {100, 200, 400}; stop at the first rung that passes.
- **Confirmatory:** drift of the mean between rungs at most 0.25 × the half-width, and
  relative change of the mean and of the standard deviation below 5%.
- α = 0.05, two-sided; intervals at 95%.

| Rung | Mean | Standard deviation | 95% BCa | Half-width | Verdict |
|---|---|---|---|---|---|
| N = 100 | +0.189628 | 0.128551 | [+0.16516, +0.21536] | 0.025097 | rejected |
| **N = 200** | **+0.185314** | 0.125780 | **[+0.16828, +0.20308]** | **0.017399** | **accepted** |

Confirmatory at N = 200: drift 0.004314 against a budget of 0.004350; relative change of the
mean 2.28% and of the standard deviation 2.16%. Seed nesting: the first 100 replications of
the N = 200 batch are identical, row by row and seed by seed, to the N = 100 batch.

## Commands

Run from the repository root.

```bash
D=outputs/analysis/ex1_mc_nominal

python scripts/make_study_design.py --fixed lambda_per_h=6.0,uas_archetype_p_mr=0.5 --design-seed 0 -o $D/design.parquet

python -m cuas_sim.cli analysis run --design $D/design.parquet --reps 100 --seed 20260804 --config configs/zida_shield_default.yaml --sysml sysml --workers 1 -o $D/datafarm_n100.parquet
python -m cuas_sim.cli analysis run --design $D/design.parquet --reps 200 --seed 20260804 --config configs/zida_shield_default.yaml --sysml sysml --workers 1 -o $D/datafarm_n200.parquet

python scripts/campaign_postprocess.py --current $D/datafarm_n200.parquet --previous $D/datafarm_n100.parquet --bootstrap-seed 20260804 -o $D/offpath_n200.json

for R in MOS-1 MOE-1 MOE-3 MOP-1 MOE-2; do
  python -m cuas_sim.cli analysis stats --datafarm $D/datafarm_n200.parquet --response $R --aor ALL --alpha 0.05 --bootstrap-seed 20260804 --n-resamples 9999 -o $D/stats_n200/$R
done

python -m cuas_sim.cli analysis report --datafarm $D/datafarm_n200.parquet --responses MOS-1,MOE-1,MOE-3,MOP-1,MOE-2,MOS-2 --aor ALL --format html,json,md --bootstrap-seed 20260804 --model-seed 20260804 -o $D/report_n200
```

`analysis stats --response MOS-2` is not run (MOS-2 has no paired row; `analysis report` and
`campaign_postprocess.py` treat it). `analysis predict` is not run: the surrogate is fitted
over design points, and there is one. `analysis stats` was also run on MOS-1 at the
rejected rung; its output is not published, since the `previous` block of
`offpath_n200.json` records the same values. The HTML, Markdown and PNG renderings written by
`analysis stats` and `analysis report` are not published.

## Artifacts

| File | Contents | Used for |
|---|---|---|
| `design.parquet` + `.meta.json` | The design point; design seed and sampler | Input of `analysis run` |
| `design.provenance.json` | Command line, commit and tree state at design time | Provenance only; changes with the commit |
| `datafarm_n200.parquet` + `.meta.json` | 200 paired replications, long format (22,618 rows) | Every result of the study |
| `datafarm_n100.parquet` + `.meta.json` | The rejected rung (11,366 rows) | Stopping rule and seed nesting |
| `offpath_n200.json` | Both rungs of the stopping rule, seed nesting, MOS-2, cost blocks | Stopping, MOS-2, rockets per engagement, leaks |
| `stats_n200/<measure>/report.json` | Test selected by the pre-checks, p-value, BCa interval, per measure | Test selection |
| `report_n200/report.json` | Consolidated report: master table, hypothesis blocks, validity | Δ and intervals of the nominal point |

All JSON artifacts have `field_version` 1.0.
