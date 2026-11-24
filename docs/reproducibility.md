# Reproducibility Guide

This guide reproduces the published artifacts of the experimental campaign and of the
verification checks, at four levels of effort. All commands run from the repository
root, in an environment installed as in the root [`README.md`](../README.md)
(Quickstart). The study records are in English, except the pre-registered stopping criteria
of EX-2 to EX-4 (`STOPPING_CRITERION.md`), which are published unchanged, in Portuguese.

> **Run on AC power.** On battery, the simulator ran about 23 times slower on the
> reference machine: the same 6,144 paired replications of EX-4 took 3,507 s on
> battery and 152 s on AC power (measured on 2026-11-19).

---

## Level 0 — Inspect the committed artifacts (no execution)

| Folder | Contents |
|--------|----------|
| [`outputs/analysis/ex1_mc_nominal/`](../outputs/analysis/ex1_mc_nominal/) | EX-1: Monte Carlo at the nominal point (λ = 6 UAS/h, p_mr = 0.5), N = 200 paired replications |
| [`outputs/analysis/ex2_lambda/`](../outputs/analysis/ex2_lambda/) | EX-2: Latin hypercube over the arrival rate λ, 16 design points × 96 replications |
| [`outputs/analysis/ex3_mix/`](../outputs/analysis/ex3_mix/) | EX-3: Latin hypercube over the multi-rotor fraction p_mr, 20 design points × 192 replications |
| [`outputs/analysis/ex4_interaction/`](../outputs/analysis/ex4_interaction/) | EX-4: two-dimensional Latin hypercube over λ and p_mr, 32 design points × 384 replications |
| [`outputs/verification/`](../outputs/verification/) | Artifacts of the verification checks: requirements, phases (with the logs of the re-executed replications), calculations, adherence of the incursion generator, border geometry; numbers derived from the EX-4 surface and the reformatting test of Section 3.2 |

Each study folder has a `BRIEF.md` (results and their reading), a `MANIFEST.md` (seeds,
design, stopping criterion, exact commands, artifacts) and, for EX-2 to EX-4, a
`STOPPING_CRITERION.md` (see [Stopping criteria](#stopping-criteria)).

In a multi-point design, read the results per design point (`by_design_point` in the
`offpath_*.json` files): the aggregate block over all points is a mean of the sampled
space, not an operating point.

---

## Where each result of the paper comes from

Every number printed in the paper is in a published artifact, or follows from one by the
formula given. `<EX-n>` stands for `outputs/analysis/<study folder>/`.

| Paper element | Artifact |
|---|---|
| Table 1, measures | [`metrics.md`](metrics.md); `src/cuas_sim/metrics/aggregator.py` |
| Figure 1, transformation chain of the APKWS part definition | `sysml/Technical/TechnicalComponentsPackage.sysml`; `src/cuas_sim/sysml_parser.py`, `src/cuas_sim/sysml_loader.py` |
| Verification: requirements (14 bound, 9 on model values, 18 injected violations) | `outputs/verification/sysml_requirements_audit.json` |
| Verification: phase audit (69 reruns, 828 logs, 2,484 injected violations) | `outputs/verification/phase_audit.json`, logs in `outputs/verification/phase_audit_runs/` |
| Verification: calculation guard (3 of 13 changes detected) | `outputs/verification/sysml_calculations_audit.json` |
| Verification: seed nesting, point by point and seed by seed | nesting block of each `offpath_*.json` |
| Experimental setup: EX-4 rungs (worst half-width 0.0294 at R = 192, 0.0195 at R = 384) | EX-4 `offpath_r192_lambda.json`, `offpath_r384_*.json` (`by_design_point`) |
| Experimental setup: test selected by the Shapiro–Wilk pre-check | `stats_*/<response>/report.json` of each study |
| Table 2, campaign summary | `design.parquet` (points, realized ranges) and `datafarm_*.parquet` (pairs) of each study; central results in the rows below |
| Table 3, nominal point | EX-1 `datafarm_n200.parquet` (level per arm), `report_n200/report.json` (Δ and 95% BCa interval), `offpath_n200.json` (MOS-2, rockets per engagement) |
| Intensity sweep: ΔMOS-1 from +0.2425 to +0.1519, slope, correlation, smallest lower bound; MOE-3 (p = 0.071) and MOE-1 (correlation −0.79) | EX-2 `offpath_r96.json`, `report_r96/report.json` |
| Composition sweep: ΔMOS-1 from +0.0154 to +0.4510, slope, 29.4×, correlation; MOE-3 at the fixed-wing extreme; MOP-1 (+429, +164); rockets per engagement (2.48, 2.81) | EX-3 `offpath_r192.json`, `report_r192/report.json` (per-point means of each measure); the interval of MOE-3 at the fixed-wing extreme is recomputed from `datafarm_r192.parquet`, as stated in the EX-3 `BRIEF.md` |
| Interaction: b3 = −0.015027 with interval, F(1,28) = 66.90 and p; slope +0.4318 at λ = 6 | EX-4 `surface_r384.json` (`surface_bilinear`, including `mix_slope_by_lambda`) |
| Interaction: slopes +0.4864 and +0.2579 at the realized ends of λ (fall of 47%); fall of 40.3% [26.2%; 51.6%] from λ = 6 to 17.571; cross term −0.2247; tercile slopes +0.45, +0.37, +0.29 | `outputs/verification/surface_derived_audit.json` (computed from the bilinear fit of EX-4: slope in p_mr = b2 + b3·λ, cross term = b3 × λ range × p_mr range; its guards reproduce `surface_r384.json` exactly) |
| Section 3.2, reformatting test (the regular-expression extractor recognizes one definition instead of 22; the syntax tree stays identical) | `outputs/verification/reformat_audit.json`; the reformatted input in `outputs/verification/reformat_runs/` |
| Two-dimensional design: slope on λ alone (−0.00450); rocket shortage (2.378 UAS per replication); gain positive at all 32 points, ratio of about 44× | EX-4 `offpath_r384_lambda.json`, `offpath_r384_pmr.json` (`by_design_point`, `trend`, leaks) |
| Figure 2, interaction by tercile | EX-4 `datafarm_r384.parquet` (points) and `surface_r384.json` (lines) |
| Stopping criteria | [Stopping criteria](#stopping-criteria); `STOPPING_CRITERION.md` and `MANIFEST.md` of each study |


---

## Level 1 — Verification checks (about 1 minute)

The three checkers run outside the execution path, as standalone subcommands; none of
them runs the simulation.

```bash
cuas-sim check-requirements --config configs/zida_shield_default.yaml
cuas-sim check-calculations
cuas-sim check-conformance --run outputs/verification/phase_audit_runs/EX-4/dp19_rep0
```

Each exits with code 0 on success and 1 on a violation. The published verification
artifacts are regenerated by three scripts, which write into `outputs/verification/`:

```bash
python scripts/sysml_checks_audit.py        # requirements, phases and calculations (~15 s)
python scripts/miranda_adherence_audit.py   # adherence of the incursion generator (~4 s)
python scripts/border_geometry_audit.py     # border geometry of the generator
python scripts/paper_derived_numbers_audit.py  # derived numbers of the paper (~6 s)
```

`scripts/sysml_checks_audit.py` re-executes replica 0 of each of the 69 design points of
the campaign with event logging, and audits the logged state streams; the logs are
written with a fixed gzip timestamp, so they are byte-deterministic. After running the
scripts, `git status --short outputs/verification` prints nothing when every regenerated
file matches the committed one.

### Regression ruler (about 5 seconds)

```bash
python scripts/batch_seeds.py --seeds 20
```

Twenty paired runs (seeds 1000 to 1019, nominal scenario) of the current engine. Reference
values on the reference platform, also printed on Linux by the CI:

| Quantity | Baseline | Modernized |
|---|---|---|
| Mean arrivals per run | 18.2 | 18.2 |
| MOS-1 | 0.1140 | 0.3518 |
| MOS-1 gap (modernized − baseline) | | +0.2378 |
| MOS-2 | | +0.2812 |
| MOE-1 | 0.6729 | 0.8218 |
| MOE-2 | 0.5131 | 0.3717 |
| MOE-3 | 0.2907 | 0.5961 |

Any change in these values means that the engine no longer behaves as the one that produced
the published studies.

---

## Level 2 — Reproduce EX-1 (about 1 minute)

The commands of the [EX-1 MANIFEST](../outputs/analysis/ex1_mc_nominal/MANIFEST.md),
section "Commands":

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

The commands overwrite the committed files, so git itself does the comparison (the
regenerated HTML, Markdown and PNG renderings of the reports are not published, so the
first command ignores untracked files):

```bash
git status --short --untracked-files=no outputs/analysis/ex1_mc_nominal
git diff outputs/analysis/ex1_mc_nominal
```

Expected result on the reference platform: only `design.provenance.json` changes, and
only in its provenance fields (see [Expected variance](#expected-variance)); every other
file is byte-identical. On other operating systems, see [Other platforms](#expected-variance).

---

## Level 3 — The whole campaign (about 20 to 25 minutes)

Run the studies in the order **EX-1 → EX-2 → EX-3 → EX-4**: the surface model of EX-4
(`scripts/surface_model.py --compare`) reads `ex2_lambda/offpath_r96.json` and
`ex3_mix/offpath_r192.json`. The exact commands of each study are in the section
"Commands" of its MANIFEST:

- [`ex1_mc_nominal/MANIFEST.md`](../outputs/analysis/ex1_mc_nominal/MANIFEST.md)
- [`ex2_lambda/MANIFEST.md`](../outputs/analysis/ex2_lambda/MANIFEST.md)
- [`ex3_mix/MANIFEST.md`](../outputs/analysis/ex3_mix/MANIFEST.md)
- [`ex4_interaction/MANIFEST.md`](../outputs/analysis/ex4_interaction/MANIFEST.md)

Hardware notes, from the reference machine (12 logical CPUs, 7.36 GiB of RAM):

- `--workers 6` is the ceiling for `analysis run` with this much memory; parallelism is
  per design point, so it has no effect on EX-1, which has one point.
- Never run `scripts/campaign_postprocess.py` (or `analysis stats`) at the same time as an
  `analysis run --workers 6`: the BCa interval of the aggregate block allocates a jackknife
  of size (n, n−1), about 1.2 GiB for the 12,288 pairs of EX-4.
- A study that is to be compared with the paper must pass `--config
  configs/zida_shield_default.yaml --sysml sysml` to `analysis run`. Without them the run
  falls back to the simulator defaults, with a synthetic border (this is what happened
  in the pilot study that the EX-2 and EX-3 records cite).
- `analysis run` without `--max-arrivals` lets each replication end by the spawn window;
  capping arrivals would flatten the effect of λ.

Measured on the reference machine (from a `git archive` of the development repository,
with the reference environment):

| Study | Simulated pairs | Simulation time of the original run (AC power) | Measured in the verification of 2026-11-19/20 | Byte comparison |
|---|---|---|---|---|
| EX-1, nominal point | 300 (N = 100 and N = 200) | 53 s | 59 s, the whole pipeline, AC power | all identical except 1 file, which differs only in provenance |
| EX-2, intensity λ | 2,304 | 40 s | 211 s, on battery | all identical except 3 files, which differ only in provenance |
| EX-3, threat mix | 5,760 | 82 s | 367 s, on battery | all identical except 3 files, which differ only in provenance |
| EX-4, λ × mix | 18,432 | 691 s | rung R = 192: 3,507 s on battery; replications 192–383 (6,144 pairs): 152 s on AC power; analysis of R = 384: about 400 s on AC power | all identical except 4 files, which differ only in provenance |

The comparison skips the hand-written records (`BRIEF.md`, `MANIFEST.md`,
`STOPPING_CRITERION.md`).

The `surface_r384.json` of EX-4 records as an input the grid of the exploratory surrogate
(`predict_rf/grid_prediction.csv`), which is not published. To regenerate that file byte for
byte, run the `analysis predict` command of the EX-4 MANIFEST before the surface command.

---

## Expected variance

| Component | Reproducibility |
|-----------|----------------|
| Simulation, same seed, same code and same platform | **Bit-identical**: every random number comes from one seed registry; world streams are shared by the two configurations and decision streams are drawn per configuration |
| Data farm, any number of workers | **Bit-identical**: parallel runs are written in a canonical order |
| Tests, intervals and bootstrap resamples | **Bit-identical**: every resampling seed is recorded in the artifact |
| Provenance fields: `git_head`, `git_src_clean`, `git_tree_clean` and, in EX-4, the echo of `argv` (in `design.provenance.json` and in the `design.provenance` block of the `offpath_*.json` files) | Change whenever the study is reproduced from another commit or from a tree without git |

**Line endings.** The repository stores text with LF. The data artifacts are written in
text mode, so their line endings follow the platform; with `core.autocrlf=true` on
Windows (the Git for Windows default) and the default on Linux, a checkout matches what a
regeneration writes on the same platform, and `git status` ignores line-ending
differences. The `.sysml` files and `scripts/data/*.json` are LF on every platform.

**Other platforms.** Byte identity was verified only on the reference platform (Windows
11). On Linux, with the same pinned library versions, the files written by the simulation
were not byte-identical (observed in the CI), most likely because floating-point results
differ in the last digits between operating systems. The two bit-exact regression tests
(`tests/unit/test_crn_determinism_017.py` and
`tests/property/test_uas_spawn_window_properties.py`, 23 cases) therefore run only on the
reference environment and are skipped elsewhere; the CI prints the 20-seed regression
ruler (`scripts/batch_seeds.py --seeds 20`) on Linux, and in the run of 2026-11-21 it
matched the [reference values](#regression-ruler-about-5-seconds) to the four printed
decimals.

**Development history.** This repository starts from a single commit. The commits cited
in the study records (for example under "Execution", or the `git_head` of the
provenance fields) belong to the development repository, which is not public.

---

## Environment reference

| Component | Version used to verify the published artifacts |
|-----------|------------------------------------------------|
| Operating system | Windows 11 (10.0.26200), 12 logical CPUs, 7.36 GiB of RAM |
| Python | 3.14.3 |
| `simpy` | 4.1.1 |
| `numpy` | 2.4.4 |
| `scipy` | 1.17.1 |
| `pandas` | 3.0.3 |
| `pyarrow` | 24.0.0 |
| `scikit-learn` | 1.9.0 |
| `scikit-posthocs` | 0.14.0 |
| `jinja2` | 3.1.6 |
| Full pinned list | [`requirements.txt`](../requirements.txt) (54 packages) |

The four studies ran in this environment (section "Execution" of each MANIFEST). The CI workflow
(`.github/workflows/ci.yml`) runs the test suite on Python 3.11 and 3.12 (Ubuntu).

---

## Stopping criteria

**Rule.** The four studies share one rule: the half-width of the 95% BCa bootstrap
confidence interval of the mean ΔMOS-1 (all areas of responsibility, `--aor ALL`) must
fall below a target. In multi-point studies, the rule applies to the **maximum over the
design points**. The target is a fraction of a +0.10 reference margin: ½ (0.050) when
every point lies far from the margin, and ¼ (0.025) when the curve crosses it. The
margin is the acceptance criterion set on the ΔMOS-1 gap when multi-shot re-engagement was
implemented (labelled "Spec 015 AC-3" in the records): an implementation acceptance
criterion, **not an operational requirement**.

| Study | Where the criterion is | Target | Ladder | Rejected rung | Accepted rung |
|---|---|---|---|---|---|
| EX-1 (λ = 6/h, p_mr = 0.5) | `ex1_mc_nominal/MANIFEST.md`, section "Stopping criterion"; confirmatory: drift ≤ 0.25 × half-width and relative change of mean and standard deviation < 5% | 0.025 | N ∈ {100, 200, 400} | N = 100: 0.025097 | **N = 200: 0.017399** |
| EX-2 (λ, 16 points) | `ex2_lambda/STOPPING_CRITERION.md` | 0.050 | R ∈ {48, 96, 192} | R = 48: max 0.066524 (2 of 16 points fail) | **R = 96: max 0.042498** (point 12) |
| EX-3 (mix, 20 points) | `ex3_mix/STOPPING_CRITERION.md` | 0.025 | R ∈ {96, 192, 384, 768} | R = 96: max 0.032724 (12 of 20) | **R = 192: max 0.021827** (point 8) |
| EX-4 (λ × mix, 32 points) | `ex4_interaction/STOPPING_CRITERION.md`; single primary criterion (the interaction coefficient was rejected as a primary criterion) | 0.025 | R ∈ {192, 384, 768, 1536} | R = 192: max 0.029378 (3 of 32) | **R = 384: max 0.019518** (point 19) |

The values come from `by_design_point.max_half_width` and
`by_design_point.all_points_pass` of the `offpath_*.json` files. Do not read the decision
from `stopping.verdict`: it belongs to the aggregate block (`stopping.scope` says so) and
says "ACEITAR" even at rejected rungs.

**Anteriority.** Each record states that it was written before the first batch of its
study. Version control corroborates this only for EX-3 and EX-4, whose criterion files
were committed before their first batch in the development repository, which is not
public. For EX-1, the general protocol was committed before the first batch, without the
numeric target; for EX-2, the criterion file entered with the study's closing commit.

| Study | Criterion registered in the development repository | First data batch (file time) | Blob of the criterion file |
|---|---|---|---|
| EX-1 | general protocol in `8291a2b` (2026-08-04 21:11), without the value 0.025; the MANIFEST with the value entered in `3b2d8c7` (2026-08-06 21:38) | `datafarm_n100`, 2026-08-04 21:28 | MANIFEST `9a3d321db837751a2fb10fd3dc39aaf21be0dcfe` |
| EX-2 | `3b2d8c7` (2026-08-06 21:38), closing commit of the study | `datafarm_r48`, 2026-08-06 21:16 | `8e04e07f817eec8fd7272cad275d95fab19368c3` |
| EX-3 | `18b0446` (2026-08-06 22:06) | `datafarm_r96`, 2026-08-06 22:08 | `292de8cb6c9f840a981dda611424ea7451356666` |
| EX-4 | `f076450` (2026-08-07 10:24) | `datafarm_r192`, 2026-08-07 10:29 | `02cc5d7e2ff5370cfa9f1e11f20320b235a403fc` |

The `STOPPING_CRITERION.md` files of EX-2 and EX-3 are published unchanged: `git ls-tree
HEAD outputs/analysis/ex2_lambda/STOPPING_CRITERION.md` (and the same for `ex3_mix`)
shows the blob IDs of the table. Being unchanged, they also cite records that are not
published: the pilot study (`outputs/analysis/zida_shield_me_1/`) and, in EX-2, the cost
measurement of EX-1 (`bench_cost.json`) used to estimate the run time. The EX-4 criterion
file carries a publication note and editorial changes, listed in that note, so its blob
differs from the registered one. The EX-1 MANIFEST was rewritten in English for this
release, like the other BRIEF and MANIFEST files, with the same criterion, rungs and values;
its blob differs too.
