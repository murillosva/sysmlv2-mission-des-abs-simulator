# SysML v2 Mission DES–ABS Simulator

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)
![DOI](https://img.shields.io/badge/DOI-TBD%20%28upon%20final%20acceptance%29-lightgrey.svg)

> Paired discrete-event + agent-based simulator for the **Continental
> ZIDA Shield Operation**, a counter-UAS (C-UAS) airspace policing mission over
> the Brazilian Amazon border, used to quantify the operational impact of
> modernizing the A-29 Super Tucano (baseline → A-29N) with C-UAS avionics
> (M3AR datalink, FLIR Star SAFIRE II EO/IR, APKWS laser-guided rocket).
>
> Companion artifact to the ScitePress paper *From MBSE Mission Model to Executable
> Architecture: An Exploration of Hybrid Simulation in Defense*, accepted at the
> **15th International Conference on Model-Based Software and Systems Engineering
> (MODELSWARD 2027)**, 19–21 February 2027, Valletta, Malta.

---

## Overview

The simulator executes the SysML v2 mission model of the case study in three stages.

1. **Transformation (model → configuration).** A tokenizer and a recursive-descent
   parser build a typed syntax tree of the textual notation
   (`src/cuas_sim/sysml_parser.py`). The loader (`sysml_loader.py`) resolves the mission
   configuration in increasing precedence: simulator defaults < scenario file
   (`configs/zida_shield_default.yaml`) < model; the model prevails in the technical
   fields it supplies. Three checkers use the same tree outside the execution path, as
   standalone subcommands:
   - `cuas-sim check-requirements` evaluates each bound `require constraint` against the
     resolved configuration;
   - `cuas-sim check-conformance` audits the logged state stream of each aircraft against
     the mission phases declared in the model;
   - `cuas-sim check-calculations` compares the `calc def` expressions of the model with
     the functions that compute the measures.
2. **Hybrid simulation (DES + ABS), paired.** Discrete-event simulation (SimPy) gives the
   temporal skeleton: incursion arrivals, the 15-state life cycle of each aircraft and
   serial engagement. Agents take the local decisions to pursue, fire or leave on fuel.
   Each paired run executes the baseline A-29 and the modernized A-29N against the same
   incursions, under common random numbers: four world streams (interarrival time, entry
   sector, origin and heading, archetype) are drawn once and shared, and four decision
   streams (update cadence, first-shot philosophy and distance, shot interval, shot
   outcome) are drawn per configuration; a fifth decision stream is reserved and never
   consumed.
3. **Statistical analysis layer.** Design of experiments (Latin hypercube), execution
   control (paired replications from disjoint, nested seed banks, with stopping criteria
   declared before execution) and analysis (test selection by pre-checks, bias-corrected and
   accelerated (BCa) bootstrap intervals, and a bilinear surface for the interaction in `scripts/surface_model.py`),
   reimplementing the first three steps proposed by Dantas et al. (2024) for the AsaPy
   library. The fourth step, prediction (`cuas-sim analysis predict`), is an exploratory
   study that was not concluded: it was not validated against the literature on
   predictive modelling, and it is not part of the paper nor of the master's thesis to
   which the paper belongs.

The SysML v2 model the simulator consumes was generated in our
previous work, the AIAA DATC/IEEE Digital Avionics Systems Conference (DASC 2026) paper
*"Digital Avionics Operations in Mission Simulation:
an AI-assisted MBSE Modeling Approach"*, by the pipeline published at
<https://github.com/murillosva/sysmlv2-mission-multi-agent-system>; the provenance of the
model and its three differences from that snapshot are in [`sysml/README.md`](sysml/README.md).

The documentation and the study records are in English, except the pre-registered stopping
criteria, published unchanged in Portuguese; see the index in
[`docs/README.md`](docs/README.md), which also has a
[glossary](docs/README.md#glossary) of the terms and acronyms used across the repository.

---

## Measures

The mission measures follow the hierarchy of Table 1 of the paper: measures of success
(MOS), effectiveness (MOE) and performance (MOP). The ZIDA is the continental Air Defense
Identification Zone, a band from the border line to 80 nautical miles inland. All are computed by
`MetricsAggregator` in [`src/cuas_sim/metrics/aggregator.py`](src/cuas_sim/metrics/aggregator.py).
The same definitions, with the function that computes each measure, are in
[`docs/metrics.md`](docs/metrics.md).

| Level | Measure | Definition | Computed by |
|-------|---------|------------|-------------|
| Success | MOS-1 | Fraction of incursions neutralized before crossing the 80-nautical-mile boundary of the ZIDA. | `_compute_mos1` |
| Success | MOS-2 | Fraction of the baseline configuration's evasions eliminated by the alternative configuration. | `_compute_mos2` |
| Effectiveness | MOE-1 | Mean engagement-time rate: the higher the value, the smaller the share of the available policing time taken by engagements. | `_compute_moe1` |
| Effectiveness | MOE-2 | Mean active-tracking rate during pursuit, sampled between the maximum acquisition distance and the maximum employment range. | `_compute_moe2` |
| Effectiveness | MOE-3 | Mean probability that an engagement neutralizes the UAS, with aborted engagements counting as zero. | `_compute_moe3` |
| Performance | MOP-1 | Number of heading or geographic-coordinate updates received by the interceptor per engagement. | `_compute_mop1` |
| Performance | MOP-2 | Maximum acquisition distance of the configuration. | `_mop2_row` |
| Performance | MOP-3 | Weapon employment envelope, reported as the maximum employment range. | `_mop3_row` |

The paired difference Δ = alternative − baseline is computed per replication; ΔMOS-1
is the primary response of every study. MOP-2 and MOP-3 are constants of each
configuration.

---

## Repository structure

```
sysmlv2-mission-des-abs-simulator/
├── src/cuas_sim/              # the simulator package (command: cuas-sim)
│   ├── sysml_parser.py        # SysML v2 textual notation -> typed syntax tree
│   ├── sysml_loader.py        # model -> mission configuration (on top of the YAML)
│   ├── sysml_requirements.py  # requirements checker (cuas-sim check-requirements)
│   ├── sysml_state_machine.py # phase auditor (cuas-sim check-conformance)
│   ├── sysml_calculations.py  # calculation guard (cuas-sim check-calculations)
│   ├── config.py, rng.py      # frozen mission configuration; seed registry (CRN)
│   ├── simulation/            # paired run (SimPy), persistence
│   ├── agents/, engagement/   # A-29 life cycle (15 states), dispatcher, pursuit, firing
│   ├── policies/, geometry/   # tracking and weapon laws; intercept geometry
│   ├── world/                 # incursion generation (border sectors, arrivals)
│   ├── metrics/               # MOS/MOE/MOP aggregation
│   ├── analysis/              # DoE, execution control, statistics, reports
│   │                          #   (prediction.py: exploratory, see Overview)
│   ├── optimization/          # legacy optimizer–simulator loop, not used in the paper;
│   │                          #   its seed-bank helper is reused by the analysis layer
│   └── viz/                   # replay animation (cuas-sim animate)
├── sysml/                     # the consumed SysML v2 model (27 packages), MANIFEST.sha256
├── configs/                   # scenario file and border sectors
├── scripts/                   # study design, post-processing, surface model, audits
├── tests/                     # unit, property, smoke, statistical and oracle tests
├── outputs/
│   ├── analysis/              # study records and data of EX-1..EX-4
│   └── verification/          # artifacts of the verification checks
├── docs/                      # technical documentation (index in docs/README.md)
├── case-study/                # status of the case study document
├── CITATION.cff, LICENSE, NOTICE
└── pyproject.toml, requirements.txt
```

---

## Quickstart

### Path A — Explore the published results (no execution)

Each study folder under [`outputs/analysis/`](outputs/analysis/) has a `BRIEF.md`
(results and their reading) and a `MANIFEST.md` (seeds, design, stopping criterion,
exact commands, artifacts). Level 0 of [`docs/reproducibility.md`](docs/reproducibility.md)
describes each folder.

### Path B — Verification checks and EX-1 (a few minutes, on AC power)

```bash
git clone https://github.com/murillosva/sysmlv2-mission-des-abs-simulator.git
cd sysmlv2-mission-des-abs-simulator
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps

cuas-sim check-requirements --config configs/zida_shield_default.yaml
cuas-sim check-calculations
cuas-sim check-conformance --run outputs/verification/phase_audit_runs/EX-4/dp19_rep0
```

Then reproduce EX-1, the nominal point (about 1 minute), with the commands of Level 2 of
[`docs/reproducibility.md`](docs/reproducibility.md), and compare with `git status`.
Optionally, run the test suite (436 tests, about 3 to 4 minutes; outside the reference
environment, Windows with the pinned versions, the 23 bit-exact regression cases are
skipped):

```bash
pytest tests/unit tests/property tests/smoke tests/statistical -q
```

A single paired run, and its replay:

```bash
cuas-sim run --config configs/zida_shield_default.yaml --seed 42      # writes outputs/runs/<timestamp>/
cuas-sim animate outputs/runs/<timestamp>/ --config configs/zida_shield_default.yaml
```

### Path C — The whole campaign (about 20 to 25 minutes, on AC power)

Level 3 of [`docs/reproducibility.md`](docs/reproducibility.md): the studies EX-1 → EX-2
→ EX-3 → EX-4, with the exact commands of each MANIFEST.

---

## Stack and exact versions

| Dependency | Version used to verify the published artifacts | Role |
|------------|-----------------------------------------------|------|
| Python | 3.14.3 (`requires-python >= 3.11`) | |
| `simpy` | 4.1.1 | Discrete-event engine |
| `numpy` | 2.4.4 | Seed registry (`SeedSequence`), numerics |
| `scipy` | 1.17.1 | Latin hypercube, tests, BCa bootstrap |
| `pandas` / `pyarrow` | 3.0.3 / 24.0.0 | Data farm tables (Parquet) |
| `pyyaml`, `typer`, `pyproj`, `matplotlib` | see `requirements.txt` | Configuration, CLI, projection, figures |
| `scikit-learn`, `scikit-posthocs`, `jinja2` | 1.9.0, 0.14.0, 3.1.6 | Analysis layer (extra `[analysis]`) |
| `psutil` | 7.2.2 | Cost measurement (extra `[bench]`) |

[`requirements.txt`](requirements.txt) pins all 54 packages of the environment
(`pip install -e ".[dev,analysis,bench]"`), with platform markers. The reference machine
runs Windows 11; the CI workflow installs these pinned versions and runs the test suite on
Python 3.11 and 3.12 (Ubuntu).
The simulator needs no network access and no external service at run time.

---

## Reproducibility statement

Every random number comes from one seed registry: the same seed and the same code give
bit-identical outputs, and parallel data farming is bit-identical because results are
written in a canonical order. Regenerating the published artifacts from the source of
this repository, with the reference environment, matched them byte for byte except for
provenance fields (`git_head`, `git_src_clean`, `git_tree_clean`, and an echo of `argv`
in EX-4), which change with the commit; the table is in
[`docs/reproducibility.md`](docs/reproducibility.md). Byte identity was verified only on
the reference platform (Windows 11). On Linux, with the same pinned versions, files written
by the simulation were not byte-identical (observed in the CI), most likely because
floating-point results differ in the last digits between operating systems.

The sample-size rungs and precision targets of the four studies were declared before
execution; see [Stopping criteria](docs/reproducibility.md#stopping-criteria).

Practical notes from the reference machine (12 logical CPUs, 7.36 GiB of RAM):

- run on AC power: on battery the simulator ran about 23 times slower;
- use at most `--workers 6` for `analysis run`;
- do not run `scripts/campaign_postprocess.py` at the same time as an
  `analysis run --workers 6` (the BCa interval of EX-4 allocates about 1.2 GiB);
- a study meant to be compared with the paper must pass
  `--config configs/zida_shield_default.yaml --sysml sysml` to `analysis run`.

**Scope.** Beyond verification, the scenario is hypothetical and there was no empirical
validation or face validation by experts: claims are limited to internal validity. The
modelling assumptions are listed in [`docs/assumptions.md`](docs/assumptions.md).

---

## Licenses

| Component | License | Attribution |
|---|---|---|
| Simulator source code, scripts, tests, configuration files, documentation and study records | **MIT** — see [`LICENSE`](LICENSE) | Murillo S. Szvaticsek, 2026 |
| SysML v2 model in `sysml/`, except `sysml/CoSMA/` | **MIT** | Murillo S. Szvaticsek, 2026 (generated in the DASC 2026 work) |
| CoSMA framework packages (`sysml/CoSMA/`, 3 files) | **MPL 2.0** — see [`NOTICE`](NOTICE) | © 2026 AIRBUS and its affiliates — <https://github.com/airbus/apollo-11-sysml-v2> |
| Border geometry (`scripts/data/`, border coordinates in `configs/`) | Public domain | Natural Earth — see [`NOTICE`](NOTICE) |

The MPL 2.0 permits use and redistribution with preservation of the original copyright
notices. No figure or table of the accompanying paper is redistributed in this
repository.

---

## Citing this work

If you use this software, the consumed model or the study records in your research,
please cite the accompanying paper. Bibliographic details (volume, pages, DOI, ISBN) are
TBD upon final acceptance.

```bibtex
@conference{szvaticsek2027from,
  author={Murillo S. Szvaticsek and Cesar A. C. Marcondes},
  title={From MBSE Mission Model to Executable Architecture: An Exploration of Hybrid Simulation in Defense},
  booktitle={Proceedings of the 15th International Conference on Model-Based Software and Systems Engineering - Volume TBD: MODELSWARD},
  year={2027},
  pages={TBD},
  publisher={SciTePress},
  organization={INSTICC},
  doi={TBD},
  isbn={TBD},
}
```

To cite the generation of the SysML v2 model, see the `CITATION.cff` of
<https://github.com/murillosva/sysmlv2-mission-multi-agent-system>.

---

## Publication status

Accepted as a paper at MODELSWARD 2027 (paper #12); the camera-ready version has been
submitted and final acceptance is pending. The volume, pages, DOI and ISBN will be added
to this README and to `CITATION.cff` as soon as they are assigned; until then the BibTeX
entry and the DOI badge above are provisional. The case study document is reserved until
the thesis defense (see [`case-study/README.md`](case-study/README.md)).
