# docs/

Technical documentation of the simulator, in English. The study records
([`../outputs/analysis/`](../outputs/analysis/)) are in English too, except the
pre-registered stopping criteria of EX-2 to EX-4 (`STOPPING_CRITERION.md`), which are
published unchanged, in Portuguese, as are a few text fields of the JSON artifacts; the
glossary below lists their terms.

| Document | Contents |
|----------|----------|
| [`reproducibility.md`](reproducibility.md) | Reproduction guide in four levels, where each result of the paper comes from, byte-identity table, regression ruler, environment, and the stopping criteria of the four studies |
| [`architecture.md`](architecture.md) | Module map, SysML v2 to configuration mapping, execution topology |
| [`metrics.md`](metrics.md) | What each MOS/MOE/MOP measure represents, the function that computes it, and the case-study equation numbers cited in the code |
| [`assumptions.md`](assumptions.md) | Every simplification adopted in the simulator and its expected impact on the measures |
| [`engagement_model.md`](engagement_model.md) | Engagement model per configuration: envelope trigger, heading freeze, firing philosophy and distance, dual regime of the A-29N, stop conditions, bingo fuel |
| [`analysis_layer.md`](analysis_layer.md) | Statistical analysis layer: modules, commands and artifacts, factors and responses, test selection, intervals and decision rule, consolidated report |
| [`experimental_campaign.md`](experimental_campaign.md) | Design, execution control, stopping and analysis of the four studies (EX-1 to EX-4), and how to read the published reports |
| [`odd_protocol.md`](odd_protocol.md) | ODD description of the agent-based part, a technical description of the implementation (not part of the paper) |
| [`refs.bib`](refs.bib) | Bibliography: the references of the paper, and those cited by the documentation, the code and the study records |

> **"Spec NNN" labels.** Comments in the code and in the tests, the study records and a few
> recorded fields carry labels such as "Spec 015 FR-3" or paths such as
> `specs/030-…/spec.md`. They identify the development change that introduced a behaviour,
> a requirement of that change or a check. The specification documents are not part of
> this release, which presents the simulator in its final state. Recorded labels are kept
> as written: the stopping-rule rationale in the `offpath_*.json` files, the `spec` field of
> two verification artifacts, a comment in `sysml/Technical/TechnicalComponentsPackage.sysml`
> and the pre-registered stopping criteria of EX-2 and EX-3, so that these artifacts stay
> byte-identical to the ones the studies produced. In particular, "Spec 015 AC-3" is the
> +0.10 acceptance margin on the ΔMOS-1 gap set when multi-shot re-engagement was
> implemented (see [Stopping criteria](reproducibility.md#stopping-criteria)).

> **"Article N" labels.** Labels such as "Article IV" or "Article VI" name the nine design
> principles the simulator was built under, which hold in this release:
>
> | Article | Principle |
> |---|---|
> | I | Module-first: each capability is a module with clear boundaries |
> | II | CLI-observable: components can be exercised through `cuas-sim` without a full run |
> | III | Test-oracle: tests are written, and seen to fail, before production code |
> | IV | Reproducibility: every random number comes from the `RNGRegistry`; the same seed gives bit-identical output |
> | V | SysML traceability: every component traces to the SysML v2 model |
> | VI | Formula integrity: the MOS/MOE/MOP computations follow the definitions of the case study one to one |
> | VII | Simplicity: no abstractions for hypothetical future needs |
> | VIII | Integration-first: smoke tests run the real SimPy engine, without mocks |
> | IX | Schema versioning: a structural change in an output schema bumps its `field_version` |

> **Prediction (exploratory).** The surrogate-model step (`cuas-sim analysis predict`) is
> an exploratory study that was not concluded: it was not validated against the literature
> on predictive modelling, it is not part of the published paper nor of the master's thesis
> to which the paper belongs, and its outputs are not published here.

> **Paper figures.** No figure or table of the accompanying paper is redistributed in
> this repository.

---

## Glossary

Terms and labels used across the documents and the study records (some records are in
Portuguese; their terms are listed here too).

**Mission and systems**

| Term | Meaning |
|---|---|
| A-29 / A-29N | Embraer EMB 314 Super Tucano: the baseline configuration (A-29) and the modernized alternative (A-29N) |
| UAS / C-UAS | Unmanned aircraft system / counter-UAS |
| ZIDA | Air Defense Identification Zone (Portuguese acronym). The continental ZIDA is a band from the border line to 80 nautical miles inland; crossing it without being neutralized is an evasion |
| CAP | Air combat patrol station, where each aircraft holds while policing its area |
| FARP | Forward arming and refueling point |
| SBPV | ICAO code of Porto Velho airport, the main base |
| PEA, T_PEA | Airspace policing phase (Portuguese acronym) and the time available for it at the patrol station |
| AoR | Area of responsibility of a patrol station; `--aor ALL` pools all areas |
| RTB, bingo fuel | Return to base; leaving the station when the remaining fuel only suffices to reach the refueling point (zero reserve in the model) |
| Winchester | Out of rockets |
| Leak | An incursion that evades: it crosses the ZIDA boundary without being neutralized |
| E-99M | Embraer airborne early-warning aircraft, modelled as an idealized dispatcher |
| M3AR | Rohde & Schwarz digital datalink of the A-29N |
| EO/IR | Electro-optical/infrared sensor (FLIR Star SAFIRE II on the A-29N) |
| APKWS | Advanced Precision Kill Weapon System, the laser-guidance kit for 70 mm rockets of the A-29N |
| SBAT-70 | 70 mm unguided rocket launcher of the baseline |

**Measures**

| Term | Meaning |
|---|---|
| MOS / MOE / MOP | Measures of success / effectiveness / performance; definitions in [`metrics.md`](metrics.md) |
| Δ, ΔMOS-1 | Paired difference, alternative minus baseline, in the same replication; ΔMOS-1 is the primary response |

**Modelling and simulation**

| Term | Meaning |
|---|---|
| MBSE, SysML v2 | Model-based systems engineering; Systems Modeling Language, version 2 |
| CoSMA | Complex System Modeling and Architecting methodology, which organizes the model in five layers |
| ME, MEG | Mission engineering; the US Department of Defense Mission Engineering Guide |
| DES, ABS / ABM | Discrete-event simulation; agent-based simulation / agent-based model |
| CRN | Common random numbers: both configurations of a replication face the same incursions |
| ODD | Overview, Design concepts and Details: a protocol for describing agent-based models |

**Statistics**

| Term | Meaning |
|---|---|
| DoE, LHS, OFAT | Design of experiments; Latin hypercube sampling; one factor at a time |
| Data farm | The table of every measure of every paired replication at every design point (`datafarm_*.parquet`) |
| CI (IC in the Portuguese records) | Confidence interval |
| BCa | Bias-corrected and accelerated bootstrap interval |
| ASL | Achieved significance level, the bootstrap p-value (Efron and Tibshirani) |
| EDA | Exploratory data analysis |
| AsaPy | Python library for simulation analysis; the analysis layer reimplements the first three steps of its workflow (Dantas et al., 2024) |
| RF, GBM, GP | Random forest, gradient boosting machine, Gaussian process: surrogate models of the exploratory prediction step |
| Q² | Cross-validated coefficient of determination of a surrogate |
| RMSE, MAE, OLS | Root mean squared error, mean absolute error, ordinary least squares |

**Study and development labels**

| Term | Meaning |
|---|---|
| EX-1 … EX-4 | The four studies of the campaign ([`experimental_campaign.md`](experimental_campaign.md)) |
| ME-1 | Pilot data farm run before the campaign to size the replication ladders of EX-2 and EX-3; cited by their records, not published |
| Exceeds / falls short / indeterminate | A design point whose ΔMOS-1 interval lies entirely above the +0.10 reference margin / entirely below it / contains it |
| Regression ruler (régua R0 … R3) | The 20-seed batch `scripts/batch_seeds.py --seeds 20`; R3 is the current engine ([`reproducibility.md`](reproducibility.md#regression-ruler-about-5-seconds)), R0 to R2 are earlier engine versions cited in the records |
| AC-3 | The +0.10 acceptance margin on the ΔMOS-1 gap (see the "Spec NNN" note above) |
| Spec NNN, and FR, NFR, TD, AC, Q, A, P, T, INV items | A development change, and inside it a functional or non-functional requirement, technical decision, acceptance criterion, question decided by the researcher, assumption, property or test case, invariant |
| TR-xxx, MR-xxx, FR-xxx (three digits) | Identifiers of the technical, mission and functional requirements declared in the SysML v2 model (`sysml/Requirements/`) |
| SR-1 … SR-4b | Changes of the SysML integration: parser (SR-1), requirements checker (SR-2), phase auditor (SR-3), weapon parameters read from the model (SR-4), calculation guard (SR-4b) |
| AN-0 … AN-3, AN-report | Stages of the analysis layer: design (AN-0), data farming (AN-1), statistical analysis (AN-2), exploratory prediction (AN-3), consolidated report (AN-report) |
| Article N | Design principles (table above) |
