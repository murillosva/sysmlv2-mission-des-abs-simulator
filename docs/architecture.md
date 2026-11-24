# Architecture

`cuas-sim` turns a SysML v2 mission model into the configuration of a paired hybrid
discrete-event / agent-based simulation and analyzes its outputs statistically:

```
   ┌──────────────────────────┐
   │  SysML v2 model           │  27 CoSMA packages (sysml/), from the previous work
   └──────────┬───────────────┘
              │ sysml_parser → typed syntax tree → loader + 3 checkers
              ▼
   ┌──────────────────────────┐
   │  MissionConfig            │  resolved: Python defaults < YAML < model
   └──────────┬───────────────┘
              │ paired run under common random numbers
              ▼
   ┌──────────────────────────┐    MOS / MOE / MOP    ┌──────────────────┐
   │  MissionSimulator         │ ─────────────────────▶│  Aggregator       │
   │  (DES + ABS)              │                       │  → analysis layer │
   └──────────────────────────┘                       └──────────────────┘
```

> **Legacy:** the optimizer–simulator (OPT↔SIM) loop (`optimization/`,
> `replanning_signal.json`) remains in the code but is outside the scope of the paper and
> of the simulator's current plans.

## SysML v2 → Python module mapping

| SysML file(s) read | Python module | What it reads |
|---|---|---|
| `Technical/TechnicalComponentsPackage.sysml` | `cuas_sim.sysml_loader.SysMLLoader` | `part def` values (kinematics, sensors, weapons) → `MissionConfig` |
| `Requirements/*.sysml` (4 files) | `cuas_sim.sysml_requirements` | `requirement def` + `require constraint` (traceability matrix; 14 bound and checked) |
| `Purpose/MissionPhasesPackage.sysml` | `cuas_sim.sysml_state_machine` | `state def` phase order (phase-conformance oracle) |
| `Analysis/CalculationsPackage.sysml` | `cuas_sim.sysml_calculations` | `calc def` measure expressions (calculation guard) |
| unit literals in the files above | `cuas_sim.sysml_loader._to_si` | unit conversion to SI |

Geography is declared in `Purpose/ContextPackage.sysml` but not read: the simulator defaults
reproduce the declared values. The other packages declare intent, capability or function and
are not read.

## Runtime topology

```
┌─────────────────────┐
│   simpy.Environment │
└──┬────────────┬─────┘
   │            │
   │            │     ┌─────────────────────┐
   │            ├────▶│ UASSpawner          │ ── world streams ──▶ both stores
   │            │     └─────────────────────┘
   │            │
   ▼            ▼
┌──────────────┐  ┌──────────────┐
│ store_b      │  │ store_m      │
│ (baseline)   │  │ (modernized) │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ E99MDispatch │  │ E99MDispatch │
│ baseline     │  │ modernized   │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
   {A29}×6           {A29}×6
   baseline          modernized
       │                 │
       └───────┬─────────┘
               ▼
        EngagementController → RunLogger → MetricsAggregator (→ ReplanningSignal, legacy)
```

## Key data classes

* `MissionConfig` — frozen dataclass, hashed for provenance.
* `UASBirth` — frozen, pushed by the spawner to both stores.
* `EngagementOutcome` — single source of truth per engagement.
* `RunLogger` — accumulates engagements, leaks, state events.
* `PairedResult` — bundles the above plus the aggregated DataFrame.
