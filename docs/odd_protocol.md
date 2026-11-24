# ODD Protocol of the Agent-Based Part

Description of the agent-based part of the simulator in the ODD protocol (*Overview, Design
concepts, Details*) of Grimm et al. (2020), the standard for describing agent-based models.
The simulator is a hybrid simulation (Brailsford et al., 2019): a discrete-event skeleton
(SimPy) carries the timing of incursion arrivals and of the aircraft life cycle, and the
decisions of each aircraft are taken by agents. This protocol is a technical description of
the implementation; it is not part of the paper. Each element points to the module that
implements it; the code remains the source of truth. Paths are relative to `src/cuas_sim/`.

---

## 1. Purpose and patterns

**Purpose.** Quantify, in a hypothetical counter-UAS scenario (Continental ZIDA Shield
Operation), the operational gain of the modernized configuration (A-29N: EO/IR sensor, M3AR
datalink, guided APKWS rocket) over the baseline (A-29: naked-eye acquisition, unguided
SBAT-70 rocket), under incursions arriving as a Poisson process. The primary measure is
MOS-1, the fraction of incursions neutralized before crossing the 80-nautical-mile boundary
of the ZIDA, and the gain is read in the paired difference ΔMOS-1.

**Patterns** the model produces, used as qualitative checks of its structure (not as
empirical validation, which was not performed):

- the modernized configuration neutralizes a larger fraction of incursions (EX-1);
- tail chase, Winchester (out of rockets) and bingo fuel emerge from geometry, firing policy
  and serial engagement, and appear as leaks by reason (`offpath_*.json` of each study);
- the serial engagement saturates as the arrival rate grows, and the gain falls with λ
  (EX-2).

---

## 2. Entities, state variables and scales

| Entity | ODD type | Module | Main state variables |
|---|---|---|---|
| A-29 / A-29N | agent | `agents/a29.py`, `states.py` | state of the 15-state machine, `rockets_left`, fuel clock, assigned patrol station (CAP) and FARP, serial inbox (SimPy Store) |
| E-99M dispatcher | idealized agent | `agents/e99m_dispatcher.py` | global nearest-free assignment; `PendingPool` of incursions waiting for a free aircraft |
| UAS (fixed-wing / multi-rotor) | target agent | `agents/uas.py`; archetypes in `config.py` | position (straight, constant-velocity flight), heading, archetype (fixed-wing 130 m/s and 14 m; multi-rotor 29 m/s and 1 m), spawn time |
| FARP, CAP | passive objects | `agents/farp.py`, `config.py` | geographic coordinates |
| Border, ZIDA, areas of responsibility | environment | `world/border_polygon.py`, `world/boundary.py`, `world/aor.py` | 27 border sectors (north to south); the scored ZIDA is an 80-nmi disc around each incursion's spawn point; sector–area–aircraft map |

The phase order of the state machine is checked against the mission phases of the SysML v2
model by `cuas-sim check-conformance` (`sysml_state_machine.py`); the combat states
(`PURSUING`, `ENGAGING`, `RETURNING_TO_CAP`, `RTB_FUEL_LIMIT`) are extensions of the
modelled phases.

**Scales.** Space: local equirectangular projection centred at (−5°, −68°), over an area of
operations of about 2,000 × 1,000 km, with a scale error below 0.5% (`geometry/projection.py`;
[`assumptions.md`](assumptions.md) §1); Haversine distances only for the policing time and
the ZIDA check. Time: missions of hours (A-29 endurance 5.2 h), advanced by events, not by a
fixed step; the pursuit is integrated in discrete steps only during an engagement.

---

## 3. Process overview and scheduling

The schedule is event-driven (SimPy). In each paired run (`simulation/run.py::run_paired`):

1. **Arrivals.** `world/arrivals.py::UASSpawner` draws incursions as a Poisson process of
   rate λ and pushes each one to the stores of both configurations, so both face the same
   incursions.
2. **Spawn window.** Arrivals stop when both configurations have reached their first bingo
   fuel; the window is recorded in `window.json`.
3. **Handoff.** The E-99M dispatcher assigns each incursion to the nearest free A-29 (ties
   broken by aircraft number); when all are busy, the incursion waits in the `PendingPool`.
4. **Aircraft life cycle.** Each of the 6 + 6 A-29 is a SimPy process that walks the state
   machine; engagement is serial (one UAS at a time, arrivals wait in the inbox).
5. **Engagement.** `engagement/controller.py`: pursuit by discrete pure-pursuit steps, then
   shots until a hit, the floor, Winchester, bingo fuel or infeasibility
   ([`engagement_model.md`](engagement_model.md)).
6. **Aggregation.** `metrics/aggregator.py` computes the MOS/MOE/MOP measures
   ([`metrics.md`](metrics.md)).

The runtime topology is drawn in [`architecture.md`](architecture.md).

---

## 4. Design concepts

- **Basic principles.** Hybrid discrete-event and agent-based simulation (Brailsford et al.,
  2019); the MOS/MOE/MOP hierarchy of mission engineering (DoD, 2023); the interception
  doctrine of the case study, in which an engagement ends in neutralization or evasion.
- **Emergence.** Tail chase, Winchester, bingo fuel and saturation emerge from the
  interaction of geometry, firing policy and the serial queue; they are not coded as
  high-level rules.
- **Adaptation, objectives, learning.** None. The aircraft do not adjust their traits, do not
  maximize an objective at run time and do not learn; behaviour follows doctrine (firing
  philosophy, nearest-free dispatch). The zero-reserve bingo rule is a deterministic trigger.
- **Prediction.** Geometric only: `geometry/intercept.py` (`time_to_separation`,
  `pure_pursuit_step`) predicts the future separation to time the shots, and
  `geometry/projection.py::t_to_farp` predicts the time back to the FARP for bingo fuel.
- **Sensing.** `policies/tracking.py`: naked-eye acquisition in the baseline, with the angular
  probability `min(1, L_U / (d · θ_acuity))` and θ_acuity = 1 arcmin; EO/IR tracking in the
  14–20 km annulus in the modernized configuration. E-99M detection is idealized
  (instantaneous), an upper bound ([`assumptions.md`](assumptions.md) §3).
- **Interaction.** A-29 × UAS through the engagement controller; E-99M × A-29 through the
  handoff; contention through the serial inbox and the `PendingPool`.
- **Stochasticity.** Every random number comes from the `RNGRegistry` (`rng.py`). Four world
  streams (interarrival time, entry sector, origin and heading, archetype) are shared by the
  two configurations; four decision streams (update cadence, first-shot philosophy and
  distance, shot interval, shot outcome) are drawn per configuration. The same seed gives
  bit-identical output.
- **Collectives.** The fleet of six aircraft with global dispatch forms a weak collective,
  without a command hierarchy. Coordination is hub-and-spoke: after an engagement, the A-29
  flies back to its own CAP before accepting a new target ([`assumptions.md`](assumptions.md)
  §10).
- **Observation.** Each run writes `engagements.parquet`, `spawns.parquet`, `metrics.parquet`,
  `events_*.jsonl.gz` and `window.json`; `viz/` replays a run afterwards (`cuas-sim animate`).

---

## 5. Initialization

The initial state is the frozen `MissionConfig` (`config.py`), resolved in increasing
precedence: Python defaults < scenario file (YAML) < SysML v2 model, which prevails in the
technical fields it supplies (`MissionConfig.from_yaml_with_sysml`). It includes the FARPs
(SBRB, SBCZ, SDCG, SWKU), the six CAPs, the flight, sensor and weapon parameters of each
configuration, λ, the master seed and the spawn window. The configuration is hashed for
provenance.

---

## 6. Input data

The external source that parameterizes the model is the SysML v2 mission model (27 packages,
[`../sysml/README.md`](../sysml/README.md)). `sysml_parser.py` builds a typed syntax tree of
the textual notation; `sysml_loader.py` reads the part definitions with values into the
configuration, so each value is traceable to the definition that declares it; and
`sysml_requirements.py` checks the resolved configuration against the `require constraint`
of the requirements (`cuas-sim check-requirements`). The seven files read, by four modules,
are listed in [`architecture.md`](architecture.md).

---

## 7. Submodels

| Submodel | Module | Content |
|---|---|---|
| Tracking (sensing) | `policies/tracking.py` | baseline angular law with 1-arcmin acuity; modernized annulus |
| Weapon | `policies/weapon.py` | `P_hit` linear in distance; first-shot distance by philosophy and Triangular draw |
| Cumulative neutralization | `engagement/controller.py` | `P_neut = 1 − ∏(1 − P_hit,k)` |
| Interception geometry | `geometry/intercept.py` | `time_to_separation`, `pure_pursuit_step` |
| Bingo fuel and geometric return | `geometry/projection.py`, `agents/a29.py` | `t_to_farp` |
| Measures | `metrics/aggregator.py` | MOS/MOE/MOP ([`metrics.md`](metrics.md)) |

The engagement submodels are detailed in [`engagement_model.md`](engagement_model.md).
