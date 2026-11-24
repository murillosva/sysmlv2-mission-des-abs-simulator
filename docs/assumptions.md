# Modelling assumptions

This document lists every simplification adopted in the simulator and the
expected impact on each MOS/MOE/MOP. Each assumption names the module that
implements it; paths are relative to `src/cuas_sim/`.

## 1. 2D equirectangular projection in the AoO

The Earth is locally projected to a Cartesian (x, y) plane centered at
`(lat₀, lon₀) = (-5°, -68°)`. Distortion grows quadratically with distance
from the centroid; across the ~2 000 × 1 000 km AoO the worst-case scale
error is ~0.5 % near the meridian extremes.

* **Impact**: < 1 % bias on engagement geometry. MOE-1 (engagement time) is
  most affected; MOS-1/MOS-2 negligibly so since the ZIDA check uses the
  exact Haversine on raw lat/lon.

## 2. Pre-PEA phases are symbolic time-outs

Phases 1 (SBPV → FARP), 2 (FARP → CAP), and 3 (HOLDING) are modeled as
deterministic timeouts using the durations of Table 3.2 of the case study. No fuel burn, no
weather, no tanker rendezvous, no failure stochasticity.

* **Impact**: PEA window start time is fixed at `t_pea_start_s = 9 180 s`
  across A-29s (synchronized engagement). Post-mission analysis won't
  capture asymmetric arrival risk. Set `skip_pre_pea: true` for fast runs.

## 3. E-99M dispatcher is idealized

Every UAS is detected instantaneously at the moment it crosses the border;
handoff to the nearest free A-29 has zero latency. This is an explicit
assumption of the case study (Section 3.3.2.1 of the thesis): the E-99M is an
idealized asset, always present and available, with full coverage and no
faults in the radar track passed to the interceptors.

* **Impact**: Upper-bounds MOE-1 (we measure A-29 reaction; we don't account
  for AEW&C confusion or saturation); real leaks would be larger.

## 4. Default border polygon is an APPROXIMATION

The **synthetic** default polygon (used only when no YAML is loaded, i.e. the
test path) is a chain of 27 equal-length geodesic segments along an
approximation of the CO/PE border, ordered north→south.
Production runs always load the exact 27-sector polyline from
`configs/zida_shield_default.yaml`, whose vertices and arcs come from Natural
Earth 50m, with the sector method adapted from Miranda (2025).

* **Impact (synthetic only)**: The chain is **not to scale** — the straight
  anchor-to-anchor geodesic is ~1 346 km, so its segments are ~50 km, half the
  100-km border arc of the real sectors; sector geometry shifts by tens of km
  vs the YAML polygon. It also carries no `arc`, so ω falls back to the chord.
  Never used in production — the YAML block always overrides it.

### 4a. Cartographic base

The border polyline is from **Natural Earth 50m** — a generalised world-scale
representation, not a legal boundary demarcation. Comparison with the 10m
resolution of the same product shows a mean offset of 0.46 km and a max of
3.44 km between the two. Therefore the origin points are **exact relative to the
line the model adopts**, and subject to ~km-order cartographic error relative to
the juridically demarcated border. Under `spawn_interpolation="arc"` every origin
falls exactly on the adopted polyline (0 % inside national territory), replacing
the chord interpolation, under which roughly half the targets were born inside Brazil.

Those figures are **reproducible**, not narrative: `scripts/border_geometry_audit.py`
recomputes them from the simulator's own `_spawn_coord` / `spawn_in_sector` and
writes `outputs/verification/border_geometry_audit.json`. Measured on the chord
(201 ω samples per sector): **53.4 %** of origins inside Brazil weighting sectors
uniformly, **52.6 %** weighting by the sector-selection pmf, max depth
**20.1 km**, mean offset **5.10 km** from the border line, max arc↔chord
separation **42.26 km** (sector 21).

### 4b. The replay draws the border on the chords

`viz/animate.py` draws the border and offsets the 80-NM ZIDA band from the
27 **chords** (28 vertices). With `spawn_interpolation="arc"`, spawn points land
on the real border, which departs from the drawn chord by up to **42.3 km** in
the most sinuous sectors (21, 3, 4). This visual displacement is a **known
artifact of the replay**, with no effect on the measures.

### 4c. TWO distinct notions of "ZIDA" — scored vs drawn

There are two objects called *ZIDA* in this codebase and they are **not the same
shape**. Conflating them produces plausible-sounding but false causal claims, so
the distinction is recorded here explicitly.

| | **Scored ZIDA** (the metric) | **Drawn ZIDA** (the picture) |
|---|---|---|
| Where | `world/boundary.py::ZIDABoundary` | `viz/animate.py::_zida_band_lonlat` |
| Shape | a **disc of 80 nmi centred on each incursion's own spawn point** | a **fixed band** of constant 80-NM depth, offset inward from the border |
| Per what | one disc **per UAS** | one band for the whole theatre |
| Feeds | MOS-1 success/leak; `t_ev` in the MOE-1 numerator; `pending-evaded` | nothing — presentation only |

`ZIDABoundary.is_inside(spawn_point, current_point)` is `haversine(spawn, pos) <=
radius`. The docstring states the case-study intent: *"the ZIDA is a depth
defined from the entry point, not a fixed polygon — each incursion has its own
ZIDA centered on its spawn."*

**Consequence — the scored ZIDA is invariant under translation of the spawn
point.** Displacing an origin inland does *not* shorten the interceptor's
window, because the disc moves with it. Nor does it shorten `T_PEA`:

* `geometry/haversine.py::t_pea_seconds` is `R - 2*d(FARP, CAP)/v` — its
  arguments are `(farp, cap, endurance_s, cruise_speed_mps)`. **No spawn term.**
* the emergent path (`metrics/aggregator.py::_t_pea_for_aor`) is
  `t_RTB_FUEL_LIMIT - t_pea_start_s`, i.e. the A-29's own fuel clock. It depends
  on spawn geometry only indirectly, through how far the aircraft had to chase.

What a displaced spawn *does* change is the **CAP↔origin geometry**, which drives
pursuit duration and the distance-dependent `p_track`. Measured over the whole
polygon, the chord shortcut moves origins on average **1.27 km closer** to the
serving AoR's CAP — the opposite direction from "less time to react".

> **Therefore:** `spawn_interpolation="arc"` is justified on **geometric and
> distributional fidelity** (origins on the border; uniform ω meaning uniform in
> border length — see `scripts/border_geometry_audit.py`), not on "the target
> would be notified inside national territory and shorten the PEA window". That
> reasoning silently assumes the *drawn* band.

## 5. MOE-2 is zero when no annulus samples are taken

If an engagement has no heading-update sample in the pursuit annulus
(envelope_max, tracking_max], its per-engagement MOE-2 is **0**, not
masked-out. This biases the aggregate MOE-2 downward when arrivals are
short-lived.

* **Impact**: Conservative for modernized config which has a wide 14–20 km
  annulus; bigger impact on baseline (2–13 km annulus) at high λ.

## 6. Spherical Earth, with two radii

Haversine distances use the mean Earth radius,
`EARTH_MEAN_RADIUS_M = 6 371 000 m` (`geometry/haversine.py`), following the
spherical-Earth assumption of the case study. The local projection of §1 uses
the WGS-84 equatorial radius, `EARTH_RADIUS_M = 6 378 137 m`
(`geometry/projection.py`). The two radii differ by about 0.1 %.

* **Impact**: < 0.1 % on T_PEA,k and the ZIDA boundary distance check.

## 7. Constant-velocity UAS kinematics

UAS fly straight constant-heading constant-speed from spawn until killed or
end-of-PEA. No evasive maneuvers, no air-traffic conflict avoidance.

* **Impact**: Optimistic for the A-29 (constant heading is the easiest
  intercept geometry). MOS-1 is therefore an **upper bound**.

## 8. Serial A-29 engagements

Each A-29 engages one UAS at a time; concurrent arrivals back up in the
inbox and are handled FIFO. No formation tactics or multi-target tracking.

* **Impact**: At low λ this is fully realistic; at high λ (≥ 18/h) the
  serial bottleneck dominates and leak rate spikes.

## 9. Discrete pure-pursuit chase, frozen-heading firing

The PURSUING phase is **not** a single closed-form intercept.
The A-29 re-aims its heading at the UAS's *current* position once per tracking
update and flies that heading for `dt` seconds
(`geometry.intercept.pure_pursuit_step`), tracing a lag-pursuit (tail-chase)
curve. At weapon-envelope entry the heading is **frozen** at the bearing to the
UAS, and the rest of the engagement (ENGAGING) flies that straight heading;
firing instants are mapped from firing distances by the closed-form
`time_to_separation`. Speed is constant throughout (no in-flight
speed re-tasking).

* **Impact**: The frozen-heading firing geometry is what makes tail-chase and
  Winchester (out-of-rockets) outcomes **emerge** for unfavourable geometries,
  rather than being assumed away by a guaranteed collision solution. The
  constant-speed assumption can still slightly over-estimate intercept
  feasibility for very long baseline pursuits. See
  [`docs/engagement_model.md`](engagement_model.md) §2.

## 10. Return-to-CAP is hub-and-spoke (no mid-return re-tasking)

After an engagement ends (kill, miss, or floor) the A-29 flies a single atomic
straight leg from the engagement end-point back to its **own CAP** before it can
accept the next UAS; the following pursuit is then geometrically anchored at that
CAP (`agents/a29.py::A29._engage`). The A-29 does **not** divert mid-return to a
closer fresh incursion, and the E-99M dispatcher ranks candidates by
CAP→UAS distance, not by the A-29's live position. Equivalently: each A-29 holds
to its assigned CAP/sector between engagements (hub-and-spoke discipline) rather
than free-roaming opportunistically.

This is a *deliberate, material* simplification, not a negligible one. A
sensitivity study run during development, with an earlier version of the engine
(20 paired seeds; the study is not part of this release), found that:

* return legs are **long** (median about 176 km, about 20 min) and take about
  **20 %** of the fleet's on-station time, so the effect is not small;
* the effect is **near-symmetric across the two configurations** (return-leg
  fraction 0.198 in both), so the **configuration comparison** is robust to the
  simplification, while the *absolute* effectiveness levels are affected by it;
* a prototype of opportunistic re-tasking (interruptible return and pursuit from
  the live position) was **net-negative**: it increased leaks by about a third and
  reduced the MOS-1 gap, because aircraft left their sectors (coverage gaps) and
  chained engagements without refuelling (more bingo aborts).

* **Impact**: The configuration comparison is robust (symmetric); absolute MOS-1
  levels are a lower bound relative to a smarter tasking. The deep-penetration
  pattern (many engagements start with the UAS already inside the weapon envelope,
  through the `PendingPool`) is a **capacity and coverage limit** of the
  six-aircraft fleet relative to the arrival rate, not an artifact of the return
  leg. A tasking model with live-position dispatch and fuel-aware re-tasking is
  future work.
