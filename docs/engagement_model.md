# Engagement Model (ENGAGING phase) per Configuration

How the engagement of the A-29 is modelled in the simulator, separating what the two
configurations share from what distinguishes the **baseline** (A-29 / unguided SBAT-70
rocket) from the **modernized** configuration (A-29N / guided APKWS rocket). This document
is the reference for the firing mechanics; the general modelling assumptions are in
[`assumptions.md`](assumptions.md).

## Sources

| Layer | File | Role |
|-------|------|------|
| Life-cycle orchestration | `agents/a29.py` (`_engage`) | State transitions, SimPy clock, rocket consumption |
| Engagement engine | `engagement/controller.py` (`EngagementController.run` + `_engage_after_envelope`) | Pursuit, firing decision and shot sequence, outcome |
| Lethality | `policies/weapon.py` | `P_hit` per shot, firing philosophy, guided/unguided regime |
| Tracking | `policies/tracking.py` | Update cadence that sets when ENGAGING starts; MOE-2 sampling |
| Geometry | `geometry/intercept.py` | `time_to_separation`, `pure_pursuit_step` |

Paths are relative to `src/cuas_sim/`; line numbers refer to this release.

---

## 1. Where ENGAGING fits in the life cycle

The combat sequence of an A-29 is:

```
AT_CAP_PEA → PURSUING → ENGAGING → (RETURNING_TO_CAP | RTB_FUEL_LIMIT) → AT_CAP_PEA
```

The split of responsibilities matters:

- **PURSUING** — the A-29 closes the distance. It is modelled as discrete *pure pursuit*:
  at each tracking update the A-29 turns its heading to the **current** position of the UAS
  and flies for `dt` seconds (`controller.py:198–284`, `pure_pursuit_step`). MOE-2 is sampled
  here.
- **ENGAGING** (weapon engagement) — starts when the separation falls inside the **maximum
  envelope of the weapon**. This is the phase this document details: how the A-29 decides to
  fire, how the shots behave and when they stop. All the logic is in
  `_engage_after_envelope` (`controller.py:339–522`).

The controller solves **all the kinematics analytically** and returns a single
`EngagementOutcome`. `agents/a29.py` only advances the SimPy clock with
`yield env.timeout(...)` to the time marks of that outcome, which keeps the engagement
**serial and first-in, first-out** (§8).

---

## 2. Trigger for ENGAGING and heading freeze

### 2.1 Trigger — entry into the weapon envelope

During PURSUING, the separation `sep` is checked at each step. ENGAGING is triggered when:

```python
if sep <= env_max:        # controller.py:247
    t_envelope_entry = t
    break
```

`env_max` is **the main parameter that distinguishes the configurations in this
transition**:

| Configuration | `envelope_max_m` (ENGAGING trigger) |
|---------------|-------------------------------------|
| **baseline** | **2,000 m** |
| **modernized** | **14,000 m** |

That is, the A-29N enters the engagement **7 times farther** from the target. This is the
first-order effect of the weapon modernization: more room and time to re-engage before
bingo fuel.

### 2.2 Heading freeze (central kinematic assumption)

At the exact instant of envelope entry, the heading of the A-29 is **frozen** on the bearing
to the UAS at that moment (`controller.py:292–294`):

```python
heading_at_envelope = math.atan2(uas_xy[1] - a29_xy[1], uas_xy[0] - a29_xy[0])
```

From then on, during the **whole** ENGAGING phase, the A-29 flies a **straight line** on
that fixed heading; unlike PURSUING, there is no further heading change. The UAS keeps a
constant velocity. Every later separation is computed from this fixed-heading kinematics by
`time_to_separation`.

**Consequence:** a *tail chase* emerges naturally. Against targets that do not fly straight
away from it, the A-29 closes; when the target is fast or the geometry is unfavourable, the
A-29 may run out of rockets without a kill (Winchester). This behaviour emerges; it is not
coded explicitly.

The geometric instant of closest approach on the fixed heading is `t_intercept`
(`controller.py:295–303`), computed by `time_to_separation(..., target_sep_m=0)`. No shot
can be scheduled **after** `t_intercept`.

---

## 3. How the aircraft decides to fire

> **There is no "is this shot good enough?" gate.** Once inside the envelope, the A-29
> **always** fires. What is drawn is the *firing philosophy*, which sets the **distance** of
> the first shot. Long-range shots with low (or zero) `P_hit` **are taken** and consume a
> rocket.

The draw happens in `draw_firing_distance` (`weapon.py:141–153`, called at
`controller.py:378`) and has **two steps**:

1. **Philosophy** — uniform discrete draw with probability 1/3 each (`rng.integers(0, 3)`):
   `0 = short`, `1 = medium`, `2 = long` (the `DetermineFirePhilosophy` of the case study).
2. **Distance** — **continuous** `Triangular(low, mode, high)` inside the sub-band of that
   philosophy. The envelope is divided into three (`range/3`); the `mode` is anchored at the
   model's `firingDistance{phil}Philosophy` point (= minimum / centre / maximum).

Sub-bands (`firing_subband`, `weapon.py:94–113`):

| Philosophy | Baseline `[500, 2000]` m | Modernized `[2000, 14000]` m |
|------------|--------------------------|------------------------------|
| short (idx 0) | `]500, 1000]`, mode 500 | `]2000, 6000]`, mode 2000 |
| medium (idx 1) | `]1000, 1500]`, mode 1250 | `]6000, 10000]`, mode 8000* |
| long (idx 2) | `]1500, 2000]`, mode 2000 | `]10000, 14000]`, mode 14000 |

\* The `mode` of medium is the centre of the envelope, **clamped** inside the sub-band
(`weapon.py:110–113`); this matters only for non-canonical envelopes.

### 3.1 When the first shot happens in time

The drawn distance `d1` is converted into the **instant** at which the real separation
reaches `d1`, by `_t_at_sep` (`controller.py:385–395`), using `time_to_separation` on the
frozen heading:

- If `d1 ≥ separation at envelope entry` (the typical case of the *long* philosophy), the
  shot is fired **immediately** at envelope entry (`t_envelope_entry`).
- Otherwise, the A-29 closes on the fixed heading until the separation reaches `d1` and fires
  there, bounded by `t_intercept`.

**Practical difference between configurations:** since the modernized envelope reaches
14 km, a modernized *long* philosophy fires at about 14 km (very low `P_hit`, see §4), while
the baseline can fire only from 2 km. The A-29N has much more room to start early and
re-engage several times as it closes.

---

## 4. Shot behaviour

The hit probability of **each** shot depends on the **real separation at that shot** and on
the archetype of the target, through `p_hit_at` (`weapon.py:164–173`):

$$P_{hit}(d) = M \cdot 0{.}80 \cdot \left(1 - \frac{d - d_{min}}{d_{max} - d_{min}}\right)$$

- `0.80` = maximum `P_hit` (reached at the minimum range of the envelope), field
  `p_hit_at_min_range` (`config.py:287`).
- `M` = **dimensional penalty**: `0.5` for multi-rotor, `1.0` for fixed-wing
  (`config.py:288–289`, Table 3.3 of the case study).
- **Linear** decay with distance: a shot at the `d_max` of the envelope has `P_hit ≈ 0`.

### 4.1 The crucial difference: the dual regime of the A-29N

This is the subtlest and most important point of the lethality model. The **baseline** has
**a single curve**. The **modernized** configuration has **two regimes**, with a **sharp**
switch at 2 km (`weapon.py:171–172`, `weapon.py:219–229`):

| Regime | Range | Curve |
|--------|-------|-------|
| **Baseline** | `[500, 2000]` m | unguided SBAT-70, single |
| **A-29N — guided** | `[2000, 14000]` m | guided APKWS |
| **A-29N — below 2 km** | `[500, 2000]` m | **falls back to the baseline curve** (`unguided_below`) |

That is: when it closes **below 2 km**, the A-29N fires like a baseline A-29 (same `P_hit`
curve, same `M`, same `p_hit_max`). The switch at 2 km is **sharp, with no smoothing**, a
modelling decision faithful to the case study.

**Firing floor** (minimum separation below which no shot is fired, `weapon.py:175–181`):

| Configuration | `floor_m` |
|---------------|-----------|
| baseline | 500 m (= `envelope_min`) |
| modernized | 500 m (from `unguided_below`, **not** from the 2 km of the guided regime) |

### 4.2 Realization of the hit (Bernoulli)

`P_hit` is only the probability. The actual hit of each shot `k` is a Bernoulli draw on a
dedicated stream (`controller.py:453–459`):

```python
if float(rng_p_hit.random()) < p_k:
    is_neut = True; t_kill = t_k; kill_sep = sep_k; n_shots = k; break
```

The loop **stops at the first hit**; this sets `n_shots` (rockets consumed until the kill)
and `t_kill`.

---

## 5. Re-engagement: when and how firing stops

The number and the pace of the shots depend on the philosophy drawn in §3.

### 5.1 short (idx 0) → single shot

`controller.py:383, 420–421`: a single shot. If it misses, there is **no** re-engagement
and the UAS escapes (leak). The *short* philosophy is identified by `philosophy_idx == 0`,
**not** by the distance (the short sub-band is `]min, min+R/3]`, strictly above
`envelope_min`).

### 5.2 medium / long → multiple re-engagement

`controller.py:419–441`: after the first shot, the A-29 fires again at intervals
`Δt ~ Triangular(8, 12, 18) s` (stream `reengagement_interval`, `config.py:294`),
recomputing the real separation at each new firing instant. Since the A-29 keeps closing on
the fixed heading, the separation **decreases** at each shot and `P_hit` **increases**.

The loop continues **until** one of the following stop conditions:

| # | Stop condition | Location | Result |
|---|----------------|----------|--------|
| 1 | **Hit** (Bernoulli success) | `controller.py:454` | UAS neutralized; stops at the first hit |
| 2 | **Past `t_intercept`** (closest approach) | `controller.py:428` | no further closing is possible; stops |
| 3 | **Separation ≤ floor** (500 m) | `controller.py:439–440` | too close; stops |
| 4 | **Rocket budget exhausted** | `controller.py:422` (`len(shots) < budget`) | Winchester |
| 5 | **Bingo fuel** | `controller.py:430–436` | aborts; the Bernoulli of that shot is **not** resolved (§6) |

The rocket budget per sortie is **per configuration** (`config.py:300–301`,
`rocket_budget_for` at `config.py:435`): both default to 49 (SBAT-70 and APKWS, read from
the SysML v2 model).

> **Important:** the re-engagement logic, the philosophies and the stop conditions are
> **identical** in the two configurations. What changes are the **parameters** (envelope,
> `P_hit` curve, tracking cadence). The modernization does not change *how* the decision to
> fire is taken; it changes *where* and *with what lethality*.

---

## 6. Bingo fuel (truncation by fuel)

Evaluated **before each shot** by `_bingo_at` (`controller.py:397–403`) and also **at each
step** during PURSUING (`controller.py:212–245`). The predicate has **zero reserve**:

$$(t_{exhaust} - t_k) \le t\_to\_farp(\text{pos}, \text{farp})$$

If the remaining fuel **no longer covers the flight back to the forward arming and refueling
point (FARP)**, the engagement is truncated and marked `bingo_aborted`, and the A-29 returns
to base (`A29State.RTB_FUEL_LIMIT`). The released UAS goes to the `PendingPool` if a teammate
is free; otherwise it becomes a `bingo-fuel` leak (`a29.py:233–251`).

`t_exhaust = t_farp_departure + a29_endurance_s` (`a29.py:301–304`;
`a29_endurance_s = 5.2 h`). `farp_xy=None` or `fuel_exhaust_t=inf` disable the trigger (for
callers that do not model fuel).

---

## 7. Engagement outputs and coupling with the measures

`EngagementOutcome` (`controller.py:42–93`) is the single source per engagement. Fine
distinctions worth recording:

- **MOS-1 (realized neutralization, inside the 80-nmi ZIDA)** uses the **realized
  Bernoulli** (stop at the first hit). `is_neutralized_before_zida` checks whether the kill
  point is inside the 80-nmi disc around the spawn point (`controller.py:478–485`).
- **MOE-3** uses `cumulative_neutralization(p_seq)` = $1 - \prod_k (1 - p_k)$ over **all
  planned shots** (`controller.py:445–446`, `535–541`). It is the **expected** value, not the
  realization.
- **MOE-1** — numerator = `t_end − t_pursuit_start_s` (`controller.py:487–489`); the final
  aggregation is `clamp(1 − mean(t_ef/T_PEA), 0, 1)` with an evasion penalty. The evasion
  instant `t_ev` (projected crossing of the 80 nmi) is stored in `t_ev_s` for engagements
  without neutralization (`controller.py:159–169`, `521–522`).

Diagnostic fields: `n_shots`, `rockets_remaining`, `end_xy` (kill point, last shot, or bingo
position), `bingo_aborted`/`bingo_xy`/`bingo_t_s`, `feasible`.

---

## 8. Time orchestration (serial engagement)

`agents/a29.py` (`_engage`, `a29.py:191–289`) turns the analytical outcome into SimPy time:

1. `_set_state(PURSUING)`; calls `controller.run(...)`.
2. Consumes rockets: `rockets_left -= outcome.n_shots` (`a29.py:228`).
3. Advances the clock: `timeout(t_envelope_entry − t_notify)` (PURSUING), then
   `_set_state(ENGAGING)` and `timeout(t_fire − now)` (`a29.py:257–266`).
4. Applies the outcome to the UAS (`mark_killed` / `mark_leak`) and logs the engagement.
5. Returns to the CAP from the **real end point** (`end_xy`), not by a symmetric outbound
   duration (`a29.py:280–289`).

Since the clock advances to the end of the engagement, concurrent arrivals wait in the inbox
(SimPy Store) and are served **first-in, first-out**: an A-29 engages **one** UAS at a time.
This produces saturation at high λ: when incursions outnumber the free aircraft, they queue
and leaks grow.

---

## 9. Baseline × modernized comparison

| Aspect | **Baseline (A-29 / SBAT-70)** | **Modernized (A-29N / APKWS)** |
|--------|-------------------------------|--------------------------------|
| Enters ENGAGING at | ≤ 2 km | ≤ 14 km |
| Firing envelope | `[500, 2000]` m, single regime | `[2000, 14000]` m guided **+** `[500, 2000]` m unguided below 2 km (sharp switch) |
| `P_hit` curve | linear in distance, maximum `0.80·M` at minimum range | linear, guided; **becomes the baseline curve** below 2 km |
| Firing floor | 500 m | 500 m (through the unguided regime) |
| Philosophy and distance decision | 1/3 discrete + Triangular in the sub-band | identical, with much larger sub-bands |
| Re-engagement | short = 1 shot; medium/long fire again at `Triangular(8, 12, 18)` s | identical |
| Stop conditions | hit / `t_intercept` / floor / Winchester / bingo | identical |
| Rocket budget | 49 (SBAT-70) | 49 (APKWS) |
| Tracking that sets the start | 12.5 s reaction + updates U(5, 60) s (naked eye) | updates U(1, 10) s (EO/IR sensor and datalink), no reaction floor |
| MOE-2 sampling annulus | `(2 km, 13 km]` | `(14 km, 20 km]` |

The modernization **does not change the firing decision logic** (same philosophy, same
re-engagement, same stop conditions). It changes the **parameters**: a much larger envelope
(more attempts before bingo), a guided curve above 2 km, and faster tracking, which makes the
A-29N start engaging earlier and accumulate more attempts per target.

---

## 10. Random number streams used (reproducibility)

All through the `RNGRegistry`, separated by configuration (`a29.py:218–221`):

| Stream | Use in the engagement |
|--------|-----------------------|
| `tracking_updates` | update cadence (sets when ENGAGING starts) + MOE-2 sampling in PURSUING |
| `firing_phil` | philosophy (idx) **and** Triangular distance of the first shot |
| `reengagement_interval` | `Triangular(8, 12, 18)` interval between shots k ≥ 2 |
| `p_hit_outcome` | Bernoulli hit of each shot |

The same seed gives bit-identical output.

---

## 11. Parameters and sources (defaults)

| Parameter | Default | Source (`config.py`) |
|-----------|---------|----------------------|
| `a29_cruise_speed_mps` | 144.0 | `:245` |
| `a29_endurance_s` | 18,720 (5.2 h) | `:246` |
| `baseline_envelope_m` | `(500, 2000)` | `:285` |
| `modernized_envelope_m` | `(2000, 14000)` | `:286` |
| `p_hit_at_min_range` | 0.80 | `:287` |
| `dimensional_penalty_mr` / `_fw` | 0.5 / 1.0 | `:288–289` |
| `reengagement_triangular_s` | `(8, 12, 18)` | `:294` |
| `rocket_budget_per_sortie` | 49 | `:295` |
| `n_infeasible_updates` | 3 | `:304` |
| `baseline_update_interval_s` | `(5, 60)` | `:266` |
| `modernized_update_interval_s` | `(1, 10)` | `:275` |

Resolution precedence, in increasing order: **Python defaults < YAML < SysML v2 model** —
`SysMLOverrides.apply_to` is applied on top of the YAML in the fields that the model supplies
(see `MissionConfig.from_yaml_with_sysml` in `src/cuas_sim/config.py`). The weapon parameters
(envelope, `P_hit`, penalties, capacity) are read from the APKWS and SBAT-70 `part def` of the
model.

---

## 12. Assumptions and limitations

1. **Frozen heading in ENGAGING** — after envelope entry the A-29 does not turn; it flies
   straight until the kill or a stop. Optimistic against targets flying straight away;
   produces emergent tail chase and Winchester.
2. **Always fires inside the envelope** — there is no decision "not to fire because `P_hit`
   is low". *Long* shots with `P_hit ≈ 0` are taken and consume a rocket.
3. **short = single shot** — no second chance; a miss is a leak.
4. **Sharp regime switch at 2 km** (A-29N) — no smoothing between guided APKWS and
   unguided SBAT-70 (faithful to the case study).
5. **Multiple shots as a visual approximation** — the simulation stores the `t_fire` of the
   first shot and `t_kill`; the visualization (`viz/`) reconstructs the intermediate shots
   (not bit-exact; see `src/cuas_sim/viz/missile_flyout.py` and `viz/hud.py`).
6. **Zero-reserve bingo** — the A-29 engages until the last instant that still allows the
   return to the FARP; there is no fuel safety margin.
