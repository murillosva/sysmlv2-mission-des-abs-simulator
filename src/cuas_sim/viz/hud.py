"""Per-A-29 heads-up display for the matplotlib animator.

Renders a compact text box beside each active A-29 marker showing the four
emergent quantities that the simulator already persists but the bare colored
dot cannot convey: the lifecycle **state**, **rockets**
remaining, **fuel** margin, and the **target** UAS. Two of the model's most
important behaviours — Winchester (rockets exhausted before kill) and
bingo-fuel abort — are present in the run-dir but invisible without this layer.

Design — thin render, pure logic
--------------------------------
All four quantities are reconstructed from data already in the run directory
(``engagements.parquet`` + ``events_<config>.jsonl.gz``); none is persisted
directly. The reconstruction lives in **pure** helpers
(:func:`fuel_fraction`, :func:`rockets_remaining_at`,
:func:`active_engagement_at`, :func:`a29_state_at`, :func:`fuel_anchor_s`) that
take plain tuples and return plain values — no RNG, no I/O, no global
side-effects, in the same spirit as
:mod:`cuas_sim.viz.replay` and :mod:`cuas_sim.viz.missile_flyout`.
:func:`render_a29_hud` is the only function that touches matplotlib, and it
only formats the values its caller already computed.

Fuel reconstruction (approximate — documented)
----------------------------------------------
The agent's fuel clock (``A29._t_farp_departure`` / ``_fuel_exhaust_t``)
is internal and never logged. The animator reconstructs it from
the state-transition stream: the most recent transition that *fills the tank* —
``TRANSIT_TO_FARP`` (SBPV takeoff with a full tank), ``REPOSITIONING_TO_CAP``
(FARP→CAP departure, where ``A29.run`` sets ``_t_farp_departure``) or
``AT_FARP_POST_PEA`` (the post-PEA refuel that resets it) — at or before ``t_s``
is the anchor (:func:`fuel_anchor_s`). The SBPV→FARP transit therefore burns from
the takeoff anchor, then tops back up at the FARP (``AT_FARP_PRE_PEA`` reads a
full tank); the CAP departure re-anchors for the patrol. With ``skip_pre_pea``
there is no transit/reposition event, so the caller passes the PEA-start time as
the fallback (matching the agent, which anchors the clock at PEA start in that
path). The fraction is then ``1 − (t_s − anchor) / a29_endurance_s`` clamped to
``[0, 1]`` — equivalent to ``(fuel_exhaust_t − t_s) / a29_endurance_s``.
This is an **approximate reconstruction**: it tracks the same linear burn
the agent uses (the transit leg reuses ``a29_endurance_s`` as the burn-rate
reference) but cannot see sub-step detail the sim never wrote.

Rockets reconstruction
----------------------
Rockets are the sortie's rocket budget minus the rockets fired at or before ``t_s``,
floored at 0 (Winchester). Each rocket leaves the count at its own fire instant
(:func:`shot_fire_times` spreads the ``n_shots`` of an engagement over its firing
window). A FARP rearm (``AT_FARP_POST_PEA``) reloads the launcher, so only
shots fired after the most recent refuel count (:func:`rockets_remaining_at`).

Design constraints:

* Reproducibility — the logic helpers are pure; the HUD is
  deterministic per run-dir.
* Formula integrity — the HUD only **reads** persisted
  values; it changes no MOS/MOE/MOP formula and feeds none.
* Scope — the last of the three replay modules of ``viz/`` (with
  :mod:`cuas_sim.viz.replay` and :mod:`cuas_sim.viz.missile_flyout`).

SysML traceability:

* ``a29_endurance_s`` ↦ ``TechnicalComponentsPackage`` A-29 endurance.
* ``rocket_budget_per_sortie`` ↦ ``TechnicalComponentsPackage`` launcher
  capacity.
* State names ↦ ``MissionExecutionPackage`` snapshots (see
  :mod:`cuas_sim.states`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cuas_sim.states import A29State

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.text import Annotation


#: States in which the A-29 is on the ground or freshly refuelled, so the HUD
#: shows a full tank. The aircraft burns fuel while airborne — including the
#: SBPV→FARP transit (``TRANSIT_TO_FARP``), after which it tops up at the FARP
#: (``AT_FARP_PRE_PEA``) before the CAP departure.
_FUEL_FULL_STATES: frozenset[str] = frozenset({
    A29State.AT_BASE.value,
    A29State.AT_FARP_PRE_PEA.value,
    A29State.AT_FARP_POST_PEA.value,
    A29State.MISSION_COMPLETE.value,
})

#: States for which the HUD is suppressed — a parked aircraft adds no
#: information and would only clutter runs with many simultaneous engagements.
_HUD_HIDDEN_STATES: frozenset[str] = frozenset({
    A29State.AT_BASE.value,
    A29State.MISSION_COMPLETE.value,
})

#: Transitions that fill the tank and therefore re-anchor the fuel clock: the
#: SBPV takeoff (``TRANSIT_TO_FARP``, full tank leaving base), the CAP departure
#: (``REPOSITIONING_TO_CAP``, where ``A29.run`` sets ``_t_farp_departure`` after
#: the FARP top-up), and the post-PEA FARP refuel (``AT_FARP_POST_PEA``, where
#: ``A29._recover`` resets it). The latest anchor at or before ``t_s`` wins, so
#: the FARP top-up (no event, but ``AT_FARP_PRE_PEA`` reads full) and the CAP
#: departure correctly supersede the takeoff anchor.
_FUEL_ANCHOR_STATES: frozenset[str] = frozenset({
    A29State.TRANSIT_TO_FARP.value,
    A29State.REPOSITIONING_TO_CAP.value,
    A29State.AT_FARP_POST_PEA.value,
})

#: Concise, title-case label for each lifecycle state, shown on the HUD's first
#: line in place of the verbose SCREAMING_SNAKE enum name. The animator's A-29
#: STATE legend (``viz/animate.py``) reads its labels from here too, so the
#: legend and the HUD always name a state alike; unknown states fall through to
#: the raw value via :func:`state_short_label`.
_STATE_SHORT: dict[str, str] = {
    A29State.AT_BASE.value:              "At Base",
    A29State.TRANSIT_TO_FARP.value:      "To FARP",
    A29State.AT_FARP_PRE_PEA.value:      "At FARP",
    A29State.REPOSITIONING_TO_CAP.value: "To CAP",
    A29State.HOLDING_ORBIT.value:        "Holding",
    A29State.AT_CAP_PEA.value:           "On Station",
    A29State.PURSUING.value:             "Pursuing",
    A29State.ENGAGING.value:             "Engaging",
    A29State.RETURNING_TO_CAP.value:     "Return to CAP",
    A29State.RTB_BINGO.value:            "Bingo",
    A29State.DEPARTING_CAP.value:        "Depart CAP",
    A29State.RTB_FUEL_LIMIT.value:       "Bingo / RTB",
    A29State.AT_FARP_POST_PEA.value:     "At FARP",
    A29State.RECOVERING_TO_BASE.value:   "Recovering",
    A29State.MISSION_COMPLETE.value:     "Mission Complete",
}


# ─── Pure logic helpers ────────────────────────────────────────────────────────


def state_short_label(state: str) -> str:
    """Concise title-case label for a HUD/legend state (pure).

    Maps a lifecycle state value (e.g. ``"TRANSIT_TO_FARP"``) to a short,
    title-case label (``"Transit"``); falls back to the raw value for any
    state not in :data:`_STATE_SHORT`.
    """
    return _STATE_SHORT.get(state, state)


def a29_state_at(
    events: Sequence[tuple[float, str]],
    t_s: float,
    default: str = A29State.AT_BASE.value,
) -> str:
    """A-29 lifecycle state at ``t_s`` from its transition log.

    Returns the ``new``-state of the latest transition whose timestamp is
    ``≤ t_s``; ``default`` (AT_BASE) before the first transition. ``events`` is
    the ``(t_ev, state_value)`` list sorted ascending — identical derivation to
    the animator's ``_state_colors`` closure (single source of truth).
    """
    state = default
    for t_ev, st in events:
        if t_ev <= t_s:
            state = st
        else:
            break
    return state


def is_in_flight(state: str) -> bool:
    """True when ``state`` burns tracked fuel (airborne).

    Grounded / freshly-refuelled states (:data:`_FUEL_FULL_STATES`) read a full
    tank; everything airborne — from the ``TRANSIT_TO_FARP`` takeoff onward —
    burns fuel.
    """
    return state not in _FUEL_FULL_STATES


def should_render_hud(state: str) -> bool:
    """Whether to draw the HUD for an A-29 in ``state``.

    Suppressed for AT_BASE / MISSION_COMPLETE; shown for every active state.
    """
    return state not in _HUD_HIDDEN_STATES


def fuel_anchor_s(
    events: Sequence[tuple[float, str]],
    t_s: float,
    fallback_s: float,
) -> float:
    """Mission time the tank was last filled, at or before ``t_s``.

    Scans the transition log for the most recent fuel-refilling transition
    (:data:`_FUEL_ANCHOR_STATES`). Returns ``fallback_s`` when none is found —
    e.g. ``skip_pre_pea`` runs log no ``REPOSITIONING_TO_CAP``; the agent then
    anchors the clock at the PEA start, which the caller passes as the fallback.
    """
    anchor = fallback_s
    for t_ev, st in events:
        if t_ev > t_s:
            break
        if st in _FUEL_ANCHOR_STATES:
            anchor = t_ev
    return anchor


def fuel_fraction(
    t_s: float,
    t_farp_departure_s: float,
    a29_endurance_s: float,
    in_flight: bool,
) -> float:
    """Remaining-fuel fraction in ``[0, 1]`` at ``t_s``.

    On the ground / refuelled (``in_flight=False``) the tank is full (1.0). In
    flight the fraction is ``1 − (t_s − t_farp_departure_s) / a29_endurance_s``
    — i.e. ``(fuel_exhaust_t − t_s) / a29_endurance_s`` with
    ``fuel_exhaust_t = t_farp_departure_s + a29_endurance_s`` —
    clamped to ``[0, 1]``. Pure approximation of the agent's internal clock;
    see the module docstring.
    """
    if not in_flight:
        return 1.0
    if a29_endurance_s <= 0.0:
        return 0.0
    frac = 1.0 - (t_s - t_farp_departure_s) / a29_endurance_s
    return max(0.0, min(1.0, frac))


def shot_fire_times(
    t_first_fire_s: float,
    t_last_fire_s: float,
    n_shots: int,
) -> list[float]:
    """Per-shot fire instants of one engagement, for the rocket tally.

    The simulator persists only the first shot's ``t_fire_s`` (and ``t_kill_s``)
    — never each shot's instant — so the ``n_shots`` rounds are reconstructed by
    spreading them uniformly across the firing window ``[t_first_fire_s,
    t_last_fire_s]`` (the same window the missile-flyout layer draws across, so
    the HUD ticks down in step with the visible missiles). One shot fires at
    ``t_first_fire_s``; a zero-span window (instant kill) bursts them all there.
    Visual approximation, not bit-exact.
    """
    n = max(0, n_shots)
    if n == 0:
        return []
    if n == 1 or t_last_fire_s <= t_first_fire_s:
        return [t_first_fire_s] * n
    step = (t_last_fire_s - t_first_fire_s) / (n - 1)
    return [t_first_fire_s + i * step for i in range(n)]


def rockets_remaining_at(
    fire_times_s: Sequence[float],
    t_s: float,
    rocket_budget: int,
    refuel_times_s: Sequence[float] = (),
) -> int:
    """Rockets left at ``t_s``.

    ``fire_times_s`` is the flat list of individual shot-fire instants across
    all of the A-29's engagements (see :func:`shot_fire_times`). The launcher
    starts at ``rocket_budget`` (loaded at PEA start) and each
    rocket leaves the count **at its own fire instant** — not at the pursuit
    start — so the HUD decrements as the A-29 actually shoots, in step with the
    missile-flyout layer. The tally is floored at 0 (Winchester). A FARP rearm
    (``AT_FARP_POST_PEA``, times in ``refuel_times_s``) reloads the launcher, so
    only rounds fired after the most recent rearm at or before ``t_s`` count.
    """
    last_refuel = max((r for r in refuel_times_s if r <= t_s), default=None)
    floor_t = last_refuel if last_refuel is not None else float("-inf")
    used = sum(1 for ft in fire_times_s if floor_t < ft <= t_s)
    return max(0, rocket_budget - used)


def active_engagement_at(
    windows: Sequence[tuple[float, float, str]],
    t_s: float,
) -> str | None:
    """Target UAS id of the engagement whose window contains ``t_s``.

    ``windows`` is ``(t_start_s, t_end_s, uas_id)`` per engagement (inclusive
    bounds). Returns the id of the first window containing ``t_s``, else
    ``None`` (the A-29 is between engagements). ``t_end_s`` is ``t_kill_s`` for
    a neutralisation, else the last-shot instant — set consistently by the
    caller with the fire/missile-flyout layers.
    """
    for t_start, t_end, uid in windows:
        if t_start <= t_s <= t_end:
            return uid
    return None


# ─── Rendering ─────────────────────────────────────────────────────────────────


def render_a29_hud(
    ax: Axes,
    a29_id: str,
    state: str,
    rockets_remaining: int,
    fuel_pct: float,
    target_uas_id: str | None,
    marker_lonlat: tuple[float, float],
    accent_color: str = "#66FFFF",
) -> Annotation:
    """Draw a compact HUD box adjacent to an A-29 marker.

    Renders three small lines beside ``marker_lonlat`` (lon, lat in the
    animator's native frame): ``"<id>  <short-state>"`` (via
    :func:`state_short_label`), ``"R:<rockets>  F:<fuel>%"``, and a target line
    (``"→ U<id>"`` while engaging, a bare ``"-"`` otherwise: "no target", not
    "idle", since the aircraft is usually flying). Uses ``ax.annotate``
    with a fixed pixel offset so the box does not scale with zoom,
    ``fontsize≈6.5`` and a low-opacity backing box (to keep the map legible
    with many simultaneous A-29s). ``accent_color`` tints the box border so it
    matches the aircraft's current phase colour in the bottom-bar legend
    (defaults to cyan). Returns the :class:`matplotlib.text.Annotation` so the
    caller can remove it next frame.
    """
    fuel_line = f"R:{rockets_remaining}  F:{round(fuel_pct * 100.0)}%"
    target_line = f"→ U{target_uas_id}" if target_uas_id is not None else "-"
    text = f"{a29_id}  {state_short_label(state)}\n{fuel_line}\n{target_line}"

    return ax.annotate(
        text,
        xy=marker_lonlat,
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.5,
        family="monospace",
        color="#DCE6F0",
        alpha=0.8,
        zorder=15,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#0d1b2a",
            "edgecolor": accent_color,
            "linewidth": 0.6,
            "alpha": 0.55,
        },
    )
