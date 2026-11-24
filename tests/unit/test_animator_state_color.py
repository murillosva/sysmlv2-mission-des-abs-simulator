"""Unit oracle for the A-29 state→colour map of the matplotlib animator.

Verifies: Spec 019 §3.1 FR-10

FR-10 [MUST]: ``_STATE_COLORS`` must contain an entry for
``A29State.RTB_FUEL_LIMIT`` (Spec 018 FR-7 unified bingo/RTB state) with a
colour visibly distinct from every other state.

Red phase (Article III): both tests below FAIL until ``_STATE_COLORS`` gains
the ``RTB_FUEL_LIMIT`` entry. Importing ``_STATE_COLORS`` does NOT import
matplotlib (the animator imports pyplot lazily inside ``animate()``), so this
stays a fast, dependency-light unit test.
"""

from __future__ import annotations

import pytest

from cuas_sim.states import A29State
from cuas_sim.sysml_state_machine import _STATE_PHASE_BINDING
from cuas_sim.viz.animate import _STATE_COLORS, _STATE_COLOUR_ALIAS, _STATE_LEGEND

pytestmark = pytest.mark.viz


def test_state_colors_includes_rtb_fuel_limit() -> None:
    assert A29State.RTB_FUEL_LIMIT in _STATE_COLORS, (
        "FR-10: _STATE_COLORS must map A29State.RTB_FUEL_LIMIT."
    )
    colour = _STATE_COLORS[A29State.RTB_FUEL_LIMIT]
    assert isinstance(colour, str) and colour.startswith("#"), (
        f"RTB_FUEL_LIMIT colour must be a hex string, got {colour!r}."
    )


#: SysML ``MissionPhasesPackage`` order (Spec 022 oracle), for the legend check.
_PHASE_ORDER = (
    "Initial",
    "TransitToFARP",
    "FARPToCAPRepositioning",
    "CAPHoldingOrbit",
    "AirspacePolicing",
    "CAPToFARPReturn",
    "FARPToBaseRecovery",
    "MissionComplete",
)


def test_every_state_has_a_colour() -> None:
    # A state missing here would paint its marker the white fallback: adding a
    # member to A29State must come with a colour (or an alias) in the animator.
    missing = [s for s in A29State if s not in _STATE_COLORS]
    assert not missing, f"A29State members without a marker colour: {missing}"


def test_legend_lists_each_shown_state_once_in_phase_order() -> None:
    # The A-29 STATE legend names every state that holds the screen exactly
    # once, in the SysML phase order; the folded (zero-duration / legacy)
    # states are exactly the rest.
    shown = [state for state, _ in _STATE_LEGEND]
    assert len(shown) == len(set(shown)), "a state appears twice in the legend"
    assert set(shown).isdisjoint(_STATE_COLOUR_ALIAS)
    assert set(shown) | set(_STATE_COLOUR_ALIAS) == set(A29State)
    phase_idx = [_PHASE_ORDER.index(_STATE_PHASE_BINDING[s.value]) for s in shown]
    assert phase_idx == sorted(phase_idx), f"legend out of phase order: {shown}"


def test_folded_states_wear_a_legend_colour() -> None:
    legend_colours = dict(_STATE_LEGEND)
    for alias, shown in _STATE_COLOUR_ALIAS.items():
        assert shown in legend_colours, f"{alias} folds into {shown}, absent from the legend"
        assert _STATE_COLORS[alias] == legend_colours[shown]


def test_rtb_fuel_limit_colour_is_distinct() -> None:
    colour = _STATE_COLORS[A29State.RTB_FUEL_LIMIT].upper()
    others = [
        c.upper()
        for s, c in _STATE_COLORS.items()
        if s != A29State.RTB_FUEL_LIMIT
    ]
    assert colour not in others, (
        "FR-10: RTB_FUEL_LIMIT colour must be visually distinct from all "
        f"other states; {colour} collides with another entry."
    )
