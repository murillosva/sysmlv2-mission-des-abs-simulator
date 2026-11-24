"""Property-based oracles for SR-3 state-machine conformance (Spec 022 §5.2).

Group B (Red phase, Article III): MUST FAIL until ``load_phases`` /
``check_conformance`` are implemented (Group C).

P1 — any phase-index-monotonic sequence (with the mandatory phase visited before
     recovery) is conformant.
P2 — inserting a strictly-lower-phase state into a conformant sequence always
     yields an ``out_of_order`` violation (the check has teeth).
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from cuas_sim.sysml_state_machine import StateEvent, check_conformance, load_phases

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"

# Local canonical phase → states table (mirrors spec §3.1 FR-6); used only to
# GENERATE candidate sequences. The conformance oracle itself comes from
# load_phases(sysml/), so a divergence here would be a real bug.
_PHASE_STATES: list[list[str]] = [
    ["AT_BASE"],                                                       # 0 Initial
    ["TRANSIT_TO_FARP", "AT_FARP_PRE_PEA"],                            # 1
    ["REPOSITIONING_TO_CAP"],                                          # 2
    ["HOLDING_ORBIT"],                                                 # 3
    ["AT_CAP_PEA", "PURSUING", "ENGAGING", "RETURNING_TO_CAP"],        # 4 AirspacePolicing
    ["RTB_FUEL_LIMIT", "DEPARTING_CAP", "RTB_BINGO", "AT_FARP_POST_PEA"],  # 5
    ["RECOVERING_TO_BASE"],                                           # 6
    ["MISSION_COMPLETE"],                                             # 7
]


def _chain(states: list[str]) -> list[StateEvent]:
    return [
        StateEvent(t_s=10.0 * (i + 1), old=states[i], new=states[i + 1])
        for i in range(len(states) - 1)
    ]


@st.composite
def _monotonic_states(draw: st.DrawFn, min_max_phase: int = 0) -> list[str]:
    """Draw a non-decreasing phase-index state sequence visiting phases 0..k."""
    k = draw(st.integers(min_value=max(0, min_max_phase), max_value=7))
    out: list[str] = []
    for phase in range(k + 1):
        n = draw(st.integers(min_value=1, max_value=3))
        for _ in range(n):
            out.append(draw(st.sampled_from(_PHASE_STATES[phase])))
    return out


@given(states=_monotonic_states())
def test_monotonic_sequences_conform(states: list[str]) -> None:
    model = load_phases(_SYSML_ROOT)
    report = check_conformance({"A-29-1": _chain(states)}, model)
    assert report.conformant is True, report.violations


@given(data=st.data())
def test_inserting_backstep_breaks_conformance(data: st.DataObject) -> None:
    model = load_phases(_SYSML_ROOT)
    states = data.draw(_monotonic_states(min_max_phase=1))
    idx = model.phase_index_of_state
    # Find the first position whose running-max phase index is >= 1, then insert
    # AT_BASE (phase 0) right after it: 0 < running_max ⇒ out_of_order.
    running_max = -1
    insert_at = None
    for i, s in enumerate(states):
        running_max = max(running_max, idx[s])
        if running_max >= 1:
            insert_at = i + 1
            break
    assert insert_at is not None  # min_max_phase=1 guarantees a phase >= 1
    mutated = [*states[:insert_at], "AT_BASE", *states[insert_at:]]
    report = check_conformance({"A-29-1": _chain(mutated)}, model)
    assert report.conformant is False
    assert any(v.kind == "out_of_order" for v in report.violations)
