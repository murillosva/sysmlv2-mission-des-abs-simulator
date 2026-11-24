"""Known-answer tests for SR-3 state-machine conformance (Spec 022).

Group B (Red phase, Article III): these MUST FAIL until Group C lands the parser
extension (``state def``), the phase oracle (``load_phases``), the event reader
(``load_events``) and the conformance check (``check_conformance`` /
``validate_conformance``).
"""

from __future__ import annotations

import gzip
import io
import json
import logging
from pathlib import Path

import pytest

from cuas_sim.states import A29State
from cuas_sim.sysml_parser import parse_package
from cuas_sim.sysml_state_machine import (
    ConformanceError,
    StateEvent,
    check_conformance,
    load_events,
    load_phases,
    validate_conformance,
)

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"
_MISSION_PHASES = _SYSML_ROOT / "Purpose" / "MissionPhasesPackage.sysml"

_PHASE_NAMES_IN_ORDER = (
    "Initial",
    "TransitToFARP",
    "FARPToCAPRepositioning",
    "CAPHoldingOrbit",
    "AirspacePolicing",
    "CAPToFARPReturn",
    "FARPToBaseRecovery",
    "MissionComplete",
)

#: Raw parser order = file order (spec §5.1 T1): the 6 ``:> Phase`` blocks first,
#: then the special ``Initial`` / ``MissionComplete`` states declared at the end of
#: the file. ``load_phases`` later reorders these into the oracle order above
#: (Initial → 0, MissionComplete → 7); the AST parser preserves file order.
_STATE_DEFS_FILE_ORDER = (
    "TransitToFARP",
    "FARPToCAPRepositioning",
    "CAPHoldingOrbit",
    "AirspacePolicing",
    "CAPToFARPReturn",
    "FARPToBaseRecovery",
    "Initial",
    "MissionComplete",
)


def _chain(states: list[str], dt: float = 10.0) -> list[StateEvent]:
    """Build the StateEvent list for a chronological chain of states.

    ``states[0]`` is the implicit initial state (becomes the first event's ``old``).
    """
    events: list[StateEvent] = []
    for i in range(len(states) - 1):
        events.append(StateEvent(t_s=dt * (i + 1), old=states[i], new=states[i + 1]))
    return events


# ─── T1: parser reads the real MissionPhasesPackage ─────────────────────────


def test_parser_reads_mission_phases() -> None:
    pkg = parse_package(_MISSION_PHASES.read_text(encoding="utf-8"))
    names = [sd.name for sd in pkg.state_defs]
    assert names == list(_STATE_DEFS_FILE_ORDER)
    by_name = {sd.name: sd for sd in pkg.state_defs}
    for phase in _PHASE_NAMES_IN_ORDER[1:7]:  # the 6 :> Phase blocks
        assert by_name[phase].specializes == ("Phase",), phase
    for special in ("Initial", "MissionComplete"):
        assert by_name[special].specializes == ()
        assert by_name[special].actions == ()
    step_types = [a.type_ref for a in by_name["AirspacePolicing"].actions]
    assert "DetectUASByE99M" in step_types


# ─── T2: load_phases derives the ordered oracle ─────────────────────────────


def test_load_phases_oracle_order() -> None:
    model = load_phases(_SYSML_ROOT)
    assert model.oracle_loaded is True
    assert tuple(p.sysml_name for p in model.phases) == _PHASE_NAMES_IN_ORDER
    assert tuple(p.index for p in model.phases) == tuple(range(8))
    idx = model.phase_index_of_state
    assert idx[A29State.AT_BASE.value] == 0
    assert idx[A29State.TRANSIT_TO_FARP.value] == 1
    assert idx[A29State.AT_CAP_PEA.value] == 4
    assert idx[A29State.PURSUING.value] == 4
    assert idx[A29State.RTB_FUEL_LIMIT.value] == 5
    assert idx[A29State.MISSION_COMPLETE.value] == 7
    # All 15 enum states are bound.
    for st in A29State:
        assert st.value in idx, st.value


# ─── T3: happy path is conformant, combat states are extensions ─────────────


def test_happy_path_conformant_with_extensions() -> None:
    model = load_phases(_SYSML_ROOT)
    chain = _chain([
        "AT_BASE", "TRANSIT_TO_FARP", "AT_FARP_PRE_PEA", "REPOSITIONING_TO_CAP",
        "HOLDING_ORBIT", "AT_CAP_PEA",
        "PURSUING", "ENGAGING", "RETURNING_TO_CAP", "AT_CAP_PEA",  # cycle 1
        "PURSUING", "AT_CAP_PEA",                                  # cycle 2 (no kill)
        "RTB_FUEL_LIMIT", "AT_FARP_POST_PEA", "RECOVERING_TO_BASE", "MISSION_COMPLETE",
    ])
    report = check_conformance({"A-29-1": chain}, model)
    assert report.conformant is True
    assert report.violations == []
    assert report.extensions_used.get("PURSUING") == 2
    assert report.extensions_used.get("ENGAGING") == 1
    assert report.extensions_used.get("RETURNING_TO_CAP") == 1
    assert report.extensions_used.get("RTB_FUEL_LIMIT") == 1


# ─── T4: out-of-order transition is a violation (the test with teeth) ────────


def test_out_of_order_violation() -> None:
    model = load_phases(_SYSML_ROOT)
    chain = _chain([
        "AT_BASE", "TRANSIT_TO_FARP", "REPOSITIONING_TO_CAP", "HOLDING_ORBIT",
        "AT_CAP_PEA", "HOLDING_ORBIT",  # phase 4 -> phase 3 backstep
    ])
    report = check_conformance({"A-29-1": chain}, model)
    assert report.conformant is False
    ooo = [v for v in report.violations if v.kind == "out_of_order"]
    assert len(ooo) >= 1
    v = ooo[0]
    assert v.from_state == "AT_CAP_PEA"
    assert v.to_state == "HOLDING_ORBIT"
    assert v.a29_id == "A-29-1"
    assert v.t_s > 0.0


# ─── T5: missing mandatory phase (AirspacePolicing) is a violation ──────────


def test_missing_mandatory_phase_violation() -> None:
    model = load_phases(_SYSML_ROOT)
    chain = _chain([
        "AT_BASE", "TRANSIT_TO_FARP", "REPOSITIONING_TO_CAP", "HOLDING_ORBIT",
        "RTB_FUEL_LIMIT", "AT_FARP_POST_PEA", "RECOVERING_TO_BASE", "MISSION_COMPLETE",
    ])
    report = check_conformance({"A-29-1": chain}, model)
    assert report.conformant is False
    miss = [v for v in report.violations if v.kind == "missing_phase"]
    assert len(miss) >= 1
    assert "AirspacePolicing" in miss[0].detail


# ─── T6: unknown state is a violation ───────────────────────────────────────


def test_unknown_state_violation() -> None:
    model = load_phases(_SYSML_ROOT)
    chain = _chain(["AT_BASE", "TRANSIT_TO_FARP", "WARP_DRIVE", "AT_CAP_PEA"])
    report = check_conformance({"A-29-1": chain}, model)
    assert report.conformant is False
    unk = [v for v in report.violations if v.kind == "unknown_state"]
    assert len(unk) >= 1
    assert unk[0].to_state == "WARP_DRIVE"
    assert unk[0].t_s > 0.0


# ─── T7: leniency — missing oracle never raises, check is vacuously conformant


def test_lenient_missing_oracle(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        model = load_phases(tmp_path)  # no Purpose/MissionPhasesPackage.sysml
    assert model.oracle_loaded is False
    chain = _chain(["AT_BASE", "AT_CAP_PEA", "HOLDING_ORBIT"])  # would be a backstep
    report = check_conformance({"A-29-1": chain}, model)
    assert report.conformant is True
    assert report.violations == []
    assert report.oracle_loaded is False


# ─── T8: both Spec 018 flag paths accepted ──────────────────────────────────


def test_legacy_flag_path_accepted() -> None:
    model = load_phases(_SYSML_ROOT)
    chain = _chain([
        "AT_BASE", "TRANSIT_TO_FARP", "AT_FARP_PRE_PEA", "REPOSITIONING_TO_CAP",
        "HOLDING_ORBIT", "AT_CAP_PEA",
        "DEPARTING_CAP", "AT_FARP_POST_PEA", "RECOVERING_TO_BASE", "MISSION_COMPLETE",
    ])
    report = check_conformance({"A-29-1": chain}, model)
    assert report.conformant is True


# ─── FR-7: load_events filters info-only (old == new) records ───────────────


def test_load_events_filters_info_only(tmp_path: Path) -> None:
    run_dir = tmp_path
    rows = [
        {"config": "baseline", "a29": "A-29-1", "t_s": 0.0,
         "old": "AT_BASE", "new": "TRANSIT_TO_FARP"},
        {"config": "baseline", "a29": "A-29-1", "t_s": 5.0,
         "old": "RTB_FUEL_LIMIT_INFO", "new": "RTB_FUEL_LIMIT_INFO"},  # info-only
        {"config": "baseline", "a29": "A-29-1", "t_s": 6.0,
         "old": "TRANSIT_TO_FARP", "new": "AT_FARP_PRE_PEA"},
    ]
    path = run_dir / "events_baseline.jsonl.gz"
    with (
        open(path, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8", write_through=True) as fh,
    ):
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # An (empty) modernized file so the reader does not choke on its absence.
    (run_dir / "events_modernized.jsonl.gz").write_bytes(
        gzip.compress(b"", mtime=0)
    )

    events = load_events(run_dir)
    evs = events["baseline"]["A-29-1"]
    assert len(evs) == 2  # info-only row dropped
    assert all(e.old != e.new for e in evs)
    assert [e.new for e in evs] == ["TRANSIT_TO_FARP", "AT_FARP_PRE_PEA"]


# ─── FR-11: validate_conformance is the only port that raises ───────────────


def _write_events(run_dir: Path, config: str, a29: str, states: list[str]) -> None:
    path = run_dir / f"events_{config}.jsonl.gz"
    with (
        open(path, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8", write_through=True) as fh,
    ):
        for i in range(len(states) - 1):
            fh.write(json.dumps({
                "config": config, "a29": a29, "t_s": 10.0 * (i + 1),
                "old": states[i], "new": states[i + 1],
            }) + "\n")


def test_validate_conformance_raises_on_violation(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    _write_events(bad, "baseline", "A-29-1",
                  ["AT_BASE", "AT_CAP_PEA", "HOLDING_ORBIT"])  # backstep
    (bad / "events_modernized.jsonl.gz").write_bytes(gzip.compress(b"", mtime=0))
    with pytest.raises(ConformanceError):
        validate_conformance(bad, _SYSML_ROOT)

    good = tmp_path / "good"
    good.mkdir()
    _write_events(good, "baseline", "A-29-1",
                  ["AT_BASE", "TRANSIT_TO_FARP", "AT_FARP_PRE_PEA",
                   "REPOSITIONING_TO_CAP", "HOLDING_ORBIT", "AT_CAP_PEA",
                   "RTB_FUEL_LIMIT", "AT_FARP_POST_PEA", "RECOVERING_TO_BASE",
                   "MISSION_COMPLETE"])
    (good / "events_modernized.jsonl.gz").write_bytes(gzip.compress(b"", mtime=0))
    validate_conformance(good, _SYSML_ROOT)  # must NOT raise
