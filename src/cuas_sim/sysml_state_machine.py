"""SysML v2 state-machine conformance.

Reads the ordered mission phases of ``sysml/Purpose/MissionPhasesPackage.sysml``
(via the AST parser) and checks that the ``StateChangeEvent``
stream persisted to ``events_<config>.jsonl.gz`` conforms to that phase ordering.
Python-only combat/failure states (``PURSUING``, ``ENGAGING``, ``RETURNING_TO_CAP``,
``RTB_FUEL_LIMIT``/``RTB_BINGO``/``DEPARTING_CAP``) are documented **extensions**,
not violations.

Two contracts, deliberately separated (as in the requirements checker):

* :func:`load_phases` / :func:`load_events` / :func:`check_conformance` are
  **lenient** — they NEVER raise on ``.sysml`` or log content.
* :func:`validate_conformance` is the **only** entry point that may raise
  :class:`ConformanceError`.

The module is pure (no RNG, no global state; the only I/O is the file reads in
``load_phases`` / ``load_events``).
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cuas_sim.sysml_parser import SysMLParseError, SysMLStateDef, parse_package

logger = logging.getLogger(__name__)

__all__ = [
    "ConformanceError",
    "ConformanceReport",
    "ConformanceViolation",
    "PhaseModel",
    "PhaseSpec",
    "StateEvent",
    "check_conformance",
    "load_events",
    "load_phases",
    "validate_conformance",
]


class ConformanceError(Exception):
    """Raised by the explicit validation step when ≥1 conformance violation exists."""


@dataclass(frozen=True, slots=True)
class StateEvent:
    """One A-29 state transition read back from ``events_<config>.jsonl.gz``."""

    t_s: float
    old: str
    new: str


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """One SysML mission phase: its file-order index and the A29 states it covers."""

    sysml_name: str
    index: int
    is_special: bool
    action_steps: tuple[str, ...]
    states: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseModel:
    """The ordered phase oracle derived from ``MissionPhasesPackage`` + the binding."""

    package: str
    phases: tuple[PhaseSpec, ...]
    phase_index_of_state: dict[str, int]
    mandatory_phases: frozenset[str]
    oracle_loaded: bool


@dataclass(frozen=True, slots=True)
class ConformanceViolation:
    """A single non-conformance between the logged stream and the SysML oracle."""

    a29_id: str
    t_s: float
    kind: str  # "out_of_order" | "missing_phase" | "unknown_state"
    from_state: str | None
    to_state: str | None
    detail: str


@dataclass(slots=True)
class ConformanceReport:
    """Outcome of :func:`check_conformance` — pure classification, never raises."""

    oracle_loaded: bool = False
    violations: list[ConformanceViolation] = field(default_factory=list)
    extensions_used: dict[str, int] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def conformant(self) -> bool:
        return not self.violations


# ─── Binding: A29State → SysML phase name ───────────────────
#
# Explicit, documented table (analogous to ``sysml_requirements._BINDINGS``)
# covering all 15 ``A29State`` values. The Python-only combat/failure states
# (PURSUING/ENGAGING/RETURNING_TO_CAP, RTB_FUEL_LIMIT/DEPARTING_CAP/RTB_BINGO)
# have no ``state def`` of their own — they map by documentation to the SysML
# phase that contains them (AirspacePolicing / CAPToFARPReturn). Both spawn-window
# flag paths (RTB_FUEL_LIMIT default; DEPARTING_CAP/RTB_BINGO legacy) map to the
# same CAPToFARPReturn phase, so neither is a violation.
_STATE_PHASE_BINDING: dict[str, str] = {
    "AT_BASE": "Initial",
    "TRANSIT_TO_FARP": "TransitToFARP",
    "AT_FARP_PRE_PEA": "TransitToFARP",
    "REPOSITIONING_TO_CAP": "FARPToCAPRepositioning",
    "HOLDING_ORBIT": "CAPHoldingOrbit",
    "AT_CAP_PEA": "AirspacePolicing",
    "PURSUING": "AirspacePolicing",
    "ENGAGING": "AirspacePolicing",
    "RETURNING_TO_CAP": "AirspacePolicing",
    "RTB_FUEL_LIMIT": "CAPToFARPReturn",
    "DEPARTING_CAP": "CAPToFARPReturn",
    "RTB_BINGO": "CAPToFARPReturn",
    "AT_FARP_POST_PEA": "CAPToFARPReturn",
    "RECOVERING_TO_BASE": "FARPToBaseRecovery",
    "MISSION_COMPLETE": "MissionComplete",
}

#: Python-only combat/failure states — documented extensions, never violations.
#: Counted and reported in :attr:`ConformanceReport.extensions_used`.
_EXTENSION_STATES: frozenset[str] = frozenset({
    "PURSUING", "ENGAGING", "RETURNING_TO_CAP",
    "RTB_FUEL_LIMIT", "DEPARTING_CAP", "RTB_BINGO",
})

#: The mission phase a sortie MUST visit before recovery/terminal.
_MANDATORY_PHASES: frozenset[str] = frozenset({"AirspacePolicing"})

#: First recovery/terminal phase — reaching its index (or beyond) without a
#: mandatory phase is a ``missing_phase`` violation.
_RECOVERY_PHASE: str = "CAPToFARPReturn"

#: The two special (non-``:> Phase``) states that bracket the mission, in oracle
#: order: ``Initial`` first, ``MissionComplete`` last.
_SPECIAL_FIRST: str = "Initial"
_SPECIAL_LAST: str = "MissionComplete"


def _empty_model() -> PhaseModel:
    """A vacuous oracle (file missing / parse failed) — checks are conformant."""
    return PhaseModel(
        package="",
        phases=(),
        phase_index_of_state={},
        mandatory_phases=_MANDATORY_PHASES,
        oracle_loaded=False,
    )


def load_phases(root: Path | str = "sysml") -> PhaseModel:
    """Parse ``MissionPhasesPackage`` into the ordered phase oracle.

    The 6 ``:> Phase`` states keep their file order; the special ``Initial`` /
    ``MissionComplete`` states bracket them (indices 0 and 7). LENIENT — a missing
    file or :class:`SysMLParseError` yields ``PhaseModel(oracle_loaded=False)``
    (WARNING logged), never raising.
    """
    path = Path(root) / "Purpose" / "MissionPhasesPackage.sysml"
    if not path.is_file():
        logger.warning("Missing SysML phase oracle: %s — conformance is vacuous", path)
        return _empty_model()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read %s: %s — conformance is vacuous", path, exc)
        return _empty_model()
    try:
        pkg = parse_package(text)
    except SysMLParseError as exc:
        logger.warning("SysML parse failed for %s: %s — conformance is vacuous", path, exc)
        return _empty_model()

    by_name = {sd.name: sd for sd in pkg.state_defs}
    phase_defs = [sd for sd in pkg.state_defs if sd.specializes == ("Phase",)]
    ordered: list[SysMLStateDef] = []
    if _SPECIAL_FIRST in by_name:
        ordered.append(by_name[_SPECIAL_FIRST])
    ordered.extend(phase_defs)
    if _SPECIAL_LAST in by_name:
        ordered.append(by_name[_SPECIAL_LAST])

    states_of_phase: dict[str, list[str]] = {}
    for state_value, phase_name in _STATE_PHASE_BINDING.items():
        states_of_phase.setdefault(phase_name, []).append(state_value)

    phases: list[PhaseSpec] = []
    name_to_index: dict[str, int] = {}
    for index, sd in enumerate(ordered):
        phases.append(PhaseSpec(
            sysml_name=sd.name,
            index=index,
            is_special=sd.specializes != ("Phase",),
            action_steps=tuple(a.type_ref or a.name for a in sd.actions),
            states=tuple(states_of_phase.get(sd.name, ())),
        ))
        name_to_index[sd.name] = index

    phase_index_of_state = {
        state_value: name_to_index[phase_name]
        for state_value, phase_name in _STATE_PHASE_BINDING.items()
        if phase_name in name_to_index
    }
    return PhaseModel(
        package=pkg.name,
        phases=tuple(phases),
        phase_index_of_state=phase_index_of_state,
        mandatory_phases=_MANDATORY_PHASES,
        oracle_loaded=True,
    )


def load_events(run_dir: Path | str) -> dict[str, dict[str, list[StateEvent]]]:
    """Read ``events_<config>.jsonl.gz`` into ``{config: {a29: [StateEvent, …]}}``.

    Mirrors ``viz/animate.py::_parse_events``: info-only diagnostic records
    (``old == new``, e.g. ``RTB_FUEL_LIMIT_INFO``) are dropped; events are grouped
    by A-29 and sorted stably by ``t_s``. A missing file yields an
    empty config; LENIENT — never raises on log content.
    """
    run_path = Path(run_dir)
    out: dict[str, dict[str, list[StateEvent]]] = {}
    for config in ("baseline", "modernized"):
        by_a29: dict[str, list[StateEvent]] = {}
        out[config] = by_a29
        path = run_path / f"events_{config}.jsonl.gz"
        if not path.is_file():
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        old = str(obj["old"])
                        new = str(obj["new"])
                        if old == new:  # info-only diagnostic, not a transition
                            continue
                        by_a29.setdefault(str(obj["a29"]), []).append(
                            StateEvent(t_s=float(obj["t_s"]), old=old, new=new)
                        )
                    except (ValueError, KeyError, TypeError) as exc:
                        logger.warning("Skipping malformed event in %s: %s", path, exc)
        except OSError as exc:
            logger.warning("Cannot read %s: %s — skipping config %s", path, exc, config)
            continue
        for evs in by_a29.values():
            evs.sort(key=lambda e: e.t_s)  # stable: ties keep append/causal order
    return out


def check_conformance(
    events: Mapping[str, Sequence[StateEvent]], phases: PhaseModel
) -> ConformanceReport:
    """Classify a single config's per-A29 event stream against the oracle.

    PURE (no RNG, no I/O, no global state). Applies the non-decreasing phase-index
    rule (``running_max``); flags ``out_of_order`` / ``missing_phase`` /
    ``unknown_state`` and counts documented extensions. If the oracle
    was not loaded, the report is vacuously conformant.
    """
    report = ConformanceReport(oracle_loaded=phases.oracle_loaded)
    if not phases.oracle_loaded:
        return report

    idx = phases.phase_index_of_state
    recovery_index = next(
        (p.index for p in phases.phases if p.sysml_name == _RECOVERY_PHASE), None
    )
    mandatory_index = {
        p.sysml_name: p.index
        for p in phases.phases
        if p.sysml_name in phases.mandatory_phases
    }

    for a29_id, evs in events.items():
        if not evs:
            continue
        running_max = -1
        visited: set[int] = set()

        seed_idx = idx.get(evs[0].old)
        if seed_idx is None:
            report.violations.append(ConformanceViolation(
                a29_id=a29_id, t_s=evs[0].t_s, kind="unknown_state",
                from_state=None, to_state=evs[0].old,
                detail=f"unknown state {evs[0].old!r} (initial) not in phase binding",
            ))
        else:
            running_max = seed_idx
            visited.add(seed_idx)

        for ev in evs:
            new_idx = idx.get(ev.new)
            if new_idx is None:
                report.violations.append(ConformanceViolation(
                    a29_id=a29_id, t_s=ev.t_s, kind="unknown_state",
                    from_state=ev.old, to_state=ev.new,
                    detail=f"unknown state {ev.new!r} not in phase binding",
                ))
                continue
            if ev.new in _EXTENSION_STATES:
                report.extensions_used[ev.new] = report.extensions_used.get(ev.new, 0) + 1
            if new_idx < running_max:
                report.violations.append(ConformanceViolation(
                    a29_id=a29_id, t_s=ev.t_s, kind="out_of_order",
                    from_state=ev.old, to_state=ev.new,
                    detail=(f"phase backstep {ev.old} (>= idx {running_max}) "
                            f"-> {ev.new} (idx {new_idx})"),
                ))
            running_max = max(running_max, new_idx)
            visited.add(new_idx)

        if recovery_index is not None and running_max >= recovery_index:
            for phase_name, phase_idx in mandatory_index.items():
                if phase_idx not in visited:
                    report.violations.append(ConformanceViolation(
                        a29_id=a29_id, t_s=evs[-1].t_s, kind="missing_phase",
                        from_state=None, to_state=None,
                        detail=(f"reached recovery/terminal without mandatory "
                                f"phase {phase_name}"),
                    ))
        report.details[a29_id] = f"{len(evs)} transitions"
    return report


def validate_conformance(run_dir: Path | str, root: Path | str = "sysml") -> None:
    """Raise :class:`ConformanceError` if any A-29 stream is non-conformant.

    This is the ONLY entry point that raises; ``load_phases`` / ``load_events`` /
    ``check_conformance`` are always lenient.
    """
    phases = load_phases(root)
    events = load_events(run_dir)
    offending: list[str] = []
    for config, by_a29 in events.items():
        report = check_conformance(by_a29, phases)
        for v in report.violations:
            offending.append(
                f"  - [{config}] {v.a29_id} @ {v.t_s:.1f}s {v.kind}: {v.detail}"
            )
    if offending:
        raise ConformanceError(
            "State-machine conformance violations vs SysML MissionPhasesPackage:\n"
            + "\n".join(offending)
        )
