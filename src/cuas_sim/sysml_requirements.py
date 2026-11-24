"""SysML v2 requirements + constraint validation.

Loads the ``requirement def`` / ``require constraint`` blocks of the auto-generated
CoSMA ``Requirements/*.sysml`` packages (via the AST parser),
binds the checkable Technical requirements to :class:`~cuas_sim.config.MissionConfig`
fields, and offers an **explicit, opt-in** validation step plus a traceability matrix.

Two contracts, deliberately separated:

* :func:`load_requirements` is **lenient** — it NEVER raises on ``.sysml`` content
  (a :class:`~cuas_sim.sysml_parser.SysMLParseError` is logged and that file skipped),
  honouring the loader invariant "simulation never crashes because of .sysml issues".
* :func:`validate_against_requirements` (and :meth:`MissionConfig.validate_against_requirements`)
  is the **only** entry point that may raise :class:`RequirementsValidationError`.

The module is pure (no RNG, no global state; the only I/O is the file read in
``load_requirements``).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cuas_sim.config import MissionConfig
from cuas_sim.sysml_loader import _to_si
from cuas_sim.sysml_parser import (
    ConstraintExpr,
    ConstraintLiteral,
    ConstraintRef,
    ConstraintUnary,
    SysMLAttribute,
    SysMLParseError,
    SysMLRequirementDef,
    parse_package,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ParsedRequirement",
    "RequirementBinding",
    "RequirementsReport",
    "RequirementsValidationError",
    "TraceabilityRow",
    "build_traceability_matrix",
    "check_requirements",
    "evaluate_constraint",
    "load_requirements",
    "validate_against_requirements",
]

#: Equality tolerance for ``==`` between numeric values.
_EQ_REL_TOL = 1e-9
_EQ_ABS_TOL = 1e-9
#: Nautical mile → metre (matches ``sysml_loader`` unit table; TR-030 binding).
_NMI_TO_M = 1852.0


class RequirementsValidationError(Exception):
    """Raised by the explicit validation step when ≥1 bound constraint is violated."""


@dataclass(frozen=True, slots=True)
class ParsedRequirement:
    """One ``requirement def`` with SI-converted literal attributes and its constraint."""

    req_id: str
    name: str
    package: str
    attributes: dict[str, float | int | str | bool]
    constraint: ConstraintExpr | None
    refinements: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RequirementBinding:
    """Maps a requirement's ``actual*`` term(s) to MissionConfig value extractor(s).

    Each extractor returns a **list** of SI values (1 for a scalar binding, ≥2 for a
    range such as ``baseline_update_interval_s``); the constraint must hold for every
    combination.
    """

    actuals: dict[str, Callable[[MissionConfig], list[float]]]
    config_fields: str
    spec_ref: str


@dataclass(frozen=True, slots=True)
class TraceabilityRow:
    """One row of the requirement → config-field → module traceability matrix."""

    req_id: str
    name: str
    package: str
    config_fields: str
    spec_ref: str
    bound: bool


@dataclass(slots=True)
class RequirementsReport:
    """Outcome of :func:`check_requirements` — pure classification, never raises."""

    satisfied: list[str] = field(default_factory=list)
    violated: list[str] = field(default_factory=list)
    unbound: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)


# ─── Binding table — the curated requirement→config map ──
#
# Each extractor returns config value(s) in the SAME SI unit as the SysML limit.
# All bound fields are already SI (m, s, m/s, dimensionless) except
# ``zida_boundary_nmi`` (nautical miles), which TR-030 converts × 1852.

_BINDINGS: dict[str, RequirementBinding] = {
    "TR-001": RequirementBinding(
        {"actualEmploymentRange": lambda c: [c.baseline_envelope_m[1]]},
        "baseline_envelope_m[1]", "policies/weapon.py",
    ),
    "TR-002": RequirementBinding(
        {"actualEmploymentRange": lambda c: [c.modernized_envelope_m[1]]},
        "modernized_envelope_m[1]", "policies/weapon.py",
    ),
    "TR-003": RequirementBinding(
        {"actualUpdateInterval": lambda c: list(c.baseline_update_interval_s)},
        "baseline_update_interval_s", "policies/tracking.py",
    ),
    "TR-004": RequirementBinding(
        {"actualUpdateInterval": lambda c: list(c.modernized_update_interval_s)},
        "modernized_update_interval_s", "policies/tracking.py",
    ),
    "TR-005": RequirementBinding(
        {
            "actualAcquisitionDistance": lambda c: [c.baseline_max_acquisition_m],
            "actualAcquisitionTime": lambda c: [c.baseline_reaction_time_min_s],
        },
        "baseline_max_acquisition_m, baseline_reaction_time_min_s", "policies/tracking.py",
    ),
    "TR-006": RequirementBinding(
        {
            "actualLaserRangefinderRange": lambda c: [c.modernized_eoir_max_m],
            "actualRangefinderAccuracy": lambda c: [c.modernized_eoir_accuracy_m],
        },
        "modernized_eoir_max_m, modernized_eoir_accuracy_m", "policies/tracking.py",
    ),
    "TR-010": RequirementBinding(
        {"actualCruiseSpeed": lambda c: [c.a29_cruise_speed_mps]},
        "a29_cruise_speed_mps", "agents/a29.py",
    ),
    "TR-011": RequirementBinding(
        {"actualEndurance": lambda c: [c.a29_endurance_s]},
        "a29_endurance_s", "agents/a29.py",
    ),
    "TR-016": RequirementBinding(
        {"actualDimensionalPenalty": lambda c: [c.dimensional_penalty_mr]},
        "dimensional_penalty_mr", "policies/weapon.py",
    ),
    "TR-021": RequirementBinding(
        {
            "actualMultiRotorProbability": lambda c: [c.uas_archetype_p_mr],
            "actualFixedWingProbability": lambda c: [1.0 - c.uas_archetype_p_mr],
        },
        "uas_archetype_p_mr", "world/arrivals.py",
    ),
    "TR-023": RequirementBinding(
        {"actualPEACommencementTime": lambda c: [c.t_pea_start_s]},
        "t_pea_start_s", "world/arrivals.py",
    ),
    "TR-028": RequirementBinding(
        {"actualMaxSpeed": lambda c: [c.uas_mr.max_speed_mps]},
        "uas_mr.max_speed_mps", "agents/uas.py",
    ),
    "TR-029": RequirementBinding(
        {"actualMaxSpeed": lambda c: [c.uas_fw.max_speed_mps]},
        "uas_fw.max_speed_mps", "agents/uas.py",
    ),
    "TR-030": RequirementBinding(
        {"actualZIDABoundaryDistance": lambda c: [c.zida_boundary_nmi * _NMI_TO_M]},
        "zida_boundary_nmi", "world/boundary.py",
    ),
}


# ─── Loading (lenient — never raises) ────────────────


def _attr_value_si(a: SysMLAttribute) -> float | int | str | bool | None:
    """SI-convert a requirement attribute literal (``None`` for value-less actuals)."""
    if a.value is None:
        return None
    if isinstance(a.value, bool):
        return a.value
    if isinstance(a.value, (int, float)) and a.unit is not None:
        si_val, _ = _to_si(float(a.value), a.unit)
        return si_val
    return a.value  # unit-less number or string


def _to_parsed(rd: SysMLRequirementDef, package: str) -> ParsedRequirement:
    attrs: dict[str, float | int | str | bool] = {}
    for a in rd.attributes:
        v = _attr_value_si(a)
        if v is not None:
            attrs[a.name] = v
    constraint = rd.constraints[0].expr if rd.constraints else None
    return ParsedRequirement(
        req_id=rd.short_name or rd.name,
        name=rd.name,
        package=package,
        attributes=attrs,
        constraint=constraint,
        refinements=rd.refinements,
    )


def load_requirements(root: Path | str) -> list[ParsedRequirement]:
    """Parse all ``Requirements/*.sysml`` packages. NEVER raises."""
    req_dir = Path(root) / "Requirements"
    out: list[ParsedRequirement] = []
    if not req_dir.is_dir():
        logger.warning("Missing SysML Requirements dir: %s — no requirements loaded", req_dir)
        return out
    for path in sorted(req_dir.glob("*.sysml")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read %s: %s — skipping", path, exc)
            continue
        try:
            pkg = parse_package(text)
        except SysMLParseError as exc:
            logger.warning("SysML parse failed for %s: %s — skipping file", path, exc)
            continue
        for rd in pkg.requirement_defs:
            out.append(_to_parsed(rd, pkg.name))
    return out


# ─── Evaluation (pure) ──────────────────────────────────────────────────────


def _eval(expr: ConstraintExpr, env: dict[str, float | bool | str]) -> float | bool | str:
    if isinstance(expr, ConstraintLiteral):
        return expr.value
    if isinstance(expr, ConstraintRef):
        return env[expr.name]
    if isinstance(expr, ConstraintUnary):
        v = _eval(expr.operand, env)
        if expr.op == "not":
            return not v
        if expr.op == "-":
            return -float(v)
        raise SysMLParseError(f"unknown unary op {expr.op!r}")
    # ConstraintBinary
    if expr.op == "and":
        return bool(_eval(expr.left, env)) and bool(_eval(expr.right, env))
    if expr.op == "or":
        return bool(_eval(expr.left, env)) or bool(_eval(expr.right, env))
    left = _eval(expr.left, env)
    right = _eval(expr.right, env)
    if expr.op == "==":
        if _is_number(left) and _is_number(right):
            return math.isclose(float(left), float(right),
                                rel_tol=_EQ_REL_TOL, abs_tol=_EQ_ABS_TOL)
        return left == right
    if expr.op == "<=":
        return left <= right  # type: ignore[operator]
    if expr.op == ">=":
        return left >= right  # type: ignore[operator]
    if expr.op == "<":
        return left < right  # type: ignore[operator]
    if expr.op == ">":
        return left > right  # type: ignore[operator]
    raise SysMLParseError(f"unknown binary op {expr.op!r}")


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def evaluate_constraint(expr: ConstraintExpr, env: dict[str, float | bool | str]) -> bool:
    """Evaluate a constraint tree to a boolean."""
    return bool(_eval(expr, env))


# ─── Checking + validation ──────────────────────────────────────────────────


def check_requirements(
    cfg: MissionConfig, requirements: list[ParsedRequirement]
) -> RequirementsReport:
    """Classify each requirement as satisfied / violated / unbound. Pure."""
    report = RequirementsReport()
    for req in requirements:
        binding = _BINDINGS.get(req.req_id)
        if binding is None or req.constraint is None:
            report.unbound.append(req.req_id)
            continue
        limits = dict(req.attributes)
        actual_names = list(binding.actuals)
        value_lists = [binding.actuals[name](cfg) for name in actual_names]
        violated = False
        offending: dict[str, float] | None = None
        for combo in itertools.product(*value_lists):
            env: dict[str, float | bool | str] = dict(limits)
            for name, val in zip(actual_names, combo, strict=True):
                env[name] = val
            try:
                ok = evaluate_constraint(req.constraint, env)
            except (KeyError, TypeError) as exc:  # defensive — binding bug, not a config fault
                violated = True
                offending = dict(zip(actual_names, combo, strict=True))
                report.details[req.req_id] = f"{req.name}: evaluation error ({exc})"
                break
            if not ok:
                violated = True
                offending = dict(zip(actual_names, combo, strict=True))
                break
        if violated:
            report.violated.append(req.req_id)
            report.details.setdefault(
                req.req_id,
                f"{req.name}: {binding.config_fields} violates constraint; actuals={offending}",
            )
        else:
            report.satisfied.append(req.req_id)
            report.details[req.req_id] = f"{req.name}: OK ({binding.config_fields})"
    return report


def validate_against_requirements(cfg: MissionConfig, root: Path | str = "sysml") -> None:
    """Raise :class:`RequirementsValidationError` if any bound constraint is violated.

    This is the ONLY entry point that raises; loading
    requirements (:func:`load_requirements`) is always lenient.
    """
    report = check_requirements(cfg, load_requirements(root))
    if report.violated:
        lines = "\n".join(f"  - {rid}: {report.details.get(rid, '')}" for rid in report.violated)
        raise RequirementsValidationError(
            "MissionConfig violates SysML require constraint(s):\n" + lines
        )


# ─── Traceability matrix ───────────────────────────────────


def build_traceability_matrix(requirements: list[ParsedRequirement]) -> list[TraceabilityRow]:
    """Build the requirement → config-field → spec matrix (all requirements listed)."""
    rows: list[TraceabilityRow] = []
    for req in requirements:
        binding = _BINDINGS.get(req.req_id)
        if binding is not None:
            rows.append(TraceabilityRow(
                req.req_id, req.name, req.package, binding.config_fields, binding.spec_ref, True,
            ))
        else:
            rows.append(TraceabilityRow(
                req.req_id, req.name, req.package, "—", "—", False,
            ))
    return rows
