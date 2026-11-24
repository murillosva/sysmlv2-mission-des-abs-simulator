"""Unit tests for SysML requirements parsing + constraint validation (Spec 021 SR-2).

Group B (Red phase): these MUST FAIL until the parser is extended for
``requirement def``/``require constraint`` and ``sysml_requirements`` is
implemented (Spec 021 Group C).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.sysml_parser import (
    ConstraintBinary,
    ConstraintRef,
    ConstraintUnary,
    SysMLPackage,
    parse_package,
)
from cuas_sim.sysml_requirements import (
    RequirementsValidationError,
    build_traceability_matrix,
    check_requirements,
    evaluate_constraint,
    load_requirements,
    validate_against_requirements,
)

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"
_CONFIG = Path(__file__).parent.parent.parent / "configs" / "zida_shield_default.yaml"

#: The 14 bound requirements (Spec 021 §7.2).
_BOUND_IDS = {
    "TR-001", "TR-002", "TR-003", "TR-004", "TR-005", "TR-006",
    "TR-010", "TR-011", "TR-016", "TR-021", "TR-023", "TR-028", "TR-029", "TR-030",
}


def _by_id(reqs: object) -> dict[str, object]:
    return {r.req_id: r for r in reqs}  # type: ignore[attr-defined]


def _attr(rd: object, name: str) -> object:
    for a in rd.attributes:  # type: ignore[attr-defined]
        if a.name == name:
            return a
    raise AssertionError(f"attribute {name!r} not found")


def _default_cfg() -> MissionConfig:
    if not _SYSML_ROOT.exists() or not _CONFIG.exists():
        pytest.skip("sysml/ or default config not present")
    return MissionConfig.from_yaml_with_sysml(_CONFIG, sysml_root=_SYSML_ROOT)


# ─── T1 — parse requirement def + require constraint ─────────────────────────


def test_parse_requirement_def_and_constraint() -> None:
    text = (
        "package P {\n"
        "  requirement def <'TR-001'> FooReq {\n"
        '    doc /* a } brace and "quote" inside */\n'
        '    @Rationale { text = "why"; }\n'
        "    attribute actualEmploymentRange :> ISQ::length;\n"
        "    attribute maxEmploymentRange :> ISQ::length = 2 [km];\n"
        "    require constraint {\n"
        "      actualEmploymentRange <= maxEmploymentRange\n"
        "    }\n"
        "    #refinement dependency 'TR-001' to FunctionalRequirementsPackage::'FR-007';\n"
        "  }\n"
        "}\n"
    )
    pkg = parse_package(text)
    assert isinstance(pkg, SysMLPackage)
    assert pkg.part_defs == ()
    assert len(pkg.requirement_defs) == 1
    rd = pkg.requirement_defs[0]
    assert rd.short_name == "TR-001"
    assert rd.name == "FooReq"

    actual = _attr(rd, "actualEmploymentRange")
    assert actual.type_ref == "ISQ::length" and actual.value is None and actual.unit is None
    mx = _attr(rd, "maxEmploymentRange")
    assert mx.value == 2.0 and mx.unit == "km"

    assert len(rd.constraints) == 1
    assert rd.constraints[0].expr == ConstraintBinary(
        "<=", ConstraintRef("actualEmploymentRange"), ConstraintRef("maxEmploymentRange")
    )


# ─── T2 — expression precedence / parens / unary minus ───────────────────────


def test_parse_constraint_expression_precedence() -> None:
    text = (
        "package P {\n"
        "  requirement def <'TR-003'> R {\n"
        "    attribute actualUpdateInterval :> ISQ::time;\n"
        "    attribute minUpdateInterval :> ISQ::time = 5 [s];\n"
        "    attribute maxUpdateInterval :> ISQ::time = 60 [s];\n"
        "    require constraint {\n"
        "      (actualUpdateInterval >= minUpdateInterval) and\n"
        "      (actualUpdateInterval <= maxUpdateInterval)\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    rd = parse_package(text).requirement_defs[0]
    assert rd.constraints[0].expr == ConstraintBinary(
        "and",
        ConstraintBinary(">=", ConstraintRef("actualUpdateInterval"),
                         ConstraintRef("minUpdateInterval")),
        ConstraintBinary("<=", ConstraintRef("actualUpdateInterval"),
                         ConstraintRef("maxUpdateInterval")),
    )

    # Unary minus.
    text2 = (
        "package P {\n"
        "  requirement def <'TR-020'> R {\n"
        "    attribute actualHeadingVariation :> ISQ::planeAngle;\n"
        "    attribute maxHeadingVariation :> ISQ::planeAngle = 45 ['°'];\n"
        "    require constraint { actualHeadingVariation >= -maxHeadingVariation }\n"
        "  }\n"
        "}\n"
    )
    rd2 = parse_package(text2).requirement_defs[0]
    assert rd2.constraints[0].expr == ConstraintBinary(
        ">=", ConstraintRef("actualHeadingVariation"),
        ConstraintUnary("-", ConstraintRef("maxHeadingVariation")),
    )


# ─── T3 — evaluate_constraint known-answer ───────────────────────────────────


def test_evaluate_constraint_known_answer() -> None:
    le = ConstraintBinary("<=", ConstraintRef("a"), ConstraintRef("b"))
    assert evaluate_constraint(le, {"a": 2000.0, "b": 2000.0}) is True
    assert evaluate_constraint(le, {"a": 2500.0, "b": 2000.0}) is False

    eq = ConstraintBinary("==", ConstraintRef("a"), ConstraintRef("b"))
    assert evaluate_constraint(eq, {"a": 144.0, "b": 144.0}) is True
    assert evaluate_constraint(eq, {"a": 0.333333333, "b": 1 / 3}) is True  # isclose
    assert evaluate_constraint(eq, {"a": 144.0, "b": 130.0}) is False


# ─── T4 — load_requirements over the real sysml/ ─────────────────────────────


def test_load_requirements_real_sysml() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    reqs = _by_id(load_requirements(_SYSML_ROOT))
    tr_count = sum(1 for k in reqs if k.startswith("TR-"))
    assert tr_count >= 30

    tr001 = reqs["TR-001"]
    assert tr001.attributes["maxEmploymentRange"] == 2000.0  # 2 km -> SI
    assert tr001.constraint is not None

    tr030 = reqs["TR-030"]
    assert tr030.attributes["requiredZIDABoundaryDistance"] == 148160.0  # 80 nmi -> SI

    assert reqs["SHN-001"].constraint is None  # prose-only stakeholder need


# ─── T5 — validation PASSES on default config ────────────────────────────────


def test_validation_passes_on_default_config() -> None:
    cfg = _default_cfg()
    cfg.validate_against_requirements(sysml_root=_SYSML_ROOT)  # must not raise
    validate_against_requirements(cfg, _SYSML_ROOT)  # function form, must not raise

    report = check_requirements(cfg, load_requirements(_SYSML_ROOT))
    assert report.violated == []
    assert _BOUND_IDS.issubset(set(report.satisfied))


# ─── T6 — validation RAISES on violation ─────────────────────────────────────


def test_validation_raises_on_violation() -> None:
    cfg = _default_cfg()

    cfg_tr001 = cfg.replaced(baseline_envelope_m=(500.0, 2500.0))
    with pytest.raises(RequirementsValidationError) as exc1:
        cfg_tr001.validate_against_requirements(sysml_root=_SYSML_ROOT)
    assert "TR-001" in str(exc1.value)

    cfg_tr003 = cfg.replaced(baseline_update_interval_s=(5.0, 70.0))
    with pytest.raises(RequirementsValidationError) as exc3:
        cfg_tr003.validate_against_requirements(sysml_root=_SYSML_ROOT)
    assert "TR-003" in str(exc3.value)

    cfg_tr030 = cfg.replaced(zida_boundary_nmi=90.0)
    with pytest.raises(RequirementsValidationError) as exc30:
        cfg_tr030.validate_against_requirements(sysml_root=_SYSML_ROOT)
    assert "TR-030" in str(exc30.value)


# ─── T7 — traceability matrix ────────────────────────────────────────────────


def test_traceability_matrix() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    rows = {row.req_id: row for row in build_traceability_matrix(load_requirements(_SYSML_ROOT))}

    assert rows["TR-001"].bound is True
    assert "baseline_envelope_m" in rows["TR-001"].config_fields

    assert rows["TR-007"].bound is False
    assert rows["TR-007"].config_fields == "—"

    for rid in _BOUND_IDS:
        assert rows[rid].bound is True, rid
