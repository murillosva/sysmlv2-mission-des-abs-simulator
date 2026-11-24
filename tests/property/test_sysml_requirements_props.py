"""Property-based tests for SysML requirements validation (Spec 021 SR-2).

Group B (Red phase): MUST FAIL until Spec 021 Group C is implemented.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cuas_sim.config import MissionConfig
from cuas_sim.sysml_parser import parse_package
from cuas_sim.sysml_requirements import (
    RequirementsValidationError,
    check_requirements,
    load_requirements,
    validate_against_requirements,
)

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"
_CONFIG = Path(__file__).parent.parent.parent / "configs" / "zida_shield_default.yaml"

_REQ_TEXT = (
    "package P {\n"
    "  requirement def <'TR-001'> R {\n"
    "    attribute actualEmploymentRange :> ISQ::length;\n"
    "    attribute maxEmploymentRange :> ISQ::length = 2 [km];\n"
    "    require constraint { actualEmploymentRange <= maxEmploymentRange }\n"
    "  }\n"
    "}\n"
)


def _default_cfg() -> MissionConfig:
    if not _SYSML_ROOT.exists() or not _CONFIG.exists():
        pytest.skip("sysml/ or default config not present")
    return MissionConfig.from_yaml_with_sysml(_CONFIG, sysml_root=_SYSML_ROOT)


@given(ws=st.integers(min_value=0, max_value=8))
@settings(max_examples=200)
def test_parse_determinism(ws: int) -> None:
    spaced = _REQ_TEXT.replace(" <= ", " <= " + " " * ws + "// c\n      ")
    a = parse_package(_REQ_TEXT)
    b = parse_package(_REQ_TEXT)
    assert a == b  # determinism
    # Reformatting (extra whitespace + a line comment) preserves the AST.
    assert parse_package(spaced).requirement_defs[0].constraints[0].expr == \
        a.requirement_defs[0].constraints[0].expr


@given(
    base_max=st.floats(min_value=500.0, max_value=4000.0),
    upd_max=st.floats(min_value=20.0, max_value=120.0),
)
@settings(max_examples=200, deadline=None)
def test_check_partitions_bindings(base_max: float, upd_max: float) -> None:
    cfg = _default_cfg().replaced(
        baseline_envelope_m=(500.0, base_max),
        baseline_update_interval_s=(5.0, upd_max),
    )
    reqs = load_requirements(_SYSML_ROOT)
    report = check_requirements(cfg, reqs)  # pure, never raises
    classified = set(report.satisfied) | set(report.violated)
    # No binding may be both satisfied and violated.
    assert set(report.satisfied).isdisjoint(set(report.violated))
    # The bound TR-001 / TR-003 are always classified (never dropped).
    for rid in ("TR-001", "TR-003"):
        assert rid in classified


@given(base_max=st.floats(min_value=500.0, max_value=4000.0))
@settings(max_examples=200, deadline=None)
def test_validate_iff_check_violated(base_max: float) -> None:
    cfg = _default_cfg().replaced(baseline_envelope_m=(500.0, base_max))
    reqs = load_requirements(_SYSML_ROOT)
    has_violation = check_requirements(cfg, reqs).violated != []
    raised = False
    try:
        validate_against_requirements(cfg, _SYSML_ROOT)
    except RequirementsValidationError:
        raised = True
    assert raised == has_violation
