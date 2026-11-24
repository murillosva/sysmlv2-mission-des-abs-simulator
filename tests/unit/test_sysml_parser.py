"""Unit tests for the SysML v2 AST parser (Spec 020).

Group B (Red phase): these MUST FAIL until ``sysml_parser.parse_package`` is
implemented (Spec 020 Group C).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cuas_sim.sysml_parser import (
    SysMLAttribute,
    SysMLPackage,
    SysMLParseError,
    parse_package,
)

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"
_TECH = _SYSML_ROOT / "Technical" / "TechnicalComponentsPackage.sysml"


def _attr(pd: object, name: str) -> SysMLAttribute:
    """Return the attribute named ``name`` from a SysMLPartDef."""
    for a in pd.attributes:  # type: ignore[attr-defined]
        if a.name == name:
            return a
    raise AssertionError(f"attribute {name!r} not found")


def test_parse_part_def_mixed_attributes() -> None:
    text = (
        "package P {\n"
        "  part def Foo :> System {\n"
        '    doc /* contains a } brace and a "quote" */\n'
        "    attribute cruiseSpeed :> ISQ::speed = 144 [m/s];\n"
        "    attribute capacity : ScalarValues::Integer = 49;\n"
        "    attribute pTrack : ScalarValues::Real = 0.2;\n"
        '    attribute lat : ScalarValues::String = "7°35\'59.86\\"S";\n'
        "    attribute ang :> ISQ::planeAngle = 1.0 ['arcmin'];\n"
        "    port p : SomePort;\n"
        "  }\n"
        "}\n"
    )
    pkg = parse_package(text)
    assert isinstance(pkg, SysMLPackage)
    assert pkg.name == "P"
    assert len(pkg.part_defs) == 1  # the } inside doc did NOT close the part def
    pd = pkg.part_defs[0]
    assert pd.name == "Foo"
    assert pd.specializes == ("System",)

    # Attribute order preserved; doc + port skipped.
    assert tuple(a.name for a in pd.attributes) == (
        "cruiseSpeed", "capacity", "pTrack", "lat", "ang",
    )

    cs = _attr(pd, "cruiseSpeed")
    assert cs.type_ref == "ISQ::speed" and cs.value == 144.0 and cs.unit == "m/s"

    cap = _attr(pd, "capacity")
    assert cap.type_ref == "ScalarValues::Integer" and cap.value == 49 and cap.unit is None

    pt = _attr(pd, "pTrack")
    assert pt.type_ref == "ScalarValues::Real" and pt.value == 0.2 and pt.unit is None

    lat = _attr(pd, "lat")
    # Escaped quote resolved: \" -> "
    assert lat.value == "7°35'59.86\"S" and lat.unit is None

    ang = _attr(pd, "ang")
    assert ang.value == 1.0 and ang.unit == "arcmin"


def test_literal_robustness() -> None:
    text = (
        "package P {\n"
        "  part def F :> A, B {\n"
        "    attribute neg :> ISQ::length = -61 [m];\n"
        "    attribute sci1 :> ISQ::frequency = 2.0E9 [Hz];\n"
        "    attribute sci2 :> ISQ::frequency = 900.0E6 [Hz];\n"
        "    attribute slew : ScalarValues::Real = 60 ['°'/s];\n"
        "    attribute flag : ScalarValues::Boolean = true;\n"
        "  }\n"
        "}\n"
    )
    pkg = parse_package(text)
    pd = pkg.part_defs[0]
    assert pd.specializes == ("A", "B")
    assert _attr(pd, "neg").value == -61.0 and _attr(pd, "neg").unit == "m"
    assert _attr(pd, "sci1").value == 2.0e9 and _attr(pd, "sci1").unit == "Hz"
    assert _attr(pd, "sci2").value == 900.0e6
    slew = _attr(pd, "slew")
    assert slew.value == 60 and slew.unit == "'°'/s"  # not fully quote-wrapped → raw
    assert _attr(pd, "flag").value is True and _attr(pd, "flag").unit is None


def test_empty_and_import_only_package() -> None:
    pkg = parse_package("package Q {\n  private import SI::*;\n}\n")
    assert pkg.name == "Q" and pkg.part_defs == ()


def test_unbalanced_braces_raises() -> None:
    with pytest.raises(SysMLParseError):
        parse_package("package P {\n  part def F {\n    attribute x = 1 [m];\n")


def test_attribute_type_stability() -> None:
    if not _TECH.exists():
        pytest.skip("sysml/ not present")
    pkg = parse_package(_TECH.read_text(encoding="utf-8"))
    seen = 0
    for pd in pkg.part_defs:
        for a in pd.attributes:
            seen += 1
            if a.type_ref == "ScalarValues::String":
                assert isinstance(a.value, str) and a.unit is None
            elif a.type_ref == "ScalarValues::Integer":
                assert isinstance(a.value, int) and not isinstance(a.value, bool)
                assert a.unit is None
            elif a.type_ref == "ScalarValues::Boolean":
                assert isinstance(a.value, bool)
            # Numeric attributes (ISQ::*) with a bracketed unit carry unit != None;
            # bare ScalarValues::Real have no unit.
            if a.unit is not None:
                assert isinstance(a.value, (int, float)) and not isinstance(a.value, bool)
    assert seen > 50  # the real package has many attributes
