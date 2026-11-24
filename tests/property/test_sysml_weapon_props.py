"""Property tests for the APKWS dimensionless whitelist (Spec 023 / SR-4).

P1 (whitelist closed) is a regression guard — it holds before and after the
implementation. P2 (AST↔regex symmetry) has teeth: it fails until both engines
surface the whitelisted attributes identically.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from cuas_sim.sysml_loader import (
    SysMLLoader,
    SysMLOverrides,
    _attrs_from_ast,
    _parse_numeric_attrs,
)
from cuas_sim.sysml_parser import parse_package

_WHITELIST = {
    "hitProbabilityAtMinRange",
    "dimensionalPenaltyFixedWing",
    "dimensionalPenaltyMultiRotor",
    "launcherRocketCapacity",
}

_SYNTH = """package TechnicalComponentsPackage {{
    part def APKWSLaserGuidanceKitWithSBAT70Rocket :> System {{
        attribute minRangeWithGuidance :> ISQ::length = 2 [km];
        attribute maxRangeWithGuidance :> ISQ::length = 14 [km];
        attribute hitProbabilityAtMinRange : ScalarValues::Real = {phit}.0;
        attribute dimensionalPenaltyFixedWing : ScalarValues::Real = 1.0;
        attribute dimensionalPenaltyMultiRotor : ScalarValues::Real = 0.5;
        attribute launcherRocketCapacity : ScalarValues::Integer = {cap};
        attribute firingDistanceShortPhilosophy :> ISQ::length = 2 [km];
        attribute firingDistanceLongPhilosophy :> ISQ::length = 14 [km];
    }}
}}
"""


@given(name=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=14))
def test_whitelist_does_not_leak(name: str) -> None:
    if name in _WHITELIST:
        return
    body = f"\tattribute {name} : ScalarValues::Real = 3.0;\n"
    text = "package P {\n\tpart def Foo :> System {\n" + body + "\t}\n}\n"
    pd = parse_package(text).part_defs[0]
    ast_attrs, _strs, _acuity = _attrs_from_ast(pd)
    rx_attrs = _parse_numeric_attrs(body)
    # A unit-less attribute whose name is not whitelisted must be invisible to
    # BOTH engines (Q-1 preserved), and both engines agree.
    assert name not in ast_attrs
    assert name not in rx_attrs
    assert ast_attrs == rx_attrs


@given(
    cap=st.integers(min_value=1, max_value=200),
    phit=st.integers(min_value=0, max_value=100),
)
def test_ast_regex_symmetry_whitelist(cap: int, phit: int) -> None:
    text = _SYNTH.format(phit=phit, cap=cap)
    loader = SysMLLoader.__new__(SysMLLoader)  # no filesystem root needed for dispatch

    ov_ast = SysMLOverrides()
    for pd in parse_package(text).part_defs:
        attrs, strs, acuity = _attrs_from_ast(pd)
        loader._dispatch_part_def(pd.name, attrs, strs, acuity, ov_ast, "synthetic")

    ov_rx = SysMLOverrides()
    loader._legacy_regex_extract(text, ov_rx, "synthetic")

    # Both engines produce a structurally identical SysMLOverrides (incl. the
    # provenance order) — the bit-identity contract under the whitelist.
    assert ov_ast == ov_rx
    assert ov_ast.modernized_rocket_budget == cap
    assert ov_ast.modernized_p_hit_max == phit / 100.0
