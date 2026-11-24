"""Known-answer tests for the APKWS weapon extraction (Spec 023 / SR-4).

Red phase (Article III): these MUST fail until ``sysml_loader`` gains the APKWS
branch, the SBAT-70 baseline-capacity extension, the dimensionless whitelist and
the new ``SysMLOverrides`` fields, and ``MissionConfig`` gains the per-config
rocket budgets (``rocket_budget_for``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.sysml_loader import SysMLLoader, _attrs_from_ast
from cuas_sim.sysml_parser import parse_package

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"
_TECH = _SYSML_ROOT / "Technical" / "TechnicalComponentsPackage.sysml"

#: Minimal Technical package carrying only the APKWS part def — used to exercise
#: the dispatch in isolation (T4 percent→fraction).
_SYNTH_APKWS = """package TechnicalComponentsPackage {{
    part def APKWSLaserGuidanceKitWithSBAT70Rocket :> System {{
        attribute minRangeWithGuidance :> ISQ::length = 2 [km];
        attribute maxRangeWithGuidance :> ISQ::length = 14 [km];
        attribute hitProbabilityAtMinRange : ScalarValues::Real = {phit};
        attribute dimensionalPenaltyFixedWing : ScalarValues::Real = 1.0;
        attribute dimensionalPenaltyMultiRotor : ScalarValues::Real = 0.5;
        attribute launcherRocketCapacity : ScalarValues::Integer = {cap};
        attribute firingDistanceShortPhilosophy :> ISQ::length = 2 [km];
        attribute firingDistanceLongPhilosophy :> ISQ::length = 14 [km];
    }}
}}
"""


def _part_def(name: str):
    pkg = parse_package(_TECH.read_text(encoding="utf-8"))
    return next(pd for pd in pkg.part_defs if pd.name == name)


def test_apkws_envelope() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    ov = SysMLLoader(_SYSML_ROOT).load()
    assert ov.modernized_envelope_m == (2000.0, 14000.0)
    assert any("modernized_envelope_m" in path and "APKWS" in path
               for _, path in ov.provenance)


def test_apkws_coeffs_and_per_config_capacity() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    ov = SysMLLoader(_SYSML_ROOT).load()
    assert ov.modernized_p_hit_max == 0.80
    assert ov.dimensional_penalty_fw == 1.0
    assert ov.dimensional_penalty_mr == 0.5
    assert ov.modernized_rocket_budget == 49   # from APKWS
    assert ov.baseline_rocket_budget == 49     # from SBAT-70


def test_rocket_budget_for_per_config() -> None:
    cfg = MissionConfig(baseline_rocket_budget=10, modernized_rocket_budget=20)
    assert cfg.rocket_budget_for("baseline") == 10
    assert cfg.rocket_budget_for("modernized") == 20
    # Fallback to the shared default when the per-config field is None.
    cfg2 = MissionConfig(rocket_budget_per_sortie=33)
    assert cfg2.rocket_budget_for("baseline") == 33
    assert cfg2.rocket_budget_for("modernized") == 33


def test_apply_to_value_idempotent() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    cfg = MissionConfig()
    ov = SysMLLoader(_SYSML_ROOT).load()
    cfg2 = ov.apply_to(cfg)
    # Weapon parameters land on their config fields but the resolved values are
    # the pre-SR-4 defaults (metric-neutral by value).
    assert cfg2.modernized_envelope_m == cfg.modernized_envelope_m == (2000.0, 14000.0)
    assert cfg2.p_hit_at_min_range == cfg.p_hit_at_min_range == 0.80
    assert cfg2.dimensional_penalty_fw == cfg.dimensional_penalty_fw == 1.0
    assert cfg2.dimensional_penalty_mr == cfg.dimensional_penalty_mr == 0.5
    assert cfg.rocket_budget_for("modernized") == cfg2.rocket_budget_for("modernized") == 49
    assert cfg.rocket_budget_for("baseline") == cfg2.rocket_budget_for("baseline") == 49


def test_p_hit_percent_to_fraction(tmp_path: Path) -> None:
    tech = tmp_path / "Technical"
    tech.mkdir()
    (tech / "TechnicalComponentsPackage.sysml").write_text(
        _SYNTH_APKWS.format(phit="60.0", cap="49"), encoding="utf-8"
    )
    ov = SysMLLoader(tmp_path).load()
    assert ov.modernized_p_hit_max == 0.60  # 60.0 percent → 0.60 fraction


def test_whitelist_surfaces_apkws_dimensionless() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    apkws = _part_def("APKWSLaserGuidanceKitWithSBAT70Rocket")
    attrs, _strs, _acuity = _attrs_from_ast(apkws)
    for name in (
        "hitProbabilityAtMinRange",
        "dimensionalPenaltyFixedWing",
        "dimensionalPenaltyMultiRotor",
        "launcherRocketCapacity",
    ):
        assert name in attrs, name


def test_whitelist_closed_flir_not_surfaced() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    flir = _part_def("FLIRStarSAFIREIIEOIRSensor")
    attrs, _strs, _acuity = _attrs_from_ast(flir)
    # The FLIR dimensionless reals are NOT whitelisted (FR-7) — they stay out.
    # (slewRate/azimuthCoverage carry units and are legitimately surfaced by the
    # pre-existing unit rule — they are not the whitelist's concern.)
    assert "trackingProbabilityFixedWing" not in attrs
    assert "trackingProbabilityMultiRotor" not in attrs
    ov = SysMLLoader(_SYSML_ROOT).load()
    assert ov.modernized_track_p_fw is None
    assert ov.modernized_track_p_mr is None


def test_malformed_no_raise(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tech = tmp_path / "Technical"
    tech.mkdir()
    # Unbalanced braces — structurally broken.
    (tech / "TechnicalComponentsPackage.sysml").write_text(
        "package TechnicalComponentsPackage {\n"
        "  part def APKWSLaserGuidanceKitWithSBAT70Rocket :> System {\n"
        "    attribute hitProbabilityAtMinRange : ScalarValues::Real = 80.0;\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        ov = SysMLLoader(tmp_path).load()  # must NOT raise (INV-1)
    assert ov is not None
