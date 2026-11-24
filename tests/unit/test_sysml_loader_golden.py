"""Golden / bit-identical tests for the AST-backed SysML loader (Spec 020).

Group B (Red phase): these MUST FAIL until the loader is refactored to the AST
engine and the legacy regex oracle ``_legacy_regex_load`` is preserved
(Spec 020 Group C, FR-4/FR-6/TD-4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cuas_sim.sysml_loader import SysMLLoader, _legacy_regex_load

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"

# Golden snapshot captured from the regex engine at checkpoint-2026-06-16
# (Spec 020 §5.1 T2). The AST engine must reproduce these exactly.
#
# Spec 023 (SR-4) re-baseline: the APKWS weapon model is now SysML-sourced, so
# ``modernized_envelope_m`` is populated and five new weapon fields appear; the
# baseline rocket capacity also surfaces (per-config). This is a *deliberate*
# re-baseline (metric-neutral by value — every value equals the config default).
# Provenance grows 11 → 17 (5 from APKWS + 1 from the SBAT-70 capacity).
_GOLDEN: dict[str, object] = {
    "a29_cruise_speed_mps": 144.0,
    "a29_endurance_s": 18720.0,
    "uas_mr_max_speed_mps": 29.0,
    "uas_mr_dimension_m": 1.0,
    "uas_mr_ceiling_m": 6000.0,
    "uas_fw_max_speed_mps": 130.0,
    "uas_fw_dimension_m": 14.0,
    "uas_fw_ceiling_m": 18000.0,
    "baseline_update_interval_s": (5.0, 60.0),
    "baseline_max_acquisition_m": 13000.0,
    "baseline_visual_acuity_rad": 0.0002908882086657216,
    "modernized_update_interval_s": (1.0, 10.0),
    "modernized_eoir_max_m": 20000.0,
    "modernized_eoir_accuracy_m": 5.0,
    "modernized_track_p_fw": None,  # FLIR still out of scope (Spec 023 FR-7)
    "modernized_track_p_mr": None,  # FLIR still out of scope (Spec 023 FR-7)
    "baseline_envelope_m": (500.0, 2000.0),
    # Spec 023 (SR-4) — APKWS weapon model now SysML-sourced.
    "modernized_envelope_m": (2000.0, 14000.0),
    "modernized_p_hit_max": 0.80,           # 80.0 percent → 0.80 fraction
    "dimensional_penalty_fw": 1.0,
    "dimensional_penalty_mr": 0.5,
    "modernized_rocket_budget": 49,         # from APKWS launcherRocketCapacity
    "baseline_rocket_budget": 49,           # from SBAT-70 launcherRocketCapacity
    "main_base": None,  # Q-2 deferred — geography stays default-sourced
}


def test_loader_known_answer() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    ov = SysMLLoader(_SYSML_ROOT).load()
    for field_name, expected in _GOLDEN.items():
        assert getattr(ov, field_name) == expected, field_name
    assert ov.farps == {}  # Q-2 deferred
    assert len(ov.provenance) == 17  # Spec 023 SR-4 re-baseline (was 11)


def test_ast_vs_regex_bit_identical() -> None:
    if not _SYSML_ROOT.exists():
        pytest.skip("sysml/ not present")
    ast_ov = SysMLLoader(_SYSML_ROOT).load()
    regex_ov = _legacy_regex_load(_SYSML_ROOT)
    # Full structural equality, including farps, main_base and provenance ORDER.
    assert ast_ov == regex_ov
    assert ast_ov.provenance == regex_ov.provenance


def test_malformed_model_warns_no_raise(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tech_dir = tmp_path / "Technical"
    tech_dir.mkdir(parents=True)
    # Unbalanced braces — structurally broken.
    (tech_dir / "TechnicalComponentsPackage.sysml").write_text(
        "package TechnicalComponentsPackage {\n"
        "  part def Broken :> System {\n"
        "    attribute cruiseSpeed :> ISQ::speed = 144 [m/s];\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        ov = SysMLLoader(tmp_path).load()  # must NOT raise (INV-1)
    assert ov is not None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "parse" in caplog.text.lower()
