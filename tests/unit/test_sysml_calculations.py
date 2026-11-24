"""Known-answer tests for SR-4b calc-def ↔ metrics guard (Spec 024).

Group B (Red phase, Article III): these MUST FAIL until Group C lands the parser
extension (``calc def``), the arithmetic expression evaluator, the adjudication
table (``_ADJUDICATIONS``), the guard (``check_calculations``) and the opt-in
validation (``validate_calculations``).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from cuas_sim.sysml_calculations import (
    _ADJUDICATIONS,
    CalculationsValidationError,
    build_adjudication_matrix,
    check_calculations,
    evaluate_expr,
    load_calculations,
    validate_calculations,
)
from cuas_sim.sysml_parser import parse_package

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"
_CALC_PKG = _SYSML_ROOT / "Analysis" / "CalculationsPackage.sysml"

_CALC_NAMES_IN_ORDER = (
    "calculateNeutralizationRate",
    "calculateEvasionReductionRate",
    "calculateEngagementTimeRate",
    "calculateTrackingRate",
    "calculateNeutralizationProbability",
    "calculateUpdateRate",
    "calculateAcquisitionRange",
    "calculateWeaponEnvelope",
)


def _calc_defs_by_name() -> dict[str, object]:
    pkg = parse_package(_CALC_PKG.read_text(encoding="utf-8"))
    return {c.name: c for c in pkg.calc_defs}


# ─── T1: parser reads the 8 calc defs ───────────────────────────────────────


def test_parser_reads_8_calc_defs() -> None:
    pkg = parse_package(_CALC_PKG.read_text(encoding="utf-8"))
    names = tuple(c.name for c in pkg.calc_defs)
    assert names == _CALC_NAMES_IN_ORDER
    nr = {c.name: c for c in pkg.calc_defs}["calculateNeutralizationRate"]
    input_names = {i.name for i in nr.inputs}
    assert {"Nn", "Nt", "Config"} <= input_names
    assert nr.return_name == "NeutralizationRate"
    assert nr.expr is not None  # Nn / Nt


# ─── T2: arithmetic evaluation of the parsed return expression ──────────────


def test_arithmetic_expr_eval() -> None:
    by_name = _calc_defs_by_name()
    nr = by_name["calculateNeutralizationRate"].expr  # type: ignore[attr-defined]
    assert evaluate_expr(nr, {"Nn": 7, "Nt": 10}) == pytest.approx(0.7)
    er = by_name["calculateEvasionReductionRate"].expr  # type: ignore[attr-defined]
    # (rB - rM) / rM with rB=0.5, rM=0.2 -> 0.3 / 0.2 = 1.5 (parens + precedence)
    assert evaluate_expr(er, {"rB": 0.5, "rM": 0.2}) == pytest.approx(1.5)
    we = by_name["calculateWeaponEnvelope"].expr  # type: ignore[attr-defined]
    assert evaluate_expr(we, {"maxRange": 14000.0, "minRange": 2000.0}) == pytest.approx(12000.0)


# ─── T3: CONSISTENT guard with teeth (mutation breaks it) ───────────────────


def test_consistent_guard_has_teeth() -> None:
    calcs = load_calculations(_SYSML_ROOT)
    report = check_calculations(calcs, _ADJUDICATIONS)
    assert "calculateNeutralizationRate" in report.consistent  # MOS-1
    assert report.failed_guards == []

    # Mutation: simulate a Python aggregator drift on MOS-1's reference value.
    adj = _ADJUDICATIONS["calculateNeutralizationRate"]
    assert adj.aggregator_eval is not None
    mutated = dict(_ADJUDICATIONS)
    mutated["calculateNeutralizationRate"] = replace(
        adj, aggregator_eval=lambda: adj.aggregator_eval() + 0.5  # type: ignore[misc]
    )
    drift = check_calculations(calcs, mutated)
    assert "calculateNeutralizationRate" in drift.failed_guards
    with pytest.raises(CalculationsValidationError):
        validate_calculations(_SYSML_ROOT, adjudications=mutated)


# ─── T4: divergent verdicts never fail the guard ────────────────────────────


def test_divergent_verdicts_dont_fail_guard() -> None:
    calcs = load_calculations(_SYSML_ROOT)
    report = check_calculations(calcs, _ADJUDICATIONS)
    assert "calculateEvasionReductionRate" in report.accepted  # MOS-2
    assert "calculateWeaponEnvelope" in report.accepted  # MOP-3
    assert "calculateUpdateRate" in report.mapping_fixed  # MOP-1 -> update_interval_s
    assert report.failed_guards == []
    validate_calculations(_SYSML_ROOT)  # does not raise on the real table


# ─── T5: MOE-1 reconciled (Spec 025) — now CONSISTENT with teeth, 0 ESCALATE ─


def test_moe1_now_consistent_with_teeth() -> None:
    # Spec 025 (Article VI §9) closed the SR-4b ESCALATE backlog: the Python
    # _compute_moe1 was reconciled with Eq. (3.5), so calculateEngagementTimeRate
    # flips DIVERGENCE-ESCALATE -> CONSISTENT and the numeric guard gets teeth.
    calcs = load_calculations(_SYSML_ROOT)
    report = check_calculations(calcs, _ADJUDICATIONS)
    assert "calculateEngagementTimeRate" in report.consistent  # MOE-1
    assert report.escalated == []                              # no more backlog
    assert "calculateEngagementTimeRate" not in report.failed_guards
    validate_calculations(_SYSML_ROOT)  # real table -> no raise

    # Teeth: a drift in the MOE-1 aggregator reference fails the guard.
    adj = _ADJUDICATIONS["calculateEngagementTimeRate"]
    assert adj.aggregator_eval is not None
    mutated = dict(_ADJUDICATIONS)
    mutated["calculateEngagementTimeRate"] = replace(
        adj, aggregator_eval=lambda: adj.aggregator_eval() + 0.5  # type: ignore[misc]
    )
    drift = check_calculations(calcs, mutated)
    assert "calculateEngagementTimeRate" in drift.failed_guards
    with pytest.raises(CalculationsValidationError):
        validate_calculations(_SYSML_ROOT, adjudications=mutated)


# ─── T6: leniency — missing / broken package never raises ───────────────────


def test_lenient_missing_or_broken(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert load_calculations(tmp_path) == []  # no Analysis/CalculationsPackage.sysml
    analysis = tmp_path / "Analysis"
    analysis.mkdir()
    (analysis / "CalculationsPackage.sysml").write_text(
        "package X { calc def Bad {", encoding="utf-8"
    )  # unbalanced braces
    with caplog.at_level(logging.WARNING):
        result = load_calculations(tmp_path)
    assert result == []  # broken file skipped, not raised
    report = check_calculations([], _ADJUDICATIONS)
    assert report.failed_guards == []  # vacuously OK


# ─── T7: adjudication matrix is complete (8 rows, verdict counts) ───────────


def test_adjudication_matrix_complete() -> None:
    # Spec 025 reconciled MOE-1 -> CONSISTENT: 4/2/1/1 became 5/2/1/0.
    rows = build_adjudication_matrix()
    assert len(rows) == 8
    assert len({r.calc_name for r in rows}) == 8
    counts = Counter(r.verdict for r in rows)
    assert counts == Counter({
        "CONSISTENT": 5,
        "DIVERGENCE-ACCEPTED": 2,
        "MAPPING-FIX": 1,
        "DIVERGENCE-ESCALATE": 0,
    })


# ─── FR-11: validate raises only on a CONSISTENT guard failure ──────────────


def test_validate_raises_only_on_consistent_guard_fail() -> None:
    validate_calculations(_SYSML_ROOT)  # real table -> no raise
    mutated = dict(_ADJUDICATIONS)
    adj = mutated["calculateTrackingRate"]  # MOE-2, CONSISTENT
    mutated["calculateTrackingRate"] = replace(adj, aggregator_eval=lambda: -1.0)
    with pytest.raises(CalculationsValidationError):
        validate_calculations(_SYSML_ROOT, adjudications=mutated)
