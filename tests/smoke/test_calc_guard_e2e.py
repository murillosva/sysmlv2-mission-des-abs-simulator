"""End-to-end SR-4b calc-def guard over the real SysML + aggregator (Spec 024 §6).

Group B (Red phase, Article III): MUST FAIL until Group C is implemented.

Read-only integration: parses the real ``CalculationsPackage`` and runs the
adjudication guard against the **real** ``metrics/aggregator.py`` — no SimPy, no
RNG, no run (SR-4b is read-only / metric-neutral, distinct from the SimPy smokes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cuas_sim.sysml_calculations import (
    _ADJUDICATIONS,
    build_adjudication_matrix,
    check_calculations,
    load_calculations,
    validate_calculations,
)

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"

pytestmark = pytest.mark.smoke

_CONSISTENT = (
    "calculateNeutralizationRate",        # MOS-1
    "calculateTrackingRate",              # MOE-2
    "calculateNeutralizationProbability",  # MOE-3
    "calculateAcquisitionRange",          # MOP-2
)


def test_real_pipeline_consistent_guards_pass() -> None:
    calcs = load_calculations(_SYSML_ROOT)
    assert len(calcs) == 8
    report = check_calculations(calcs, _ADJUDICATIONS)
    for name in _CONSISTENT:
        assert name in report.consistent, name
    assert report.failed_guards == []
    assert len(build_adjudication_matrix()) == 8
    validate_calculations(_SYSML_ROOT)  # real pipeline does not raise
