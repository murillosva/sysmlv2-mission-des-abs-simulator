"""End-to-end conformance over a real paired run (Spec 022 §5.3 / AC-3).

Group B (Red phase, Article III): MUST FAIL until Group C is implemented.

Runs ``run_paired`` with real SimPy (no mocks, Article VIII), persists the run,
then asserts the persisted ``events_<config>.jsonl.gz`` streams conform to the
SysML phase oracle for BOTH configs and BOTH Spec 018 flag paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.simulation.run import run_paired
from cuas_sim.sysml_state_machine import (
    check_conformance,
    load_events,
    load_phases,
    validate_conformance,
)

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("uas_spawn_window_enabled", [True, False])
def test_paired_run_is_conformant(tmp_path: Path, uas_spawn_window_enabled: bool) -> None:
    cfg = MissionConfig(uas_spawn_window_enabled=uas_spawn_window_enabled)
    run_paired(cfg, seed=42, outputs_root=tmp_path, max_arrivals=20, persist=True)

    run_dir = next((tmp_path / "runs").iterdir())

    # The opt-in validation port must not raise on a real run.
    validate_conformance(run_dir, _SYSML_ROOT)

    # Both configs explicitly conformant, with real combat extensions present.
    model = load_phases(_SYSML_ROOT)
    assert model.oracle_loaded is True
    events = load_events(run_dir)
    for config in ("baseline", "modernized"):
        report = check_conformance(events[config], model)
        assert report.conformant is True, (config, report.violations)
