"""Minimal CLI smoke covering Spec 018 FR-16 window summary in cli.py.


Pre-existing context: ``src/cuas_sim/cli.py`` was 0% covered before Spec 018
(see ``memory/project_type_lint_debt_followup.md``). This file adds focused
coverage for the Spec 018 additions (the window-summary print block) without
inflating the module — heavy commands (run/sweep/loop_run/loop_step) remain
exercised end-to-end by ``tests/smoke/`` and by D-4 of ``tasks.md``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from cuas_sim import cli as cli_module
from cuas_sim.cli import app
from cuas_sim.config import MissionConfig
from cuas_sim.metrics.records import RunLogger
from cuas_sim.metrics.replanning import ReplanningSignal
from cuas_sim.simulation.run import PairedResult

runner = CliRunner()


def test_cli_info_command() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "cuas-sim" in result.stdout


def test_cli_validate_command() -> None:
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0


def test_cli_animate_missing_dir() -> None:
    result = runner.invoke(app, ["animate", str(Path("/__nope__"))])
    assert result.exit_code == 1


def test_cli_check_border_missing_config() -> None:
    result = runner.invoke(
        app, ["check-border", "--config", str(Path("/__nope__.yaml"))],
    )
    assert result.exit_code == 1


def test_cli_loop_step_state_missing() -> None:
    result = runner.invoke(
        app, ["loop", "step", "--state", str(Path("/__nope__.json"))],
    )
    assert result.exit_code == 1


def test_cli_loop_step_no_args() -> None:
    result = runner.invoke(app, ["loop", "step"])
    assert result.exit_code == 1


def test_cli_loop_run_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["loop", "run", "--config", str(tmp_path / "nope.yaml"),
         "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 1


def test_cli_check_requirements_default_passes() -> None:
    if not Path("sysml").exists():
        return  # sysml/ not present — skip
    result = runner.invoke(
        app, ["check-requirements", "--config", "configs/zida_shield_default.yaml",
              "--no-matrix"],
    )
    assert result.exit_code == 0, result.stdout
    assert "All 14 bound requirements satisfied" in result.stdout


def test_cli_check_requirements_violation_exits_1(tmp_path: Path) -> None:
    if not Path("sysml").exists():
        return
    # zida_boundary_nmi is NOT a SysML-provided override (golden = None), so the YAML
    # value wins; 90 nmi != 80 nmi violates TR-030.
    bad = tmp_path / "bad.yaml"
    bad.write_text("zida_boundary_nmi: 90.0\n", encoding="utf-8")
    result = runner.invoke(app, ["check-requirements", "--config", str(bad), "--no-matrix"])
    assert result.exit_code == 1
    assert "TR-030" in result.stdout


def _build_fake_paired_result(window_payload: dict[str, object] | None) -> PairedResult:
    cfg = MissionConfig.from_yaml(Path("configs/zida_shield_default.yaml"))
    log = RunLogger()
    if window_payload is not None:
        log.set_window(window_payload)
    metrics = pd.DataFrame([
        {"config_name": "baseline", "aor": "ALL", "metric": "MOS-1", "value": 0.42},
        {"config_name": "modernized", "aor": "ALL", "metric": "MOS-1", "value": 0.81},
    ])
    signal = ReplanningSignal(
        triggered=False,
        mos1_modernized=0.81,
        mos1_target=0.80,
        leaks_per_aor={},
        leaks_per_sector={},
        leak_share_per_aor={},
        hint="",
    )
    return PairedResult(
        cfg=cfg, seed=0, log=log, metrics=metrics,
        replanning_signal=signal, total_arrivals=10,
    )


def test_cli_run_prints_window_summary(monkeypatch, tmp_path: Path) -> None:
    window_payload = {
        "field_version": "1.0",
        "uas_spawn_window_enabled": True,
        "t_window_start_s": 9180.0,
        "t_window_end_s": 24512.4,
        "t_window_end_reason": "all_first_bingo",
        "horizon_cap_s": 40000.0,
        "baseline_first_bingo": {"a29_id": "A-29-1", "t_s": 24500.0},
        "modernized_first_bingo": {"a29_id": "A-29-4", "t_s": 24512.4},
        "t_rtb_fuel_limit_per_a29": {"baseline": {}, "modernized": {}},
    }
    fake = _build_fake_paired_result(window_payload)

    def _fake_run_paired(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake

    monkeypatch.setattr(cli_module, "run_paired", _fake_run_paired)
    result = runner.invoke(
        app, ["run", "--config", "configs/zida_shield_default.yaml",
              "--outputs", str(tmp_path), "--seed", "0"],
    )
    assert result.exit_code == 0, result.stdout
    assert "Window: [9180.0 -> 24512.4]" in result.stdout
    assert "reason: all_first_bingo" in result.stdout
    assert "baseline   first bingo: A-29-1 @ 24500.0 s" in result.stdout
    assert "modernized first bingo: A-29-4 @ 24512.4 s" in result.stdout


def test_cli_run_window_summary_handles_none_bingo(monkeypatch, tmp_path: Path) -> None:
    window_payload = {
        "field_version": "1.0",
        "uas_spawn_window_enabled": True,
        "t_window_start_s": 9180.0,
        "t_window_end_s": 40000.0,
        "t_window_end_reason": "horizon_cap",
        "horizon_cap_s": 40000.0,
        "baseline_first_bingo": None,
        "modernized_first_bingo": None,
        "t_rtb_fuel_limit_per_a29": {"baseline": {}, "modernized": {}},
    }
    fake = _build_fake_paired_result(window_payload)
    monkeypatch.setattr(cli_module, "run_paired", lambda *a, **k: fake)
    result = runner.invoke(
        app, ["run", "--config", "configs/zida_shield_default.yaml",
              "--outputs", str(tmp_path), "--seed", "0", "--skip-pre-pea"],
    )
    assert result.exit_code == 0, result.stdout
    assert "(none -- horizon_cap)" in result.stdout
