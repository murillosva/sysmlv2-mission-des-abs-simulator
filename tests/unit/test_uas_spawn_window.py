"""Unit tests for Spec 018 — UAS spawn window & unified RTB state.

Covers Known-Answer Tests T1–T8 from `spec.md` §5.1:
- T1: flag default is True (FR-1).
- T2: flag=False requires phase4_pea_s > 0 (FR-2).
- T3: A29State.RTB_FUEL_LIMIT enum member exists (FR-7).
- T4: first_bingo_event.succeed is idempotent under `if not event.triggered` (FR-10).
- T5: _t_pea_for_aor returns emergent T_PEA when RTB observed (FR-11).
- T6: _t_pea_for_aor falls back to Haversine when RTB not registered (FR-12).
- T7: _t_pea_for_aor uses Haversine when flag=False (FR-11 legacy branch).
- T8: write_window_json produces JSON v1.0 with all 9 keys (FR-13, FR-14).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import simpy

from cuas_sim.agents.a29 import A29
from cuas_sim.config import A29Schedule, MissionConfig
from cuas_sim.geometry.haversine import t_pea_seconds
from cuas_sim.metrics.aggregator import MetricsAggregator
from cuas_sim.metrics.records import RunLogger
from cuas_sim.states import A29State

# ─── B-1.1 / T1 — Flag default ───────────────────────────────────────────────

def test_flag_default_true() -> None:
    cfg = MissionConfig()
    assert cfg.uas_spawn_window_enabled is True


# ─── B-1.2 / T2 — Validation of flag=False + phase4_pea_s ───────────────────

def test_flag_false_requires_phase4_positive() -> None:
    base = MissionConfig()
    bad_schedules = dict(base.a29_schedules)
    a29_id = next(iter(bad_schedules))
    orig = bad_schedules[a29_id]
    bad_schedules[a29_id] = A29Schedule(
        farp_name=orig.farp_name,
        cap_index=orig.cap_index,
        phase1_sbpv_to_farp_s=orig.phase1_sbpv_to_farp_s,
        phase2_farp_to_cap_s=orig.phase2_farp_to_cap_s,
        phase3_holding_s=orig.phase3_holding_s,
        phase4_pea_s=0.0,
        phase5_cap_to_farp_s=orig.phase5_cap_to_farp_s,
        phase6_farp_to_sbpv_s=orig.phase6_farp_to_sbpv_s,
    )
    # flag=False + phase4_pea_s=0 ⇒ ValueError mentioning phase4_pea_s
    with pytest.raises(ValueError, match="phase4_pea_s"):
        MissionConfig(uas_spawn_window_enabled=False, a29_schedules=bad_schedules)
    # flag=True + phase4_pea_s=0 ⇒ no error (field ignored)
    MissionConfig(uas_spawn_window_enabled=True, a29_schedules=bad_schedules)


# ─── B-2.1 / T3 — Enum RTB_FUEL_LIMIT exists ─────────────────────────────────

def test_rtb_fuel_limit_enum_member() -> None:
    assert A29State.RTB_FUEL_LIMIT.value == "RTB_FUEL_LIMIT"
    # Legacy members must remain (back-compat for flag=False).
    assert A29State.DEPARTING_CAP.value == "DEPARTING_CAP"
    assert A29State.RTB_BINGO.value == "RTB_BINGO"


# ─── B-3.1 / T4 — first_bingo_event idempotency ──────────────────────────────

def test_first_bingo_event_idempotent() -> None:
    # The production pattern: ``if not event.triggered: event.succeed(...)``.
    # We verify both the simpy mechanism AND that A29 carries the wiring
    # attribute (the agent must accept a ``first_bingo_event`` after Spec 018).
    env = simpy.Environment()
    evt = env.event()
    first_payload = {"a29_id": "A-29-1", "t_s": 10000.0}
    second_payload = {"a29_id": "A-29-2", "t_s": 10001.0}

    if not evt.triggered:
        evt.succeed(value=first_payload)
    assert evt.triggered
    # Second attempt must be guarded and a no-op.
    if not evt.triggered:
        evt.succeed(value=second_payload)
    # Payload is the FIRST a29 (idempotent — no overwrite).
    assert evt.value == first_payload

    # A29 must expose ``first_bingo_event`` as a constructor-settable attribute
    # (slots=True dataclass — verify via fields).
    field_names = {f.name for f in dataclasses.fields(A29)}
    assert "first_bingo_event" in field_names, (
        "A29 must accept first_bingo_event (Spec 018 FR-10 / TD-1)"
    )


# ─── B-4 helpers (build a MetricsAggregator with logger access) ──────────────

def _aggregator_for(cfg: MissionConfig) -> tuple[MetricsAggregator, RunLogger]:
    log = RunLogger()
    agg = MetricsAggregator(cfg=cfg)
    agg.log = log  # type: ignore[attr-defined]
    return agg, log


# ─── B-4.1 / T5 — _t_pea_for_aor emergent (flag=True + RTB observed) ────────

def test_t_pea_emergent_flag_true() -> None:
    cfg = MissionConfig(uas_spawn_window_enabled=True)
    agg, log = _aggregator_for(cfg)
    aor_name = next(iter(cfg.aor_to_a29))  # e.g. "AoR-1"
    a29_id = cfg.aor_to_a29[aor_name]      # e.g. "A-29-1"

    t_rtb = 24512.4
    log.log_rtb_fuel_limit(
        config_name="baseline", a29_id=a29_id, t_s=t_rtb,
        pos_xy=(0.0, 0.0), t_pea_actual_s=t_rtb - cfg.t_pea_start_s,
    )
    got = agg._t_pea_for_aor(aor_name, "baseline")
    assert got == pytest.approx(t_rtb - cfg.t_pea_start_s, abs=1e-9)


# ─── B-4.2 / T6 — _t_pea_for_aor falls back to Haversine ────────────────────

def test_t_pea_fallback_haversine_flag_true_no_rtb() -> None:
    cfg = MissionConfig(uas_spawn_window_enabled=True)
    agg, _ = _aggregator_for(cfg)
    aor_name = next(iter(cfg.aor_to_a29))
    a29_id = cfg.aor_to_a29[aor_name]
    sched = cfg.a29_schedules[a29_id]
    expected = t_pea_seconds(
        farp=cfg.farps[sched.farp_name],
        cap=cfg.caps[f"CAP-{sched.cap_index}"],
        endurance_s=cfg.a29_endurance_s,
        cruise_speed_mps=cfg.a29_cruise_speed_mps,
    )
    # No log entries → fallback Haversine, regardless of config_name passed.
    got = agg._t_pea_for_aor(aor_name, "baseline")
    assert got == pytest.approx(expected, rel=1e-12)


# ─── B-4.3 / T7 — _t_pea_for_aor legacy Haversine (flag=False) ──────────────

def test_t_pea_haversine_legacy_flag_false() -> None:
    cfg = MissionConfig(uas_spawn_window_enabled=False)
    agg, log = _aggregator_for(cfg)
    aor_name = next(iter(cfg.aor_to_a29))
    a29_id = cfg.aor_to_a29[aor_name]
    sched = cfg.a29_schedules[a29_id]
    expected = t_pea_seconds(
        farp=cfg.farps[sched.farp_name],
        cap=cfg.caps[f"CAP-{sched.cap_index}"],
        endurance_s=cfg.a29_endurance_s,
        cruise_speed_mps=cfg.a29_cruise_speed_mps,
    )
    # Even with the log populated, legacy mode must ignore it.
    log.log_rtb_fuel_limit(
        config_name="baseline", a29_id=a29_id, t_s=24512.4,
        pos_xy=(0.0, 0.0), t_pea_actual_s=99.0,
    )
    got = agg._t_pea_for_aor(aor_name, "baseline")
    assert got == pytest.approx(expected, rel=1e-12)


# ─── B-5.1 / T8 — write_window_json schema v1.0 ─────────────────────────────

REQUIRED_WINDOW_KEYS = {
    "field_version",
    "uas_spawn_window_enabled",
    "t_window_start_s",
    "t_window_end_s",
    "t_window_end_reason",
    "horizon_cap_s",
    "baseline_first_bingo",
    "modernized_first_bingo",
    "t_rtb_fuel_limit_per_a29",
}

VALID_REASONS = {"all_first_bingo", "horizon_cap", "max_arrivals", "legacy_horizon"}


def test_window_json_schema_v1(tmp_path: Path) -> None:
    log = RunLogger()
    payload = {
        "field_version": "1.0",
        "uas_spawn_window_enabled": True,
        "t_window_start_s": 9180.0,
        "t_window_end_s": 24512.4,
        "t_window_end_reason": "all_first_bingo",
        "horizon_cap_s": 35000.0,
        "baseline_first_bingo": {"a29_id": "A-29-3", "t_s": 24512.4},
        "modernized_first_bingo": {"a29_id": "A-29-1", "t_s": 25103.7},
        "t_rtb_fuel_limit_per_a29": {
            "baseline": {"A-29-1": 25103.7},
            "modernized": {"A-29-1": 25103.7},
        },
    }
    log.set_window(payload)
    out_path = tmp_path / "window.json"
    log.write_window_json(out_path)

    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(parsed.keys()) == REQUIRED_WINDOW_KEYS
    assert parsed["field_version"] == "1.0"
    assert parsed["t_window_end_reason"] in VALID_REASONS
