"""Aggregator tests with synthetic engagement records.

Note: Spec 018 — _t_pea_for_aor source change only; Eqs. 3.5-3.7 form unchanged.
B-8.4 will parametrize the T_PEA-related cases on uas_spawn_window_enabled.
"""

from __future__ import annotations

import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.engagement.controller import EngagementOutcome
from cuas_sim.metrics.aggregator import MetricsAggregator
from cuas_sim.metrics.records import RunLogger
from cuas_sim.metrics.replanning import build_replanning_signal


def _outcome(
    *,
    cfg_name: str,
    aor: str = "AoR-1",
    sector: int = 30,
    archetype: str = "fixed-wing",
    is_neut: bool,
    is_neut_before_zida: bool,
    moe2_mean: float = 0.5,
    moe2_n: int = 4,
    p_hit: float = 0.6,
    moe1: float = 30.0,
    mop1: int = 2,
) -> EngagementOutcome:
    """Helper to fabricate an EngagementOutcome with sensible defaults."""
    return EngagementOutcome(
        config_name=cfg_name, a29_id="A-29", uas_id=1,
        aor_name=aor, sector_index=sector, archetype_name=archetype,
        t_pursuit_start_s=0.0, t_envelope_entry_s=10.0,
        t_fire_s=15.0, t_kill_s=15.0 if is_neut else float("nan"),
        initial_distance_m=10000.0, intercept_distance_m=2000.0,
        firing_distance_m=1000.0,
        is_neutralized=is_neut, is_neutralized_before_zida=is_neut_before_zida,
        moe1_engagement_time_s=moe1,
        moe2_samples=[moe2_mean] * moe2_n,
        moe3_p_hit_expected=p_hit,
        mop1_heading_update_count=mop1,
        mop2_acquisition_range_m=13000.0,
        mop3_envelope_max_m=2000.0,
        feasible=True,
    )


@pytest.fixture
def synthetic_log_perfect_modern() -> RunLogger:
    """Perfect modernized neutralization, mediocre baseline."""
    log = RunLogger()
    # 6 baseline engagements: 3 killed-before-ZIDA, 3 leaked
    for i in range(3):
        log.log_engagement(_outcome(
            cfg_name="baseline",
            is_neut=True, is_neut_before_zida=True,
            p_hit=0.5, moe2_mean=0.4, moe1=50.0, mop1=8,
        ))
    for i in range(3):
        log.log_engagement(_outcome(
            cfg_name="baseline",
            is_neut=False, is_neut_before_zida=False,
            p_hit=0.5, moe2_mean=0.4, moe1=50.0, mop1=8,
        ))
    # 6 modernized engagements: all killed-before-ZIDA
    for i in range(6):
        log.log_engagement(_outcome(
            cfg_name="modernized",
            is_neut=True, is_neut_before_zida=True,
            p_hit=0.75, moe2_mean=0.7, moe1=20.0, mop1=3,
        ))
    return log


def test_mos1_overall_perfect_modern(
    synthetic_log_perfect_modern: RunLogger,
) -> None:
    cfg = MissionConfig()
    agg = MetricsAggregator(cfg)
    df = agg.compute_all(synthetic_log_perfect_modern, total_arrivals=6)
    mos1_b = df[(df["config_name"] == "baseline") & (df["metric"] == "MOS-1") & (df["aor"] == "ALL")]
    mos1_m = df[(df["config_name"] == "modernized") & (df["metric"] == "MOS-1") & (df["aor"] == "ALL")]
    assert mos1_b["value"].iloc[0] == pytest.approx(0.5, abs=1e-6)
    assert mos1_m["value"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_mos2_perfect_modern_one(
    synthetic_log_perfect_modern: RunLogger,
) -> None:
    cfg = MissionConfig()
    agg = MetricsAggregator(cfg)
    df = agg.compute_all(synthetic_log_perfect_modern, total_arrivals=6)
    mos2 = df[df["metric"] == "MOS-2"]["value"].iloc[0]
    # leaks_baseline = 3, leaks_modern = 0 → MOS-2 = 1.
    assert mos2 == pytest.approx(1.0, abs=1e-6)


def test_moe3_mean_is_simple_average(
    synthetic_log_perfect_modern: RunLogger,
) -> None:
    cfg = MissionConfig()
    agg = MetricsAggregator(cfg)
    df = agg.compute_all(synthetic_log_perfect_modern, total_arrivals=6)
    moe3_m = df[(df["config_name"] == "modernized") & (df["metric"] == "MOE-3") & (df["aor"] == "ALL")]
    assert moe3_m["value"].iloc[0] == pytest.approx(0.75, abs=1e-6)
    moe3_b = df[(df["config_name"] == "baseline") & (df["metric"] == "MOE-3") & (df["aor"] == "ALL")]
    assert moe3_b["value"].iloc[0] == pytest.approx(0.5, abs=1e-6)


def test_moe2_zero_when_no_annulus_samples() -> None:
    """A record with empty moe2_samples must yield mean=0."""
    log = RunLogger()
    log.log_engagement(_outcome(
        cfg_name="baseline",
        is_neut=True, is_neut_before_zida=True,
        moe2_mean=0.0, moe2_n=0,
    ))
    cfg = MissionConfig()
    df = MetricsAggregator(cfg).compute_all(log, total_arrivals=1)
    moe2 = df[(df["config_name"] == "baseline") & (df["metric"] == "MOE-2") & (df["aor"] == "ALL")]
    assert moe2["value"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_replanning_signal_not_triggered_above_target(
    synthetic_log_perfect_modern: RunLogger,
) -> None:
    cfg = MissionConfig()
    agg = MetricsAggregator(cfg)
    df = agg.compute_all(synthetic_log_perfect_modern, total_arrivals=6)
    sig = build_replanning_signal(cfg, df, synthetic_log_perfect_modern, mos1_target=0.80)
    assert sig.triggered is False
    assert sig.mos1_modernized == pytest.approx(1.0)


def test_replanning_signal_triggered_below_target() -> None:
    log = RunLogger()
    # 5 modernized engagements: only 2 kills in ZIDA → MOS-1 = 0.4 < 0.80
    for i in range(2):
        log.log_engagement(_outcome(
            cfg_name="modernized", is_neut=True, is_neut_before_zida=True,
            p_hit=0.6, moe2_mean=0.5, moe2_n=3,
        ))
    for i in range(3):
        log.log_engagement(_outcome(
            cfg_name="modernized", aor="AoR-1", sector=30,
            is_neut=False, is_neut_before_zida=False,
        ))
    cfg = MissionConfig()
    df = MetricsAggregator(cfg).compute_all(log, total_arrivals=5)
    sig = build_replanning_signal(cfg, df, log, mos1_target=0.80)
    assert sig.triggered is True
    assert "AoR-1" in sig.leaks_per_aor
    assert sig.hint  # non-empty hint string


# ─── Spec 015: cumulative neutralization (Eq. 3.15) ────────────────────────


def test_cumulative_neutralization_known_answer() -> None:
    from cuas_sim.engagement.controller import cumulative_neutralization

    # 1 − (1−0)(1−0.4)(1−0.8) = 1 − 0.6·0.2 = 0.88
    assert cumulative_neutralization([0.0, 0.4, 0.8]) == pytest.approx(0.88)
    assert cumulative_neutralization([0.8]) == pytest.approx(0.8)
    assert cumulative_neutralization([]) == pytest.approx(0.0)
    assert cumulative_neutralization([1.0, 0.5]) == pytest.approx(1.0)


def test_moe3_is_mean_of_cumulative_pneut_over_engagements() -> None:
    # Spec 015 FR-5: MOE-3 = mean of the per-engagement cumulative P_neut,i.
    # FR-8: MOE-2 (annulus mean) stays decoupled from the multi-shot fields.
    log = RunLogger()
    # Engagement A: P_neut from [0, 0.4, 0.8] = 0.88; B: 0.40.
    log.log_engagement(_outcome(
        cfg_name="modernized", is_neut=True, is_neut_before_zida=True,
        p_hit=0.88, moe2_mean=0.3, moe2_n=4,
    ))
    log.log_engagement(_outcome(
        cfg_name="modernized", is_neut=True, is_neut_before_zida=True,
        p_hit=0.40, moe2_mean=0.7, moe2_n=4,
    ))
    df = MetricsAggregator(MissionConfig()).compute_all(log, total_arrivals=2)

    def _val(metric: str) -> float:
        sub = df[(df["config_name"] == "modernized")
                 & (df["metric"] == metric) & (df["aor"] == "ALL")]
        return float(sub["value"].iloc[0])

    assert _val("MOE-3") == pytest.approx((0.88 + 0.40) / 2, abs=1e-9)  # FR-5
    assert _val("MOE-2") == pytest.approx((0.3 + 0.7) / 2, abs=1e-9)    # FR-8


# ─── Spec 018 — _t_pea_for_aor source change (B-8.4) ────────────────────────


@pytest.mark.parametrize("uas_spawn_window_enabled", [True, False])
def test_compute_all_structure_invariant_under_flag(
    synthetic_log_perfect_modern: RunLogger,
    uas_spawn_window_enabled: bool,
) -> None:
    # Spec 018 — _t_pea_for_aor source change only; Eqs. 3.5–3.7 form unchanged.
    # The output structure of ``compute_all`` (set of (config, aor, metric)
    # tuples and MOE-1 row presence) must be invariant across the flag.
    cfg = MissionConfig(uas_spawn_window_enabled=uas_spawn_window_enabled)
    agg = MetricsAggregator(cfg)
    df = agg.compute_all(synthetic_log_perfect_modern, total_arrivals=6)

    triples = {(row["config_name"], row["aor"], row["metric"])
               for _, row in df.iterrows()}
    # Both modes must emit MOE-1 rows per config × AoR (and ALL).
    assert ("baseline", "ALL", "MOE-1") in triples
    assert ("modernized", "ALL", "MOE-1") in triples
    # The set of metric names produced is the same in both modes.
    assert {row[2] for row in triples} == {
        "MOS-1", "MOS-2", "MOE-1", "MOE-2", "MOE-3", "MOP-1", "MOP-2", "MOP-3",
    }


@pytest.mark.parametrize("uas_spawn_window_enabled", [True, False])
def test_t_pea_for_aor_signature_accepts_config_name(
    uas_spawn_window_enabled: bool,
) -> None:
    # FR-11: _t_pea_for_aor MUST accept (aor_name, config_name=None) — additive.
    cfg = MissionConfig(uas_spawn_window_enabled=uas_spawn_window_enabled)
    agg = MetricsAggregator(cfg)
    agg.log = RunLogger()  # type: ignore[attr-defined]
    aor_name = next(iter(cfg.aor_to_a29))
    # Call site adopted in C-6.3 — must not raise.
    value_with = agg._t_pea_for_aor(aor_name, "baseline")
    value_without = agg._t_pea_for_aor(aor_name)
    assert isinstance(value_with, float)
    assert isinstance(value_without, float)
