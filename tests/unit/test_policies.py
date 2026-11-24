"""Unit tests for the tracking and weapon policies (Eqs. 3.8–3.15)."""

from __future__ import annotations

import numpy as np
import pytest

from cuas_sim.config import MissionConfig, UASArchetypeSpec
from cuas_sim.policies.tracking import make_tracking_policy
from cuas_sim.policies.weapon import make_weapon_policy


@pytest.fixture
def cfg() -> MissionConfig:
    return MissionConfig()


# ─── tracking ──────────────────────────────────────────────────────────────
# Note: see also Spec 016 §5.1 for the
# new angular formula P_track,bsl = min(1, L_U / (d · θ_acuity)) that replaces
# the size_factor · range_factor heuristic of the earlier tracking formula. Some baseline tests
# below will need updated expected values once Spec 016 lands.


def test_baseline_p_track_at_min_range_fw(cfg: MissionConfig) -> None:
    """At small d in the engagement envelope, P_track,bsl(FW) saturates at 1.0
    via the angular-acuity formula min(1, L_U/(d·θ_acuity)) (Spec 016 §9)."""
    pol = make_tracking_policy(cfg, "baseline")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    rng = np.random.default_rng(0)
    # d=500m: FW (14m) at 500m subtends 14/(500·2.9088e-4)=96.3 ⇒ saturates at 1.0.
    assert pol.p_track(500.0, fw, rng) == pytest.approx(1.0)


def test_baseline_p_track_above_max_range_is_zero(cfg: MissionConfig) -> None:
    """Above d_max=13km the baseline P_track is hard-clipped to zero (Spec 016 §9)."""
    pol = make_tracking_policy(cfg, "baseline")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    assert pol.p_track(cfg.baseline_max_acquisition_m + 1.0, fw,
                       np.random.default_rng(0)) == pytest.approx(0.0)


def test_baseline_p_track_outside_range_is_zero(cfg: MissionConfig) -> None:
    pol = make_tracking_policy(cfg, "baseline")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    assert pol.p_track(50_000.0, fw, np.random.default_rng(0)) == 0.0
    assert pol.p_track(-1.0, fw, np.random.default_rng(0)) == 0.0


def test_modernized_p_track_outside_annulus(cfg: MissionConfig) -> None:
    """Outside (14, 20] km the modernized P_track is zero (Eq. 3.13)."""
    pol = make_tracking_policy(cfg, "modernized")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    rng = np.random.default_rng(0)
    assert pol.p_track(13_000.0, fw, rng) == 0.0
    assert pol.p_track(20_500.0, fw, rng) == 0.0


def test_modernized_p_track_fw_returns_one(cfg: MissionConfig) -> None:
    pol = make_tracking_policy(cfg, "modernized")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    assert pol.p_track(17_000.0, fw, np.random.default_rng(0)) == pytest.approx(1.0)


def test_modernized_p_track_mr_returns_point_two(cfg: MissionConfig) -> None:
    pol = make_tracking_policy(cfg, "modernized")
    mr = UASArchetypeSpec(name="multi-rotor", max_speed_mps=29, dimension_m=1.0, ceiling_m=6000)
    assert pol.p_track(16_000.0, mr, np.random.default_rng(0)) == pytest.approx(0.2, abs=1e-10)


def test_modernized_p_track_at_eoir_min_is_zero(cfg: MissionConfig) -> None:
    pol = make_tracking_policy(cfg, "modernized")
    mr = UASArchetypeSpec(name="multi-rotor", max_speed_mps=29, dimension_m=1.0, ceiling_m=6000)
    assert pol.p_track(14_000.0, mr, np.random.default_rng(0)) == 0.0


def test_modernized_p_track_above_eoir_max_is_zero(cfg: MissionConfig) -> None:
    pol = make_tracking_policy(cfg, "modernized")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    assert pol.p_track(21_000.0, fw, np.random.default_rng(0)) == 0.0


def test_modernized_p_track_fw_constant_in_annulus(cfg: MissionConfig) -> None:
    pol = make_tracking_policy(cfg, "modernized")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    rng = np.random.default_rng(0)
    for d in (14_001.0, 17_000.0, 20_000.0):
        assert pol.p_track(d, fw, rng) == pytest.approx(1.0), f"failed at d={d}"


def test_baseline_first_update_interval_floor(cfg: MissionConfig) -> None:
    pol = make_tracking_policy(cfg, "baseline")
    rng = np.random.default_rng(42)
    results = [pol.first_update_interval(rng) for _ in range(200)]
    assert all(r >= cfg.baseline_reaction_time_min_s for r in results)
    assert all(r <= cfg.baseline_update_interval_s[1] for r in results)


def test_baseline_p_track_mr_at_5km(cfg: MissionConfig) -> None:
    """Spec 016 §9: angular-acuity formula resolves a 1m MR at 5km well above
    the 1-arcmin threshold ⇒ P_track ≈ 0.688 (vs ≈0.044 under the old
    size-factor × range-decay heuristic — see Spec 016 §10 rationale).
    """
    pol = make_tracking_policy(cfg, "baseline")
    mr = UASArchetypeSpec(name="multi-rotor", max_speed_mps=29, dimension_m=1.0, ceiling_m=6000)
    # min(1, L_U / (d · θ_acuity)) = min(1, 1 / (5000 · 2.9088e-4)) ≈ 0.6876
    expected = min(1.0, 1.0 / (5_000.0 * cfg.baseline_visual_acuity_rad))
    assert pol.p_track(5_000.0, mr, np.random.default_rng(0)) == pytest.approx(expected, abs=1e-9)


def test_annulus_predicate(cfg: MissionConfig) -> None:
    pol_b = make_tracking_policy(cfg, "baseline")
    assert pol_b.in_sampling_annulus(2_500.0)
    assert not pol_b.in_sampling_annulus(1_500.0)
    assert pol_b.in_sampling_annulus(13_000.0)
    assert not pol_b.in_sampling_annulus(13_500.0)

    pol_m = make_tracking_policy(cfg, "modernized")
    assert pol_m.in_sampling_annulus(15_000.0)
    assert not pol_m.in_sampling_annulus(13_000.0)
    assert pol_m.in_sampling_annulus(20_000.0)
    assert not pol_m.in_sampling_annulus(20_500.0)


# ─── weapon ────────────────────────────────────────────────────────────────


def test_p_hit_at_min_range_fw_max(cfg: MissionConfig) -> None:
    """At the envelope minimum and fixed-wing target, P_hit = M·0.80·1.0 = 0.80."""
    pol = make_weapon_policy(cfg, "modernized")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    p = pol.p_hit_expected(pol.envelope_min_m, fw)
    assert p == pytest.approx(0.80, abs=1e-9)


def test_p_hit_at_max_range_is_zero(cfg: MissionConfig) -> None:
    pol = make_weapon_policy(cfg, "modernized")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    assert pol.p_hit_expected(pol.envelope_max_m, fw) == pytest.approx(0.0, abs=1e-9)


def test_p_hit_mr_half_of_fw(cfg: MissionConfig) -> None:
    """At any in-envelope range, multi-rotor M=0.5 ⇒ P_hit_mr = 0.5 · P_hit_fw."""
    pol = make_weapon_policy(cfg, "baseline")
    mr = UASArchetypeSpec(name="multi-rotor", max_speed_mps=29, dimension_m=1.0, ceiling_m=6000)
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    for d in (500.0, 1_000.0, 1_500.0):
        p_mr = pol.p_hit_expected(d, mr)
        p_fw = pol.p_hit_expected(d, fw)
        assert p_mr == pytest.approx(0.5 * p_fw, rel=1e-9)


def test_p_hit_realised_obeys_distribution(cfg: MissionConfig) -> None:
    """Bernoulli mean ≈ expected P_hit over many trials."""
    pol = make_weapon_policy(cfg, "baseline")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0, ceiling_m=18000)
    rng = np.random.default_rng(7)
    trials = 5_000
    hits = sum(pol.p_hit_realised(800.0, fw, rng)[0] for _ in range(trials))
    expected = pol.p_hit_expected(800.0, fw)
    assert hits / trials == pytest.approx(expected, abs=0.025)


def test_firing_distance_is_triangular_within_subband(cfg: MissionConfig) -> None:
    # Note: Spec 017 §5.4 — replaces the Spec 009 discrete {min,mid,max} draw
    # with a continuous Triangular within each philosophy's range/3 sub-band.
    from cuas_sim.policies.weapon import firing_subband

    pol = make_weapon_policy(cfg, "modernized")
    rng = np.random.default_rng(0)
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(3_000):
        dist, idx = pol.draw_firing_distance(rng)
        low, _mode, high = firing_subband(pol.envelope_min_m, pol.envelope_max_m, idx)
        assert low - 1e-6 <= dist <= high + 1e-6
        counts[idx] += 1
    # Each philosophy is still drawn ≈ 1/3 of the time (discrete philosophy draw).
    for v in counts.values():
        assert 800 <= v <= 1_200


# ─── Spec 015: two-regime weapon (FR-3) ────────────────────────────────────


def test_modernized_regime_switch_at_2km() -> None:
    cfg = MissionConfig()
    w = make_weapon_policy(cfg, "modernized")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0,
                          ceiling_m=18000)
    # Guided just above 2 km → max 0.8·M (M_fw=1).
    assert w.p_hit_at(2_000.0, fw) == pytest.approx(0.8, abs=1e-9)
    assert w.p_hit_at(14_000.0, fw) == pytest.approx(0.0, abs=1e-9)
    # Unguided just below 2 km → ~0; rises to 0.8 at 0.5 km.
    assert w.p_hit_at(1_999.0, fw) == pytest.approx(0.0, abs=2e-3)
    assert w.p_hit_at(500.0, fw) == pytest.approx(0.8, abs=1e-9)
    assert w.floor_m == pytest.approx(500.0)


def test_baseline_single_regime() -> None:
    cfg = MissionConfig()
    w = make_weapon_policy(cfg, "baseline")
    fw = UASArchetypeSpec(name="fixed-wing", max_speed_mps=130, dimension_m=14.0,
                          ceiling_m=18000)
    assert w.p_hit_at(500.0, fw) == pytest.approx(0.8, abs=1e-9)
    assert w.p_hit_at(2_000.0, fw) == pytest.approx(0.0, abs=1e-9)
    assert w.floor_m == pytest.approx(500.0)
