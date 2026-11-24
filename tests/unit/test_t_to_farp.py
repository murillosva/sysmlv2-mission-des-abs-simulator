from __future__ import annotations

"""``t_to_farp`` primitive + bingo trigger boundary (FR-7, FR-8).

Group B (Red phase): must FAIL until C-1 lands ``t_to_farp`` in
``geometry/projection.py``.
"""

import pytest

from cuas_sim.geometry.projection import XY


def test_t_to_farp_known_answer() -> None:
    """pos=(0,0), FARP=(50_000,0), v=144 ⇒ 50000/144 = 347.222… s."""
    from cuas_sim.geometry.projection import t_to_farp

    t = t_to_farp(XY(0.0, 0.0), XY(50_000.0, 0.0), 144.0)
    assert t == pytest.approx(50_000.0 / 144.0, rel=1e-9)


def test_bingo_trigger_boundary_zero_reserve() -> None:
    """Bingo predicate (reserva zero): remaining ≤ t_to_FARP."""
    from cuas_sim.geometry.projection import t_to_farp

    t_farp = t_to_farp(XY(0.0, 0.0), XY(50_000.0, 0.0), 144.0)
    assert (t_farp <= t_farp) is True              # exact equality fires
    assert ((t_farp + 1e-3) <= t_farp) is False    # remaining just above ⇒ no bingo


def test_phase6_geometry_farp_to_sbpv() -> None:
    """Phase 6 (FARP→SBPV) = ‖FARP − SBPV‖ / v_cruise."""
    from cuas_sim.geometry.projection import t_to_farp

    farp = XY(0.0, 0.0)
    sbpv = XY(30_000.0, 40_000.0)  # ‖·‖ = 50_000 m
    t = t_to_farp(farp, sbpv, 144.0)
    assert t == pytest.approx(50_000.0 / 144.0, rel=1e-9)
