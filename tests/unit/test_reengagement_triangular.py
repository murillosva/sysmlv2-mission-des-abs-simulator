from __future__ import annotations

"""Triangular re-engagement interval T(8,12,18) s (FR-3).

Group B (Red phase): must FAIL until C-3 adds ``reengagement_triangular_s`` to
``MissionConfig`` and the controller switches ``uniform`` → ``triangular``.
"""

import numpy as np
import pytest

from cuas_sim.config import MissionConfig


def test_reengagement_triangular_config_default() -> None:
    """The re-engagement interval is parameterised as Triangular(8, 12, 18) s."""
    cfg = MissionConfig()
    assert cfg.reengagement_triangular_s == (8.0, 12.0, 18.0)


def test_reengagement_triangular_bounds_and_mean() -> None:
    """Draws stay in [8, 18] s; sample mean ≈ (8+12+18)/3 = 12.667 s."""
    cfg = MissionConfig()
    rng = np.random.default_rng(7)
    draws = np.array([rng.triangular(*cfg.reengagement_triangular_s) for _ in range(10_000)])
    assert draws.min() >= 8.0
    assert draws.max() <= 18.0
    assert float(draws.mean()) == pytest.approx((8 + 12 + 18) / 3, abs=0.1)
