"""Shared test fixtures."""

from __future__ import annotations

import pytest

from cuas_sim.config import MissionConfig


@pytest.fixture
def default_cfg() -> MissionConfig:
    """Plain default config (no YAML, no SysML)."""
    return MissionConfig()


@pytest.fixture
def fast_cfg() -> MissionConfig:
    """A small/fast configuration for smoke tests."""
    return MissionConfig(
        lambda_per_sec=6.0 / 3600.0,   # 6/h
        skip_pre_pea=True,
    )
