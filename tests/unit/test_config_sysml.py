"""Tests for the MissionConfig and SysML loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cuas_sim.config import MissionConfig
from cuas_sim.sysml_loader import SysMLLoader


def test_default_config_is_hashable() -> None:
    cfg = MissionConfig()
    h = cfg.config_hash()
    assert isinstance(h, str)
    assert len(h) == 64


def test_default_config_round_trip_through_dict() -> None:
    cfg = MissionConfig()
    d = cfg.to_dict()
    assert "lambda_per_sec" in d
    assert d["a29_endurance_s"] == cfg.a29_endurance_s


def test_replaced_returns_new_instance() -> None:
    cfg = MissionConfig()
    cfg2 = cfg.replaced(lambda_per_sec=0.01)
    assert cfg.lambda_per_sec != cfg2.lambda_per_sec
    assert cfg.config_hash() != cfg2.config_hash()


def test_from_yaml_minimal(tmp_path: Path) -> None:
    """A YAML with just one override merges correctly with defaults."""
    p = tmp_path / "small.yaml"
    p.write_text(yaml.safe_dump({"lambda_per_sec": 0.005, "skip_pre_pea": True}))
    cfg = MissionConfig.from_yaml(p)
    assert cfg.lambda_per_sec == 0.005
    assert cfg.skip_pre_pea is True
    # Defaults preserved
    assert cfg.zida_boundary_nmi == 80.0


def test_from_yaml_unknown_keys_warn(tmp_path: Path, caplog) -> None:
    p = tmp_path / "junk.yaml"
    p.write_text(yaml.safe_dump({"not_a_real_field": 999}))
    with caplog.at_level("WARNING"):
        cfg = MissionConfig.from_yaml(p)
    assert "Unknown" in caplog.text
    assert cfg.lambda_per_sec > 0


def test_sysml_loader_handles_missing_root() -> None:
    with pytest.raises(FileNotFoundError):
        SysMLLoader(Path("/nonexistent")).load()


def test_sysml_loader_extracts_overrides() -> None:
    """The bundled sysml/ directory yields ≥ one provenance entry."""
    root = Path(__file__).parent.parent.parent / "sysml"
    if not root.exists():
        pytest.skip("sysml/ not present in test environment")
    ov = SysMLLoader(root).load()
    # Whatever else, we expect a baseline_max_acquisition or the M3AR
    # update interval to be picked up.
    assert ov.provenance, "No SysML overrides extracted"


# ─── Spec 015: re-engagement config + RNG stream ───────────────────────────


def test_reengagement_config_defaults() -> None:
    cfg = MissionConfig()
    assert cfg.reengagement_interval_s == (10.0, 20.0)
    assert cfg.rocket_budget_per_sortie == 49


def test_reengagement_rng_stream_split_per_config() -> None:
    from cuas_sim.rng import RNGRegistry

    reg = RNGRegistry.from_master_seed(42)
    b = reg.per_config("baseline")
    m = reg.per_config("modernized")
    assert hasattr(b, "reengagement_interval")
    # Independent streams per config (CRN: decision streams split).
    assert b.reengagement_interval.random() != m.reengagement_interval.random()
