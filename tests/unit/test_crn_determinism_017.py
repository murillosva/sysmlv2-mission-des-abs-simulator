from __future__ import annotations

# Note: Spec 018 parametrizes this test on uas_spawn_window_enabled in [True, False]
# and adds ``test_back_compat_flag_false_bit_identical`` against
# ``tests/fixtures/spec018_back_compat_hashes.json`` (AC-2 / P5).
"""Determinism under CRN (P7, Article IV).

This test PASSES from the start (the current model is deterministic) and MUST
stay green after Spec 017 — same master seed ⇒ bit-identical metrics, even with
the Triangular firing/re-engagement draws and the geometric bingo.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.simulation.run import run_paired


def _mos1_all(df: pd.DataFrame, cfg_name: str) -> float:
    sub = df[(df["metric"] == "MOS-1") & (df["aor"] == "ALL")
             & (df["config_name"] == cfg_name)]
    return float(sub["value"].iloc[0])


def test_same_seed_bit_identical(tmp_path: Path) -> None:
    cfg = MissionConfig(lambda_per_sec=12.0 / 3600.0, skip_pre_pea=True)
    res1 = run_paired(cfg, seed=2024, outputs_root=tmp_path / "a", max_arrivals=20)
    res2 = run_paired(cfg, seed=2024, outputs_root=tmp_path / "b", max_arrivals=20)

    assert _mos1_all(res1.metrics, "baseline") == _mos1_all(res2.metrics, "baseline")
    assert _mos1_all(res1.metrics, "modernized") == _mos1_all(res2.metrics, "modernized")


# ─── Spec 018 — Back-compat bit-identity (AC-2 / P5) ────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "spec018_back_compat_hashes.json"
_CONFIG_PATH = _REPO_ROOT / "configs" / "zida_shield_default.yaml"
_SYSML_ROOT = _REPO_ROOT / "sysml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_fixture() -> dict[str, dict[str, str]]:
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"A-5 fixture missing: {_FIXTURE_PATH}")
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))["hashes"]


# Use a small subset of seeds for fast CI; expand to all 20 after Green phase.
_SUBSET_SEEDS = ["1000", "1010", "1019"]

# The fixture hashes the raw bytes of files written on the reference
# environment. Those bytes are not portable: the Parquet metadata embeds the
# pandas/pyarrow versions, and on Linux the bytes differ even with the same
# versions (observed in CI). The bit-exact comparison therefore runs only there.
_REFERENCE_ENV = (
    sys.platform == "win32" and pd.__version__ == "3.0.3" and pa.__version__ == "24.0.0"
)
_reference_env_only = pytest.mark.skipif(
    not _REFERENCE_ENV,
    reason="bit-exact fixture recorded on the reference environment "
    "(Windows, pandas 3.0.3, pyarrow 24.0.0)",
)


@_reference_env_only
@pytest.mark.parametrize(
    "seed_str",
    _SUBSET_SEEDS if _FIXTURE_PATH.exists() else [],
)
def test_back_compat_flag_false_bit_identical(seed_str: str) -> None:
    expected = _load_fixture()[seed_str]
    seed = int(seed_str)
    cfg = MissionConfig.from_yaml_with_sysml(_CONFIG_PATH, sysml_root=_SYSML_ROOT)
    # Spec 030 FR-18: reproduce the pre-018 baseline exactly by pinning the
    # legacy generation flags AND the historical truncated sigma (the YAML now
    # carries full-precision sigma per FR-9a; the pdf path consumes it).
    cfg = cfg.replaced(
        uas_spawn_window_enabled=False,
        spawn_interpolation="chord",
        sector_draw_rule="pdf",
        heading_inward_rule="east_flip",
        sector_sigma=4.3333,
    )
    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)
        run_paired(
            cfg, seed=seed,
            outputs_root=tmp_root,
            slug=f"back_compat_unit_seed{seed}",
            max_arrivals=50,
            persist=True,
        )
        run_dir = next((tmp_root / "runs").iterdir())
        got_metrics = _sha256(run_dir / "metrics.parquet")
        got_bsl = _sha256(run_dir / "events_baseline.jsonl.gz")
        got_mod = _sha256(run_dir / "events_modernized.jsonl.gz")
    assert got_metrics == expected["metrics_sha256"], (
        f"metrics.parquet hash mismatch for seed {seed} (flag=False vs pre-018)"
    )
    assert got_bsl == expected["events_baseline_sha256"]
    assert got_mod == expected["events_modernized_sha256"]
