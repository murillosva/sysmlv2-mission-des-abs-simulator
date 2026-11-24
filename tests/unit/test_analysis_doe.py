"""Unit — AN-1 DoE / LHS (Spec 026 §5.1). Red phase first (Article III)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from cuas_sim.analysis.doe import Factor, create_design, write_design


def test_lhs_deterministic_scaled() -> None:
    factors = [
        Factor("lambda_per_h", "numeric", 2.0, 18.0),
        Factor("uas_archetype_p_mr", "numeric", 0.0, 1.0),
    ]
    df = create_design(factors, n_samples=4, design_seed=42)

    assert list(df["design_point_id"]) == [0, 1, 2, 3]

    # Oracle = the live scipy primitive (same d, seed, scramble, single .random).
    u = qmc.LatinHypercube(d=2, seed=42, scramble=True).random(4)
    exp_lambda = 2.0 + u[:, 0] * (18.0 - 2.0)
    exp_pmr = 0.0 + u[:, 1] * (1.0 - 0.0)
    np.testing.assert_array_equal(df["lambda_per_h"].to_numpy(), exp_lambda)
    np.testing.assert_array_equal(df["uas_archetype_p_mr"].to_numpy(), exp_pmr)

    assert ((df["lambda_per_h"] >= 2.0) & (df["lambda_per_h"] <= 18.0)).all()
    assert ((df["uas_archetype_p_mr"] >= 0.0) & (df["uas_archetype_p_mr"] <= 1.0)).all()

    # Bit-identical across calls (FR-3).
    df2 = create_design(factors, n_samples=4, design_seed=42)
    assert df.equals(df2)


def test_type_casts() -> None:
    factors = [
        Factor("n", "integer", 1, 6),
        Factor("phil", "categorical", options=("short", "medium", "long")),
        Factor("flag", "boolean"),
    ]
    df = create_design(factors, n_samples=8, design_seed=7)

    assert df["n"].map(lambda x: isinstance(x, (int, np.integer))).all()
    assert df["n"].between(1, 6).all()
    assert set(df["phil"].unique()).issubset({"short", "medium", "long"})
    assert df["flag"].map(lambda x: isinstance(x, (bool, np.bool_))).all()
    assert not df.isna().any().any()


def test_write_design_deterministic_sidecar(tmp_path: Path) -> None:
    factors = [Factor("lambda_per_h", "numeric", 2.0, 18.0)]
    df = create_design(factors, n_samples=4, design_seed=42)
    meta = {"lambda_per_h": {"kind": "numeric", "low": 2.0, "high": 18.0, "options": None}}

    p1 = tmp_path / "a" / "design.parquet"
    p2 = tmp_path / "b" / "design.parquet"
    for p in (p1, p2):
        write_design(df, p, design_seed=42, sampler="LatinHypercube",
                     n_samples=4, factor_meta=meta)

    assert p1.read_bytes() == p2.read_bytes()  # FR-5 byte-determinism
    side1 = p1.with_name(p1.name + ".meta.json")
    side2 = p2.with_name(p2.name + ".meta.json")
    assert side1.read_bytes() == side2.read_bytes()
    m = json.loads(side1.read_text(encoding="utf-8"))
    assert m["field_version"] == "1.0"
    assert m["design_seed"] == 42
    assert m["sampler"] == "LatinHypercube"
    assert m["n_samples"] == 4
