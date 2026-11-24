"""Property-based oracles for the AN-report consolidated layer (Spec 029 §5.2)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from cuas_sim.analysis.report import (
    CAUSALITY_CAVEAT,
    NO_GENERALIZE_CAVEAT,
    build_consolidated_report,
    render_md,
)
from cuas_sim.analysis.reporting import render_html

_RESPONSES = ["MOS-1", "MOE-1"]
_AORS = ["ALL"]


def _make_datafarm(
    seed: int = 0, n_points: int = 10, n_reps: int = 6,
    responses: tuple[str, ...] = ("MOS-1", "MOE-1"),
) -> pd.DataFrame:
    """Synthetic AN-1 datafarm (tidy/long) with a controlled mix-dominated signal."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dp in range(n_points):
        lam = 2.0 + 16.0 * dp / (n_points - 1)
        pmr = dp / (n_points - 1)
        for rep in range(n_reps):
            sd = 1000 * dp + rep
            base = {"design_point_id": dp, "lambda_per_h": lam,
                    "uas_archetype_p_mr": pmr, "rep": rep, "seed": sd}
            for resp in responses:
                bsl = float(rng.uniform(0.05, 0.20))
                gain = 0.25 * pmr + 0.004 * lam + float(rng.normal(0.0, 0.01))
                mod = bsl + gain
                for cfg, val in (("baseline", bsl), ("modernized", mod),
                                 ("delta", mod - bsl)):
                    rows.append({**base, "config": cfg, "aor": "ALL",
                                 "metric": resp, "value": val})
            rows.append({**base, "config": "modernized", "aor": "ALL",
                         "metric": "rockets_per_engagement",
                         "value": float(rng.uniform(1, 5))})
    return pd.DataFrame(rows)


def _build(seed: int = 0) -> dict:
    df = _make_datafarm(seed=seed)
    return build_consolidated_report(
        df, responses=_RESPONSES, aors=_AORS, alpha=0.05,
        bootstrap_seed=7, model_seed=13, n_resamples=299, grid_resolution=6,
    )


def test_contract_determinism() -> None:
    r1 = _build(seed=0)
    r2 = _build(seed=0)
    assert r1 == r2
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_three_layer_determinism() -> None:
    report = _build(seed=0)
    assert render_md(report) == render_md(report)
    assert render_html(report, {}) == render_html(report, {})


def _numbers(text: str) -> set[str]:
    # capture decimals incl. scientific notation (e.g. p-values '1.63e-11') as one token
    return set(re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", text))


def test_presentation_subset_of_contract() -> None:
    report = _build(seed=0)
    md = render_md(report)

    # allowed = every numeric leaf of the contract, in each format the renderer uses
    allowed: set[str] = set()

    def _walk(obj: object) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, (int, float)):
            x = float(obj)
            allowed.add(f"{x:.4f}")
            allowed.add(f"{x:.3f}")
            allowed.add(f"{x:.2f}")
            allowed.add(f"{x:.4g}")
            allowed.add(str(obj))

    _walk(report)
    md_numbers = _numbers(md)
    unexplained = {n for n in md_numbers if n not in allowed}
    assert not unexplained, f"presentation introduced numbers absent from contract: {unexplained}"


def test_decision_index_gate_monotonic() -> None:
    from cuas_sim.analysis.report import build_decision_index

    master = [
        {"response": "MOS-1", "aor": "AoR-hi", "effect": 0.30,
         "ci_low": 0.20, "ci_high": 0.40, "p_value": 1e-3, "test": "wilcoxon"},
        {"response": "MOS-1", "aor": "AoR-lo", "effect": 0.10,
         "ci_low": 0.05, "ci_high": 0.15, "p_value": 1e-3, "test": "wilcoxon"},
        {"response": "MOS-1", "aor": "AoR-ns", "effect": 0.50,
         "ci_low": -0.10, "ci_high": 1.10, "p_value": 0.30, "test": "wilcoxon"},
    ]
    idx = build_decision_index(master, primary_response="MOS-1", cost_by_aor={})
    order = [r["aor"] for r in idx["ranking"]]

    # among significant, larger effect ranks first
    assert order.index("AoR-hi") < order.index("AoR-lo")
    # a non-significant scenario (CI crosses zero) never outranks a significant one,
    # despite the largest raw effect
    sig_positions = [order.index("AoR-hi"), order.index("AoR-lo")]
    assert order.index("AoR-ns") > max(sig_positions)


def test_validity_always_present() -> None:
    for seed in (0, 1, 2):
        report = _build(seed=seed)
        v = report["validity"]
        assert v["factor_ranges"]  # ranges declared
        assert v["n_design_points"] >= 1
        assert set(v["seeds"]) >= {"bootstrap", "model"}
        assert NO_GENERALIZE_CAVEAT in v["caveats"]
        assert CAUSALITY_CAVEAT in v["caveats"]
