"""Property — AN-1 datafarm delta consistency (Spec 026 §5.2 P4). Red phase first."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from cuas_sim.analysis.experiment import _delta_rows


@given(
    vb=st.floats(-5, 5, allow_nan=False),
    vm=st.floats(-5, 5, allow_nan=False),
)
@settings(max_examples=50)
def test_delta_consistency(vb: float, vm: float) -> None:
    rows = [
        {"design_point_id": 0, "rep": 0, "seed": 1, "lambda_per_h": 6.0,
         "uas_archetype_p_mr": 0.5, "config": "baseline", "aor": "ALL",
         "metric": "MOS-1", "value": vb},
        {"design_point_id": 0, "rep": 0, "seed": 1, "lambda_per_h": 6.0,
         "uas_archetype_p_mr": 0.5, "config": "modernized", "aor": "ALL",
         "metric": "MOS-1", "value": vm},
    ]
    deltas = [r for r in _delta_rows(rows) if r["config"] == "delta"]
    assert len(deltas) == 1
    assert deltas[0]["value"] == (vm - vb)
