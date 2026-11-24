"""Property-based oracle for the AN-2 report determinism (Spec 027 §5.2 P4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cuas_sim.analysis.report import build_report


def _make_datafarm(seed: int = 0, n_points: int = 4, n_reps: int = 10) -> pd.DataFrame:
    """Minimal synthetic AN-1 datafarm (tidy/long) — no simulation run."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dp in range(n_points):
        lam = 2.0 + dp * 4.0
        pmr = 0.1 + dp * 0.2
        for rep in range(n_reps):
            sd = 1000 * dp + rep
            bsl = float(rng.uniform(0.05, 0.20))
            mod = bsl + float(rng.uniform(0.05, 0.30))  # modernized > baseline
            base = {"design_point_id": dp, "lambda_per_h": lam,
                    "uas_archetype_p_mr": pmr, "rep": rep, "seed": sd, "aor": "ALL"}
            for cfg, val in (("baseline", bsl), ("modernized", mod), ("delta", mod - bsl)):
                rows.append({**base, "config": cfg, "metric": "MOS-1", "value": val})
            rows.append({**base, "config": "modernized",
                         "metric": "rockets_per_engagement", "value": float(rng.uniform(1, 5))})
    return pd.DataFrame(rows)


def test_report_deterministic() -> None:
    df = _make_datafarm(seed=0)

    r1 = build_report(df, response="MOS-1", bootstrap_seed=7, n_resamples=499)
    r2 = build_report(df, response="MOS-1", bootstrap_seed=7, n_resamples=499)

    assert r1 == r2
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
