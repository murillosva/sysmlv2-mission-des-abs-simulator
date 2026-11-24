"""Compare two steps of the analysis layer with the AsaPy library itself.

Run in a separate environment where AsaPy (Dantas et al., 2024) is installed; it cannot be
installed next to the simulator (see docs/analysis_layer.md, section 7). On a fixed toy
data set, compares what is methodologically comparable: the feature-score ranking and the
Pareto set. Exits with 0 on agreement, 1 on divergence and 2 when AsaPy is not importable.

Usage:
    python scripts/asapy_oracle_check.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from cuas_sim.analysis.features import feature_scores, pareto_front


def main() -> int:
    try:
        import asapy
    except ImportError:
        print("[SKIP] asapy not importable in this environment "
              "(install it in a separate oracle venv to run this check).")
        return 2

    ok = True

    rng = np.random.default_rng(7)
    n = 80
    x1 = rng.uniform(0.0, 10.0, n)
    x2 = rng.uniform(0.0, 10.0, n)
    y = 5.0 * x1 + 0.2 * x2 + rng.normal(0.0, 0.5, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    ours = [s.factor for s in feature_scores(df, response="y", factors=["x1", "x2"],
                                             scoring="f_regression")]
    theirs = list(
        asapy.Analysis.feature_score(df, ["x1", "x2"], "y", "f_regression")
        .sort_values("score", ascending=False)["feature"]
    )
    print(f"feature_score ranking: ours={ours} asapy={theirs} -> {'OK' if ours == theirs else 'MISMATCH'}")
    ok = ok and ours == theirs

    pts = pd.DataFrame({"gain": [0.30, 0.34, 0.20, 0.10], "cost": [1.0, 3.2, 0.5, 5.0]})
    our_front = pareto_front(pts, maximize=["gain"], minimize=["cost"])
    our_set = {
        (round(g, 6), round(c, 6))
        for g, c in zip(our_front["gain"], our_front["cost"], strict=True)
    }
    their_df = asapy.Analysis.pareto_front(pts, list_min=["cost"], list_max=["gain"])
    their_set = {
        (round(g, 6), round(c, 6))
        for g, c in zip(their_df["gain"], their_df["cost"], strict=True)
    }
    print(f"pareto set: ours={our_set} asapy={their_set} -> {'OK' if our_set == their_set else 'MISMATCH'}")
    ok = ok and our_set == their_set

    print("RESULT:", "agreement with AsaPy" if ok else "DIVERGENCE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
