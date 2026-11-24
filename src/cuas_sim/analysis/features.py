"""Feature scoring + Pareto front.

Pure module: no ``RNGRegistry``, no I/O, no global state.
``scikit-learn`` is imported **lazily inside the functions** (optional ``[analysis]``
extra) so ``import cuas_sim.analysis`` works without it; ``mutual_info_regression``
is seeded for determinism. ``pareto_front`` is pure NumPy.

Mirrors ``asapy.analysis.feature_score`` / ``asapy.analysis.pareto_front``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cuas_sim.analysis.stats import DEFAULT_BOOTSTRAP_SEED

__all__ = ["FeatureScore", "feature_scores", "pareto_front"]

_INSTALL_HINT = (
    "scikit-learn is required for feature scoring; install the analysis extra: "
    'pip install -e ".[analysis]"'
)


@dataclass(frozen=True)
class FeatureScore:
    """One factor's score for a response."""

    factor: str
    score: float


def _score_func(scoring: str, seed: int) -> Any:
    try:
        from sklearn.feature_selection import (
            f_regression,
            mutual_info_regression,
            r_regression,
        )
    except ImportError as exc:  # pragma: no cover — exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc

    if scoring == "f_regression":
        return f_regression
    if scoring == "r_regression":
        return r_regression
    if scoring == "mutual_info_regression":
        return lambda x, y: mutual_info_regression(x, y, random_state=seed)
    raise ValueError(
        f"unknown scoring {scoring!r} (use f_regression/r_regression/mutual_info_regression)"
    )


def feature_scores(
    table: pd.DataFrame,
    *,
    response: str,
    factors: Sequence[str],
    scoring: str = "f_regression",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[FeatureScore]:
    """Rank ``factors`` by relevance to ``response``.

    ``scoring`` in {``f_regression``, ``r_regression``, ``mutual_info_regression``};
    ``mutual_info_regression`` uses ``random_state=seed`` (determinism).
    Returns scores sorted descending.
    """
    from sklearn.feature_selection import SelectKBest

    cols = list(factors)
    x = table[cols].to_numpy(dtype=float)
    y = table[response].to_numpy(dtype=float)
    selector = SelectKBest(score_func=_score_func(scoring, seed), k="all")
    selector.fit(x, y)
    scores = [
        FeatureScore(name, float(s))
        for name, s in zip(cols, selector.scores_, strict=True)
    ]
    scores.sort(key=lambda fs: fs.score, reverse=True)
    return scores


def pareto_front(
    points: pd.DataFrame,
    *,
    maximize: Sequence[str] = (),
    minimize: Sequence[str] = (),
) -> pd.DataFrame:
    """Non-dominated subset (gain x cost) by NumPy non-dominance.

    ``p`` is dominated by ``q`` iff ``q`` is >= ``p`` on every ``maximize`` column
    and <= ``p`` on every ``minimize`` column, with strict inequality somewhere.
    """
    hi = points[list(maximize)].to_numpy(dtype=float)
    lo = points[list(minimize)].to_numpy(dtype=float)
    n = len(points)

    keep: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            ge_max = bool(np.all(hi[j] >= hi[i]))
            le_min = bool(np.all(lo[j] <= lo[i]))
            strict = bool(np.any(hi[j] > hi[i])) or bool(np.any(lo[j] < lo[i]))
            if ge_max and le_min and strict:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return points.iloc[keep]
