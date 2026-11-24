"""Execution Control — chunked batch + early-stop heuristic.

Pure module. Mirrors ``asapy.execution_controller.ExecutionController``
and ``basic_stop`` minus the pause/resume/pickle machinery (not required by any FR).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from cuas_sim.optimization.loop_controller import _compute_ci

__all__ = ["ExecutionController", "convergence_stop", "non_stop_func"]

SimFunc = Callable[[pd.DataFrame], pd.DataFrame]
StopFunc = Callable[[pd.DataFrame, pd.DataFrame], bool]


class ExecutionController:
    """Run ``sim_func`` over a design in chunks; stop early if ``stop_func`` says so."""

    def __init__(
        self, sim_func: SimFunc, stop_func: StopFunc, chunk_size: int = 0,
    ) -> None:
        self._sim_func = sim_func
        self._stop_func = stop_func
        self._chunk_size = chunk_size

    def run(self, design: pd.DataFrame) -> pd.DataFrame:
        """Execute the design in chunks; return accumulated tidy rows.

        ``stop_func`` is consulted after each chunk with ``(current_chunk_result,
        accumulated_before_this_chunk)``; a truthy result stops early (mirrors
        ``asapy...ExecutionController.run``). ``chunk_size <= 0`` => single chunk.
        """
        n = len(design)
        if self._chunk_size and self._chunk_size > 0:
            chunks = [
                design.iloc[i:i + self._chunk_size]
                for i in range(0, n, self._chunk_size)
            ]
        else:
            chunks = [design]

        accumulated: pd.DataFrame | None = None
        for chunk in chunks:
            partial = self._sim_func(chunk)
            previous = accumulated if accumulated is not None else partial.iloc[0:0]
            stop = self._stop_func(partial, previous)
            accumulated = (
                partial if accumulated is None
                else pd.concat([accumulated, partial], ignore_index=True)
            )
            if stop:
                break

        if accumulated is None:
            return pd.DataFrame()
        return accumulated.reset_index(drop=True)


def non_stop_func(current: pd.DataFrame, all_previous: pd.DataFrame) -> bool:
    """Never stop early (the default)."""
    return False


def _target_values(df: pd.DataFrame, target: str) -> list[float]:
    if df is None or len(df) == 0:
        return []
    if target == "delta_mos1":
        sub = df[
            (df["config"] == "delta")
            & (df["metric"] == "MOS-1")
            & (df["aor"] == "ALL")
        ]
    else:
        sub = df[df["metric"] == target] if "metric" in df.columns else df
    return [float(v) for v in sub["value"]] if "value" in sub.columns else []


def _rel_change(prev: float, cur: float) -> float:
    return abs(cur - prev) / abs(prev) if prev != 0 else abs(cur - prev)


def convergence_stop(threshold: float, target: str = "delta_mos1") -> StopFunc:
    """Stop when relative change of mean AND std of ``target`` < ``threshold``.

    The target's confidence interval is computed via ``_compute_ci`` (reused);
    a degenerate/non-finite CI never forces a stop.
    """

    def _stop(current: pd.DataFrame, all_previous: pd.DataFrame) -> bool:
        prev = _target_values(all_previous, target)
        if len(prev) < 2:
            return False
        cur = _target_values(
            pd.concat([all_previous, current], ignore_index=True), target
        )
        if len(cur) < 2:
            return False
        rel_mean = _rel_change(float(np.mean(prev)), float(np.mean(cur)))
        rel_std = _rel_change(
            float(np.std(prev, ddof=1)), float(np.std(cur, ddof=1))
        )
        ci_lo, ci_hi = _compute_ci(cur)
        ci_ok = math.isfinite(ci_lo) and math.isfinite(ci_hi)
        return ci_ok and rel_mean < threshold and rel_std < threshold

    return _stop
