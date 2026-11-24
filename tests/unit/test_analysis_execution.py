"""Unit — AN-1 ExecutionController (Spec 026 §5.1 T3). Red phase first (Article III)."""

from __future__ import annotations

import pandas as pd

from cuas_sim.analysis.execution import (
    ExecutionController,
    convergence_stop,
    non_stop_func,
)


def _delta_df(values: list[float]) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        "config": ["delta"] * n, "metric": ["MOS-1"] * n,
        "aor": ["ALL"] * n, "value": values,
    })


def _design(n: int) -> pd.DataFrame:
    return pd.DataFrame({"design_point_id": range(n)})


def _identity_sim(chunk: pd.DataFrame) -> pd.DataFrame:
    """One tidy row per design point in the chunk."""
    return chunk.assign(value=1.0)


def test_chunking_and_early_stop() -> None:
    design = _design(10)

    def stop_after_first(current: pd.DataFrame, all_previous: pd.DataFrame) -> bool:
        return True  # stop after the first chunk

    out = ExecutionController(_identity_sim, stop_after_first, chunk_size=3).run(design)
    assert len(out) == 3  # only the first chunk processed

    out_all = ExecutionController(_identity_sim, non_stop_func, chunk_size=3).run(design)
    assert len(out_all) == 10  # every chunk processed

    out_single = ExecutionController(_identity_sim, non_stop_func, chunk_size=0).run(design)
    assert len(out_single) == 10  # chunk_size=0 -> single chunk


def test_convergence_stop_triggers_on_stable_target() -> None:
    stop = convergence_stop(threshold=0.05, target="delta_mos1")
    prev = _delta_df([0.20, 0.20, 0.20, 0.20])
    current = _delta_df([0.20, 0.20])
    assert stop(current, prev) is True
    # Fewer than 2 accumulated points -> never stop.
    assert stop(_delta_df([0.2]), _delta_df([0.2])) is False


def test_convergence_stop_holds_on_moving_target() -> None:
    stop = convergence_stop(threshold=0.01, target="delta_mos1")
    prev = _delta_df([0.10, 0.10, 0.10])
    current = _delta_df([0.50, 0.50, 0.50])  # large jump -> not converged
    assert stop(current, prev) is False
