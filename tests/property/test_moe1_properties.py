"""Property-based oracles for the MOE-1 <-> Eq. (3.5) reconciliation (Spec 025).

Group B (Red phase, Article III): these MUST FAIL until ``_compute_moe1`` is
rewritten (complement ``1 -`` + evasion penalty ``(1+E/N)`` + per-aircraft
``T_PEA`` + final clamp [0,1]) and ``clip_moe1_to_01`` flips to ``True``.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cuas_sim.config import MissionConfig
from cuas_sim.metrics.aggregator import MetricsAggregator

_TPEA = 8000.0


class _FixedTPea(MetricsAggregator):
    """Aggregator with a constant ``T_PEA`` so the MOE-1 *formula* is isolated.

    Subclassing (rather than monkeypatching) keeps the patch out of Hypothesis'
    function-scoped-fixture health check — the override is constant across all
    generated examples by construction.
    """

    def _t_pea_for_aor(self, aor_name: str, config_name: str | None = None) -> float:
        return _TPEA


def _moe1_all(df: pd.DataFrame) -> float:
    rows = _FixedTPea(cfg=MissionConfig())._compute_moe1(df, "baseline")
    return next(float(r["value"]) for r in rows
               if r["aor"] == "ALL" and r["metric"] == "MOE-1")


def _neut_df(t_efs: list[float]) -> pd.DataFrame:
    return pd.DataFrame([{
        "aor_name": "AoR-1", "config_name": "baseline", "is_neutralized": True,
        "moe1_engagement_time_s": v, "t_ev_s": float("nan"), "t_pursuit_start_s": 0.0,
    } for v in t_efs])


# ─── P1 — monotonia: engajamento mais rápido ⇒ MOE-1 não diminui ────────────


@given(
    t_efs=st.lists(st.floats(min_value=1.0, max_value=5000.0), min_size=1, max_size=8),
    factor=st.floats(min_value=0.1, max_value=1.0),
)
def test_monotonic_faster_higher(t_efs: list[float], factor: float) -> None:
    base = _moe1_all(_neut_df(t_efs))
    faster = _moe1_all(_neut_df([v * factor for v in t_efs]))  # all ≤ original
    assert faster >= base - 1e-9


# ─── P2 — penalidade (1+E/N)=2 num cenário all-evaded (penalidade ≥ 1) ───────


@given(ds=st.lists(st.floats(min_value=0.0, max_value=3000.0), min_size=1, max_size=8))
def test_penalty_all_evaded(ds: list[float]) -> None:
    tpea = _TPEA
    df = pd.DataFrame([{
        "aor_name": "AoR-1", "config_name": "baseline", "is_neutralized": False,
        "moe1_engagement_time_s": 12345.0,  # MUST be ignored — evaded uses t_ev
        "t_ev_s": d, "t_pursuit_start_s": 0.0,
    } for d in ds])
    actual = _moe1_all(df)
    # all-evaded ⇒ E = N ⇒ pen = 1 + N/N = 2.
    ref = 1.0 - sum(2.0 * d / tpea for d in ds) / len(ds)
    ref = max(0.0, min(1.0, ref))
    assert actual == pytest.approx(ref, abs=1e-9)


# ─── P3 — faixa clampada [0,1] ──────────────────────────────────────────────


@given(t_efs=st.lists(st.floats(min_value=0.0, max_value=40000.0), min_size=1, max_size=8))
def test_range_clamped(t_efs: list[float]) -> None:
    val = _moe1_all(_neut_df(t_efs))
    assert -1e-9 <= val <= 1.0 + 1e-9
