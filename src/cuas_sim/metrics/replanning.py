"""Replanning signal — closes the OPT ↔ SIM loop.

If the modernized configuration of a paired run fails the acceptance
criterion ``MOS-1 ≥ MOS-1_target`` (default target 0.80),
the simulator emits ``replanning_signal.json`` with the per-AoR and
per-sector leak distribution plus a textual hint summarizing where the
mission is hemorrhaging. The upstream Set Covering Problem (Szvaticsek et
al., 2025) consumes this signal and re-optimizes the FARP/CAP placement.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.metrics.records import RunLogger
    from cuas_sim.world.aor import AoRMap


@dataclass(slots=True)
class ReplanningSignal:
    """Payload emitted when MOS-1 < target for the modernized configuration."""

    triggered: bool
    mos1_modernized: float
    mos1_target: float
    leaks_per_aor: dict[str, int]
    leaks_per_sector: dict[int, int]
    leak_share_per_aor: dict[str, float]
    hint: str
    field_version: str = "1.0"
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_version": self.field_version,
            "triggered": self.triggered,
            "mos1_modernized": self.mos1_modernized,
            "mos1_target": self.mos1_target,
            "leaks_per_aor": dict(self.leaks_per_aor),
            "leaks_per_sector": {str(k): v for k, v in self.leaks_per_sector.items()},
            "leak_share_per_aor": dict(self.leak_share_per_aor),
            "hint": self.hint,
            "metadata": dict(self.metadata),
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_replanning_signal(
    cfg: MissionConfig,
    metrics_df: pd.DataFrame,
    log: RunLogger,
    mos1_target: float = 0.80,
) -> ReplanningSignal:
    """Construct the replanning signal from the aggregated metrics."""
    # MOS-1 for the modernized config, overall (ALL)
    mos1_mod = 0.0
    if not metrics_df.empty:
        row = metrics_df[
            (metrics_df["config_name"] == "modernized")
            & (metrics_df["metric"] == "MOS-1")
            & (metrics_df["aor"] == "ALL")
        ]
        if not row.empty:
            mos1_mod = float(row["value"].iloc[0])

    triggered = mos1_mod < mos1_target

    # Leak distribution for the modernized config: every engagement with
    # is_neutralized_before_zida == False, plus every unhandled leak.
    leaks_per_aor: Counter[str] = Counter()
    leaks_per_sector: Counter[int] = Counter()
    for e in log.engagements:
        if e.fields_["config_name"] != "modernized":
            continue
        if not e.fields_["is_neutralized_before_zida"]:
            leaks_per_aor[e.fields_["aor_name"]] += 1
            leaks_per_sector[e.fields_["sector_index"]] += 1
    for leak in log.leaks:
        if leak["config_name"] != "modernized":
            continue
        sec = leak["sector_index"]
        leaks_per_sector[sec] += 1
        for aor_name, secs in cfg.aor_to_sectors.items():
            if sec in secs:
                leaks_per_aor[aor_name] += 1
                break

    total = sum(leaks_per_aor.values())
    share = {a: c / total for a, c in leaks_per_aor.items()} if total > 0 else {}

    hint = _build_hint(triggered, share, leaks_per_sector, cfg)

    return ReplanningSignal(
        triggered=triggered,
        mos1_modernized=mos1_mod,
        mos1_target=mos1_target,
        leaks_per_aor=dict(leaks_per_aor),
        leaks_per_sector=dict(leaks_per_sector),
        leak_share_per_aor=share,
        hint=hint,
        metadata={"total_modernized_leaks": int(total)},
    )


def _build_hint(
    triggered: bool,
    share: dict[str, float],
    leaks_per_sector: Counter[int],
    cfg: MissionConfig,
) -> str:
    if not triggered:
        return "MOS-1 modernized meets target — no replanning required."
    if not share:
        return ("MOS-1 modernized below target but no per-AoR leaks recorded "
                "(potential cause: very low arrival rate). Increase λ or "
                "extend horizon.")
    worst_aor = max(share, key=share.__getitem__)
    worst_share = share[worst_aor]
    secs = [s for s in leaks_per_sector if s in cfg.aor_to_sectors.get(worst_aor, [])]
    secs_top = sorted(secs, key=lambda s: leaks_per_sector[s], reverse=True)[:3]
    sec_text = ", ".join(f"S{s}({leaks_per_sector[s]})" for s in secs_top)
    return (
        f"{worst_aor} carries {worst_share:.0%} of modernized leaks "
        f"(hot sectors: {sec_text}). Consider re-locating "
        f"{cfg.aor_to_a29.get(worst_aor, 'the assigned A-29')}'s CAP/FARP "
        f"closer to that band, or splitting the AoR with an extra airframe."
    )


# ─── v2.0 types ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ReplicationMetrics:
    """Per-replication MOS-1 summary for the OPT↔SIM loop."""

    seed: int
    mos1_overall: float
    mos1_by_aor: dict[str, float]


@dataclass(slots=True)
class ReplanningSignalV2:
    """v2.0 replanning signal payload.

    Extends v1.0 with CI bounds, per-AoR MOS-1, active FARP/CAP state, and
    iteration metadata required by the OPT↔SIM loop controller.
    ``field_version = "2.0"`` is always the first JSON key.
    """

    # v1.0 fields (preserved for backward compatibility)
    triggered: bool
    mos1_modernized: float
    mos1_target: float
    leaks_per_aor: dict[str, int]
    leaks_per_sector: dict[int, int]
    leak_share_per_aor: dict[str, float]
    advisory_hint: str
    metadata: dict[str, str | int | float]
    field_version: str = "2.0"

    # v2.0 fields
    mos1_by_aor: dict[str, float] = field(default_factory=dict)
    ci_95_mos1_lower: float = float("nan")
    ci_95_mos1_upper: float = float("nan")
    cap_positions: dict[str, dict[str, float]] = field(default_factory=dict)
    active_farps: list[str] = field(default_factory=list)
    iteration: int = 0
    replication_count: int = 0
    seed_set: list[int] = field(default_factory=list)
    farp_candidate_distances_km: dict[str, float] = field(default_factory=dict)
    lambda_per_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        # field_version MUST be first key
        return {
            "field_version": self.field_version,
            "triggered": self.triggered,
            "mos1_modernized": self.mos1_modernized,
            "mos1_target": self.mos1_target,
            "leaks_per_aor": dict(self.leaks_per_aor),
            "leaks_per_sector": {str(k): v for k, v in self.leaks_per_sector.items()},
            "leak_share_per_aor": dict(self.leak_share_per_aor),
            "_advisory_hint": self.advisory_hint,
            "metadata": dict(self.metadata),
            "mos1_by_aor": dict(self.mos1_by_aor),
            "ci_95_mos1_lower": self.ci_95_mos1_lower,
            "ci_95_mos1_upper": self.ci_95_mos1_upper,
            "cap_positions": {k: dict(v) for k, v in self.cap_positions.items()},
            "active_farps": list(self.active_farps),
            "iteration": self.iteration,
            "replication_count": self.replication_count,
            "seed_set": list(self.seed_set),
            "farp_candidate_distances_km": dict(self.farp_candidate_distances_km),
            "lambda_per_sec": self.lambda_per_sec,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_replanning_signal_v2(
    mos1_modernized: float,
    mos1_target: float,
    leaks: list[dict[str, Any]],
    aor_map: AoRMap,
    replication_results: list[ReplicationMetrics],
    iteration: int,
    config: MissionConfig,
    seed_set: list[int],
    farp_distances_km: dict[str, float],
) -> ReplanningSignalV2:
    """Build the v2.0 replanning signal from N Monte Carlo replications.

    ``mos1_modernized`` is the mean across all ``replication_results``.
    CI bounds are computed via Student-t (same replication batch).
    ``leaks`` should be pre-filtered or contain only modernized-config entries.
    """
    triggered = mos1_modernized < mos1_target

    # ── Leak distribution (per AoR and sector) ──────────────────────────────
    leaks_per_aor: Counter[str] = Counter()
    leaks_per_sector: Counter[int] = Counter()
    for lk in leaks:
        if lk.get("config_name", "modernized") != "modernized":
            continue
        sec = int(lk["sector_index"])
        leaks_per_sector[sec] += 1
        aor_name = aor_map.sector_to_aor.get(sec)
        if aor_name:
            leaks_per_aor[aor_name] += 1

    total = sum(leaks_per_aor.values())
    share: dict[str, float] = (
        {a: c / total for a, c in leaks_per_aor.items()} if total > 0 else {}
    )

    if not share:
        advisory = "No leaks detected." if not triggered else (
            "MOS-1 below target but no per-AoR leaks recorded."
        )
    else:
        worst_aor = max(share, key=share.__getitem__)
        advisory = (
            f"{worst_aor} carries {share[worst_aor]:.0%} of modernized leaks "
            f"(iteration {iteration}). Consider re-positioning FARPs/CAPs in that AoR."
        )

    # ── CI from replication batch ──────────────────────────
    n_rep = len(replication_results)
    if n_rep >= 2:
        from scipy.stats import t as t_dist
        mos1_values = [r.mos1_overall for r in replication_results]
        mean_v = sum(mos1_values) / n_rep
        variance = sum((x - mean_v) ** 2 for x in mos1_values) / (n_rep - 1)
        se = math.sqrt(variance / n_rep)
        t_crit = float(t_dist.ppf(0.975, df=n_rep - 1))
        ci_lower = mean_v - t_crit * se
        ci_upper = mean_v + t_crit * se
    else:
        ci_lower = ci_upper = float("nan")

    # ── Per-AoR MOS-1 (average across replications) ─────────────────────────
    mos1_by_aor: dict[str, float] = {}
    if replication_results:
        all_aors = set(aor_map.aor_to_sectors.keys())
        for aor in sorted(all_aors):
            vals = [r.mos1_by_aor.get(aor, 1.0) for r in replication_results]
            mos1_by_aor[aor] = sum(vals) / len(vals)

    # ── Active FARP/CAP state from config ───────────────────────────────────
    active_farps = sorted(config.farps.keys())
    cap_positions = {
        f"CAP-{sched.cap_index}": {
            "lat": config.caps[f"CAP-{sched.cap_index}"].lat,
            "lon": config.caps[f"CAP-{sched.cap_index}"].lon,
        }
        for sched in config.a29_schedules.values()
        if f"CAP-{sched.cap_index}" in config.caps
    }

    return ReplanningSignalV2(
        triggered=triggered,
        mos1_modernized=mos1_modernized,
        mos1_target=mos1_target,
        leaks_per_aor=dict(leaks_per_aor),
        leaks_per_sector=dict(leaks_per_sector),
        leak_share_per_aor=share,
        advisory_hint=advisory,
        metadata={"total_modernized_leaks": int(total)},
        field_version="2.0",
        mos1_by_aor=mos1_by_aor,
        ci_95_mos1_lower=ci_lower,
        ci_95_mos1_upper=ci_upper,
        cap_positions=cap_positions,
        active_farps=active_farps,
        iteration=iteration,
        replication_count=n_rep,
        seed_set=list(seed_set),
        farp_candidate_distances_km=dict(farp_distances_km),
        lambda_per_sec=config.lambda_per_sec,
    )
