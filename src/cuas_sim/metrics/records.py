"""Per-run, per-event log container.

The :class:`RunLogger` accumulates two streams in memory while the simulation
runs (engagements + events) and serializes them to Parquet + JSONL.gz at the
end. Memory pressure: at λ = 6 UAS/h over a 5 h mission with 6 A-29s the
event stream is well under 100k lines, so a list-of-dicts is fine.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cuas_sim.engagement.controller import EngagementOutcome
    from cuas_sim.world.arrivals import UASBirth


@dataclass(slots=True)
class StateChangeEvent:
    """A single A-29 state transition (or notable info-level event)."""

    config: str
    a29: str
    t_s: float
    old: str
    new: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps({
            "config": self.config, "a29": self.a29, "t_s": self.t_s,
            "old": self.old, "new": self.new, **self.extra,
        }) + "\n"


@dataclass(slots=True)
class EngagementRecord:
    """Flat dictionary view of an :class:`EngagementOutcome`."""

    fields_: dict[str, Any]

    @classmethod
    def from_outcome(cls, outcome: EngagementOutcome) -> EngagementRecord:
        moe2_samples = outcome.moe2_samples
        moe2_mean = sum(moe2_samples) / len(moe2_samples) if moe2_samples else 0.0
        return cls(fields_={
            "config_name": outcome.config_name,
            "a29_id": outcome.a29_id,
            "uas_id": outcome.uas_id,
            "aor_name": outcome.aor_name,
            "sector_index": outcome.sector_index,
            "archetype_name": outcome.archetype_name,
            "t_pursuit_start_s": outcome.t_pursuit_start_s,
            "t_envelope_entry_s": outcome.t_envelope_entry_s,
            "t_fire_s": outcome.t_fire_s,
            "t_kill_s": outcome.t_kill_s,
            "initial_distance_m": outcome.initial_distance_m,
            "intercept_distance_m": outcome.intercept_distance_m,
            "firing_distance_m": outcome.firing_distance_m,
            "is_neutralized": outcome.is_neutralized,
            "is_neutralized_before_zida": outcome.is_neutralized_before_zida,
            "moe1_engagement_time_s": outcome.moe1_engagement_time_s,
            # Evasion instant (80 nmi crossing) for MOE-1, Eq. (3.6).
            "t_ev_s": outcome.t_ev_s,
            "moe2_mean_per_engagement": moe2_mean,
            "moe2_n_samples": len(moe2_samples),
            "moe3_p_hit_expected": outcome.moe3_p_hit_expected,
            "mop1_heading_update_count": outcome.mop1_heading_update_count,
            "mop2_acquisition_range_m": outcome.mop2_acquisition_range_m,
            "mop3_envelope_max_m": outcome.mop3_envelope_max_m,
            # Multi-shot diagnostics (internal run-log only;
            # no external-schema change → no field_version bump).
            "n_shots": outcome.n_shots,
            "rockets_remaining": outcome.rockets_remaining,
            "feasible": outcome.feasible,
            "note": outcome.note,
        })


@dataclass(slots=True)
class RunLogger:
    """Aggregates engagements, state changes, leaks, and UAS births for one paired run."""

    engagements: list[EngagementRecord] = field(default_factory=list)
    events: dict[str, list[StateChangeEvent]] = field(default_factory=dict)
    leaks: list[dict[str, Any]] = field(default_factory=list)
    uas_births: list[dict[str, Any]] = field(default_factory=list)
    # Instant of RTB_FUEL_LIMIT per (config, a29_id) for
    # ``_t_pea_for_aor`` (emergent T_PEA) and ``window.json``.
    t_rtb_fuel_limit_per_a29: dict[tuple[str, str], float] = field(default_factory=dict)
    # Payload of ``window.json`` (filled by
    # ``MissionSimulator.run_paired`` and serialized by ``write_window_json``).
    window: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for cfg in ("baseline", "modernized"):
            self.events.setdefault(cfg, [])

    # ── Mutators ─────────────────────────────────────────────────────────

    def log_state(self, *, config: str, a29: str, t_s: float, old: str, new: str) -> None:
        self.events.setdefault(config, []).append(
            StateChangeEvent(config=config, a29=a29, t_s=t_s, old=old, new=new)
        )

    def log_engagement(self, outcome: EngagementOutcome) -> None:
        self.engagements.append(EngagementRecord.from_outcome(outcome))

    def log_uas_birth(
        self,
        uas_id: int,
        spawn_time_s: float,
        lat: float,
        lon: float,
        heading_rad: float,
        speed_mps: float,
        archetype_name: str,
        sector_index: int,
    ) -> None:
        self.uas_births.append({
            "uas_id": uas_id,
            "spawn_time_s": spawn_time_s,
            "lat": lat,
            "lon": lon,
            "heading_rad": heading_rad,
            "speed_mps": speed_mps,
            "archetype_name": archetype_name,
            "sector_index": sector_index,
        })

    def log_uas_leak(self, config: str, birth: UASBirth, reason: str) -> None:
        self.leaks.append({
            "config_name": config,
            "uas_id": birth.uas_id,
            "spawn_time_s": birth.spawn_time_s,
            "sector_index": birth.sector_index,
            "archetype_name": birth.archetype.name,
            "reason": reason,
        })

    def log_rtb_fuel_limit(
        self,
        config_name: str,
        a29_id: str,
        t_s: float,
        pos_xy: tuple[float, float],
        t_pea_actual_s: float,
    ) -> None:
        """Record an RTB_FUEL_LIMIT instant for the (config, a29_id) pair.

        Stores ``t_s`` in :attr:`t_rtb_fuel_limit_per_a29`
        and emits a structured diagnostic event in
        ``events_<config>.jsonl.gz`` with the abort position and the actual
        PEA duration. The new-state transition itself (``→ RTB_FUEL_LIMIT``)
        is logged separately by :meth:`A29._set_state`.
        """
        # Idempotent record: first RTB wins (subsequent calls for the same
        # (config, a29_id) are silently ignored — an A-29 only RTBs once
        # per sortie).
        key = (config_name, a29_id)
        if key not in self.t_rtb_fuel_limit_per_a29:
            self.t_rtb_fuel_limit_per_a29[key] = float(t_s)
            self.events.setdefault(config_name, []).append(
                StateChangeEvent(
                    config=config_name, a29=a29_id, t_s=float(t_s),
                    old="RTB_FUEL_LIMIT_INFO", new="RTB_FUEL_LIMIT_INFO",
                    extra={
                        "pos_xy": [float(pos_xy[0]), float(pos_xy[1])],
                        "t_pea_actual_s": float(t_pea_actual_s),
                    },
                )
            )

    def set_window(self, payload: dict[str, Any]) -> None:
        """Store the ``window.json`` payload for later serialization."""
        self.window = dict(payload)

    def write_window_json(self, path: Path) -> None:
        """Write ``window.json`` (schema v1.0).

        Falls back to an empty schema-v1.0 shell if :meth:`set_window` was
        never called (e.g. unit tests). The 9 required keys are always
        present in the output.
        """
        empty: dict[str, Any] = {
            "field_version": "1.0",
            "uas_spawn_window_enabled": False,
            "t_window_start_s": 0.0,
            "t_window_end_s": 0.0,
            "t_window_end_reason": "legacy_horizon",
            "horizon_cap_s": 0.0,
            "baseline_first_bingo": None,
            "modernized_first_bingo": None,
            "t_rtb_fuel_limit_per_a29": {"baseline": {}, "modernized": {}},
        }
        payload = {**empty, **self.window} if self.window else empty
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ── Serialisers ──────────────────────────────────────────────────────

    def write_engagements_parquet(self, path: Path) -> None:
        # Lazy import to keep pandas optional at module-load time.
        import pandas as pd

        if not self.engagements:
            df = pd.DataFrame(columns=list(_EMPTY_ENGAGEMENT_COLUMNS))
        else:
            df = pd.DataFrame([e.fields_ for e in self.engagements])
        df.to_parquet(path, engine="pyarrow", index=False)

    def write_events_jsonl_gz(self, path: Path, config: str) -> None:
        # Deterministic gzip output across
        # Python sessions: zero out the mtime field in the gzip header (and
        # disable the filename so the OS-time/name don't leak into hashes).
        with (
            open(path, "wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
            io.TextIOWrapper(gz, encoding="utf-8", write_through=True) as fh,
        ):
            for ev in self.events.get(config, []):
                fh.write(ev.to_jsonl())

    def write_spawns_parquet(self, path: Path) -> None:
        import pandas as pd

        if not self.uas_births:
            df = pd.DataFrame(columns=[
                "uas_id", "spawn_time_s", "lat", "lon",
                "heading_rad", "speed_mps", "archetype_name", "sector_index",
            ])
        else:
            df = pd.DataFrame(self.uas_births)
        df.to_parquet(path, engine="pyarrow", index=False)

    def write_leaks_parquet(self, path: Path) -> None:
        import pandas as pd

        if not self.leaks:
            df = pd.DataFrame(columns=[
                "config_name", "uas_id", "spawn_time_s",
                "sector_index", "archetype_name", "reason",
            ])
        else:
            df = pd.DataFrame(self.leaks)
        df.to_parquet(path, engine="pyarrow", index=False)


_EMPTY_ENGAGEMENT_COLUMNS = (
    "config_name", "a29_id", "uas_id", "aor_name", "sector_index", "archetype_name",
    "t_pursuit_start_s", "t_envelope_entry_s", "t_fire_s", "t_kill_s",
    "initial_distance_m", "intercept_distance_m", "firing_distance_m",
    "is_neutralized", "is_neutralized_before_zida",
    "moe1_engagement_time_s", "t_ev_s", "moe2_mean_per_engagement", "moe2_n_samples",
    "moe3_p_hit_expected",
    "mop1_heading_update_count", "mop2_acquisition_range_m", "mop3_envelope_max_m",
    "n_shots", "rockets_remaining",
    "feasible", "note",
)
