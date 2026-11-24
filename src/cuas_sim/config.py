"""Frozen MissionConfig — the single source of truth for the simulation.

The simulator never reads parameters from anywhere other than a
:class:`MissionConfig` instance. Override precedence when the model is loaded
(:meth:`MissionConfig.from_yaml_with_sysml`):

1. **SysML v2 model** under ``sysml/`` (auto-extracted, highest priority: its
   overrides are applied on top of the YAML, in the fields it supplies)
2. **YAML** file (user-edited, medium priority)
3. **Python defaults** in this module (lowest priority)

Every override is logged at INFO level via :mod:`cuas_sim.sysml_loader`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from cuas_sim.geometry.coordinates import GeoPoint

logger = logging.getLogger(__name__)


# ─── Sub-records ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class A29Schedule:
    """Pre-PEA and post-PEA durations from case study Table 3.2 (per aircraft).

    Durations are in seconds.
    """

    farp_name: str
    cap_index: int
    phase1_sbpv_to_farp_s: float
    phase2_farp_to_cap_s: float
    phase3_holding_s: float
    phase4_pea_s: float
    phase5_cap_to_farp_s: float
    phase6_farp_to_sbpv_s: float

    @property
    def t_cap_arrival_s(self) -> float:
        """Cumulative time at which the A-29 arrives at the CAP (s)."""
        return self.phase1_sbpv_to_farp_s + self.phase2_farp_to_cap_s

    @property
    def t_pea_end_s(self) -> float:
        """Mission time at which Phase 4 (PEA) ends for this A-29 (s).

        Anchored at the synchronized T+153 min start.
        """
        return T_PEA_START_DEFAULT_S + self.phase4_pea_s


@dataclass(frozen=True, slots=True)
class UASArchetypeSpec:
    """Kinematic spec of a UAS archetype (case study Tables 11–12)."""

    name: str
    max_speed_mps: float
    dimension_m: float
    ceiling_m: float


@dataclass(frozen=True, slots=True)
class BorderSegment:
    """One 100-km segment of the Brazilian dry-border generating polygon.

    Per Miranda (2025), the full polygon has 92 segments numbered 1–92. This
    study restricts to a 27-sector strip along the BR-CO/BR-PE border, numbered
    1–27 from north to south. Each segment is a directed edge from ``start`` to
    ``end``.

    ``arc`` is the exact 100-km border polyline with cumulative
    curvilinear abscissa: a tuple of ``(lat_deg, lon_deg, s_km)`` vertices,
    ``s_km`` measured from ``start``. Empty by default (``()``) — a segment
    without an arc falls back to chord interpolation.
    """

    index: int
    start: GeoPoint
    end: GeoPoint
    arc: tuple[tuple[float, float, float], ...] = ()


# ─── Default constants from the case study ────────────────────────────────

#: Mission time at which Phase 4 (PEA) synchronously starts for all A-29s.
#: Case study Section 3.3.3.4: T+153 min = 9 180 s.
T_PEA_START_DEFAULT_S: float = 153.0 * 60.0


def _default_main_base() -> GeoPoint:
    """SBPV / Porto Velho (RO) — Table A.1 of the case study."""
    return GeoPoint.from_dms("8°42'59.61\"S", "63°54'7.46\"O", name="SBPV")


def _default_farps() -> dict[str, GeoPoint]:
    """FARP coordinates from case study Table A.1."""
    return {
        "SBCZ": GeoPoint.from_dms("7°35'59.86\"S", "72°46'23.12\"O", name="SBCZ"),
        "SWKU": GeoPoint.from_dms("1°7'11.98\"N", "66°50'29.59\"O", name="SWKU"),
        "SDCG": GeoPoint.from_dms("3°28'9.00\"S", "68°55'16.00\"O", name="SDCG"),
        "SBRB": GeoPoint.from_dms("9°52'12.82\"S", "67°53'44.23\"O", name="SBRB"),
    }


def _default_caps() -> dict[str, GeoPoint]:
    """CAP coordinates from case study Table A.1.

    The case study renumbers Szvaticsek (2025)'s 16 ZIDA surveillance points
    down to six CAP points (numbered 1–6) supported by four FARPs per Figure 3.3.
    """
    return {
        "CAP-1": GeoPoint.from_dms("10°18'1.36\"S", "69°56'9.04\"O", name="CAP-1"),
        "CAP-2": GeoPoint.from_dms("8°42'11.51\"S", "72°8'25.17\"O", name="CAP-2"),
        "CAP-3": GeoPoint.from_dms("5°59'49.82\"S", "72°23'35.12\"O", name="CAP-3"),
        "CAP-4": GeoPoint.from_dms("4°41'29.38\"S", "70°1'15.70\"O", name="CAP-4"),
        "CAP-5": GeoPoint.from_dms("2°12'46.78\"S", "68°55'38.91\"O", name="CAP-5"),
        "CAP-6": GeoPoint.from_dms("0°25'53.75\"N", "68°18'47.86\"O", name="CAP-6"),
    }


def _default_a29_schedules() -> dict[str, A29Schedule]:
    """Per-aircraft schedule from case study Table 3.2.

    Durations are in seconds. ``cap_index`` follows the case study numbering
    (1-based).
    """
    M = 60.0
    return {
        "A-29-1": A29Schedule(
            farp_name="SBRB", cap_index=1,
            phase1_sbpv_to_farp_s=53 * M, phase2_farp_to_cap_s=26 * M,
            phase3_holding_s=74 * M, phase4_pea_s=(3 * 60 + 5) * M,
            phase5_cap_to_farp_s=26 * M, phase6_farp_to_sbpv_s=53 * M,
        ),
        "A-29-2": A29Schedule(
            farp_name="SBCZ", cap_index=2,
            phase1_sbpv_to_farp_s=114 * M, phase2_farp_to_cap_s=16 * M,
            phase3_holding_s=23 * M, phase4_pea_s=(4 * 60 + 16) * M,
            phase5_cap_to_farp_s=16 * M, phase6_farp_to_sbpv_s=114 * M,
        ),
        "A-29-3": A29Schedule(
            farp_name="SBCZ", cap_index=3,
            phase1_sbpv_to_farp_s=114 * M, phase2_farp_to_cap_s=21 * M,
            phase3_holding_s=18 * M, phase4_pea_s=(4 * 60 + 11) * M,
            phase5_cap_to_farp_s=21 * M, phase6_farp_to_sbpv_s=114 * M,
        ),
        "A-29-4": A29Schedule(
            farp_name="SDCG", cap_index=4,
            phase1_sbpv_to_farp_s=93 * M, phase2_farp_to_cap_s=21 * M,
            phase3_holding_s=39 * M, phase4_pea_s=(3 * 60 + 51) * M,
            phase5_cap_to_farp_s=21 * M, phase6_farp_to_sbpv_s=93 * M,
        ),
        "A-29-5": A29Schedule(
            farp_name="SDCG", cap_index=5,
            phase1_sbpv_to_farp_s=93 * M, phase2_farp_to_cap_s=16 * M,
            phase3_holding_s=44 * M, phase4_pea_s=(3 * 60 + 56) * M,
            phase5_cap_to_farp_s=16 * M, phase6_farp_to_sbpv_s=93 * M,
        ),
        "A-29-6": A29Schedule(
            farp_name="SWKU", cap_index=6,
            phase1_sbpv_to_farp_s=132 * M, phase2_farp_to_cap_s=21 * M,
            phase3_holding_s=0 * M, phase4_pea_s=(4 * 60 + 30) * M,
            phase5_cap_to_farp_s=21 * M, phase6_farp_to_sbpv_s=132 * M,
        ),
    }


def _default_aor_to_sectors() -> dict[str, list[int]]:
    """Default contiguous sector → AoR mapping (sectors 1–27, 6 AoRs).

    Geographically consistent with the N→S polygon (sector 1 at
    the north, sector 27 at the south). CAP-6 is northernmost (+0.43°N, sectors
    1-7); CAP-1 is southernmost (-10.3°S, sectors 22-27). Matches the production
    ``configs/zida_shield_default.yaml`` block. Applies only when MissionConfig
    is constructed without a YAML file (test path).
    """
    return {
        "AoR-6": [1, 2, 3, 4, 5, 6, 7],
        "AoR-5": [8, 9, 10],
        "AoR-4": [11, 12, 13, 14, 15],
        "AoR-3": [16, 17],
        "AoR-2": [18, 19, 20, 21],
        "AoR-1": [22, 23, 24, 25, 26, 27],
    }


# ─── MissionConfig ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MissionConfig:
    """All simulation parameters in a single immutable record.

    Values come from (in decreasing precedence) the SysML v2 model under
    ``sysml/`` (in the fields it supplies), the user YAML, and the defaults
    declared in this module.
    """

    # ─── Geographic ────────────────────────────────────────────────────────
    aoo_centroid_lat: float = -5.0
    aoo_centroid_lon: float = -68.0
    main_base: GeoPoint = field(default_factory=_default_main_base)
    farps: dict[str, GeoPoint] = field(default_factory=_default_farps)
    caps: dict[str, GeoPoint] = field(default_factory=_default_caps)
    zida_boundary_nmi: float = 80.0

    # ─── Border polygon (Miranda 2025) ─────────────────────────────────────
    border_sectors: tuple[BorderSegment, ...] | None = None  # populated by loader
    sector_min: int = 1
    sector_max: int = 27
    sector_mu: float = 14.0
    sector_sigma: float = 4.3333
    heading_perpendicular_deviation_deg: float = 15.0
    # ─── UAS generation rules (Miranda 2025) ──────────────────
    #: Curve ω runs along for the spawn point: "arc" (border arc) or
    #: "chord" (straight vertex-to-vertex, legacy).
    spawn_interpolation: str = "arc"
    #: Sector draw: "cdf" (inverse-CDF + ±3σ rejection) or "pdf" (legacy
    #: renormalised discrete normal).
    sector_draw_rule: str = "cdf"
    #: Inward perpendicular: "literal" (az−90°, as in Miranda) or "east_flip"
    #: (legacy east-component heuristic).
    heading_inward_rule: str = "literal"

    # ─── AoR mapping ───────────────────────────────────────────────────────
    aor_to_sectors: dict[str, list[int]] = field(default_factory=_default_aor_to_sectors)
    #: AoR name → assigned A-29 designator. Default contiguous (AoR-k → A-29-k).
    aor_to_a29: dict[str, str] = field(
        default_factory=lambda: {f"AoR-{i}": f"A-29-{i}" for i in range(1, 7)}
    )

    # ─── Aircraft (airframe-common) ────────────────────────────────────────
    a29_cruise_speed_mps: float = 144.0
    a29_endurance_s: float = 5.2 * 3600.0
    a29_schedules: dict[str, A29Schedule] = field(default_factory=_default_a29_schedules)

    # ─── UAS archetypes ────────────────────────────────────────────────────
    uas_mr: UASArchetypeSpec = field(
        default_factory=lambda: UASArchetypeSpec(
            name="multi-rotor", max_speed_mps=29.0, dimension_m=1.0, ceiling_m=6_000.0
        )
    )
    uas_fw: UASArchetypeSpec = field(
        default_factory=lambda: UASArchetypeSpec(
            name="fixed-wing", max_speed_mps=130.0, dimension_m=14.0, ceiling_m=18_000.0
        )
    )
    uas_archetype_p_mr: float = 0.5

    # ─── Arrival process ───────────────────────────────────────────────────
    lambda_per_sec: float = 1.0 / 600.0  # 1 UAS / 10 min

    # ─── Tracking — baseline (Eqs. 3.8 and 3.12) ────────────────────────
    baseline_update_interval_s: tuple[float, float] = (5.0, 60.0)
    baseline_max_acquisition_m: float = 13_000.0
    baseline_dim_ref_m: float = 14.0  # deprecated (earlier tracking formula)
    # Table 3.3 / MOP-2 — PilotVisualAcquisitionSystemNakedEye.minAcquisitionAndReactionTime
    baseline_reaction_time_min_s: float = 12.5
    # Naked-eye angular acuity threshold (1 arcmin = π/(180·60) rad)
    baseline_visual_acuity_rad: float = 2.9088820866572160e-4

    # ─── Tracking — modernized (Eqs. 3.9 and 3.13) ────────────────────────
    modernized_update_interval_s: tuple[float, float] = (1.0, 10.0)
    modernized_eoir_min_m: float = 14_000.0
    modernized_eoir_max_m: float = 20_000.0
    modernized_eoir_accuracy_m: float = 5.0   # a_EO/IR
    # deprecated: the value derives from min(1, L_U/σ_s); kept for YAML compatibility
    modernized_track_p_fw: float = 1.0
    # deprecated: the value derives from min(1, L_U/σ_s); kept for YAML compatibility
    modernized_track_p_mr: float = 0.2

    # ─── Weapon ───────────────────────────────────────────────────────────
    baseline_envelope_m: tuple[float, float] = (500.0, 2_000.0)
    modernized_envelope_m: tuple[float, float] = (2_000.0, 14_000.0)
    p_hit_at_min_range: float = 0.80   # 0.80 in Eq. (3.14)
    dimensional_penalty_mr: float = 0.5   # M for multi-rotor, Table 3.3
    dimensional_penalty_fw: float = 1.0   # M for fixed-wing, Table 3.3
    # Re-engagement model. ``reengagement_interval_s`` is deprecated
    # (kept for YAML back-compat); the inter-shot interval is
    # now ``Triangular(*reengagement_triangular_s)`` on the same decision stream.
    reengagement_interval_s: tuple[float, float] = (10.0, 20.0)
    reengagement_triangular_s: tuple[float, float, float] = (8.0, 12.0, 18.0)
    rocket_budget_per_sortie: int = 49
    # Per-config rocket capacity, SysML-sourced from each
    # weapon part def (baseline ← SBAT-70, modernized ← APKWS). ``None`` falls
    # back to the shared ``rocket_budget_per_sortie`` (back-compat). Both default
    # to 49 in the SysML model ⇒ metric-neutral by value.
    baseline_rocket_budget: int | None = None
    modernized_rocket_budget: int | None = None
    # Number of consecutive non-decreasing-separation steps that
    # mark an engagement as infeasible during PURSUING discrete integration.
    n_infeasible_updates: int = 3

    # ─── Mission timeline ─────────────────────────────────────────────────
    t_pea_start_s: float = T_PEA_START_DEFAULT_S  # 9 180 s = 153 min
    # Dynamic spawn window + unified RTB_FUEL_LIMIT state.
    # Default ``True`` activates: (a) ``simpy.AllOf`` spawn window in
    # ``MissionSimulator.run_paired``, (b) ``A29State.RTB_FUEL_LIMIT`` in
    # ``A29._recover``, (c) emergent ``T_PEA`` in
    # ``MetricsAggregator._t_pea_for_aor``. ``False`` preserves the earlier
    # legacy path bit-exactly.
    uas_spawn_window_enabled: bool = True

    # ─── Recovery / bingo fuel ─────────────────────────────────
    # Refuel/rearm time at the FARP. Default 0 s: SysML ``RefuelAtFARP``
    # declares "operation disregarded in simulation timing". Parametrisable for
    # future sensitivity studies.
    farp_refuel_time_s: float = 0.0
    # Geometric recovery phases 5/6. When True (default) the CAP→FARP and
    # FARP→SBPV legs use ``t_to_farp`` geometry; when False the legacy fixed
    # ``A29Schedule.phase5/6`` durations are used (regression / old-run repro).
    geometric_recovery_phases: bool = True

    # ─── Simulation control ───────────────────────────────────────────────
    skip_pre_pea: bool = False
    # MOE-1 is a completion-time rate clamped to [0,1] (Eq. 3.5).
    # Default True makes the clamped form canonical; set False to inspect the
    # raw unclamped rate. Only ``_compute_moe1`` reads this (MOE-1-scoped).
    clip_moe1_to_01: bool = True
    log_level: str = "INFO"

    # ─── Persistence ──────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Validate the UAS generation rule flags.
        if self.spawn_interpolation not in ("arc", "chord"):
            raise ValueError(
                f"spawn_interpolation must be 'arc' or 'chord', got {self.spawn_interpolation!r}"
            )
        if self.sector_draw_rule not in ("cdf", "pdf"):
            raise ValueError(
                f"sector_draw_rule must be 'cdf' or 'pdf', got {self.sector_draw_rule!r}"
            )
        if self.heading_inward_rule not in ("literal", "east_flip"):
            raise ValueError(
                f"heading_inward_rule must be 'literal' or 'east_flip', "
                f"got {self.heading_inward_rule!r}"
            )
        # Validate engagement parameters.
        if self.baseline_visual_acuity_rad <= 0:
            raise ValueError(
                f"baseline_visual_acuity_rad must be > 0, got {self.baseline_visual_acuity_rad}"
            )
        if self.n_infeasible_updates < 1:
            raise ValueError(
                f"n_infeasible_updates must be >= 1, got {self.n_infeasible_updates}"
            )
        # Triangular(left ≤ mode ≤ right) precondition.
        left, mode, right = self.reengagement_triangular_s
        if not (left <= mode <= right):
            raise ValueError(
                "reengagement_triangular_s must satisfy left <= mode <= right, "
                f"got {self.reengagement_triangular_s}"
            )
        # Legacy path requires positive phase4_pea_s on every
        # A-29 (it gates the PEA end via env.timeout). When flag=True the field
        # is ignored by the PEA loop (its duration is emergent).
        if not self.uas_spawn_window_enabled:
            for a29_id, sched in self.a29_schedules.items():
                if sched.phase4_pea_s <= 0:
                    raise ValueError(
                        f"phase4_pea_s must be > 0 when uas_spawn_window_enabled=False "
                        f"({a29_id}={sched.phase4_pea_s})"
                    )

    def config_hash(self) -> str:
        """SHA-256 hash of the resolved config, for traceability."""
        d = self.to_dict()
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Deep-convert to a JSON-serialisable dict."""
        data: dict[str, Any] = _to_serializable(self)
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> MissionConfig:
        """Load from a YAML file (defaults fill in missing keys)."""
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls._from_dict(data)

    @classmethod
    def from_yaml_with_sysml(
        cls,
        yaml_path: str | Path,
        sysml_root: str | Path | None = None,
    ) -> MissionConfig:
        """Load YAML, then overlay SysML overrides if a SysML root is given.

        Precedence, increasing: Python defaults < YAML < SysML. The SysML overrides
        are applied on top of the YAML, so the model prevails in the fields it supplies.
        """
        cfg = cls.from_yaml(yaml_path)
        if sysml_root is None:
            return cfg
        # Local import to avoid a hard dependency cycle at module load time.
        from cuas_sim.sysml_loader import SysMLLoader

        overrides = SysMLLoader(Path(sysml_root)).load()
        return overrides.apply_to(cfg)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MissionConfig:
        """Convert a plain dict (from YAML) to a MissionConfig.

        Unknown keys raise a warning and are skipped. Sub-records (GeoPoint,
        UASArchetypeSpec, A29Schedule) are reconstructed if present.
        """
        kwargs: dict[str, Any] = {}
        known_names = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k not in known_names:
                logger.warning("Unknown MissionConfig key in YAML: %s", k)
                continue
            kwargs[k] = _from_serializable(k, v)
        return cls(**kwargs)

    def replaced(self, **changes: Any) -> MissionConfig:
        """Functional update — returns a new frozen instance with overrides."""
        return replace(self, **changes)

    def rocket_budget_for(self, config_name: str) -> int:
        """Rockets loaded per sortie for ``config_name``.

        Per-config SysML-sourced capacity (``modernized`` ← APKWS, ``baseline`` ←
        SBAT-70); falls back to the shared ``rocket_budget_per_sortie`` when the
        per-config field is ``None`` (back-compat).
        """
        per_cfg = (
            self.modernized_rocket_budget
            if config_name == "modernized"
            else self.baseline_rocket_budget
        )
        return per_cfg if per_cfg is not None else self.rocket_budget_per_sortie

    def validate_against_requirements(self, sysml_root: str | Path = "sysml") -> None:
        """Validate this config against the SysML ``require constraint`` blocks.

        Opt-in: NOT called in ``__post_init__``. Raises
        :class:`~cuas_sim.sysml_requirements.RequirementsValidationError` if any bound
        requirement constraint is violated; loading the requirements is always lenient.
        """
        # Local import to avoid a hard dependency cycle at module load time.
        from cuas_sim.sysml_requirements import validate_against_requirements

        validate_against_requirements(self, sysml_root)


# ─── (de)serialisation helpers ─────────────────────────────────────────────


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, MissionConfig):
        return {f.name: _to_serializable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, GeoPoint):
        return {"lat": obj.lat, "lon": obj.lon, "name": obj.name}
    if isinstance(obj, (UASArchetypeSpec, A29Schedule, BorderSegment)):
        return {f.name: _to_serializable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _from_serializable(field_name: str, value: Any) -> Any:
    """Best-effort reconstruction of sub-records from raw YAML dicts."""
    if field_name == "main_base":
        return _geopoint_from(value)
    if field_name in ("farps", "caps"):
        return {k: _geopoint_from(v) for k, v in value.items()}
    if field_name == "a29_schedules":
        return {k: A29Schedule(**v) for k, v in value.items()}
    if field_name == "uas_mr" or field_name == "uas_fw":
        return UASArchetypeSpec(**value)
    if field_name == "border_sectors" and value is not None:
        return tuple(
            BorderSegment(
                index=int(v["index"]),
                start=_geopoint_from(v["start"]),
                end=_geopoint_from(v["end"]),
                # Arc polyline; absent -> () (chord fallback).
                arc=tuple(
                    (float(a), float(b), float(c)) for a, b, c in v.get("arc", ())
                ),
            )
            for v in value
        )
    if field_name in ("baseline_update_interval_s", "modernized_update_interval_s",
                      "baseline_envelope_m", "modernized_envelope_m",
                      "reengagement_interval_s", "reengagement_triangular_s") \
            and isinstance(value, list):
        return tuple(value)
    return value


def _geopoint_from(value: Any) -> GeoPoint:
    if isinstance(value, GeoPoint):
        return value
    if isinstance(value, dict):
        if "lat_dms" in value and "lon_dms" in value:
            return GeoPoint.from_dms(value["lat_dms"], value["lon_dms"], value.get("name", ""))
        return GeoPoint(lat=float(value["lat"]), lon=float(value["lon"]),
                        name=value.get("name", ""))
    raise TypeError(f"Cannot build GeoPoint from {value!r}")
