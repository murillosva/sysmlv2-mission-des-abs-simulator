"""Medium-scope SysML v2 loader.

Parses the ``Technical`` package of the 27-file CoSMA model under ``sysml/`` and
extracts:

* numeric attributes with SI units (e.g. ``[m]``, ``[s]``, ``[kn]``, ``[km]``)
* the angular ``visualAcuityThreshold`` (arcmin → rad)
* (geography is parsed into the AST but kept default-sourced; its extraction is deferred)

The extraction engine is a **real AST parser**
(:mod:`cuas_sim.sysml_parser`): the text is parsed into ``SysMLPackage →
SysMLPartDef → SysMLAttribute`` and dispatched by ``part def`` name. The legacy
regex extractor is **retained** as a fallback so that the invariant *"Simulation
must never crash because of .sysml issues"* holds: on a structurally malformed
file the loader WARNs and falls back to the regex engine, then to defaults.

The :class:`SysMLOverrides` dataclass packages whatever was successfully
extracted, and :meth:`apply_to` merges it onto an existing :class:`MissionConfig`
(SysML > defaults, but YAML beats both — see :mod:`cuas_sim.config`).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cuas_sim.config import MissionConfig
from cuas_sim.geometry.coordinates import GeoPoint, parse_dms
from cuas_sim.sysml_parser import SysMLParseError, SysMLPartDef, parse_package

logger = logging.getLogger(__name__)

#: SysML geography (FARP/SBPV coordinates) extraction is
#: deferred. The AST parses the coordinate strings correctly (the regex engine
#: never could, due to escaped quotes), but enabling it would change the golden
#: ``SysMLOverrides`` and is metric-neutral only because the SysML coordinates
#: equal the Python defaults. Flipping this flag is a separate spec.
_ENABLE_SYSML_GEOGRAPHY: Final[bool] = False

# ─── Legacy regex toolkit (retained as fallback + golden oracle) ────────────

#: ``attribute foo :> ISQ::length = 13 [km]`` or ``= 144 [m/s]``.
_ATTR_NUMERIC = re.compile(
    r"""attribute\s+(?P<name>\w+)\s*(?::>\s*\w[\w:]*\s*)?=\s*
        (?P<val>-?\d+(?:\.\d+)?)\s*\[(?P<unit>[^\]]+)\]""",
    re.VERBOSE,
)

#: ``attribute latitude :> ScalarValues::String = "8°42'59.61\"S";``.
_ATTR_STRING = re.compile(
    r"""attribute\s+(?P<name>\w+)\s*(?::\s*ScalarValues::String\s*)?=
        \s*"(?P<val>[^"]+)"\s*;""",
    re.VERBOSE,
)

#: ``part def Foo :> Bar { ... }``. Captures definition name and body.
_PART_DEF = re.compile(
    r"part\s+def\s+(?P<name>\w+)\s*(?::>\s*\w+\s*)?\{(?P<body>.*?)^\s*\}",
    re.DOTALL | re.MULTILINE,
)

#: ``attribute visualAcuityThreshold :> ISQ::planeAngle = 1.0 ['arcmin'];``
#: The angular unit (arcmin/arcsec/rad) lies outside the SI-length/time table.
_ACUITY_ATTR = re.compile(
    r"""attribute\s+visualAcuityThreshold\s*(?::>\s*\w[\w:]*\s*)?=\s*
        (?P<val>-?\d+(?:\.\d+)?)\s*\['(?P<unit>[^']+)'\]""",
    re.VERBOSE,
)

#: Dimensionless numeric attributes the loader DOES
#: surface despite carrying no ``[unit]``. The general rule "numeric ⇒ has unit"
#: would exclude them (so every *other* unit-less Real stays unread); this
#: explicit name whitelist is the narrow exception. Applied **identically** by
#: the AST adapter (:func:`_attrs_from_ast`) and the regex engine
#: (:func:`_parse_numeric_attrs`) so AST==regex holds. Unit semantics (percent
#: ``/100``, ``int`` cast) live in the shared dispatch, not here.
_DIMENSIONLESS_WHITELIST: Final[frozenset[str]] = frozenset(
    {
        "hitProbabilityAtMinRange",
        "dimensionalPenaltyFixedWing",
        "dimensionalPenaltyMultiRotor",
        "launcherRocketCapacity",
    }
)

#: ``attribute hitProbabilityAtMinRange : ScalarValues::Real = 80.0;``
#: (no bracketed unit). Only whitelisted names are kept — see _parse_numeric_attrs.
_ATTR_DIMENSIONLESS = re.compile(
    r"""attribute\s+(?P<name>\w+)\s*:\s*ScalarValues::(?:Real|Integer)\s*=\s*
        (?P<val>-?\d+(?:\.\d+)?)\s*;""",
    re.VERBOSE,
)

_UNIT_CONVERSIONS_TO_SI: dict[str, tuple[str, float]] = {
    "m": ("m", 1.0),
    "km": ("m", 1_000.0),
    "nmi": ("m", 1_852.0),
    "s": ("s", 1.0),
    "min": ("s", 60.0),
    "h": ("s", 3_600.0),
    "m/s": ("m/s", 1.0),
    "kn": ("m/s", 0.5144444444444444),
    "kg": ("kg", 1.0),
    "Hz": ("Hz", 1.0),
}


def _to_si(value: float, unit: str) -> tuple[float, str]:
    """Convert a parsed (value, unit) tuple to SI. Returns the SI value+unit."""
    unit = unit.strip()
    if unit not in _UNIT_CONVERSIONS_TO_SI:
        return value, unit  # unknown unit — leave it alone, caller decides
    si_unit, factor = _UNIT_CONVERSIONS_TO_SI[unit]
    return value * factor, si_unit


def _acuity_to_rad(value: float, unit: str) -> float | None:
    """Convert an angular acuity literal to radians."""
    if unit == "arcmin":
        return value * math.pi / (180.0 * 60.0)
    if unit == "arcsec":
        return value * math.pi / (180.0 * 3600.0)
    if unit in ("rad", "radian", "radians"):
        return value
    logger.warning("Unknown unit %r for visualAcuityThreshold — ignored", unit)
    return None


# ─── Overrides dataclass ───────────────────────────────────────────────────


@dataclass(slots=True)
class SysMLOverrides:
    """Whatever the loader managed to extract from the SysML model.

    Every field is optional. :meth:`apply_to` only updates the corresponding
    :class:`MissionConfig` field if the extracted value is not ``None``.
    """

    # Aircraft
    a29_cruise_speed_mps: float | None = None
    a29_endurance_s: float | None = None
    # UAS archetypes (kinematic only)
    uas_mr_max_speed_mps: float | None = None
    uas_mr_dimension_m: float | None = None
    uas_mr_ceiling_m: float | None = None
    uas_fw_max_speed_mps: float | None = None
    uas_fw_dimension_m: float | None = None
    uas_fw_ceiling_m: float | None = None
    # Tracking — baseline
    baseline_update_interval_s: tuple[float, float] | None = None
    baseline_max_acquisition_m: float | None = None
    # Naked-eye visual acuity threshold (in radians, parsed
    # from PilotVisualAcquisitionSystemNakedEye.visualAcuityThreshold)
    baseline_visual_acuity_rad: float | None = None
    # Tracking — modernized
    modernized_update_interval_s: tuple[float, float] | None = None
    modernized_eoir_max_m: float | None = None
    modernized_eoir_accuracy_m: float | None = None
    modernized_track_p_fw: float | None = None
    modernized_track_p_mr: float | None = None
    # Weapon
    baseline_envelope_m: tuple[float, float] | None = None
    modernized_envelope_m: tuple[float, float] | None = None
    # APKWS modernized weapon coefficients (Eq. 3.14) + per-config
    # rocket capacity. All metric-neutral by value (== config defaults).
    modernized_p_hit_max: float | None = None
    dimensional_penalty_fw: float | None = None
    dimensional_penalty_mr: float | None = None
    baseline_rocket_budget: int | None = None
    modernized_rocket_budget: int | None = None
    # Geographic — populated from TechnicalComponentsPackage doc strings
    main_base: GeoPoint | None = None
    farps: dict[str, GeoPoint] = field(default_factory=dict)

    # Provenance — list of (source_file, override_path) tuples for audit log
    provenance: list[tuple[str, str]] = field(default_factory=list)

    def apply_to(self, cfg: MissionConfig) -> MissionConfig:
        """Return a new MissionConfig with non-None overrides applied."""
        changes: dict[str, Any] = {}
        # Scalars
        for key in (
            "a29_cruise_speed_mps",
            "a29_endurance_s",
            "baseline_max_acquisition_m",
            "baseline_visual_acuity_rad",
            "modernized_eoir_accuracy_m",
            "modernized_track_p_fw",
            "modernized_track_p_mr",
            # Override attr name == config field name.
            "dimensional_penalty_fw",
            "dimensional_penalty_mr",
            "baseline_rocket_budget",
            "modernized_rocket_budget",
        ):
            v = getattr(self, key)
            if v is not None:
                changes[key] = v
                logger.info("SysML override applied: %s = %s", key, v)
        # Modernized P_hit maps to the shared p_hit_at_min_range.
        if self.modernized_p_hit_max is not None:
            changes["p_hit_at_min_range"] = self.modernized_p_hit_max
            logger.info(
                "SysML override applied: p_hit_at_min_range = %s", self.modernized_p_hit_max
            )
        # UAS archetypes — rebuild as new spec objects
        if any(getattr(self, f) is not None for f in
               ("uas_mr_max_speed_mps", "uas_mr_dimension_m", "uas_mr_ceiling_m")):
            existing = cfg.uas_mr
            from cuas_sim.config import UASArchetypeSpec  # local import to break cycle
            changes["uas_mr"] = UASArchetypeSpec(
                name=existing.name,
                max_speed_mps=self.uas_mr_max_speed_mps or existing.max_speed_mps,
                dimension_m=self.uas_mr_dimension_m or existing.dimension_m,
                ceiling_m=self.uas_mr_ceiling_m or existing.ceiling_m,
            )
            logger.info("SysML override applied: uas_mr = %s", changes["uas_mr"])
        if any(getattr(self, f) is not None for f in
               ("uas_fw_max_speed_mps", "uas_fw_dimension_m", "uas_fw_ceiling_m")):
            existing = cfg.uas_fw
            from cuas_sim.config import UASArchetypeSpec
            changes["uas_fw"] = UASArchetypeSpec(
                name=existing.name,
                max_speed_mps=self.uas_fw_max_speed_mps or existing.max_speed_mps,
                dimension_m=self.uas_fw_dimension_m or existing.dimension_m,
                ceiling_m=self.uas_fw_ceiling_m or existing.ceiling_m,
            )
            logger.info("SysML override applied: uas_fw = %s", changes["uas_fw"])
        # Tuples
        for key in (
            "baseline_update_interval_s",
            "modernized_update_interval_s",
            "baseline_envelope_m",
            "modernized_envelope_m",
        ):
            v = getattr(self, key)
            if v is not None:
                changes[key] = v
                logger.info("SysML override applied: %s = %s", key, v)
        # Geography
        if self.main_base is not None:
            changes["main_base"] = self.main_base
            logger.info("SysML override applied: main_base = %s", self.main_base)
        if self.farps:
            merged = {**cfg.farps, **self.farps}
            changes["farps"] = merged
            logger.info("SysML override applied: farps = %s", list(self.farps))
        return cfg.replaced(**changes)


# ─── Loader ────────────────────────────────────────────────────────────────


class SysMLLoader:
    """Walks a SysML v2 model directory and produces a :class:`SysMLOverrides`.

    The loader never raises on a bad file — at worst it logs a warning and
    moves on, leaving the corresponding field as ``None``. By design,
    *"simulation must never crash because of .sysml issues."*
    The primary engine is the AST parser; the regex extractor is
    the fallback (:meth:`_legacy_regex_extract`).
    """

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"SysML root not found: {root}")
        self.root = root

    def load(self) -> SysMLOverrides:
        ov = SysMLOverrides()
        self._extract_technical_components(ov)
        return ov

    # ── TechnicalComponentsPackage ────────────────────────────────────────

    def _extract_technical_components(self, ov: SysMLOverrides) -> None:
        path = self.root / "Technical" / "TechnicalComponentsPackage.sysml"
        if not path.exists():
            logger.warning("Missing SysML file: %s — using defaults", path)
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return

        source = str(path)
        try:
            pkg = parse_package(text)
        except SysMLParseError as exc:
            logger.warning(
                "SysML AST parse failed for %s: %s — falling back to regex engine",
                path, exc,
            )
            self._legacy_regex_extract(text, ov, source)
            return

        for pd in pkg.part_defs:
            attrs, strs, acuity = _attrs_from_ast(pd)
            self._dispatch_part_def(pd.name, attrs, strs, acuity, ov, source)

    def _dispatch_part_def(
        self,
        name: str,
        attrs: dict[str, float],
        strs: dict[str, str],
        acuity: tuple[float, str] | None,
        ov: SysMLOverrides,
        source: str,
    ) -> None:
        # ── A-29 airframe (baseline or modernized — share fields) ────────
        if name in (
            "EmbraerEMB314SuperTucanoAirframeBaseline",
            "EmbraerEMB314SuperTucanoAirframeModernized",
        ):
            if (v := attrs.get("cruiseSpeed")) is not None:
                ov.a29_cruise_speed_mps = v
                ov.provenance.append((source, f"{name}.cruiseSpeed → a29_cruise_speed_mps"))
            if (v := attrs.get("enduranceWith2ExternalTanksAndEOIR")) is not None:
                ov.a29_endurance_s = v
                ov.provenance.append((source, f"{name}.endurance → a29_endurance_s"))
            return

        # ── VHF radio (baseline update interval) ─────────────────────────
        if name == "VHFVoiceCommunicationRadio":
            tmin = attrs.get("updateIntervalMin")
            tmax = attrs.get("updateIntervalMax")
            if tmin is not None and tmax is not None:
                ov.baseline_update_interval_s = (tmin, tmax)
                ov.provenance.append((source, f"{name} → baseline_update_interval_s"))
            return

        # ── Naked-eye visual system (baseline max acquisition + angular acuity) ──
        if name == "PilotVisualAcquisitionSystemNakedEye":
            if (v := attrs.get("maxVisualAcquisitionDistanceMax")) is not None:
                ov.baseline_max_acquisition_m = v
                ov.provenance.append((source, f"{name} → baseline_max_acquisition_m"))
            # visualAcuityThreshold is angular (arcmin/arcsec/rad).
            if acuity is not None:
                rad = _acuity_to_rad(acuity[0], acuity[1])
                if rad is not None:
                    ov.baseline_visual_acuity_rad = rad
                    logger.info(
                        "SysML override applied: baseline_visual_acuity_rad = %s rad "
                        "(from %s %s)", rad, acuity[0], acuity[1],
                    )
                    ov.provenance.append(
                        (source, f"{name}.visualAcuityThreshold → baseline_visual_acuity_rad")
                    )
            return

        # ── M3AR datalink (modernized update interval) ───────────────────
        if name == "RohdeAndSchwarzM3ARMR6000RLDigitalDatalink":
            tmin = attrs.get("updateIntervalMin")
            tmax = attrs.get("updateIntervalMax")
            if tmin is not None and tmax is not None:
                ov.modernized_update_interval_s = (tmin, tmax)
                ov.provenance.append((source, f"{name} → modernized_update_interval_s"))
            return

        # ── FLIR Star SAFIRE II EO/IR ────────────────────────────────────
        if name == "FLIRStarSAFIREIIEOIRSensor":
            if (v := attrs.get("laserRangefinderMaxRange")) is not None:
                ov.modernized_eoir_max_m = v
                ov.provenance.append((source, f"{name} → modernized_eoir_max_m"))
            if (v := attrs.get("rangeAccuracy")) is not None:
                ov.modernized_eoir_accuracy_m = v
                ov.provenance.append((source, f"{name} → modernized_eoir_accuracy_m"))
            if (v := attrs.get("trackingProbabilityFixedWing")) is not None:
                ov.modernized_track_p_fw = v
                ov.provenance.append((source, f"{name} → modernized_track_p_fw"))
            if (v := attrs.get("trackingProbabilityMultiRotor")) is not None:
                ov.modernized_track_p_mr = v
                ov.provenance.append((source, f"{name} → modernized_track_p_mr"))
            return

        # ── SBAT-70 unguided rocket (baseline envelope + capacity) ────────
        if name == "SBAT70UnguidedRocketLauncherPod":
            short = attrs.get("firingDistanceShortPhilosophy")
            long_ = attrs.get("firingDistanceLongPhilosophy")
            if short is not None and long_ is not None:
                ov.baseline_envelope_m = (short, long_)
                ov.provenance.append((source, f"{name} → baseline_envelope_m"))
            # Baseline rocket capacity (per-config).
            if (v := attrs.get("launcherRocketCapacity")) is not None:
                ov.baseline_rocket_budget = int(v)
                ov.provenance.append((source, f"{name} → baseline_rocket_budget"))
            return

        # ── APKWS guided rocket (modernized weapon model) ──
        if name == "APKWSLaserGuidanceKitWithSBAT70Rocket":
            short = attrs.get("firingDistanceShortPhilosophy")
            long_ = attrs.get("firingDistanceLongPhilosophy")
            if short is not None and long_ is not None:
                ov.modernized_envelope_m = (short, long_)
                ov.provenance.append((source, f"{name} → modernized_envelope_m"))
            if (v := attrs.get("hitProbabilityAtMinRange")) is not None:
                ov.modernized_p_hit_max = v / 100.0  # percent → fraction
                ov.provenance.append((source, f"{name} → modernized_p_hit_max"))
            if (v := attrs.get("dimensionalPenaltyFixedWing")) is not None:
                ov.dimensional_penalty_fw = v
                ov.provenance.append((source, f"{name} → dimensional_penalty_fw"))
            if (v := attrs.get("dimensionalPenaltyMultiRotor")) is not None:
                ov.dimensional_penalty_mr = v
                ov.provenance.append((source, f"{name} → dimensional_penalty_mr"))
            if (v := attrs.get("launcherRocketCapacity")) is not None:
                ov.modernized_rocket_budget = int(v)
                ov.provenance.append((source, f"{name} → modernized_rocket_budget"))
            return

        # ── UAS archetypes ────────────────────────────────────────────────
        if name == "MultiRotorUASIllicit":
            if (v := attrs.get("maxSpeed")) is not None:
                ov.uas_mr_max_speed_mps = v
            if (v := attrs.get("maxDimensionWidth")) is not None:
                ov.uas_mr_dimension_m = v
            if (v := attrs.get("ceiling")) is not None:
                ov.uas_mr_ceiling_m = v
            return
        if name == "FixedWingUASIllicit":
            if (v := attrs.get("maxSpeed")) is not None:
                ov.uas_fw_max_speed_mps = v
            if (v := attrs.get("maxDimensionWidth")) is not None:
                ov.uas_fw_dimension_m = v
            if (v := attrs.get("ceiling")) is not None:
                ov.uas_fw_ceiling_m = v
            return

        # ── FARPs (geographic) ────────────────────────────────────────────
        if name.startswith("FARPPoint"):
            # Geography extraction deferred (default-sourced).
            if _ENABLE_SYSML_GEOGRAPHY:
                farp_code = name.replace("FARPPoint", "")
                lat = strs.get("latitude")
                lon = strs.get("longitude")
                if lat and lon:
                    try:
                        ov.farps[farp_code] = GeoPoint(
                            lat=parse_dms(lat), lon=parse_dms(lon), name=farp_code
                        )
                    except ValueError as exc:
                        logger.warning("Cannot parse coords for %s: %s", farp_code, exc)
            return

        # ── SBPV main base ────────────────────────────────────────────────
        if name == "GovernadorJorgeTeixeiraDeOliveiraAerodromeSBPV":
            # Geography extraction deferred (default-sourced).
            if _ENABLE_SYSML_GEOGRAPHY:
                lat = strs.get("latitude")
                lon = strs.get("longitude")
                if lat and lon:
                    try:
                        ov.main_base = GeoPoint(
                            lat=parse_dms(lat), lon=parse_dms(lon), name="SBPV"
                        )
                    except ValueError as exc:
                        logger.warning("Cannot parse SBPV coords: %s", exc)
            return

    # ── Legacy regex fallback ──────────────────────

    def _legacy_regex_extract(
        self, text: str, ov: SysMLOverrides, source: str
    ) -> None:
        """Regex-based extraction (the earlier engine), used as a fallback.

        Iterates ``part def`` bodies with :data:`_PART_DEF`, computes the same
        ``(attrs, strs, acuity)`` inputs the AST path produces, and dispatches.
        Never raises — at worst it extracts nothing.
        """
        for m in _PART_DEF.finditer(text):
            name = m.group("name")
            body = m.group("body")
            attrs = _parse_numeric_attrs(body)
            strs = _parse_string_attrs(body)
            acuity: tuple[float, str] | None = None
            ang = _ACUITY_ATTR.search(body)
            if ang is not None:
                acuity = (float(ang.group("val")), ang.group("unit").strip())
            self._dispatch_part_def(name, attrs, strs, acuity, ov, source)


def _attrs_from_ast(
    pd: SysMLPartDef,
) -> tuple[dict[str, float], dict[str, str], tuple[float, str] | None]:
    """Adapt a :class:`SysMLPartDef` to the dispatch inputs.

    ``attrs`` mirrors the legacy ``_parse_numeric_attrs`` selection rule —
    numeric attributes that carry a bracketed unit, SI-converted (this naturally
    excludes unit-less ``ScalarValues::Real``). ``strs`` are the
    ``ScalarValues::String`` values. ``acuity`` is the raw ``(value, unit)`` of
    ``visualAcuityThreshold`` if present.
    """
    attrs: dict[str, float] = {}
    strs: dict[str, str] = {}
    acuity: tuple[float, str] | None = None
    for a in pd.attributes:
        if a.type_ref == "ScalarValues::String" and isinstance(a.value, str):
            strs[a.name] = a.value
        if (
            a.unit is not None
            and isinstance(a.value, (int, float))
            and not isinstance(a.value, bool)
        ):
            si_val, _ = _to_si(float(a.value), a.unit)
            attrs[a.name] = si_val
            if a.name == "visualAcuityThreshold":
                acuity = (float(a.value), a.unit)
        # Whitelisted unit-less numerics (stored raw as
        # float; the dispatch applies percent/int conversions). Mirrors the regex
        # engine's _ATTR_DIMENSIONLESS so AST==regex holds.
        elif (
            a.unit is None
            and a.name in _DIMENSIONLESS_WHITELIST
            and isinstance(a.value, (int, float))
            and not isinstance(a.value, bool)
        ):
            attrs[a.name] = float(a.value)
    return attrs, strs, acuity


def _legacy_regex_load(root: Path) -> SysMLOverrides:
    """Run the legacy regex extractor over ``root`` (oracle of the golden test)."""
    ov = SysMLOverrides()
    path = root / "Technical" / "TechnicalComponentsPackage.sysml"
    if not path.exists():
        return ov
    text = path.read_text(encoding="utf-8")
    SysMLLoader(root)._legacy_regex_extract(text, ov, str(path))
    return ov


def _parse_numeric_attrs(body: str) -> dict[str, float]:
    """Return ``{attribute_name → value_in_SI}`` from a part-def body."""
    out: dict[str, float] = {}
    for m in _ATTR_NUMERIC.finditer(body):
        name = m.group("name")
        raw = float(m.group("val"))
        unit = m.group("unit")
        si_val, _ = _to_si(raw, unit)
        # If the same attribute appears multiple times in the same body
        # (e.g. SysML duplicates `rateOfClimb`), the LAST canonical SI form wins.
        out[name] = si_val
    # Surface whitelisted unit-less numerics (no `[unit]`),
    # mirroring the AST adapter's branch so the two engines stay bit-identical.
    for m in _ATTR_DIMENSIONLESS.finditer(body):
        name = m.group("name")
        if name in _DIMENSIONLESS_WHITELIST:
            out[name] = float(m.group("val"))
    return out


def _parse_string_attrs(body: str) -> dict[str, str]:
    """Return ``{attribute_name → string_value}``."""
    return {m.group("name"): m.group("val") for m in _ATTR_STRING.finditer(body)}
