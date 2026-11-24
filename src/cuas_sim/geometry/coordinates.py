"""DMS coordinate parsing and GeoPoint dataclass.

The SysML v2 model and the case study tables encode coordinates as
degrees-minutes-seconds strings (e.g. ``"8°42'59.61\\"S"``). This module parses
them to signed decimal degrees and packages them as immutable :class:`GeoPoint`
records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# ─── DMS parser ────────────────────────────────────────────────────────────

#: Hemisphere markers. We accept "S" / "N" / "E" / "W", plus "O" (oeste,
#: Portuguese for west) used in the original case study tables.
_HEMI_SIGN: Final[dict[str, int]] = {
    "N": +1,
    "S": -1,
    "E": +1,
    "W": -1,
    "O": -1,  # case study uses "O" for "Oeste" (Portuguese)
}

# Pattern accepting variations like:
#   "8°42'59.61\"S"       (typical)
#   "1° 7'11.98\"N"       (one or more spaces)
#   "0°25'53.75\"N"       (zero degrees)
#   "63° 54'7.46\"O"      (Portuguese hemisphere)
# The pattern is intentionally tolerant of inner whitespace.
_DMS_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^\s*
    (?P<deg>-?\d+(?:\.\d+)?)   # degrees (allow signed for already-decimal)
    \s*°\s*
    (?P<min>\d+(?:\.\d+)?)     # minutes
    \s*'\s*
    (?P<sec>\d+(?:\.\d+)?)     # seconds
    \s*(?:\"|″)\s*             # double-quote or unicode prime
    (?P<hemi>[NSEWO])          # hemisphere
    \s*$
    """,
    re.VERBOSE,
)


def parse_dms(dms: str) -> float:
    """Parse a degrees-minutes-seconds latitude or longitude string.

    Parameters
    ----------
    dms
        DMS string such as ``"8°42'59.61\\"S"`` or ``"63°54'7.46\\"O"``.
        Accepts inner whitespace.

    Returns
    -------
    float
        Signed decimal degrees (positive N/E, negative S/W).

    Raises
    ------
    ValueError
        If the string does not match the expected DMS pattern.
    """
    m = _DMS_RE.match(dms)
    if m is None:
        raise ValueError(f"Cannot parse DMS coordinate: {dms!r}")
    deg = float(m.group("deg"))
    minutes = float(m.group("min"))
    seconds = float(m.group("sec"))
    hemi = m.group("hemi")
    sign = _HEMI_SIGN[hemi]
    # Magnitude in decimal degrees, then sign by hemisphere.
    magnitude = abs(deg) + minutes / 60.0 + seconds / 3600.0
    return sign * magnitude


# ─── GeoPoint ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """Immutable geographic point (decimal degrees, WGS-84).

    Attributes
    ----------
    lat
        Latitude in decimal degrees (positive North).
    lon
        Longitude in decimal degrees (positive East).
    name
        Optional label (e.g. ``"SBPV"``, ``"CAP-3"``).
    """

    lat: float
    lon: float
    name: str = ""

    @classmethod
    def from_dms(cls, lat_dms: str, lon_dms: str, name: str = "") -> GeoPoint:
        """Construct a GeoPoint from two DMS strings (lat, lon)."""
        return cls(lat=parse_dms(lat_dms), lon=parse_dms(lon_dms), name=name)

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"Latitude out of range: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"Longitude out of range: {self.lon}")
