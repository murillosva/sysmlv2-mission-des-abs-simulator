"""Sector ↔ AoR ↔ A-29 mapping.

The case study assigns each sector to exactly one AoR and each AoR to exactly
one A-29. This module provides O(1) bidirectional lookup tables, fail-fast
validation, and an inverse traversal for the OPT loop's replanning signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from cuas_sim.config import MissionConfig


@dataclass(frozen=True, slots=True)
class AoRMap:
    """Bidirectional lookup tables for sector ↔ AoR ↔ A-29.

    Parameters
    ----------
    sector_to_aor
        ``{sector_index: aor_name}``. Every spawned sector must be present.
    aor_to_a29
        ``{aor_name: a29_designator}``.
    aor_to_sectors
        ``{aor_name: [sector_indices]}`` (the input view).
    a29_to_aor
        Inverse of ``aor_to_a29``.
    """

    sector_to_aor: dict[int, str]
    aor_to_a29: dict[str, str]
    aor_to_sectors: dict[str, list[int]]
    a29_to_aor: dict[str, str]

    @classmethod
    def from_config(cls, cfg: MissionConfig) -> AoRMap:
        """Build the maps from a :class:`MissionConfig`, validating coverage."""
        # Forward: sector → AoR
        sector_to_aor: dict[int, str] = {}
        for aor_name, secs in cfg.aor_to_sectors.items():
            for s in secs:
                if s in sector_to_aor:
                    raise ValueError(
                        f"Sector {s} is mapped to both {sector_to_aor[s]} and {aor_name}"
                    )
                sector_to_aor[s] = aor_name

        # Verify all sectors in [sector_min, sector_max] are covered
        full = set(range(cfg.sector_min, cfg.sector_max + 1))
        missing = full - set(sector_to_aor)
        if missing:
            raise ValueError(
                f"AoR mapping does not cover sectors: {sorted(missing)}. "
                "Every sector in [sector_min, sector_max] must be assigned."
            )

        # Inverse: A-29 → AoR
        a29_to_aor: dict[str, str] = {}
        for aor_name, a29 in cfg.aor_to_a29.items():
            if a29 in a29_to_aor:
                raise ValueError(f"A-29 '{a29}' is assigned to two AoRs")
            a29_to_aor[a29] = aor_name

        # Cross-check: every AoR in aor_to_sectors has an A-29
        for aor_name in cfg.aor_to_sectors:
            if aor_name not in cfg.aor_to_a29:
                raise ValueError(f"AoR '{aor_name}' has no A-29 assigned")

        return cls(
            sector_to_aor=sector_to_aor,
            aor_to_a29=dict(cfg.aor_to_a29),
            aor_to_sectors={k: list(v) for k, v in cfg.aor_to_sectors.items()},
            a29_to_aor=a29_to_aor,
        )

    # ─── Lookup ──────────────────────────────────────────────────────────

    def aor_of_sector(self, sector_idx: int) -> str:
        return self.sector_to_aor[sector_idx]

    def a29_of_sector(self, sector_idx: int) -> str:
        return self.aor_to_a29[self.sector_to_aor[sector_idx]]

    def a29_of_aor(self, aor_name: str) -> str:
        return self.aor_to_a29[aor_name]

    def aor_of_a29(self, a29: str) -> str:
        return self.a29_to_aor[a29]

    def all_aors(self) -> list[str]:
        return list(self.aor_to_sectors)
