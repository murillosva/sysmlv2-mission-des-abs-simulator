"""Centralized random number generation registry.

Implements the Common Random Numbers (CRN) discipline: in any paired run, the
**world streams** (UAS arrivals, sector choice, spawn geometry, archetype) are
**shared** between the baseline and modernized configurations, while the
**decision streams** (tracking timers, firing philosophy, P_hit Bernoulli,
P_track sampling) are **split** per configuration. The split is essential
because the two configurations sample these streams at different rates and
points; sharing them would induce a spurious correlation. The shared streams
allow paired-difference statistical tests.

The split is achieved with :class:`numpy.random.SeedSequence` ``spawn`` —
a single master seed deterministically produces a hierarchy of independent
sub-streams. Verified by ``tests/property/test_crn.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# ─── Stream identifiers ────────────────────────────────────────────────────

#: Number of world-shared streams.
_N_WORLD: Final[int] = 4
#: Number of decision streams per configuration.
#: (tracking_updates, firing_phil, p_hit_outcome, p_track_sampling,
#: reengagement_interval). Adding a stream at the end preserves
#: all prior streams bit-identically (SeedSequence children are independent
#: of ``spawn(n)``), so CRN/bit-identity of existing streams is unaffected.
_N_DECISION: Final[int] = 5

#: Sentinel for the two configurations in a paired run.
ConfigName = str  # "baseline" | "modernized"


@dataclass(slots=True)
class RNGRegistry:
    """Centralized RNG hub for a paired run.

    Attributes
    ----------
    arrivals
        Poisson inter-arrival sampling. **World-shared.**
    sector_choice
        Sector inverse-CDF sampling. **World-shared.**
    spawn_geometry
        ω (spawn-point coefficient) and heading deviation. **World-shared.**
    archetype
        Multi-rotor vs fixed-wing draw. **World-shared.**
    tracking_updates
        Per-config dict. Each draws U(t_min, t_max) heading update intervals.
    firing_phil
        Per-config dict. Discrete uniform over {short, medium, long}.
    p_hit_outcome
        Per-config dict. Bernoulli(P_hit) realisation for MOS-1.
    p_track_sampling
        Per-config dict. Reserved for any Bernoulli realisation of P_track
        (the current implementation uses the expected value; this is kept
        for symmetry and forward compatibility).
    reengagement_interval
        Per-config dict. U(reengagement_interval_s) between successive shots
        in the multi-shot re-engagement loop.
    """

    master_seed: int
    arrivals: np.random.Generator
    sector_choice: np.random.Generator
    spawn_geometry: np.random.Generator
    archetype: np.random.Generator
    tracking_updates: dict[ConfigName, np.random.Generator]
    firing_phil: dict[ConfigName, np.random.Generator]
    p_hit_outcome: dict[ConfigName, np.random.Generator]
    p_track_sampling: dict[ConfigName, np.random.Generator]
    reengagement_interval: dict[ConfigName, np.random.Generator]

    @classmethod
    def from_master_seed(cls, master_seed: int) -> RNGRegistry:
        """Build a registry from a single integer master seed.

        Hierarchy::

            master_seed
            ├── world_seeds[0:4]    →  arrivals, sector, spawn_geom, archetype
            ├── baseline_seeds[0:5] →  tracking, firing, p_hit, p_track, reengage
            └── modernized_seeds[0:5] → tracking, firing, p_hit, p_track, reengage
        """
        ss = np.random.SeedSequence(master_seed)
        world_seeds = ss.spawn(_N_WORLD)
        ss_b, ss_m = ss.spawn(2)
        b_seeds = ss_b.spawn(_N_DECISION)
        m_seeds = ss_m.spawn(_N_DECISION)

        def _gens(seeds: list[np.random.SeedSequence]) -> list[np.random.Generator]:
            return [np.random.default_rng(s) for s in seeds]

        world = _gens(world_seeds)
        b = _gens(b_seeds)
        m = _gens(m_seeds)

        return cls(
            master_seed=master_seed,
            arrivals=world[0],
            sector_choice=world[1],
            spawn_geometry=world[2],
            archetype=world[3],
            tracking_updates={"baseline": b[0], "modernized": m[0]},
            firing_phil={"baseline": b[1], "modernized": m[1]},
            p_hit_outcome={"baseline": b[2], "modernized": m[2]},
            p_track_sampling={"baseline": b[3], "modernized": m[3]},
            reengagement_interval={"baseline": b[4], "modernized": m[4]},
        )

    # ─── Convenience helpers ──────────────────────────────────────────────

    def per_config(self, cfg_name: ConfigName) -> _PerConfigRNG:
        """Bundle the decision streams for a single configuration."""
        return _PerConfigRNG(
            tracking_updates=self.tracking_updates[cfg_name],
            firing_phil=self.firing_phil[cfg_name],
            p_hit_outcome=self.p_hit_outcome[cfg_name],
            p_track_sampling=self.p_track_sampling[cfg_name],
            reengagement_interval=self.reengagement_interval[cfg_name],
        )


@dataclass(slots=True)
class _PerConfigRNG:
    """Convenience bundle for a single config's decision streams."""

    tracking_updates: np.random.Generator
    firing_phil: np.random.Generator
    p_hit_outcome: np.random.Generator
    p_track_sampling: np.random.Generator
    reengagement_interval: np.random.Generator
