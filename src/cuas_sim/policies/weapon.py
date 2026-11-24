"""Weapon policies — Eqs. (3.14)–(3.15) of the case study.

**Firing philosophy** (case study Section 3.3.4.2 and Table 3.3):

The philosophy is drawn with probability 1/3 each (short / medium / long —
``DetermineFirePhilosophy``), and the firing **distance** is drawn from a continuous
``Triangular(low, mode, high)`` *inside* the chosen philosophy's sub-band. The
envelope is split in three (``range/3``); the Triangular ``mode`` is anchored at
the SysML ``firingDistance{Short,Medium,Long}Philosophy`` point, which coincides
with ``{envelope_min, midpoint, envelope_max}`` (long → max, medium → centre,
short → min):

* **Baseline (SBAT-70 unguided)**, envelope ``[500, 2000]`` m: short ``]500, 1000]``,
  medium ``]1000, 1500]``, long ``]1500, 2000]``.
* **Modernized (APKWS guided)**, envelope ``[2000, 14000]`` m: short ``]2000, 6000]``,
  medium ``]6000, 10000]``, long ``]10000, 14000]``.

**Hit probability** (Eq. 3.14):

.. math::

    P_{\\text{hit}}(d_{\\text{fire}}) = M \\cdot 0.80 \\cdot
        \\bigl(1 - \\frac{d_{\\text{fire}} - d_{\\min}}{d_{\\max} - d_{\\min}}\\bigr)

where :math:`M` is the **dimensional penalty** (case study Table 3.3):
``M = 0.5`` for multi-rotor, ``M = 1.0`` for fixed-wing. The reference factor
``0.80`` is the maximum P_hit (achieved at the envelope minimum range).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cuas_sim.config import MissionConfig, UASArchetypeSpec


class WeaponPolicy(Protocol):
    """Common interface for the baseline and modernized weapon policies."""

    # Read-only members: the implementations are frozen dataclasses.
    @property
    def name(self) -> str:  # pragma: no cover
        ...

    @property
    def envelope_min_m(self) -> float:  # pragma: no cover
        ...

    @property
    def envelope_max_m(self) -> float:  # pragma: no cover
        ...

    @property
    def p_hit_max(self) -> float:  # pragma: no cover
        ...

    @property
    def M_mr(self) -> float:  # pragma: no cover
        ...

    @property
    def M_fw(self) -> float:  # pragma: no cover
        ...

    def draw_firing_distance(
        self, rng_decision: np.random.Generator
    ) -> tuple[float, int]:  # pragma: no cover
        ...

    def p_hit_expected(self, firing_distance_m: float, archetype: UASArchetypeSpec) -> float:  # pragma: no cover
        ...

    def p_hit_at(self, distance_m: float, archetype: UASArchetypeSpec) -> float:  # pragma: no cover
        """Eq. (3.14) at ``distance_m`` selecting the active weapon regime."""
        ...

    @property
    def floor_m(self) -> float:  # pragma: no cover
        """Minimum firing separation (loop floor) for the configuration."""
        ...

    def p_hit_realised(
        self,
        firing_distance_m: float,
        archetype: UASArchetypeSpec,
        rng_decision: np.random.Generator,
    ) -> tuple[bool, float]:  # pragma: no cover
        ...


def firing_subband(
    envelope_min_m: float, envelope_max_m: float, idx: int
) -> tuple[float, float, float]:
    """Triangular ``(low, mode, high)`` for philosophy ``idx``.

    The envelope ``[min, max]`` is split into three equal sub-bands of width
    ``R/3 = (max − min)/3``; ``idx`` selects short (0), medium (1) or long (2).
    The ``mode`` is the SysML ``firingDistance{phil}Philosophy`` anchor — which
    equals ``{min, midpoint, max}`` — giving short its min bias and long its
    max bias.
    """
    R = envelope_max_m - envelope_min_m
    # Snap the partition endpoints exactly to the envelope so the union of the
    # three sub-bands is precisely ``]min, max]`` (no float drift at idx 0/2).
    low = envelope_min_m if idx == 0 else envelope_min_m + idx * R / 3.0
    high = envelope_max_m if idx == 2 else envelope_min_m + (idx + 1) * R / 3.0
    mode = (envelope_min_m, 0.5 * (envelope_min_m + envelope_max_m), envelope_max_m)[idx]
    # Clamp the SysML anchor into [low, high] (numpy.triangular precondition);
    # only matters for non-canonical envelopes where R/3 rounds adversely.
    mode = min(max(mode, low), high)
    return low, mode, high


# ─── Concrete policy ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _BaseWeaponPolicy:
    envelope_min_m: float
    envelope_max_m: float
    p_hit_max: float
    M_mr: float
    M_fw: float
    name: str
    #: For the A-29N, the SBAT-70 unguided sub-regime used
    #: below the APKWS guided minimum (2 km). ``None`` for the baseline
    #: (single regime).
    unguided_below: _BaseWeaponPolicy | None = None

    def __post_init__(self) -> None:
        # A degenerate envelope makes the range/3 split ill-posed.
        if self.envelope_max_m <= self.envelope_min_m:
            raise ValueError(
                f"envelope_max_m ({self.envelope_max_m}) must exceed "
                f"envelope_min_m ({self.envelope_min_m})"
            )

    def draw_firing_distance(self, rng_decision: np.random.Generator) -> tuple[float, int]:
        """Triangular firing distance within the drawn philosophy sub-band.

        The discrete 1/3 philosophy draw (``DetermineFirePhilosophy``) is kept;
        the distance is then a ``Triangular(low, mode, high)`` over the sub-band
        ``range/3``, with ``mode`` anchored at the SysML
        ``firingDistance{phil}Philosophy`` point. Returns ``(distance_m, idx)``
        where ``idx`` is the philosophy (0=short, 1=medium, 2=long); the index
        is consumed by the controller to keep short = single-shot.
        """
        idx = int(rng_decision.integers(0, 3))
        low, mode, high = firing_subband(self.envelope_min_m, self.envelope_max_m, idx)
        return float(rng_decision.triangular(low, mode, high)), idx

    def p_hit_expected(self, firing_distance_m: float, archetype: UASArchetypeSpec) -> float:
        """Eq. (3.14): expected hit probability at a given firing distance."""
        if firing_distance_m < self.envelope_min_m or firing_distance_m > self.envelope_max_m:
            return 0.0
        span = self.envelope_max_m - self.envelope_min_m
        M = self.M_mr if archetype.name == "multi-rotor" else self.M_fw
        range_factor = 1.0 - (firing_distance_m - self.envelope_min_m) / span
        return float(M * self.p_hit_max * max(0.0, range_factor))

    def p_hit_at(self, distance_m: float, archetype: UASArchetypeSpec) -> float:
        """Eq. (3.14) at ``distance_m``, selecting the active regime.

        For the A-29N, separations below the APKWS guided minimum
        (``envelope_min_m`` = 2 km) fall back to the SBAT-70 unguided baseline
        curve (``unguided_below``). Sharp regime switch — no smoothing.
        """
        if self.unguided_below is not None and distance_m < self.envelope_min_m:
            return self.unguided_below.p_hit_at(distance_m, archetype)
        return self.p_hit_expected(distance_m, archetype)

    @property
    def floor_m(self) -> float:
        """Minimum firing separation: the unguided floor if a sub-regime
        exists (A-29N → 0.5 km), else this policy's envelope minimum."""
        if self.unguided_below is not None:
            return self.unguided_below.envelope_min_m
        return self.envelope_min_m

    def p_hit_realised(
        self,
        firing_distance_m: float,
        archetype: UASArchetypeSpec,
        rng_decision: np.random.Generator,
    ) -> tuple[bool, float]:
        """Bernoulli realisation for MOS-1.

        Returns ``(is_neutralized, p_hit)`` where ``p_hit`` is the
        Eq. (3.14) value (also the per-shot term of Eq. 3.15).
        """
        p = self.p_hit_expected(firing_distance_m, archetype)
        u = float(rng_decision.random())
        return (u < p, p)


# Concrete typed aliases — distinct names help debugging in event logs.
class BaselineWeaponPolicy(_BaseWeaponPolicy):
    """SBAT-70 unguided rocket weapon policy."""


class ModernizedWeaponPolicy(_BaseWeaponPolicy):
    """APKWS laser-guided rocket weapon policy."""


# ─── Factory ───────────────────────────────────────────────────────────────


def make_weapon_policy(cfg: MissionConfig, config_name: str) -> WeaponPolicy:
    """Build the weapon policy for ``baseline`` or ``modernized``."""
    unguided: _BaseWeaponPolicy | None = None
    if config_name == "baseline":
        envelope = cfg.baseline_envelope_m
        cls: type[_BaseWeaponPolicy] = BaselineWeaponPolicy
    elif config_name == "modernized":
        envelope = cfg.modernized_envelope_m
        cls = ModernizedWeaponPolicy
        # SBAT-70 unguided sub-regime below 2 km — reuses the
        # exact baseline curve (envelope 0.5–2 km, same dimensional M).
        unguided = BaselineWeaponPolicy(
            envelope_min_m=cfg.baseline_envelope_m[0],
            envelope_max_m=cfg.baseline_envelope_m[1],
            p_hit_max=cfg.p_hit_at_min_range,
            M_mr=cfg.dimensional_penalty_mr,
            M_fw=cfg.dimensional_penalty_fw,
            name="modernized-unguided",
        )
    else:
        raise ValueError(f"Unknown config name: {config_name!r}")
    return cls(
        envelope_min_m=envelope[0],
        envelope_max_m=envelope[1],
        p_hit_max=cfg.p_hit_at_min_range,
        M_mr=cfg.dimensional_penalty_mr,
        M_fw=cfg.dimensional_penalty_fw,
        name=config_name,
        unguided_below=unguided,
    )
