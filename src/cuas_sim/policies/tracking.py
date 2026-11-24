"""Tracking policies — Eqs. (3.8)–(3.9) and (3.12)–(3.13) of the case study.

The **baseline** A-29 uses naked-eye visual acquisition with VHF voice
heading updates every 5..60 s (uniform, Eq. 3.8). Per Eq. (3.12):

.. math::

    P_{\\text{track,bsl}}(d, L_U)
        = \\min\\!\\bigl(1,\\; \\frac{L_U}{d \\cdot \\theta_{\\text{acuity}}}\\bigr),
    \\quad 0 < d \\le d_{\\max} = 13\\,\\mathrm{km},

with :math:`\\theta_{\\text{acuity}} = 2.9088 \\times 10^{-4}` rad
(1 arcmin, naked-eye visual acuity threshold sourced from SysML
``PilotVisualAcquisitionSystemNakedEye.visualAcuityThreshold``). Returns
``0`` outside the valid range.

The first update interval for the baseline applies a minimum pilot reaction
time of 12.5 s (Table 3.3, MOP-2) via :meth:`BaselineTrackingPolicy.first_update_interval`.

The **modernized** A-29N uses the FLIR Star SAFIRE II EO/IR plus M3AR
digital datalink (1..10 s updates, Eq. 3.9). Per Eq. (3.13):

.. math::

    P_{\\text{track,mod}}(d_k, c_{\\mathrm{UAS}})
        = \\min\\!\\bigl(1,\\; \\frac{L_U}{\\sigma_s}\\bigr),
    \\quad 14 < d_k \\le 20\\,\\mathrm{km},

with :math:`\\sigma_s = 5\\,\\mathrm{m}` (accuracy of the EO/IR laser rangefinder). This value
is **constant** throughout the annulus — there is no range degradation within
the EO/IR operating window. For fixed-wing (L_U = 14 m): min(1, 14/5) = 1.0.
For multi-rotor (L_U = 1 m): min(1, 1/5) = 0.2.

.. note::

   **Annulus sampling.** :math:`P_{\\text{track}}` is sampled **only** in the
   annulus between the weapon-envelope maximum and the tracking maximum (pursuit
   phase). For baseline: :math:`d_k \\in (2, 13]` km. For modernized:
   :math:`d_k \\in (14, 20]` km.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cuas_sim.config import MissionConfig, UASArchetypeSpec


class TrackingPolicy(Protocol):
    """Common interface for the baseline and modernized tracking policies."""

    # Read-only members: the implementations are frozen dataclasses.
    @property
    def name(self) -> str:  # pragma: no cover
        ...

    @property
    def sampling_annulus_min_m(self) -> float:  # pragma: no cover
        ...

    @property
    def sampling_annulus_max_m(self) -> float:  # pragma: no cover
        ...

    @property
    def update_interval_min_s(self) -> float:  # pragma: no cover
        ...

    @property
    def update_interval_max_s(self) -> float:  # pragma: no cover
        ...

    def p_track(
        self, distance_m: float, archetype: UASArchetypeSpec, rng_decision: np.random.Generator
    ) -> float:  # pragma: no cover  -  protocol stub
        ...

    def first_update_interval(self, rng_decision: np.random.Generator) -> float:  # pragma: no cover
        ...

    def next_update_interval(self, rng_decision: np.random.Generator) -> float:  # pragma: no cover
        ...

    def in_sampling_annulus(self, distance_m: float) -> bool:  # pragma: no cover
        ...


# ─── Baseline ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BaselineTrackingPolicy:
    """Visual + VHF voice — Eqs. (3.8) and (3.12).

    ``dim_ref_m`` is retained for YAML backwards-compat but is **deprecated**:
    the angular-acuity formula does not depend on
    it. New code must consume ``visual_acuity_rad`` instead.
    """

    update_interval_min_s: float
    update_interval_max_s: float
    max_acquisition_m: float
    dim_ref_m: float  # deprecated — kept for YAML compatibility
    sampling_annulus_min_m: float
    sampling_annulus_max_m: float
    visual_acuity_rad: float = 2.9088e-4
    reaction_time_min_s: float = 12.5
    name: str = "baseline"

    def p_track(
        self,
        distance_m: float,
        archetype: UASArchetypeSpec,
        rng_decision: np.random.Generator,  # unused, kept for protocol symmetry
    ) -> float:
        r"""Baseline ``P_track`` (Eq. 3.12).

        :math:`P_{\text{track,bsl}}(d, L_U) = \min(1,\; L_U / (d \cdot \theta_{\text{acuity}}))`
        for :math:`d \in (0, d_{\max}]`, and ``0`` otherwise.
        """
        if distance_m <= 0 or distance_m > self.max_acquisition_m:
            return 0.0
        sigma_s = distance_m * self.visual_acuity_rad
        if sigma_s == 0.0:
            # Subnormal-distance underflow: L_U / 0 is the saturation limit of
            # the angular-acuity formula, which clips to 1.0 anyway.
            return 1.0
        return float(min(1.0, archetype.dimension_m / sigma_s))

    def first_update_interval(self, rng_decision: np.random.Generator) -> float:
        """First tracking update — enforces 12.5 s pilot reaction floor (Table 3.3, MOP-2)."""
        raw = float(rng_decision.uniform(self.update_interval_min_s, self.update_interval_max_s))
        return max(self.reaction_time_min_s, raw)

    def next_update_interval(self, rng_decision: np.random.Generator) -> float:
        return float(
            rng_decision.uniform(self.update_interval_min_s, self.update_interval_max_s)
        )

    def in_sampling_annulus(self, distance_m: float) -> bool:
        return self.sampling_annulus_min_m < distance_m <= self.sampling_annulus_max_m


# ─── Modernized ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModernizedTrackingPolicy:
    """EO/IR + M3AR datalink — Eqs. (3.9) and (3.13)."""

    update_interval_min_s: float
    update_interval_max_s: float
    eoir_min_m: float
    eoir_max_m: float
    eoir_accuracy_m: float
    # deprecated: unused by p_track(); kept for YAML compatibility
    p_fw: float
    # deprecated: unused by p_track(); kept for YAML compatibility
    p_mr: float
    sampling_annulus_min_m: float
    sampling_annulus_max_m: float
    name: str = "modernized"

    def p_track(
        self,
        distance_m: float,
        archetype: UASArchetypeSpec,
        rng_decision: np.random.Generator,  # unused — closed-form expected value
    ) -> float:
        """Modernized ``P_track`` per Eq. (3.13): ``min(1, L_U / σ_s)``.

        Constant throughout the EO/IR operating annulus ``(eoir_min, eoir_max]``.
        Returns 0 outside that annulus.
        """
        if distance_m <= self.eoir_min_m or distance_m > self.eoir_max_m:
            return 0.0
        # Eq. (3.13): P_track,M = min(1, L_U / σ_s) — constant in annulus [eoir_min, eoir_max]
        return float(min(1.0, archetype.dimension_m / self.eoir_accuracy_m))

    def first_update_interval(self, rng_decision: np.random.Generator) -> float:
        """No pilot reaction delay for A-29N — delegates to next_update_interval."""
        return self.next_update_interval(rng_decision)

    def next_update_interval(self, rng_decision: np.random.Generator) -> float:
        return float(
            rng_decision.uniform(self.update_interval_min_s, self.update_interval_max_s)
        )

    def in_sampling_annulus(self, distance_m: float) -> bool:
        return self.sampling_annulus_min_m < distance_m <= self.sampling_annulus_max_m


# ─── Factory ───────────────────────────────────────────────────────────────


def make_tracking_policy(cfg: MissionConfig, config_name: str) -> TrackingPolicy:
    """Build the tracking policy for ``baseline`` or ``modernized``."""
    if config_name == "baseline":
        return BaselineTrackingPolicy(
            update_interval_min_s=cfg.baseline_update_interval_s[0],
            update_interval_max_s=cfg.baseline_update_interval_s[1],
            max_acquisition_m=cfg.baseline_max_acquisition_m,
            dim_ref_m=cfg.baseline_dim_ref_m,
            sampling_annulus_min_m=cfg.baseline_envelope_m[1],
            sampling_annulus_max_m=cfg.baseline_max_acquisition_m,
            visual_acuity_rad=cfg.baseline_visual_acuity_rad,
            reaction_time_min_s=cfg.baseline_reaction_time_min_s,
        )
    if config_name == "modernized":
        return ModernizedTrackingPolicy(
            update_interval_min_s=cfg.modernized_update_interval_s[0],
            update_interval_max_s=cfg.modernized_update_interval_s[1],
            eoir_min_m=cfg.modernized_eoir_min_m,
            eoir_max_m=cfg.modernized_eoir_max_m,
            eoir_accuracy_m=cfg.modernized_eoir_accuracy_m,
            p_fw=cfg.modernized_track_p_fw,
            p_mr=cfg.modernized_track_p_mr,
            sampling_annulus_min_m=cfg.modernized_envelope_m[1],
            sampling_annulus_max_m=cfg.modernized_eoir_max_m,
        )
    raise ValueError(f"Unknown config name: {config_name!r}")
