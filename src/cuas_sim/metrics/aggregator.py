"""MOS/MOE/MOP aggregator.

All formulas trace to the case study measures (Section 3.3.4; map in ``docs/metrics.md``):

* **MOS-1** (Eq. 3.2): Neutralization rate within the ZIDA.

  .. math::

       \\mathrm{MOS\\text{-}1} = \\frac{\\#\\{\\text{kills inside ZIDA}\\}}
                                       {\\#\\{\\text{total incursions}\\}}

* **MOS-2** (Eqs. 3.3–3.4): Evasion-reduction rate. The denominator is the baseline
  evasion count, so the value lies in (−∞, 1]; with no baseline evasion it is 1 if the
  modernized configuration has none either, and 0 otherwise:

  .. math::

       \\mathrm{MOS\\text{-}2} = 1 - \\frac{\\text{leaks}_\\text{mod}}
                                          {\\text{leaks}_\\text{bsl}}

* **MOE-1** (Eqs. 3.5–3.7): engagement-time **rate**
  :math:`1 - \\frac{1}{N} \\sum_i t_{ef,i} / T_{\\text{PEA},i}`, clamped to [0, 1], with
  the evasion penalty of Eq. 3.6 and each aircraft's own ``T_PEA`` (Eq. 3.7).
* **MOE-2** (Eqs. 3.10–3.11): Mean of per-engagement :math:`P_\\text{track}` averages over
  the annulus between the employment and acquisition ranges (zero without samples).
* **MOE-3** (Eqs. 3.15–3.16): Mean per-engagement **cumulative neutralization** probability
  :math:`P_{\\text{neut},i} = 1 - \\prod_k (1 - P_{\\text{hit},k})` over the planned shots.
* **MOP-1** (Eq. 3.17): Mean heading-update count per engagement.
* **MOP-2** (Eq. 3.18): Maximum acquisition distance (case study Table 3.3).
* **MOP-3** (Eq. 3.19): Weapon envelope maximum (case study Table 3.3).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from cuas_sim.geometry.haversine import t_pea_seconds

if TYPE_CHECKING:
    from cuas_sim.config import MissionConfig
    from cuas_sim.metrics.records import RunLogger


@dataclass(slots=True)
class MetricsAggregator:
    """Builds the long-format ``metrics.parquet`` from a :class:`RunLogger`."""

    cfg: MissionConfig
    # Captured by ``compute_all`` so ``_t_pea_for_aor`` can
    # read ``t_rtb_fuel_limit_per_a29``. ``None`` keeps standalone callers
    # (unit tests of helper methods) working — they default to the Haversine
    # fallback.
    log: RunLogger | None = field(default=None)

    # ─── Top level ───────────────────────────────────────────────────────

    def compute_all(self, log: RunLogger, total_arrivals: int) -> pd.DataFrame:
        """Compute every metric per config and per (config, AoR).

        Returns
        -------
        DataFrame
            Long format: ``config_name``, ``aor`` (or "ALL"), ``metric``, ``value``.
        """
        # Make the RunLogger available to ``_t_pea_for_aor``
        # without changing the signature of the helper loops in
        # ``_compute_moe1``.
        self.log = log
        rows: list[dict[str, float | str]] = []
        eng_df = self._engagements_df(log)

        for config_name in ("baseline", "modernized"):
            cfg_df = eng_df[eng_df["config_name"] == config_name] if not eng_df.empty else eng_df

            # MOS-1 (overall and per AoR)
            rows.extend(self._compute_mos1(cfg_df, config_name, total_arrivals, log))
            # MOE-1 / MOE-2 / MOE-3 (per AoR, overall is the mean)
            rows.extend(self._compute_moe1(cfg_df, config_name))
            rows.extend(self._compute_moe2(cfg_df, config_name))
            rows.extend(self._compute_moe3(cfg_df, config_name))
            # MOP-1 (per AoR, overall mean)
            rows.extend(self._compute_mop1(cfg_df, config_name))
            # MOP-2 / MOP-3 (config-level constants)
            rows.append(self._mop2_row(config_name))
            rows.append(self._mop3_row(config_name))

        # MOS-2 (a single number per run, cross-config)
        rows.extend(self._compute_mos2(eng_df, total_arrivals, log))

        return pd.DataFrame(rows)

    # ─── MOS-1 ───────────────────────────────────────────────────────────

    def _compute_mos1(
        self, cfg_df: pd.DataFrame, config_name: str,
        total_arrivals: int, log: RunLogger,
    ) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []

        if cfg_df.empty or total_arrivals == 0:
            rows.append({"config_name": config_name, "aor": "ALL",
                         "metric": "MOS-1", "value": 0.0})
            return rows

        # Count per-AoR (denominator: incursions PER AoR, computed from the
        # full arrival set, not just engagements — leaks count too).
        # Engagements give us the AoR of every UAS that was at least notified;
        # we add the unhandled leaks (rare under the idealized E-99M).
        eng_per_aor = cfg_df.groupby("aor_name")
        kills_per_aor = eng_per_aor["is_neutralized_before_zida"].sum().to_dict()
        eng_count_per_aor = eng_per_aor.size().to_dict()

        # Total incursions per AoR = engagements + unhandled leaks for that config
        unhandled = [lk for lk in log.leaks if lk["config_name"] == config_name]
        leaks_per_aor: dict[str, int] = defaultdict(int)
        for leak in unhandled:
            # Map sector → AoR via the cfg (best-effort)
            for aor_name, secs in self.cfg.aor_to_sectors.items():
                if leak["sector_index"] in secs:
                    leaks_per_aor[aor_name] += 1
                    break

        all_aors = set(eng_count_per_aor) | set(leaks_per_aor)
        total_kills = 0
        total_incursions = 0
        for aor in sorted(all_aors):
            denom = int(eng_count_per_aor.get(aor, 0)) + int(leaks_per_aor.get(aor, 0))
            kills = int(kills_per_aor.get(aor, 0))
            total_kills += kills
            total_incursions += denom
            mos1 = (kills / denom) if denom > 0 else 0.0
            rows.append({"config_name": config_name, "aor": aor,
                         "metric": "MOS-1", "value": float(mos1)})

        mos1_all = (total_kills / total_incursions) if total_incursions > 0 else 0.0
        rows.append({"config_name": config_name, "aor": "ALL",
                     "metric": "MOS-1", "value": float(mos1_all)})
        return rows

    # ─── MOS-2 ───────────────────────────────────────────────────────────

    def _compute_mos2(
        self, eng_df: pd.DataFrame, total_arrivals: int, log: RunLogger,
    ) -> list[dict[str, float | str]]:
        leaks_b = self._count_leaks(eng_df, "baseline", log)
        leaks_m = self._count_leaks(eng_df, "modernized", log)
        mos2 = (1.0 if leaks_m == 0 else 0.0) if leaks_b == 0 else 1.0 - leaks_m / leaks_b
        return [{"config_name": "modernized_vs_baseline", "aor": "ALL",
                 "metric": "MOS-2", "value": float(mos2)}]

    def _count_leaks(self, eng_df: pd.DataFrame, config_name: str, log: RunLogger) -> int:
        """Total incursions that were NOT neutralized inside the ZIDA."""
        if eng_df.empty:
            return sum(1 for lk in log.leaks if lk["config_name"] == config_name)
        sub = eng_df[eng_df["config_name"] == config_name]
        engaged_leaks = int((~sub["is_neutralized_before_zida"]).sum())
        unhandled_leaks = sum(1 for lk in log.leaks if lk["config_name"] == config_name)
        return engaged_leaks + unhandled_leaks

    # ─── MOE-1 ───────────────────────────────────────────────────────────

    def _compute_moe1(
        self, cfg_df: pd.DataFrame, config_name: str,
    ) -> list[dict[str, float | str]]:
        r"""MOE-1 completion-time **rate** (case study Eqs. 3.5–3.7).

        .. math::

            \mathrm{MOE\text{-}1} = 1 - \frac{1}{N}\sum_i
                \frac{t_{ef,i}}{T_{\mathrm{PEA},i}}

        Per-engagement numerator (Eq. 3.6):

        * **neutralised**: :math:`t_{ef,i} = t_{neut,i} - t_{not,i}`
          (= ``moe1_engagement_time_s``);
        * **evaded**: :math:`t_{ef,i} = \max(0,\, t_{ev,i} - t_{not,i})\,(1 + E/N)`.

        Every logged engagement enters ``N``
        and every **non-neutralised** one is *evaded* (counted in ``E`` and
        penalised); pure leaks are out of scope. ``T_PEA`` is the per-aircraft
        emergent value. The final per-AoR / ALL value is the
        Eq. 3.5 complement ``1 - mean(ratio)``, clamped to [0,1] when
        ``clip_moe1_to_01``.
        """
        rows: list[dict[str, float | str]] = []
        if cfg_df.empty:
            rows.append({"config_name": config_name, "aor": "ALL",
                         "metric": "MOE-1", "value": 0.0})
            return rows

        # Config-level evasion penalty (Eq. 3.6): E evasions over N engagements.
        n_total = len(cfg_df)
        e_evasions = int((~cfg_df["is_neutralized"].astype(bool)).sum())
        penalty = 1.0 + (e_evasions / n_total if n_total > 0 else 0.0)

        ratios: list[float] = []
        per_aor_ratios: dict[str, list[float]] = defaultdict(list)
        for _, row in cfg_df.iterrows():
            aor = row["aor_name"]
            t_pea = self._t_pea_for_aor(aor, row["config_name"])
            if t_pea <= 0:
                continue
            q = self._engagement_tef(row, penalty) / t_pea
            ratios.append(q)
            per_aor_ratios[aor].append(q)

        for aor, vals in per_aor_ratios.items():
            rows.append({"config_name": config_name, "aor": aor,
                         "metric": "MOE-1", "value": self._moe1_from_ratios(vals)})
        rows.append({"config_name": config_name, "aor": "ALL",
                     "metric": "MOE-1", "value": self._moe1_from_ratios(ratios)})
        return rows

    @staticmethod
    def _engagement_tef(row: pd.Series, penalty: float) -> float:
        """Per-engagement Eq. 3.6 numerator :math:`t_{ef}` in seconds.

        Neutralised ⇒ ``moe1_engagement_time_s`` (= ``t_kill − t_notify``).
        Evaded ⇒ ``max(0, t_ev − t_notify)·(1 + E/N)``. When ``t_ev_s`` is
        missing/NaN (e.g. legacy synthetic records), the
        numerator base falls back to ``moe1_engagement_time_s``.
        """
        if bool(row["is_neutralized"]):
            return float(row["moe1_engagement_time_s"])
        t_not = float(row["t_pursuit_start_s"])
        t_ev = float(row.get("t_ev_s", float("nan")))
        base = (t_ev - t_not) if math.isfinite(t_ev) else float(row["moe1_engagement_time_s"])
        return max(0.0, base) * penalty

    def _moe1_from_ratios(self, ratios: list[float]) -> float:
        """Eq. 3.5 complement ``1 - mean(ratios)``, clamped to [0,1]."""
        if not ratios:
            return 0.0
        value = 1.0 - sum(ratios) / len(ratios)
        if self.cfg.clip_moe1_to_01:
            value = min(1.0, max(0.0, value))
        return float(value)

    def _t_pea_for_aor(
        self, aor_name: str, config_name: str | None = None,
    ) -> float:
        """:math:`T_\\text{PEA,k}` for the A-29 assigned to ``aor_name``.

        When ``uas_spawn_window_enabled=True`` **and** the
        ``RunLogger`` recorded ``RTB_FUEL_LIMIT`` for that (config, A-29),
        return the emergent value ``t_RTB_FUEL_LIMIT - t_pea_start_s``.
        Otherwise (legacy ``flag=False``, or A-29 whose RTB was not observed —
        per-A-29 fallback) return the nominal value of
        :func:`cuas_sim.geometry.haversine.t_pea_seconds`. The emergent value is
        Eq. (3.7); the MOE-1 ratio loop in :meth:`_compute_moe1` is the same
        whichever source applies.
        """
        a29_id = self.cfg.aor_to_a29.get(aor_name)
        if a29_id is None:
            return 0.0
        # Emergent T_PEA when enabled and observed.
        if (
            self.cfg.uas_spawn_window_enabled
            and config_name is not None
            and self.log is not None
        ):
            t_rtb = self.log.t_rtb_fuel_limit_per_a29.get((config_name, a29_id))
            if t_rtb is not None:
                return float(t_rtb) - float(self.cfg.t_pea_start_s)
        # Fallback: nominal Haversine value (legacy path, or RTB not observed).
        sched = self.cfg.a29_schedules[a29_id]
        farp = self.cfg.farps[sched.farp_name]
        cap = self.cfg.caps[f"CAP-{sched.cap_index}"]
        return t_pea_seconds(
            farp=farp, cap=cap,
            endurance_s=self.cfg.a29_endurance_s,
            cruise_speed_mps=self.cfg.a29_cruise_speed_mps,
        )

    # ─── MOE-2 ───────────────────────────────────────────────────────────

    def _compute_moe2(
        self, cfg_df: pd.DataFrame, config_name: str,
    ) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        if cfg_df.empty:
            rows.append({"config_name": config_name, "aor": "ALL",
                         "metric": "MOE-2", "value": 0.0})
            return rows
        # Per-engagement: mean of in-annulus P_track samples (already
        # computed into moe2_mean_per_engagement). If n_samples == 0, the
        # value is 0 (Eq. 3.10).
        per_aor = cfg_df.groupby("aor_name")
        for aor, sub in per_aor:
            v = float(sub["moe2_mean_per_engagement"].mean())
            rows.append({"config_name": config_name, "aor": str(aor),
                         "metric": "MOE-2", "value": v})
        v_all = float(cfg_df["moe2_mean_per_engagement"].mean())
        rows.append({"config_name": config_name, "aor": "ALL",
                     "metric": "MOE-2", "value": v_all})
        return rows

    # ─── MOE-3 ───────────────────────────────────────────────────────────

    def _compute_moe3(
        self, cfg_df: pd.DataFrame, config_name: str,
    ) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        if cfg_df.empty:
            rows.append({"config_name": config_name, "aor": "ALL",
                         "metric": "MOE-3", "value": 0.0})
            return rows
        # ``moe3_p_hit_expected`` now carries the per-engagement
        # cumulative P_neut,i (Eq. 3.15). MOE-3 is its mean (Eq. 3.16);
        # the field name predates the multi-shot model.
        per_aor = cfg_df.groupby("aor_name")
        for aor, sub in per_aor:
            v = float(sub["moe3_p_hit_expected"].mean())
            rows.append({"config_name": config_name, "aor": str(aor),
                         "metric": "MOE-3", "value": v})
        v_all = float(cfg_df["moe3_p_hit_expected"].mean())
        rows.append({"config_name": config_name, "aor": "ALL",
                     "metric": "MOE-3", "value": v_all})
        return rows

    # ─── MOP-1 ───────────────────────────────────────────────────────────

    def _compute_mop1(
        self, cfg_df: pd.DataFrame, config_name: str,
    ) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        if cfg_df.empty:
            rows.append({"config_name": config_name, "aor": "ALL",
                         "metric": "MOP-1", "value": 0.0})
            return rows
        per_aor = cfg_df.groupby("aor_name")
        for aor, sub in per_aor:
            v = float(sub["mop1_heading_update_count"].mean())
            rows.append({"config_name": config_name, "aor": str(aor),
                         "metric": "MOP-1", "value": v})
        v_all = float(cfg_df["mop1_heading_update_count"].mean())
        rows.append({"config_name": config_name, "aor": "ALL",
                     "metric": "MOP-1", "value": v_all})
        return rows

    # ─── MOP-2 / MOP-3 (config-level constants) ─────────────────────────

    def _mop2_row(self, config_name: str) -> dict[str, float | str]:
        v = self.cfg.baseline_max_acquisition_m if config_name == "baseline" \
            else self.cfg.modernized_eoir_max_m
        return {"config_name": config_name, "aor": "ALL",
                "metric": "MOP-2", "value": float(v)}

    def _mop3_row(self, config_name: str) -> dict[str, float | str]:
        env = self.cfg.baseline_envelope_m if config_name == "baseline" \
            else self.cfg.modernized_envelope_m
        return {"config_name": config_name, "aor": "ALL",
                "metric": "MOP-3", "value": float(env[1])}

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _engagements_df(log: RunLogger) -> pd.DataFrame:
        if not log.engagements:
            return pd.DataFrame()
        return pd.DataFrame([e.fields_ for e in log.engagements])
