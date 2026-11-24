"""Mission orchestrator — wires the SimPy environment and all agents.

A single :meth:`MissionSimulator.run_paired` call:

1. Builds the projection, border polygon, AoR map.
2. Builds the per-config tracking + weapon policies.
3. Spins up the SimPy environment, the world-shared spawner, two
   E-99M dispatchers, and 12 A-29s (6 per configuration).
4. Runs until both configurations close their PEA windows.
5. Returns the run logger + the aggregated metrics dataframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import simpy

from cuas_sim.agents.a29 import A29
from cuas_sim.agents.e99m_dispatcher import E99MDispatcher, PendingPool
from cuas_sim.geometry.projection import LocalProjection
from cuas_sim.metrics.aggregator import MetricsAggregator
from cuas_sim.metrics.records import RunLogger
from cuas_sim.metrics.replanning import ReplanningSignal, build_replanning_signal
from cuas_sim.policies.tracking import make_tracking_policy
from cuas_sim.policies.weapon import make_weapon_policy
from cuas_sim.rng import RNGRegistry
from cuas_sim.world.aor import AoRMap
from cuas_sim.world.arrivals import UASSpawner
from cuas_sim.world.border_polygon import default_border_polygon

if TYPE_CHECKING:
    import pandas as pd

    from cuas_sim.config import MissionConfig


@dataclass(slots=True)
class PairedResult:
    """Bundle returned by a paired baseline + modernized run."""

    cfg: MissionConfig
    seed: int
    log: RunLogger
    metrics: pd.DataFrame
    replanning_signal: ReplanningSignal
    total_arrivals: int


class MissionSimulator:
    """One-shot mission simulator (a fresh instance per paired run)."""

    def __init__(self, cfg: MissionConfig) -> None:
        self.cfg = cfg
        self.projection = LocalProjection(
            lat0_deg=cfg.aoo_centroid_lat,
            lon0_deg=cfg.aoo_centroid_lon,
        )
        self.polygon = default_border_polygon(cfg)
        self.aor_map = AoRMap.from_config(cfg)

    def run_paired(
        self,
        seed: int,
        *,
        horizon_s: float | None = None,
        max_arrivals: int | None = None,
        mos1_target: float = 0.80,
    ) -> PairedResult:
        """Execute one paired baseline + modernized run."""
        rng = RNGRegistry.from_master_seed(seed)
        log = RunLogger()

        env = simpy.Environment()

        # Per-config dispatch stores: each fed by the world-shared spawner.
        store_b: simpy.Store = simpy.Store(env)
        store_m: simpy.Store = simpy.Store(env)

        # Per-config first-bingo events; ``AllOf``
        # closes the spawn window once both configs have produced a bingo.
        baseline_first_bingo = env.event()
        modernized_first_bingo = env.event()
        end_window: simpy.Event | None
        if self.cfg.uas_spawn_window_enabled:
            end_window = simpy.events.AllOf(
                env, [baseline_first_bingo, modernized_first_bingo]
            )
        else:
            end_window = None  # legacy spawner runs until horizon_s

        # World-shared spawner pushes each UASBirth into BOTH stores.
        spawner = UASSpawner(cfg=self.cfg, polygon=self.polygon, rng=rng, run_logger=log)
        env.process(spawner.run(
            env, sinks=[store_b, store_m],
            horizon_s=horizon_s, end_event=end_window, max_arrivals=max_arrivals,
        ))

        # Per-config A-29 inboxes
        inboxes_b: dict[str, simpy.Store] = {
            a29_id: simpy.Store(env) for a29_id in self.cfg.a29_schedules
        }
        inboxes_m: dict[str, simpy.Store] = {
            a29_id: simpy.Store(env) for a29_id in self.cfg.a29_schedules
        }

        # Precompute CAP XY positions (shared across both configs)
        cap_xy_map: dict[str, tuple[float, float]] = {
            a29_id: (lambda xy: (xy.x, xy.y))(
                self.projection.project(self.cfg.caps[f"CAP-{schedule.cap_index}"])
            )
            for a29_id, schedule in self.cfg.a29_schedules.items()
        }

        # A-29 agents — built first so dispatcher can hold references to query .is_free
        agents_b: dict[str, A29] = {}
        agents_m: dict[str, A29] = {}

        # One pending pool per configuration, shared by that
        # config's dispatcher and its six A-29s.
        pools: dict[str, PendingPool] = {}

        for config_name, inboxes, agents, first_bingo_evt in (
            ("baseline",   inboxes_b, agents_b, baseline_first_bingo),
            ("modernized", inboxes_m, agents_m, modernized_first_bingo),
        ):
            tracking_policy = make_tracking_policy(self.cfg, config_name)
            weapon_policy = make_weapon_policy(self.cfg, config_name)
            per_cfg_rng = rng.per_config(config_name)
            pool = PendingPool(
                config_name=config_name, projection=self.projection,
                run_logger=log,
            )
            pools[config_name] = pool
            for a29_id, schedule in self.cfg.a29_schedules.items():
                aor_name = self.aor_map.aor_of_a29(a29_id)
                agent = A29(
                    a29_id=a29_id, schedule=schedule, aor_name=aor_name,
                    cfg=self.cfg, aor_map=self.aor_map,
                    projection=self.projection,
                    tracking=tracking_policy, weapon=weapon_policy,
                    rng=per_cfg_rng, run_logger=log, config_name=config_name,
                    pending_pool=pool,
                    # Shared first-bingo handle per config.
                    first_bingo_event=first_bingo_evt,
                )
                agents[a29_id] = agent
                env.process(agent.run(env, inboxes[a29_id]))
            # Let each A-29 see its config's fleet so a bingo
            # abort can hand the released UAS to a free teammate (else it leaks).
            for agent in agents.values():
                agent.fleet = agents

        # Dispatchers (after agents are created so references are valid)
        for config_name, store, inboxes, agents in (
            ("baseline",   store_b, inboxes_b, agents_b),
            ("modernized", store_m, inboxes_m, agents_m),
        ):
            env.process(E99MDispatcher(
                config_name=config_name,
                aor_map=self.aor_map,
                a29_agents=agents,
                cap_xy_map=cap_xy_map,
                projection=self.projection,
                pending_pool=pools[config_name],
            ).run(env, inbox=store, outboxes=inboxes))

        # Effective horizon defaults to the latest mission completion.
        if horizon_s is None:
            horizon_s = self._compute_default_horizon()

        env.run(until=horizon_s)

        # Build the ``window.json`` payload
        # from the post-run state of the AllOf event and the spawner log.
        window_payload = self._build_window_payload(
            log=log,
            baseline_first_bingo=baseline_first_bingo,
            modernized_first_bingo=modernized_first_bingo,
            horizon_s=horizon_s,
            max_arrivals=max_arrivals,
        )
        log.set_window(window_payload)

        # Aggregation
        agg = MetricsAggregator(self.cfg)
        # Total arrivals: derived from the engagement + leak log
        total_arrivals = self._count_arrivals(log)
        metrics = agg.compute_all(log, total_arrivals=total_arrivals)
        replanning = build_replanning_signal(self.cfg, metrics, log, mos1_target=mos1_target)

        return PairedResult(
            cfg=self.cfg, seed=seed, log=log,
            metrics=metrics, replanning_signal=replanning,
            total_arrivals=total_arrivals,
        )

    # ─── helpers ─────────────────────────────────────────────────────────

    def _compute_default_horizon(self) -> float:
        """Latest of (T_PEA_start + phase4 + phase5 + phase6) across A-29s."""
        end_times = [
            self.cfg.t_pea_start_s + s.phase4_pea_s + s.phase5_cap_to_farp_s
            + s.phase6_farp_to_sbpv_s
            for s in self.cfg.a29_schedules.values()
        ]
        return float(max(end_times)) + 60.0  # small safety margin

    def _build_window_payload(
        self,
        *,
        log: RunLogger,
        baseline_first_bingo: simpy.Event,
        modernized_first_bingo: simpy.Event,
        horizon_s: float,
        max_arrivals: int | None,
    ) -> dict[str, object]:
        """Build the ``window.json`` payload.

        Determines ``t_window_end_reason`` from the post-run state of the
        per-config first-bingo events and the spawner log; ``t_window_end_s``
        is ``min(t_AllOf, horizon_s, t_max_arrivals)``.
        """
        bsl_done = baseline_first_bingo.triggered
        mod_done = modernized_first_bingo.triggered
        bsl_payload: dict[str, Any] | None = (
            baseline_first_bingo.value if bsl_done else None
        )
        mod_payload: dict[str, Any] | None = (
            modernized_first_bingo.value if mod_done else None
        )

        if not self.cfg.uas_spawn_window_enabled:
            t_window_end_reason = "legacy_horizon"
            t_window_end_s = float(horizon_s)
        elif bsl_done and mod_done:
            assert bsl_payload is not None and mod_payload is not None
            t_window_end_reason = "all_first_bingo"
            t_window_end_s = max(
                float(bsl_payload["t_s"]), float(mod_payload["t_s"]),
            )
        elif (
            max_arrivals is not None
            and len(log.uas_births) >= max_arrivals
        ):
            t_window_end_reason = "max_arrivals"
            t_window_end_s = (
                float(log.uas_births[-1]["spawn_time_s"]) if log.uas_births
                else float(self.cfg.t_pea_start_s)
            )
        else:
            t_window_end_reason = "horizon_cap"
            t_window_end_s = float(horizon_s)

        per_a29: dict[str, dict[str, float]] = {"baseline": {}, "modernized": {}}
        for (cfg_name, a29_id), t_s in log.t_rtb_fuel_limit_per_a29.items():
            per_a29.setdefault(cfg_name, {})[a29_id] = float(t_s)

        return {
            "field_version": "1.0",
            "uas_spawn_window_enabled": self.cfg.uas_spawn_window_enabled,
            "t_window_start_s": float(self.cfg.t_pea_start_s),
            "t_window_end_s": float(t_window_end_s),
            "t_window_end_reason": t_window_end_reason,
            "horizon_cap_s": float(horizon_s),
            "baseline_first_bingo": bsl_payload,
            "modernized_first_bingo": mod_payload,
            "t_rtb_fuel_limit_per_a29": per_a29,
        }

    def _count_arrivals(self, log: RunLogger) -> int:
        """Total arrivals = engagements (per config) + unhandled leaks.

        Since the two configurations see the same arrivals (CRN), we use the
        baseline count as the canonical total.
        """
        baseline_eng = sum(1 for e in log.engagements
                           if e.fields_["config_name"] == "baseline")
        baseline_leaks = sum(1 for lk in log.leaks if lk["config_name"] == "baseline")
        return baseline_eng + baseline_leaks
