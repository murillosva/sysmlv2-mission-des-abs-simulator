"""Poisson UAS spawner — the world-shared arrival process.

A SimPy generator that emits one ``UASBirth`` event per realisation of the
homogeneous Poisson process with rate :math:`\\lambda` (per second). The
spawner is **world-shared**: both the baseline and modernized configurations
of a paired run consume the same arrival sequence in lockstep via SimPy
``Store``s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cuas_sim.geometry.coordinates import GeoPoint
from cuas_sim.world.border_polygon import BorderPolygon, sample_sector

if TYPE_CHECKING:
    import simpy

    from cuas_sim.config import MissionConfig, UASArchetypeSpec
    from cuas_sim.metrics.records import RunLogger
    from cuas_sim.rng import RNGRegistry


@dataclass(frozen=True, slots=True)
class UASBirth:
    """A single UAS arrival realisation produced by the spawner.

    All four random draws (sector, spawn point, heading, archetype) are
    pre-resolved at birth time so consumers (paired baseline + modernized)
    see identical inputs.
    """

    uas_id: int
    spawn_time_s: float
    sector_index: int
    point: GeoPoint
    heading_rad: float
    archetype: UASArchetypeSpec
    omega: float


def uas_spawner(
    env: simpy.Environment,
    cfg: MissionConfig,
    polygon: BorderPolygon,
    rng: RNGRegistry,
    sinks: list[simpy.Store],
    *,
    horizon_s: float | None = None,
    end_event: simpy.Event | None = None,
    max_arrivals: int | None = None,
    run_logger: RunLogger | None = None,
) -> Generator[Any, Any, None]:
    """SimPy generator that produces UAS births and pushes them to each sink.

    When :class:`MissionSimulator` passes an
    ``end_event=simpy.AllOf([baseline_first_bingo, modernized_first_bingo])``,
    the spawner stops emitting once both configs have had their first
    ``RTB_FUEL_LIMIT``. The start gate ``cfg.t_pea_start_s`` equals
    ``max_{i}(phase1+phase2)`` over the A-29 schedule by construction
    (validated in :meth:`MissionConfig.__post_init__`); no code change is
    needed for the start.

    Parameters
    ----------
    env
        SimPy environment.
    cfg
        Mission configuration. ``cfg.lambda_per_sec`` is the Poisson rate.
    polygon
        Border polygon (the 27 BR-CO/BR-PE sectors, indexed 1..27 north→south).
    rng
        CRN registry — only the four world-shared streams are used here.
    sinks
        One SimPy ``Store`` per paired configuration. Each birth is pushed to
        **every** sink, so the two configurations consume the same sequence.
    horizon_s
        Wall-clock stopping time (in simulation seconds). If ``None``,
        relies on ``end_event``.
    end_event
        Alternative stopping condition (e.g. a SimPy event fired when
        Phase 4 ends in both configs, or the ``AllOf`` of per-config
        first-bingo events).
    max_arrivals
        Safety cap on the total number of arrivals (default: no cap).
    """
    uas_id = 0
    # Gate: hold spawner until PEA begins. Pre-PEA UAS are filtered by each
    # A-29 agent anyway (a29.py line 90), but emitting them fills stores and
    # shows them in the animation before any A-29 is on station.
    pea_wait = cfg.t_pea_start_s - float(env.now)
    if pea_wait > 0:
        yield env.timeout(pea_wait)
    while True:
        # Inter-arrival sample — Poisson process ⇒ Exp(1/λ).
        # Using exponential(scale=1/λ) keeps the draw consistent regardless
        # of the underlying generator's float internals.
        dt = float(rng.arrivals.exponential(scale=1.0 / cfg.lambda_per_sec))
        yield env.timeout(dt)

        if horizon_s is not None and env.now > horizon_s:
            return
        if end_event is not None and end_event.triggered:
            return
        if max_arrivals is not None and uas_id >= max_arrivals:
            return

        uas_id += 1
        sector_idx = sample_sector(polygon, rng.sector_choice,
                                   mu=cfg.sector_mu, sigma=cfg.sector_sigma,
                                   rule=cfg.sector_draw_rule)
        draw = polygon.spawn_in_sector(
            sector_idx, rng.spawn_geometry, cfg.heading_perpendicular_deviation_deg,
            interpolation=cfg.spawn_interpolation,
            heading_rule=cfg.heading_inward_rule,
        )
        is_mr = bool(rng.archetype.random() < cfg.uas_archetype_p_mr)
        archetype = cfg.uas_mr if is_mr else cfg.uas_fw

        birth = UASBirth(
            uas_id=uas_id,
            spawn_time_s=float(env.now),
            sector_index=sector_idx,
            point=draw.point,
            heading_rad=draw.heading_rad,
            archetype=archetype,
            omega=draw.omega,
        )
        if run_logger is not None:
            run_logger.log_uas_birth(
                uas_id=birth.uas_id,
                spawn_time_s=birth.spawn_time_s,
                lat=birth.point.lat,
                lon=birth.point.lon,
                heading_rad=birth.heading_rad,
                speed_mps=birth.archetype.max_speed_mps,
                archetype_name=birth.archetype.name,
                sector_index=birth.sector_index,
            )
        for s in sinks:
            s.put(birth)


# Convenience alias for the generator-yield typing
from collections.abc import Generator  # noqa: E402  (placed after to avoid stub clash)

# ─── Spawner thin wrapper class (for the orchestrator) ────────────────────


@dataclass(slots=True)
class UASSpawner:
    """OO wrapper around :func:`uas_spawner` for the simulation orchestrator."""

    cfg: MissionConfig
    polygon: BorderPolygon
    rng: RNGRegistry
    run_logger: RunLogger | None = None

    def run(
        self,
        env: simpy.Environment,
        sinks: list[simpy.Store],
        *,
        horizon_s: float | None = None,
        end_event: simpy.Event | None = None,
        max_arrivals: int | None = None,
    ) -> Generator[Any, Any, None]:
        return uas_spawner(
            env, self.cfg, self.polygon, self.rng, sinks,
            horizon_s=horizon_s, end_event=end_event, max_arrivals=max_arrivals,
            run_logger=self.run_logger,
        )
