"""High-level convenience entry points for paired runs."""

from __future__ import annotations

import logging
from pathlib import Path

from cuas_sim.config import MissionConfig
from cuas_sim.simulation.mission import MissionSimulator, PairedResult
from cuas_sim.simulation.persistence import RunDirectory, persist_run

logger = logging.getLogger(__name__)


def run_paired(
    cfg: MissionConfig,
    seed: int,
    *,
    outputs_root: Path | None = None,
    slug: str | None = None,
    horizon_s: float | None = None,
    max_arrivals: int | None = None,
    mos1_target: float = 0.80,
    persist: bool = True,
) -> PairedResult:
    """Execute one paired run and (optionally) persist it.

    Parameters
    ----------
    cfg
        Resolved mission configuration (already merged with SysML overrides).
    seed
        Master seed for the CRN registry.
    outputs_root
        Root directory for runs. Defaults to ``./outputs`` if persisting.
    slug
        Free-form slug for the run-directory name.
    horizon_s
        Stop time in simulation seconds. ``None`` → derived from schedules.
    max_arrivals
        Safety cap on the spawner.
    mos1_target
        Target threshold for the replanning signal.
    persist
        If ``True``, write the three-layer outputs to disk.
    """
    sim = MissionSimulator(cfg)
    result = sim.run_paired(
        seed=seed, horizon_s=horizon_s, max_arrivals=max_arrivals,
        mos1_target=mos1_target,
    )

    if persist:
        out = outputs_root or Path("outputs")
        slug = slug or f"seed{seed}_lam{cfg.lambda_per_sec * 3600:.2f}ph"
        run_dir = RunDirectory.create(out, slug=slug)
        persist_run(result, run_dir)
        logger.info("Run persisted to %s", run_dir.root)

    return result
