"""λ-sweep — replicates one MissionConfig across (λ, seed) pairs.

Outputs a long-format ``sweep.parquet`` with one row per (λ, seed, config, AoR,
metric, value) combination, suitable for the figures the parent paper will
extend (heatmaps, paired plots, MOE-1/-2/-3 surfaces).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cuas_sim.config import MissionConfig
from cuas_sim.simulation.run import run_paired

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SweepRow:
    """Flat row of a sweep result."""

    fields_: dict[str, Any]


@dataclass(slots=True)
class LambdaSweep:
    """A grid of (λ, seed) pairs over a base :class:`MissionConfig`."""

    base_cfg: MissionConfig
    lambdas_per_hour: list[float]
    seeds: list[int]
    workers: int = 1
    outputs_root: Path = field(default_factory=lambda: Path("outputs"))
    mos1_target: float = 0.80

    def run(self) -> pd.DataFrame:
        """Execute the full sweep, returning the flat dataframe."""
        tasks: list[tuple[int, MissionConfig, int]] = []
        for lam_h in self.lambdas_per_hour:
            cfg = self.base_cfg.replaced(lambda_per_sec=lam_h / 3600.0)
            for seed in self.seeds:
                tasks.append((seed, cfg, seed * 1_000_003 + hash(lam_h) % 997))

        all_rows: list[dict[str, Any]] = []
        if self.workers <= 1:
            for _, cfg, seed in tasks:
                all_rows.extend(self._one_task(cfg, seed))
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(_run_one, cfg, seed, self.mos1_target,
                                       self.outputs_root): (cfg, seed)
                           for _, cfg, seed in tasks}
                for fut in as_completed(futures):
                    cfg, seed = futures[fut]
                    all_rows.extend(fut.result())
                    logger.info("Completed task seed=%d λ=%.2f/h", seed,
                                cfg.lambda_per_sec * 3600)

        df = pd.DataFrame(all_rows)
        out = self.outputs_root / "sweeps"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "sweep.parquet", engine="pyarrow", index=False)
        return df

    def _one_task(self, cfg: MissionConfig, seed: int) -> list[dict[str, Any]]:
        return _run_one(cfg, seed, self.mos1_target, self.outputs_root)


def _run_one(
    cfg: MissionConfig, seed: int, mos1_target: float, outputs_root: Path,
) -> list[dict[str, Any]]:
    """Top-level for ProcessPoolExecutor (must be picklable)."""
    res = run_paired(
        cfg, seed=seed, outputs_root=outputs_root, mos1_target=mos1_target,
        persist=True, slug=f"sweep_seed{seed}_lam{cfg.lambda_per_sec * 3600:.2f}ph",
    )
    out: list[dict[str, Any]] = []
    for _, row in res.metrics.iterrows():
        out.append({
            "lambda_per_hour": cfg.lambda_per_sec * 3600.0,
            "seed": seed,
            **row.to_dict(),
        })
    return out
