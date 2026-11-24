"""Persistence layer — writes the three-layer output tree per run.

Layout::

    outputs/runs/<timestamp>_<slug>/
        ├── run.yaml                  ← seed, λ, config hash, version
        ├── engagements.parquet       ← one row per engagement
        ├── metrics.parquet           ← long-format aggregate metrics
        ├── leaks.parquet             ← unhandled leak records
        ├── spawns.parquet            ← UAS birth positions/headings (for animation replay)
        ├── events_baseline.jsonl.gz
        ├── events_modernized.jsonl.gz
        ├── window.json               ← dynamic spawn window state (v1.0)
        └── replanning_signal.json    ← only if MOS-1 < target
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import yaml

from cuas_sim import __version__
from cuas_sim.simulation.mission import PairedResult


@dataclass(slots=True)
class RunDirectory:
    """Path bundle for a single paired-run output directory."""

    root: Path
    run_yaml: Path
    engagements_parquet: Path
    metrics_parquet: Path
    leaks_parquet: Path
    spawns_parquet: Path
    events_baseline: Path
    events_modernized: Path
    replanning_json: Path
    # Dynamic spawn window artifact (schema v1.0).
    window_json: Path

    @classmethod
    def create(cls, outputs_root: Path, slug: str = "paired") -> RunDirectory:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
        root = outputs_root / "runs" / f"{ts}_{slug}"
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            run_yaml=root / "run.yaml",
            engagements_parquet=root / "engagements.parquet",
            metrics_parquet=root / "metrics.parquet",
            leaks_parquet=root / "leaks.parquet",
            spawns_parquet=root / "spawns.parquet",
            events_baseline=root / "events_baseline.jsonl.gz",
            events_modernized=root / "events_modernized.jsonl.gz",
            replanning_json=root / "replanning_signal.json",
            window_json=root / "window.json",
        )


def persist_run(result: PairedResult, run_dir: RunDirectory) -> None:
    """Serialize a :class:`PairedResult` to disk."""
    # 1. Engagements + leaks + metrics + UAS spawn positions
    result.log.write_engagements_parquet(run_dir.engagements_parquet)
    result.log.write_leaks_parquet(run_dir.leaks_parquet)
    result.log.write_spawns_parquet(run_dir.spawns_parquet)
    result.metrics.to_parquet(run_dir.metrics_parquet, engine="pyarrow", index=False)

    # 2. Event streams
    result.log.write_events_jsonl_gz(run_dir.events_baseline, "baseline")
    result.log.write_events_jsonl_gz(run_dir.events_modernized, "modernized")

    # 3. run.yaml — auditable provenance
    payload = {
        "package_version": __version__,
        "seed": result.seed,
        "lambda_per_sec": result.cfg.lambda_per_sec,
        "lambda_per_hour": result.cfg.lambda_per_sec * 3600.0,
        "total_arrivals_baseline": result.total_arrivals,
        "config_hash": result.cfg.config_hash(),
        "config": result.cfg.to_dict(),
    }
    run_dir.run_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # 4. Replanning signal (only when triggered)
    if result.replanning_signal.triggered:
        result.replanning_signal.write(run_dir.replanning_json)

    # 5. window.json (always emitted, schema v1.0).
    result.log.write_window_json(run_dir.window_json)
