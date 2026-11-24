"""Data-farm runner — paired CRN replications -> tidy rows.

Consumes ``MetricsAggregator.compute_all`` via ``run_paired`` (read-only) and never
recomputes MOS/MOE/MOP. ``run_paired`` is always called with
``persist=False``. Cost responses mirror ``scripts/batch_seeds.py``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from cuas_sim.analysis.doe import Factor
from cuas_sim.analysis.execution import ExecutionController, StopFunc
from cuas_sim.config import MissionConfig
from cuas_sim.optimization.loop_controller import _crn_seeds
from cuas_sim.simulation.run import run_paired

FIELD_VERSION = "1.0"

__all__ = ["FIELD_VERSION", "factors_from_names", "run_data_farm", "write_datafarm"]

# Study-factor presets resolved by the CLI.
_STUDY_FACTORS: dict[str, Factor] = {
    "lambda": Factor("lambda_per_h", "numeric", 2.0, 18.0),
    "p_mr": Factor("uas_archetype_p_mr", "numeric", 0.0, 1.0),
}

# Canonical column order + tidy row sort key.
_COLUMNS = [
    "design_point_id", "lambda_per_h", "uas_archetype_p_mr",
    "rep", "seed", "config", "aor", "metric", "value",
]
_SORT_KEY = ["design_point_id", "rep", "config", "aor", "metric"]


def factors_from_names(names: Sequence[str]) -> list[Factor]:
    """Resolve CLI ``--factors`` names to :class:`Factor` presets."""
    out: list[Factor] = []
    for raw in names:
        key = raw.strip()
        if key not in _STUDY_FACTORS:
            known = ", ".join(sorted(_STUDY_FACTORS))
            raise ValueError(f"unknown factor preset {key!r} (known: {known})")
        out.append(_STUDY_FACTORS[key])
    return out


def _design_point_to_config(base_cfg: MissionConfig, row: dict[str, Any]) -> MissionConfig:
    """Map a design-point row to a ``MissionConfig`` via ``.replaced``."""
    changes: dict[str, Any] = {}
    if "lambda_per_h" in row:
        changes["lambda_per_sec"] = float(row["lambda_per_h"]) / 3600.0
    if "uas_archetype_p_mr" in row:
        changes["uas_archetype_p_mr"] = float(row["uas_archetype_p_mr"])
    return base_cfg.replaced(**changes)


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Synthesize paired ``config="delta"`` rows = modernized - baseline."""
    by_key: dict[tuple[Any, Any, Any, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["config"] not in ("baseline", "modernized"):
            continue
        key = (r["design_point_id"], r["rep"], r["aor"], r["metric"])
        by_key[key][r["config"]] = r

    out: list[dict[str, Any]] = []
    for pair in by_key.values():
        if "baseline" in pair and "modernized" in pair:
            b, m = pair["baseline"], pair["modernized"]
            out.append({
                "design_point_id": m["design_point_id"],
                "lambda_per_h": m["lambda_per_h"],
                "uas_archetype_p_mr": m["uas_archetype_p_mr"],
                "rep": m["rep"],
                "seed": m["seed"],
                "config": "delta",
                "aor": m["aor"],
                "metric": m["metric"],
                "value": m["value"] - b["value"],
            })
    return out


def _cost_rows(res: Any, base: dict[str, Any]) -> list[dict[str, Any]]:
    """Cost responses (rockets/leaks) from the run logs — mirrors ``batch_seeds``."""
    rows: list[dict[str, Any]] = []

    shots: dict[str, list[int]] = defaultdict(list)
    for e in res.log.engagements:
        f = e.fields_
        shots[f["config_name"]].append(int(f.get("n_shots", 0)))
    for cfg_name in ("baseline", "modernized"):
        vals = shots.get(cfg_name, [])
        if vals:
            rows.append({**base, "config": cfg_name, "aor": "ALL",
                         "metric": "rockets_per_engagement",
                         "value": sum(vals) / len(vals)})

    leaks: dict[tuple[str, str], int] = defaultdict(int)
    for lk in res.log.leaks:
        leaks[(lk["config_name"], lk["reason"])] += 1
    for (cfg_name, reason), cnt in sorted(leaks.items()):
        rows.append({**base, "config": cfg_name, "aor": "ALL",
                     "metric": f"leak_{reason}", "value": float(cnt)})

    return rows


def _run_design_point(
    base_cfg: MissionConfig,
    row: dict[str, Any],
    n_reps: int,
    base_seed: int,
    max_arrivals: int | None,
) -> list[dict[str, Any]]:
    """Top-level (picklable) runner for one design point."""
    design_point_id = int(row["design_point_id"])
    lam_h = float(row["lambda_per_h"])
    p_mr = float(row["uas_archetype_p_mr"])
    cfg = _design_point_to_config(base_cfg, row)
    seeds = _crn_seeds(base_seed, design_point_id, n_reps)

    rows: list[dict[str, Any]] = []
    for rep, seed in enumerate(seeds):
        res = run_paired(cfg, seed=seed, max_arrivals=max_arrivals, persist=False)
        base = {
            "design_point_id": design_point_id, "lambda_per_h": lam_h,
            "uas_archetype_p_mr": p_mr, "rep": rep, "seed": int(seed),
        }
        for _, mrow in res.metrics.iterrows():
            rows.append({**base, "config": str(mrow["config_name"]),
                         "aor": str(mrow["aor"]), "metric": str(mrow["metric"]),
                         "value": float(mrow["value"])})
        rows.extend(_cost_rows(res, base))
    return rows


def _finalize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Add Δ rows, enforce dtypes + canonical column order/sort."""
    all_rows = rows + _delta_rows(rows)
    df = pd.DataFrame(all_rows, columns=_COLUMNS)
    if df.empty:
        return df
    df = df.sort_values(_SORT_KEY, kind="mergesort").reset_index(drop=True)
    for col in ("design_point_id", "rep", "seed"):
        df[col] = df[col].astype("int64")
    for col in ("lambda_per_h", "uas_archetype_p_mr", "value"):
        df[col] = df[col].astype("float64")
    for col in ("config", "aor", "metric"):
        df[col] = df[col].astype(str)
    return df[_COLUMNS]


def run_data_farm(
    base_cfg: MissionConfig,
    design: pd.DataFrame,
    *,
    n_reps: int,
    base_seed: int,
    max_arrivals: int | None = None,
    workers: int = 1,
    stop_func: StopFunc | None = None,
    chunk_size: int = 0,
) -> pd.DataFrame:
    """Run the full data farm, returning the tidy/long DataFrame.

    Deterministic row order regardless of worker completion order (canonical sort).
    With ``stop_func`` set, wraps the run in an :class:`ExecutionController` (early-stop).
    """
    if stop_func is not None:
        def _sim(chunk: pd.DataFrame) -> pd.DataFrame:
            acc: list[dict[str, Any]] = []
            for _, row in chunk.iterrows():
                acc.extend(
                    _run_design_point(base_cfg, row.to_dict(), n_reps, base_seed,
                                      max_arrivals)
                )
            return pd.DataFrame(acc)

        controlled = ExecutionController(_sim, stop_func, chunk_size).run(design)
        return _finalize(controlled.to_dict("records"))

    rows: list[dict[str, Any]] = []
    point_rows = [r for _, r in design.iterrows()]
    if workers <= 1:
        for row in point_rows:
            rows.extend(
                _run_design_point(base_cfg, row.to_dict(), n_reps, base_seed,
                                  max_arrivals)
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_design_point, base_cfg, row.to_dict(), n_reps,
                            base_seed, max_arrivals)
                for row in point_rows
            ]
            for fut in as_completed(futures):
                rows.extend(fut.result())
    return _finalize(rows)


def write_datafarm(
    df: pd.DataFrame,
    path: Path,
    *,
    base_seed: int,
    n_reps: int,
    max_arrivals: int | None,
    design_ref: str,
) -> None:
    """Persist ``datafarm.parquet`` + deterministic sidecar JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    meta = {
        "field_version": FIELD_VERSION,
        "base_seed": base_seed,
        "n_reps": n_reps,
        "max_arrivals": max_arrivals,
        "design_ref": design_ref,
    }
    meta_path = path.with_name(path.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
