"""Design of Experiments — Latin Hypercube Sampling.

Pure module: the only RNG is the explicit ``design_seed`` passed
to :func:`create_design` (meta-level). No ``RNGRegistry``, no global
RNG state, no implicit I/O. Mirrors ``asapy.doe.Doe.create`` on ``scipy.stats.qmc``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import qmc

FIELD_VERSION = "1.0"

__all__ = ["FIELD_VERSION", "Factor", "create_design", "write_design"]


@dataclass(frozen=True)
class Factor:
    """A single DoE factor.

    ``kind`` is one of ``"numeric"`` / ``"integer"`` (use ``low``/``high``),
    ``"categorical"`` (use ``options``), or ``"boolean"`` (no extra params).
    """

    name: str
    kind: str = "numeric"
    low: float | None = None
    high: float | None = None
    options: tuple[Any, ...] | None = None


def _require_range(f: Factor) -> tuple[float, float]:
    if f.low is None or f.high is None or f.low >= f.high:
        raise ValueError(f"factor {f.name!r} requires low < high")
    return float(f.low), float(f.high)


def _scale_factor(f: Factor, col: Any) -> Any:
    """Scale a unit column ``col in [0,1)`` to the factor's domain."""
    if f.kind == "numeric":
        low, high = _require_range(f)
        return (low + col * (high - low)).astype(float)
    if f.kind == "integer":
        low, high = _require_range(f)
        scaled = low + col * (high - low)
        return [math.floor(v) for v in scaled]
    if f.kind == "categorical":
        if not f.options:
            raise ValueError(f"factor {f.name!r} requires non-empty options")
        k = len(f.options)
        return [f.options[min(int(v * k), k - 1)] for v in col]
    if f.kind == "boolean":
        return [bool(round(v)) for v in col]
    raise ValueError(f"unknown factor kind: {f.kind!r}")


def create_design(
    factors: Sequence[Factor], n_samples: int, design_seed: int,
) -> pd.DataFrame:
    """Generate ``n_samples`` LHS design points over ``factors``.

    Deterministic given ``(factors, n_samples, design_seed)`` — the unit sample is a
    single ``scipy.stats.qmc.LatinHypercube(d=len(factors), seed=design_seed,
    scramble=True).random(n_samples)`` call (column ``i`` <-> factor ``i``).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not factors:
        raise ValueError("at least one factor is required")

    sampler = qmc.LatinHypercube(d=len(factors), seed=design_seed, scramble=True)
    unit = sampler.random(n_samples)

    data: dict[str, Any] = {"design_point_id": list(range(n_samples))}
    for i, f in enumerate(factors):
        data[f.name] = _scale_factor(f, unit[:, i])
    return pd.DataFrame(data)


def write_design(
    df: pd.DataFrame,
    path: Path,
    *,
    design_seed: int,
    sampler: str,
    n_samples: int,
    factor_meta: dict[str, Any],
) -> None:
    """Persist ``design.parquet`` + deterministic sidecar JSON.

    The sidecar (``<path>.meta.json``) is byte-deterministic: ``sort_keys=True`` and
    no timestamp.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    meta = {
        "field_version": FIELD_VERSION,
        "design_seed": design_seed,
        "sampler": sampler,
        "n_samples": n_samples,
        "factors": factor_meta,
    }
    meta_path = path.with_name(path.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
