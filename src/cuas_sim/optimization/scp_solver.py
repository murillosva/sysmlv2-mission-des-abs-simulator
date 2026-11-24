"""Weighted multi-cover Set Covering Problem solver.

Implements the SCP re-solve step of the OPT↔SIM loop.
Uses ``scipy.optimize.milp`` for exact binary MILP.

Literature:
    Haight & ReVelle (2000) — set cover with demand weights.
    Szvaticsek (2025) — original R implementation (TCC 2024).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cuas_sim.geometry.coordinates import GeoPoint


@dataclass(slots=True)
class OptimizerDirective:
    """SCP solution delivered to the loop controller for the next iteration."""

    iteration: int
    activate_farps: list[str]
    cap_adjustments: dict[str, dict[str, float]]
    farp_coverage_matrix: dict[str, list[int]]
    scp_parameters: dict[str, object]
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "iteration": self.iteration,
            "activate_farps": list(self.activate_farps),
            "cap_adjustments": dict(self.cap_adjustments),
            "farp_coverage_matrix": {k: list(v) for k, v in self.farp_coverage_matrix.items()},
            "scp_parameters": dict(self.scp_parameters),
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def compute_coverage_matrix(
    farps: dict[str, GeoPoint],
    caps: dict[str, GeoPoint],
    endurance_s: float,
    cruise_speed_mps: float,
) -> np.ndarray:
    """Build binary coverage matrix A where A[j, k] = 1 if FARP j covers CAP k.

    Coverage is defined as t_v = endurance - 2·d/v > 0, i.e., the FARP is
    close enough to the CAP that the A-29 has positive patrol time remaining.

    Parameters
    ----------
    farps
        Candidate FARP positions (ordered dict — row order preserved).
    caps
        ZIDA anchor positions (ordered dict — column order preserved).
    endurance_s
        A-29 endurance in seconds (case study: 5.2 h = 18 720 s).
    cruise_speed_mps
        A-29 cruise speed in m/s (case study: 144 m/s).

    Returns
    -------
    np.ndarray
        Shape (J, K): J = len(farps), K = len(caps).
    """
    from cuas_sim.geometry.haversine import haversine_m

    farp_list = list(farps.values())
    cap_list = list(caps.values())
    J = len(farp_list)
    K = len(cap_list)
    A = np.zeros((J, K), dtype=np.int8)
    for j, farp in enumerate(farp_list):
        for k, cap in enumerate(cap_list):
            d_m = haversine_m(farp, cap)
            t_v = endurance_s - 2.0 * d_m / cruise_speed_mps
            if t_v > 0:
                A[j, k] = 1
    return A


def compute_demand(
    mos1_by_aor: dict[str, float],
    aor_names: list[str],
    target: float = 0.80,
) -> np.ndarray:
    """Weighted demand vector for the multi-cover SCP.

    Demand rule:
        demand[k] = ceil(1 + max(0, target − mos1_by_aor[aor_k]) / 0.3)

    AoRs above target get demand = 1 (standard cover). AoRs well below target
    require multiple-cover (demand ≥ 2) to force more FARPs into that region.
    """
    return np.array([
        math.ceil(1 + max(0.0, target - mos1_by_aor.get(aor, 0.0)) / 0.3)
        for aor in aor_names
    ], dtype=np.int64)


def solve(
    coverage_matrix: np.ndarray,
    demand: np.ndarray,
    candidate_names: list[str],
) -> tuple[list[str], float]:
    """Solve the weighted multi-cover SCP via MILP.

    Minimise sum(x_j) subject to A.T @ x ≥ demand, x_j ∈ {0,1}.

    Parameters
    ----------
    coverage_matrix
        Shape (J, K) binary matrix: A[j, k] = 1 if FARP j covers anchor k.
    demand
        Shape (K,) integer demand per anchor point (≥ 1).
    candidate_names
        FARP names corresponding to coverage_matrix rows.

    Returns
    -------
    (active_farps, objective_value)
        active_farps: ICAO codes of selected FARPs.
        objective_value: number of FARPs activated.

    Raises
    ------
    ValueError
        If the problem is infeasible (no combination of FARPs covers all anchors).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    J = coverage_matrix.shape[0]
    K = coverage_matrix.shape[1]

    if J == 0 or K == 0:
        raise ValueError("Coverage matrix must be non-empty (infeasible: no candidates).")

    c = np.ones(J, dtype=float)
    A_mat = coverage_matrix.T.astype(float)  # shape (K, J)
    lb = demand.astype(float)
    ub = np.full(K, np.inf)

    constraints = LinearConstraint(A_mat, lb=lb, ub=ub)
    integrality = np.ones(J)
    bounds = Bounds(lb=np.zeros(J), ub=np.ones(J))

    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

    if result.status != 0 or result.x is None:
        raise ValueError(
            f"SCP infeasible or solver failed (status={result.status}): {result.message}"
        )

    active = [candidate_names[j] for j in range(J) if result.x[j] > 0.5]
    return active, float(result.fun)


def solve_initial(
    farp_names: list[str],
    coverage_matrix: np.ndarray,
    iteration: int = 0,
) -> OptimizerDirective:
    """Initial SCP solve with demand=1 (standard cover — no feedback yet)."""
    K = coverage_matrix.shape[1]
    demand = np.ones(K, dtype=np.int64)
    try:
        active_farps, obj = solve(coverage_matrix, demand, farp_names)
    except ValueError:
        active_farps = list(farp_names)  # fallback: activate all
        obj = float(len(farp_names))

    coverage_dict = {
        farp_names[j]: coverage_matrix[j].tolist()
        for j in range(len(farp_names))
    }
    return OptimizerDirective(
        iteration=iteration,
        activate_farps=active_farps,
        cap_adjustments={},
        farp_coverage_matrix=coverage_dict,
        scp_parameters={"objective_value": obj, "demand_mode": "standard"},
    )


def resolve(
    farp_names: list[str],
    coverage_matrix: np.ndarray,
    mos1_by_aor: dict[str, float],
    aor_names: list[str],
    iteration: int,
    target: float = 0.80,
) -> OptimizerDirective:
    """Re-solve SCP with weighted demand based on per-AoR MOS-1 deficit."""
    demand = compute_demand(mos1_by_aor, aor_names, target=target)
    try:
        active_farps, obj = solve(coverage_matrix, demand, farp_names)
    except ValueError:
        active_farps = list(farp_names)  # fallback: activate all
        obj = float(len(farp_names))

    coverage_dict = {
        farp_names[j]: coverage_matrix[j].tolist()
        for j in range(len(farp_names))
    }
    return OptimizerDirective(
        iteration=iteration,
        activate_farps=active_farps,
        cap_adjustments={},
        farp_coverage_matrix=coverage_dict,
        scp_parameters={
            "objective_value": obj,
            "demand_mode": "weighted",
            "demand_by_aor": {aor: int(d) for aor, d in zip(aor_names, demand, strict=True)},
        },
    )
