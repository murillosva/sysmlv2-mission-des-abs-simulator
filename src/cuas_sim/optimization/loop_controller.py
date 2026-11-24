"""OPT↔SIM loop controller — SAA with Rinott correction.

Legacy optimizer–simulator loop, not used in the studies of the paper. It implements
the iterative FARP/CAP placement loop:
the SCP optimizer resolves FARP activation after each simulation batch;
the loop stops when the 95% CI lower bound of MOS-1 exceeds the target
(Gate 1), when improvement stagnates (Gate 2), or when the iteration
budget is exhausted (Gate 3).

LoopState, init_loop_state() and step_loop() provide single-step operation via
the Python API and the CLI.

Common random numbers: all seeds derive from ``numpy.random.SeedSequence``
with a single base seed — same seed ⟹ bit-identical results.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cuas_sim.config import MissionConfig
    from cuas_sim.world.aor import AoRMap

logger = logging.getLogger(__name__)


# ─── Result types ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class IterationRecord:
    """Audit record for one loop iteration."""

    iteration: int
    n_replications: int
    mos1_mean: float
    ci_lower: float
    ci_upper: float
    active_farps: list[str]
    gate_1_accepted: bool
    gate_2_stagnation: bool


@dataclass(slots=True)
class LoopResult:
    """Full result of ``run_opt_sim_loop``."""

    converged: bool
    reason: str  # "accepted" | "stagnation" | "budget" | "infeasible"
    iterations: int
    final_mos1: float
    final_ci_lower: float
    active_farps: list[str]
    last_signal_path: Path | None
    history: list[IterationRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "converged": self.converged,
            "reason": self.reason,
            "iterations": self.iterations,
            "final_mos1": self.final_mos1,
            "final_ci_lower": self.final_ci_lower,
            "active_farps": list(self.active_farps),
            "last_signal_path": str(self.last_signal_path) if self.last_signal_path else None,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ─── Loop state ────────────────────────────────────────────────────


@dataclass(slots=True)
class LoopState:
    """Persistent inter-iteration state for step_loop().

    ``iteration`` is the 1-indexed number of the *next* iteration to execute.
    After ``init_loop_state()`` returns, ``iteration == 1``.
    After ``step_loop()`` completes iteration k, ``iteration == k + 1``.
    """

    config_base: MissionConfig
    config_current: MissionConfig
    run_dir: Path
    loop_dir: Path
    iteration: int
    history: list[IterationRecord]
    prev_mos1: float | None
    stagnation_count: int
    last_signal_path: Path | None
    coverage_matrix: np.ndarray
    farp_names: list[str]
    aor_names: list[str]
    aor_map: AoRMap
    I_max: int
    N0: int
    N_final: int
    target_mos1: float
    alpha: float
    delta_stagnation: float
    base_seed: int
    max_arrivals: int | None
    farp_distances_km: dict[str, float]

    def done(self) -> bool:
        """True when iteration budget is exhausted (Gate 3 pre-check)."""
        return self.iteration > self.I_max

    def to_json_dict(self, config_path: Path) -> dict[str, object]:
        """Serialize for CLI persistence (loop_state.json schema v1.0).

        ``config_path`` must point to the original YAML so deserialization
        can reconstruct ``config_base``.
        """
        return {
            "schema_version": "1.0",
            "config_path": str(config_path),
            "run_dir": str(self.run_dir),
            "iteration": self.iteration,
            "current_active_farps": sorted(self.config_current.farps.keys()),
            "prev_mos1": self.prev_mos1,
            "stagnation_count": self.stagnation_count,
            "last_signal_path": str(self.last_signal_path) if self.last_signal_path else None,
            "history": [
                {
                    "iteration": r.iteration,
                    "n_replications": r.n_replications,
                    "mos1_mean": r.mos1_mean,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                    "active_farps": list(r.active_farps),
                    "gate_1_accepted": r.gate_1_accepted,
                    "gate_2_stagnation": r.gate_2_stagnation,
                }
                for r in self.history
            ],
            "I_max": self.I_max,
            "N0": self.N0,
            "N_final": self.N_final,
            "target_mos1": self.target_mos1,
            "alpha": self.alpha,
            "delta_stagnation": self.delta_stagnation,
            "base_seed": self.base_seed,
            "max_arrivals": self.max_arrivals,
            "farp_distances_km": dict(self.farp_distances_km),
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> LoopState:
        """Deserialize from loop_state.json.

        Reloads ``config_base`` from the stored YAML path and recomputes
        ``coverage_matrix`` deterministically. Requires the original YAML
        to be accessible at the stored ``config_path``.
        """
        from cuas_sim.config import MissionConfig
        from cuas_sim.optimization.directive_builder import apply_directive
        from cuas_sim.optimization.scp_solver import OptimizerDirective, compute_coverage_matrix
        from cuas_sim.world.aor import AoRMap

        config_path = Path(d["config_path"])
        config_base = MissionConfig.from_yaml_with_sysml(str(config_path))

        directive = OptimizerDirective(
            iteration=d["iteration"] - 1,
            activate_farps=list(d["current_active_farps"]),
            cap_adjustments={},
            farp_coverage_matrix={},
            scp_parameters={},
        )
        config_current = apply_directive(config_base, directive)

        coverage_matrix = compute_coverage_matrix(
            farps=config_base.farps,
            caps=config_base.caps,
            endurance_s=config_base.a29_endurance_s,
            cruise_speed_mps=config_base.a29_cruise_speed_mps,
        )
        farp_names = list(config_base.farps.keys())
        aor_names = list(config_base.aor_to_sectors.keys())
        aor_map = AoRMap.from_config(config_base)

        history = [
            IterationRecord(
                iteration=r["iteration"],
                n_replications=r["n_replications"],
                mos1_mean=r["mos1_mean"],
                ci_lower=r["ci_lower"],
                ci_upper=r["ci_upper"],
                active_farps=r["active_farps"],
                gate_1_accepted=r["gate_1_accepted"],
                gate_2_stagnation=r["gate_2_stagnation"],
            )
            for r in d["history"]
        ]

        lsp = d.get("last_signal_path")
        return cls(
            config_base=config_base,
            config_current=config_current,
            run_dir=Path(d["run_dir"]),
            loop_dir=Path(d["run_dir"]) / "loop",
            iteration=d["iteration"],
            history=history,
            prev_mos1=d.get("prev_mos1"),
            stagnation_count=int(d.get("stagnation_count", 0)),
            last_signal_path=Path(lsp) if lsp else None,
            coverage_matrix=coverage_matrix,
            farp_names=farp_names,
            aor_names=aor_names,
            aor_map=aor_map,
            I_max=int(d["I_max"]),
            N0=int(d["N0"]),
            N_final=int(d["N_final"]),
            target_mos1=float(d["target_mos1"]),
            alpha=float(d["alpha"]),
            delta_stagnation=float(d["delta_stagnation"]),
            base_seed=int(d["base_seed"]),
            max_arrivals=d.get("max_arrivals"),
            farp_distances_km=dict(d.get("farp_distances_km", {})),
        )


# ─── Statistical helpers ──────────────────────────────────────────────────────


def _rinott_n(
    pilot: list[float],
    k: int,
    alpha: float,
    n0: int,
    delta: float,
) -> int:
    """Rinott sample-size correction.

    Returns ``N_i = max(n0, ceil(h² · S² / Δ²))`` where h is the Bonferroni-
    corrected t-critical value (conservative approximation of the Rinott
    constant; Bechhofer-Santner-Goldsman tables give h(6,0.05,10) ≈ 2.94,
    this approximation yields ≈ 2.96).
    """
    from scipy.stats import t as t_dist

    n = len(pilot)
    if n < 2:
        return n0

    h = float(t_dist.ppf(1.0 - alpha / (2 * k), df=n0 - 1))
    s2 = float(np.var(pilot, ddof=1))
    n_req = math.ceil((h ** 2) * s2 / (delta ** 2))
    return max(n0, n_req)


def _compute_ci(
    values: list[float],
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% CI for the mean via Student-t (edge case: N<2 → value).

    Returns ``(lower, upper)``.
    """
    n = len(values)
    if n < 2:
        v = values[0] if values else 0.0
        return v, v
    from scipy.stats import t as t_dist

    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / math.sqrt(n)
    t_crit = float(t_dist.ppf(1.0 - alpha / 2, df=n - 1))
    return mean - t_crit * se, mean + t_crit * se


# ─── CRN seed helpers ─────────────────────────────────────────────────────────


def _crn_seeds(base_seed: int, iteration: int, n: int) -> list[int]:
    """Non-overlapping seed bank for iteration ``iteration``.

    Hierarchy: ``base_seed → iter[iteration] → rep[0..n-1]``.
    """
    ss = np.random.SeedSequence(base_seed)
    iter_children = ss.spawn(iteration + 1)
    rep_children = iter_children[iteration].spawn(n)
    return [int(rc.generate_state(1)[0]) for rc in rep_children]


# ─── Replication helpers ──────────────────────────────────────────────────────


def _extract_mos1_overall(metrics: Any) -> float:
    """Extract overall MOS-1 for the modernized config from a metrics DataFrame."""
    if metrics.empty:
        return 0.0
    row = metrics[
        (metrics["config_name"] == "modernized")
        & (metrics["metric"] == "MOS-1")
        & (metrics["aor"] == "ALL")
    ]
    return float(row["value"].iloc[0]) if not row.empty else 0.0


def _extract_mos1_by_aor(metrics: Any) -> dict[str, float]:
    """Extract per-AoR MOS-1 for the modernized config from a metrics DataFrame."""
    if metrics.empty:
        return {}
    sub = metrics[
        (metrics["config_name"] == "modernized")
        & (metrics["metric"] == "MOS-1")
        & (metrics["aor"] != "ALL")
    ]
    return {str(row["aor"]): float(row["value"]) for _, row in sub.iterrows()}


def _run_replications(
    config: MissionConfig,
    seeds: list[int],
    max_arrivals: int | None,
    mos1_target: float,
) -> tuple[list[object], list[dict[str, object]]]:
    """Run ``len(seeds)`` paired replications; return (ReplicationMetrics list, aggregated leaks).

    Returns a tuple of:
    - list[ReplicationMetrics]: per-replication MOS-1 stats
    - list[dict[str, object]]: all modernized leaks aggregated across replications
    """
    from cuas_sim.metrics.replanning import ReplicationMetrics
    from cuas_sim.simulation.mission import MissionSimulator

    rep_results = []
    all_leaks: list[dict[str, object]] = []
    for seed in seeds:
        sim = MissionSimulator(config)
        pr = sim.run_paired(seed=seed, max_arrivals=max_arrivals, mos1_target=mos1_target)
        rep_results.append(ReplicationMetrics(
            seed=seed,
            mos1_overall=_extract_mos1_overall(pr.metrics),
            mos1_by_aor=_extract_mos1_by_aor(pr.metrics),
        ))
        all_leaks.extend(
            lk for lk in pr.log.leaks if lk.get("config_name", "modernized") == "modernized"
        )
    return rep_results, all_leaks  # type: ignore[return-value]


# ─── Step API ──────────────────────────────────────────────────────


def init_loop_state(
    config_base: MissionConfig,
    run_dir: Path,
    I_max: int = 8,
    N0: int = 10,
    N_final: int = 30,
    target_mos1: float = 0.80,
    alpha: float = 0.05,
    delta_stagnation: float = 0.01,
    base_seed: int = 42,
    farp_distances_km: dict[str, float] | None = None,
    max_arrivals: int | None = None,
) -> LoopState:
    """Initialise the loop state and execute iteration 0.

    Iteration 0 is the initial SCP solve using only SysML parameters — no
    simulation feedback. Returns a ``LoopState`` with ``iteration=1`` ready
    for the first call to ``step_loop()``.
    """
    from cuas_sim.optimization.directive_builder import apply_directive
    from cuas_sim.optimization.scp_solver import compute_coverage_matrix, solve_initial
    from cuas_sim.world.aor import AoRMap

    run_dir.mkdir(parents=True, exist_ok=True)
    loop_dir = run_dir / "loop"
    loop_dir.mkdir(exist_ok=True)

    if farp_distances_km is None:
        farp_distances_km = {}

    farp_names = list(config_base.farps.keys())
    coverage_matrix = compute_coverage_matrix(
        farps=config_base.farps,
        caps=config_base.caps,
        endurance_s=config_base.a29_endurance_s,
        cruise_speed_mps=config_base.a29_cruise_speed_mps,
    )
    aor_names = list(config_base.aor_to_sectors.keys())
    aor_map = AoRMap.from_config(config_base)

    directive = solve_initial(farp_names, coverage_matrix, iteration=0)
    iter0_dir = loop_dir / "iter_000"
    iter0_dir.mkdir(exist_ok=True)
    directive.write(iter0_dir / "optimizer_directive.json")
    config_current = apply_directive(config_base, directive)

    return LoopState(
        config_base=config_base,
        config_current=config_current,
        run_dir=run_dir,
        loop_dir=loop_dir,
        iteration=1,
        history=[],
        prev_mos1=None,
        stagnation_count=0,
        last_signal_path=None,
        coverage_matrix=coverage_matrix,
        farp_names=farp_names,
        aor_names=aor_names,
        aor_map=aor_map,
        I_max=I_max,
        N0=N0,
        N_final=N_final,
        target_mos1=target_mos1,
        alpha=alpha,
        delta_stagnation=delta_stagnation,
        base_seed=base_seed,
        max_arrivals=max_arrivals,
        farp_distances_km=farp_distances_km,
    )


def step_loop(
    state: LoopState,
) -> tuple[LoopState, IterationRecord, bool, str]:
    """Execute exactly one loop iteration.

    Runs iteration ``state.iteration``, checks convergence gates, and returns:
    - The updated ``LoopState`` with ``iteration`` incremented by 1.
    - The ``IterationRecord`` for this iteration.
    - ``done``: True when a termination gate fires.
    - ``reason``: ``"accepted"`` | ``"stagnation"`` | ``"budget"`` |
      ``"infeasible"`` when done; ``"running"`` otherwise.

    Raises ``RuntimeError`` if called when ``state.done()`` is already True.
    """
    if state.done():
        raise RuntimeError(
            f"Loop already exhausted after {state.I_max} iterations "
            f"(iteration={state.iteration} > I_max={state.I_max})."
        )

    from cuas_sim.metrics.replanning import build_replanning_signal_v2
    from cuas_sim.optimization.directive_builder import apply_directive
    from cuas_sim.optimization.scp_solver import resolve

    i = state.iteration
    iter_dir = state.loop_dir / f"iter_{i:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: N0 pilot replications ───────────────────────────────────────
    pilot_seeds = _crn_seeds(state.base_seed, iteration=i, n=state.N0)
    pilot_results, pilot_leaks = _run_replications(
        state.config_current, pilot_seeds[:state.N0], state.max_arrivals, state.target_mos1,
    )
    pilot_mos1 = [r.mos1_overall for r in pilot_results]  # type: ignore[attr-defined]

    # ── Rinott correction → N_i ───────────────────────────────────────────────
    n_i = min(state.N_final, _rinott_n(
        pilot_mos1, k=max(1, len(state.aor_names)),
        alpha=state.alpha, n0=state.N0, delta=0.05,
    ))
    if n_i > state.N0:
        logger.info("Iter %d: Rinott → N_i=%d (pilot N0=%d)", i, n_i, state.N0)

    # ── Phase 2: additional replications if needed ────────────────────────────
    all_seeds = _crn_seeds(state.base_seed, iteration=i, n=n_i)
    if n_i > state.N0:
        extra_seeds = all_seeds[state.N0:n_i]
        extra_results, extra_leaks = _run_replications(
            state.config_current, extra_seeds, state.max_arrivals, state.target_mos1,
        )
        all_results = [*pilot_results, *extra_results]
        all_leaks = pilot_leaks + extra_leaks
    else:
        all_results = pilot_results
        all_leaks = pilot_leaks

    mos1_values = [r.mos1_overall for r in all_results]  # type: ignore[attr-defined]
    mos1_mean = float(np.mean(mos1_values))
    ci_lower, ci_upper = _compute_ci(mos1_values, alpha=state.alpha)

    # ── Build and write v2 signal ─────────────────────────────────────────────
    signal = build_replanning_signal_v2(
        mos1_modernized=mos1_mean,
        mos1_target=state.target_mos1,
        leaks=all_leaks,
        aor_map=state.aor_map,
        replication_results=all_results,  # type: ignore[arg-type]
        iteration=i,
        config=state.config_current,
        seed_set=[int(s) for s in all_seeds[:n_i]],
        farp_distances_km=state.farp_distances_km,
    )
    signal_path = iter_dir / "replanning_signal.json"
    signal.write(signal_path)

    active_farps = sorted(state.config_current.farps.keys())
    logger.info(
        "Iter %d | N=%d | MOS-1=%.4f | CI=[%.4f,%.4f] | FARPs=%s",
        i, n_i, mos1_mean, ci_lower, ci_upper, active_farps,
    )

    # ── Gate 1 — Acceptance ───────────────────────────────────────────────────
    gate1 = ci_lower >= state.target_mos1

    # ── Gate 2 — Stagnation ───────────────────────────────────────────────────
    gate2 = False
    new_stagnation = state.stagnation_count
    if state.prev_mos1 is not None and abs(mos1_mean - state.prev_mos1) < state.delta_stagnation:
        new_stagnation += 1
        gate2 = new_stagnation >= 2
    else:
        new_stagnation = 0

    record = IterationRecord(
        iteration=i,
        n_replications=n_i,
        mos1_mean=mos1_mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        active_farps=active_farps,
        gate_1_accepted=gate1,
        gate_2_stagnation=gate2,
    )
    new_history = [*state.history, record]

    if gate1:
        done, reason = True, "accepted"
        new_config = state.config_current
        new_prev_mos1 = state.prev_mos1
    elif gate2:
        done, reason = True, "stagnation"
        new_config = state.config_current
        new_prev_mos1 = state.prev_mos1
    else:
        new_prev_mos1 = mos1_mean
        new_config = state.config_current
        done = i >= state.I_max
        reason = "budget" if done else "running"

        if not done:
            # ── SCP re-solve for next iteration ──────────────────────────────
            mos1_by_aor = {
                k: float(np.mean([r.mos1_by_aor.get(k, 1.0)  # type: ignore[attr-defined]
                                   for r in all_results]))
                for k in state.aor_names
            }
            try:
                directive = resolve(
                    state.farp_names, state.coverage_matrix, mos1_by_aor, state.aor_names,
                    iteration=i, target=state.target_mos1,
                )
                directive.write(iter_dir / "optimizer_directive.json")
                new_config = apply_directive(state.config_base, directive)
            except ValueError:
                logger.warning("Iter %d: SCP infeasible — aborting loop.", i)
                done, reason = True, "infeasible"

    new_state = LoopState(
        config_base=state.config_base,
        config_current=new_config,
        run_dir=state.run_dir,
        loop_dir=state.loop_dir,
        iteration=i + 1,
        history=new_history,
        prev_mos1=new_prev_mos1,
        stagnation_count=new_stagnation,
        last_signal_path=signal_path,
        coverage_matrix=state.coverage_matrix,
        farp_names=state.farp_names,
        aor_names=state.aor_names,
        aor_map=state.aor_map,
        I_max=state.I_max,
        N0=state.N0,
        N_final=state.N_final,
        target_mos1=state.target_mos1,
        alpha=state.alpha,
        delta_stagnation=state.delta_stagnation,
        base_seed=state.base_seed,
        max_arrivals=state.max_arrivals,
        farp_distances_km=state.farp_distances_km,
    )

    return new_state, record, done, reason


# ─── Main loop ────────────────────────────────────────────────────────────────


def run_opt_sim_loop(
    config_base: MissionConfig,
    run_dir: Path,
    I_max: int = 8,
    N0: int = 10,
    N_final: int = 30,
    target_mos1: float = 0.80,
    alpha: float = 0.05,
    delta_stagnation: float = 0.01,
    base_seed: int = 42,
    farp_distances_km: dict[str, float] | None = None,
    max_arrivals: int | None = None,
) -> LoopResult:
    """Execute the OPT↔SIM loop with SAA and Rinott correction.

    Thin wrapper over ``init_loop_state()`` + ``step_loop()``.
    Public signature is unchanged — zero breaking change for existing callers.

    Gates:
        1. Acceptance — stop when ``ci_95_mos1_lower ≥ target_mos1``.
        2. Stagnation — stop when ``|mos1_i − mos1_{i-1}| < delta_stagnation``
           for 2 consecutive iterations.
        3. Budget     — stop after ``I_max`` iterations.

    CRN: each iteration uses an independent seed bank derived from
    ``numpy.random.SeedSequence(base_seed)``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "loop.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    try:
        state = init_loop_state(
            config_base=config_base,
            run_dir=run_dir,
            I_max=I_max,
            N0=N0,
            N_final=N_final,
            target_mos1=target_mos1,
            alpha=alpha,
            delta_stagnation=delta_stagnation,
            base_seed=base_seed,
            farp_distances_km=farp_distances_km,
            max_arrivals=max_arrivals,
        )

        reason = "budget"
        while not state.done():
            state, _record, done, reason = step_loop(state)
            if done:
                if reason == "infeasible":
                    final_rec = state.history[-1]
                    result = LoopResult(
                        converged=False,
                        reason="infeasible",
                        iterations=final_rec.iteration,
                        final_mos1=final_rec.mos1_mean,
                        final_ci_lower=final_rec.ci_lower,
                        active_farps=final_rec.active_farps,
                        last_signal_path=state.last_signal_path,
                        history=state.history,
                    )
                    result.write(run_dir / "loop_result.json")
                    return result
                break

        final_record = state.history[-1] if state.history else None
        result = LoopResult(
            converged=(reason == "accepted"),
            reason=reason,
            iterations=len(state.history),
            final_mos1=final_record.mos1_mean if final_record else 0.0,
            final_ci_lower=final_record.ci_lower if final_record else 0.0,
            active_farps=final_record.active_farps if final_record else [],
            last_signal_path=state.last_signal_path,
            history=state.history,
        )
        result.write(run_dir / "loop_result.json")
        return result
    finally:
        logger.removeHandler(fh)
