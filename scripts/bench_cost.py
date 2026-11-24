"""Measure the computational cost of the simulator (wall time, CPU time, memory).

A performance measurement is an experiment too: a single run is one sample, subject to
background load, frequency scaling and temperature. This script runs ``k`` repetitions of
``N`` paired replications and reports the median (with minimum and maximum) of three
phases:

* **startup**: loading the scenario file and parsing the SysML v2 model, a fixed cost per
  invocation;
* **engine**: a plain loop of ``run_paired`` (the DES + ABS engine);
* **farm**: ``run_data_farm``, what ``analysis run`` calls: the engine plus the conversion
  to long-format rows, the paired-difference rows and the canonical sort.

The difference ``farm - engine`` is the per-replication cost of the analysis layer,
measured rather than inferred from two separate runs.

Reported quantities:

* wall time and process CPU time (``time.process_time``, user + system); the second is
  insensitive to background load and I/O waits;
* utilization per core (CPU / wall; about 100% for CPU-bound work) and of the machine
  (CPU / (wall x cores), which exposes idle parallelism);
* throughput: replications per second and logged records per second (SimPy exposes no
  scheduler event counter, so logged records are a labelled proxy, not events per second);
* real-time factor (simulated seconds per wall-clock second);
* peak resident memory.

With ``--datafarm`` it adds a precision-versus-cost table: since the interval shrinks as
1/sqrt(N), halving the half-width costs four times the CPU time.

Usage::

    python scripts/bench_cost.py --reps 20 --repeats 5 \
        --datafarm outputs/analysis/ex1_mc_nominal/datafarm_n200.parquet \
        -o outputs/analysis/ex1_mc_nominal/bench_cost.json

Runs ``run_paired`` and ``run_data_farm`` with ``persist=False`` only. Console output is
ASCII. The JSON is deterministic except for the measured times, which are the object of
the measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from cuas_sim.analysis.experiment import run_data_farm
from cuas_sim.config import MissionConfig
from cuas_sim.optimization.loop_controller import _crn_seeds
from cuas_sim.simulation.run import run_paired

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign_postprocess import paired_delta  # sibling script (scripts/ is not a package)

Z_95 = 1.959963984540054  # two-sided 95% normal quantile
HALF_WIDTH_TARGETS = (0.05, 0.025, 0.0174, 0.01, 0.005)


def _git_head() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout.strip()


def _environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "simpy", "pyarrow"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # the environment block is best effort
            versions[name] = "n/a"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "total_ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "packages": versions,
        "git_head": _git_head(),
    }


def _measure(fn: Any) -> dict[str, Any]:
    """Run ``fn`` measuring wall time, process CPU time and peak resident memory."""
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    w0, c0 = time.perf_counter(), time.process_time()
    payload = fn()
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    rss = max(rss0, proc.memory_info().rss)
    return {"wall_s": wall, "cpu_s": cpu, "peak_rss_bytes": rss, "payload": payload}


def _engine_loop(cfg: MissionConfig, seeds: list[int]) -> dict[str, float]:
    """Plain loop of ``run_paired``, counting the units of work."""
    records = 0
    sim_seconds = 0.0
    arrivals = 0
    for seed in seeds:
        res = run_paired(cfg, seed=seed, persist=False)
        records += (
            sum(len(v) for v in res.log.events.values())
            + len(res.log.engagements) + len(res.log.leaks) + len(res.log.uas_births)
        )
        sim_seconds += float(res.log.window["t_window_end_s"])
        arrivals += int(res.total_arrivals)
    return {"records": float(records), "sim_seconds": sim_seconds,
            "arrivals": float(arrivals)}


def _farm_call(cfg: MissionConfig, design: pd.DataFrame, n_reps: int,
               base_seed: int) -> dict[str, float]:
    df = run_data_farm(cfg, design, n_reps=n_reps, base_seed=base_seed,
                       max_arrivals=None, workers=1)
    return {"tidy_rows": float(len(df))}


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _phase_block(runs: list[dict[str, Any]], n_reps: int, cores: int) -> dict[str, Any]:
    wall = [r["wall_s"] for r in runs]
    cpu = [r["cpu_s"] for r in runs]
    w_med, c_med = statistics.median(wall), statistics.median(cpu)
    return {
        "n_repeats": len(runs),
        "n_reps_per_repeat": n_reps,
        "wall_s": _stats(wall),
        "cpu_s": _stats(cpu),
        "wall_ms_per_replication": w_med / n_reps * 1000.0,
        "cpu_ms_per_replication": c_med / n_reps * 1000.0,
        "utilization_per_core_pct": c_med / w_med * 100.0 if w_med else None,
        "utilization_machine_pct": c_med / (w_med * cores) * 100.0 if w_med else None,
        "replications_per_s": n_reps / w_med if w_med else None,
        "peak_rss_mib": max(r["peak_rss_bytes"] for r in runs) / 2**20,
    }


def precision_cost_table(
    datafarm: pd.DataFrame, *, response: str, aor: str, ms_per_replication: float,
) -> dict[str, Any]:
    """N and cost required for each half-width target (interval ~ 1/sqrt(N))."""
    delta = paired_delta(datafarm, response, aor)
    sigma = float(delta.std(ddof=1))
    rows = []
    for target in HALF_WIDTH_TARGETS:
        n_needed = (Z_95 * sigma / target) ** 2
        rows.append({
            "half_width_target": target,
            "n_required": int(-(-n_needed // 1)),  # ceiling
            "wall_s_estimated": n_needed * ms_per_replication / 1000.0,
        })
    return {
        "response": response,
        "aor": aor,
        "sigma_per_replication": sigma,
        "n_observed": int(delta.size),
        "half_width_observed": Z_95 * sigma / (delta.size ** 0.5),
        "law": "semi-largura ~ z*sigma/raiz(N); cortar pela metade custa 4x CPU",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure the computational cost of the simulator (wall, CPU, memory)."
    )
    ap.add_argument("--config", type=Path, default=Path("configs/zida_shield_default.yaml"))
    ap.add_argument("--sysml", type=Path, default=Path("sysml"))
    ap.add_argument("--reps", type=int, default=20,
                    help="Paired replications per repetition")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Repetitions of the measurement (the median is reported)")
    ap.add_argument("--base-seed", type=int, default=20260804)
    ap.add_argument("--datafarm", type=Path, default=None,
                    help="datafarm.parquet for the precision-versus-cost table (optional)")
    ap.add_argument("--response", default="MOS-1")
    ap.add_argument("--aor", default="ALL")
    ap.add_argument("--output", "-o", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.repeats < 1 or args.reps < 1:
        print("[ERROR] --reps and --repeats must be >= 1", file=sys.stderr)
        return 1

    cores = os.cpu_count() or 1
    startup = _measure(
        lambda: MissionConfig.from_yaml_with_sysml(args.config, sysml_root=args.sysml)
    )
    cfg = startup.pop("payload")
    seeds = _crn_seeds(args.base_seed, 0, args.reps)
    design = pd.DataFrame({"design_point_id": [0], "lambda_per_h": [6.0],
                           "uas_archetype_p_mr": [0.5]})

    _engine_loop(cfg, seeds[:1])  # warm-up, not measured

    engine_runs = [_measure(lambda: _engine_loop(cfg, seeds)) for _ in range(args.repeats)]
    farm_runs = [
        _measure(lambda: _farm_call(cfg, design, args.reps, args.base_seed))
        for _ in range(args.repeats)
    ]

    engine = _phase_block(engine_runs, args.reps, cores)
    farm = _phase_block(farm_runs, args.reps, cores)

    work = engine_runs[0]["payload"]
    engine["records_per_replication"] = work["records"] / args.reps
    engine["records_per_s"] = work["records"] / statistics.median(
        [r["wall_s"] for r in engine_runs]
    )
    engine["records_note"] = (
        "PROXY do trabalho do modelo (registros logados: transicoes de estado + "
        "engajamentos + leaks + nascimentos). O SimPy nao expoe contador de "
        "eventos do escalonador, entao NAO chamar de 'eventos/s'"
    )
    engine["sim_hours_per_replication"] = work["sim_seconds"] / args.reps / 3600.0
    engine["real_time_factor"] = work["sim_seconds"] / statistics.median(
        [r["wall_s"] for r in engine_runs]
    )
    engine["arrivals_per_replication"] = work["arrivals"] / args.reps
    farm["tidy_rows_per_replication"] = farm_runs[0]["payload"]["tidy_rows"] / args.reps

    overhead_ms = farm["wall_ms_per_replication"] - engine["wall_ms_per_replication"]
    layer = {
        "wall_ms_per_replication": overhead_ms,
        "share_of_farm_pct": (
            overhead_ms / farm["wall_ms_per_replication"] * 100.0
            if farm["wall_ms_per_replication"] else None
        ),
        "note": (
            "conversao para linhas tidy + linhas delta + ordenacao canonica; "
            "MEDIDO como farm - engine, nao inferido"
        ),
    }

    report: dict[str, Any] = {
        "environment": _environment(),
        "settings": {
            "config": str(args.config), "sysml_root": str(args.sysml),
            "n_reps_per_repeat": args.reps, "n_repeats": args.repeats,
            "base_seed": args.base_seed, "workers": 1,
        },
        "startup": {"wall_s": startup["wall_s"], "cpu_s": startup["cpu_s"],
                    "note": "carga do YAML + parse do SysML; custo fixo por invocacao"},
        "engine": engine,
        "farm": farm,
        "analysis_layer_overhead": layer,
        "precision_cost": None,
    }

    if args.datafarm is not None:
        if not args.datafarm.exists():
            print(f"[ERROR] datafarm not found: {args.datafarm}", file=sys.stderr)
            return 1
        report["precision_cost"] = precision_cost_table(
            pd.read_parquet(args.datafarm), response=args.response, aor=args.aor,
            ms_per_replication=farm["wall_ms_per_replication"],
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")

    print(f"=== computational cost -> {args.output} ===")
    print(f"environment: {cores} logical cores | "
          f"{report['environment']['total_ram_gib']} GiB | "
          f"Python {report['environment']['python']}")
    print(f"measurement: {args.repeats} repetitions x {args.reps} paired replications (median)")
    print(f"startup : {startup['wall_s']:.2f} s (scenario file + model, fixed per invocation)")
    for name, blk in (("engine", engine), ("farm  ", farm)):
        print(f"{name}  : {blk['wall_ms_per_replication']:.1f} ms/replication wall | "
              f"{blk['cpu_ms_per_replication']:.1f} ms CPU | "
              f"{blk['utilization_per_core_pct']:.1f}% of a core, "
              f"{blk['utilization_machine_pct']:.1f}% of the machine")
    print(f"  wall-time spread (engine): min {engine['wall_s']['min']:.3f} s / "
          f"median {engine['wall_s']['median']:.3f} s / max {engine['wall_s']['max']:.3f} s")
    print(f"layer   : +{layer['wall_ms_per_replication']:.1f} ms/replication "
          f"({layer['share_of_farm_pct']:.0f}% of the farm cost)")
    print(f"throughput: {farm['replications_per_s']:.2f} replications/s | "
          f"{engine['records_per_s']:.0f} records/s (proxy) | "
          f"peak RSS {farm['peak_rss_mib']:.1f} MiB")
    print(f"scale   : {engine['sim_hours_per_replication']:.2f} h of mission/replication | "
          f"real-time factor {engine['real_time_factor']:.0f}x | "
          f"{engine['arrivals_per_replication']:.1f} arrivals/replication")
    pc = report["precision_cost"]
    if pc is not None:
        print(f"precision x cost ({pc['response']}, sigma={pc['sigma_per_replication']:.4f}):")
        for row in pc["rows"]:
            print(f"  half-width {row['half_width_target']:.4f} -> "
                  f"N={row['n_required']:6d}  ~{row['wall_s_estimated']:8.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
