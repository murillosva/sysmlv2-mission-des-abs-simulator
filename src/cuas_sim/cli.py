"""Typer CLI — ``cuas-sim`` command."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from cuas_sim import __version__
from cuas_sim.config import MissionConfig
from cuas_sim.simulation.run import run_paired
from cuas_sim.simulation.sweep import LambdaSweep

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger(__name__)


@app.callback()
def _setup_logging(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Common options."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


@app.command()
def info() -> None:
    """Print package version and dependency banner."""
    typer.echo(f"cuas-sim v{__version__}")
    typer.echo("Paired DES + ABS Monte Carlo for the Continental ZIDA Shield case study.")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="YAML mission config"),
    seed: int = typer.Option(0, "--seed", "-s", help="Master CRN seed"),
    sysml_root: Path | None = typer.Option(
        Path("sysml"), "--sysml", help="SysML v2 root (None disables override)"),
    outputs: Path = typer.Option(Path("outputs"), "--outputs", "-o"),
    skip_pre_pea: bool = typer.Option(False, "--skip-pre-pea",
                                       help="Bypass Phase-1/2/3 symbolic timeouts"),
    horizon_s: float | None = typer.Option(None, "--horizon-s"),
    mos1_target: float = typer.Option(0.80, "--mos1-target"),
) -> None:
    """Execute one paired baseline + modernized run."""
    if sysml_root is not None and sysml_root.exists():
        cfg = MissionConfig.from_yaml_with_sysml(config, sysml_root=sysml_root)
    else:
        cfg = MissionConfig.from_yaml(config)
    if skip_pre_pea:
        cfg = cfg.replaced(skip_pre_pea=True)
    typer.echo(f"Running paired run (seed={seed}, lambda={cfg.lambda_per_sec * 3600:.2f}/h)")
    res = run_paired(cfg, seed=seed, outputs_root=outputs, horizon_s=horizon_s,
                     mos1_target=mos1_target, persist=True)
    typer.echo(f"  total arrivals: {res.total_arrivals}")
    mos1 = res.metrics[
        (res.metrics["metric"] == "MOS-1") & (res.metrics["aor"] == "ALL")
    ]
    for _, row in mos1.iterrows():
        typer.echo(f"  MOS-1 ({row['config_name']}): {row['value']:.3f}")
    typer.echo(f"  Replanning signal triggered: {res.replanning_signal.triggered}")
    # Paired-run window summary (avoid Unicode arrows for
    # cp1252 Windows console compatibility).
    window = res.log.window
    if window:
        t_start = float(window.get("t_window_start_s", 0.0))
        t_end = float(window.get("t_window_end_s", 0.0))
        reason = window.get("t_window_end_reason", "unknown")
        duration = max(0.0, t_end - t_start)
        typer.echo(
            f"  Window: [{t_start:.1f} -> {t_end:.1f}] s "
            f"({duration:.1f} s, reason: {reason})"
        )
        for cfg_name in ("baseline", "modernized"):
            payload = window.get(f"{cfg_name}_first_bingo")
            if payload:
                typer.echo(
                    f"    {cfg_name:<10} first bingo: "
                    f"{payload['a29_id']} @ {float(payload['t_s']):.1f} s"
                )
            else:
                typer.echo(
                    f"    {cfg_name:<10} first bingo: (none -- {reason})"
                )


@app.command()
def sweep(
    config: Path = typer.Option(..., "--config", "-c", help="YAML sweep config"),
    workers: int = typer.Option(1, "--workers", "-w"),
    outputs: Path = typer.Option(Path("outputs"), "--outputs", "-o"),
) -> None:
    """Execute a lambda sweep on a (lambda, seed) grid (legacy; the studies use ``analysis``)."""
    import yaml

    data = yaml.safe_load(config.read_text()) or {}
    base = data.get("base", {})
    lambdas_per_hour = data["lambdas_per_hour"]
    seeds = data["seeds"]
    sysml_root = data.get("sysml_root")
    if sysml_root and Path(sysml_root).exists():
        # Materialize base config via temporary file
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(base, tf)
            tmp_path = tf.name
        base_cfg = MissionConfig.from_yaml_with_sysml(tmp_path, sysml_root=Path(sysml_root))
    else:
        base_cfg = MissionConfig._from_dict(base)
    s = LambdaSweep(base_cfg=base_cfg, lambdas_per_hour=lambdas_per_hour,
                    seeds=seeds, workers=workers, outputs_root=outputs)
    df = s.run()
    typer.echo(f"Sweep complete: {len(df)} rows → {outputs / 'sweeps' / 'sweep.parquet'}")


@app.command()
def validate(
    sysml_root: Path = typer.Option(Path("sysml"), "--sysml"),
) -> None:
    """Smoke-test the SysML loader without running the simulator."""
    from cuas_sim.sysml_loader import SysMLLoader

    ov = SysMLLoader(sysml_root).load()
    typer.echo("SysML loader provenance:")
    for src, what in ov.provenance:
        typer.echo(f"  • {what} (from {src})")
    if not ov.provenance:
        typer.echo("  (no overrides extracted)")


@app.command(name="check-requirements")
def check_requirements(
    config: Path = typer.Option(
        Path("configs/zida_shield_default.yaml"), "--config", "-c",
        help="Mission YAML config (resolved with SysML overrides)",
    ),
    sysml_root: Path = typer.Option(Path("sysml"), "--sysml", help="SysML v2 root"),
    matrix: bool = typer.Option(True, "--matrix/--no-matrix",
                                help="Print the requirement→config→spec traceability matrix"),
) -> None:
    """Validate a MissionConfig against SysML ``require constraint`` blocks.

    Does NOT run the simulation. Exit 0 if no violation, 1 otherwise.
    """
    from cuas_sim import sysml_requirements as reqmod

    if not config.exists():
        typer.echo(f"[ERROR] Config not found: {config}", err=True)
        raise typer.Exit(1)
    if sysml_root.exists():
        cfg = MissionConfig.from_yaml_with_sysml(config, sysml_root=sysml_root)
    else:
        cfg = MissionConfig.from_yaml(config)

    reqs = reqmod.load_requirements(sysml_root)
    report = reqmod.check_requirements(cfg, reqs)

    typer.echo(f"Requirements parsed: {len(reqs)} | bound checked: "
               f"{len(report.satisfied) + len(report.violated)}")
    for rid in report.satisfied:
        typer.echo(f"  [OK]    {rid}: {report.details.get(rid, '')}")
    for rid in report.violated:
        typer.echo(f"  [FAIL]  {rid}: {report.details.get(rid, '')}")

    if matrix:
        typer.echo("\nTraceability matrix (requirement -> config field -> module):")
        for row in reqmod.build_traceability_matrix(reqs):
            mark = "bound" if row.bound else "info "
            # ASCII-only output for cp1252 Windows consoles (the model uses an em-dash
            # sentinel for unbound rows; render it as '-' here).
            fields = row.config_fields if row.bound else "-"
            spec_ref = row.spec_ref if row.bound else "-"
            typer.echo(f"  [{mark}] {row.req_id:<8} {fields}  ({spec_ref})")

    if report.violated:
        typer.echo(f"\n{len(report.violated)} requirement violation(s).", err=True)
        raise typer.Exit(1)
    typer.echo(f"\nAll {len(report.satisfied)} bound requirements satisfied.")


@app.command(name="check-conformance")
def check_conformance_cmd(
    run: Path = typer.Option(..., "--run", help="Run output dir (outputs/runs/<ts>/)"),
    sysml_root: Path = typer.Option(Path("sysml"), "--sysml", help="SysML v2 root"),
) -> None:
    """Audit a finished run's A-29 state stream against the SysML phase oracle.

    Reads ``events_<config>.jsonl.gz`` only; does NOT run the simulation.
    Exit 0 if conformant, 1 if any violation.
    """
    from cuas_sim import sysml_state_machine as sm

    if not run.exists():
        typer.echo(f"[ERROR] Run directory not found: {run}", err=True)
        raise typer.Exit(1)

    phases = sm.load_phases(sysml_root)
    if not phases.oracle_loaded:
        typer.echo("[WARN] SysML phase oracle not loaded -- conformance is vacuous.")
    events = sm.load_events(run)

    total_violations = 0
    for config in ("baseline", "modernized"):
        by_a29 = events.get(config, {})
        report = sm.check_conformance(by_a29, phases)
        exts = ", ".join(f"{k}={v}" for k, v in sorted(report.extensions_used.items()))
        status = "CONFORMANT" if report.conformant else f"{len(report.violations)} VIOLATION(S)"
        typer.echo(
            f"[{config:<10}] {len(by_a29)} A-29 stream(s) | {status} "
            f"| extensions: {exts or 'none'}"
        )
        for v in report.violations:
            typer.echo(f"    [FAIL] {v.a29_id} @ {v.t_s:.1f}s {v.kind}: {v.detail}")
        total_violations += len(report.violations)

    if total_violations:
        typer.echo(f"\n{total_violations} conformance violation(s).", err=True)
        raise typer.Exit(1)
    typer.echo("\nAll A-29 state streams conform to the SysML phase ordering.")


@app.command(name="check-calculations")
def check_calculations_cmd(
    sysml_root: Path = typer.Option(Path("sysml"), "--sysml", help="SysML v2 root"),
    matrix: bool = typer.Option(True, "--matrix/--no-matrix",
                                help="Print the calc def -> metric -> verdict matrix"),
) -> None:
    """Audit the SysML ``calc def`` formulas against ``metrics/aggregator.py``.

    Reads ``Analysis/CalculationsPackage.sysml`` only; does NOT run the simulation.
    Exit 0 if no CONSISTENT guard failed, 1 otherwise. ESCALATE verdicts are listed as backlog
    (they never cause exit 1). ASCII output (cp1252-safe).
    """
    from cuas_sim import sysml_calculations as calcmod

    calcs = calcmod.load_calculations(sysml_root)
    report = calcmod.check_calculations(calcs)

    typer.echo(
        f"Calc defs parsed: {len(calcs)} | consistent: {len(report.consistent)} | "
        f"accepted: {len(report.accepted)} | mapping-fix: {len(report.mapping_fixed)} | "
        f"escalate: {len(report.escalated)} | failed: {len(report.failed_guards)}"
    )
    for name in report.consistent:
        typer.echo(f"  [OK]      {name}: {report.details.get(name, '')}")
    for name in report.failed_guards:
        typer.echo(f"  [FAIL]    {name}: {report.details.get(name, '')}")
    for name in report.accepted:
        typer.echo(f"  [ACCEPT]  {name}: {report.details.get(name, '')}")
    for name in report.mapping_fixed:
        typer.echo(f"  [MAP-FIX] {name}: {report.details.get(name, '')}")
    for name in report.escalated:
        typer.echo(f"  [ESCAL]   {name}: {report.details.get(name, '')}")
    for name in report.unbound:
        typer.echo(f"  [INFO]    {name}: {report.details.get(name, '')}")

    if matrix:
        typer.echo("\nAdjudication matrix (calc def -> metric -> function -> verdict):")
        for row in calcmod.build_adjudication_matrix():
            typer.echo(
                f"  {row.metric_id:<6} {row.calc_name:<34} {row.verdict:<20} {row.aggregator_ref}"
            )

    if report.failed_guards:
        typer.echo(f"\n{len(report.failed_guards)} CONSISTENT guard failure(s).", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"\nAll {len(report.consistent)} CONSISTENT guard(s) pass; "
        f"{len(report.escalated)} escalation(s) listed as backlog."
    )


@app.command()
def animate(
    run_dir: Path = typer.Argument(..., help="Run output directory (outputs/runs/<timestamp>/)"),
    config: Path = typer.Option(
        Path("configs/zida_shield_default.yaml"), "--config", "-c",
        help="Mission YAML config (same one used for the original run)",
    ),
    fps: int = typer.Option(20, "--fps", help="Frames per second"),
    speed: float = typer.Option(
        60.0, "--speed",
        help="Simulated seconds per real second (60 = 1 sim-min per real second)",
    ),
) -> None:
    """Replay a post-run animation in an interactive matplotlib window."""
    from cuas_sim.viz.animate import animate_run

    if not run_dir.exists():
        typer.echo(f"[ERROR] Run directory not found: {run_dir}", err=True)
        raise typer.Exit(1)
    if not config.exists():
        typer.echo(f"[ERROR] Config not found: {config}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Animating {run_dir.name}  (speed={speed}x, fps={fps})")
    animate_run(run_dir, config, fps=fps, speed_factor=speed)


@app.command()
def check_border(
    config: Path = typer.Option(
        Path("configs/zida_shield_default.yaml"), "--config", "-c",
        help="Mission YAML config (must have border_sectors defined)",
    ),
) -> None:
    """Plot a static geographic check of the border polygon (no run needed)."""
    from cuas_sim.viz.animate import plot_border_check

    if not config.exists():
        typer.echo(f"[ERROR] Config not found: {config}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Plotting border check for {config} ...")
    plot_border_check(config)


loop_app = typer.Typer(
    no_args_is_help=True,
    help="Legacy optimizer-simulator loop (not used in the studies).",
)
app.add_typer(loop_app, name="loop")


@loop_app.command(name="run")
def loop_run(
    config: Path = typer.Option(..., "--config", "-c", help="YAML mission config"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for loop artifacts"),
    sysml_root: Path | None = typer.Option(
        Path("sysml"), "--sysml", help="SysML v2 root (None disables override)",
    ),
    seed: int = typer.Option(42, "--seed", help="Base CRN seed"),
    i_max: int = typer.Option(8, "--i-max", help="Maximum iterations"),
    n0: int = typer.Option(10, "--n0", help="Pilot replications (Rinott N0)"),
    n_final: int = typer.Option(30, "--n-final", help="Max replications cap"),
    target_mos1: float = typer.Option(0.80, "--target-mos1", help="Acceptance threshold MOS-1"),
    max_arrivals: int | None = typer.Option(None, "--max-arrivals", hidden=True),
) -> None:
    """Execute the legacy OPT<->SIM loop (SAA + Rinott correction)."""
    from cuas_sim.optimization.loop_controller import run_opt_sim_loop

    if not config.exists():
        typer.echo(f"[ERROR] Config not found: {config}", err=True)
        raise typer.Exit(1)

    if sysml_root is not None and sysml_root.exists():
        cfg = MissionConfig.from_yaml_with_sysml(config, sysml_root=sysml_root)
    else:
        cfg = MissionConfig.from_yaml(config)

    output.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"Loop OPT<->SIM iniciado  "
        f"(seed={seed}, I_max={i_max}, N0={n0}, N_final={n_final}, target_mos1={target_mos1:.2f})"
    )

    result = run_opt_sim_loop(
        config_base=cfg,
        run_dir=output,
        I_max=i_max,
        N0=n0,
        N_final=n_final,
        target_mos1=target_mos1,
        base_seed=seed,
        max_arrivals=max_arrivals,
    )

    ci_upper = result.history[-1].ci_upper if result.history else float("nan")
    farps_str = ", ".join(result.active_farps) if result.active_farps else "(nenhuma)"
    typer.echo("Loop OPT<->SIM concluido")
    typer.echo(f"  Razao      : {result.reason}")
    typer.echo(f"  Convergiu  : {result.converged}")
    typer.echo(
        f"  MOS-1      : {result.final_mos1:.3f}"
        f"  CI [{result.final_ci_lower:.3f}, {ci_upper:.3f}]"
    )
    typer.echo(f"  Iteracoes  : {result.iterations} / {i_max}")
    typer.echo(f"  FARPs      : {farps_str}  ({len(result.active_farps)} ativas)")
    if result.last_signal_path:
        typer.echo(f"  Signal     : {result.last_signal_path}")


@loop_app.command(name="step")
def loop_step(
    config: Path | None = typer.Option(
        None, "--config", "-c",
        help="YAML mission config (required for first step; omit when resuming via --state)",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Output directory for loop artifacts (required for first step)",
    ),
    state_file: Path | None = typer.Option(
        None, "--state", help="loop_state.json from a previous step (resume mode)",
    ),
    sysml_root: Path | None = typer.Option(
        Path("sysml"), "--sysml", help="SysML v2 root (first step only)",
    ),
    seed: int = typer.Option(42, "--seed", help="Base CRN seed (first step only)"),
    i_max: int = typer.Option(8, "--i-max", help="Maximum iterations (first step only)"),
    n0: int = typer.Option(10, "--n0", help="Pilot replications (first step only)"),
    n_final: int = typer.Option(30, "--n-final", help="Max replications cap (first step only)"),
    target_mos1: float = typer.Option(0.80, "--target-mos1", help="Acceptance threshold MOS-1"),
    max_arrivals: int | None = typer.Option(None, "--max-arrivals", hidden=True),
) -> None:
    """Execute exactly one iteration of the legacy OPT<->SIM loop (step API).

    First step:  cuas-sim loop step --config <yaml> --output <dir> --seed 42
    Resume:      cuas-sim loop step --state <dir>/loop_state.json
    """
    import json as _json

    from cuas_sim.optimization.loop_controller import LoopState, init_loop_state, step_loop

    if state_file is not None:
        # ── Resume mode ────────────────────────────────────────────────────────
        if not state_file.exists():
            typer.echo(f"[ERROR] State file not found: {state_file}", err=True)
            raise typer.Exit(1)
        raw = _json.loads(state_file.read_text(encoding="utf-8"))
        config_path = Path(raw["config_path"])
        loop_state_obj = LoopState.from_json_dict(raw)
        save_path = state_file
    else:
        # ── First step ─────────────────────────────────────────────────────────
        if config is None or output is None:
            typer.echo(
                "[ERROR] For the first step provide --config and --output "
                "(or use --state to resume).",
                err=True,
            )
            raise typer.Exit(1)
        if not config.exists():
            typer.echo(f"[ERROR] Config not found: {config}", err=True)
            raise typer.Exit(1)

        if sysml_root is not None and sysml_root.exists():
            cfg = MissionConfig.from_yaml_with_sysml(config, sysml_root=sysml_root)
        else:
            cfg = MissionConfig.from_yaml(config)

        output.mkdir(parents=True, exist_ok=True)
        loop_state_obj = init_loop_state(
            config_base=cfg,
            run_dir=output,
            I_max=i_max,
            N0=n0,
            N_final=n_final,
            target_mos1=target_mos1,
            base_seed=seed,
            max_arrivals=max_arrivals,
        )
        config_path = config
        save_path = output / "loop_state.json"

    if loop_state_obj.done():
        typer.echo("Loop already complete — no more iterations to run.")
        raise typer.Exit(0)

    loop_state_obj, record, done, reason = step_loop(loop_state_obj)

    remaining = max(0, loop_state_obj.I_max - loop_state_obj.iteration + 1)
    typer.echo(
        f"Iteration {record.iteration} | N={record.n_replications} | "
        f"MOS-1={record.mos1_mean:.4f} | "
        f"CI=[{record.ci_lower:.4f},{record.ci_upper:.4f}] | "
        f"FARPs={record.active_farps}"
    )
    if done:
        typer.echo(f"Loop complete: {reason}")
    else:
        typer.echo(f"Status: running ({remaining} iterations remaining)")

    save_path.write_text(
        _json.dumps(loop_state_obj.to_json_dict(config_path), indent=2),
        encoding="utf-8",
    )
    typer.echo(f"State saved to {save_path}")


analysis_app = typer.Typer(
    no_args_is_help=True,
    help="Data farming and statistical analysis: design, run, stats, predict, report."
)
app.add_typer(analysis_app, name="analysis")


@analysis_app.command(name="design")
def analysis_design(
    factors: str = typer.Option(
        "lambda,p_mr", "--factors", help="Comma list of factor presets (lambda,p_mr)",
    ),
    samples: int = typer.Option(20, "--samples", help="Number of LHS design points (N)"),
    seed: int = typer.Option(42, "--seed", help="Design seed (meta-level LHS RNG)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output design.parquet path"),
) -> None:
    """Generate an LHS design of experiments (does NOT run the simulation)."""
    from cuas_sim.analysis.doe import create_design, write_design
    from cuas_sim.analysis.experiment import factors_from_names

    try:
        flist = factors_from_names(factors.split(","))
    except ValueError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1) from None

    df = create_design(flist, n_samples=samples, design_seed=seed)
    factor_meta = {
        f.name: {
            "kind": f.kind, "low": f.low, "high": f.high,
            "options": list(f.options) if f.options else None,
        }
        for f in flist
    }
    write_design(
        df, output, design_seed=seed, sampler="LatinHypercube",
        n_samples=samples, factor_meta=factor_meta,
    )
    names = ", ".join(f.name for f in flist)
    typer.echo(
        f"Design written: {output}  (points={samples}, factors=[{names}], seed={seed})"
    )


@analysis_app.command(name="run")
def analysis_run(
    design: Path = typer.Option(..., "--design", help="design.parquet from 'analysis design'"),
    reps: int = typer.Option(10, "--reps", help="Paired CRN replications per point (R)"),
    seed: int = typer.Option(42, "--seed", help="Base CRN seed"),
    max_arrivals: int | None = typer.Option(
        None, "--max-arrivals",
        help="Spawner cap; omit to rely on the spawn window "
             "(needs uas_spawn_window_enabled=True or the sim may not terminate)",
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output datafarm.parquet path"),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Base YAML mission config (default: built-in defaults)",
    ),
    sysml_root: Path | None = typer.Option(Path("sysml"), "--sysml", help="SysML v2 root"),
    workers: int = typer.Option(1, "--workers", help="Parallel processes"),
) -> None:
    """Run the paired data farm over a design (RUNS the simulation)."""
    import pandas as pd

    from cuas_sim.analysis.experiment import run_data_farm, write_datafarm

    if not design.exists():
        typer.echo(f"[ERROR] Design not found: {design}", err=True)
        raise typer.Exit(1)

    if config is not None and config.exists():
        if sysml_root is not None and sysml_root.exists():
            base_cfg = MissionConfig.from_yaml_with_sysml(config, sysml_root=sysml_root)
        else:
            base_cfg = MissionConfig.from_yaml(config)
    else:
        base_cfg = MissionConfig()

    if max_arrivals is None and not base_cfg.uas_spawn_window_enabled:
        typer.echo(
            "[WARN] --max-arrivals omitted and uas_spawn_window_enabled=False: "
            "the simulation may not terminate.",
            err=True,
        )

    design_df = pd.read_parquet(design)
    typer.echo(
        f"Data farm: points={len(design_df)} reps={reps} seed={seed} "
        f"max_arrivals={max_arrivals} workers={workers}"
    )
    df = run_data_farm(
        base_cfg, design_df, n_reps=reps, base_seed=seed,
        max_arrivals=max_arrivals, workers=workers,
    )
    write_datafarm(
        df, output, base_seed=seed, n_reps=reps,
        max_arrivals=max_arrivals, design_ref=str(design),
    )
    gap_rows = df[
        (df["config"] == "delta") & (df["metric"] == "MOS-1") & (df["aor"] == "ALL")
    ]
    gap = float(gap_rows["value"].mean()) if not gap_rows.empty else float("nan")
    typer.echo(f"Datafarm written: {output}  (rows={len(df)})")
    typer.echo(f"MOS-1 gap (mod-bsl) mean: {gap:+.4f}")


@analysis_app.command(name="stats")
def analysis_stats(
    datafarm: Path = typer.Option(
        ..., "--datafarm", help="datafarm.parquet from 'analysis run'"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    response: str = typer.Option("MOS-1", "--response", help="Target response metric"),
    aor: str = typer.Option("ALL", "--aor", help="Area of responsibility to analyze"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level"),
    bootstrap_seed: int = typer.Option(
        12345, "--bootstrap-seed", help="Bootstrap RNG seed (recorded in report.json)"
    ),
    n_resamples: int = typer.Option(9999, "--n-resamples", help="Bootstrap resamples"),
    no_figures: bool = typer.Option(
        False, "--no-figures", help="Skip PNG figures (headless/CI)"
    ),
) -> None:
    """Analyze a data farm: hypothesis test + features + Pareto + EDA (read-only)."""
    import pandas as pd

    from cuas_sim.analysis.report import build_report, write_report

    if not datafarm.exists():
        typer.echo(f"[ERROR] Datafarm not found: {datafarm}", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(datafarm)
    report = build_report(
        df, response=response, aor=aor, alpha=alpha,
        bootstrap_seed=bootstrap_seed, n_resamples=n_resamples,
    )
    write_report(report, output, with_figures=not no_figures)

    h = report["hypothesis_tests"][0]
    ci = h["ci"]
    ci_txt = "-" if ci is None else f"[{ci['low']:.4f}, {ci['high']:.4f}] ({ci['method']})"
    bp = "-" if h["bootstrap_test"] is None else f"{h['bootstrap_test']['p_value']:.4f}"
    ranking = report["feature_scores"][0]["ranking"]
    top = ranking[0]["factor"] if ranking else "-"
    typer.echo(f"Report written: {output}  (response={response}, aor={aor})")
    typer.echo(
        f"test={h['test']} p={h['p_value']:.4g} reject_h0={h['reject_h0']} "
        f"effect={h['effect']:+.4f} CI={ci_txt} bootstrap_p={bp}"
    )
    typer.echo(f"top factor={top} | pareto points={len(report['pareto_front']['points'])}")


@analysis_app.command(name="predict")
def analysis_predict(
    datafarm: Path = typer.Option(
        ..., "--datafarm", help="datafarm.parquet from 'analysis run'"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    response: str = typer.Option("MOS-1", "--response", help="Target response metric"),
    aor: str = typer.Option("ALL", "--aor", help="Area of responsibility to analyze"),
    model: str = typer.Option("rf", "--model", help="Surrogate model: rf/gbm/gp"),
    seed: int = typer.Option(
        20260701, "--seed", help="Model seed (meta-level; recorded in surrogate.json)"
    ),
    grid_resolution: int = typer.Option(
        25, "--grid-resolution", help="Dense lambda x mix grid resolution"
    ),
    tune: bool = typer.Option(False, "--tune", help="GridSearchCV hyperparameter tuning"),
    no_figures: bool = typer.Option(
        False, "--no-figures", help="Skip the surface PNG (headless/CI)"
    ),
) -> None:
    """Train a surrogate over a data farm and predict the response surface (read-only)."""
    import pandas as pd

    from cuas_sim.analysis.prediction import build_surrogate_report
    from cuas_sim.analysis.report import write_surrogate

    if not datafarm.exists():
        typer.echo(f"[ERROR] Datafarm not found: {datafarm}", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(datafarm)
    try:
        report, grid = build_surrogate_report(
            df, response=response, aor=aor, model=model,
            model_seed=seed, tune=tune, grid_resolution=grid_resolution,
        )
    except ImportError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1) from None
    write_surrogate(report, output, grid, with_figures=not no_figures)

    cv = report["cv"]["mean"]
    r2 = "-" if cv.get("r2") is None else f"{cv['r2']:.4f}"
    mae = "-" if cv.get("mae") is None else f"{cv['mae']:.4f}"
    imps = report["feature_importances"]
    top = imps[0]["factor"] if imps else "-"
    gsum = report["grid_prediction"]["summary"]
    typer.echo(f"Surrogate written: {output}  (model={model}, response={response}, n_train={report['n_train']})")
    typer.echo(
        f"CV R2={r2} MAE={mae} (folds={report['cv']['n_folds']}) | top factor={top}"
    )
    typer.echo(
        f"grid {grid_resolution}x{grid_resolution}: predicted min={gsum['min']:.4f} "
        f"max={gsum['max']:.4f} mean={gsum['mean']:.4f}"
    )


@analysis_app.command(name="report")
def analysis_report(
    datafarm: Path = typer.Option(
        ..., "--datafarm", help="datafarm.parquet from 'analysis run'"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    responses: str = typer.Option(
        "MOS-1,MOE-1,MOE-2,MOE-3", "--responses", help="CSV of responses to consolidate"
    ),
    aor: str = typer.Option(
        "ALL", "--aor", help="Base AoR (ALL sweeps the AoRs present in the datafarm)"
    ),
    fmt: str = typer.Option(
        "html,json", "--format", help="Subset of html,json,md (json is always written)"
    ),
    bootstrap_seed: int = typer.Option(
        20260701, "--bootstrap-seed", help="Bootstrap RNG seed (recorded in report.json)"
    ),
    model_seed: int = typer.Option(
        20260701, "--model-seed", help="Surrogate RNG seed (recorded in report.json)"
    ),
    no_figures: bool = typer.Option(
        False, "--no-figures", help="Skip PNG figures (headless/CI); html without <img>"
    ),
) -> None:
    """Consolidate stats+features+prediction into a friendly report (read-only, no sim)."""
    import json as _json

    import pandas as pd

    from cuas_sim.analysis.report import (
        build_consolidated_report,
        render_figures_png,
        write_consolidated_report,
    )
    from cuas_sim.analysis.reporting import write_html

    if not datafarm.exists():
        typer.echo(f"[ERROR] Datafarm not found: {datafarm}", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(datafarm)
    meta_path = datafarm.with_name(datafarm.name + ".meta.json")
    study_meta: dict[str, object] = {}
    if meta_path.exists():
        study_meta = _json.loads(meta_path.read_text(encoding="utf-8"))

    resp_list = [r.strip() for r in responses.split(",") if r.strip()]
    formats = [f.strip() for f in fmt.split(",") if f.strip()]
    if aor == "ALL":
        present = sorted(str(a) for a in df["aor"].unique() if str(a) != "ALL")
        aors = ["ALL", *present]
    else:
        aors = [aor]

    try:
        report = build_consolidated_report(
            df, responses=resp_list, aors=aors,
            bootstrap_seed=bootstrap_seed, model_seed=model_seed, study_meta=study_meta,
        )
        figures = {} if no_figures else render_figures_png(report)
        write_consolidated_report(report, output, formats=formats, figures_png=figures)
        if "html" in formats:
            write_html(report, output, figures)
    except ImportError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1) from None

    di = report["decision_index"]["ranking"]
    if di:
        top = di[0]
        flag = "significativo" if top["significant"] else "evidencia insuficiente"
        decision = f"{top['aor']} dMOS-1={top['effect']:+.4f} ({flag})"
    else:
        decision = "-"
    used = ",".join(report["study"]["responses"])
    typer.echo(
        f"report -> {output} | responses={used} | format={','.join(formats)}"
    )
    typer.echo(f"decision: {decision}")


if __name__ == "__main__":
    app()
