"""Regression ruler: paired runs over a fixed seed set.

Runs the baseline and the modernized configuration on seeds 1000 to 1000+N-1 of the nominal
scenario and prints, as measured (no threshold is enforced):

* mean MOS-1 of each configuration and the gap (modernized - baseline);
* mean MOS-2, MOE-1, MOE-2 and MOE-3;
* leaks by reason, rockets per engagement and mean pursuit time per configuration;
* the shot-count histogram of the modernized configuration.

The reference values for 20 seeds are in docs/reproducibility.md; any change means that the
engine no longer behaves as the one that produced the published studies.

``--flag-comparison`` runs the seed set with and without the spawn window
(``uas_spawn_window_enabled`` False, then True) and reports the distribution of the number
of arrivals. ``--spawn-interpolation``, ``--sector-draw-rule`` and ``--heading-inward-rule``
override the incursion-generation rules; the defaults of the configuration are ``arc``,
``cdf`` and ``literal``, and the alternatives are the earlier rules.

Usage:
    python scripts/batch_seeds.py [--seeds N] [--config PATH]
    python scripts/batch_seeds.py --seeds 20 --flag-comparison
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from cuas_sim.config import MissionConfig
from cuas_sim.simulation.run import run_paired


def _metric(df, config_name: str, metric: str) -> float:
    sub = df[(df["config_name"] == config_name)
             & (df["metric"] == metric) & (df["aor"] == "ALL")]
    return float(sub["value"].iloc[0]) if not sub.empty else float("nan")


def _arrivals_for_flag(cfg: MissionConfig, seeds: list[int]) -> list[int]:
    # Number of UAS emitted by the spawner, the quantity the spawn window controls.
    # ``res.total_arrivals`` counts baseline engagements plus leaks (the subset that
    # reached the pipeline) and would mask the effect of the window on the source.
    out: list[int] = []
    for seed in seeds:
        res = run_paired(cfg, seed=seed, persist=False)
        out.append(int(len(res.log.uas_births)))
    return out


def _quantiles(xs: list[int]) -> tuple[float, float, float]:
    if not xs:
        nan = float("nan")
        return nan, nan, nan
    sorted_xs = sorted(xs)
    p25 = statistics.quantiles(sorted_xs, n=4)[0] if len(sorted_xs) > 1 else float(sorted_xs[0])
    p50 = statistics.median(sorted_xs)
    p75 = statistics.quantiles(sorted_xs, n=4)[2] if len(sorted_xs) > 1 else float(sorted_xs[0])
    return float(p25), float(p50), float(p75)


def _run_flag_comparison(cfg_base: MissionConfig, seeds: list[int]) -> dict:
    cfg_legacy = cfg_base.replaced(uas_spawn_window_enabled=False)
    cfg_window = cfg_base.replaced(uas_spawn_window_enabled=True)
    print(f"\n--- flag comparison: fixed-window run ({len(seeds)} seeds) ---")
    legacy = _arrivals_for_flag(cfg_legacy, seeds)
    print(f"--- flag comparison: spawn-window run ({len(seeds)} seeds) ---")
    window = _arrivals_for_flag(cfg_window, seeds)

    p25_l, p50_l, p75_l = _quantiles(legacy)
    p25_w, p50_w, p75_w = _quantiles(window)
    gap_abs = p50_l - p50_w
    gap_rel = (gap_abs / p50_l) if p50_l > 0 else float("nan")
    target_met = math.isfinite(gap_rel) and gap_rel >= 0.05

    print()
    print("=== --flag-comparison (#arrivals per seed) ===")
    print(f"seeds                       : {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    print()
    print("| flag                          |   P25 |   P50 |   P75 | min | max |")
    print("|-------------------------------|------:|------:|------:|----:|----:|")
    print(
        f"| uas_spawn_window_enabled=False| {p25_l:5.1f} | {p50_l:5.1f} | {p75_l:5.1f} "
        f"| {min(legacy) if legacy else float('nan'):3.0f} "
        f"| {max(legacy) if legacy else float('nan'):3.0f} |"
    )
    print(
        f"| uas_spawn_window_enabled=True | {p25_w:5.1f} | {p50_w:5.1f} | {p75_w:5.1f} "
        f"| {min(window) if window else float('nan'):3.0f} "
        f"| {max(window) if window else float('nan'):3.0f} |"
    )
    print()
    print(f"median absolute gap (legacy - window): {gap_abs:+.1f}")
    if math.isfinite(gap_rel):
        print(
            f"median relative gap                  : "
            f"{gap_rel:+.4f} ({gap_rel * 100:+.2f} %)"
        )
    else:
        print("median relative gap                  : nan (legacy median is 0)")
    print(f"reduction of the median >= 5 %       : {target_met}")

    report = {
        "spec": "018",
        "feature": "uas_spawn_window",
        "seeds": seeds,
        "arrivals_legacy": legacy,
        "arrivals_window": window,
        "p25_p50_p75_legacy": [p25_l, p50_l, p75_l],
        "p25_p50_p75_window": [p25_w, p50_w, p75_w],
        "median_absolute_gap": gap_abs,
        "median_relative_gap": gap_rel,
        "ac1_target_met": target_met,
    }
    print()
    print("--- JSON ---")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--config", type=Path,
                    default=Path("configs/zida_shield_default.yaml"))
    ap.add_argument("--flag-comparison", action="store_true",
                    help="compare uas_spawn_window_enabled True vs False")
    # Overrides of the incursion-generation rules; omit them to use the configuration values.
    ap.add_argument("--spawn-interpolation", choices=["arc", "chord"], default=None,
                    help="override cfg.spawn_interpolation")
    ap.add_argument("--sector-draw-rule", choices=["cdf", "pdf"], default=None,
                    help="override cfg.sector_draw_rule")
    ap.add_argument("--heading-inward-rule", choices=["literal", "east_flip"], default=None,
                    help="override cfg.heading_inward_rule")
    args = ap.parse_args()

    cfg = MissionConfig.from_yaml_with_sysml(args.config, sysml_root="sysml")

    overrides = {
        k: v for k, v in (
            ("spawn_interpolation", args.spawn_interpolation),
            ("sector_draw_rule", args.sector_draw_rule),
            ("heading_inward_rule", args.heading_inward_rule),
        ) if v is not None
    }
    if overrides:
        cfg = cfg.replaced(**overrides)
        print(f"[overrides applied] {overrides}")

    if args.flag_comparison:
        seeds = [1000 + i for i in range(args.seeds)]
        _run_flag_comparison(cfg, seeds)
        return

    (mos1_b, mos1_m, mos2, moe1_b, moe1_m,
     moe2_b, moe2_m, moe3_b, moe3_m) = ([] for _ in range(9))
    arrivals: list[int] = []
    leak_reasons: Counter[str] = Counter()        # config-tagged below
    leak_by_cfg: dict[str, Counter[str]] = {"baseline": Counter(),
                                            "modernized": Counter()}
    shots_m: Counter[int] = Counter()             # modernized shot-count hist
    rockets_used_m: list[int] = []
    rockets_used_b: list[int] = []
    pursuing_b: list[float] = []                  # PURSUING duration (s)
    pursuing_m: list[float] = []

    for i in range(args.seeds):
        seed = 1000 + i
        res = run_paired(cfg, seed=seed, persist=False)
        df = res.metrics
        arrivals.append(res.total_arrivals)
        mos1_b.append(_metric(df, "baseline", "MOS-1"))
        mos1_m.append(_metric(df, "modernized", "MOS-1"))
        mos2.append(_metric(df, "modernized_vs_baseline", "MOS-2"))
        moe1_b.append(_metric(df, "baseline", "MOE-1"))
        moe1_m.append(_metric(df, "modernized", "MOE-1"))
        moe2_b.append(_metric(df, "baseline", "MOE-2"))
        moe2_m.append(_metric(df, "modernized", "MOE-2"))
        moe3_b.append(_metric(df, "baseline", "MOE-3"))
        moe3_m.append(_metric(df, "modernized", "MOE-3"))
        for lk in res.log.leaks:
            leak_by_cfg[lk["config_name"]][lk["reason"]] += 1
            leak_reasons[lk["reason"]] += 1
        for e in res.log.engagements:
            f = e.fields_
            n = int(f.get("n_shots", 0))
            t_pursuit = float(f.get("t_pursuit_start_s", float("nan")))
            t_env = float(f.get("t_envelope_entry_s", float("nan")))
            pursuing_dur = t_env - t_pursuit
            if f["config_name"] == "modernized":
                shots_m[n] += 1
                rockets_used_m.append(n)
                if math.isfinite(pursuing_dur) and pursuing_dur >= 0.0:
                    pursuing_m.append(pursuing_dur)
            else:
                rockets_used_b.append(n)
                if math.isfinite(pursuing_dur) and pursuing_dur >= 0.0:
                    pursuing_b.append(pursuing_dur)

    def _m(xs: list[float]) -> float:
        return statistics.fmean(xs) if xs else float("nan")

    gap = _m(mos1_m) - _m(mos1_b)
    print(f"\n=== Regression ruler - {args.seeds} seeds (1000..{1000 + args.seeds - 1}) ===")
    print(f"mean arrivals/run        : {_m([float(a) for a in arrivals]):.1f}")
    print(f"MOS-1 baseline (mean)    : {_m(mos1_b):.4f}")
    print(f"MOS-1 modernized (mean)  : {_m(mos1_m):.4f}")
    print(f"MOS-1 gap (mod - bsl)    : {gap:+.4f}   (reference margin +0.10)")
    print(f"MOS-2 (mean)             : {_m(mos2):+.4f}")
    print(f"MOE-1 baseline / mod     : {_m(moe1_b):.4f} / {_m(moe1_m):.4f}")
    print(f"MOE-3 baseline / mod     : {_m(moe3_b):.4f} / {_m(moe3_m):.4f}")
    print(f"MOE-2 baseline / mod     : {_m(moe2_b):.4f} / {_m(moe2_m):.4f}")
    print(f"rockets/engagement bsl   : {_m([float(x) for x in rockets_used_b]):.2f}")
    print(f"rockets/engagement mod   : {_m([float(x) for x in rockets_used_m]):.2f}")
    print(f"PURSUING time bsl (mean) : {_m(pursuing_b):.1f} s")
    print(f"PURSUING time mod (mean) : {_m(pursuing_m):.1f} s")
    print(f"leak reasons (all cfg)   : {dict(leak_reasons)}")
    print(f"  baseline               : {dict(leak_by_cfg['baseline'])}")
    print(f"  modernized             : {dict(leak_by_cfg['modernized'])}")
    print(f"modernized shot-count hist: {dict(sorted(shots_m.items()))}")


if __name__ == "__main__":
    main()
