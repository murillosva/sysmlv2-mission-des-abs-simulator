"""Stopping rule and the treatments outside the standard path of the command line.

Read-only over ``datafarm.parquet``: it recomputes no MOS/MOE/MOP measure and only
regroups what ``MetricsAggregator`` wrote. It reuses the public ``bootstrap_effect`` of the
analysis layer.

Six things the command line does not do:

1. **Nesting of the seed banks**: checks, per design point, that the larger batch contains
   the smaller as an exact prefix. If it fails, comparing two rungs of the ladder is no
   longer convergence on the same sequence, the stopping protocol is invalid, and the
   script exits with code 1.
2. **Stopping rule**: half-width of the 95% bootstrap interval of ΔMOS-1, with the
   confirmatory check of the stability of the estimate. In a multi-point design the
   primary criterion is the maximum half-width over the design points, not the aggregate
   value, which is a mean of the sampled space.
3. **MOS-2 as a one-sample statistic**: MOS-2 is recorded with
   ``config="modernized_vs_baseline"``, has no ``delta`` row and makes ``analysis stats``
   raise ValueError. It gets a one-sample bootstrap of the mean (H0: MOS-2 = 0).
4. **Leaks with zero fill**: ``leak_<reason>`` exists only in the replications where that
   reason occurred, so an absent row means zero leaks; the leaks are reindexed over
   (point x replication x configuration x reason) with zeros before averaging.
5. **Rockets per engagement**: here a zero fill would be wrong, since an absent row means
   no engagement in the replication and the mean is undefined. It is reported as missing,
   with the effective N, and paired only where both arms have the row.
6. **Trend over the factor** (multi-point): least-squares slope of the mean Δ per design
   point against the varied factor, with a 95% interval from a bootstrap over the points,
   for the confirmatory check of the precision of the trend.

In a multi-point design, pooling the replications of all points gives the mean marginal
effect over the sampled range, which is not an operating point. The aggregate block is
labelled with an explicit ``scope``, and the stopping rule does not use it.

Shape of the output: a datafarm with one design point (EX-1) produces the single-point
form; the multi-point keys (``by_design_point``, ``trend``, ``cost_by_design_point``,
``nesting_check.n_design_points``, ``stopping.scope``) appear only with two or more
points, and the aggregate ``cost`` block only with one. ``--assert-baseline <published
json>`` checks that every value of a published artifact is reproduced unchanged.

The array of differences is built as ``report._paired_arrays`` builds it (paired on
``(design_point_id, rep, seed)`` and sorted), so that the effect and the interval match
those of ``report.json`` to the last digit.

Usage::

    # EX-1 (one point)
    python scripts/campaign_postprocess.py \
        --current  outputs/analysis/ex1_mc_nominal/datafarm_n200.parquet \
        --previous outputs/analysis/ex1_mc_nominal/datafarm_n100.parquet \
        --bootstrap-seed 20260804 \
        -o outputs/analysis/ex1_mc_nominal/offpath_n200.json

    # EX-2 (16 points, Latin hypercube in lambda)
    python scripts/campaign_postprocess.py \
        --current  outputs/analysis/ex2_lambda/datafarm_r96.parquet \
        --previous outputs/analysis/ex2_lambda/datafarm_r48.parquet \
        --factor lambda_per_h --half-width 0.05 --ladder 48,96,192 \
        --design outputs/analysis/ex2_lambda/design.parquet \
        --bootstrap-seed 20260806 \
        -o outputs/analysis/ex2_lambda/offpath_r96.json

Console output is ASCII; the JSON is byte-deterministic (sorted keys, no timestamp).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cuas_sim.analysis.stats import bootstrap_effect

# The canonical sort key of the datafarm.
SORT_KEY = ["design_point_id", "rep", "config", "aor", "metric"]
PAIR_KEY = ["design_point_id", "rep", "seed"]

# Thresholds declared before each study, as fractions of the +0.10 reference margin:
# EX-1 (one point):    0.025 = 1/4 of the margin.
# EX-2 (16 points):    0.050 = 1/2 of the margin, applied to the worst point.
# EX-3, EX-4:          0.025 = 1/4 of the margin, applied to the worst point (the curve
#                      crosses the margin, so each point must be classified against it,
#                      not only resolved).
DEFAULT_HALF_WIDTH_TARGET = 0.025  # in units of MOS-1
AC3_MARGIN = 0.10  # reference margin on the MOS-1 gap (labelled "Spec 015 AC-3")
DEFAULT_CONVERGENCE = 0.05  # relative change of the mean and of the deviation between batches
DEFAULT_DRIFT_FRACTION = 0.25  # |mean_c - mean_p| <= 0.25 * half-width_c


def _f(x: Any) -> float | None:
    """float(), with NaN and inf mapped to None (JSON-safe)."""
    if x is None:
        return None
    v = float(x)
    return v if np.isfinite(v) else None


def _fmt_p(p: float | None, n_resamples: int) -> str:
    """An ASL of exactly 0 means that no resample reached the effect: p < 1/B."""
    if p is None:
        return "-"
    if p == 0.0:
        return f"<{1.0 / max(n_resamples, 1):.1e}"
    return f"{p:.4g}"


def paired_delta(df: pd.DataFrame, metric: str, aor: str) -> np.ndarray:
    """Paired differences modernized - baseline, in the order of ``report._paired_arrays``."""
    sub = df[(df["metric"] == metric) & (df["aor"] == aor)]
    bsl = sub[sub["config"] == "baseline"].set_index(PAIR_KEY)["value"]
    mod = sub[sub["config"] == "modernized"].set_index(PAIR_KEY)["value"]
    common = bsl.index.intersection(mod.index).sort_values()
    return (mod.loc[common].to_numpy(float) - bsl.loc[common].to_numpy(float))


def _boot_block(values: np.ndarray, *, seed: int, n_resamples: int) -> dict[str, Any]:
    """Effect, 95% BCa interval and ASL p-value under H0 (Efron and Tibshirani, 1993, ch. 16)."""
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None,
                "half_width": None, "ci_method": None, "p_value": None}
    res = bootstrap_effect(values, bootstrap_seed=seed, n_resamples=n_resamples)
    half = None if res.ci_low is None else (res.ci_high - res.ci_low) / 2.0
    return {
        "n": int(values.size),
        "mean": _f(values.mean()),
        "std": _f(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci_low": _f(res.ci_low),
        "ci_high": _f(res.ci_high),
        "half_width": _f(half),
        "ci_method": res.ci_method,
        "p_value": _f(res.p_value),
        "n_resamples": int(res.n_resamples),
        "bootstrap_seed": int(res.bootstrap_seed),
    }


def check_nesting(current: pd.DataFrame, previous: pd.DataFrame) -> dict[str, Any]:
    """Does the larger batch contain the smaller as an exact prefix, per design point?"""
    n_prev = int(previous["rep"].max()) + 1
    head = current[current["rep"] < n_prev]

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(SORT_KEY, kind="mergesort").reset_index(drop=True)

    a, b = _norm(head), _norm(previous)
    rows_equal = bool(a.shape == b.shape and a.equals(b))

    # Seed banks are compared per design point: with several points, the same ``rep``
    # carries different seeds at different points (key = (dp, rep)).
    def _seeds(df: pd.DataFrame) -> dict[tuple[int, int], int]:
        uniq = df[["design_point_id", "rep", "seed"]].drop_duplicates()
        return {(int(d), int(r)): int(s) for d, r, s in uniq.to_numpy().tolist()}

    seeds_cur, seeds_prev = _seeds(head), _seeds(previous)
    seeds_equal = seeds_cur == seeds_prev
    per_point = sorted({int(d) for d, _ in seeds_prev})
    # The note of the single-point path keeps the exact wording of the published EX-1
    # artifact (offpath_n200.json), so that the command of its MANIFEST reproduces the
    # file byte for byte.
    note = (
        "lote maior contem o menor como prefixo exato -- '100 vs 200' e "
        "convergencia sobre a MESMA sequencia (sec.7.2)"
        if len(per_point) <= 1 else
        "lote maior contem o menor como prefixo exato EM CADA ponto de projeto "
        "-- a comparacao entre degraus e convergencia sobre a MESMA sequencia "
        "(sec.7.2)"
    )
    out = {
        "n_reps_previous": n_prev,
        "n_reps_current": int(current["rep"].max()) + 1,
        "seed_banks_identical": seeds_equal,
        "rows_identical": rows_equal,
        "passed": bool(rows_equal and seeds_equal),
        "note": note,
    }
    if len(per_point) > 1:  # multi-point key: see the shape rule in the module docstring
        out["n_design_points"] = len(per_point)
    return out


def stopping_block(
    current: pd.DataFrame, previous: pd.DataFrame | None, *,
    response: str, aor: str, seed: int, n_resamples: int,
    half_width_target: float, convergence: float, drift_fraction: float,
) -> dict[str, Any]:
    """Primary criterion (half-width) and confirmatory check (stability), aggregate block."""
    cur = _boot_block(paired_delta(current, response, aor), seed=seed,
                      n_resamples=n_resamples)
    prev = (None if previous is None else
            _boot_block(paired_delta(previous, response, aor), seed=seed,
                        n_resamples=n_resamples))

    half = cur["half_width"]
    primary = bool(half is not None and half <= half_width_target)

    secondary: dict[str, Any] | None = None
    if prev is not None and prev["mean"] is not None and cur["mean"] is not None:
        drift = abs(cur["mean"] - prev["mean"])
        rel_mean = drift / abs(prev["mean"]) if prev["mean"] != 0 else drift
        p_std, c_std = prev["std"] or 0.0, cur["std"] or 0.0
        rel_std = abs(c_std - p_std) / abs(p_std) if p_std != 0 else abs(c_std - p_std)
        drift_ok = bool(half is not None and drift <= drift_fraction * half)
        secondary = {
            "drift": _f(drift),
            "drift_budget": _f(None if half is None else drift_fraction * half),
            "drift_ok": drift_ok,
            "rel_change_mean": _f(rel_mean),
            "rel_change_std": _f(rel_std),
            "convergence_ok": bool(rel_mean < convergence and rel_std < convergence),
            "passed": bool(drift_ok and rel_mean < convergence and rel_std < convergence),
        }

    return {
        "response": response,
        "aor": aor,
        "current": cur,
        "previous": prev,
        "primary": {
            "half_width": half,
            "target": half_width_target,
            "passed": primary,
            "rule": "semi-largura do IC95% bootstrap de Delta MOS-1 <= alvo",
        },
        "secondary": secondary,
        "verdict": "ACEITAR N" if primary else "ESTENDER N (proximo degrau da escada)",
    }


def mos2_one_sample(df: pd.DataFrame, *, seed: int, n_resamples: int) -> dict[str, Any]:
    """MOS-2 outside the standard path: one-sample bootstrap."""
    sub = df[(df["metric"] == "MOS-2") & (df["aor"] == "ALL")].sort_values("rep")
    values = sub["value"].to_numpy(float)
    block = _boot_block(values, seed=seed, n_resamples=n_resamples)
    block["config_in_datafarm"] = "modernized_vs_baseline"
    block["h0"] = "media de MOS-2 = 0"
    block["note"] = (
        "NAO e teste pareado: MOS-2 ja e um contraste por construcao e nao gera "
        "linha delta; analysis stats --response MOS-2 levanta ValueError"
    )
    block["fraction_positive"] = _f(np.mean(values > 0.0)) if values.size else None
    return block


def _series_by_rep(df: pd.DataFrame, metric: str, config: str) -> pd.Series:
    """Series indexed by replication; requires a datafarm of ONE design point.

    With several points there are several rows per ``rep`` and the reindex would raise
    ``cannot reindex on an axis with duplicate labels``. Callers always pass the slice of
    one point (see ``_split_points``).
    """
    sub = df[(df["metric"] == metric) & (df["config"] == config) & (df["aor"] == "ALL")]
    return sub.set_index("rep")["value"].sort_index()


def leak_block(df: pd.DataFrame, *, seed: int, n_resamples: int) -> dict[str, Any]:
    """Leaks reindexed with zero fill (absent == zero occurrences)."""
    reps = sorted(int(r) for r in df["rep"].unique())
    reasons = sorted(m for m in df["metric"].astype(str).unique() if m.startswith("leak_"))
    out: dict[str, Any] = {
        "reindex_rule": "produto (rep x config x motivo) com zero-fill; ausente = zero",
        "n_reps": len(reps),
        "reasons": {},
    }
    for reason in reasons:
        bsl = _series_by_rep(df, reason, "baseline").reindex(reps, fill_value=0.0)
        mod = _series_by_rep(df, reason, "modernized").reindex(reps, fill_value=0.0)
        delta = mod.to_numpy(float) - bsl.to_numpy(float)
        out["reasons"][reason] = {
            "baseline_mean": _f(bsl.mean()),
            "modernized_mean": _f(mod.mean()),
            "reps_with_row_baseline": int(_series_by_rep(df, reason, "baseline").size),
            "reps_with_row_modernized": int(_series_by_rep(df, reason, "modernized").size),
            "paired_delta": _boot_block(delta, seed=seed, n_resamples=n_resamples),
        }
    return out


def rockets_block(df: pd.DataFrame, *, seed: int, n_resamples: int) -> dict[str, Any]:
    """Rockets per engagement: missing is NOT zero (mean conditional on engaging)."""
    bsl = _series_by_rep(df, "rockets_per_engagement", "baseline")
    mod = _series_by_rep(df, "rockets_per_engagement", "modernized")
    common = bsl.index.intersection(mod.index).sort_values()
    delta = mod.loc[common].to_numpy(float) - bsl.loc[common].to_numpy(float)
    n_reps = int(df["rep"].nunique())
    return {
        "missing_rule": (
            "ausente = replica SEM engajamento -> media indefinida, NAO zero; "
            "nao aplicar zero-fill (contraste com leak_*)"
        ),
        "n_reps": n_reps,
        "baseline": {"n_eff": int(bsl.size), "mean": _f(bsl.mean()) if bsl.size else None},
        "modernized": {"n_eff": int(mod.size), "mean": _f(mod.mean()) if mod.size else None},
        "paired_delta": _boot_block(delta, seed=seed, n_resamples=n_resamples),
        "paired_note": f"pareado so nas {len(common)} de {n_reps} replicas com a linha nos dois bracos",
    }


# ─── multi-point: per design point, sparsity and trend ──────────────────────


def _split_points(df: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    """Slices per design point, in id order (determinism)."""
    return [(int(dp), df[df["design_point_id"] == dp])
            for dp in sorted(df["design_point_id"].unique())]


def degenerate_reps(sub: pd.DataFrame) -> dict[str, Any]:
    """Replications without engagement in EITHER arm, a proxy of replications with no arrival.

    The aggregator writes ``MOS-1 = 0.0`` when ``total_arrivals == 0``, so the pair enters Δ
    and pulls the mean towards zero without any signal. They are counted and reported;
    a study that reaches lambda near 2 per hour can produce them.
    """
    reps = {int(r) for r in sub["rep"].unique()}
    rk = sub[sub["metric"] == "rockets_per_engagement"]
    engaged = {int(r) for r in rk["rep"].unique()}
    no_eng = sorted(reps - engaged)
    mos1 = sub[(sub["metric"] == "MOS-1") & (sub["aor"] == "ALL")
               & (sub["config"] == "delta")].set_index("rep")["value"]
    zero_delta = [r for r in no_eng if r in mos1.index and float(mos1.loc[r]) == 0.0]
    return {
        "n_reps": len(reps),
        "n_no_engagement_either_arm": len(no_eng),
        "n_degenerate_zero_delta": len(zero_delta),
        "fraction_degenerate": _f(len(zero_delta) / len(reps)) if reps else None,
        "rule": (
            "sem linha rockets_per_engagement em nenhum braco = replica sem "
            "engajamento; MOS-1 sai 0.0 nos dois bracos e o par entra no Delta "
            "puxando para zero (ausente != zero, sec.5.4)"
        ),
    }


def by_design_point(
    df: pd.DataFrame, *, response: str, aor: str, factor: str,
    seed: int, n_resamples: int, half_width_target: float,
) -> dict[str, Any]:
    """Block per design point: Δ, interval, sparsity and effective N (not pooled)."""
    points: list[dict[str, Any]] = []
    for dp, sub in _split_points(df):
        blk = _boot_block(paired_delta(sub, response, aor), seed=seed,
                          n_resamples=n_resamples)
        rk = rockets_block(sub, seed=seed, n_resamples=n_resamples)
        points.append({
            "design_point_id": dp,
            factor: _f(sub[factor].iloc[0]) if factor in sub.columns else None,
            **blk,
            "half_width_ok": (None if blk["half_width"] is None
                              else bool(blk["half_width"] <= half_width_target)),
            "sparsity": degenerate_reps(sub),
            "n_eff_rockets": {"baseline": rk["baseline"]["n_eff"],
                              "modernized": rk["modernized"]["n_eff"]},
            "mos2": mos2_one_sample(sub, seed=seed, n_resamples=n_resamples),
        })

    halves = [(p["design_point_id"], p["half_width"]) for p in points
              if p["half_width"] is not None]
    worst = max(halves, key=lambda t: t[1]) if halves else (None, None)
    return {
        "response": response,
        "aor": aor,
        "factor": factor,
        "n_design_points": len(points),
        "points": points,
        "max_half_width": worst[1],
        "worst_design_point_id": worst[0],
        "all_points_pass": bool(halves and worst[1] <= half_width_target),
        "target": half_width_target,
        "rule": (
            "criterio PRIMARIO do EX-2: max sobre os pontos de projeto da "
            "semi-largura do IC95% BCa <= alvo. O `max` e declarado a priori "
            "como estatistica -- nao se escolhe o ponto depois de ver os dados"
        ),
    }


def trend_block(
    df: pd.DataFrame, *, response: str, aor: str, factor: str,
    seed: int, n_resamples: int,
) -> dict[str, Any]:
    """Least-squares slope of the mean Δ per point against the factor, with a bootstrap interval.

    Confirmatory check of the precision of the trend: the half-width of the interval of the
    TOTAL change over the sampled range (range x half-width of the slope interval) must stay
    under the same anchor as the primary criterion.
    """
    rows = []
    for _dp, sub in _split_points(df):
        d = paired_delta(sub, response, aor)
        if d.size and factor in sub.columns:
            rows.append((float(sub[factor].iloc[0]), float(d.mean())))
    if len(rows) < 3:
        return {"factor": factor, "n_points": len(rows), "slope": None,
                "note": "pontos de projeto insuficientes para tendencia"}

    x = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)

    # Bootstrap over the DESIGN POINTS (the independent unit; Kleijnen, 2015).
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    slopes = np.empty(n_resamples)
    for i in range(n_resamples):
        xb, yb = x[idx[i]], y[idx[i]]
        slopes[i] = np.polyfit(xb, yb, 1)[0] if xb.std() > 0 else np.nan
    slopes = slopes[np.isfinite(slopes)]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    span = float(x.max() - x.min())
    half_slope = float((hi - lo) / 2.0)
    return {
        "factor": factor,
        "n_points": len(x),
        "slope": _f(slope),
        "intercept": _f(intercept),
        "slope_ci_low": _f(lo),
        "slope_ci_high": _f(hi),
        "slope_half_width": _f(half_slope),
        "ci_method": "percentile bootstrap over design points",
        "n_resamples": int(slopes.size),
        "factor_span": _f(span),
        "total_change_over_span": _f(slope * span),
        "total_change_half_width": _f(half_slope * span),
        "pearson": _f(np.corrcoef(x, y)[0, 1]),
        "note": (
            "unidade de reamostragem = PONTO DE PROJETO (replicas de um ponto "
            "sao correlacionadas; reamostrar replicas inflaria a precisao)"
        ),
    }


def _design_meta(design: Path | None) -> dict[str, Any] | None:
    """Echo the design sidecar (the design seed does not reach report.json)."""
    if design is None:
        return None
    meta_path = design.with_name(design.name + ".meta.json")
    prov_path = design.with_name(design.stem + ".provenance.json")
    out: dict[str, Any] = {"design_ref": str(design)}
    if meta_path.exists():
        out["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    if prov_path.exists():
        out["provenance"] = json.loads(prov_path.read_text(encoding="utf-8"))
    out["why"] = (
        "validity.seeds.design sai null no report.json (o analysis report so le o "
        "sidecar do datafarm) -- este eco fecha a rastreabilidade do design_seed "
        "sem tocar a camada (docs/experimental_campaign.md sec.7.5 item 2)"
    )
    return out


def _margin_fraction_label(half_width: float) -> str:
    """Label of the threshold anchor as a fraction of the +0.10 reference margin.

    The text of ``half_width_rationale`` is derived from the target, so that it is correct
    for every study; for ``half_width=0.05`` it returns exactly "1/2", the wording of the
    published ``offpath_r96.json`` of EX-2.
    """
    frac = half_width / AC3_MARGIN
    for denom in (2, 3, 4, 5, 8, 10):
        if abs(frac - 1.0 / denom) < 1e-9:
            return f"1/{denom}"
    return f"{frac:g}x"


def _compare_baseline(report: dict[str, Any], baseline_path: Path) -> tuple[bool, list[str]]:
    """Non-regression: every value of the published artifact must be reproduced unchanged."""
    old = json.loads(baseline_path.read_text(encoding="utf-8"))
    diffs: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict):
            if not isinstance(b, dict):
                diffs.append(f"{path}: tipo mudou")
                return
            for k in a:
                if k not in b:
                    diffs.append(f"{path}.{k}: chave sumiu")
                else:
                    walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                diffs.append(f"{path}: lista mudou")
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, float) and isinstance(b, float):
            if a != b:
                diffs.append(f"{path}: {a!r} -> {b!r}")
        elif a != b:
            diffs.append(f"{path}: {a!r} -> {b!r}")

    walk(old, report, "")
    return (not diffs), diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stopping rule, MOS-2 and cost blocks outside the standard path."
    )
    ap.add_argument("--current", type=Path, required=True,
                    help="datafarm.parquet of the current batch (the largest of the ladder)")
    ap.add_argument("--previous", type=Path, default=None,
                    help="datafarm.parquet of the previous batch (nested in the current one)")
    ap.add_argument("--response", default="MOS-1", help="Response of the stopping rule")
    ap.add_argument("--aor", default="ALL", help="Area of responsibility of the stopping rule")
    ap.add_argument("--factor", default=None,
                    help="Varied factor (e.g. lambda_per_h); enables the multi-point blocks")
    ap.add_argument("--design", type=Path, default=None,
                    help="design.parquet, to echo the design seed of its sidecar")
    ap.add_argument("--ladder", default=None,
                    help="Ladder declared before the study, CSV (e.g. 48,96,192)")
    ap.add_argument("--bootstrap-seed", type=int, default=20260804,
                    help="Seed of the bootstrap (recorded in the output)")
    ap.add_argument("--n-resamples", type=int, default=9999)
    ap.add_argument("--half-width", type=float, default=DEFAULT_HALF_WIDTH_TARGET,
                    help="Half-width target of the 95%% interval (units of the response)")
    ap.add_argument("--convergence", type=float, default=DEFAULT_CONVERGENCE,
                    help="Maximum relative change of mean and deviation between batches")
    ap.add_argument("--drift-fraction", type=float, default=DEFAULT_DRIFT_FRACTION,
                    help="Maximum drift of the mean, as a fraction of the half-width")
    ap.add_argument("--assert-baseline", type=Path, default=None,
                    help="Published JSON to check (non-regression); exit 1 on divergence")
    ap.add_argument("--output", "-o", type=Path, required=True, help="Output JSON")
    args = ap.parse_args(argv)

    for path in (args.current, args.previous):
        if path is not None and not path.exists():
            print(f"[ERROR] datafarm not found: {path}", file=sys.stderr)
            return 1

    current = pd.read_parquet(args.current)
    previous = pd.read_parquet(args.previous) if args.previous else None
    n_points = int(current["design_point_id"].nunique())
    multi = n_points > 1

    if multi and args.factor is None:
        print(f"[ERROR] the datafarm has {n_points} design points: --factor is "
              "REQUIRED (without it the per-replication blocks would silently pool "
              "different points)", file=sys.stderr)
        return 1

    nesting = None if previous is None else check_nesting(current, previous)
    stopping = stopping_block(
        current, previous, response=args.response, aor=args.aor,
        seed=args.bootstrap_seed, n_resamples=args.n_resamples,
        half_width_target=args.half_width, convergence=args.convergence,
        drift_fraction=args.drift_fraction,
    )
    if multi:
        stopping["scope"] = (
            "AGREGADO sobre os pontos de projeto = efeito marginal MEDIO na faixa "
            "amostrada; NAO e um ponto de operacao. O criterio primario do EX-2 e o "
            "max por ponto, em by_design_point -- nao este bloco"
        )
        stopping["primary"]["governs_stop"] = False

    # Cost blocks: in a multi-point design, pooling leaks and rockets would mix
    # different lambdas; the right aggregation is per point (the pooled value would be the
    # mean of the space, not a scenario). Only the single-point path writes the aggregate
    # `cost` block.
    report: dict[str, Any] = {
        "inputs": {
            "current": str(args.current),
            "previous": None if args.previous is None else str(args.previous),
            "bootstrap_seed": args.bootstrap_seed,
            "n_resamples": args.n_resamples,
        },
        "criteria_declared_a_priori": {
            "half_width_target": args.half_width,
            "half_width_rationale": (
                "1/4 da margem de aceitacao do Spec 015 AC-3 (gap MOS-1 >= +0.10); "
                "~10% do gap esperado (+0.2378, regua R3)"
            ),
            "ladder": [100, 200, 400],
            "convergence_threshold": args.convergence,
            "drift_fraction": args.drift_fraction,
            "alpha": 0.05,
            "confidence_level": 0.95,
        },
        "nesting_check": nesting,
        "stopping": stopping,
        "mos2_one_sample": mos2_one_sample(
            current, seed=args.bootstrap_seed, n_resamples=args.n_resamples
        ),
    }
    if args.ladder:
        report["criteria_declared_a_priori"]["ladder"] = [
            int(v) for v in args.ladder.split(",") if v.strip()
        ]
    if multi:
        report["criteria_declared_a_priori"]["half_width_rationale"] = (
            f"{_margin_fraction_label(args.half_width)} da margem de aceitacao do "
            "Spec 015 AC-3 (gap MOS-1 >= +0.10), aplicado ao PIOR ponto de projeto: "
            "cada ponto da curva deve decidir sozinho se supera a margem"
        )
        report["criteria_declared_a_priori"]["primary_rule"] = (
            "max sobre pontos de projeto da semi-largura do IC95% BCa <= alvo"
        )
        report["mos2_one_sample"]["scope"] = (
            "POOLADO sobre pontos de projeto = media do espaco; o tratamento por "
            "ponto esta em by_design_point[].mos2"
        )
        report["by_design_point"] = by_design_point(
            current, response=args.response, aor=args.aor, factor=args.factor,
            seed=args.bootstrap_seed, n_resamples=args.n_resamples,
            half_width_target=args.half_width,
        )
        report["trend"] = trend_block(
            current, response=args.response, aor=args.aor, factor=args.factor,
            seed=args.bootstrap_seed, n_resamples=args.n_resamples,
        )
        report["cost_by_design_point"] = {
            str(dp): {
                "leaks": leak_block(sub, seed=args.bootstrap_seed,
                                    n_resamples=args.n_resamples),
                "rockets_per_engagement": rockets_block(
                    sub, seed=args.bootstrap_seed, n_resamples=args.n_resamples
                ),
            }
            for dp, sub in _split_points(current)
        }
    else:
        report["cost"] = {
            "leaks": leak_block(current, seed=args.bootstrap_seed,
                                n_resamples=args.n_resamples),
            "rockets_per_engagement": rockets_block(
                current, seed=args.bootstrap_seed, n_resamples=args.n_resamples
            ),
        }
    if args.design is not None:
        report["design"] = _design_meta(args.design)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")

    cur = stopping["current"]
    print(f"=== campaign postprocess -> {args.output} ===")
    print(f"design points: {n_points} | replications per point: "
          f"{int(current['rep'].max()) + 1}")
    if nesting is not None:
        flag = "OK" if nesting["passed"] else "FAILED"
        print(f"seed-bank nesting ({nesting['n_reps_previous']} in "
              f"{nesting['n_reps_current']}, {n_points} points): {flag}")
    label = "AGGREGATE (mean of the space)" if multi else f"N={cur['n']}"
    print(f"{args.response} ({args.aor}) {label}: effect {cur['mean']:+.4f} "
          f"CI95 [{cur['ci_low']:+.4f}, {cur['ci_high']:+.4f}] ({cur['ci_method']})")
    if multi:
        bdp = report["by_design_point"]
        print(f"  PRIMARY CRITERION: max half-width {bdp['max_half_width']:.4f} "
              f"(point {bdp['worst_design_point_id']}) vs target {args.half_width:.4f} "
              f"-> {'MET' if bdp['all_points_pass'] else 'NOT met'}")
        n_fail = sum(1 for p in bdp["points"] if p["half_width_ok"] is False)
        print(f"  points above the target: {n_fail} of {bdp['n_design_points']}")
        tr = report["trend"]
        if tr.get("slope") is not None:
            print(f"  trend: slope {tr['slope']:+.5f} per unit of "
                  f"{tr['factor']} CI95 [{tr['slope_ci_low']:+.5f}, "
                  f"{tr['slope_ci_high']:+.5f}]")
            print(f"    total change over the range ({tr['factor_span']:.2f}): "
                  f"{tr['total_change_over_span']:+.4f} "
                  f"+/- {tr['total_change_half_width']:.4f}")
        deg = sum(p["sparsity"]["n_degenerate_zero_delta"] for p in bdp["points"])
        print(f"  degenerate replications (no engagement in either arm): {deg}")
    else:
        print(f"  half-width {cur['half_width']:.4f} vs target {args.half_width:.4f} "
              f"-> {'MET' if stopping['primary']['passed'] else 'NOT met'}")
    sec = stopping["secondary"]
    if sec is not None:
        print(f"  stability: drift {sec['drift']:.4f} (budget "
              f"{sec['drift_budget']:.4f}), rel. change mean {sec['rel_change_mean']:.4f} / "
              f"deviation {sec['rel_change_std']:.4f} -> "
              f"{'OK' if sec['passed'] else 'NOT OK'}")
    if not multi:
        print(f"  VERDICT: {stopping['verdict']}")
    m2 = report["mos2_one_sample"]
    if m2["n"]:
        print(f"MOS-2 (one sample) N={m2['n']}: mean {m2['mean']:+.4f} "
              f"CI95 [{m2['ci_low']:+.4f}, {m2['ci_high']:+.4f}] "
              f"p {_fmt_p(m2['p_value'], args.n_resamples)}")
    if not multi:
        for reason, blk in report["cost"]["leaks"]["reasons"].items():
            d = blk["paired_delta"]
            print(f"{reason}: bsl {blk['baseline_mean']:.3f} -> mod "
                  f"{blk['modernized_mean']:.3f} (delta {d['mean']:+.3f})")
        rk = report["cost"]["rockets_per_engagement"]
        b_m, m_m = rk["baseline"]["mean"], rk["modernized"]["mean"]
        print(f"rockets/eng: bsl {b_m:.3f} (n={rk['baseline']['n_eff']}) -> "
              f"mod {m_m:.3f} (n={rk['modernized']['n_eff']}) of {rk['n_reps']} replications")

    rc = 0
    if args.assert_baseline is not None:
        ok, diffs = _compare_baseline(report, args.assert_baseline)
        if ok:
            print(f"NON-REGRESSION vs {args.assert_baseline}: OK (every published value "
                  "reproduced)")
        else:
            print(f"[ERROR] NON-REGRESSION vs {args.assert_baseline}: "
                  f"{len(diffs)} divergence(s)", file=sys.stderr)
            for d in diffs[:20]:
                print(f"  {d}", file=sys.stderr)
            rc = 1

    if nesting is not None and not nesting["passed"]:
        print("[ERROR] seed banks NOT nested: the stopping protocol requires an "
              "exact prefix. Review before citing the study.",
              file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
