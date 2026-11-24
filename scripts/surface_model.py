"""Surface model with the interaction term, for a two-factor design (EX-4).

Read-only over ``datafarm.parquet``: it recomputes no MOS/MOE/MOP measure and only
regroups what ``MetricsAggregator`` wrote. ``campaign_postprocess.py`` accepts one
``--factor`` and its trend block is a simple regression on that factor, which cannot reach
the interaction coefficient; this script fits the matrix ``[1, L, p, L*p]``.

Outputs, in one JSON file:

1. **Bilinear surface** ``Delta = b0 + b1*L + b2*p + b3*L*p`` fitted by ordinary least
   squares to the per-design-point means of the paired difference; the design point is
   the independent unit (the replications of a point are correlated; Kleijnen, 2015).
2. **Interval of each coefficient** from a bootstrap over the design points, including
   the interaction coefficient ``b3``.
3. **Partial F test of additivity** (bilinear against the additive model).
4. **Slope in the mix by lambda**, ``b2 + b3*L``, with its interval.
5. **Frontier of the margin**: ``p*(L)`` such that ``Delta(L, p*)`` equals the margin
   (default +0.10), with a band from the same bootstrap over the design points.
6. **Quadratic model** (``+ L^2, p^2``) as a robustness check, not as the headline: the
   value of ``b3`` depends on the model.
7. **Cross-check against measured lines** (``--compare``): evaluates the fitted surface
   (and, with ``--grid``, the grid of the exploratory surrogate) along the lines
   p_mr = 0.5 and lambda = 6 and compares it with the points measured by EX-2 and EX-3 on
   disjoint seed banks.

Usage::

    python scripts/surface_model.py \\
        --datafarm outputs/analysis/ex4_interaction/datafarm_r384.parquet \\
        --design   outputs/analysis/ex4_interaction/design.parquet \\
        --bootstrap-seed 20260808 \\
        --compare outputs/analysis/ex2_lambda/offpath_r96.json \\
        --compare outputs/analysis/ex3_mix/offpath_r192.json \\
        --grid    outputs/analysis/ex4_interaction/predict_rf/grid_prediction.csv \\
        -o outputs/analysis/ex4_interaction/surface_r384.json

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
from scipy import stats

# Pairing key, as in the analysis layer and campaign_postprocess.py.
PAIR_KEY = ["design_point_id", "rep", "seed"]

LAMBDA_COL = "lambda_per_h"
PMR_COL = "uas_archetype_p_mr"

AC3_MARGIN = 0.10  # reference margin on the MOS-1 gap (labelled "AC-3" in the records)
# Intensities at which the slope in the mix and the frontier are reported: the nominal
# lambda = 6, the ends of the nominal range and an intermediate value.
REPORT_LAMBDAS = (2.0, 6.0, 12.0, 18.0)


def _f(x: Any) -> float | None:
    """float(), with NaN and inf mapped to None (JSON-safe)."""
    if x is None:
        return None
    v = float(x)
    return v if np.isfinite(v) else None


def point_table(df: pd.DataFrame, response: str, aor: str) -> pd.DataFrame:
    """Mean and standard deviation of the paired difference per design point, with both factors."""
    sub = df[(df["metric"] == response) & (df["aor"] == aor)]
    bsl = sub[sub["config"] == "baseline"].set_index(PAIR_KEY)["value"]
    mod = sub[sub["config"] == "modernized"].set_index(PAIR_KEY)["value"]
    common = bsl.index.intersection(mod.index).sort_values()
    delta = (mod.loc[common] - bsl.loc[common]).rename("delta").reset_index()
    agg = delta.groupby("design_point_id")["delta"].agg(["mean", "std", "count"])
    fac = df.groupby("design_point_id")[[LAMBDA_COL, PMR_COL]].first()
    return agg.join(fac).sort_index()


def _design_matrix(lam: np.ndarray, pmr: np.ndarray, *, quadratic: bool) -> np.ndarray:
    cols = [np.ones(lam.size), lam, pmr, lam * pmr]
    if quadratic:
        cols += [lam ** 2, pmr ** 2]
    return np.column_stack(cols)


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Coefficients, R^2 and residual sum of squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    ss_res = float(r @ r)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")), ss_res


def _frontier(beta: np.ndarray, lam: float, margin: float) -> float:
    """p* such that Delta(lam, p*) equals the margin, in the bilinear model."""
    den = beta[2] + beta[3] * lam
    if den == 0.0:
        return float("nan")
    return (margin - beta[0] - beta[1] * lam) / den


def fit_surface(
    pts: pd.DataFrame, *, seed: int, n_resamples: int, margin: float,
    quadratic: bool = False,
) -> dict[str, Any]:
    """Fit and bootstrap intervals over the design points (the independent unit)."""
    lam = pts[LAMBDA_COL].to_numpy(float)
    pmr = pts[PMR_COL].to_numpy(float)
    y = pts["mean"].to_numpy(float)
    n = lam.size

    X = _design_matrix(lam, pmr, quadratic=quadratic)
    beta, r2, ss_res = _ols(X, y)
    names = ["intercept", "lambda", "p_mr", "lambda_x_p_mr"]
    if quadratic:
        names += ["lambda_sq", "p_mr_sq"]

    # Partial F test of additivity: bilinear against [1, L, p] (no cross term).
    Xa = np.column_stack([np.ones(n), lam, pmr])
    _, r2_add, ss_add = _ols(Xa, y)
    k = X.shape[1]
    dfree = n - k
    f_stat = ((ss_add - ss_res) / (ss_res / dfree)) if dfree > 0 and ss_res > 0 else float("nan")
    p_add = float(stats.f.sf(f_stat, 1, dfree)) if np.isfinite(f_stat) else None

    # Bootstrap over the design points (Kleijnen, 2015).
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_beta = np.full((n_resamples, k), np.nan)
    boot_slope = np.full((n_resamples, len(REPORT_LAMBDAS)), np.nan)
    boot_front = np.full((n_resamples, len(REPORT_LAMBDAS)), np.nan)
    for i in range(n_resamples):
        j = idx[i]
        Xb = _design_matrix(lam[j], pmr[j], quadratic=quadratic)
        # A degenerate resample (rank-deficient) is discarded, not repaired.
        if np.linalg.matrix_rank(Xb) < k:
            continue
        b, *_ = np.linalg.lstsq(Xb, y[j], rcond=None)
        boot_beta[i] = b
        for c, L in enumerate(REPORT_LAMBDAS):
            boot_slope[i, c] = b[2] + b[3] * L
            boot_front[i, c] = _frontier(b, L, margin)

    ok = np.isfinite(boot_beta[:, 0])
    n_ok = int(ok.sum())

    def _ci(col: np.ndarray) -> tuple[float | None, float | None, float | None]:
        v = col[np.isfinite(col)]
        if v.size < 100:
            return None, None, None
        lo, hi = np.percentile(v, [2.5, 97.5])
        return _f(lo), _f(hi), _f((hi - lo) / 2.0)

    coefficients = {}
    for c, name in enumerate(names):
        lo, hi, hw = _ci(boot_beta[:, c])
        coefficients[name] = {
            "estimate": _f(beta[c]), "ci_low": lo, "ci_high": hi, "half_width": hw,
            "excludes_zero": (None if lo is None else bool(lo > 0 or hi < 0)),
        }

    mix_slope, frontier = {}, {}
    for c, L in enumerate(REPORT_LAMBDAS):
        lo, hi, hw = _ci(boot_slope[:, c])
        mix_slope[f"{L:g}"] = {
            "estimate": _f(beta[2] + beta[3] * L), "ci_low": lo, "ci_high": hi,
            "half_width": hw,
        }
        if not quadratic:
            lo2, hi2, hw2 = _ci(boot_front[:, c])
            frontier[f"{L:g}"] = {
                "estimate": _f(_frontier(beta, L, margin)), "ci_low": lo2,
                "ci_high": hi2, "half_width": hw2,
            }

    out: dict[str, Any] = {
        "model": "quadratic [1,L,p,L*p,L^2,p^2]" if quadratic else "bilinear [1,L,p,L*p]",
        "n_design_points": n,
        "coefficients": coefficients,
        "r_squared": _f(r2),
        "r_squared_additive": _f(r2_add),
        "residual_sd": _f(np.sqrt(ss_res / dfree)) if dfree > 0 else None,
        "additivity_F_test": {
            "F": _f(f_stat), "df_num": 1, "df_den": int(dfree), "p_value": p_add,
            "rejects_additivity": (None if p_add is None else bool(p_add < 0.05)),
            "note": ("mesmo teste que o piloto ME-1 rejeitou com F(1,28)=11.88, "
                     "p=0.0018 (docs/experimental_campaign.md sec.8.4)"),
        },
        "mix_slope_by_lambda": mix_slope,
        "bootstrap": {
            "unit": "design point (Kleijnen 2015)",
            "n_resamples_requested": int(n_resamples),
            "n_resamples_used": n_ok,
            "seed": int(seed),
            "ci_method": "percentile bootstrap over design points",
        },
    }
    if not quadratic:
        out["frontier_p_mr_at_margin"] = frontier
        out["frontier_margin"] = margin
        out["frontier_rule"] = (
            "p*(L) = (margem - b0 - b1*L) / (b2 + b3*L); so e afirmavel onde cai "
            "DENTRO da faixa realizada de p_mr -- fora dela, reportar 'sem "
            "cruzamento na faixa amostrada', nunca extrapolar"
        )
    return out


# The value column of grid_prediction.csv is "predicted"; "prediction" is accepted too.
_GRID_VALUE_COLUMNS = ("predicted", "prediction")


def _grid_value_column(grid: pd.DataFrame) -> str:
    for name in _GRID_VALUE_COLUMNS:
        if name in grid.columns:
            return name
    raise KeyError(
        f"grid_prediction.csv has no value column {_GRID_VALUE_COLUMNS}: "
        f"columns present = {list(grid.columns)}"
    )


def _bilinear_grid_lookup(grid: pd.DataFrame, lam: float, pmr: float) -> float:
    """Bilinear interpolation on the grid of the surrogate (grid_prediction.csv)."""
    ls = np.sort(grid[LAMBDA_COL].unique())
    ps = np.sort(grid[PMR_COL].unique())
    value_col = _grid_value_column(grid)
    piv = grid.pivot_table(index=LAMBDA_COL, columns=PMR_COL, values=value_col).to_numpy()
    li = int(np.clip(np.searchsorted(ls, lam) - 1, 0, len(ls) - 2))
    pi = int(np.clip(np.searchsorted(ps, pmr) - 1, 0, len(ps) - 2))
    l0, l1, p0, p1 = ls[li], ls[li + 1], ps[pi], ps[pi + 1]
    tl = 0.0 if l1 == l0 else (lam - l0) / (l1 - l0)
    tp = 0.0 if p1 == p0 else (pmr - p0) / (p1 - p0)
    return float(
        piv[li, pi] * (1 - tl) * (1 - tp) + piv[li + 1, pi] * tl * (1 - tp)
        + piv[li, pi + 1] * (1 - tl) * tp + piv[li + 1, pi + 1] * tl * tp
    )


def compare_block(
    beta: np.ndarray, offpath: Path, grid: pd.DataFrame | None,
) -> dict[str, Any]:
    """Cross-check against the points measured by a one-factor study."""
    data = json.loads(offpath.read_text(encoding="utf-8"))
    pts = data["by_design_point"]["points"]
    factor = data["by_design_point"]["factor"]
    rows = []
    for p in pts:
        if factor == LAMBDA_COL:
            lam, pmr = float(p[LAMBDA_COL]), 0.5
        else:
            lam, pmr = 6.0, float(p[PMR_COL])
        measured = float(p["mean"])
        surf = float(beta[0] + beta[1] * lam + beta[2] * pmr + beta[3] * lam * pmr)
        row = {LAMBDA_COL: lam, PMR_COL: pmr, "measured": measured,
               "surface": surf, "residual_surface": surf - measured}
        if grid is not None:
            g = _bilinear_grid_lookup(grid, lam, pmr)
            row["surrogate"] = g
            row["residual_surrogate"] = g - measured
        rows.append(row)

    def _rmse(key: str) -> float | None:
        v = np.array([r[key] for r in rows if key in r], dtype=float)
        return _f(np.sqrt(float((v ** 2).mean()))) if v.size else None

    out = {
        "source": str(offpath),
        "factor_varied_there": factor,
        "n_points": len(rows),
        "line": ("p_mr = 0.5" if factor == LAMBDA_COL else "lambda = 6.0"),
        "rmse_surface_vs_measured": _rmse("residual_surface"),
        "max_abs_residual_surface": _f(max(abs(r["residual_surface"]) for r in rows)),
        "points": rows,
        "note": (
            "os pontos medidos vem de um estudo com banco de seeds DISJUNTO, "
            "entao a concordancia e replicacao independente, nao aritmetica "
            "circular (docs/experimental_campaign.md sec.4.1/4.2)"
        ),
    }
    if grid is not None:
        out["rmse_surrogate_vs_measured"] = _rmse("residual_surrogate")
        out["max_abs_residual_surrogate"] = _f(
            max(abs(r["residual_surrogate"]) for r in rows)
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Surface with the lambda x mix interaction and the frontier of the margin."
    )
    ap.add_argument("--datafarm", type=Path, required=True)
    ap.add_argument("--design", type=Path, default=None,
                    help="design.parquet, to echo the design seed of its sidecar")
    ap.add_argument("--response", default="MOS-1")
    ap.add_argument("--aor", default="ALL")
    ap.add_argument("--bootstrap-seed", type=int, default=20260808)
    ap.add_argument("--n-resamples", type=int, default=9999)
    ap.add_argument("--margin", type=float, default=AC3_MARGIN,
                    help="Decision margin for the frontier (default 0.10)")
    ap.add_argument("--compare", type=Path, action="append", default=None,
                    help="offpath_*.json of a one-factor study (repeatable)")
    ap.add_argument("--grid", type=Path, default=None,
                    help="grid_prediction.csv of the surrogate, for the 2-D cross-check")
    ap.add_argument("--output", "-o", type=Path, required=True)
    args = ap.parse_args(argv)

    if not args.datafarm.exists():
        print(f"[ERROR] datafarm not found: {args.datafarm}", file=sys.stderr)
        return 1
    df = pd.read_parquet(args.datafarm)
    n_pts = int(df["design_point_id"].nunique())
    if n_pts < 4:
        print(f"[ERROR] a surface with interaction needs >= 4 design points "
              f"(the datafarm has {n_pts})", file=sys.stderr)
        return 1
    for col in (LAMBDA_COL, PMR_COL):
        if float(df[col].std(ddof=0)) <= 0.0:
            print(f"[ERROR] factor {col} has zero variance in this datafarm: the "
                  "interaction coefficient has no degree of freedom (cross design; "
                  "see docs/analysis_layer.md, section 3)",
                  file=sys.stderr)
            return 1

    pts = point_table(df, args.response, args.aor)
    bil = fit_surface(pts, seed=args.bootstrap_seed, n_resamples=args.n_resamples,
                      margin=args.margin, quadratic=False)
    quad = fit_surface(pts, seed=args.bootstrap_seed, n_resamples=args.n_resamples,
                       margin=args.margin, quadratic=True)

    beta = np.array([bil["coefficients"][k]["estimate"] for k in
                     ("intercept", "lambda", "p_mr", "lambda_x_p_mr")], dtype=float)

    grid = pd.read_csv(args.grid) if args.grid and args.grid.exists() else None
    comparisons = [compare_block(beta, p, grid) for p in (args.compare or [])
                   if p.exists()]

    report: dict[str, Any] = {
        "inputs": {
            "datafarm": str(args.datafarm),
            "response": args.response,
            "aor": args.aor,
            "bootstrap_seed": args.bootstrap_seed,
            "n_resamples": args.n_resamples,
            "margin": args.margin,
            "grid": None if grid is None else str(args.grid),
        },
        "realized_ranges": {
            LAMBDA_COL: [_f(pts[LAMBDA_COL].min()), _f(pts[LAMBDA_COL].max())],
            PMR_COL: [_f(pts[PMR_COL].min()), _f(pts[PMR_COL].max())],
        },
        "headline_model": "bilinear",
        "headline_model_rationale": (
            "o piloto ME-1 mediu b3 = -0.014776 num ajuste BILINEAR; comparar com "
            "ele exige o mesmo modelo. O quadratico entra como ROBUSTEZ (o b3 e "
            "dependente de modelo: no ME-1 vale -0.01478 bilinear e -0.01023 "
            "quadratico) -- declarado antes de rodar"
        ),
        "surface_bilinear": bil,
        "surface_quadratic": quad,
        "design_points": [
            {"design_point_id": int(i), LAMBDA_COL: _f(r[LAMBDA_COL]),
             PMR_COL: _f(r[PMR_COL]), "mean": _f(r["mean"]), "std": _f(r["std"]),
             "n": int(r["count"])}
            for i, r in pts.iterrows()
        ],
        "cross_checks": comparisons,
    }
    if args.design is not None:
        meta = args.design.with_name(args.design.name + ".meta.json")
        if meta.exists():
            report["design"] = json.loads(meta.read_text(encoding="utf-8"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")

    # ---- console (ASCII) ----
    c = bil["coefficients"]
    print(f"=== surface model ({args.response}, aor={args.aor}) -> {args.output} ===")
    print(f"design points: {bil['n_design_points']} | R2 bilinear "
          f"{bil['r_squared']:.4f} (additive {bil['r_squared_additive']:.4f})")
    print(f"realized range: lambda [{report['realized_ranges'][LAMBDA_COL][0]:.3f}, "
          f"{report['realized_ranges'][LAMBDA_COL][1]:.3f}]  p_mr "
          f"[{report['realized_ranges'][PMR_COL][0]:.4f}, "
          f"{report['realized_ranges'][PMR_COL][1]:.4f}]")
    for name in ("intercept", "lambda", "p_mr", "lambda_x_p_mr"):
        b = c[name]
        star = "" if b["ci_low"] is None else ("  *excludes zero" if b["excludes_zero"] else "  (crosses zero)")
        print(f"  {name:<16} {b['estimate']:+.6f}  CI95 [{b['ci_low']:+.6f}, "
              f"{b['ci_high']:+.6f}]  hw {b['half_width']:.6f}{star}")
    ft = bil["additivity_F_test"]
    print(f"  additivity: F(1,{ft['df_den']}) = {ft['F']:.4f}, p = {ft['p_value']:.6f} -> "
          f"{'additivity REJECTED' if ft['rejects_additivity'] else 'not rejected'}")
    print("  slope in the mix by lambda:")
    for k, v in bil["mix_slope_by_lambda"].items():
        print(f"    lambda={k:>4}: {v['estimate']:+.4f} CI95 [{v['ci_low']:+.4f}, "
              f"{v['ci_high']:+.4f}]")
    print(f"  frontier p* (Delta = {args.margin:+.2f}) by lambda:")
    lo_p, hi_p = report["realized_ranges"][PMR_COL]
    for k, v in bil["frontier_p_mr_at_margin"].items():
        est = v["estimate"]
        inside = (est is not None and lo_p <= est <= hi_p)
        tag = "" if inside else "   [OUTSIDE the sampled range: not supported]"
        print(f"    lambda={k:>4}: p* = {est:+.4f} CI95 [{v['ci_low']:+.4f}, "
              f"{v['ci_high']:+.4f}]{tag}")
    print(f"  robustness (quadratic): b3 = "
          f"{quad['coefficients']['lambda_x_p_mr']['estimate']:+.6f} CI95 "
          f"[{quad['coefficients']['lambda_x_p_mr']['ci_low']:+.6f}, "
          f"{quad['coefficients']['lambda_x_p_mr']['ci_high']:+.6f}]")
    for cc in comparisons:
        line = f"cross-check vs {cc['line']} ({cc['n_points']} measured points, disjoint seeds)"
        print(f"{line}: RMSE surface {cc['rmse_surface_vs_measured']:.5f}", end="")
        if "rmse_surrogate_vs_measured" in cc:
            print(f" | RMSE surrogate {cc['rmse_surrogate_vs_measured']:.5f}")
        else:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
