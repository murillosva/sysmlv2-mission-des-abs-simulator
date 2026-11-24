"""EDA + report assembly/persistence.

``build_report`` is **pure** (assembles the ``report.json`` dict from ``stats``/
``features``); ``write_report`` is the only place with I/O. ``matplotlib`` is
imported **lazily** inside the ``_render_*`` helpers (viz/hud.py pattern) —
importing this module does not force a GUI backend.

``report.json`` carries its own ``field_version``, independent of
``replanning_signal`` / ``design.parquet`` / ``datafarm.parquet``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cuas_sim.analysis.features import feature_scores, pareto_front
from cuas_sim.analysis.stats import (
    CAUSALITY_CAVEAT,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_RESAMPLES,
    SCHOBER_CUTOFFS,
    analyze_relationship,
    bin_factor_levels,
    correlation_strength,
    detect_outliers,
    fit_distribution,
    kgroup_hypothesis_test,
    paired_hypothesis_test,
)

FIELD_VERSION = "1.0"

# Consolidated-report contract version. Independent of the per-response
# ``report.json``; the ``study.kind == "consolidated"`` tag distinguishes
# the two artifacts.
CONSOLIDATED_FIELD_VERSION = "1.0"

_FACTORS = ("lambda_per_h", "uas_archetype_p_mr")
_COST_METRIC = "rockets_per_engagement"

# Versioned thresholds for the consolidated-report insights and decision index.
# Literature-anchored; recorded verbatim in report.json for auditability.
SURROGATE_Q2_MIN = 0.5  # Golbraikh & Tropsha (2002): acceptable predictive model
DESIGN_POINTS_PER_FACTOR = 10  # Loeppky, Sacks & Welch (2009): n ~ 10 x factors
MIN_PAIRS_FOR_POWER = 34  # Cohen (1988): paired t, d=0.5, power 0.80, alpha 0.05 two-sided

# ASCII-only insight caveats. PT accents are cp1252-safe but
# the known-answer oracles pin ASCII tokens, so the templates stay accent-free.
# CAUSALITY_CAVEAT is defined in stats.py (used by analyze_relationship) and re-exported.
NO_GENERALIZE_CAVEAT = "nao generalizar alem do espaco amostrado (faixas dos fatores)"

__all__ = [
    "CAUSALITY_CAVEAT",
    "CONSOLIDATED_FIELD_VERSION",
    "DESIGN_POINTS_PER_FACTOR",
    "FIELD_VERSION",
    "MIN_PAIRS_FOR_POWER",
    "NO_GENERALIZE_CAVEAT",
    "SURROGATE_Q2_MIN",
    "build_consolidated_report",
    "build_decision_index",
    "build_report",
    "build_validity_block",
    "correlation_insight",
    "dominant_factor_insight",
    "evidence_sufficiency",
    "per_design_point",
    "render_figures_png",
    "render_md",
    "significance_insight",
    "write_consolidated_report",
    "write_report",
    "write_surrogate",
]


def _paired_arrays(
    datafarm: pd.DataFrame, response: str, aor: str
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned (baseline, modernized) arrays matched by (design_point_id, rep, seed)."""
    sub = datafarm[(datafarm["metric"] == response) & (datafarm["aor"] == aor)]
    key = ["design_point_id", "rep", "seed"]
    bsl = sub[sub["config"] == "baseline"].set_index(key)["value"]
    mod = sub[sub["config"] == "modernized"].set_index(key)["value"]
    common = bsl.index.intersection(mod.index).sort_values()  # deterministic order
    return bsl.loc[common].to_numpy(dtype=float), mod.loc[common].to_numpy(dtype=float)


def per_design_point(datafarm: pd.DataFrame, response: str, aor: str) -> pd.DataFrame:
    """One row per design point: factors + mean paired delta (``gain``) + mean cost.

    Shared training table for the analysis (features/Pareto/EDA) and the prediction
    (surrogate): a single owner guarantees the surrogate's feature importances
    corroborate the ``feature_scores`` on the *same* per-design-point table.
    """
    delta = (
        datafarm[(datafarm["metric"] == response) & (datafarm["aor"] == aor)
                 & (datafarm["config"] == "delta")]
        .groupby("design_point_id")["value"].mean().rename("gain")
    )
    cost = (
        datafarm[(datafarm["metric"] == _COST_METRIC) & (datafarm["config"] == "modernized")]
        .groupby("design_point_id")["value"].mean().rename("cost")
    )
    factors = datafarm.groupby("design_point_id")[list(_FACTORS)].first()
    out = factors.join(delta, how="inner")
    out = out.join(cost, how="left")
    return out.reset_index().sort_values("design_point_id").reset_index(drop=True)


def _hypothesis_block(
    datafarm: pd.DataFrame, response: str, aor: str, *,
    alpha: float, bootstrap_seed: int, n_resamples: int,
) -> dict[str, Any]:
    bsl, mod = _paired_arrays(datafarm, response, aor)
    res = paired_hypothesis_test(
        bsl.tolist(), mod.tolist(), alpha=alpha,
        bootstrap_seed=bootstrap_seed, n_resamples=n_resamples,
    )
    boot = res.bootstrap
    return {
        "response": response,
        "aor": aor,
        "n_pairs": int(bsl.size),
        "pre_checks": [
            {"name": pc.name, "statistic": pc.statistic, "p_value": pc.p_value,
             "passed": pc.passed}
            for pc in res.pre_checks
        ],
        "test": res.test,
        "statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "effect": res.effect,
        "ci": None if boot is None else {
            "low": boot.ci_low, "high": boot.ci_high,
            "method": boot.ci_method, "n_resamples": boot.n_resamples,
        },
        "bootstrap_test": None if boot is None else {
            "p_value": boot.p_value, "n_resamples": boot.n_resamples,
            "bootstrap_seed": boot.bootstrap_seed,
        },
    }


def _feature_block(per_dp: pd.DataFrame, *, response: str, seed: int) -> dict[str, Any]:
    factors = [f for f in _FACTORS if per_dp[f].std(ddof=0) > 0]
    if len(per_dp) < 2 or not factors:
        return {"response": response, "scoring": "f_regression", "ranking": [],
                "note": "insufficient design points / factor variance"}
    scores = feature_scores(per_dp, response="gain", factors=factors,
                            scoring="f_regression", seed=seed)
    return {
        "response": response,
        "scoring": "f_regression",
        "ranking": [{"factor": s.factor, "score": s.score} for s in scores],
    }


def _pareto_block(per_dp: pd.DataFrame) -> dict[str, Any]:
    points = per_dp.dropna(subset=["gain", "cost"])
    if points.empty:
        return {"objectives": {"maximize": ["gain"], "minimize": ["cost"]}, "points": []}
    front = pareto_front(points, maximize=["gain"], minimize=["cost"])
    return {
        "objectives": {"maximize": ["gain"], "minimize": ["cost"]},
        "points": [
            {"design_point_id": int(r["design_point_id"]),
             "gain": float(r["gain"]), "cost": float(r["cost"])}
            for _, r in front.iterrows()
        ],
    }


def _eda_block(per_dp: pd.DataFrame, *, response: str) -> dict[str, Any]:
    gain = per_dp["gain"].to_numpy(dtype=float)
    correlation: dict[str, float | None] = {}
    for f in _FACTORS:
        col = per_dp[f].to_numpy(dtype=float)
        if col.std() == 0 or gain.std() == 0 or len(gain) < 2:
            correlation[f] = None
        else:
            correlation[f] = float(np.corrcoef(col, gain)[0, 1])
    outliers = detect_outliers(gain, method="IQR")
    return {
        "response": response,
        "summary": {
            "count": int(gain.size),
            "mean": float(gain.mean()),
            "std": float(gain.std(ddof=1)) if gain.size > 1 else 0.0,
            "min": float(gain.min()),
            "max": float(gain.max()),
        },
        "correlation": correlation,
        "outlier_count": len(outliers.indices),
        "figures": None,
    }


def _factor_ranges(datafarm: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for f in _FACTORS:
        if f in datafarm.columns:
            col = datafarm[f].to_numpy(dtype=float)
            ranges[f] = {"min": float(col.min()), "max": float(col.max())}
    return ranges


def build_report(
    datafarm: pd.DataFrame,
    *,
    response: str = "MOS-1",
    aor: str = "ALL",
    alpha: float = 0.05,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    study_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``report.json`` dict from the datafarm.

    Pure: read-only over ``datafarm``; deterministic given the
    same ``datafarm`` + ``bootstrap_seed``.
    """
    per_dp = per_design_point(datafarm, response, aor)
    study: dict[str, Any] = {
        "response": response,
        "aor": aor,
        "alpha": alpha,
        "bootstrap_seed": bootstrap_seed,
        "n_resamples": n_resamples,
        "n_design_points": int(datafarm["design_point_id"].nunique()),
        "factors": _factor_ranges(datafarm),
    }
    if study_meta:
        study.update(study_meta)

    return {
        "field_version": FIELD_VERSION,
        "study": study,
        "hypothesis_tests": [
            _hypothesis_block(datafarm, response, aor, alpha=alpha,
                              bootstrap_seed=bootstrap_seed, n_resamples=n_resamples)
        ],
        "feature_scores": [_feature_block(per_dp, response=response, seed=bootstrap_seed)],
        "pareto_front": _pareto_block(per_dp),
        "eda": _eda_block(per_dp, response=response),
    }


def _report_md(report: dict[str, Any]) -> str:
    """ASCII-only Markdown mirror of report.json."""
    s = report["study"]
    lines = [
        "# Analysis report",
        "",
        f"- field_version: {report['field_version']}",
        f"- response: {s['response']} | aor: {s['aor']} | alpha: {s['alpha']}",
        f"- design_points: {s['n_design_points']} | bootstrap_seed: {s['bootstrap_seed']}",
        "",
        "## Hypothesis tests",
    ]
    for h in report["hypothesis_tests"]:
        ci = h["ci"]
        ci_txt = "-" if ci is None else f"[{ci['low']:.4f}, {ci['high']:.4f}] ({ci['method']})"
        bt = h["bootstrap_test"]
        bp = "-" if bt is None else f"{bt['p_value']:.4f}"
        lines += [
            f"- {h['response']} ({h['aor']}, n={h['n_pairs']}): test={h['test']} "
            f"p={h['p_value']:.4g} reject_h0={h['reject_h0']}",
            f"  effect={h['effect']:.4f} CI={ci_txt} bootstrap_p={bp}",
        ]
    lines += ["", "## Feature scores"]
    for fb in report["feature_scores"]:
        rank = " > ".join(f"{r['factor']}({r['score']:.3f})" for r in fb["ranking"]) or "-"
        lines.append(f"- {fb['response']} [{fb['scoring']}]: {rank}")
    lines += ["", "## Pareto front (gain x cost)"]
    for p in report["pareto_front"]["points"]:
        lines.append(
            f"- dp {p['design_point_id']}: gain={p['gain']:.4f} cost={p['cost']:.4f}"
        )
    eda = report["eda"]
    lines += [
        "",
        "## EDA",
        f"- gain: n={eda['summary']['count']} mean={eda['summary']['mean']:.4f} "
        f"std={eda['summary']['std']:.4f} outliers={eda['outlier_count']}",
    ]
    return "\n".join(lines) + "\n"


def _render_figures(report: dict[str, Any], out_dir: Path) -> None:  # pragma: no cover
    """Render EDA/feature/Pareto figures (lazy matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fb = report["feature_scores"][0]["ranking"]
    if fb:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([r["factor"] for r in fb], [r["score"] for r in fb])
        ax.set_title("Feature importance")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_score.png")
        plt.close(fig)

    pts = report["pareto_front"]["points"]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([p["cost"] for p in pts], [p["gain"] for p in pts])
        ax.set_xlabel("cost")
        ax.set_ylabel("gain")
        ax.set_title("Pareto front")
        fig.tight_layout()
        fig.savefig(out_dir / "pareto_front.png")
        plt.close(fig)


def write_report(
    report: dict[str, Any],
    out_dir: Path,
    *,
    with_figures: bool = True,
) -> None:
    """Persist ``report.json`` (sort_keys, no timestamp) + ``report.md`` + figures.

    ``report.json`` is byte-deterministic; figures (PNG) are written
    via lazily-imported matplotlib unless ``with_figures`` is False.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report_md(report), encoding="utf-8")
    if with_figures:
        _render_figures(report, out_dir)


def write_surrogate(
    report: dict[str, Any],
    out_dir: Path,
    grid_df: pd.DataFrame,
    *,
    with_figures: bool = True,
) -> None:
    """Persist ``surrogate.json`` + ``grid_prediction.csv`` (+ surface figure).

    ``surrogate.json`` is byte-deterministic; the dense grid goes to
    ``grid_prediction.csv`` and the optional lambda x mix surface figure via lazily
    imported matplotlib unless ``with_figures`` is False. This is the
    I/O sink of the prediction, so ``prediction.py`` stays pure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surrogate.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
    )
    grid_df.to_csv(out_dir / "grid_prediction.csv", index=False)
    if with_figures:
        _render_surface(grid_df, out_dir)


# ─── Consolidated report layer ────────────────────────────────
# Pure rule helpers + orchestrator + writers. Read-only over the datafarm:
# reuses the analysis and prediction blocks; never recomputes MOS/MOE/MOP.


def _json_safe(obj: Any) -> Any:
    """Coerce numpy scalars -> python and non-finite floats -> None."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _ci_excludes_zero(lo: float | None, hi: float | None) -> bool:
    if lo is None or hi is None:
        return False
    return (lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)


def _significance(
    p_value: float | None, lo: float | None, hi: float | None, alpha: float
) -> tuple[bool, bool]:
    """Return (significant, borderline). NHST reject AND CI-excludes-zero agree."""
    reject = p_value is not None and float(p_value) <= alpha
    excludes = _ci_excludes_zero(lo, hi)
    return (reject and excludes), (reject != excludes)


def significance_insight(
    response: str, block: dict[str, Any], *, alpha: float = 0.05
) -> dict[str, Any]:
    """NHST significance insight. Pure; ASCII (cp1252-safe).

    ``block`` = ``{effect, ci_low, ci_high, p_value, test}``. Phrasing follows
    Bussab & Morettin (2010): reject / do-not-reject H0 at ``alpha``. When not
    significant the insight adds the "evidencia insuficiente" clause.
    """
    effect = float(block["effect"])
    lo = block.get("ci_low")
    hi = block.get("ci_high")
    p = block.get("p_value")
    test = block.get("test") or "teste"
    significant, borderline = _significance(p, lo, hi, alpha)
    lo_s = "-" if lo is None else f"{float(lo):.2f}"
    hi_s = "-" if hi is None else f"{float(hi):.2f}"
    p_s = "-" if p is None else f"{float(p):.4g}"
    direction = "modernized > baseline" if effect >= 0.0 else "modernized < baseline"
    ci_rel = "exclui zero" if _ci_excludes_zero(lo, hi) else "cruza zero"
    if significant:
        text = (
            f"{response}: efeito {effect:+.2f} significativo (alpha={alpha:g}). "
            f"Rejeita-se a hipotese nula de igualdade entre as configuracoes "
            f"({test}, p={p_s}; IC 95% BCa [{lo_s}, {hi_s}] {ci_rel}), "
            f"indicando {direction}."
        )
    else:
        text = (
            f"{response}: efeito {effect:+.2f} nao significativo (alpha={alpha:g}). "
            f"Nao se rejeita a hipotese nula ({test}, p={p_s}; "
            f"IC 95% BCa [{lo_s}, {hi_s}] {ci_rel}); "
            f"evidencia insuficiente; rodar mais replicas/pontos."
        )
    return {
        "response": response, "effect": effect,
        "significant": bool(significant), "borderline": bool(borderline), "text": text,
    }


def correlation_insight(factor: str, r: float | None) -> dict[str, Any]:
    """Correlation-strength insight + causality caveat. Pure; ASCII."""
    strength = correlation_strength(r)
    if r is None:
        rv: float | None = None
        text = f"{factor}: correlacao indeterminada (variancia nula); {CAUSALITY_CAVEAT}."
    else:
        rv = float(r)
        text = f"{factor}: correlacao {strength} (r={rv:+.2f}); {CAUSALITY_CAVEAT}."
    return {"factor": factor, "r": rv, "strength": strength,
            "caveat": CAUSALITY_CAVEAT, "text": text}


def dominant_factor_insight(ranking: list[dict[str, Any]]) -> dict[str, Any]:
    """Dominant-factor insight from feature scores. Pure; no cutoff ratio."""
    if not ranking:
        return {"factor": None, "score": None, "text": "sem fatores com variancia"}
    top = ranking[0]
    others = ", ".join(f"{r['factor']} {float(r['score']):.2f}" for r in ranking[1:])
    tail = f" vs {others})" if others else ")"
    text = f"fator dominante: {top['factor']} (feature score {float(top['score']):.2f}{tail}"
    return {"factor": top["factor"], "score": float(top["score"]), "text": text}


def evidence_sufficiency(
    *,
    p_value: float | None,
    ci_low: float | None,
    ci_high: float | None,
    alpha: float,
    n_design_points: int,
    n_factors: int,
    n_pairs: int,
    surrogate_q2: float | None,
) -> dict[str, Any]:
    """Evidence-sufficiency gate. Pure; literature-anchored."""
    reasons: list[str] = []
    significant, _ = _significance(p_value, ci_low, ci_high, alpha)
    if not significant:
        reasons.append("nao-significancia (p > alpha ou IC cruza zero; Bussab & Morettin 2010)")
    floor = DESIGN_POINTS_PER_FACTOR * max(n_factors, 1)
    if n_design_points < floor:
        reasons.append(
            f"pontos de projeto insuficientes (N={n_design_points} < {floor}; Loeppky 2009)"
        )
    if n_pairs < MIN_PAIRS_FOR_POWER:
        reasons.append(
            f"pares insuficientes (n={n_pairs} < {MIN_PAIRS_FOR_POWER}; Cohen 1988)"
        )
    if surrogate_q2 is not None and float(surrogate_q2) < SURROGATE_Q2_MIN:
        reasons.append(
            f"surrogate nao-confiavel (Q2={float(surrogate_q2):.2f} < {SURROGATE_Q2_MIN}; "
            "Golbraikh & Tropsha 2002)"
        )
    sufficient = not reasons
    text = (
        "evidencia suficiente para a conclusao no espaco amostrado"
        if sufficient
        else "evidencia insuficiente; rodar mais replicas/pontos (" + "; ".join(reasons) + ")"
    )
    return {"sufficient": sufficient, "reasons": reasons, "text": text}


def build_validity_block(
    *,
    factor_ranges: dict[str, dict[str, float]],
    n_design_points: int,
    n_reps: int | None,
    seeds: dict[str, int | None],
    max_arrivals: int | None,
) -> dict[str, Any]:
    """Validity/scope section. Pure; declares the sampled space + caveats."""
    return {
        "factor_ranges": factor_ranges,
        "n_design_points": int(n_design_points),
        "n_reps": None if n_reps is None else int(n_reps),
        "seeds": dict(seeds),
        "max_arrivals": max_arrivals,
        "byte_determinism_note": (
            "byte-identidade das 3 camadas por-ambiente (mesmas versoes de matplotlib/jinja2)"
        ),
        "caveats": [NO_GENERALIZE_CAVEAT, CAUSALITY_CAVEAT],
    }


def build_decision_index(
    master_table: list[dict[str, Any]],
    *,
    primary_response: str = "MOS-1",
    cost_by_aor: dict[str, dict[str, float]] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Decision index: rank AoRs by significant primary effect. Pure.

    Prioritizes NEUTRALIZATION (primary paired effect = dMOS-1), gated by significance:
    significant scenarios first (effect desc), non-significant demoted and flagged.
    Cost is an ANNOTATION (rockets/eng + leaks), NEVER weighted/normalized — the
    index is a reporting composite, not a paper metric.
    """
    cost_by_aor = cost_by_aor or {}
    rows = [r for r in master_table if r.get("response") == primary_response]
    entries: list[dict[str, Any]] = []
    for r in rows:
        significant, borderline = _significance(
            r.get("p_value"), r.get("ci_low"), r.get("ci_high"), alpha
        )
        aor_key = str(r.get("aor"))
        entries.append({
            "aor": r.get("aor"),
            "effect": float(r["effect"]),
            "significant": bool(significant),
            "borderline": bool(borderline),
            "cost": cost_by_aor.get(aor_key, {}),
            "note": "" if significant else
                    "evidencia insuficiente (nao-significativo: IC cruza zero ou p > alpha)",
        })
    sig = sorted((e for e in entries if e["significant"]),
                 key=lambda e: e["effect"], reverse=True)
    non_sig = sorted((e for e in entries if not e["significant"]),
                     key=lambda e: e["effect"], reverse=True)
    ranked = sig + non_sig
    for i, e in enumerate(ranked, start=1):
        e["rank"] = i
    return {
        "label": "reporting composite (nao metrica do paper)",
        "primary_response": primary_response,
        "criterion": (
            "rank AoRs by significant paired dMOS-1 (prioriza neutralizacao); "
            "custo anotado, nao ponderado"
        ),
        "ranking": ranked,
    }


# ─── internal contract builders ─────────────────────────────────────────────


def _master_row(
    datafarm: pd.DataFrame, response: str, aor: str, *,
    alpha: float, bootstrap_seed: int, n_resamples: int,
) -> dict[str, Any] | None:
    """One master-table row (paired for MOS-1/MOE-*; absolute for MOS-2)."""
    if response == "MOS-2":  # already comparative -> absolute
        sub = datafarm[(datafarm["metric"] == "MOS-2") & (datafarm["aor"] == aor)]
        if sub.empty:
            return None
        return {
            "response": response, "config": "absolute", "aor": aor,
            "effect": float(sub["value"].mean()), "ci_low": None, "ci_high": None,
            "ci_method": None, "p_value": None, "test": "absolute",
            "significant": None, "borderline": False, "n_pairs": len(sub),
        }
    bsl, mod = _paired_arrays(datafarm, response, aor)
    if bsl.size == 0:
        return None
    res = paired_hypothesis_test(
        bsl.tolist(), mod.tolist(), alpha=alpha,
        bootstrap_seed=bootstrap_seed, n_resamples=n_resamples,
    )
    boot = res.bootstrap
    lo = boot.ci_low if boot else None
    hi = boot.ci_high if boot else None
    significant, borderline = _significance(res.p_value, lo, hi, alpha)
    return {
        "response": response, "config": "modernized-baseline", "aor": aor,
        "effect": res.effect, "ci_low": lo, "ci_high": hi,
        "ci_method": boot.ci_method if boot else None,
        "p_value": res.p_value, "test": res.test,
        "significant": bool(significant), "borderline": bool(borderline),
        "n_pairs": int(bsl.size),
    }


def _response_block(
    datafarm: pd.DataFrame, per_dp: pd.DataFrame, response: str, *,
    alpha: float, bootstrap_seed: int, n_resamples: int,
) -> dict[str, Any]:
    """Per-response block: hypothesis + features + Pareto + EDA/hist + fit + insight."""
    fblock = _feature_block(per_dp, response=response, seed=bootstrap_seed)
    ranking = fblock["ranking"]
    pareto = _pareto_block(per_dp)
    eda = _eda_block(per_dp, response=response)
    gain = per_dp["gain"].to_numpy(dtype=float)
    if gain.size >= 1:
        counts, edges = np.histogram(gain, bins=min(10, max(1, gain.size)))
        eda["histogram"] = {"counts": [int(c) for c in counts],
                            "edges": [float(e) for e in edges]}
    else:
        eda["histogram"] = {"counts": [], "edges": []}
    fits = fit_distribution(gain.tolist()) if gain.size >= 2 else []
    best = fits[0] if fits else None
    fit = None if best is None else {
        "name": best.name, "ks_statistic": best.ks_statistic, "p_value": best.p_value,
    }
    hblock = _hypothesis_block(
        datafarm, response, "ALL", alpha=alpha,
        bootstrap_seed=bootstrap_seed, n_resamples=n_resamples,
    )
    ci = hblock["ci"] or {}
    insight = significance_insight(
        response,
        {"effect": hblock["effect"], "ci_low": ci.get("low"), "ci_high": ci.get("high"),
         "p_value": hblock["p_value"], "test": hblock["test"]},
        alpha=alpha,
    )
    return {
        "response": response,
        "hypothesis": hblock,
        "feature_scores": ranking,
        "dominant_factor": dominant_factor_insight(ranking),
        "pareto_front": pareto,
        "eda": eda,
        "fit": fit,
        "outlier_count": eda["outlier_count"],
        "insight": insight,
    }


def _relationship_dict(response: str, factor: str, rel: Any) -> dict[str, Any]:
    ins = correlation_insight(factor, rel.pearson)
    return {
        "response": response, "factor": factor, "col_x": rel.col_x, "col_y": rel.col_y,
        "n": rel.n, "pearson": rel.pearson, "spearman": rel.spearman,
        "kendall": rel.kendall, "slope": rel.slope, "intercept": rel.intercept,
        "r2": rel.r2, "mse": rel.mse, "strength": rel.strength,
        "caveat": rel.caveat, "insight": ins["text"],
    }


def _anova_by_levels_block(
    per_dp: pd.DataFrame, response: str, factor: str, *, n_levels: int = 3,
) -> dict[str, Any]:
    """k-group test over factor levels. Reuses ``kgroup_hypothesis_test``."""
    base = {"response": response, "factor": factor, "n_levels": n_levels}
    if per_dp[factor].nunique() < n_levels or len(per_dp) < n_levels:
        return {**base, "test": None, "p_value": None,
                "note": "insufficient distinct levels"}
    levels = bin_factor_levels(per_dp, factor, n_levels=n_levels, method="quantile")
    lv = np.asarray(levels)
    order = ["low", "mid", "high"] if n_levels == 3 else sorted(set(levels))
    gain = per_dp["gain"].to_numpy(dtype=float)
    groups = [gain[lv == lab].tolist() for lab in order if bool((lv == lab).any())]
    if len(groups) < 2:
        return {**base, "test": None, "p_value": None,
                "note": "insufficient non-empty levels"}
    res = kgroup_hypothesis_test(groups)
    return {
        **base, "n_levels": len(groups), "test": res.test, "statistic": res.statistic,
        "p_value": res.p_value, "reject_h0": res.reject_h0,
        "posthoc": res.posthoc if isinstance(res.posthoc, dict) else None,
    }


def _cost_by_aor(datafarm: pd.DataFrame, aors: Sequence[str]) -> dict[str, dict[str, float]]:
    """Cost annotation (modernized rockets/eng + leak totals). Per-config, not per-AoR."""
    rk = datafarm[(datafarm["metric"] == _COST_METRIC) & (datafarm["config"] == "modernized")]
    rockets = float(rk["value"].mean()) if not rk.empty else None
    leaks_df = datafarm[
        datafarm["metric"].astype(str).str.startswith("leak_")
        & (datafarm["config"] == "modernized")
    ]
    leaks = float(leaks_df["value"].sum()) if not leaks_df.empty else None
    entry: dict[str, float] = {}
    if rockets is not None:
        entry["rockets_per_engagement"] = rockets
    if leaks is not None:
        entry["leaks"] = leaks
    return {aor: dict(entry) for aor in aors}


def build_consolidated_report(
    datafarm: pd.DataFrame,
    *,
    responses: Sequence[str],
    aors: Sequence[str],
    alpha: float = 0.05,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    model_seed: int = 20260701,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    grid_resolution: int = 25,
    study_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Orchestrate stats/features/prediction into the consolidated contract.

    Pure: read-only over ``datafarm``; deterministic given the same
    ``datafarm`` + seeds + parameters. Reuses the analysis and prediction blocks so
    the consolidated numbers equal the per-response artifacts.
    """
    study_meta = study_meta or {}
    metrics_present = set(datafarm["metric"].astype(str).unique())
    aors_present = set(datafarm["aor"].astype(str).unique())
    factors = [
        f for f in _FACTORS
        if f in datafarm.columns and float(datafarm[f].std(ddof=0)) > 0.0
    ]
    n_factors = len(factors)
    n_design_points = int(datafarm["design_point_id"].nunique())
    factor_ranges = _factor_ranges(datafarm)

    aors_eff: list[str] = []
    for a in aors:
        if (a == "ALL" or a in aors_present) and a not in aors_eff:
            aors_eff.append(a)
    if not aors_eff:
        aors_eff = ["ALL"]

    responses_eff = [r for r in responses if r in metrics_present]
    skipped = [r for r in responses if r not in metrics_present]

    master_table: list[dict[str, Any]] = []
    responses_blocks: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    anova_by_levels: list[dict[str, Any]] = []

    for response in responses_eff:
        per_dp = per_design_point(datafarm, response, "ALL")
        for aor in aors_eff:
            row = _master_row(
                datafarm, response, aor, alpha=alpha,
                bootstrap_seed=bootstrap_seed, n_resamples=n_resamples,
            )
            if row is not None:
                master_table.append(row)
        if response != "MOS-2":
            responses_blocks.append(_response_block(
                datafarm, per_dp, response, alpha=alpha,
                bootstrap_seed=bootstrap_seed, n_resamples=n_resamples,
            ))
            for f in factors:
                relationships.append(
                    _relationship_dict(response, f, analyze_relationship(per_dp, f, "gain"))
                )
            for f in factors:
                anova_by_levels.append(_anova_by_levels_block(per_dp, response, f))

    primary = responses_eff[0] if responses_eff else (responses[0] if responses else "MOS-1")

    surrogate_block: dict[str, Any] | None = None
    surrogate_q2: float | None = None
    if responses_eff and primary != "MOS-2" and n_design_points >= 2 and factors:
        from cuas_sim.analysis.prediction import build_surrogate_report

        srep, _grid = build_surrogate_report(
            datafarm, response=primary, aor="ALL", model="rf",
            model_seed=model_seed, grid_resolution=grid_resolution,
        )
        surrogate_q2 = srep["cv"]["mean"].get("r2")
        surrogate_block = {
            "response": primary, "model": srep["model"]["kind"], "q2": surrogate_q2,
            "n_train": srep["n_train"], "grid_summary": srep["grid_prediction"]["summary"],
        }

    cost_by_aor = _cost_by_aor(datafarm, aors_eff)
    decision_index = build_decision_index(
        master_table, primary_response=primary, cost_by_aor=cost_by_aor, alpha=alpha
    )

    primary_all = next(
        (r for r in master_table if r["response"] == primary and r["aor"] == "ALL"), None
    )
    headline = (
        significance_insight(primary, primary_all, alpha=alpha) if primary_all
        else {"text": "sem dados para a resposta primaria", "significant": False}
    )
    primary_block = next((b for b in responses_blocks if b["response"] == primary), None)
    dominant = (
        dominant_factor_insight(primary_block["feature_scores"]) if primary_block
        else {"factor": None, "score": None, "text": "-"}
    )
    n_pairs_primary = int(primary_all["n_pairs"]) if primary_all else 0
    evidence = evidence_sufficiency(
        p_value=(primary_all or {}).get("p_value"),
        ci_low=(primary_all or {}).get("ci_low"),
        ci_high=(primary_all or {}).get("ci_high"),
        alpha=alpha, n_design_points=n_design_points, n_factors=n_factors,
        n_pairs=n_pairs_primary, surrogate_q2=surrogate_q2,
    )

    seeds: dict[str, int | None] = {
        "design": study_meta.get("design_seed"),
        "base": study_meta.get("base_seed"),
        "bootstrap": bootstrap_seed,
        "model": model_seed,
    }
    validity = build_validity_block(
        factor_ranges=factor_ranges, n_design_points=n_design_points,
        n_reps=study_meta.get("n_reps"), seeds=seeds,
        max_arrivals=study_meta.get("max_arrivals"),
    )

    thresholds = {
        "alpha": alpha,
        "schober_cutoffs": list(SCHOBER_CUTOFFS),
        "surrogate_q2_min": SURROGATE_Q2_MIN,
        "design_points_per_factor": DESIGN_POINTS_PER_FACTOR,
        "min_pairs_for_power": MIN_PAIRS_FOR_POWER,
        "anova_n_levels": 3,
        "references": {
            "significance": "Bussab & Morettin (2010)",
            "correlation_strength": "Schober, Boer & Schwarte (2018)",
            "design_points": "Loeppky, Sacks & Welch (2009)",
            "power": "Cohen (1988)",
            "surrogate_q2": "Golbraikh & Tropsha (2002)",
        },
    }

    study: dict[str, Any] = {
        "id": study_meta.get("id", "consolidated"),
        "kind": "consolidated",
        "responses": list(responses_eff),
        "responses_skipped": skipped,
        "aors": list(aors_eff),
        "alpha": alpha,
        "bootstrap_seed": bootstrap_seed,
        "model_seed": model_seed,
        "n_design_points": n_design_points,
        "factors": factor_ranges,
    }

    decision_summary = {
        "primary_response": primary,
        "headline": headline,
        "dominant_factor": dominant,
        "evidence": evidence,
        "surrogate": surrogate_block,
        "note": "the decision index below is a reporting composite (nao metrica do paper)",
    }

    report: dict[str, Any] = {
        "field_version": CONSOLIDATED_FIELD_VERSION,
        "study": study,
        "thresholds": thresholds,
        "validity": validity,
        "decision_summary": decision_summary,
        "master_table": master_table,
        "responses": responses_blocks,
        "relationships": relationships,
        "anova_by_levels": anova_by_levels,
        "decision_index": decision_index,
        "figures": None,
    }
    safe: dict[str, Any] = _json_safe(report)
    return safe


# ─── presentation: markdown fallback + figures ──────────────────────────────


def _fmt(x: Any) -> str:
    """4-decimal fixed float (or '-' for None) — the sole numeric format in report.md."""
    if x is None:
        return "-"
    return f"{float(x):.4f}"


def _fmtp(p: Any) -> str:
    """4-significant-figure p-value (or '-')."""
    if p is None:
        return "-"
    return f"{float(p):.4g}"


def render_md(report: dict[str, Any]) -> str:
    """Render the consolidated ``report.md``. Pure; ASCII; contract-only.

    Every displayed number is a formatting of a ``report.json`` leaf via
    ``_fmt``/``_fmtp``/``str``; the presentation never computes.
    """
    s = report["study"]
    di = report["decision_index"]
    ds = report["decision_summary"]
    lines: list[str] = [
        "# Consolidated report",
        "",
        f"- field_version: {report['field_version']}",
        f"- study: {s.get('id', 'consolidated')} | kind: {s.get('kind')}",
        f"- responses: {', '.join(s.get('responses', []))} | "
        f"aors: {', '.join(s.get('aors', []))} | alpha: {s.get('alpha')}",
        f"- design_points: {s.get('n_design_points')} | "
        f"bootstrap_seed: {s.get('bootstrap_seed')} | model_seed: {s.get('model_seed')}",
        "",
        "## Decision summary",
        f"- primary: {ds.get('primary_response')}",
        f"- headline: {ds.get('headline', {}).get('text', '-')}",
        f"- {ds.get('dominant_factor', {}).get('text', '-')}",
        f"- evidence: {ds.get('evidence', {}).get('text', '-')}",
        "",
        f"## Decision index [{di.get('label')}]",
        f"- criterion: {di.get('criterion')}",
    ]
    for e in di.get("ranking", []):
        cost = e.get("cost", {})
        cost_s = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(cost.items())) or "-"
        flag = "significativo" if e.get("significant") else "evidencia insuficiente"
        lines.append(
            f"  {e.get('rank')}. {e.get('aor')}: effect={_fmt(e.get('effect'))} "
            f"({flag}) | cost: {cost_s}"
        )

    lines += ["", "## Master table (response | config | aor | effect | CI | p | test | sig)"]
    for r in report["master_table"]:
        ci_txt = (
            "-" if r.get("ci_low") is None
            else f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] ({r.get('ci_method')})"
        )
        lines.append(
            f"- {r['response']} | {r.get('config')} | {r['aor']}: "
            f"effect={_fmt(r['effect'])} CI={ci_txt} p={_fmtp(r.get('p_value'))} "
            f"test={r.get('test')} sig={r.get('significant')} n={r.get('n_pairs')}"
        )

    lines += ["", "## Responses"]
    for b in report["responses"]:
        top = b["feature_scores"][0]["factor"] if b.get("feature_scores") else "-"
        fit = b.get("fit")
        fit_s = "-" if fit is None else f"{fit['name']} (ks={_fmt(fit['ks_statistic'])})"
        lines += [
            f"- {b['response']}: {b['insight']['text']}",
            f"  top factor={top} | fit={fit_s} | outliers={b.get('outlier_count')}",
        ]

    lines += ["", "## Relationships (factor x gain)"]
    for rel in report["relationships"]:
        lines.append(
            f"- {rel['response']} {rel['factor']}: pearson={_fmt(rel['pearson'])} "
            f"spearman={_fmt(rel['spearman'])} kendall={_fmt(rel['kendall'])} "
            f"slope={_fmt(rel['slope'])} r2={_fmt(rel['r2'])} [{rel['strength']}]"
        )

    lines += ["", "## ANOVA by factor levels"]
    for a in report["anova_by_levels"]:
        lines.append(
            f"- {a['response']} {a['factor']} (levels={a.get('n_levels')}): "
            f"test={a.get('test')} p={_fmtp(a.get('p_value'))}"
        )

    v = report["validity"]
    lines += [
        "",
        "## Validity / scope",
        f"- design_points: {v.get('n_design_points')} | n_reps: {v.get('n_reps')} | "
        f"max_arrivals: {v.get('max_arrivals')}",
        f"- seeds: {', '.join(f'{k}={v2}' for k, v2 in sorted(v.get('seeds', {}).items()))}",
    ]
    for cav in v.get("caveats", []):
        lines.append(f"- caveat: {cav}")
    lines.append(f"- {v.get('byte_determinism_note')}")
    return "\n".join(lines) + "\n"


def render_figures_png(report: dict[str, Any]) -> dict[str, bytes]:  # pragma: no cover
    """Render the consolidated figures as standalone PNG bytes (lazy matplotlib).

    Deterministic per-environment: PNG metadata is stripped (no Software/date chunk) so
    two runs on the same matplotlib produce byte-identical bytes.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: dict[str, bytes] = {}

    def _save(fig: Any, key: str) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", metadata={"Software": None}, dpi=100)
        plt.close(fig)
        figs[key] = buf.getvalue()

    blocks = report.get("responses", [])
    primary = report["decision_summary"].get("primary_response")
    primary_block = next((b for b in blocks if b["response"] == primary), None)

    if primary_block and primary_block.get("feature_scores"):
        fs = primary_block["feature_scores"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([r["factor"] for r in fs], [r["score"] for r in fs])
        ax.set_title(f"Feature importance ({primary})")
        fig.tight_layout()
        _save(fig, "feature_importance")

        pts = primary_block["pareto_front"]["points"]
        if pts:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter([p["cost"] for p in pts], [p["gain"] for p in pts])
            ax.set_xlabel("cost")
            ax.set_ylabel("gain")
            ax.set_title(f"Pareto front ({primary})")
            fig.tight_layout()
            _save(fig, "pareto")

        hist = primary_block["eda"].get("histogram", {})
        if hist.get("counts"):
            edges = hist["edges"]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
            widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(centers, hist["counts"], width=widths)
            ax.set_xlabel("gain")
            ax.set_ylabel("count")
            ax.set_title(f"Distribution ({primary})")
            fig.tight_layout()
            _save(fig, "histogram")

    ranking = report["decision_index"].get("ranking", [])
    if ranking:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([e["aor"] for e in ranking], [e["effect"] for e in ranking])
        ax.set_ylabel("effect (paired dMOS-1)")
        ax.set_title("Decision index")
        fig.tight_layout()
        _save(fig, "decision_index")

    return figs


def write_consolidated_report(
    report: dict[str, Any],
    out_dir: Path,
    *,
    formats: Sequence[str] = ("html", "json"),
    figures_png: dict[str, bytes] | None = None,
) -> None:
    """Persist the consolidated ``report.json`` (+ ``report.md`` + standalone PNGs).

    ``report.json`` is ALWAYS written (the contract html/md derive from),
    byte-deterministic (``sort_keys``, no timestamp). ``report.md`` only when requested.
    The HTML dashboard is written by ``reporting.write_html``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if figures_png:
        report["figures"] = {k: f"{k}.png" for k in sorted(figures_png)}
        for key, data in figures_png.items():
            (out_dir / f"{key}.png").write_bytes(data)
    else:
        report["figures"] = None
    (out_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
    )
    if "md" in formats:
        (out_dir / "report.md").write_text(render_md(report), encoding="utf-8")


def _render_surface(grid_df: pd.DataFrame, out_dir: Path) -> None:  # pragma: no cover
    """Render the lambda x mix response-surface figure (lazy matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = [c for c in grid_df.columns if c != "predicted"]
    if len(factors) != 2:
        return
    fx, fy = factors
    pivot = grid_df.pivot_table(index=fy, columns=fx, values="predicted")
    fig, ax = plt.subplots(figsize=(6, 5))
    mesh = ax.pcolormesh(pivot.columns, pivot.index, pivot.to_numpy(), shading="auto")
    fig.colorbar(mesh, ax=ax, label="predicted")
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.set_title("Surrogate response surface")
    fig.tight_layout()
    fig.savefig(out_dir / "surface.png")
    plt.close(fig)
