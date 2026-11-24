"""Known-answer tests for the AN-report consolidated reporting layer (Spec 029 §5.1)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from cuas_sim.analysis.report import (
    CAUSALITY_CAVEAT,
    NO_GENERALIZE_CAVEAT,
    build_decision_index,
    build_validity_block,
    correlation_insight,
    render_md,
    significance_insight,
)
from cuas_sim.analysis.reporting import render_html
from cuas_sim.analysis.stats import (
    analyze_relationship,
    bin_factor_levels,
    kgroup_hypothesis_test,
)


def test_analyze_relationship_known() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    res = analyze_relationship(df, "x", "y")
    assert res.pearson is not None and abs(res.pearson - 1.0) < 1e-9
    assert res.spearman is not None and abs(res.spearman - 1.0) < 1e-9
    assert res.kendall is not None and abs(res.kendall - 1.0) < 1e-9
    assert res.slope is not None and abs(res.slope - 2.0) < 1e-9
    assert res.intercept is not None and abs(res.intercept - 0.0) < 1e-9
    assert res.r2 is not None and abs(res.r2 - 1.0) < 1e-9
    assert res.strength == "muito forte"  # |r| >= 0.90


def test_insight_significance_ci_excludes_zero() -> None:
    block = {"effect": 0.17, "ci_low": 0.16, "ci_high": 0.19,
             "p_value": 1e-4, "test": "wilcoxon"}
    ins = significance_insight("MOS-1", block, alpha=0.05)
    assert ins["significant"] is True
    for token in ("MOS-1", "+0.17", "Rejeita-se", "[0.16", "0.19]", "alpha=0.05"):
        assert token in ins["text"], token
    assert "modernized > baseline" in ins["text"]
    ins["text"].encode("cp1252")  # cp1252-safe (FR-14)


def test_insight_insufficient_evidence_ci_crosses_zero() -> None:
    block = {"effect": 0.01, "ci_low": -0.02, "ci_high": 0.04,
             "p_value": 0.4, "test": "wilcoxon"}
    ins = significance_insight("MOS-1", block, alpha=0.05)
    assert ins["significant"] is False
    assert "Nao se rejeita" in ins["text"]
    assert "evidencia insuficiente; rodar mais replicas/pontos" in ins["text"]
    ins["text"].encode("cp1252")


def test_correlation_strength_labels_schober() -> None:
    expected = {0.05: "negligivel", 0.20: "fraca", 0.50: "moderada",
                0.80: "forte", 0.95: "muito forte"}
    for r, label in expected.items():
        ins = correlation_insight("mix", r)
        assert ins["strength"] == label, (r, ins["strength"])
        assert ins["caveat"] == CAUSALITY_CAVEAT


def test_bin_factor_levels_anova() -> None:
    lam = np.linspace(2.0, 18.0, 30)
    rng = np.random.default_rng(0)
    gain = 0.01 * lam + rng.normal(0.0, 0.02, 30)
    table = pd.DataFrame({"lambda_per_h": lam, "gain": gain})

    levels = bin_factor_levels(table, "lambda_per_h", n_levels=3, method="quantile")
    assert len(levels) == 30
    assert set(levels) == {"low", "mid", "high"}
    counts = Counter(levels)
    assert all(counts[label] > 0 for label in ("low", "mid", "high"))

    # anova-by-levels REUSES the existing k-group test (no reimplementation)
    lv = np.asarray(levels)
    groups = [table["gain"].to_numpy()[lv == label].tolist()
              for label in ("low", "mid", "high")]
    res = kgroup_hypothesis_test(groups)
    assert res.test in ("f_oneway", "welch_anova", "kruskal")
    assert 0.0 <= res.p_value <= 1.0


def test_validity_scope_section() -> None:
    vb = build_validity_block(
        factor_ranges={"lambda_per_h": {"min": 2.0, "max": 18.0},
                       "uas_archetype_p_mr": {"min": 0.0, "max": 1.0}},
        n_design_points=32,
        n_reps=24,
        seeds={"design": 42, "base": 42, "bootstrap": 20260701, "model": 20260701},
        max_arrivals=None,
    )
    assert vb["factor_ranges"]["lambda_per_h"]["max"] == 18.0
    assert vb["factor_ranges"]["uas_archetype_p_mr"]["min"] == 0.0
    assert vb["n_design_points"] == 32
    assert vb["n_reps"] == 24
    assert set(vb["seeds"]) == {"design", "base", "bootstrap", "model"}
    assert NO_GENERALIZE_CAVEAT in vb["caveats"]
    assert CAUSALITY_CAVEAT in vb["caveats"]


def test_decision_index_prioritizes_neutralization() -> None:
    master = [
        {"response": "MOS-1", "aor": "AoR-1", "effect": 0.20,
         "ci_low": 0.15, "ci_high": 0.25, "p_value": 1e-3, "test": "wilcoxon"},
        {"response": "MOS-1", "aor": "AoR-2", "effect": 0.05,
         "ci_low": 0.02, "ci_high": 0.08, "p_value": 1e-2, "test": "wilcoxon"},
        {"response": "MOS-1", "aor": "AoR-3", "effect": 0.30,
         "ci_low": -0.05, "ci_high": 0.65, "p_value": 0.20, "test": "wilcoxon"},
    ]
    cost = {
        "AoR-1": {"rockets_per_engagement": 3.0, "leaks": 10.0},
        "AoR-2": {"rockets_per_engagement": 1.0, "leaks": 2.0},
        "AoR-3": {"rockets_per_engagement": 5.0, "leaks": 20.0},
    }
    idx = build_decision_index(master, primary_response="MOS-1", cost_by_aor=cost)

    ranking = idx["ranking"]
    order = [r["aor"] for r in ranking]
    # significant ordered by effect desc: AoR-1 (0.20) then AoR-2 (0.05); AoR-3 demoted
    assert order.index("AoR-1") < order.index("AoR-2")
    assert order.index("AoR-2") < order.index("AoR-3")

    by_aor = {r["aor"]: r for r in ranking}
    assert by_aor["AoR-1"]["significant"] is True
    assert by_aor["AoR-3"]["significant"] is False
    assert "evidencia insuficiente" in by_aor["AoR-3"]["note"]
    # cost is an ANNOTATION (present) but did NOT reorder (AoR-2 cheapest yet 2nd)
    assert by_aor["AoR-1"]["cost"]["rockets_per_engagement"] == 3.0
    # explicit reporting-composite label (not a paper metric) — INV-7
    assert "reporting composite (nao metrica do paper)" in idx["label"]


def _fixed_report() -> dict:
    """A minimal but complete consolidated contract (no datafarm needed) — INV-3."""
    return {
        "field_version": "1.0",
        "study": {"id": "test", "kind": "consolidated", "alpha": 0.05,
                  "responses": ["MOS-1"], "aors": ["ALL"],
                  "n_design_points": 8, "factors": {}},
        "thresholds": {"surrogate_q2_min": 0.5},
        "validity": {"factor_ranges": {}, "n_design_points": 8, "n_reps": 6,
                     "seeds": {"design": 1, "base": 1, "bootstrap": 2, "model": 3},
                     "max_arrivals": None,
                     "caveats": [NO_GENERALIZE_CAVEAT, CAUSALITY_CAVEAT]},
        "decision_summary": {
            "primary_response": "MOS-1",
            "headline": {"text": "MOS-1: efeito +0.17 significativo", "significant": True},
            "dominant_factor": {"factor": "uas_archetype_p_mr", "score": 161.0,
                                "text": "fator dominante: uas_archetype_p_mr "
                                        "(feature score 161.00 vs lambda_per_h 5.70)"},
            "evidence": {"sufficient": True, "reasons": [],
                         "text": "evidencia suficiente para a conclusao no espaco amostrado"},
            "surrogate": None,
            "note": "the decision index below is a reporting composite (nao metrica do paper)",
        },
        "master_table": [
            {"response": "MOS-1", "config": "modernized-baseline", "aor": "ALL",
             "effect": 0.17, "ci_low": 0.16, "ci_high": 0.19, "ci_method": "BCa",
             "p_value": 0.0001, "test": "wilcoxon", "significant": True, "n_pairs": 48},
        ],
        "responses": [
            {"response": "MOS-1",
             "feature_scores": [{"factor": "uas_archetype_p_mr", "score": 161.0},
                                {"factor": "lambda_per_h", "score": 5.7}],
             "dominant_factor": {"factor": "uas_archetype_p_mr", "score": 161.0,
                                 "text": "fator dominante: uas_archetype_p_mr "
                                         "(feature score 161.00 vs lambda_per_h 5.70)"},
             "eda": {"summary": {"count": 8, "mean": 0.17, "std": 0.02,
                                 "min": 0.1, "max": 0.2},
                     "outlier_count": 0,
                     "histogram": {"counts": [1, 3, 4], "edges": [0.1, 0.13, 0.16, 0.2]}},
             "fit": {"name": "norm", "ks_statistic": 0.1, "p_value": 0.9},
             "insight": {"text": "MOS-1: efeito +0.17 significativo", "significant": True}},
        ],
        "relationships": [
            {"response": "MOS-1", "factor": "uas_archetype_p_mr", "col_x": "uas_archetype_p_mr",
             "col_y": "gain", "pearson": 0.92, "spearman": 0.9, "kendall": 0.8,
             "slope": 0.25, "intercept": 0.01, "r2": 0.85, "strength": "muito forte",
             "caveat": CAUSALITY_CAVEAT},
        ],
        "anova_by_levels": [
            {"response": "MOS-1", "factor": "lambda_per_h", "n_levels": 3,
             "test": "kruskal", "p_value": 0.03, "reject_h0": True},
        ],
        "decision_index": {
            "label": "reporting composite (nao metrica do paper)",
            "primary_response": "MOS-1",
            "ranking": [{"aor": "ALL", "effect": 0.17, "significant": True, "rank": 1,
                         "cost": {"rockets_per_engagement": 3.2, "leaks": 5.0}, "note": ""}],
        },
        "figures": None,
    }


def test_presentation_formats_contract_only() -> None:
    report = _fixed_report()
    # presentation is a pure function of the contract — NO datafarm required
    md = render_md(report)
    html = render_html(report, {})

    assert isinstance(md, str) and isinstance(html, str)
    # contract numbers surface in both layers (traceable to report.json)
    assert "0.1700" in md or "0.17" in md
    assert "MOS-1" in md and "MOS-1" in html
    assert "reporting composite (nao metrica do paper)" in md
    # ASCII-safe markdown fallback (FR-14)
    md.encode("cp1252")
    # self-contained HTML (offline, no CDN)
    assert "<html" in html.lower()
    assert "http://" not in html and "https://" not in html
