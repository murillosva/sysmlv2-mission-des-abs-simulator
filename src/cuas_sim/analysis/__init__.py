"""Statistical analysis layer — data farming over the paired simulator.

The data-farming modules implement the first two AsaPy steps (Dantas et al. 2024):

* ``doe``        — Design of Experiments (Latin Hypercube Sampling).
* ``execution``  — Execution Control (chunked early-stop heuristic).
* ``experiment`` — the data-farm runner (paired CRN replications -> tidy rows).

The layer is **read-only** over the metrics: it *consumes*
``MetricsAggregator.compute_all`` and never recomputes MOS/MOE/MOP. The DoE
sampler's ``design_seed`` is a meta-level RNG, recorded in the
artifacts; it does not touch the simulation's ``RNGRegistry`` bit-identity.
"""

from __future__ import annotations

from cuas_sim.analysis.doe import Factor, create_design, write_design
from cuas_sim.analysis.execution import (
    ExecutionController,
    convergence_stop,
    non_stop_func,
)
from cuas_sim.analysis.experiment import (
    factors_from_names,
    run_data_farm,
    write_datafarm,
)
from cuas_sim.analysis.features import FeatureScore, feature_scores, pareto_front
from cuas_sim.analysis.prediction import (
    CVScores,
    SurrogateFit,
    build_surrogate_report,
    cross_validate_surrogate,
    feature_importances,
    predict_grid,
    train_surrogate,
)
from cuas_sim.analysis.report import (
    build_consolidated_report,
    build_report,
    per_design_point,
    render_figures_png,
    render_md,
    write_consolidated_report,
    write_report,
    write_surrogate,
)
from cuas_sim.analysis.reporting import render_html, write_html
from cuas_sim.analysis.stats import (
    BootstrapResult,
    DistFit,
    HypothesisResult,
    OutlierResult,
    PreCheck,
    RelationshipResult,
    analyze_relationship,
    bin_factor_levels,
    bootstrap_effect,
    correlation_strength,
    detect_outliers,
    fit_distribution,
    kgroup_hypothesis_test,
    paired_hypothesis_test,
)

__all__ = [
    "BootstrapResult",
    "CVScores",
    "DistFit",
    "ExecutionController",
    "Factor",
    "FeatureScore",
    "HypothesisResult",
    "OutlierResult",
    "PreCheck",
    "RelationshipResult",
    "SurrogateFit",
    "analyze_relationship",
    "bin_factor_levels",
    "bootstrap_effect",
    "build_consolidated_report",
    "build_report",
    "build_surrogate_report",
    "convergence_stop",
    "correlation_strength",
    "create_design",
    "cross_validate_surrogate",
    "detect_outliers",
    "factors_from_names",
    "feature_importances",
    "feature_scores",
    "fit_distribution",
    "kgroup_hypothesis_test",
    "non_stop_func",
    "paired_hypothesis_test",
    "pareto_front",
    "per_design_point",
    "predict_grid",
    "render_figures_png",
    "render_html",
    "render_md",
    "run_data_farm",
    "train_surrogate",
    "write_consolidated_report",
    "write_datafarm",
    "write_design",
    "write_html",
    "write_report",
    "write_surrogate",
]
