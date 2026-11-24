"""SysML v2 ``calc def`` ↔ metrics guard + the researcher's adjudication.

Parses the 8 ``calc def`` of ``sysml/Analysis/CalculationsPackage.sysml`` (via the
AST parser), binds each to the ``metrics/aggregator.py``
function it claims to define through an explicit, versioned **adjudication table**
(``_ADJUDICATIONS`` — the researcher's verdicts), and offers (a) a
numeric **regression guard with teeth** for the ``CONSISTENT`` metrics and (b) a
documented adjudication matrix.

Two contracts, deliberately separated (as in the requirements checker and the phase
auditor):

* :func:`load_calculations` / :func:`check_calculations` are **lenient** — they
  NEVER raise on ``.sysml`` content.
* :func:`validate_calculations` is the **only** entry point that may raise
  :class:`CalculationsValidationError` (when a ``CONSISTENT`` guard fails — i.e.
  the frozen aggregator drifted away from the SysML formula it matches).

The module is read-only and metric-neutral: it **reads** the frozen aggregator on
synthetic in-memory inputs, never reimplements or runs the simulation. The guard imports of pandas/MissionConfig/MetricsAggregator
are **lazy** (inside the reference helpers), so importing this module is cheap.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from cuas_sim.sysml_parser import (
    ConstraintBinary,
    ConstraintLiteral,
    ConstraintRef,
    ConstraintUnary,
    SysMLParseError,
    parse_package,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cuas_sim.sysml_parser import ConstraintExpr, SysMLCalcInput

logger = logging.getLogger(__name__)

__all__ = [
    "Adjudication",
    "AdjudicationRow",
    "CalcReport",
    "CalculationsValidationError",
    "ParsedCalc",
    "build_adjudication_matrix",
    "check_calculations",
    "evaluate_expr",
    "load_calculations",
    "validate_calculations",
]

#: Guard tolerance for SysML-formula vs aggregator comparison.
_REL_TOL = 1e-9
_ABS_TOL = 1e-12


class CalculationsValidationError(Exception):
    """Raised by the opt-in validation step when ≥1 ``CONSISTENT`` guard fails."""


@dataclass(frozen=True, slots=True)
class ParsedCalc:
    """One ``calc def`` with its inputs and parsed return expression."""

    name: str
    package: str
    inputs: tuple[SysMLCalcInput, ...]
    return_name: str | None
    return_type: str | None
    expr: ConstraintExpr | None


@dataclass(frozen=True, slots=True)
class Adjudication:
    """One curated verdict mapping a ``calc def`` to its aggregator function.

    ``verdict`` is one of ``CONSISTENT`` / ``DIVERGENCE-ACCEPTED`` /
    ``DIVERGENCE-ESCALATE`` / ``MAPPING-FIX``. For ``CONSISTENT``
    entries, ``synthetic_inputs`` + ``aggregator_eval`` carry the numeric guard:
    the SysML ``expr`` is evaluated on ``synthetic_inputs`` and compared with
    ``aggregator_eval()`` (the value the bound aggregator function produces on the
    equivalent synthetic inputs). For the other verdicts both are ``None``.
    """

    calc_name: str
    metric_id: str
    aggregator_ref: str
    verdict: str
    rationale: str
    spec_ref: str
    synthetic_inputs: dict[str, float] | None = None
    aggregator_eval: Callable[[], float] | None = None


@dataclass(frozen=True, slots=True)
class AdjudicationRow:
    """One row of the ``calc def -> metric -> função -> veredito`` matrix."""

    calc_name: str
    metric_id: str
    aggregator_ref: str
    verdict: str
    spec_ref: str


@dataclass(slots=True)
class CalcReport:
    """Outcome of :func:`check_calculations` — pure classification, never raises."""

    consistent: list[str] = field(default_factory=list)
    accepted: list[str] = field(default_factory=list)
    escalated: list[str] = field(default_factory=list)
    mapping_fixed: list[str] = field(default_factory=list)
    unbound: list[str] = field(default_factory=list)
    failed_guards: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)


# ─── Arithmetic evaluation (pure) ───────────────────────────────────────────


def evaluate_expr(expr: ConstraintExpr, env: dict[str, float]) -> float:
    """Evaluate a parsed ``calc def`` return expression to a float.

    Supports ``+ - * /`` (binary), unary ``-``, references (looked up in ``env``)
    and numeric literals. Pure — no RNG, no I/O.
    """
    if isinstance(expr, ConstraintLiteral):
        return float(expr.value)
    if isinstance(expr, ConstraintRef):
        return float(env[expr.name])
    if isinstance(expr, ConstraintUnary):
        v = evaluate_expr(expr.operand, env)
        if expr.op == "-":
            return -v
        raise SysMLParseError(f"unsupported unary op {expr.op!r} in calc expr")
    if isinstance(expr, ConstraintBinary):
        left = evaluate_expr(expr.left, env)
        right = evaluate_expr(expr.right, env)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        raise SysMLParseError(f"unsupported binary op {expr.op!r} in calc expr")
    raise SysMLParseError(f"unsupported expr node {type(expr).__name__}")


# ─── Aggregator reference values (the guard's teeth — read-only, lazy import) ─
#
# Each helper invokes the REAL ``metrics/aggregator.py`` function on a minimal
# synthetic in-memory input chosen to match the paired ``synthetic_inputs`` of
# the SysML formula. If the aggregator's aggregation drifts, the returned value
# changes and the guard fails (drift detector).


def _all_value(rows: list[dict[str, float | str]], metric: str) -> float:
    for row in rows:
        if row.get("aor") == "ALL" and row.get("metric") == metric:
            return float(row["value"])
    raise KeyError(f"no ALL row for {metric}")


def _mos1_ref() -> float:
    # 7 kills / 10 incursions = 0.7  (matches synthetic {Nn: 7, Nt: 10}).
    from types import SimpleNamespace

    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.metrics.aggregator import MetricsAggregator

    df = pd.DataFrame({
        "aor_name": ["AoR-1"] * 10,
        "is_neutralized_before_zida": [True] * 7 + [False] * 3,
    })
    agg = MetricsAggregator(cfg=MissionConfig())
    log = SimpleNamespace(leaks=[])  # duck-typed RunLogger: only .leaks is read here
    rows = agg._compute_mos1(df, "baseline", 10, log)  # type: ignore[arg-type]
    return _all_value(rows, "MOS-1")


def _moe2_ref() -> float:
    # mean(0.25, 0.5, 0.75) = 0.5  (matches synthetic {totalMOE2Sum: 1.5, N: 3}).
    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.metrics.aggregator import MetricsAggregator

    df = pd.DataFrame({"aor_name": ["A"] * 3, "moe2_mean_per_engagement": [0.25, 0.5, 0.75]})
    rows = MetricsAggregator(cfg=MissionConfig())._compute_moe2(df, "baseline")
    return _all_value(rows, "MOE-2")


def _moe3_ref() -> float:
    # mean(0.25, 0.5, 0.75) = 0.5  (matches synthetic {totalPhitSum: 1.5, N: 3}).
    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.metrics.aggregator import MetricsAggregator

    df = pd.DataFrame({"aor_name": ["A"] * 3, "moe3_p_hit_expected": [0.25, 0.5, 0.75]})
    rows = MetricsAggregator(cfg=MissionConfig())._compute_moe3(df, "baseline")
    return _all_value(rows, "MOE-3")


def _mop2_ref() -> float:
    # _mop2_row returns the configured max acquisition distance (default 13 km).
    from cuas_sim.config import MissionConfig
    from cuas_sim.metrics.aggregator import MetricsAggregator

    row = MetricsAggregator(cfg=MissionConfig())._mop2_row("baseline")
    return float(row["value"])


def _moe1_ref() -> float:
    # 10 neutralized engagements with t_ef = 0.1 * T_PEA give a ratio
    # of 0.1 each, so MOE-1 = 1 - mean(0.1) = 0.9, independent of the numeric
    # T_PEA. The paired synthetic SysML inputs {totalTefSum: 1, N: 10, Tmax: 1}
    # evaluate 1 - 1/(10*1) = 0.9 as well. The guard exercises the REAL
    # _compute_moe1 (the '1 -' complement and the per-aircraft T_PEA denominator),
    # so any drift of the aggregator from the formula changes the value and trips it.
    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.metrics.aggregator import MetricsAggregator

    cfg = MissionConfig()
    agg = MetricsAggregator(cfg=cfg)
    aor = next(iter(cfg.aor_to_a29))
    tpea = agg._t_pea_for_aor(aor)
    df = pd.DataFrame({
        "aor_name": [aor] * 10,
        "config_name": ["baseline"] * 10,
        "is_neutralized": [True] * 10,
        "moe1_engagement_time_s": [0.1 * tpea] * 10,
        "t_ev_s": [float("nan")] * 10,
        "t_pursuit_start_s": [0.0] * 10,
    })
    rows = agg._compute_moe1(df, "baseline")
    return _all_value(rows, "MOE-1")


# ─── Adjudication table ──────────────────────────────
#
# The researcher's verdicts (2026-06-19). For CONSISTENT entries the guard
# compares ``evaluate_expr(expr, synthetic_inputs)`` with ``aggregator_eval()``.

_SPEC = "Spec 024 §8.1"
_SPEC_025 = "Spec 025 §5.4"

_ADJUDICATIONS: dict[str, Adjudication] = {
    "calculateNeutralizationRate": Adjudication(
        "calculateNeutralizationRate", "MOS-1",
        "metrics.aggregator._compute_mos1 (Nn/Nt)", "CONSISTENT",
        "SysML, paper Eq.1 and Python agree: kills inside ZIDA / total incursions.",
        _SPEC, synthetic_inputs={"Nn": 7.0, "Nt": 10.0}, aggregator_eval=_mos1_ref,
    ),
    "calculateEvasionReductionRate": Adjudication(
        "calculateEvasionReductionRate", "MOS-2",
        "metrics.aggregator._compute_mos2 (1 - leaks_m/leaks_b)", "DIVERGENCE-ACCEPTED",
        "SysML matches paper Eq.2 (denom rM); Python uses rB, which respects the "
        "Table 20 [0,1] bound that Eq.2 violates. Python accepted; paper errata candidate.",
        _SPEC,
    ),
    "calculateEngagementTimeRate": Adjudication(
        "calculateEngagementTimeRate", "MOE-1",
        "metrics.aggregator._compute_moe1 (1 - mean tef/T_PEA, Eq.5/6)", "CONSISTENT",
        "Spec 025 (Article VI section-9) reconciled _compute_moe1 with Eq.5: the '1 -' "
        "complement and the evasion penalty (1+E/N) were restored, so the outer aggregation "
        "1 - Sum(tef)/(N*Tmax) now matches the calc def. Inner per-engagement tef/penalty/t_ev "
        "and the per-aircraft T_PEA (Spec 018) are below calc-def granularity (sanctioned, "
        "like MOE-2/MOE-3). Closes the SR-4b ESCALATE backlog.",
        _SPEC_025, synthetic_inputs={"totalTefSum": 1.0, "N": 10.0, "Tmax": 1.0},
        aggregator_eval=_moe1_ref,
    ),
    "calculateTrackingRate": Adjudication(
        "calculateTrackingRate", "MOE-2",
        "metrics.aggregator._compute_moe2 (mean per-engagement)", "CONSISTENT",
        "Outer aggregation (Eq.10, Sum/N) matches. Inner per-step P_track was upgraded to "
        "angular acuity (Spec 016), below calc-def granularity — sanctioned, not a drift.",
        _SPEC, synthetic_inputs={"totalMOE2Sum": 1.5, "N": 3.0}, aggregator_eval=_moe2_ref,
    ),
    "calculateNeutralizationProbability": Adjudication(
        "calculateNeutralizationProbability", "MOE-3",
        "metrics.aggregator._compute_moe3 (mean P_neut)", "CONSISTENT",
        "Outer aggregation (Eq.19, Sum/N) matches. Inner term is the multi-shot cumulative "
        "P_neut over continuous firing distances (Spec 015/017), below calc-def granularity.",
        _SPEC, synthetic_inputs={"totalPhitSum": 1.5, "N": 3.0}, aggregator_eval=_moe3_ref,
    ),
    "calculateUpdateRate": Adjudication(
        "calculateUpdateRate", "MOP-1",
        "config.*_update_interval_s (MAPPING-FIX: not _compute_mop1's count)", "MAPPING-FIX",
        "SysML (min+max)/2 is the update INTERVAL = the paper's MOP-1 ([1,60] s). Python's "
        "_compute_mop1 is an emergent heading-update COUNT — a different quantity. Re-bind the "
        "calc def to the update_interval_s config parameter; no formula change (metric-neutral).",
        _SPEC,
    ),
    "calculateAcquisitionRange": Adjudication(
        "calculateAcquisitionRange", "MOP-2",
        "metrics.aggregator._mop2_row (max acquisition)", "CONSISTENT",
        "SysML, paper (Table 18) and Python agree: the maximum acquisition distance.",
        _SPEC, synthetic_inputs={"maxAcquisitionDistance": 13000.0}, aggregator_eval=_mop2_ref,
    ),
    "calculateWeaponEnvelope": Adjudication(
        "calculateWeaponEnvelope", "MOP-3",
        "metrics.aggregator._mop3_row (envelope[1] max)", "DIVERGENCE-ACCEPTED",
        "SysML reports the envelope WIDTH (max-min), Python the MAX, the paper Table 20 the "
        "interval [0.5,14]. A reported constant that feeds no computation - representation choice.",
        _SPEC,
    ),
}


# ─── Loading (lenient — never raises) ─────────────────────────


def load_calculations(root: Path | str = "sysml") -> list[ParsedCalc]:
    """Parse ``Analysis/CalculationsPackage.sysml``. NEVER raises."""
    path = Path(root) / "Analysis" / "CalculationsPackage.sysml"
    if not path.is_file():
        logger.warning("Missing SysML calculations package: %s — none loaded", path)
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read %s: %s — skipping", path, exc)
        return []
    try:
        pkg = parse_package(text)
    except SysMLParseError as exc:
        logger.warning("SysML parse failed for %s: %s — skipping file", path, exc)
        return []
    return [
        ParsedCalc(c.name, pkg.name, c.inputs, c.return_name, c.return_type, c.expr)
        for c in pkg.calc_defs
    ]


# ─── Checking + validation ──────────────────────────────────────────────────


def _run_guard(calc: ParsedCalc | None, adj: Adjudication) -> tuple[bool, str]:
    """Numeric known-answer guard for a CONSISTENT verdict. Returns (ok, detail)."""
    if calc is None:
        return True, f"{adj.metric_id}: calc def absent — guard skipped"
    if calc.expr is None:
        return True, f"{adj.metric_id}: non-evaluable return expr — guard skipped"
    if adj.synthetic_inputs is None or adj.aggregator_eval is None:
        return True, f"{adj.metric_id}: no guard artifacts — skipped"
    sysml_val = evaluate_expr(calc.expr, adj.synthetic_inputs)
    agg_val = adj.aggregator_eval()
    ok = math.isclose(sysml_val, agg_val, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    detail = (
        f"{adj.metric_id}: SysML={sysml_val:.6g} vs aggregator={agg_val:.6g} "
        f"[{adj.aggregator_ref}]" + ("" if ok else " -- DRIFT")
    )
    return ok, detail


def check_calculations(
    calcs: list[ParsedCalc],
    adjudications: dict[str, Adjudication] | None = None,
) -> CalcReport:
    """Classify each ``calc def`` by verdict and run the CONSISTENT guards.

    PURE (no RNG, no global state). Reads the frozen aggregator on synthetic
    in-memory inputs for CONSISTENT metrics; never raises.
    """
    table = _ADJUDICATIONS if adjudications is None else adjudications
    report = CalcReport()
    by_name = {c.name: c for c in calcs}
    for name, adj in table.items():
        calc = by_name.get(name)
        if adj.verdict == "CONSISTENT":
            ok, detail = _run_guard(calc, adj)
            report.details[name] = detail
            (report.consistent if ok else report.failed_guards).append(name)
        elif adj.verdict == "DIVERGENCE-ACCEPTED":
            report.accepted.append(name)
            report.details[name] = f"{adj.metric_id} ACCEPTED: {adj.rationale}"
        elif adj.verdict == "MAPPING-FIX":
            report.mapping_fixed.append(name)
            report.details[name] = f"{adj.metric_id} MAPPING-FIX -> {adj.aggregator_ref}"
        elif adj.verdict == "DIVERGENCE-ESCALATE":
            report.escalated.append(name)
            report.details[name] = (
                f"{adj.metric_id} ESCALATE (candidate formula correction): {adj.rationale}"
            )
        else:
            report.unbound.append(name)
            report.details[name] = f"{adj.metric_id}: unknown verdict {adj.verdict!r}"
    for c in calcs:
        if c.name not in table:
            report.unbound.append(c.name)
            report.details[c.name] = "no adjudication entry (unbound)"
    return report


def validate_calculations(
    root: Path | str = "sysml",
    adjudications: dict[str, Adjudication] | None = None,
) -> None:
    """Raise :class:`CalculationsValidationError` if a CONSISTENT guard fails.

    The ONLY entry point that raises; ``load_calculations`` / ``check_calculations``
    are always lenient. ESCALATE verdicts are backlog — they never cause a raise.
    """
    report = check_calculations(load_calculations(root), adjudications)
    if report.failed_guards:
        lines = "\n".join(f"  - {n}: {report.details.get(n, '')}" for n in report.failed_guards)
        raise CalculationsValidationError(
            "CONSISTENT calc-def guard(s) failed (the aggregator drifted from the SysML formula):\n" + lines
        )


# ─── Traceability matrix ────────────────────────────────────────────


def build_adjudication_matrix(
    adjudications: dict[str, Adjudication] | None = None,
) -> list[AdjudicationRow]:
    """Build the full 8-row adjudication/traceability matrix."""
    table = _ADJUDICATIONS if adjudications is None else adjudications
    return [
        AdjudicationRow(a.calc_name, a.metric_id, a.aggregator_ref, a.verdict, a.spec_ref)
        for a in table.values()
    ]
