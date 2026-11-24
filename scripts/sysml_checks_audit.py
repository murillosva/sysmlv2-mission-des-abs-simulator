"""Audit of the three checks that use the SysML v2 model as a reference.

Three parts, each with cases that must pass and cases that must fail, written as
byte-deterministic artifacts in ``outputs/verification/``:

* ``requirements``: the requirements checker (``sysml_requirements``);
* ``phases``: the phase auditor (``sysml_state_machine``), on replications of the campaign
  rerun with the event log written;
* ``calculations``: the calculation guard (``sysml_calculations``).

The checkers themselves only print to the console; this script records what each one
checks, on what, and with what result, including what it does not cover. It is read-only
over the simulator's code; its only execution is the rerun of existing replications (part
``phases``), which produces no new experimental data.

Cases, declared before the first run:

Requirements
  * the nominal configuration, resolved as in the studies (scenario file and model);
    expected: the 14 bound requirements satisfied;
  * the 69 design points of the campaign (base configuration with the lambda and p_mr of
    each study's ``design.parquet``); expected: the points with p_mr != 0.5 violate
    TR-021 (equal class probabilities) and only it, a deliberate departure of the design;
  * 18 injected violations, one per clause of the constraints of the 14 requirements
    (10% beyond the limit); expected: each one fails on its target requirement only.
Phases
  * fixed sample: replication 0 of each of the 69 design points, rerun with the same
    engine, configuration and seed as in the campaign, writing the event log;
  * a replication counts as audited only if its rerun measures are bit-identical to the
    datafarm; a divergence is reported, not hidden;
  * expected: every state stream conforms;
  * cases that must fail: in each stream (replication x configuration x A-29), three
    injected violations (phase regression, missing mandatory phase, state without a
    phase); expected: all detected, with the right type.
Calculations
  * the 8 ``calc def``: expression, what the guard compares and the verdict;
  * 13 in-memory mutations of the aggregator (one at a time, the original restored);
    mutation M04 reproduces a defect of MOE-1 that the comparison with the model revealed
    during development and that was corrected before the campaign.

Usage::

    python scripts/sysml_checks_audit.py                  # the three parts
    python scripts/sysml_checks_audit.py --part requirements
    python scripts/sysml_checks_audit.py --out-dir <dir>  # another destination

Console output is ASCII.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import statistics
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FIELD_VERSION = "1.0"
SAMPLE_REP = 0  # declared sampling rule: replication 0 of each design point
OUT_DIR = REPO / "outputs" / "verification"
SYSML = REPO / "sysml"
SCENARIO_YAML = REPO / "configs" / "zida_shield_default.yaml"
ANALYSIS = REPO / "outputs" / "analysis"
INJECTED_STATE = "INJECTED_UNKNOWN_STATE"
LAMBDA_COL = "lambda_per_h"
PMR_COL = "uas_archetype_p_mr"

#: Directory of each study and the datafarm of its accepted rung.
STUDY_DIRS = {"EX-1": "ex1_mc_nominal", "EX-2": "ex2_lambda", "EX-3": "ex3_mix",
              "EX-4": "ex4_interaction"}
ACCEPTED_DATAFARM = {"EX-1": "datafarm_n200.parquet", "EX-2": "datafarm_r96.parquet",
                     "EX-3": "datafarm_r192.parquet", "EX-4": "datafarm_r384.parquet"}


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


# ==========================================================================
# Resolved configuration and campaign design
# ==========================================================================
def _study_dir(study: str) -> Path:
    return ANALYSIS / STUDY_DIRS[study]


def _load_design(study: str) -> pd.DataFrame:
    return pd.read_parquet(_study_dir(study) / "design.parquet")


class _Capture:
    """Collect the messages of a logger while active, without echoing them."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        self.messages: list[str] = []
        sink = self.messages

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                sink.append(record.getMessage())

        self._handler = _Handler(level=logging.INFO)
        self._state: tuple[int, bool] = (0, True)

    def __enter__(self) -> _Capture:
        self._state = (self._logger.level, self._logger.propagate)
        # A global ``logging.disable(...)`` set by the caller would also silence this
        # handler (and the provenance would come out empty with no error): re-enable it
        # during the capture and restore the caller's level on exit.
        self._disabled = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self._logger.setLevel(20)  # INFO
        self._logger.propagate = False
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc: object) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._state[0])
        self._logger.propagate = self._state[1]
        logging.disable(self._disabled)


def _resolved_config() -> Any:
    """``MissionConfig`` resolved as in the studies (scenario file and SysML v2 model)."""
    from cuas_sim.config import MissionConfig

    with _Capture("cuas_sim.sysml_loader"), _Capture("cuas_sim.config"):
        return MissionConfig.from_yaml_with_sysml(SCENARIO_YAML, SYSML)


#: UAS archetypes: the loader applies them without recording provenance (the
#: ``MultiRotorUASIllicit``/``FixedWingUASIllicit`` branch of ``_dispatch_part_def``), so the
#: element name is declared here and checked against the syntax tree in
#: ``_config_provenance`` (model value in SI == resolved value).
_UAS_ELEMENTS = {
    "uas_mr": ("MultiRotorUASIllicit", ("maxSpeed", "maxDimensionWidth", "ceiling")),
    "uas_fw": ("FixedWingUASIllicit", ("maxSpeed", "maxDimensionWidth", "ceiling")),
}


def _config_provenance(config: Path = SCENARIO_YAML,
                       sysml_root: Path = SYSML) -> pd.DataFrame:
    """Effective source of each field of the resolved ``MissionConfig``.

    ``model``    -- the loader applied the value of a ``part def``; ``apply_to`` overrides
                    the scenario file, so the effective precedence is model > scenario >
                    default;
    ``scenario`` -- key present in the scenario file and not overridden;
    ``default``  -- default value of ``MissionConfig``.

    The applied fields come from the messages that ``apply_to`` itself emits
    ("SysML override applied: <field> = ..."), not from a comparison of values: the model
    values equal those of the scenario and of the defaults, so a comparison of values
    would distinguish nothing (see ``scenario_equals_resolved``).
    """
    import re
    from dataclasses import fields

    import yaml

    from cuas_sim.config import MissionConfig
    from cuas_sim.sysml_loader import SysMLLoader, _to_si
    from cuas_sim.sysml_parser import parse_package

    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    default = MissionConfig()
    with _Capture("cuas_sim.config"), _Capture("cuas_sim.sysml_loader"):
        scenario = MissionConfig.from_yaml(config)
        overrides = SysMLLoader(sysml_root).load()
    with _Capture("cuas_sim.sysml_loader") as cap, _Capture("cuas_sim.config"):
        resolved = overrides.apply_to(scenario)
    applied = {m.group(1) for msg in cap.messages
               if (m := re.match(r"SysML override applied: (\w+) = ", msg))}
    if overrides.provenance and not applied:
        raise AssertionError("the loader recorded provenance, but no application was captured")

    elements: dict[str, list[str]] = {}
    for _src, what in overrides.provenance:
        elem, _, target = what.partition(" → ")
        # apply_to writes the P_hit of the APKWS into the shared field p_hit_at_min_range.
        target = "p_hit_at_min_range" if target == "modernized_p_hit_max" else target
        elements.setdefault(target, []).append(elem)
    tech = parse_package((sysml_root / "Technical" / "TechnicalComponentsPackage.sysml")
                         .read_text(encoding="utf-8"))
    by_name = {p.name: p for p in tech.part_defs}
    for fld, (part, attrs) in _UAS_ELEMENTS.items():
        spec = getattr(resolved, fld)
        got = {a.name: a for a in by_name[part].attributes}
        model_si = tuple(_to_si(float(got[a].value), got[a].unit or "")[0] for a in attrs)
        expect = (spec.max_speed_mps, spec.dimension_m, spec.ceiling_m)
        if model_si != expect:
            raise AssertionError(f"{fld}: model {model_si} != resolved {expect}")
        elements[fld] = [f"{part}.{a}" for a in attrs]

    rows = []
    for f in fields(MissionConfig):
        name = f.name
        if name in applied:
            origin = "model"
        elif name in raw:
            origin = "scenario"
        else:
            origin = "default"
        rows.append({
            "field": name,
            "origin": origin,
            "model_elements": "; ".join(elements.get(name, [])),
            "read_not_applied": bool(name in elements and name not in applied),
            "in_scenario": name in raw,
            "scenario_equals_resolved": bool(getattr(scenario, name) == getattr(resolved, name)),
            "default_equals_resolved": bool(getattr(default, name) == getattr(resolved, name)),
            "resolved": repr(getattr(resolved, name)),
            # Value the loader extracted from the model (applied or not); None if none.
            "model_value": repr(getattr(overrides, "modernized_p_hit_max"
                                        if name == "p_hit_at_min_range" else name, None)),
        })
    return pd.DataFrame(rows)


def _expr_text(expr: Any) -> str:
    """Return expression of a ``calc def`` in infix notation (for reports)."""
    from cuas_sim.sysml_parser import (
        ConstraintBinary,
        ConstraintLiteral,
        ConstraintRef,
        ConstraintUnary,
    )

    prec = {"+": 1, "-": 1, "*": 2, "/": 2}

    def walk(e: Any, parent: int = 0, right: bool = False) -> str:
        if isinstance(e, ConstraintLiteral):
            return str(e.value)
        if isinstance(e, ConstraintRef):
            return str(e.name)
        if isinstance(e, ConstraintUnary):
            return f"{e.op}{walk(e.operand, 3)}"
        if not isinstance(e, ConstraintBinary):
            return "?"
        p = prec[e.op]
        txt = f"{walk(e.left, p)} {e.op} {walk(e.right, p, True)}"
        need = p < parent or (right and p == parent)
        return f"({txt})" if need else txt

    return walk(expr)



# ==========================================================================
# Requirements
# ==========================================================================
def _limits(req: Any) -> dict[str, float]:
    return {k: float(v) for k, v in req.attributes.items() if isinstance(v, (int, float))}


#: One violation per constraint CLAUSE: (requirement, clause, builder). The injected
#: value comes from the LIMIT declared in the model (x1.1 above a ceiling or an equality;
#: x0.9 below a floor), never typed in.
INJECTIONS: list[tuple[str, str, Callable[[Any, dict[str, float]], Any]]] = [
    ("TR-001", "actualEmploymentRange <= maxEmploymentRange",
     lambda c, L: c.replaced(baseline_envelope_m=(
         c.baseline_envelope_m[0], L["maxEmploymentRange"] * 1.1))),
    ("TR-002", "actualEmploymentRange <= maxEmploymentRange",
     lambda c, L: c.replaced(modernized_envelope_m=(
         c.modernized_envelope_m[0], L["maxEmploymentRange"] * 1.1))),
    ("TR-003", "actualUpdateInterval >= minUpdateInterval",
     lambda c, L: c.replaced(baseline_update_interval_s=(
         L["minUpdateInterval"] * 0.9, c.baseline_update_interval_s[1]))),
    ("TR-003", "actualUpdateInterval <= maxUpdateInterval",
     lambda c, L: c.replaced(baseline_update_interval_s=(
         c.baseline_update_interval_s[0], L["maxUpdateInterval"] * 1.1))),
    ("TR-004", "actualUpdateInterval >= minUpdateInterval",
     lambda c, L: c.replaced(modernized_update_interval_s=(
         L["minUpdateInterval"] * 0.9, c.modernized_update_interval_s[1]))),
    ("TR-004", "actualUpdateInterval <= maxUpdateInterval",
     lambda c, L: c.replaced(modernized_update_interval_s=(
         c.modernized_update_interval_s[0], L["maxUpdateInterval"] * 1.1))),
    ("TR-005", "actualAcquisitionDistance <= maxAcquisitionDistance",
     lambda c, L: c.replaced(baseline_max_acquisition_m=L["maxAcquisitionDistance"] * 1.1)),
    ("TR-005", "actualAcquisitionTime >= minAcquisitionTime",
     lambda c, L: c.replaced(baseline_reaction_time_min_s=L["minAcquisitionTime"] * 0.9)),
    ("TR-006", "actualLaserRangefinderRange <= maxLaserRangefinderRange",
     lambda c, L: c.replaced(modernized_eoir_max_m=L["maxLaserRangefinderRange"] * 1.1)),
    ("TR-006", "actualRangefinderAccuracy <= maxRangefinderAccuracy",
     lambda c, L: c.replaced(modernized_eoir_accuracy_m=L["maxRangefinderAccuracy"] * 1.1)),
    ("TR-010", "actualCruiseSpeed == requiredCruiseSpeed",
     lambda c, L: c.replaced(a29_cruise_speed_mps=L["requiredCruiseSpeed"] * 1.1)),
    ("TR-011", "actualEndurance <= maxEndurance",
     lambda c, L: c.replaced(a29_endurance_s=L["maxEndurance"] * 1.1)),
    ("TR-016", "actualDimensionalPenalty == requiredDimensionalPenalty",
     lambda c, L: c.replaced(dimensional_penalty_mr=L["requiredDimensionalPenalty"] * 1.1)),
    ("TR-021", "actualMultiRotorProbability == actualFixedWingProbability == required",
     lambda c, L: c.replaced(uas_archetype_p_mr=L["requiredProbability"] * 1.1)),
    ("TR-023", "actualPEACommencementTime == requiredPEACommencementTime",
     lambda c, L: c.replaced(t_pea_start_s=L["requiredPEACommencementTime"] * 1.1)),
    ("TR-028", "actualMaxSpeed <= requiredMaxSpeed",
     lambda c, L: c.replaced(uas_mr=replace(c.uas_mr, max_speed_mps=L["requiredMaxSpeed"] * 1.1))),
    ("TR-029", "actualMaxSpeed <= requiredMaxSpeed",
     lambda c, L: c.replaced(uas_fw=replace(c.uas_fw, max_speed_mps=L["requiredMaxSpeed"] * 1.1))),
    ("TR-030", "actualZIDABoundaryDistance == requiredZIDABoundaryDistance",
     lambda c, L: c.replaced(
         zida_boundary_nmi=L["requiredZIDABoundaryDistance"] * 1.1 / 1852.0)),
]

#: Field -> key of the source table (``config_provenance``).
_FIELD_KEY = {"uas_mr.max_speed_mps": "uas_mr", "uas_fw.max_speed_mps": "uas_fw",
              "baseline_envelope_m[1]": "baseline_envelope_m",
              "modernized_envelope_m[1]": "modernized_envelope_m"}


def build_requirements() -> dict[str, Any]:
    from cuas_sim.analysis.experiment import _design_point_to_config
    from cuas_sim.sysml_requirements import (
        _BINDINGS,
        build_traceability_matrix,
        check_requirements,
        load_requirements,
    )

    cfg = _resolved_config()
    reqs = load_requirements(SYSML)
    by_id = {r.req_id: r for r in reqs}
    prov = _config_provenance().set_index("field")

    nominal = check_requirements(cfg, reqs)
    bound = []
    for rid, binding in _BINDINGS.items():
        fields = [f.strip() for f in binding.config_fields.split(",")]
        origins = [str(prov.loc[_FIELD_KEY.get(f, f), "origin"]) for f in fields]
        bound.append({
            "req_id": rid, "name": by_id[rid].name, "limits": _limits(by_id[rid]),
            "actuals": {k: [float(v) for v in fn(cfg)] for k, fn in binding.actuals.items()},
            "config_fields": fields, "field_origins": origins,
            "compares_model_with_model": set(origins) == {"model"},
            "satisfied": rid in nominal.satisfied,
        })

    matrix = [{"req_id": r.req_id, "name": r.name, "package": r.package, "bound": r.bound,
               "config_fields": r.config_fields if r.bound else None}
              for r in build_traceability_matrix(reqs)]
    with_constraint = {r.req_id for r in reqs if r.constraint is not None}

    points = []
    for study in STUDY_DIRS:
        design = _load_design(study)
        for row in design.to_dict("records"):
            pcfg = _design_point_to_config(cfg, row)
            rep = check_requirements(pcfg, reqs)
            points.append({"study": study, "design_point_id": int(row["design_point_id"]),
                           "lambda_per_h": float(row[LAMBDA_COL]),
                           "uas_archetype_p_mr": float(row[PMR_COL]),
                           "violated": sorted(rep.violated)})
    viol_sets = Counter(tuple(p["violated"]) for p in points)

    injected = []
    for rid, clause, build in INJECTIONS:
        bad_cfg = build(cfg, _limits(by_id[rid]))
        rep = check_requirements(bad_cfg, reqs)
        injected.append({"req_id": rid, "clause": clause, "violated": sorted(rep.violated),
                         "detected": rid in rep.violated,
                         "only_target": sorted(rep.violated) == [rid],
                         "detail": rep.details.get(rid, "")})

    return {
        "field_version": FIELD_VERSION,
        "what": ("verificador de requisitos (src/cuas_sim/sysml_requirements.py): limite do "
                 "requirement def em SI x valor da MissionConfig na unidade do limite; "
                 "igualdade com math.isclose(rel 1e-9, abs 1e-9)"),
        "scope": {"requirements_parsed": len(reqs),
                  "by_package": dict(sorted(Counter(r.package for r in reqs).items())),
                  "with_require_constraint": len(with_constraint),
                  "bound_to_config_fields": len(_BINDINGS),
                  "unbound_listed_only": len(reqs) - len(_BINDINGS)},
        "nominal": {"satisfied": len(nominal.satisfied), "violated": sorted(nominal.violated),
                    "bound": bound,
                    "compares_model_with_model": sum(b["compares_model_with_model"]
                                                     for b in bound)},
        "campaign_design_points": {
            "n": len(points),
            "violation_sets": {",".join(k) or "(nenhuma)": v for k, v in sorted(viol_sets.items())},
            "points": points,
            "note": ("o desenho experimental varia p_mr de proposito; TR-021 fixa a "
                     "equiprobabilidade (p_mr = 0,5). Os pontos nunca passaram pelo "
                     "verificador na campanha: este artefato e a primeira aplicacao."),
        },
        "injected_violations": {
            "n": len(injected), "detected": sum(i["detected"] for i in injected),
            "only_target": sum(i["only_target"] for i in injected), "cases": injected,
        },
        "traceability_matrix": matrix,
    }


# ==========================================================================
# Calculations
# ==========================================================================
def _rows(fn: Callable[[Any], float], metric: str, column: str) -> Callable[..., Any]:
    def _m(self: Any, cfg_df: Any, config_name: str) -> list[dict[str, Any]]:
        if cfg_df.empty:
            return [{"config_name": config_name, "aor": "ALL", "metric": metric, "value": 0.0}]
        return [{"config_name": config_name, "aor": "ALL", "metric": metric,
                 "value": float(fn(cfg_df[column]))}]
    return _m


def _mutations() -> list[tuple[str, str, str, str, Any]]:
    """(id, measure, description, method, replacement), each applied on its own."""
    from cuas_sim.metrics.aggregator import MetricsAggregator as A

    o_mos1, o_tef, o_tpea, o_mop2 = (A._compute_mos1, A._engagement_tef, A._t_pea_for_aor,
                                     A._mop2_row)

    def mos1_scale(self: Any, df: Any, cfg: str, total: int, log: Any) -> Any:
        return [{**r, "value": r["value"] * 1.1} for r in o_mos1(self, df, cfg, total, log)]

    def mos1_no_leaks(self: Any, df: Any, cfg: str, total: int, log: Any) -> Any:
        return o_mos1(self, df, cfg, total, SimpleNamespace(leaks=[]))

    def mos2_model(self: Any, eng_df: Any, total: int, log: Any) -> Any:
        b = self._count_leaks(eng_df, "baseline", log)
        m = self._count_leaks(eng_df, "modernized", log)
        return [{"config_name": "modernized_vs_baseline", "aor": "ALL", "metric": "MOS-2",
                 "value": float((b - m) / m if m else 0.0)}]

    def moe1_no_complement(self: Any, ratios: list[float]) -> float:
        return float(sum(ratios) / len(ratios)) if ratios else 0.0

    def moe1_no_clamp(self: Any, ratios: list[float]) -> float:
        return float(1.0 - sum(ratios) / len(ratios)) if ratios else 0.0

    def moe1_fleet_tpea(self: Any, aor: str, config_name: str | None = None) -> float:
        return max(o_tpea(self, a, config_name) for a in self.cfg.aor_to_a29)

    def mop2_mod(self: Any, config_name: str) -> Any:
        if config_name == "baseline":
            return o_mop2(self, config_name)
        return {"config_name": config_name, "aor": "ALL", "metric": "MOP-2",
                "value": float(self.cfg.modernized_eoir_min_m)}

    def mop3_width(self: Any, config_name: str) -> Any:
        env = (self.cfg.baseline_envelope_m if config_name == "baseline"
               else self.cfg.modernized_envelope_m)
        return {"config_name": config_name, "aor": "ALL", "metric": "MOP-3",
                "value": float(env[1] - env[0])}

    return [
        ("M01", "MOS-1", "valor x1,1 (deriva grosseira)", "_compute_mos1", mos1_scale),
        ("M02", "MOS-1", "denominador sem os vazamentos", "_compute_mos1", mos1_no_leaks),
        ("M03", "MOS-2", "trocar pela expressao do modelo (rB-rM)/rM", "_compute_mos2",
         mos2_model),
        ("M04", "MOE-1", "sem o complemento '1 -' (o defeito pre-Spec 025)",
         "_moe1_from_ratios", moe1_no_complement),
        ("M05", "MOE-1", "sem a penalidade (1+E/N) dos evadidos", "_engagement_tef",
         staticmethod(lambda row, penalty: o_tef(row, 1.0))),
        ("M06", "MOE-1", "normalizado pelo T_PEA maximo da frota", "_t_pea_for_aor",
         moe1_fleet_tpea),
        ("M07", "MOE-1", "sem o clamp [0,1]", "_moe1_from_ratios", moe1_no_clamp),
        ("M08", "MOE-2", "media -> mediana", "_compute_moe2",
         _rows(statistics.median, "MOE-2", "moe2_mean_per_engagement")),
        ("M09", "MOE-3", "media -> mediana", "_compute_moe3",
         _rows(statistics.median, "MOE-3", "moe3_p_hit_expected")),
        ("M10", "MOE-3", "media -> soma", "_compute_moe3",
         _rows(sum, "MOE-3", "moe3_p_hit_expected")),
        ("M11", "MOP-1", "contagem x2", "_compute_mop1",
         _rows(lambda s: 2 * s.mean(), "MOP-1", "mop1_heading_update_count")),
        ("M12", "MOP-2", "ramo do A-29N devolve o limite inferior do EO/IR", "_mop2_row",
         mop2_mod),
        ("M13", "MOP-3", "maximo -> largura do envelope", "_mop3_row", mop3_width),
    ]


def build_calculations() -> dict[str, Any]:
    from cuas_sim.metrics.aggregator import MetricsAggregator
    from cuas_sim.sysml_calculations import (
        _ADJUDICATIONS,
        check_calculations,
        evaluate_expr,
        load_calculations,
    )

    calcs = load_calculations(SYSML)
    by = {c.name: c for c in calcs}
    base = check_calculations(calcs)
    if base.failed_guards:
        raise SystemExit(f"the guard already fails without a mutation: {base.failed_guards}")
    table = []
    for name, adj in _ADJUDICATIONS.items():
        c = by[name]
        row: dict[str, Any] = {
            "metric": adj.metric_id, "calc_def": name, "expr": _expr_text(c.expr),
            "verdict": adj.verdict, "aggregator_ref": adj.aggregator_ref,
            "rationale": adj.rationale, "compared": adj.synthetic_inputs is not None,
        }
        if adj.synthetic_inputs is not None and adj.aggregator_eval is not None:
            row.update(synthetic_inputs=adj.synthetic_inputs,
                       model_value=evaluate_expr(c.expr, adj.synthetic_inputs),
                       aggregator_value=adj.aggregator_eval(),
                       passed=name in base.consistent)
        table.append(row)

    muts = []
    for mid, metric, desc, attr, new in _mutations():
        orig = MetricsAggregator.__dict__[attr]
        try:
            setattr(MetricsAggregator, attr, new)
            rep = check_calculations(calcs)
        finally:
            setattr(MetricsAggregator, attr, orig)
        muts.append({"id": mid, "metric": metric, "mutation": desc,
                     "detected": bool(rep.failed_guards),
                     "failed_guards": sorted(_ADJUDICATIONS[n].metric_id
                                             for n in rep.failed_guards)})
    if check_calculations(calcs).failed_guards:
        raise SystemExit("restoration failed: the guard stayed broken")

    return {
        "field_version": FIELD_VERSION,
        "what": ("guarda de calculos (src/cuas_sim/sysml_calculations.py): avalia a expressao "
                 "do calc def e a funcao REAL do agregador sobre as mesmas entradas sinteticas "
                 "e exige igualdade; nos vereditos sem comparacao so registra a adjudicacao"),
        "summary": {
            "calc_defs": len(table),
            "compared_and_passed": sum(1 for r in table if r.get("passed")),
            "compared_full_expression": [r["metric"] for r in table
                                         if r["compared"] and r["metric"] in ("MOS-1", "MOP-2")],
            "compared_outer_aggregation_only": [r["metric"] for r in table if r["compared"]
                                                and r["metric"] not in ("MOS-1", "MOP-2")],
            "adjudicated_not_compared": [f"{r['metric']} ({r['verdict']})" for r in table
                                         if not r["compared"]],
            "mutations_detected": sum(m["detected"] for m in muts),
            "mutations": len(muts),
        },
        "history": {
            "spec_024": ("ee3d69b (2026-06-24): a comparacao calc def x agregador classificou o "
                         "MOE-1 como DIVERGENCE-ESCALATE -- o modelo e as Eq. 5-8 do estudo de "
                         "caso concordavam e a implementacao media a fracao da janela consumida"),
            "spec_025": ("3cfa14c (2026-06-26): implementacao corrigida (complemento e penalidade "
                         "restaurados); veredito passou a CONSISTENT com guarda numerica"),
            "campaign_start": "8291a2b (2026-08-04): desenho da campanha, posterior a correcao",
            "note": "a mutacao M04 reproduz o defeito corrigido; a guarda atual a detecta",
        },
        "calc_defs": table,
        "mutations": muts,
    }


# ==========================================================================
# Phases
# ==========================================================================
def _replica_rows(res: Any, base: dict[str, Any]) -> pd.DataFrame:
    """Datafarm rows of ONE replication, built by the runner's own functions."""
    from cuas_sim.analysis.experiment import _cost_rows, _finalize

    rows = [{**base, "config": str(m["config_name"]), "aor": str(m["aor"]),
             "metric": str(m["metric"]), "value": float(m["value"])}
            for _, m in res.metrics.iterrows()]
    rows.extend(_cost_rows(res, base))
    return _finalize(rows)


def _store_events(run_dir: Path, dest: Path) -> list[str]:
    """Copy the logs with deterministic gzip (mtime 0); return the content per config."""
    dest.mkdir(parents=True, exist_ok=True)
    texts = []
    for config in ("baseline", "modernized"):
        raw = gzip.decompress((run_dir / f"events_{config}.jsonl.gz").read_bytes())
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
            gz.write(raw)
        (dest / f"events_{config}.jsonl.gz").write_bytes(buf.getvalue())
        texts.append(raw.decode("utf-8"))
    return texts


def _inject(evs: list[Any], kind: str, phases: Any) -> list[Any] | None:
    """One violation of type ``kind`` in a real stream; None if the stream does not admit it."""
    from cuas_sim.sysml_state_machine import _STATE_PHASE_BINDING, StateEvent

    idx = phases.phase_index_of_state
    by_name = {p.sysml_name: p.index for p in phases.phases}
    policing, recovery = by_name["AirspacePolicing"], by_name["CAPToFARPReturn"]
    if kind == "out_of_order":
        for i, ev in enumerate(evs):
            if idx.get(ev.new, -1) >= policing:
                back = StateEvent(t_s=ev.t_s, old=ev.new, new="HOLDING_ORBIT")
                return [*evs[: i + 1], back, *evs[i + 1:]]
        return None
    if kind == "missing_phase":
        if max(idx.get(e.new, -1) for e in evs) < recovery:
            return None
        drop = {s for s, ph in _STATE_PHASE_BINDING.items() if ph == "AirspacePolicing"}
        return [e for e in evs if e.new not in drop]
    if kind == "unknown_state":
        return [replace(evs[0], new=INJECTED_STATE), *evs[1:]]
    raise ValueError(kind)


def build_phases(out_dir: Path) -> dict[str, Any]:
    from cuas_sim.analysis.experiment import _design_point_to_config
    from cuas_sim.simulation.run import run_paired
    from cuas_sim.sysml_state_machine import check_conformance, load_events, load_phases

    base_cfg = _resolved_config()
    phases = load_phases(SYSML)
    if not phases.oracle_loaded:
        raise SystemExit("phase oracle not loaded")
    runs_root = out_dir / "phase_audit_runs"
    replicas, neg = [], Counter()
    ext_total: Counter[str] = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        for study in STUDY_DIRS:
            farm = pd.read_parquet(_study_dir(study) / ACCEPTED_DATAFARM[study])
            design = _load_design(study)
            for row in design.to_dict("records"):
                dp = int(row["design_point_id"])
                ref = farm[(farm["design_point_id"] == dp) & (farm["rep"] == SAMPLE_REP)]
                seed = int(ref["seed"].iloc[0])
                cfg = _design_point_to_config(base_cfg, row)
                res = run_paired(cfg, seed=seed, max_arrivals=None, persist=True,
                                 outputs_root=Path(tmp), slug=f"{study}_dp{dp:02d}")
                base = {"design_point_id": dp, "lambda_per_h": float(row[LAMBDA_COL]),
                        "uas_archetype_p_mr": float(row[PMR_COL]), "rep": SAMPLE_REP,
                        "seed": seed}
                got = _replica_rows(res, base).reset_index(drop=True)
                exp = ref.reset_index(drop=True)
                keys = ["config", "aor", "metric"]
                identical = (got[keys].equals(exp[keys])
                             and np.array_equal(got["value"].to_numpy(), exp["value"].to_numpy()))
                run_dir = next(d for d in (Path(tmp) / "runs").iterdir()
                               if d.name.endswith(f"_{study}_dp{dp:02d}"))
                rel = f"{study}/dp{dp:02d}_rep{SAMPLE_REP}"
                texts = _store_events(run_dir, runs_root / rel)
                events = load_events(runs_root / rel)
                per_cfg = {}
                for config in ("baseline", "modernized"):
                    rep = check_conformance(events[config], phases)
                    ext_total.update(rep.extensions_used)
                    per_cfg[config] = {
                        "a29_streams": len(events[config]),
                        "a29_streams_conformant": (len(events[config])
                                                   - len({v.a29_id for v in rep.violations})),
                        "transitions": sum(len(v) for v in events[config].values()),
                        "conformant": rep.conformant,
                        "violations": [f"{v.a29_id} {v.kind}" for v in rep.violations],
                        "extensions_used": dict(sorted(rep.extensions_used.items())),
                    }
                    for a29, evs in events[config].items():
                        for kind in ("out_of_order", "missing_phase", "unknown_state"):
                            mutated = _inject(evs, kind, phases)
                            if mutated is None:
                                neg[f"{kind}:not_applicable"] += 1
                                continue
                            r = check_conformance({a29: mutated}, phases)
                            hit = any(v.kind == kind for v in r.violations)
                            neg[f"{kind}:{'detected' if hit else 'missed'}"] += 1
                replicas.append({
                    "study": study, "design_point_id": dp, "rep": SAMPLE_REP, "seed": seed,
                    "lambda_per_h": base["lambda_per_h"],
                    "uas_archetype_p_mr": base["uas_archetype_p_mr"],
                    "metrics_bit_identical_to_datafarm": bool(identical),
                    "metric_rows_compared": len(exp),
                    "events_dir": f"outputs/verification/phase_audit_runs/{rel}",
                    "events_lines": [t.count("\n") for t in texts],
                    "configs": per_cfg,
                })
                print(f"  {study} dp{dp:02d}: bit-identical={identical} "
                      f"conformant={all(c['conformant'] for c in per_cfg.values())}")
    audited = [r for r in replicas if r["metrics_bit_identical_to_datafarm"]]
    n_streams = sum(c["a29_streams"] for r in audited for c in r["configs"].values())
    return {
        "field_version": FIELD_VERSION,
        "what": ("auditor de fases (src/cuas_sim/sysml_state_machine.py) aplicado a replicas da "
                 "campanha re-executadas com o registro de eventos gravado"),
        "sampling_rule": (f"replica {SAMPLE_REP} de cada um dos 69 pontos de projeto dos "
                          "quatro estudos, declarada antes da execucao"),
        "oracle": [p.sysml_name for p in phases.phases],
        "summary": {
            "replicas": len(replicas),
            "metrics_bit_identical": len(audited),
            "a29_streams_audited": n_streams,
            "streams_conformant": sum(c["a29_streams_conformant"] for r in audited
                                      for c in r["configs"].values()),
            "violations": sum(len(c["violations"]) for r in audited
                              for c in r["configs"].values()),
            "extensions_used_total": dict(sorted(ext_total.items())),
        },
        "injected_violations": dict(sorted(neg.items())),
        "reproduce_one": ("cuas-sim check-conformance --run "
                          "outputs/verification/phase_audit_runs/EX-1/dp00_rep0"),
        "replicas": replicas,
    }


# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("all", "requirements", "calculations", "phases"),
                    default="all")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    out = Path(args.out_dir).resolve()
    logging.disable(logging.INFO)

    if args.part in ("all", "requirements"):
        r = build_requirements()
        _dump(r, out / "sysml_requirements_audit.json")
        inj = r["injected_violations"]
        print(f"[requirements] nominal: {r['nominal']['satisfied']} satisfied, "
              f"{r['nominal']['violated']} violated; model x model: "
              f"{r['nominal']['compares_model_with_model']} of 14")
        print(f"[requirements] campaign points: {r['campaign_design_points']['violation_sets']}")
        print(f"[requirements] injections: {inj['detected']}/{inj['n']} detected, "
              f"{inj['only_target']}/{inj['n']} on the target requirement only")
    if args.part in ("all", "calculations"):
        c = build_calculations()
        _dump(c, out / "sysml_calculations_audit.json")
        s = c["summary"]
        print(f"[calculations] compared and passed: {s['compared_and_passed']}/8; adjudicated: "
              f"{s['adjudicated_not_compared']}; mutations detected: "
              f"{s['mutations_detected']}/{s['mutations']}")
    if args.part in ("all", "phases"):
        p = build_phases(out)
        _dump(p, out / "phase_audit.json")
        s = p["summary"]
        print(f"[phases] replications: {s['replicas']}; bit-identical: {s['metrics_bit_identical']}; "
              f"streams audited: {s['a29_streams_audited']}; conformant: "
              f"{s['streams_conformant']}; violations: {s['violations']}")
        print(f"[phases] injections: {p['injected_violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
