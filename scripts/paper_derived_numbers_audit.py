"""Audit of the numbers of the paper that are derived from published artifacts.

Two parts, each written as a byte-deterministic artifact in ``outputs/verification/``:

* ``surface``: quantities of the bilinear surface of EX-4 printed in Section 4.2 and
  Figure 2 of the paper: the slope in the mix at the realized ends of lambda (2.364 and
  17.571), the declines (47% and 40.3% [26.2%; 51.6%]), the size of the cross term
  (-0.2247) and the slopes by tercile of intensity (+0.45, +0.37, +0.29).
  ``surface_r384.json`` reports the slope only at lambda = 2, 6, 12 and 18.
* ``reformat``: the test of Section 3.2. Reformatting only the whitespace of
  ``TechnicalComponentsPackage`` (closing each ``part def`` on the line of its last member)
  makes the regular-expression extractor recognize 1 definition instead of 22, while the
  syntax tree stays identical.

Guards (exit code 1 if any fails; the artifact is written anyway):

* ``surface``: the refitted surface must reproduce ``surface_r384.json`` exactly (b0 to
  b3, interval of b3, slope and interval at lambda = 6). The intervals come from 9,999
  resamples, so equality shows that the bootstrap draw is the one of
  ``scripts/surface_model.py``, and that the new intervals come from the same procedure.
  The points of Figure 2 (``_tercile_points``, the tercile rule of the published figure)
  must equal the paired means.
* ``reformat``: the reformatting may change only whitespace.
* both: each number is compared with the value printed in the paper, at the printed
  precision (``paper_claims``); a divergence fails.

The code that produced the numbers is reused, not copied: ``surface_model.point_table``
and ``_design_matrix`` for the fit, and the simulator's own extractors (``_PART_DEF``,
``_legacy_regex_extract``, ``parse_package``, ``_attrs_from_ast``, ``_dispatch_part_def``).

Input data: the ``surface`` part reads
``outputs/analysis/ex4_interaction/datafarm_r384.parquet`` and records its SHA-256, so that
anyone who reruns EX-4 can check that the input is the same. The ``reformat`` part depends
only on the model in ``sysml/``. It runs no simulation and changes nothing under ``src/``.

Usage::

    python scripts/paper_derived_numbers_audit.py                 # both parts
    python scripts/paper_derived_numbers_audit.py --part surface
    python scripts/paper_derived_numbers_audit.py --out-dir <dir> # another destination

Console output is ASCII.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surface_model as sm

from cuas_sim.sysml_loader import (
    _PART_DEF,
    SysMLLoader,
    SysMLOverrides,
    _attrs_from_ast,
)
from cuas_sim.sysml_parser import parse_package

FIELD_VERSION = "1.0"
REPO = Path(__file__).resolve().parent.parent
SYSML_ROOT = REPO / "sysml"
OUT_DIR = REPO / "outputs" / "verification"
EX4 = REPO / "outputs" / "analysis" / "ex4_interaction"
DATAFARM = EX4 / "datafarm_r384.parquet"
SURFACE = EX4 / "surface_r384.json"
RESPONSE, AOR = "MOS-1", "ALL"
NOMINAL_LAMBDA = 6.0
TECH_REL = "sysml/Technical/TechnicalComponentsPackage.sysml"
TECH = REPO / TECH_REL
REFORMATTED_REL = "reformat_runs/TechnicalComponentsPackage_reformatted.sysml"



def _tercile_points() -> pd.DataFrame:
    """Mean ΔMOS-1 per design point of EX-4, with the tercile of lambda.

    The grouping rule of the published Figure 2: cuts at the 1/3 and 2/3 quantiles of
    lambda (<=). Read directly from the datafarm.
    """
    df = pd.read_parquet(DATAFARM)
    d = df[(df["config"] == "delta") & (df["metric"] == RESPONSE) & (df["aor"] == AOR)]
    g = (d.groupby("design_point_id")
           .agg(lam=("lambda_per_h", "first"),
                pmr=("uas_archetype_p_mr", "first"),
                delta=("value", "mean"))
           .reset_index())
    assert len(g) == 32, f"expected 32 design points, found {len(g)}"
    q1, q2 = g["lam"].quantile([1 / 3, 2 / 3])
    g["t"] = g["lam"].map(lambda v: 0 if v <= q1 else (1 if v <= q2 else 2))
    return g

def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the same bytes on every OS (no CRLF translation on Windows).
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def _claim(section: str, text: str, printed: str, value: float | int | bool,
           fmt: str) -> dict[str, Any]:
    """A number printed in the paper against its recomputed value, at the printed precision."""
    shown = format(value, fmt) if not isinstance(value, bool) else str(value).lower()
    return {"section": section, "text": text, "printed": printed, "value": value,
            "format": fmt, "recomputed_as_printed": shown, "matches": shown == printed}


# ==========================================================================
# surface -- quantities derived from the bilinear surface of EX-4
# ==========================================================================

def _percentile_ci(col: np.ndarray) -> tuple[float, float]:
    """The interval of ``surface_model.fit_surface._ci`` (2.5/97.5 percentiles)."""
    v = col[np.isfinite(col)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def _with_ci(estimate: float, boot: np.ndarray) -> dict[str, float]:
    lo, hi = _percentile_ci(boot)
    return {"estimate": float(estimate), "ci_low": lo, "ci_high": hi}


def audit_surface() -> tuple[dict[str, Any], bool]:
    stored = json.loads(SURFACE.read_text(encoding="utf-8"))
    seed = int(stored["inputs"]["bootstrap_seed"])
    n_resamples = int(stored["inputs"]["n_resamples"])
    bil = stored["surface_bilinear"]

    pts = sm.point_table(pd.read_parquet(DATAFARM), RESPONSE, AOR)
    lam = pts[sm.LAMBDA_COL].to_numpy(float)
    pmr = pts[sm.PMR_COL].to_numpy(float)
    y = pts["mean"].to_numpy(float)
    n = lam.size
    beta, *_ = np.linalg.lstsq(sm._design_matrix(lam, pmr, quadratic=False), y, rcond=None)

    # The same draw as surface_model.fit_surface: default_rng(seed).integers(0, n,
    # (n_resamples, n)); rank-deficient resamples are discarded.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = np.full((n_resamples, 4), np.nan)
    for i in range(n_resamples):
        j = idx[i]
        xb = sm._design_matrix(lam[j], pmr[j], quadratic=False)
        if np.linalg.matrix_rank(xb) < 4:
            continue
        boot[i], *_ = np.linalg.lstsq(xb, y[j], rcond=None)
    ok = np.isfinite(boot[:, 0])
    b = boot[ok]

    lmin, lmax = float(lam.min()), float(lam.max())
    pmin, pmax = float(pmr.min()), float(pmr.max())

    def slope(coef: np.ndarray, at: float) -> np.ndarray:
        return coef[..., 2] + coef[..., 3] * at

    s_min, s_nom, s_max = (slope(beta, v) for v in (lmin, NOMINAL_LAMBDA, lmax))
    bs_min, bs_nom, bs_max = (slope(b, v) for v in (lmin, NOMINAL_LAMBDA, lmax))
    span = (lmax - lmin) * (pmax - pmin)

    # -- guards: exact reproduction of the recorded artifact ---------------------
    names = ["intercept", "lambda", "p_mr", "lambda_x_p_mr"]
    est_diff = max(abs(float(beta[c]) - bil["coefficients"][nm]["estimate"])
                   for c, nm in enumerate(names))
    b3_ci = _percentile_ci(b[:, 3])
    nom_ci = _percentile_ci(bs_nom)
    st_b3 = bil["coefficients"]["lambda_x_p_mr"]
    st_nom = bil["mix_slope_by_lambda"]["6"]
    g_pts = _tercile_points().set_index("design_point_id").sort_index()
    pts_diff = float(np.max(np.abs(g_pts["delta"].to_numpy() - y)))
    guards = {
        "coefficients_equal_stored": {
            "passed": est_diff == 0.0, "max_abs_diff": est_diff},
        "b3_ci_equal_stored": {
            "passed": b3_ci == (st_b3["ci_low"], st_b3["ci_high"]),
            "recomputed": list(b3_ci), "stored": [st_b3["ci_low"], st_b3["ci_high"]]},
        "mix_slope_at_6_equal_stored": {
            "passed": (float(s_nom) == st_nom["estimate"]
                       and nom_ci == (st_nom["ci_low"], st_nom["ci_high"])),
            "recomputed": [float(s_nom), *nom_ci],
            "stored": [st_nom["estimate"], st_nom["ci_low"], st_nom["ci_high"]]},
        "figure2_points_equal_paired_means": {
            "passed": pts_diff <= 1e-12, "max_abs_diff": pts_diff},
        "resamples_used": {"passed": int(ok.sum()) == bil["bootstrap"]["n_resamples_used"],
                           "recomputed": int(ok.sum()),
                           "stored": bil["bootstrap"]["n_resamples_used"]},
    }

    # -- terciles of Figure 2 (rule of _tercile_points) -------------------------
    terciles = []
    for t in (0, 1, 2):
        sub = g_pts[g_pts["t"] == t]
        lam_rep = float(sub["lam"].mean())
        terciles.append({
            "tercile": t, "n_points": len(sub), "mean_lambda_per_h": lam_rep,
            "lambda_range": [float(sub["lam"].min()), float(sub["lam"].max())],
            "mix_slope": _with_ci(float(slope(beta, lam_rep)), slope(b, lam_rep)),
        })

    decline_nom = 1.0 - s_max / s_nom
    decline_full = 1.0 - s_max / s_min
    derived = {
        "realized_lambda": [lmin, lmax],
        "realized_p_mr": [pmin, pmax],
        "mix_slope_at_lambda_min": _with_ci(float(s_min), bs_min),
        "mix_slope_at_lambda_nominal_6": _with_ci(float(s_nom), bs_nom),
        "mix_slope_at_lambda_max": _with_ci(float(s_max), bs_max),
        "decline_pct_nominal_to_lambda_max": _with_ci(
            100.0 * float(decline_nom), 100.0 * (1.0 - bs_max / bs_nom)),
        "decline_pct_lambda_min_to_lambda_max": _with_ci(
            100.0 * float(decline_full), 100.0 * (1.0 - bs_max / bs_min)),
        "cross_term_over_realized_ranges": _with_ci(float(beta[3]) * span, b[:, 3] * span),
        "figure2_terciles": terciles,
    }

    m = derived
    tl = terciles
    claims = [
        _claim("4.2", "slope in pmr falls 47% across the realized intensity",
               "47", m["decline_pct_lambda_min_to_lambda_max"]["estimate"], ".0f"),
        _claim("4.2", "... from +0.4864", "+0.4864",
               m["mix_slope_at_lambda_min"]["estimate"], "+.4f"),
        _claim("4.2", "... to +0.2579", "+0.2579",
               m["mix_slope_at_lambda_max"]["estimate"], "+.4f"),
        _claim("4.2", "the cross term amounts to -0.2247 of MOS-1", "-0.2247",
               m["cross_term_over_realized_ranges"]["estimate"], "+.4f"),
        _claim("4.2/4.4", "the largest intensity drawn, 17.571 UAS/h", "17.571", lmax, ".3f"),
        _claim("4.4", "intensities drawn spanning [2.364; ...]", "2.364", lmin, ".3f"),
        _claim("4.2", "the slope falls from +0.4318", "+0.4318",
               m["mix_slope_at_lambda_nominal_6"]["estimate"], "+.4f"),
        _claim("4.2", "... by 40.3%", "40.3",
               m["decline_pct_nominal_to_lambda_max"]["estimate"], ".1f"),
        _claim("4.2", "... [26.2%; ...]", "26.2",
               m["decline_pct_nominal_to_lambda_max"]["ci_low"], ".1f"),
        _claim("4.2", "... [...; 51.6%]", "51.6",
               m["decline_pct_nominal_to_lambda_max"]["ci_high"], ".1f"),
    ]
    for t, lam_p, slope_p in zip(tl, ("4.8", "10.0", "15.3"), ("+0.45", "+0.37", "+0.29"),
                                 strict=True):
        claims.append(_claim("Fig. 2", f"legend lambda ~ {lam_p} UAS/h", lam_p,
                             t["mean_lambda_per_h"], ".1f"))
        claims.append(_claim("4.2", f"tercile slope {slope_p}", slope_p,
                             t["mix_slope"]["estimate"], "+.2f"))

    passed = all(g["passed"] for g in guards.values()) and all(c["matches"] for c in claims)
    out = {
        "field_version": FIELD_VERSION,
        "what": ("grandezas da superficie bilinear do EX-4 citadas no paper e na Figura 2 "
                 "mas ausentes de surface_r384.json (que so traz lambda = 2/6/12/18)"),
        "inputs": {
            "datafarm": _rel(DATAFARM), "datafarm_sha256": _sha256(DATAFARM),
            "datafarm_in_git": False,
            "surface": _rel(SURFACE), "surface_sha256": _sha256(SURFACE),
            "response": RESPONSE, "aor": AOR,
            "bootstrap_seed": seed, "n_resamples": n_resamples,
        },
        "method": {
            "model": "bilinear [1, L, p, L*p] por minimos quadrados sobre as 32 medias "
                     "de Delta MOS-1 por ponto de projeto (surface_model.point_table)",
            "ci": "bootstrap percentil 2,5/97,5 reamostrando PONTOS DE PROJETO; o mesmo "
                  "sorteio de surface_model.fit_surface (provado pelas guardas)",
            "mix_slope": "b2 + b3 * lambda",
            "decline_pct": "100 * (1 - slope(lambda_max) / slope(lambda_ref)), por reamostra",
            "cross_term": "b3 * (lambda_max - lambda_min) * (p_mr_max - p_mr_min), "
                          "faixas realizadas",
            "terciles": "_tercile_points: corte nos quantis 1/3 e 2/3 "
                        "de lambda (<=); a reta de cada tercil e avaliada na media de "
                        "lambda do tercil -- a mesma regra da Figura 2 publicada",
        },
        "guards": guards,
        "derived": derived,
        "paper_claims": claims,
        "passed": passed,
    }
    return out, passed


# ==========================================================================
# reformat -- robustness of the extractor to reformatting (Section 3.2)
# ==========================================================================

def close_on_last_member(text: str) -> str:
    """Close each ``part def`` on the line of its last member (only whitespace changes).

    Removes the whitespace between the last member and the brace that closes the
    definition and puts one space before it. Nested braces are respected by counting.
    """
    out: list[str] = []
    i = 0
    for m in re.finditer(r"part\s+def\s+\w+[^{;]*\{", text):
        if m.start() < i:
            continue
        depth, j = 1, m.end()
        while depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        out.append(text[i:m.end()] + text[m.end():j - 1].rstrip() + " }")
        i = j
    out.append(text[i:])
    return "".join(out)


def _ast_overrides(text: str) -> SysMLOverrides:
    """Primary path of the loader (the syntax tree), on an in-memory text."""
    ov = SysMLOverrides()
    loader = SysMLLoader(SYSML_ROOT)
    for pd_ in parse_package(text).part_defs:
        attrs, strs, acuity = _attrs_from_ast(pd_)
        loader._dispatch_part_def(pd_.name, attrs, strs, acuity, ov, TECH_REL)
    return ov


def _regex_overrides(text: str) -> SysMLOverrides:
    """Earlier regular-expression extractor (the one the syntax tree replaced)."""
    ov = SysMLOverrides()
    SysMLLoader(SYSML_ROOT)._legacy_regex_extract(text, ov, TECH_REL)
    return ov


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if dataclasses.is_dataclass(v):
        return _jsonable(dataclasses.asdict(v))
    return v


def _supplied(ov: SysMLOverrides) -> dict[str, Any]:
    """Fields filled by the extractor (except ``provenance``), in JSON form."""
    out = {}
    for f in dataclasses.fields(ov):
        v = getattr(ov, f.name)
        if f.name != "provenance" and v not in (None, {}, ()):
            out[f.name] = _jsonable(v)
    return out


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def audit_reformat(out_dir: Path) -> tuple[dict[str, Any], bool]:
    text = TECH.read_text(encoding="utf-8")
    ref = close_on_last_member(text)
    ws_only = re.sub(r"\s+", "", ref) == re.sub(r"\s+", "", text)

    ref_path = out_dir / REFORMATTED_REL
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    with ref_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(ref)

    regex_names = {k: [m.group("name") for m in _PART_DEF.finditer(t)]
                   for k, t in (("original", text), ("reformatted", ref))}
    ast = {k: parse_package(t) for k, t in (("original", text), ("reformatted", ref))}

    cap = _Capture()
    root_logger = logging.getLogger()
    root_logger.addHandler(cap)
    try:
        ov = {
            "ast_original": _ast_overrides(text),
            "ast_reformatted": _ast_overrides(ref),
            "regex_original": _regex_overrides(text),
        }
        n_before = len(cap.records)
        ov["regex_reformatted"] = _regex_overrides(ref)
        warnings_regex_reformatted = cap.records[n_before:]
    finally:
        root_logger.removeHandler(cap)

    sup = {k: _supplied(v) for k, v in ov.items()}
    base = sup["regex_original"]
    after = sup["regex_reformatted"]
    field_status = {}
    for name in sorted(set(base) | set(after)):
        if name not in after:
            status = "lost"
        elif name not in base:
            status = "spurious"
        elif after[name] != base[name]:
            status = "changed"
        else:
            status = "kept"
        field_status[name] = {"status": status, "original": base.get(name),
                              "reformatted": after.get(name)}
    counts = {s: sum(1 for v in field_status.values() if v["status"] == s)
              for s in ("kept", "lost", "changed", "spurious")}

    guards = {
        "whitespace_only": {"passed": ws_only},
        "ast_and_regex_agree_on_original": {
            "passed": sup["ast_original"] == sup["regex_original"]},
    }
    claims = [
        _claim("3.2", "a regular-expression extractor recognize one definition ...", "1",
               len(regex_names["reformatted"]), "d"),
        _claim("3.2", "... instead of 22", "22", len(regex_names["original"]), "d"),
        _claim("3.2", "silently (no warning logged by the extractor)", "0",
               len(warnings_regex_reformatted), "d"),
        _claim("3.2", "The syntax tree, by contrast, stayed identical", "true",
               ast["original"] == ast["reformatted"], ""),
        _claim("3.2", "TechnicalComponentsPackage supplies parameters through its 22 part def",
               "22", len(ast["original"].part_defs), "d"),
    ]
    passed = all(g["passed"] for g in guards.values()) and all(c["matches"] for c in claims)
    out = {
        "field_version": FIELD_VERSION,
        "what": ("robustez do extrator a reformatacao: o mesmo pacote com SO os espacos "
                 "mudados, lido pelo extrator legado por expressao regular e pelo parser"),
        "inputs": {"package": TECH_REL, "package_sha256": _sha256(TECH),
                   "reformatted": REFORMATTED_REL,
                   "reformatted_sha256": _sha256(ref_path)},
        "transformation": ("fechar cada part def na linha do ultimo membro: remover o "
                           "branco entre o ultimo membro e a chave de fechamento e por um "
                           "espaco antes dela"),
        "why_the_regex_fails": ("o padrao _PART_DEF termina em ^\\s*\\} (chave no inicio de "
                                "linha, re.MULTILINE); sem chave de definicao no inicio de "
                                "linha, o primeiro casamento so termina na chave que fecha "
                                "o PACOTE e engole as outras 21 definicoes"),
        "guards": guards,
        "regex_part_defs": {"original": len(regex_names["original"]),
                            "reformatted": len(regex_names["reformatted"]),
                            "reformatted_names": regex_names["reformatted"]},
        "ast_part_defs": {"original": len(ast["original"].part_defs),
                          "reformatted": len(ast["reformatted"].part_defs),
                          "structurally_equal": ast["original"] == ast["reformatted"]},
        "extraction": {
            "ast_supplied_fields_original": len(sup["ast_original"]),
            "ast_supplied_fields_reformatted": len(sup["ast_reformatted"]),
            "ast_extraction_identical": sup["ast_original"] == sup["ast_reformatted"],
            "regex_supplied_fields_original": len(base),
            "regex_supplied_fields_reformatted": len(after),
            "regex_field_status_counts": counts,
            "regex_field_status": field_status,
            "warnings_logged_by_regex_on_reformatted": warnings_regex_reformatted,
        },
        "paper_claims": claims,
        "passed": passed,
    }
    return out, passed


# ==========================================================================

def _print_claims(title: str, out: dict[str, Any]) -> None:
    print(f"== {title}: {'OK' if out['passed'] else 'FAILED'}")
    for name, g in out["guards"].items():
        print(f"   guard {name:<36} {'ok' if g['passed'] else 'FAILED'}")
    for c in out["paper_claims"]:
        mark = "ok" if c["matches"] else "DIVERGE"
        print(f"   [{c['section']:>7}] printed {c['printed']:>8}  recomputed "
              f"{c['recomputed_as_printed']:>8}  {mark}  ({c['text']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("surface", "reformat", "all"), default="all")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    ok = True
    if args.part in ("surface", "all"):
        out, passed = audit_surface()
        _dump(out, args.out_dir / "surface_derived_audit.json")
        _print_claims("surface", out)
        d = out["derived"]
        print(f"   decline lambda=6 -> {d['realized_lambda'][1]:.3f}: "
              f"{d['decline_pct_nominal_to_lambda_max']['estimate']:.2f}% "
              f"[{d['decline_pct_nominal_to_lambda_max']['ci_low']:.2f}; "
              f"{d['decline_pct_nominal_to_lambda_max']['ci_high']:.2f}]")
        ok &= passed
    if args.part in ("reformat", "all"):
        out, passed = audit_reformat(args.out_dir)
        _dump(out, args.out_dir / "reformat_audit.json")
        _print_claims("reformat", out)
        c = out["extraction"]["regex_field_status_counts"]
        print(f"   regex fields after reformatting: {c['kept']} kept, {c['lost']} "
              f"lost, {c['changed']} changed, {c['spurious']} spurious")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
