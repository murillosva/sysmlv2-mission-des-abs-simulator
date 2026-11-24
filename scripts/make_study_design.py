"""Build the ``design.parquet`` of a study, with fixed factors where needed.

``cuas-sim analysis design`` draws every factor by Latin hypercube and cannot fix one,
while ``analysis run`` reads both ``lambda_per_h`` and ``uas_archetype_p_mr`` from the
design. This script covers the designs of the campaign:

* a single point at fixed values (EX-1), where the factor space has dimension zero;
* a Latin hypercube in one factor with the other as a constant column (EX-2, EX-3);
* a Latin hypercube in both factors (EX-4).

It delegates to ``cuas_sim.analysis.doe.create_design`` and ``write_design`` and only adds
the constant columns.

Examples::

    # EX-1: the nominal point, one row, nothing drawn
    python scripts/make_study_design.py \
        --fixed lambda_per_h=6.0,uas_archetype_p_mr=0.5 \
        --design-seed 0 -o outputs/analysis/ex1_mc_nominal/design.parquet

    # EX-2: Latin hypercube in lambda, with p_mr fixed at its nominal value
    python scripts/make_study_design.py --lhs lambda --samples 16 \
        --design-seed 20260806 --fixed p_mr=0.5 \
        -o outputs/analysis/ex2_lambda/design.parquet

Writes ``design.parquet`` and its ``.meta.json`` sidecar (``sampler`` is "fixed",
"LatinHypercube+fixed" or "LatinHypercube", and each constant factor is listed in
``factors`` with ``kind="fixed"``), plus ``design.provenance.json`` with the command line
and the git state. Console output is ASCII.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from cuas_sim.analysis.doe import Factor, create_design, write_design
from cuas_sim.analysis.experiment import factors_from_names

# Columns that ``analysis run`` reads from every design point; the design must carry both,
# even in a one-factor or single-point study.
REQUIRED_COLUMNS = ("lambda_per_h", "uas_archetype_p_mr")

# Aliases accepted by --fixed and --lhs (the presets of the analysis layer).
PRESET_TO_COLUMN = {"lambda": "lambda_per_h", "p_mr": "uas_archetype_p_mr"}


def _resolve_column(name: str) -> str:
    key = name.strip()
    return PRESET_TO_COLUMN.get(key, key)


def _parse_fixed(raw: str | None) -> dict[str, float]:
    """Parse ``--fixed name=value[,name=value]`` into {column: value}."""
    out: dict[str, float] = {}
    if not raw:
        return out
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"--fixed expects name=value, got {item!r}")
        name, _, value = item.partition("=")
        col = _resolve_column(name)
        try:
            out[col] = float(value)
        except ValueError:
            raise ValueError(f"non-numeric value in --fixed: {item!r}") from None
    return out


def _git_provenance() -> dict[str, Any]:
    """HEAD and tree state, recorded in design.provenance.json (best effort)."""
    def _run(*args: str) -> str | None:
        try:
            res = subprocess.run(args, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return res.stdout.strip()

    head = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    src_status = _run("git", "status", "--porcelain", "--", "src")
    return {
        "git_head": head,
        "git_tree_clean": None if status is None else (status == ""),
        "git_src_clean": None if src_status is None else (src_status == ""),
    }


def build_design(
    fixed: dict[str, float],
    lhs_presets: list[str],
    n_samples: int,
    design_seed: int,
) -> tuple[pd.DataFrame, list[Factor], str]:
    """Build the design DataFrame, the Latin-hypercube factors and the sampler label."""
    overlap = sorted(set(fixed) & {f.name for f in factors_from_names(lhs_presets)}) \
        if lhs_presets else []
    if overlap:
        raise ValueError(f"factor in both --lhs and --fixed: {', '.join(overlap)}")

    if lhs_presets:
        factors = factors_from_names(lhs_presets)
        df = create_design(factors, n_samples=n_samples, design_seed=design_seed)
        sampler = "LatinHypercube+fixed" if fixed else "LatinHypercube"
    else:
        factors = []
        if n_samples != 1:
            raise ValueError("without --lhs the design has exactly 1 point (use --samples 1)")
        df = pd.DataFrame({"design_point_id": [0]})
        sampler = "fixed"

    for col, value in fixed.items():
        df[col] = float(value)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "incomplete design: missing columns "
            f"{', '.join(missing)} (analysis run reads both)"
        )

    columns = ["design_point_id", *REQUIRED_COLUMNS]
    extra = [c for c in df.columns if c not in columns]
    return df[[*columns, *extra]], factors, sampler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a design.parquet, with fixed factors where needed."
    )
    ap.add_argument("--fixed", default=None,
                    help="Constant factors: name=value[,name=value] "
                         "(lambda/p_mr or the column name)")
    ap.add_argument("--lhs", default=None,
                    help="Presets drawn by Latin hypercube (e.g. lambda); omit for a fixed design")
    ap.add_argument("--samples", type=int, default=1,
                    help="Number of Latin-hypercube points (must be 1 without --lhs)")
    ap.add_argument("--design-seed", type=int, default=0,
                    help="Seed of the Latin hypercube; inert in a fixed design")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Path of the output design.parquet")
    args = ap.parse_args(argv)

    try:
        fixed = _parse_fixed(args.fixed)
        lhs_presets = [p.strip() for p in args.lhs.split(",") if p.strip()] if args.lhs else []
        df, factors, sampler = build_design(
            fixed, lhs_presets, args.samples, args.design_seed
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    factor_meta: dict[str, Any] = {
        f.name: {"kind": f.kind, "low": f.low, "high": f.high,
                 "options": list(f.options) if f.options else None}
        for f in factors
    }
    for col, value in fixed.items():
        factor_meta[col] = {"kind": "fixed", "low": value, "high": value, "options": None}

    write_design(
        df, args.output, design_seed=args.design_seed, sampler=sampler,
        n_samples=len(df), factor_meta=factor_meta,
    )

    provenance = {
        "generator": "scripts/make_study_design.py",
        "argv": sys.argv[1:] if argv is None else list(argv),
        "sampler": sampler,
        "design_seed": args.design_seed,
        "design_seed_note": (
            "inerte: design fixo nao sorteia" if sampler == "fixed"
            else "seed do LHS (RNG meta-nivel, Article IV)"
        ),
        "fixed": {k: float(v) for k, v in sorted(fixed.items())},
        "lhs_presets": lhs_presets,
        "n_design_points": len(df),
        **_git_provenance(),
    }
    prov_path = args.output.with_name(args.output.stem + ".provenance.json")
    prov_path.write_text(json.dumps(provenance, sort_keys=True, indent=2), encoding="utf-8")

    fixed_txt = ", ".join(f"{k}={v:g}" for k, v in sorted(fixed.items())) or "-"
    lhs_txt = ", ".join(f.name for f in factors) or "-"
    print(f"Design written: {args.output}  (points={len(df)}, sampler={sampler})")
    print(f"  fixed : {fixed_txt}")
    print(f"  lhs   : {lhs_txt} (design_seed={args.design_seed})")
    print(f"  meta  : {args.output.name}.meta.json + {prov_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
