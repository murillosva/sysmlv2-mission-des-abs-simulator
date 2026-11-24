"""Property-based oracles for SR-4b calc-def guard (Spec 024 §5.2).

Group B (Red phase, Article III): MUST FAIL until the arithmetic evaluator and
``check_calculations`` are implemented (Group C).

P1 — the parsed ``+ - * /`` expression evaluates exactly like Python (precedence
     and parentheses preserved).
P2 — ``check_calculations`` is deterministic / pure (same inputs → same report).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cuas_sim.sysml_calculations import (
    _ADJUDICATIONS,
    check_calculations,
    evaluate_expr,
    load_calculations,
)
from cuas_sim.sysml_parser import parse_package

_SYSML_ROOT = Path(__file__).parent.parent.parent / "sysml"

_FINITE = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
_NONZERO = _FINITE.filter(lambda x: abs(x) > 1e-3)


def _expr_of(src: str) -> object:
    text = f"package P {{ calc def K {{ return r : ScalarValues::Real = {src}; }} }}"
    return parse_package(text).calc_defs[0].expr


# ─── P1: arithmetic precedence / parentheses preserved ──────────────────────


@given(a=_FINITE, b=_FINITE, c=_NONZERO)
def test_arithmetic_sane(a: float, b: float, c: float) -> None:
    cases = {
        "a + b * c": a + b * c,
        "(a + b) * c": (a + b) * c,
        "a - b / c": a - b / c,
        "a / c": a / c,
    }
    for src, expected in cases.items():
        got = evaluate_expr(_expr_of(src), {"a": a, "b": b, "c": c})
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-9), src


# ─── P2: check_calculations is deterministic (pure) ─────────────────────────


def test_check_deterministic() -> None:
    calcs = load_calculations(_SYSML_ROOT)
    assert check_calculations(calcs, _ADJUDICATIONS) == check_calculations(calcs, _ADJUDICATIONS)
