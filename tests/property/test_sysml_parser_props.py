"""Property-based oracle for the SysML AST parser (Spec 020 §5.2 P1).

Group B (Red phase): MUST FAIL until ``sysml_parser.parse_package`` is
implemented (Spec 020 Group C).

P1 — reformatting insensitivity: semantics-preserving reformatting of the real
``TechnicalComponentsPackage.sysml`` (extra whitespace, blank lines, trailing
``//`` comments, varied indentation) yields an IDENTICAL AST. This is the
concrete robustness win over the regex engine (FR-8).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cuas_sim.sysml_parser import parse_package

_TECH = (
    Path(__file__).parent.parent.parent
    / "sysml" / "Technical" / "TechnicalComponentsPackage.sysml"
)


def _reformat(text: str, rng: random.Random) -> str:
    """Apply semantics-preserving cosmetic mutations line by line."""
    out: list[str] = []
    for line in text.splitlines():
        # Random extra leading indentation (varied brace/attr indentation).
        if rng.random() < 0.3:
            line = (" " * rng.randint(1, 4)) + line
        # Random trailing whitespace.
        if rng.random() < 0.3:
            line = line + (" " * rng.randint(1, 3))
        # Append a line comment AFTER the complete statement (safe — the
        # statement is intact; only the injected junk is commented out).
        if rng.random() < 0.3:
            line = line + "  // injected-" + str(rng.randint(0, 999))
        out.append(line)
        # Random blank line(s) between statements.
        if rng.random() < 0.2:
            out.extend([""] * rng.randint(1, 2))
    return "\n".join(out) + "\n"


@pytest.mark.skipif(not _TECH.exists(), reason="sysml/ not present")
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=200, deadline=None)
def test_reformatting_insensitive(seed: int) -> None:
    original = _TECH.read_text(encoding="utf-8")
    base = parse_package(original)
    mutated = _reformat(original, random.Random(seed))
    assert parse_package(mutated) == base
