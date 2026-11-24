"""Statistical adherence oracle for Spec 030 — Miranda (2025) v2 generation.

Spec: Spec 030 §5.3.

The Monte Carlo batch itself lives in ``scripts/miranda_adherence_audit.py``,
which is also what produces the citable artefact
``outputs/verification/miranda_adherence_audit.json``. This module **consumes
that same function** and only asserts the tolerance bands, so the published
numbers and the oracle cannot drift apart: if the measurement changes, both the
artefact and this test see the change.

Exercises the **simulator code** (``BorderPolygon`` built from the production
YAML), not a reimplementation. Marker ``statistical`` (run with ``-m
statistical``). Pinned data source: ``scripts/data/setores_zida_27.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from miranda_adherence_audit import N_MC, build_audit

pytestmark = pytest.mark.statistical

_CONFIG_PATH = _REPO_ROOT / "configs" / "zida_shield_default.yaml"
_SYSML_ROOT = _REPO_ROOT / "sysml"


@pytest.fixture(scope="module")
def audit() -> dict:
    """One 200 000-draw batch shared by every assertion in this module."""
    return build_audit(_CONFIG_PATH, _SYSML_ROOT, N_MC)


def test_batch_is_the_published_size(audit: dict) -> None:
    # Guards the N the dissertation cites: every quantity comes from the same
    # 200 000-draw batch, and no draw is silently lost.
    assert audit["sampling"]["n_mc"] == 200_000
    assert sum(r["count"] for r in audit["per_sector"]) == 200_000


def test_sector_1_has_zero_mass(audit: dict) -> None:
    # Property of Miranda's rule, not a defect: mu - 3*sigma = 1, so every
    # accepted u is strictly above F(1) and selects k >= 2.
    assert audit["empirical"]["sector_1_count"] == 0
    assert audit["theoretical"]["p_sector_1"] == 0.0


def test_resort_rate_matches_rejected_mass(audit: dict) -> None:
    assert 0.0020 <= audit["empirical"]["resort_rate"] <= 0.0035
    assert audit["theoretical"]["rejected_mass"] == pytest.approx(0.0027, abs=5e-5)


def test_empirical_pmf_adheres_to_theoretical(audit: dict) -> None:
    assert audit["empirical"]["pmf_max_deviation"] < 0.002


def test_theoretical_pmf_mean_is_half_sector_south(audit: dict) -> None:
    # 14.5 rather than mu = 14.0 — inherent to the (k-1, k] cell convention.
    assert audit["theoretical"]["pmf_mean"] == pytest.approx(14.5, abs=1e-3)


def test_heading_dispersion(audit: dict) -> None:
    assert 14.8 <= audit["empirical"]["heading_sigma_deg"] <= 15.2
    # Untruncated by design (unlike the sector draw): ~0.27 % exceed +-45 deg.
    assert 0.996 <= audit["empirical"]["heading_within_45deg"] <= 0.999


def test_every_origin_lies_on_the_border_polyline(audit: dict) -> None:
    assert audit["empirical"]["max_offset_from_polyline_deg"] < 1e-8


def test_archetype_equiprobable(audit: dict) -> None:
    assert 0.49 <= audit["empirical"]["fraction_multirotor"] <= 0.51


def test_cdf_column_reproduces_the_published_table(audit: dict) -> None:
    # The per-sector block of the artefact is what the dissertation tabulates;
    # these are the anchor rows of that table.
    by_sector = {r["sector"]: r for r in audit["per_sector"]}
    assert by_sector[1]["cdf"] == pytest.approx(0.001349898, abs=1e-9)
    assert by_sector[14]["cdf"] == pytest.approx(0.5, abs=1e-12)
    assert by_sector[27]["cdf"] == pytest.approx(0.998650102, abs=1e-9)
    # Sector ends are the published "Latitude/Longitude final" columns.
    assert by_sector[14]["lat_final"] == pytest.approx(-4.988075, abs=5e-7)
    assert by_sector[14]["lon_final"] == pytest.approx(-72.580691, abs=5e-7)


def test_audit_is_deterministic() -> None:
    # Same seeds => same artefact. Small N keeps this assertion cheap.
    a = build_audit(_CONFIG_PATH, _SYSML_ROOT, 2_000)
    b = build_audit(_CONFIG_PATH, _SYSML_ROOT, 2_000)
    assert a == b
    assert np.isclose(a["parameters"]["sigma"], 26 / 6)
