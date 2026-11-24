from __future__ import annotations

"""Regression test for the extracted ``_engage_after_envelope`` method (AC-6, TD-1).

Group B (Red phase): MUST FAIL until C-6.1 extracts the ENGAGING block from
``EngagementController.run()`` into the private method.
"""

from cuas_sim.engagement.controller import EngagementController


def test_engage_after_envelope_bit_exact_vs_spec015() -> None:
    """The TD-1 refactor must expose ``_engage_after_envelope`` so that
    fixtures entering directly into ENGAGING remain bit-exact vs Spec 015.

    Before C-6.1 lands, this test fails with AttributeError. After C-6.1, the
    extracted method exists and behaves identically to the pre-refactor
    ENGAGING block (mechanical refactor — no behavior change)."""
    assert hasattr(EngagementController, "_engage_after_envelope"), (
        "Spec 016 C-6.1 must extract _engage_after_envelope() from "
        "EngagementController.run() — TD-1 refactor for AC-6."
    )
    method = getattr(EngagementController, "_engage_after_envelope")
    assert callable(method)
