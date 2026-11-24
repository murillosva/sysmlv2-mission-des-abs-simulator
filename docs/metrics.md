# Metrics

The MOS/MOE/MOP measures follow the Continental ZIDA Shield Operation case study. This
document gives what each measure represents and the source-code pointer that computes it.

| Measure | Definition | Implementation |
|---------|------------|----------------|
| MOS-1 | Fraction of incursions neutralized before crossing the 80-nautical-mile boundary of the ZIDA. | `MetricsAggregator._compute_mos1` |
| MOS-2 | Fraction of the baseline configuration's evasions eliminated by the alternative configuration. | `MetricsAggregator._compute_mos2` |
| MOE-1 | Mean engagement-time rate: the higher the value, the smaller the share of the available policing time taken by engagements. | `MetricsAggregator._compute_moe1` |
| MOE-2 | Mean active-tracking rate during pursuit, sampled between the maximum acquisition distance and the maximum employment range. | `MetricsAggregator._compute_moe2` |
| MOE-3 | Mean probability that an engagement neutralizes the UAS, with aborted engagements counting as zero. | `MetricsAggregator._compute_moe3` |
| MOP-1 | Number of heading or geographic-coordinate updates received by the interceptor per engagement. | `MetricsAggregator._compute_mop1` |
| MOP-2 | Maximum acquisition distance of the configuration. | `MetricsAggregator._mop2_row` |
| MOP-3 | Weapon employment envelope, reported as the maximum employment range. | `MetricsAggregator._mop3_row` |

## Equation numbers in the source code

Docstrings and comments cite equations as "Eq. (3.n)" and tables as "Table 3.n" or
"Table A.n": the numbering of Chapter 3 and Appendix A of the master's thesis, which
presents the case study (not yet public, see [`case-study/README.md`](../case-study/README.md)).

| Eq. | Quantity | Implementation |
|---|---|---|
| (3.2) | MOS-1: incursions neutralized within 80 nautical miles of their entry point, over all incursions | `MetricsAggregator._compute_mos1`, `world.boundary.ZIDABoundary` |
| (3.3)–(3.4) | MOS-2 and the evasion count of each configuration | `MetricsAggregator._compute_mos2` |
| (3.5)–(3.6) | MOE-1 and the effective time of each engagement, with the evasion penalty | `MetricsAggregator._compute_moe1` |
| (3.7) | Time available for airspace policing of each aircraft, until it leaves for fuel | `MetricsAggregator._t_pea_for_aor` |
| (3.8)–(3.9) | Update intervals, U(5, 60) s for the baseline and U(1, 10) s for the A-29N | `policies.tracking` (`next_update_interval`, `first_update_interval`) |
| (3.10)–(3.11) | MOE-2: mean tracking probability over a pursuit, then over the engagements | `metrics.records` (per engagement), `MetricsAggregator._compute_moe2` |
| (3.12)–(3.13) | Tracking probability of each configuration | `BaselineTrackingPolicy.p_track`, `ModernizedTrackingPolicy.p_track` |
| (3.14) | Hit probability, with the dimensional penalty | `policies.weapon` (`p_hit_expected`, `p_hit_at`) |
| (3.15)–(3.16) | Neutralization probability of an engagement, and MOE-3 | `engagement.controller.cumulative_neutralization`, `MetricsAggregator._compute_moe3` |
| (3.17) | MOP-1 | `MetricsAggregator._compute_mop1` |
| (3.18)–(3.19) | MOP-2 and MOP-3 | `MetricsAggregator._mop2_row`, `MetricsAggregator._mop3_row` |

Eq. (3.1) belongs to the earlier FARP placement study and is not used by the simulator.
The tables cited are Table 3.2 (nominal timeline, `config._default_a29_schedules`), Table 3.3
(parameters of each configuration, `MissionConfig` defaults), Table A.1 (coordinates of the
main base, CAP points and FARPs) and Table A.4 (cumulative probability of the sectors,
`world.border_polygon.sector_cdf`). The code writes `L_U` for the characteristic length
`c_UAS`, `sigma_s` (`eoir_accuracy_m`) for `a_EO/IR` and `theta_acuity` for `θ_ac`.

One implementation detail has no equation: when an aircraft's departure for fuel is not
observed, which happens only on the legacy fixed-window path, Eq. (3.7) is replaced by the
nominal value R − 2·d(FARP, CAP)/v of `geometry.haversine.t_pea_seconds` (Eq. 8 of the
earlier version below).

### Numbering in earlier records

The SysML model was generated from an earlier version of the case study, with its own
numbering. That numbering appears in texts recorded before the final version and kept
unchanged: the verdicts of the calculation check (`cuas-sim check-calculations` and
`outputs/verification/sysml_calculations_audit.json`) and the pre-registered stopping
criterion of EX-3. The correspondence is:

| Earlier version | Thesis | Note |
|---|---|---|
| Eq. 1 | (3.2) | |
| Eqs. 2–4 | (3.3)–(3.4) | Eq. 2 divided by the modernized evasion rate; the thesis divides by the baseline count, as the code does |
| Eqs. 5–6 | (3.5)–(3.6) | |
| Eqs. 7–8 | (3.7) | Eq. 7 averaged over the fleet the nominal time of each aircraft, given by Eq. 8; the thesis uses each aircraft's own time, until it leaves for fuel |
| Eqs. 9–10 | (3.10)–(3.11) | |
| Eq. 11 | (3.8) | |
| Eqs. 12–13 | (3.12) | Replaced by the angular-acuity form |
| Eqs. 14–15 | (3.9), (3.13) | |
| Eqs. 16–17 | (3.14) | |
| Eq. 18 | Section 3.3.4.2 | Firing philosophy, described in the text |
| Eq. 19 | (3.15)–(3.16) | Extended to engagements with several shots |
| Table 14 | Table 3.2 | |
| Tables 13 and 15–19 | Table 3.3 | |
| Table 20 | Table 3.4 | MOS-2 ranged over [0, 1]; it is (−∞, 1] in the thesis. MOP-1 was an update interval in seconds; in the thesis (Eq. 3.17) it is the number of updates per engagement, as in the code |
