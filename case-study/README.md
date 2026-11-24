# Case Study: Continental ZIDA Shield Operation

## Status

The mission case study document from which the SysML v2 model was generated is
currently **reserved** and is not included in this repository.

## Reason

The Continental ZIDA Shield Operation is a hypothetical Brazilian Air Force (FAB)
counter-UAS mission scenario developed as part of an ongoing **Master's thesis** at the
Aeronautics Institute of Technology (ITA). Publishing the document prior to thesis
defense would compromise the originality and novelty of the thesis contribution.

## What the paper and this repository provide

The "Case Study" subsection of the paper (Section 3) describes the scenario:

- Mission Engineering purpose: assessing the impact of a new capability, the A-29
  baseline against the modernized A-29N
- Area of operations: a segment of the continental Air Defense Identification Zone
  (ZIDA), a band from the border line to 80 nautical miles inland; crossing it without
  having been neutralized defines illicit evasion
- Friendly force: six A-29 per configuration at air combat patrol stations supplied by
  forward arming and refueling points, an idealized E-99M early-warning aircraft, and
  surface radars
- Incursions: slow multi-rotor or fast fixed-wing UAS
- Configurations: the baseline A-29 is vectored by voice, acquires targets by eye and
  fires unguided rockets; the A-29N receives target coordinates by datalink, tracks with
  an EO/IR sensor that designates by laser, and fires rockets with laser-guidance kits
- Measures: the MOS/MOE/MOP hierarchy of Table 1 of the paper (see the root
  [`README.md`](../README.md))

This repository provides, in turn:

- the SysML v2 model generated from the document, in [`../sysml/`](../sysml/) (provenance
  in [`../sysml/README.md`](../sysml/README.md));
- the scenario file, [`../configs/zida_shield_default.yaml`](../configs/zida_shield_default.yaml),
  with the schedule, the rules for generating incursions and for re-engagement, and the
  nominal values of the experimental factors;
- the modelling assumptions, in [`../docs/assumptions.md`](../docs/assumptions.md).

The thesis presents the case study in Chapter 3 and Appendix A. The equation and
table numbers cited in the source code follow that numbering; the correspondence
with the code is in [`../docs/metrics.md`](../docs/metrics.md).

## Availability

The case study document will be made publicly available under an open license after the
thesis is defended and deposited in the institutional repository. The `CITATION.cff` and
this file will be updated at that time with the thesis DOI and a direct download link.
This is independent of the MODELSWARD 2027 paper (see the root `README.md`).

## Contact

For academic collaboration requests related to the case study, please contact the
corresponding author, Murillo Szvaticsek (szvaticsek@ita.br).
