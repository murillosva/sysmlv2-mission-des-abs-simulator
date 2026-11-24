# sysml/ — the consumed SysML v2 mission model

This directory holds the SysML v2 model of the Continental ZIDA Shield Operation
that the simulator consumes: 27 packages following the CoSMA methodology (Purpose,
Operational, Functional, Logical and Technical layers, plus cross-cutting packages).
The model is an **input** of this repository; nothing here generates or edits it.

## Provenance

The model was generated in our previous work, the DASC 2026 paper *"Digital Avionics
Operations in Mission Simulation: an AI-assisted MBSE Modeling Approach"* (Szvaticsek,
Marcondes and Loubach, 2026), by the multi-agent pipeline published at
<https://github.com/murillosva/sysmlv2-mission-multi-agent-system>.

- Source snapshot: `models/case-study/sonnet-4.5/final` of that repository, at commit
  `33a85fa`.
- Generating run: `20260423_033801` (recorded in the `run_metadata.json` of that
  repository).

## Differences from the snapshot

Compared without carriage returns, 24 of the 27 files are identical to the snapshot.
Three differ:

| File | Change |
|---|---|
| `CoSMA/CoSMAQuantitiesAndUnitsPackage.sysml` | Header line 7: the attribution "(Szvaticsek, 2026)" is appended to the modification note. |
| `Requirements/FunctionalRequirementsPackage.sysml` | The trailing `%viz` directive (a visualization command of the SysML v2 Jupyter kernel, not part of the textual notation) and the blank line before it are removed. |
| `Technical/TechnicalComponentsPackage.sysml` | One attribute is added to each of the two rocket part definitions, `SBAT70UnguidedRocketLauncherPod` and `APKWSLaserGuidanceKitWithSBAT70Rocket`: `launcherRocketCapacity : ScalarValues::Integer = 49` (rockets per sortie, read by the loader as the rocket budget of each configuration). |

## What the simulator reads

Seven files, through four modules. The other packages are carried for traceability but
are not read.

| File | Module | Use |
|---|---|---|
| `Technical/TechnicalComponentsPackage.sysml` | `sysml_loader.py` | Technical parameters of the two configurations (kinematics, sensors, weapons), applied on top of the scenario file |
| `Requirements/*.sysml` (4 files) | `sysml_requirements.py` | `require constraint` bindings checked by `cuas-sim check-requirements` |
| `Purpose/MissionPhasesPackage.sysml` | `sysml_state_machine.py` | Phase oracle of `cuas-sim check-conformance` |
| `Analysis/CalculationsPackage.sysml` | `sysml_calculations.py` | `calc def` expressions compared by `cuas-sim check-calculations` |

Precedence, increasing: simulator defaults < scenario file (YAML) < model. The model
prevails in the technical fields it supplies. Geography is the exception: the model
declares it (`Purpose/ContextPackage.sysml`), but the loader does not read it, and the
simulator defaults reproduce the declared values.

## Integrity

`MANIFEST.sha256` lists the SHA-256 of each file, computed over the bytes with LF line
endings, as stored in git (`.gitattributes` keeps `*.sysml` in LF on every platform):

```bash
cd sysml
sha256sum -c MANIFEST.sha256
```

## Licence

The three files in `CoSMA/` derive from the CoSMA framework (© 2026 AIRBUS and its
affiliates) and remain under the Mozilla Public License 2.0, with their original notices;
see [`../NOTICE`](../NOTICE). The other 24 files are distributed under the MIT License of
this repository.
