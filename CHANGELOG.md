# Changelog

Entries below record changes and validation for each version. Historical release-candidate notes describe their status at the time.

## [1.0.1] - 1 November 2026

- Prevent reset-created entities and initial milestones from earning agent progression rewards.
- Document all 14 machine-production weights, the effective public production coefficient of 0.35, and simulated-time normalization with a one-second reference.
- Remove obsolete release-preparation wording and provide a reusable release checklist.
- Record a 301-second masked-random rollout of 132 transitions against commit d0158d9, with telemetry consistency checks and native save export.
- Independently load the exported save in Factorio 2.0.77 and advance it for 120 ticks.
- Known diagnostic limitation: 57 transitions had nonzero reward residuals; named reward components do not yet fully explain the scalar reward without the residual.

## [1.0.0] — Private release preparation

- Promote the validated `1.0.0rc6` implementation to version `1.0.0`.
- Preserve the 124-action and 266-observation interface.
- Include Lab Play and Open Play integration, documentation,
  source distribution and native Factorio save support.
- Keep the original-contributions personal non-commercial license
  and independent upstream FLE MIT notices.
- Public distribution remains pending supervisor guidance on
  licensing and applicable ownership requirements.
- Known limitation: full automated factory progression and
  post-refactor iron-plate production are not established.

## [1.0.0rc6] — release preparation (Lab/Open Play integration verified)

### Changed

- Rename private `factorio_openended_env_research_ajweuzp` to `factorio_openended_env_feasibility` and private `factorio_openended_env_research_yqeibtv` to `factorio_openended_env_extended`.
- Rename their internal classes, helper identifiers, and imports to remove random extraction suffixes while preserving the intended action and observation semantics.
- Expand the README, installation/API/development/scenario/validation/troubleshooting documentation and final-release checklist.
- Correct outdated documentation that suggested unknown scenarios were accepted.

### Validation status

- **13** regression tests passed, including an added check for stale internal absolute imports; `compileall` and `pip check` also passed.
- rc6 wheel and source distribution built; installed wheel imported successfully from outside the source tree.
- Live disposable **Open Play** test passed observation/action dimensions, furnace/drill placement and five-coal insertion for both machines; the extended and inventory runtime telemetry keys returned success.
- The rc6 controlled production probe remained **partial**: no iron output was demonstrated because harvesting/transfer actions were masked.
- Lab Play passed a 20-transition live smoke test, native save export, ZIP integrity checking and researcher-confirmed desktop Factorio loading.
- GitHub Actions passed on commit `7d41a4b`; source/wheel contents checks and installation in a fresh virtual environment passed.
- Ordered compatibility checks confirmed all 124 action names and 266 observation labels remained unchanged from rc5.
- Final source provenance, ownership and licensing review remains outstanding.
- No external API compatibility break is intended. Private Python implementation identifiers are **not** a stable supported API.

## [1.0.0rc5] — 18 October 2026

- Adopted the original-contributions FLE-Gym Personal Non-Commercial License and retained separate FLE MIT attribution.
- Nine offline tests passed, dependencies passed `pip check`, and wheel/source distributions built.
- The wheel was installed and imported successfully from `site-packages` outside the source directory.
- Earlier live Lab/Open Play 4,096-decision runs, world exports and a controlled Open Play iron-production sequence were confirmed on the validated implementation.
- Corrected telemetry key collisions and verified fuel-insertion runtime instrumentation.

The rc5 testing evidence does **not** by itself verify the rc6 refactor. See [Validation](docs/VALIDATION.md).
