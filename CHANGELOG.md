# Changelog

All dates and status claims below distinguish completed rc5 validation from changes being prepared for rc6.

## [1.0.0rc6] — release preparation (partial live verification complete)

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
- A separate rc6 **Lab Play** integration check, CI and final provenance review remain outstanding.
- No external API compatibility break is intended. Private Python implementation identifiers are **not** a stable supported API.

## [1.0.0rc5] — 18 October 2026

- Adopted the original-contributions FLE-Gym Personal Non-Commercial License and retained separate FLE MIT attribution.
- Nine offline tests passed, dependencies passed `pip check`, and wheel/source distributions built.
- The wheel was installed and imported successfully from `site-packages` outside the source directory.
- Earlier live Lab/Open Play 4,096-decision runs, world exports and a controlled Open Play iron-production sequence were confirmed on the validated implementation.
- Corrected telemetry key collisions and verified fuel-insertion runtime instrumentation.

The rc5 testing evidence does **not** by itself verify the rc6 refactor. See [Validation](docs/VALIDATION.md).
