# Final release checklist

**Current status:** version `1.0.0` is being prepared for a private GitHub release; public distribution remains pending licensing and ownership confirmation. This checklist lists conditions to complete before changing repository visibility or publishing an archive.

## rc6 evidence already collected (18 October 2026)

- 13 offline regression tests passed; Python compilation and dependency checks passed.
- rc6 wheel/sdist built, and the installed wheel imported from `site-packages` outside the checkout; a subsequent installation and API import also passed in a **fresh** virtual environment.
- Open Play four-action construction/fuel telemetry regression passed, with 266 observations and 124 actions confirmed.
- Controlled iron-production probe was partial, with no measured new output; Lab Play rc6 and GitHub Actions subsequently passed.
- This evidence does not waive any unchecked packaging, documentation, licensing or long-run validation gate below.

## Code and packaging

- [x] Apply the internal-module renames with no references to old random suffixes.
- [x] All unit/static tests pass in a fresh environment.
- [x] Full source compilation and `pip check` pass.
- [x] Build wheel and source distribution; inspect contents for generated saves, logs, hidden credentials, unrelated project files and unexpected binaries.
- [x] Install the built wheel into a clean virtual environment and import it from **outside** the source checkout.
- [x] Confirm action names/IDs, observation labels/indices, masks and telemetry keys are identical to the validated rc5 contract.
- [x] GitHub Actions passes on the new commit.

## Factorio integration

- [x] Verify a disposable Lab Play server: reset, masks, 20 actions, finite rewards, info mapping and shutdown.
- [x] Verify a separately configured disposable Open Play server with the same checks.
- [x] Repeat the controlled fuel/iron-production path if feasible; ensure action errors and machine output are recorded.
- [x] Confirm 4,096-step time-limit truncation still behaves as expected if any control-flow changes are introduced.
- [x] Confirm native save export, ZIP integrity and desktop-game load when save-related implementation changes occur.
- [x] Archive representative logs with exact software versions, commit hash, Factorio/FLE scenario and RCON mapping.
