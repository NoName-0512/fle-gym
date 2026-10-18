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
- [ ] All unit/static tests pass in a fresh environment.
- [x] Full source compilation and `pip check` pass.
- [ ] Build wheel and source distribution; inspect contents for generated saves, logs, hidden credentials, unrelated project files and unexpected binaries.
- [x] Install the built wheel into a clean virtual environment and import it from **outside** the source checkout.
- [ ] Confirm action names/IDs, observation labels/indices, masks and telemetry keys are identical to the validated rc5 contract.
- [x] GitHub Actions passes on the new commit.

## Factorio integration

- [x] Verify a disposable Lab Play server: reset, masks, 20 actions, finite rewards, info mapping and shutdown.
- [ ] Verify a separately configured disposable Open Play server with the same checks.
- [ ] Repeat the controlled fuel/iron-production path if feasible; ensure action errors and machine output are recorded.
- [ ] Confirm 4,096-step time-limit truncation still behaves as expected if any control-flow changes are introduced.
- [x] Confirm native save export, ZIP integrity and desktop-game load when save-related implementation changes occur.
- [ ] Archive representative logs with exact software versions, commit hash, Factorio/FLE scenario and RCON mapping.

## Documentation

- [ ] Install, quickstart, scenario and API examples tested against the new wheel.
- [ ] Action table is regenerated/verified against the installed runtime.
- [ ] Observation table is regenerated/verified against the installed runtime.
- [ ] Reward statements are checked against runtime diagnostics; no unsupported claim of a fully reconciled mathematical reward model.
- [ ] Release notes, version references and validation records match the final commit/tag.
- [x] README clearly distinguishes standalone wrapper from training agents, external FLE runtime and proprietary Factorio game.

## Legal and provenance review — required before public visibility

- [ ] Review each implementation file's origins and track copied/modified third-party code.
- [ ] Preserve original notices and license conditions on upstream MIT-derived material.
- [ ] Verify that the named copyright holder is authorized to license the original contributions and that any employment, institutional or collaboration terms have been reviewed.
- [ ] Confirm the custom personal non-commercial license is consistent with those rights and that upstream FLE MIT grants are not represented as restricted.
- [ ] Confirm Factorio binaries, art and other proprietary assets are **not** packaged.
- [ ] Have a qualified reviewer assess unresolved licensing/ownership questions if needed.

## Publication

- [ ] Keep the repository **private** until the preceding gates pass.
- [ ] Prepare and run the final version bump from `1.0.0rc6` to `1.0.0` **only after** validation; update `pyproject.toml`, `fle_gym/__init__.py`, docs and changelog together.
- [ ] Rebuild final archives from a clean source tree and install the new wheel.
- [ ] Tag the reviewed commit (for example, `v1.0.0`) and attach checksums and release notes if distributing the package.
- [ ] Decide explicitly whether to make the repository public; a public GitHub repository is a distribution, even if labeled non-commercial.

A checked box must reflect a completed test or documented review, not an expected outcome.
