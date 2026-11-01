# Release checklist

Use this checklist when preparing a new FLE-Gym release. Validation results for previous versions are recorded in [Validation](VALIDATION.md).

## Code and packaging

- [ ] Run the unit and static tests in a clean environment.
- [ ] Run source compilation and dependency checks.
- [ ] Build the wheel and source distribution.
- [ ] Inspect both distributions for unintended files, credentials, logs, saves, and unrelated artifacts.
- [ ] Install the wheel in a clean virtual environment and import the public API from outside the source checkout.
- [ ] Confirm continuous integration passes on the release commit.

## Interface compatibility

- [ ] Verify action IDs, ordered action names, and action-space dimensions.
- [ ] Verify ordered observation labels, transformations, and observation-space dimensions.
- [ ] Check action masks, reward diagnostics, simulated-time accounting, and reset behavior.
- [ ] Document intentional interface or behavior changes.

## Factorio integration

- [ ] Test reset, action masks, transitions, finite rewards, and shutdown on disposable Lab Play and Open Play servers.
- [ ] Test controlled machine production and record output and action diagnostics.
- [ ] Verify time-limit truncation when changing horizon or transition logic.
- [ ] Verify native save export, archive integrity, and loading in Factorio when changing save functionality.
- [ ] Record software versions, source commit, server scenarios, and test results.

## Documentation

- [ ] Check installation, quickstart, scenario, and API examples.
- [ ] Confirm action and observation references match the packaged interface.
- [ ] Confirm reward defaults, production weights, and normalization descriptions match the implementation.
- [ ] Update the changelog and validation record for the release.
- [ ] Check documentation links.

## Licensing and distribution

- [ ] Retain applicable copyright, license, and third-party notices.
- [ ] Review provenance and distribution rights for newly added material.
- [ ] Exclude Factorio binaries, game assets, and other material not licensed for redistribution.

## Publication

- [ ] Update package version references consistently.
- [ ] Build final distributions from the reviewed source commit.
- [ ] Install and check the final wheel.
- [ ] Create a version tag pointing to the reviewed commit.
- [ ] Publish release notes identifying the tag, commit, and relevant compatibility changes.
- [ ] Attach distribution archives and checksums when distributing built packages.
