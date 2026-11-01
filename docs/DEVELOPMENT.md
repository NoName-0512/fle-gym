# Development and internal compatibility

FLE-Gym's public interface is `fle_gym.api` plus the offline registry functions. The implementation under `fle_gym/_internal` descends from a validated capstone environment and retains absolute imports between compatibility modules. `fle_gym._bootstrap.activate_internal_path()` inserts the internal package directory into `sys.path` before those imports. These implementation modules/classes are **not** supported user-facing extension points.

## Working locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q fle_gym tests
python -m pip check
```

Build artifacts:

```bash
python -m pip install build
python -m build
```

`world_saves/`, `runs/`, logs, build files and Python caches are ignored by Git. Keep credentials outside the repository and do not commit native saves or experimental checkpoints by default.

## Internal modules

Internal modules implement feasibility checks, factory extensions, machine-production accounting, crafting, and research behavior.

Imports, inherited methods, and their call sites must remain consistent when internal modules change. The supported integration points are the public environment classes and registry functions; private modules and helper methods may change between versions.

## Preserving model compatibility

For every source change affecting action construction or observation generation:

1. Assert `Discrete(124)` and `Box(shape=(266,))`.
2. Confirm the **ordered** action names, not only the count, against a versioned reference.
3. Confirm the **ordered** observation feature names and normalization.
4. Verify masks, reward components, simulated time and reset behavior on disposable Lab/Open Play servers.
5. Confirm distinct inventory/factory/extended telemetry keys are preserved.
6. Build and install an actual wheel in a clean environment; rerun the minimal live smoke test.

The offline test suite covers syntax, package metadata, action count (when runtime dependencies are present), naming consistency and selected regressions. It does not replace Factorio integration testing.

## Pull-request expectations

Keep each behavior change small and testable; do not replace existing experiment data or break world-save export. Mention changes to the 124-action/266-observation contract explicitly. Documentation claims should distinguish measured live behavior from source inspection. Respect the personal non-commercial license and retain independent upstream MIT notices where required.
