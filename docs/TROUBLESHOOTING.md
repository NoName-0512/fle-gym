# Troubleshooting

## `ModuleNotFoundError: fle_gym`

Activate the virtual environment used to install the package, and install from the checked-out repository:

```bash
source .venv/bin/activate
python -m pip install -e .
python scripts/check_install.py
```

Use `python -m pip ...` so the interpreter and installer belong to the same environment. If a wheel was installed for inspection, run the import check **outside** the checkout to avoid importing from the source tree.

## `Connection refused`, RCON timeout, or no Factorio instance

FLE-Gym does not start a Factorio instance. Confirm that the FLE cluster is running and the host TCP port is exposed and reachable from Python/WSL. Check the port mapping with your Docker administration tools and pass the correct `rcon_port` and `address` to the environment. Do not assume that a second scenario is running at `27001` merely because `27000` exists.

## The wrong Lab/Open Play map appears

`FactorioOpenPlayEnv` does not change the server scenario. The Python class and FLE server launch scenario must agree. Recreating a cluster to change scenarios can replace a running world; save and back up valuable worlds first.

## `reset()` removed my existing factory

This is expected by default: FLE-Gym passes `hard_reset=True`. Do not connect a validation script to a world that must remain intact. For non-destructive alternatives, the semantics of the underlying FLE server must be evaluated separately; do not assume `hard_reset=False` is safe for every scenario.

## An action was unmasked but failed

The mask is an estimate from recent runtime state. Game state can change, prerequisites can be stricter than registry metadata, and Factorio can reject actions at execution time. Inspect the `info` dictionary, the action name, and live inventory/entity state. This is not necessarily an API-contract failure.

## `save_world()` cannot find the container or save ZIP

The export requires Docker CLI access. Supply `container="container_name"` when the host RCON port does not uniquely locate a running container. Ensure the container's Factorio saves directory is accessible and writable. ZIP export can fail even after the server acknowledges a save; the API reports an error rather than claiming success. Do not discard the world until the exported archive has passed integrity and load checks.

## Cannot overwrite an existing save

Choose a new save name and host ZIP path; intentional refusal to overwrite protects experiment artifacts.

## A saved world behaves differently afterward

By default the save workaround removes Lua functions from Factorio `storage`. This is a live-state mutation that can affect continuing gameplay. Prefer saving at the end of an episode/run. `scrub_storage=False` avoids that mutation but may not work on scenarios that contain unserializable Lua values.

## Import tests pass but a live episode fails

Offline tests cannot validate a Factorio instance, scenario setup, RCON, game recipes, or entity actions. Run a short masked-action smoke test on a **disposable** world, capture logs and the exact package version/commit, then use the [Validation](VALIDATION.md) checklist for the affected scenario.
