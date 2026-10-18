# Installation and runtime setup

FLE-Gym is a **Python client/wrapper**, not a Factorio server distribution. You must supply a working instance of Factorio Learning Environment (FLE) and the Factorio game. For upstream installation details, refer to the [FLE repository](https://github.com/JackHopkins/factorio-learning-environment). Factorio is proprietary software and must be obtained separately under its terms.

## Recommended environment

The exercised stack used Python 3.12 on WSL2 Ubuntu with Docker Desktop integration, FLE `0.4.3`, and A2A SDK `0.3.25`. Package metadata permits Python >=3.10, but not all combinations have been independently validated. Windows users may run the Python wrapper and FLE CLI from a WSL Ubuntu environment; host TCP RCON ports must be reachable from that environment.

## Install from the repository

Clone an authorized copy of the project or download its source archive, then, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_install.py
python -m pip check
```

For a regular non-editable installation, use `python -m pip install .`. A source checkout is necessary only for developer scripts, test files, and documentation. There is no claim that this package has been published to PyPI.

## Start a server for the selected scenario

Set up Factorio/FLE using upstream instructions. With the compatible `fle` CLI available, a **single-instance** example is:

```bash
fle cluster start -n 1 -s default_lab_scenario
```

For Open Play in a **separate, disposable session**:

```bash
fle cluster start -n 1 -s open_world
```

These commands may replace an existing container or world, depending on your FLE cluster setup. Do **not** run them to switch scenarios while preserving a valuable factory. Verify that a running instance exposes TCP RCON on the host port you intend to use. Earlier single-instance testing used port `27000`; this is an example, not a universal default.

`FactorioLabEnv` expects `default_lab_scenario`; `FactorioOpenPlayEnv` expects `open_world`. The Python wrapper cannot switch the server's scenario.

## Check the API (offline and live)

Offline checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q fle_gym tests
```

The offline tests do **not** prove your Factorio server is reachable. Once a disposable Lab server is running:

```bash
python scripts/smoke_test.py --rcon-port 27000 --steps 10 --seed 42
```

**Warning:** The smoke-test script calls `reset()` with a hard reset, so it will reinitialize the world on the connected Factorio instance.

For manual scenario integration testing, use `scripts/validate_scenarios.py --help`. That script also resets the connected world and can export a native save ZIP.

## Build/install a release archive

```bash
python -m pip install build
python -m build
```

The resulting wheel and sdist are written under `dist/`, which is ignored by Git. To verify a wheel is self-contained, install it into a **fresh virtual environment**, change directory outside the source checkout, and confirm `import fle_gym` resolves to `site-packages`. A wheel build does not bundle Docker, FLE's game server, or Factorio assets.

## Optional sprites

The package does not include FLE visualization assets. If you use upstream FLE rendering that requires sprites, install them through the upstream FLE process (`fle sprites`, where available) or configure `FLE_SPRITES_DIR`. FLE-Gym's current policy observation is a numeric vector; visual sprites are not required for ordinary agent observation.

## Keep experimental worlds safe

- Treat `reset()` as destructive by default (`hard_reset=True`).
- Do not start/replace clusters to inspect a saved world without understanding the server lifecycle.
- Call `save_world()` only after the last intended action because default Lua-storage scrubbing may alter the current world.
- Confirm that save ZIP archives are written to the requested host path and load in the desktop game before clearing any container.

See [Troubleshooting](TROUBLESHOOTING.md) for common failures.
