# FLE-Gym

**A Gymnasium-compatible interface for reinforcement learning in Factorio, built on the Factorio Learning Environment (FLE).**

This release contains the **Gym-facing FLE/Factorio environment only**. It exposes a stable, discrete decision interface so researchers can connect their own reinforcement-learning or quality-diversity agents to an existing Factorio server. The package provides action masks, numeric observations, reward diagnostics, scenario-specific environment classes, and native Factorio save export. It does **not** bundle a trained agent, train PPO/SAC/MAP-Elites, or manage the Factorio cluster.

**Release status:** `1.0.0` is being prepared for a private GitHub release following rc6 validation. The package passed offline regression checks, live Lab/Open Play integration, native save export, GitHub CI and clean installation. Full factory automation and rc6 iron-plate production have not been demonstrated. Public distribution remains on hold pending licensing and ownership confirmation; see [Validation](docs/VALIDATION.md).

## At a glance

| Property | Current interface |
|---|---|
| Standard | Gymnasium-style `reset` and five-value `step` |
| Action space | `Discrete(124)` high-level actions |
| Observation space | Numeric `Box`, shape `(266,)` |
| Action feasibility | Boolean mask `(124,)` via `env.action_masks()` |
| Scenarios | Lab Play (`default_lab_scenario`), Open Play (`open_world`) |
| Default time limit | 4096 environment decisions (`truncated=True`) |
| Runtime | Existing Factorio/FLE instance over TCP RCON |
| Python | Tested in Python 3.12; metadata allows Python >=3.10 |
| Primary upstream dependency | `factorio-learning-environment==0.4.3` |

Action IDs and observation indices are stable across the two wrapper classes. Feasibility and physical effects depend on the underlying game state; choosing an unmasked action does not guarantee an item is produced.

## Requirements and installation

You need Python, a working Factorio/FLE installation, and access to a running Factorio instance's **host TCP RCON port**. The wrapper does not install the Factorio game, redistribute its assets, launch Docker containers, or change a server's scenario.

From a checked-out repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_install.py
```

The package pins FLE `0.4.3` and A2A SDK `0.3.25` in its dependency metadata. For container startup, scenario selection, Windows/WSL notes, and offline installation checks, see [Installation](docs/INSTALLATION.md).

## Quick start: bring your own policy

Start an existing FLE/Factorio server in **Lab Play**, then use its exposed RCON port (typically `27000` in a single-instance local setup):

```python
import numpy as np
from fle_gym import FactorioLabEnv

rng = np.random.default_rng(42)
env = FactorioLabEnv(rcon_port=27000, seed=42, run_dir="runs/quickstart")

try:
    obs, info = env.reset(seed=42)  # resets the connected research world
    for _ in range(10):
        valid = np.flatnonzero(env.action_masks())
        if not len(valid):
            raise RuntimeError("No feasible actions")
        action = int(rng.choice(valid))  # substitute your own agent here
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.action_name(action), reward, terminated, truncated)
        if terminated or truncated:
            break
finally:
    env.close()
```

This example checks the interface, not learned factory-building performance. `env.reset()` defaults to `hard_reset=True` and **reinitializes the connected world**. Run it on an expendable test server, not a world you intend to preserve.

For Open Play, start the *server* with `open_world` and use `FactorioOpenPlayEnv` (or `make_env(scenario="open_world", ...)`). Changing Python classes alone does **not** switch the underlying server scenario.

## Validate the installation

Offline checks (do not require a Factorio server):

```bash
python -m unittest discover -s tests -v
python -m compileall -q fle_gym tests
python -m pip check
```

Once an expendable Lab Play instance is running:

```bash
python scripts/smoke_test.py --rcon-port 27000 --steps 10 --seed 42
```

The smoke test calls `reset` and therefore resets the connected research world. A 4,096-step validation can be run with `scripts/validate_scenarios.py` after configuring the corresponding server. See [Validation](docs/VALIDATION.md) for what has actually been tested, including the limits of those claims.

## Export a playable Factorio save

While `env` is still open, after its **last** action:

```python
path = env.save_world(
    "my_factory",
    "world_saves/my_factory.zip",
    rcon_port=27000,
)
print(path)
```

This uses Factorio's server save command, copies the ZIP from the selected Docker container, and checks archive integrity. The host requires Docker CLI access; set `container="your_container_name"` if port-based container discovery is ambiguous. Existing output ZIPs are not overwritten.

**Important:** The default `scrub_storage=True` removes unserializable Lua functions from Factorio `storage` before saving. This can affect further play in the same world. Save only at the end of the run, or explicitly use `scrub_storage=False` if your scenario can save safely without that workaround. Saving does not stop or destroy containers.

## Reference documentation

| Guide | Description |
|---|---|
| [Installation](docs/INSTALLATION.md) | Setup, dependencies, server connection and smoke test |
| [API](docs/API.md) | Environment constructors, resets, actions, metrics and save API |
| [Scenarios](docs/SCENARIOS.md) | Lab Play vs Open Play and scenario ownership |
| [Actions](docs/ACTIONS.md) | Ordered 124-action catalogue and action masking |
| [Observations](docs/OBSERVATIONS.md) | Ordered 266-feature schema and normalization caveats |
| [Rewards](docs/REWARDS.md) | Reward shaping, telemetry and interpretability limits |
| [Validation](docs/VALIDATION.md) | Tests completed, evidence and remaining test coverage |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common RCON, Docker, reset and packaging failures |
| [Development](docs/DEVELOPMENT.md) | Contributor checks and internal compatibility rules |
| [Release checklist](docs/RELEASE_CHECKLIST.md) | Gates before a public, final release |
| [Change log](CHANGELOG.md) | Notable package changes |


## License, upstream attribution and provenance

Copyright (c) 2026 Duran Reddy for original FLE-Gym contributions that the author has the right to license. Those contributions are offered under the [FLE-Gym Personal Non-Commercial License](LICENSE). **Commercial use requires prior written permission.** This is a source-available, non-commercial project—not an OSI-approved open-source license.

FLE-Gym depends on the separately maintained [Factorio Learning Environment](https://github.com/JackHopkins/factorio-learning-environment) (FLE); FLE's MIT license and its contributors' copyright remain applicable to FLE material. See [Third-party notices](THIRD_PARTY_NOTICES.md). FLE-Gym's non-commercial terms **do not restrict the upstream MIT rights**. Source provenance, including any third-party or institutional ownership obligations, must be assessed before a public release; see [Release checklist](docs/RELEASE_CHECKLIST.md).

Factorio game executables, game code and assets are not distributed or licensed by this project. Factorio is developed by Wube Software; this project is not affiliated with or endorsed by the Factorio developers or FLE contributors.
