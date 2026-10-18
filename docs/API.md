# Public Python API

The documented public module is `fle_gym`. All classes are thin wrappers over the internal FLE integration. The internal modules under `fle_gym/_internal` are **not** a stable external API.

## Environment construction

```python
from fle_gym import FactorioEnv, FactorioLabEnv, FactorioOpenPlayEnv, make_env

env = FactorioLabEnv(
    rcon_port=27000,
    address="localhost",
    seed=42,
    run_dir="runs/my_experiment",
    run_name="agent_01",
)
# Equivalent scenario factory:
# env = make_env(scenario="default_lab_scenario", rcon_port=27000)
```

`FactorioOpenPlayEnv` chooses the `open_world` metadata label; `FactorioLabEnv` chooses `default_lab_scenario`. The server must already be running in the corresponding scenario. `make_env` accepts **only** those two scenario strings and rejects others with `ValueError`.

| Constructor argument | Type / default | Meaning |
|---|---|---|
| `rcon_port` | `int`, `27000` | Host TCP port mapped to the FLE Factorio instance |
| `address` | `str`, `"localhost"` | RCON endpoint address |
| `scenario` | `str`, `default_lab_scenario` | Expected server scenario label; **not a server switch** |
| `seed` | `int | None`, `None` | Default reset seed |
| `run_dir` | `str | Path`, `runs/fle_gym` | Runtime log and bookkeeping directory |
| `run_name` | `str`, `external_agent` | Runtime experiment label |
| `semantic_overrides` | `dict | None` | Supported override keys from `environment_contract.py`; unrecognized keys raise `KeyError` |

## Gymnasium contract

```python
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action_index)
env.close()
```

- `reset(seed=None, options=None)` forwards to the underlying server. Unless options override it, `options={"hard_reset": True}` is used. **This is a destructive research-world reset.**
- `action_space`: `Discrete(124)`. The chosen `action_index` must be an integer ID between 0 and 123.
- `observation_space`: numeric `Box`, shape `(266,)`. Names/order are available from `env.observation_schema`.
- `terminated` indicates an internal terminal condition if applicable. `truncated=True` occurs at the default 4,096-decision time limit and is not an achievement signal.
- `info` includes dynamic action/reward/runtime diagnostics. Some fields are derived from periodically refreshed Factorio state; do not assume that every metric is continuously sampled.

## Action helpers

```python
import numpy as np

mask = env.action_masks()        # Boolean numpy array, (124,)
valid_indices = np.flatnonzero(mask)
action = int(valid_indices[0])
name = env.action_name(action)
assert env.action_index(name) == action
specs = env.action_registry     # list[dict], 124 entries
names = env.action_names        # tuple[str, ...]
```

A true mask entry indicates *currently estimated feasibility*, not guaranteed successful execution. Always inspect the resulting `info`, especially during power, material-transfer and research macro-actions. The action catalogue is documented in [Actions](ACTIONS.md).

## Observation and metrics helpers

```python
fields = env.observation_schema  # tuple[str, ...], length 266
metrics = env.get_metrics()      # live metrics; may contact the server
base = env.base_env             # advanced internal adapter; not stable API
```

The observation values often represent transformed measurements, e.g. `log1p(count)`, not raw inventory quantities. `get_metrics()` refreshes several telemetry groups by contacting the running game; it is not an offline property. See [Observations](OBSERVATIONS.md) and [Rewards](REWARDS.md).

## Server save export

```python
saved = env.save_world(
    "my_run",
    "world_saves/my_run.zip",
    container=None,
    rcon_port=27000,
    scrub_storage=True,
)
```

- Must be invoked on an **open** environment after the intended final action.
- Exports a real Factorio save from Docker and checks ZIP integrity.
- Auto-selects a Docker container by the published host TCP port if `container` is omitted; ambiguous/no matches raise an error.
- Raises if the output ZIP already exists; does not overwrite previous experiments.
- Defaults to destructive Lua-storage function scrubbing. Set `scrub_storage=False` only if the scenario is known to save without it.
- Does not stop Docker containers or destroy the underlying world.

## Offline registries

No live RCON connection is needed to inspect the package-level registry functions, although their imports require the relevant Python dependencies:

```python
from fle_gym import (
    actions, action_names, tracked_items, craft_recipes,
    research_targets, trigger_technologies,
)
print(len(actions()), len(action_names()))  # 124, 124
```

The action/observation indices are a machine-learning interface contract. If they change in future releases, trained policies must be checked for compatibility.

## Configuration fidelity

`semantic_overrides` maps supported names to reward, refresh, simulation-time and horizon settings. The default horizon is 4,096 decisions; the current environment contract **rejects horizons below 1**, even though some historical internal code descriptions discuss horizon-free evaluation. Ensure evaluation uses the same action catalogue and relevant reward/observation semantics as training. See [Rewards](REWARDS.md) for the parameter table.
