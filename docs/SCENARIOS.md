# Lab Play and Open Play

FLE-Gym supports two named configurations of the **same** 124-action, 266-observation Gymnasium interface. The scenario is chosen when the **Factorio/FLE server** is started. The Python wrapper records which scenario is expected; it does not start a container, load a map, or verify that the server was launched with a matching scenario.

| Environment | Wrapper class | `make_env` scenario | Expected FLE server scenario |
|---|---|---|---|
| Lab Play | `FactorioLabEnv` | `default_lab_scenario` | `default_lab_scenario` |
| Open Play | `FactorioOpenPlayEnv` | `open_world` | `open_world` |

## Start the matching server

If you use the FLE CLI and Docker integration, examples are:

```bash
fle cluster start -n 1 -s default_lab_scenario
# Alternatively, for a separate testing session:
fle cluster start -n 1 -s open_world
```

**Do not run these consecutively against a running world you want to keep.** Starting a cluster may replace its existing container/world. In the documented validation the scenarios were used **sequentially** on a single container with host RCON port `27000`; this does not imply both scenarios are simultaneously accessible on different ports.

## Construct an environment

```python
from fle_gym import FactorioLabEnv, FactorioOpenPlayEnv, make_env

lab = FactorioLabEnv(rcon_port=27000)  # requires Lab Play server
# open_play = FactorioOpenPlayEnv(rcon_port=27000)  # requires Open Play server
# alternate = make_env(scenario="open_world", rcon_port=27000)
```

`make_env` rejects unrecognized scenario identifiers with `ValueError` *before* attempting an RCON connection. A mismatched **known** scenario and server are not automatically detected, so operators must check the actual server configuration.

## Reset, progression and time limits

- `env.reset()` uses `options={"hard_reset": True}` unless explicitly overridden; this reinitializes the research world. Do not use it to connect non-destructively to a valuable save.
- The default horizon is 4,096 environment decisions. Reaching it yields `truncated=True`; this represents a time limit, **not** successful factory construction or research completion.
- The `scenario` label does not change the action/observation indexing. Observed content and feasible actions reflect the live map and factory.
- Save a world at the end of a disposable run with `save_world`, before closing the environment. Default Lua-storage scrubbing may disrupt continued play.

## Validation status

Both Lab Play and Open Play completed 4,096-step random-valid-action integration runs, produced native save exports, and were loaded in desktop Factorio. These tests establish basic operation and save interoperability, **not agent skill** or complete research-tree automation. See [Validation](VALIDATION.md).
