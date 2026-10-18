# FLE-Gym — standalone Gymnasium wrapper

This release contains the **Gym-facing FLE/Factorio environment only**.

## Prerequisites

- A Factorio game server must already be running with an available TCP RCON port (e.g. `27000`).
- Docker/Factorio runtime setup is separate from this Python package and is **not** bundled.
- Python dependencies install automatically; validated with FLE 0.4.3 and A2A SDK 0.3.25 in Python 3.12.

## Install

```bash
python -m pip install .
python scripts/check_install.py
```

This package declares the compatible FLE Python runtime (`factorio-learning-environment==0.4.3`) and A2A SDK (`a2a-sdk==0.3.25`) as installation dependencies. It does not bundle Factorio, Docker, or optional FLE vision sprites. If you need visual rendering, run `fle sprites` or set `FLE_SPRITES_DIR`.

## Use your own agent

```python
from fle_gym import FactorioLabEnv

env = FactorioLabEnv(rcon_port=27000, seed=42, run_dir="runs/my_agent")
obs, info = env.reset()
try:
    while True:
        mask = env.action_masks()  # Boolean feasibility mask (124,)
        action = my_agent.act(obs, mask)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()
```

You can run any algorithm and collect arbitrary metrics yourself from `(obs, reward, terminated, truncated, info)`.

## Live deployment smoke test

Ensure the existing FLE instance is already running, then:

```bash
python scripts/smoke_test.py --rcon-port 27000 --steps 10 --seed 42
```

Expected API: `Discrete(124)`, observation shape `(266,)`, Boolean `(124,)` action mask, and Gymnasium five-element `step` return. The smoke test checks actual calls; it does not train an agent.

**Important:** The release builder renames historical implementation modules, classes and imports to version-neutral names. Original source files remain untouched. This source transformation requires independent live validation: static analysis cannot guarantee all dynamic imports or string lookups are preserved.

## Save the actual Factorio game

After the final action, **before** closing or resetting the environment:

```python
save_path = env.save_world(
    "my_factory", "world_saves/my_factory.zip", rcon_port=27000
)
print("Verified Factorio save:", save_path)
```

This requests a server-side save via Factorio RCON, copies the resulting `.zip` from its Docker container, and verifies archive integrity. The host must have access to the `docker` CLI. If the running container cannot be uniquely identified from the published RCON port, supply `container="container_name"`.

**State impact:** By default this uses the original project save workaround: recursively removing unserialisable Lua functions from `storage` before requesting the save. That may disrupt further play or training in the same live world. Save only when the episode/run is complete, or set `scrub_storage=False` if you know your scenario does not need that workaround. Saving an archive does not automatically stop the container or delete the world.

## Experiment separation

Intentionally absent: `experiments/`, 5-replica scheduling, PPO/SAC/MAP-Elites training code not needed by the wrapper, monitoring dashboards, resume helpers, checkpoint orchestration, and dissertation result exporters. Lightweight in-memory episode bookkeeping necessary for observations, action validation or reward computation remains inside the wrapper. Some wrapper-native `info` telemetry may also remain because deleting it could alter the validated behaviour.

## License and attribution

FLE-Gym's original contributions are distributed under the [FLE-Gym Personal Non-Commercial License](LICENSE), copyright (c) 2026 Duran Reddy. Personal, educational, and non-commercial research use, modification and redistribution are permitted under its conditions; **commercial use requires prior written permission**. The project is **source-available, not OSI-approved open source**.

The independent [Factorio Learning Environment (FLE)](https://github.com/JackHopkins/factorio-learning-environment) remains MIT-licensed. FLE's copyright and MIT terms are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The FLE-Gym restrictions do not override rights granted under upstream FLE's MIT license. Copyright and licensing claims are limited to material the named contributor has the legal right to license; verify any institutional or other third-party IP obligations before public release.

FLE-Gym does **not** include or license Factorio game executables, code, artwork or other proprietary assets. Factorio must be acquired and operated under its own terms. The FLE-Gym maintainers are not affiliated with or endorsed by the Factorio developers or FLE's contributors.
