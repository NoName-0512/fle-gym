#!/usr/bin/env python3
"""Live Factorio validation for Lab Play and Open Play."""

import argparse
from pathlib import Path

import numpy as np

from fle_gym import FactorioLabEnv, FactorioOpenPlayEnv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scenario",
    choices=["default_lab_scenario", "open_world"],
    required=True,
)
parser.add_argument("--rcon-port", type=int, default=27000)
parser.add_argument("--steps", type=int, default=4096)
parser.add_argument("--seed", type=int, default=104)
parser.add_argument("--save-world", required=True)

args = parser.parse_args()

env_class = (
    FactorioOpenPlayEnv
    if args.scenario == "open_world"
    else FactorioLabEnv
)

output = Path("world_saves") / f"{args.save_world}.zip"

if output.exists():
    raise FileExistsError(f"Save already exists: {output}")

output.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(args.seed)
env = None

try:
    env = env_class(
        rcon_port=args.rcon_port,
        seed=args.seed,
        run_dir=f"runs/{args.scenario}_validation",
    )

    obs, info = env.reset(seed=args.seed)

    assert obs.shape == (266,)
    assert np.isfinite(obs).all()
    assert env.action_space.n == 124

    print(
        f"PASS reset: scenario={args.scenario}, "
        f"observations={obs.shape}, actions={env.action_space.n}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        mask = env.action_masks()

        assert mask.shape == (124,)
        assert mask.dtype == np.bool_

        valid = np.flatnonzero(mask)
        assert len(valid) > 0

        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (266,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)

        if step % 256 == 0 or terminated or truncated:
            print(
                f"PASS step {step}: "
                f"reward={reward:.3f} "
                f"terminated={terminated} "
                f"truncated={truncated}",
                flush=True,
            )

        if terminated or truncated:
            if step < args.steps:
                raise RuntimeError(
                    f"Episode ended at {step}, "
                    f"before requested {args.steps} steps"
                )
            break

    print(
        f"SCENARIO CONTRACT PASS: {args.steps} transitions",
        flush=True,
    )

    saved = env.save_world(
        args.save_world,
        output,
        rcon_port=args.rcon_port,
    )

    print("FACTORIO SAVE PASS:", saved, flush=True)

finally:
    if env is not None:
        env.close()
