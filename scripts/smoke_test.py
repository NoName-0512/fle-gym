#!/usr/bin/env python3
"""Minimal live Gymnasium contract smoke test; no monitoring/training."""
from __future__ import annotations
import argparse
import math
from collections.abc import Mapping
import numpy as np
from fle_gym import FactorioLabEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rcon-port", type=int, default=27000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-world", default="", help="Optional name for an actual Factorio save after last step")
    ap.add_argument("--save-path", default="", help="Host path for final .zip archive")
    ap.add_argument("--container", default=None, help="Optional explicit Factorio Docker container")
    args = ap.parse_args()

    env = FactorioLabEnv(rcon_port=args.rcon_port, seed=args.seed, run_dir="runs/public_smoke")
    try:
        assert env.action_space.n == 124, f"Unexpected action space: {env.action_space}"
        assert env.observation_space.shape == (266,), f"Unexpected observation space: {env.observation_space}"
        obs, info = env.reset()
        assert isinstance(info, Mapping), "reset info must be a mapping"
        assert np.asarray(obs).shape == (266,), "reset observation has wrong shape"
        print("PASS reset: obs=(266,), actions=124")
        rng = np.random.default_rng(args.seed)
        steps_done = 0
        for _ in range(args.steps):
            mask = np.asarray(env.action_masks())
            assert mask.shape == (124,), f"Bad mask shape: {mask.shape}"
            assert mask.dtype.kind == "b", f"Expected Boolean mask, got {mask.dtype}"
            valid = np.flatnonzero(mask)
            assert valid.size, "No feasible actions in action mask"
            action = int(rng.choice(valid))
            result = env.step(action)
            assert isinstance(result, tuple) and len(result) == 5, "step must return 5 items"
            obs, reward, terminated, truncated, info = result
            assert np.asarray(obs).shape == (266,), "step observation wrong shape"
            assert isinstance(info, Mapping), "step info must be a mapping"
            assert math.isfinite(float(reward)), "Non-finite reward"
            assert isinstance(terminated, (bool, np.bool_)), "terminated must be Boolean"
            assert isinstance(truncated, (bool, np.bool_)), "truncated must be Boolean"
            steps_done += 1
            print(f"PASS step {steps_done}: action={action} reward={float(reward):.3f} terminated={bool(terminated)} truncated={bool(truncated)}")
            if terminated or truncated:
                break
        print(f"LIVE GYM CONTRACT PASS: {steps_done} transitions")
        if args.save_world:
            if not args.save_path:
                raise SystemExit("--save-path is required when --save-world is used")
            saved = env.save_world(args.save_world, args.save_path, container=args.container, rcon_port=args.rcon_port)
            print(f"FACTORIO SAVE PASS: {saved}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
