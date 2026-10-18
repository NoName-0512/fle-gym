#!/usr/bin/env python3
"""Diagnostic progression probe for the public FLE-Gym package.

Uses public action names/masks and observations. Unlike random smoke tests, it
attempts a coherent early-game smelting sequence and distinguishes interface
success from actual construction/production. Does not export a world save. It creates a local run directory and resets the
connected Factorio world, so run only against a disposable server.
"""

import argparse
import sys

import numpy as np

from fle_gym import FactorioLabEnv, FactorioOpenPlayEnv


WATCH = (
    "inventory__coal",
    "inventory__stone",
    "inventory__iron-ore",
    "inventory__iron-plate",
    "inventory__stone-furnace",
    "inventory__burner-mining-drill",
    "entities__burner-mining-drill",
    "entities__stone-furnace",
    "entities_runtime__burner-mining-drill",
    "entities_runtime__stone-furnace",
    "machine_output_total",
    "automated_output_total",
    "manual_harvested_total",
    "manual_crafted_total",
    "global_progress",
)

# Purposeful small sequence; later actions depend on previous prerequisites.
PLAN = (
    ("harvest_coal", 2),
    ("harvest_stone", 2),
    ("harvest_iron_ore", 2),
    ("place_drill_on_iron", 1),
    ("craft_stone_furnace", 1),
    ("wait_crafting", 2),
    ("place_furnace_near_drill", 1),
    ("fuel_drill", 1),
    ("fuel_furnace", 1),
    ("insert_hand_mined_ore", 4),
    ("place_chest_at_burner_output", 1),
    ("transfer_chest_ore_to_furnace", 2),
    ("wait_research", 4),
    ("collect_iron_plate", 3),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("open_world", "default_lab_scenario"), required=True)
    parser.add_argument("--rcon-port", type=int, default=27000)
    parser.add_argument("--seed", type=int, default=105)
    args = parser.parse_args()
    klass = FactorioOpenPlayEnv if args.scenario == "open_world" else FactorioLabEnv
    env = None
    try:
        env = klass(
            rcon_port=args.rcon_port,
            seed=args.seed,
            run_dir=f"runs/probe_{args.scenario}",
        )
        obs, info = env.reset(seed=args.seed)
        names = tuple(env.observation_schema)
        assert obs.shape == (266,), f"Unexpected observation shape: {obs.shape}"
        assert env.action_space.n == 124
        assert len(names) == 266
        idx = {name: i for i, name in enumerate(names)}
        current = np.asarray(obs, dtype=float).copy()
        initial = current.copy()
        action_ids = {name: env.action_index(name) for name, _ in PLAN}
        registry = {record["name"]: record for record in env.action_registry}
        calls = 0
        changed = []
        skipped = []

        def show(title, state):
            print(f"\n========== {title} ==========", flush=True)
            for key in WATCH:
                if key in idx:
                    print(f"{key:40} {state[idx[key]]:12.6g}", flush=True)

        show("INITIAL OBSERVATION (may be normalized)", current)
        mask = np.asarray(env.action_masks(), dtype=bool)
        assert mask.shape == (124,)
        print("\n========== INITIAL ACTION FEASIBILITY ==========", flush=True)
        for action_name, _ in PLAN:
            metadata = registry[action_name]
            print(
                f"{action_name:36} mask={bool(mask[action_ids[action_name]])} "
                f"items={metadata['required_items']} entities={metadata['required_entities']}",
                flush=True,
            )

        terminated = truncated = False
        for action_name, repeat in PLAN:
            for attempt in range(1, repeat + 1):
                mask = np.asarray(env.action_masks(), dtype=bool)
                action = action_ids[action_name]
                if not mask[action]:
                    skipped.append(action_name)
                    print(f"SKIP {action_name} [{attempt}/{repeat}]: masked invalid", flush=True)
                    break
                previous = current.copy()
                current, reward, terminated, truncated, info = env.step(action)
                current = np.asarray(current, dtype=float).copy()
                assert current.shape == (266,)
                assert np.isfinite(current).all()
                assert np.isfinite(reward)
                calls += 1
                deltas = [
                    f"{key}={current[idx[key]] - previous[idx[key]]:+.4g}"
                    for key in WATCH
                    if key in idx and abs(current[idx[key]] - previous[idx[key]]) > 1e-8
                ]
                if deltas:
                    changed.append(action_name)
                details = ", ".join(deltas[:7]) if deltas else "no change in watched features"
                print(
                    f"STEP {calls:02d} {action_name:33} reward={float(reward):+.4f} "
                    f"{details}",
                    flush=True,
                )
                if isinstance(info, dict):
                    for key in ("error", "action_error", "fault", "fault_reason"):
                        if info.get(key):
                            print(f"  INFO {key}: {str(info[key])[:250]}", flush=True)
                if terminated or truncated:
                    print(f"EPISODE ENDED: terminated={terminated}, truncated={truncated}", flush=True)
                    break
            else:
                continue
            if terminated or truncated:
                break

        show("FINAL OBSERVATION (may be normalized)", current)
        print("\n========== OBSERVED DELTAS ==========", flush=True)
        for key in WATCH:
            if key in idx:
                delta = float(current[idx[key]] - initial[idx[key]])
                if abs(delta) > 1e-8:
                    print(f"{key:40} {delta:+.6g}", flush=True)

        built_keys = [
            "entities__burner-mining-drill", "entities__stone-furnace",
            "entities_runtime__burner-mining-drill", "entities_runtime__stone-furnace",
        ]
        built = any(key in idx and current[idx[key]] > initial[idx[key]] + 1e-8 for key in built_keys)
        produced = any(
            key in idx and current[idx[key]] > initial[idx[key]] + 1e-8
            for key in ("machine_output_total", "automated_output_total")
        )
        harvested = "manual_harvested_total" in idx and (
            current[idx["manual_harvested_total"]] > initial[idx["manual_harvested_total"]] + 1e-8
        )
        print(f"\nActions executed={calls}; masked stages={sorted(set(skipped))}", flush=True)
        print(f"Harvester progress={harvested}; entity growth={built}; machine output growth={produced}", flush=True)
        if built and produced:
            print("FACTORY PROGRESSION PASS: construction AND production observed", flush=True)
            return 0
        if built:
            print("PARTIAL: construction observed but no increased machine output", flush=True)
        else:
            print("INCONCLUSIVE: no new drill or furnace observed", flush=True)
        print("Review action masks, starting inventory, runtime faults, and resource reachability.", flush=True)
        return 2
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    sys.exit(main())
