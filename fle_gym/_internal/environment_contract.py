#!/usr/bin/env python3
"""Shared runtime semantic environment contract for training and evaluation.

The key runtime rule is that deterministic evaluation must present the policy with
exactly the same environment semantics used during training.  Output paths and
screenshot settings may differ, but reward coefficients, refresh cadences,
mask/feasibility behavior, Factorio time advancement, and the observation
training horizon may not.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


SEMANTIC_ARGUMENT_DEFAULTS: Mapping[str, Any] = {
    "speed": 10.0,
    "settle_seconds": 0.04,
    "production_stats_every": 3,
    "infrastructure_state_every": 3,
    "feasibility_state_every": 2,
    "training_horizon": 4096,
    "progression_shaping": True,
    "production_reward_scale": 0.35,
    "research_progress_reward": 12.0,
    "trigger_technology_reward": 3.0,
    "lab_research_completion_reward": 12.0,
    "research_activity_reward": 0.02,
    "machine_reward_reference_seconds": 1.0,
    "normalize_machine_reward_by_sim_time": True,
    "lab_ingredient_potential_reward": 20.0,
    "idle_action_penalty": 0.10,
    "environment_rejection_cooldown_steps": 24,
    "line_preflight_every": 4,
    "max_unproductive_wait_streak": 1,
    "repeated_idle_penalty": 0.08,
    "research_progress_reward_cap": 1.0,
    "debug": True,
}


def _value(arguments: Mapping[str, Any], key: str) -> Any:
    return arguments.get(key, SEMANTIC_ARGUMENT_DEFAULTS[key])


def build_environment_kwargs_contract(
    arguments: Mapping[str, Any], *, run_dir: str, run_name: str
) -> Dict[str, Any]:
    """Return the semantic kwargs shared by runtime training and evaluation."""
    training_horizon = int(_value(arguments, "training_horizon"))
    if training_horizon < 1:
        raise ValueError("runtime requires training_horizon >= 1 for policy fidelity")

    return {
        "game_speed": float(_value(arguments, "speed")),
        "settle_seconds": float(_value(arguments, "settle_seconds")),
        "pause_between_steps": True,
        "hard_reset": True,
        "clear_entities_on_reset": True,
        "run_dir": str(run_dir),
        "run_name": str(run_name),
        "production_stats_every": int(_value(arguments, "production_stats_every")),
        "infrastructure_state_every": int(
            _value(arguments, "infrastructure_state_every")
        ),
        "feasibility_state_every": int(_value(arguments, "feasibility_state_every")),
        # Critical runtime fidelity fix: evaluation keeps the training horizon in
        # the observation vector even when the evaluator measures fewer steps.
        "training_horizon_steps": training_horizon,
        "progression_shaping": bool(_value(arguments, "progression_shaping")),
        "automated_output_reward_scale": float(
            _value(arguments, "production_reward_scale")
        ),
        "research_progress_reward": float(
            _value(arguments, "research_progress_reward")
        ),
        "trigger_technology_reward": float(
            _value(arguments, "trigger_technology_reward")
        ),
        "lab_research_completion_reward": float(
            _value(arguments, "lab_research_completion_reward")
        ),
        "research_activity_reward": float(
            _value(arguments, "research_activity_reward")
        ),
        "machine_reward_reference_seconds": float(
            _value(arguments, "machine_reward_reference_seconds")
        ),
        "normalize_machine_reward_by_sim_time": bool(
            _value(arguments, "normalize_machine_reward_by_sim_time")
        ),
        "lab_ingredient_potential_reward": float(
            _value(arguments, "lab_ingredient_potential_reward")
        ),
        "idle_action_penalty": float(_value(arguments, "idle_action_penalty")),
        "environment_rejection_cooldown_steps": int(
            _value(arguments, "environment_rejection_cooldown_steps")
        ),
        "line_preflight_every": int(_value(arguments, "line_preflight_every")),
        "max_unproductive_wait_streak": int(
            _value(arguments, "max_unproductive_wait_streak")
        ),
        "repeated_idle_penalty": float(
            _value(arguments, "repeated_idle_penalty")
        ),
        "research_progress_reward_cap": float(
            _value(arguments, "research_progress_reward_cap")
        ),
        "snapshot_every": 0,
        "snapshot_on_reset": False,
        "snapshot_on_episode_end": False,
        "debug": bool(_value(arguments, "debug")),
    }


def semantic_fingerprint(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Serializable semantic settings used to prove train/eval parity."""
    kwargs = build_environment_kwargs_contract(arguments, run_dir="<runtime>", run_name="<run>")
    return {
        key: value
        for key, value in kwargs.items()
        if key not in {"run_dir", "run_name", "snapshot_every", "snapshot_on_reset", "snapshot_on_episode_end"}
    }
