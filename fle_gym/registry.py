from __future__ import annotations

"""Stable registries that external RL adapters can inspect without Lua/RCON."""

from dataclasses import asdict
from typing import Any

from ._bootstrap import activate_internal_path
activate_internal_path()
from factorio_openended_env_machine import Runtime_ACTIONS  # type: ignore  # noqa: E402
from factorio_openended_env_research_yqeibtv import Runtime_RESEARCH_TARGETS, Runtime_TRIGGER_TECHNOLOGIES, CRAFT_BY_ACTION  # type: ignore  # noqa: E402
from factorio_transition_env import TRACKED_ITEMS  # type: ignore  # noqa: E402


def actions() -> tuple[dict[str, Any], ...]:
    return tuple(asdict(spec) for spec in Runtime_ACTIONS)


def action_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in Runtime_ACTIONS)


def tracked_items() -> tuple[str, ...]:
    return tuple(str(x) for x in TRACKED_ITEMS)



def craft_recipes() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for action, spec in CRAFT_BY_ACTION.items():
        try:
            out[str(action)] = asdict(spec)
        except TypeError:
            out[str(action)] = {k: getattr(spec, k) for k in dir(spec) if not k.startswith("_") and not callable(getattr(spec, k))}
    return out

def research_targets() -> tuple[str, ...]:
    return tuple(str(x) for x in Runtime_RESEARCH_TARGETS)


def trigger_technologies() -> tuple[str, ...]:
    return tuple(str(x) for x in Runtime_TRIGGER_TECHNOLOGIES)


__all__ = ["actions", "action_names", "tracked_items", "craft_recipes", "research_targets", "trigger_technologies"]
