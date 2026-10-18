"""Reusable Gymnasium-compatible interface for the Factorio Learning Environment."""

from .api import FactorioEnv, FactorioLabEnv, FactorioOpenPlayEnv, make_env
from .registry import actions, action_names, tracked_items, craft_recipes, research_targets, trigger_technologies

__version__ = "1.0.0rc5"

__all__ = [
    "FactorioEnv", "FactorioLabEnv", "FactorioOpenPlayEnv", "make_env",
    "actions", "action_names", "tracked_items", "craft_recipes", "research_targets", "trigger_technologies",
]
