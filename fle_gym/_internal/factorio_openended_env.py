from __future__ import annotations

"""Open-ended Factorio transition environment with production, power, and research.

Unlike the original iron-plate smoke environment, this environment never ends
because a particular item or technology is reached. The caller decides how
long to train by specifying a total number of transitions or a wall-clock
limit. One factory is preserved for the entire run.

The policy receives compact numeric state. Screenshots are evidence and
monitoring output; they are not PPO observations in this MLP implementation.
"""

from collections import Counter
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from factorio_transition_env import (
    ACTIONS,
    ActionSpec,
    FactorioTransitionEnv,
    _dedent,
    inventory_to_dict,
    json_safe,
    normalise_name,
    production_stats_summary,
    safe_float,
    score_to_float,
    spaces,
)


DEFAULT_OPENENDED_INVENTORY: Dict[str, int] = {
    # Burner bootstrap.
    "burner-mining-drill": 2,
    "stone-furnace": 4,
    "coal": 200,
    "wooden-chest": 12,
    # Steam power layer.
    "offshore-pump": 2,
    "boiler": 4,
    "steam-engine": 4,
    "pipe": 160,
    "small-electric-pole": 80,
    # Electric production and research layer.
    "electric-mining-drill": 6,
    "lab": 4,
    "automation-science-pack": 300,
    # Components for later expansion actions and manual recovery.
    "transport-belt": 200,
    "inserter": 80,
    "assembling-machine-1": 6,
    "iron-plate": 150,
    "copper-plate": 150,
    "iron-gear-wheel": 80,
    "electronic-circuit": 80,
}

OPENENDED_TRACKED_ITEMS: Tuple[str, ...] = (
    "wood",
    "coal",
    "stone",
    "iron-ore",
    "copper-ore",
    "iron-plate",
    "copper-plate",
    "steel-plate",
    "iron-gear-wheel",
    "electronic-circuit",
    "automation-science-pack",
    "logistic-science-pack",
    "burner-mining-drill",
    "electric-mining-drill",
    "stone-furnace",
    "offshore-pump",
    "boiler",
    "steam-engine",
    "pipe",
    "small-electric-pole",
    "lab",
    "wooden-chest",
    "transport-belt",
    "inserter",
    "assembling-machine-1",
)

OPENENDED_TRACKED_ENTITIES: Tuple[str, ...] = (
    "burner-mining-drill",
    "electric-mining-drill",
    "stone-furnace",
    "offshore-pump",
    "boiler",
    "steam-engine",
    "small-electric-pole",
    "lab",
    "wooden-chest",
    "transport-belt",
    "inserter",
    "assembling-machine-1",
)

RESEARCH_TARGETS: Tuple[str, ...] = (
    "automation",
    "logistics",
    "steel-processing",
    "electronics",
)

INFRA_MARKER = "__FLE_RL_INFRASTRUCTURE__"
RESEARCH_QUEUE_MARKER = "__FLE_RL_RESEARCH_QUEUE__"
RUNTIME_BUILD_MARKER = "__FLE_RL_RUNTIME_BUILD__"
RUNTIME_POWER_MARKER = "__FLE_RL_RUNTIME_POWER__"

# Reward only outputs produced by machines. Manual harvesting is exposed in a
# different production-stat bucket and deliberately receives no reward.
AUTOMATED_OUTPUT_WEIGHTS: Dict[str, float] = {
    "iron-ore": 0.05,
    "copper-ore": 0.05,
    "iron-plate": 0.5,
    "copper-plate": 0.5,
    "steel-plate": 1.5,
    "iron-gear-wheel": 0.75,
    "electronic-circuit": 1.25,
    "automation-science-pack": 2.0,
    "logistic-science-pack": 3.0,
}


def _clone_base_action(name: str, **updates: Any) -> ActionSpec:
    for spec in ACTIONS:
        if spec.name == name:
            return replace(spec, **updates)
    raise KeyError(name)


def _openended_actions() -> Tuple[ActionSpec, ...]:
    # Existing FLE-0.4.3-compatible macros are reused, but their one-time flags
    # and smoke-task event bonuses are removed. Open-ended reward is calculated
    # from automated output, power, research, and verified infrastructure.
    base = [
        _clone_base_action("noop", event_bonus=0.0),
        _clone_base_action("wait", event_bonus=0.0),
        _clone_base_action(
            "place_drill_on_iron",
            event_bonus=0.0,
            single_use_flag=None,
            description="Place another burner drill on the nearest iron patch.",
        ),
        _clone_base_action(
            "place_furnace_near_drill",
            event_bonus=0.0,
            single_use_flag=None,
            description="Place another stone furnace near a burner drill.",
        ),
        _clone_base_action(
            "fuel_drill",
            event_bonus=0.0,
            single_use_flag=None,
        ),
        _clone_base_action(
            "fuel_furnace",
            event_bonus=0.0,
            single_use_flag=None,
        ),
        _clone_base_action("collect_iron_plate", event_bonus=0.0),
        _clone_base_action("harvest_coal", event_bonus=0.0),
        _clone_base_action("harvest_stone", event_bonus=0.0),
        _clone_base_action("harvest_iron_ore", event_bonus=0.0),
        _clone_base_action("craft_stone_furnace", event_bonus=0.0),
        _clone_base_action("insert_hand_mined_ore", event_bonus=0.0),
    ]

    logistics = [
        ActionSpec(
            name="place_chest_at_burner_output",
            category="logistics",
            required_items={"wooden-chest": 1},
            required_entities={"burner-mining-drill": 1},
            produced_entity="wooden-chest",
            description="Place a chest at the nearest burner drill's drop position.",
            code=_dedent(
                """
                drill_position = nearest(Prototype.BurnerMiningDrill)
                drill = get_entity(Prototype.BurnerMiningDrill, drill_position)
                move_to(drill.drop_position)
                chest = place_entity(
                    Prototype.WoodenChest,
                    position=drill.drop_position,
                    exact=True,
                )
                if chest is None:
                    raise RuntimeError("output chest placement returned None")
                print("__FLE_RL_EVENT__:burner_output_chest_placed")
                """
            ),
        ),
        ActionSpec(
            name="place_chest_at_electric_output",
            category="logistics",
            required_items={"wooden-chest": 1},
            required_entities={"electric-mining-drill": 1},
            produced_entity="wooden-chest",
            description="Place a chest at the nearest electric drill's drop position.",
            code=_dedent(
                """
                drill_position = nearest(Prototype.ElectricMiningDrill)
                drill = get_entity(Prototype.ElectricMiningDrill, drill_position)
                move_to(drill.drop_position)
                chest = place_entity(
                    Prototype.WoodenChest,
                    position=drill.drop_position,
                    exact=True,
                )
                if chest is None:
                    raise RuntimeError("electric output chest placement returned None")
                print("__FLE_RL_EVENT__:electric_output_chest_placed")
                """
            ),
        ),
        ActionSpec(
            name="transfer_chest_ore_to_furnace",
            category="transfer",
            required_entities={"wooden-chest": 1, "stone-furnace": 1},
            description="Move available iron ore from the nearest output chest to a furnace.",
            code=_dedent(
                """
                chest_position = nearest(Prototype.WoodenChest)
                furnace_position = nearest(Prototype.StoneFurnace)
                chest = get_entity(Prototype.WoodenChest, chest_position)
                furnace = get_entity(Prototype.StoneFurnace, furnace_position)
                moved = extract_item(Prototype.IronOre, chest, quantity=5)
                if moved is None or moved == 0:
                    raise RuntimeError("no iron ore available in output chest")
                try:
                    amount = max(1, min(5, int(moved)))
                except Exception:
                    amount = 1
                inserted = insert_item(Prototype.IronOre, furnace, quantity=amount)
                print("__FLE_RL_EVENT__:ore_transferred")
                print("transfer_result=" + repr((moved, inserted)))
                """
            ),
        ),
    ]

    power = [
        ActionSpec(
            name="place_offshore_pump",
            category="power",
            required_items={"offshore-pump": 1},
            produced_entity="offshore-pump",
            description="Place an offshore pump at the nearest water tile.",
            code=_dedent(
                """
                water = nearest(Resource.Water)
                move_to(water)
                pump = place_entity(Prototype.OffshorePump, position=water)
                if pump is None:
                    raise RuntimeError("offshore pump placement returned None")
                print("__FLE_RL_EVENT__:offshore_pump_placed")
                """
            ),
        ),
        ActionSpec(
            name="place_boiler_near_pump",
            category="power",
            required_items={"boiler": 1},
            required_entities={"offshore-pump": 1},
            produced_entity="boiler",
            description="Place a boiler in buildable space near an offshore pump.",
            code=_dedent(
                """
                pump_position = nearest(Prototype.OffshorePump)
                pump = get_entity(Prototype.OffshorePump, pump_position)
                box = BuildingBox(
                    width=Prototype.Boiler.WIDTH + 6,
                    height=Prototype.Boiler.HEIGHT + 6,
                )
                coordinates = nearest_buildable(
                    Prototype.Boiler,
                    box,
                    pump.position,
                )
                move_to(coordinates.center)
                boiler = place_entity(
                    Prototype.Boiler,
                    position=coordinates.center,
                    direction=Direction.LEFT,
                )
                if boiler is None:
                    raise RuntimeError("boiler placement returned None")
                print("__FLE_RL_EVENT__:boiler_placed")
                """
            ),
        ),
        ActionSpec(
            name="connect_pump_to_boiler",
            category="power",
            required_items={"pipe": 1},
            required_entities={"offshore-pump": 1, "boiler": 1},
            single_use_flag="pump_boiler_connected",
            description="Connect the nearest offshore pump to the nearest boiler with pipes.",
            code=_dedent(
                """
                pump_position = nearest(Prototype.OffshorePump)
                boiler_position = nearest(Prototype.Boiler)
                pump = get_entity(Prototype.OffshorePump, pump_position)
                boiler = get_entity(Prototype.Boiler, boiler_position)
                result = connect_entities(pump, boiler, Prototype.Pipe)
                print("__FLE_RL_EVENT__:pump_boiler_connected")
                print("connection_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="fuel_boiler",
            category="power",
            required_items={"coal": 10},
            required_entities={"boiler": 1},
            description="Insert ten coal into the nearest boiler.",
            code=_dedent(
                """
                boiler_position = nearest(Prototype.Boiler)
                boiler = get_entity(Prototype.Boiler, boiler_position)
                result = insert_item(Prototype.Coal, boiler, quantity=10)
                print("__FLE_RL_EVENT__:boiler_fuelled")
                print("fuel_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="place_steam_engine_near_boiler",
            category="power",
            required_items={"steam-engine": 1},
            required_entities={"boiler": 1},
            produced_entity="steam-engine",
            description="Place a steam engine in buildable space near the boiler.",
            code=_dedent(
                """
                boiler_position = nearest(Prototype.Boiler)
                boiler = get_entity(Prototype.Boiler, boiler_position)
                box = BuildingBox(
                    width=Prototype.SteamEngine.WIDTH + 8,
                    height=Prototype.SteamEngine.HEIGHT + 8,
                )
                coordinates = nearest_buildable(
                    Prototype.SteamEngine,
                    box,
                    boiler.position,
                )
                move_to(coordinates.center)
                engine = place_entity(
                    Prototype.SteamEngine,
                    position=coordinates.center,
                    direction=Direction.LEFT,
                )
                if engine is None:
                    raise RuntimeError("steam engine placement returned None")
                print("__FLE_RL_EVENT__:steam_engine_placed")
                """
            ),
        ),
        ActionSpec(
            name="connect_boiler_to_engine",
            category="power",
            required_items={"pipe": 1},
            required_entities={"boiler": 1, "steam-engine": 1},
            single_use_flag="boiler_engine_connected",
            description="Connect the nearest boiler to the nearest steam engine with pipes.",
            code=_dedent(
                """
                boiler_position = nearest(Prototype.Boiler)
                engine_position = nearest(Prototype.SteamEngine)
                boiler = get_entity(Prototype.Boiler, boiler_position)
                engine = get_entity(Prototype.SteamEngine, engine_position)
                result = connect_entities(boiler, engine, Prototype.Pipe)
                print("__FLE_RL_EVENT__:boiler_engine_connected")
                print("connection_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="place_small_electric_pole",
            category="power",
            required_items={"small-electric-pole": 1},
            required_entities={"steam-engine": 1},
            produced_entity="small-electric-pole",
            description=(
                "Place a persistent source pole beside the steam engine using "
                "the verified Factorio runtime builder."
            ),
            code='print("__FLE_RL_EVENT__:runtime_source_pole_requested")',
        ),
        ActionSpec(
            name="place_electric_drill_on_iron",
            category="power",
            required_items={"electric-mining-drill": 1},
            produced_entity="electric-mining-drill",
            description="Place an electric mining drill on the nearest iron patch.",
            code=_dedent(
                """
                ore = nearest(Prototype.IronOre)
                move_to(ore)
                drill = place_entity(
                    Prototype.ElectricMiningDrill,
                    position=ore,
                    direction=Direction.DOWN,
                    exact=False,
                )
                if drill is None:
                    raise RuntimeError("electric drill placement returned None")
                print("__FLE_RL_EVENT__:electric_drill_placed")
                """
            ),
        ),
        ActionSpec(
            name="connect_electric_drill_to_power",
            category="power",
            required_items={"small-electric-pole": 1},
            required_entities={"steam-engine": 1, "electric-mining-drill": 1},
            single_use_flag="electric_drill_connected",
            description="Connect an electric drill to the steam engine using electric poles.",
            code=_dedent(
                """
                engine_position = nearest(Prototype.SteamEngine)
                drill_position = nearest(Prototype.ElectricMiningDrill)
                engine = get_entity(Prototype.SteamEngine, engine_position)
                drill = get_entity(Prototype.ElectricMiningDrill, drill_position)
                result = connect_entities(
                    engine,
                    drill,
                    Prototype.SmallElectricPole,
                )
                print("__FLE_RL_EVENT__:electric_drill_connected")
                print("connection_result=" + repr(result))
                """
            ),
        ),
    ]

    research = [
        ActionSpec(
            name="craft_lab",
            category="research",
            required_items={
                "electronic-circuit": 10,
                "iron-gear-wheel": 10,
                "transport-belt": 4,
            },
            description=(
                "Craft a laboratory when none remains in the player inventory. "
                "This keeps the research layer recoverable in a continuing run."
            ),
            code=_dedent(
                """
                result = craft_item(Prototype.Lab, quantity=1)
                print("__FLE_RL_EVENT__:lab_crafted")
                print("craft_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="craft_automation_science",
            category="research",
            required_items={
                "copper-plate": 10,
                "iron-gear-wheel": 10,
            },
            description=(
                "Craft ten automation science packs when the research supply is empty."
            ),
            code=_dedent(
                """
                result = craft_item(Prototype.AutomationSciencePack, quantity=10)
                print("__FLE_RL_EVENT__:automation_science_crafted")
                print("craft_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="place_lab",
            category="research",
            required_items={"lab": 1},
            required_entities={"steam-engine": 1},
            produced_entity="lab",
            description=(
                "Place a laboratory near the steam engine without requiring a "
                "pre-existing pole. The power-connection action is atomic and "
                "creates any required poles afterwards."
            ),
            code='print("__FLE_RL_EVENT__:runtime_lab_requested")',
        ),
        ActionSpec(
            name="connect_lab_to_power",
            category="research",
            required_items={"small-electric-pole": 2},
            required_entities={"steam-engine": 1, "lab": 1},
            produced_entity="small-electric-pole",
            single_use_flag="lab_connected",
            description=(
                "Atomically create and auto-connect a verified electric-pole "
                "route from the steam engine to the laboratory."
            ),
            code='print("__FLE_RL_EVENT__:runtime_lab_power_requested")',
        ),
        ActionSpec(
            name="insert_automation_science",
            category="research",
            required_items={"automation-science-pack": 10},
            required_entities={"lab": 1},
            description="Insert ten automation science packs into the nearest lab.",
            code=_dedent(
                """
                lab_position = nearest(Prototype.Lab)
                lab = get_entity(Prototype.Lab, lab_position)
                result = insert_item(
                    Prototype.AutomationSciencePack,
                    lab,
                    quantity=10,
                )
                print("__FLE_RL_EVENT__:automation_science_inserted")
                print("insert_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="research_automation",
            category="research",
            required_entities={"lab": 1},
            description="Select automation as the current research target.",
            code='print("__FLE_RL_EVENT__:research_requested_automation")',
        ),
        ActionSpec(
            name="research_logistics",
            category="research",
            required_entities={"lab": 1},
            description="Select logistics as the current research target.",
            code='print("__FLE_RL_EVENT__:research_requested_logistics")',
        ),
        ActionSpec(
            name="research_steel_processing",
            category="research",
            required_entities={"lab": 1},
            description="Select steel processing as the current research target.",
            code='print("__FLE_RL_EVENT__:research_requested_steel")',
        ),
        ActionSpec(
            name="research_electronics",
            category="research",
            required_entities={"lab": 1},
            description="Select electronics as the current research target.",
            code='print("__FLE_RL_EVENT__:research_requested_electronics")',
        ),
        ActionSpec(
            name="wait_research",
            category="research",
            code="sleep(5)",
            settle_multiplier=0.0,
            description="Advance five seconds of game time for power, production, and research.",
        ),
    ]

    return tuple(base + logistics + power + research)


OPENENDED_ACTIONS: Tuple[ActionSpec, ...] = _openended_actions()


class FactorioOpenEndedEnv(FactorioTransitionEnv):
    """Persistent, externally bounded FLE environment.

    ``step`` always returns ``terminated=False`` and ``truncated=False``. Stop
    conditions belong to the training driver, not to a hard-coded factory goal.
    """

    def __init__(
        self,
        *,
        initial_inventory: Optional[Mapping[str, int]] = None,
        infrastructure_state_every: int = 5,
        automated_output_reward_scale: float = 1.0,
        research_progress_reward: float = 4.0,
        research_completion_reward: float = 20.0,
        first_power_reward: float = 10.0,
        infrastructure_growth_reward: float = 0.25,
        openended_step_penalty: float = 0.001,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("max_steps", 2_000_000_000)
        kwargs.setdefault("terminate_on_plate", False)
        kwargs.setdefault("all_technologies_researched", False)
        kwargs.setdefault("score_reward_scale", 0.0)
        kwargs.setdefault("iron_plate_reward", 0.0)
        kwargs.setdefault("step_penalty", 0.0)
        kwargs.setdefault("reward_clip", None)
        kwargs.setdefault("snapshot_every", 0)
        kwargs.setdefault("snapshot_on_reset", False)
        kwargs.setdefault("snapshot_on_episode_end", False)
        kwargs.setdefault("initial_inventory", initial_inventory or DEFAULT_OPENENDED_INVENTORY)

        super().__init__(**kwargs)

        self.infrastructure_state_every = max(1, int(infrastructure_state_every))
        self.automated_output_reward_scale = float(automated_output_reward_scale)
        self.research_progress_reward = float(research_progress_reward)
        self.research_completion_reward = float(research_completion_reward)
        self.first_power_reward = float(first_power_reward)
        self.infrastructure_growth_reward = float(infrastructure_growth_reward)
        self.openended_step_penalty = abs(float(openended_step_penalty))

        self.actions = OPENENDED_ACTIONS
        self.action_space = spaces.Discrete(len(self.actions))
        self.inventory = {item: 0.0 for item in OPENENDED_TRACKED_ITEMS}
        self.entity_counts = {name: 0.0 for name in OPENENDED_TRACKED_ENTITIES}
        self.flags.update(
            {
                "pump_boiler_connected": False,
                "boiler_engine_connected": False,
                "electric_drill_connected": False,
                "lab_connected": False,
                "power_online": False,
            }
        )
        self.infrastructure_state: Dict[str, Any] = self._empty_infrastructure_state()
        self.previous_infrastructure_state: Dict[str, Any] = dict(self.infrastructure_state)
        self.automated_output_total = 0.0
        self.research_reward_total = 0.0
        self.power_reward_total = 0.0
        self.milestones_seen: set[str] = set()
        self.last_openended_reward_parts: Dict[str, float] = {}
        self._last_entity_records: List[Dict[str, Any]] = []

        flag_names = (
            "drill_placed",
            "furnace_placed",
            "drill_fuelled",
            "furnace_fuelled",
            "pump_boiler_connected",
            "boiler_engine_connected",
            "electric_drill_connected",
            "lab_connected",
            "power_online",
        )
        infrastructure_features = (
            "steam_energy",
            "lab_energy",
            "electric_drill_energy",
            "research_progress",
            *(f"researched__{name}" for name in RESEARCH_TARGETS),
            *(f"available__{name}" for name in RESEARCH_TARGETS),
            "researched_count",
            "automated_output_total",
            "research_reward_total",
            "power_reward_total",
            "valid_action_fraction",
            "global_progress",
        )
        self.observation_feature_names = (
            *(f"inventory__{item}" for item in OPENENDED_TRACKED_ITEMS),
            *(f"entities__{entity}" for entity in OPENENDED_TRACKED_ENTITIES),
            *(f"flag__{name}" for name in flag_names),
            *infrastructure_features,
        )
        # Subclasses may append observation features. Keep the base count so
        # this class validates only the portion it constructs itself.
        self._base_observation_feature_count = len(self.observation_feature_names)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self._base_observation_feature_count,),
            dtype=np.float32,
        )

        self._write_openended_config()

    @staticmethod
    def _convert_initial_inventory(inventory: Mapping[str, int]) -> Dict[str, int]:
        """FLE 0.4.3 serialises this mapping directly through JSON."""
        converted: Dict[str, int] = {}
        for raw_name, amount in inventory.items():
            name = normalise_name(raw_name)
            if not name:
                raise ValueError(f"Invalid inventory item name: {raw_name!r}")
            amount_int = int(amount)
            if amount_int < 0:
                raise ValueError(f"Inventory quantity cannot be negative: {raw_name!r}")
            converted[name] = amount_int
        return converted

    @staticmethod
    def _empty_infrastructure_state() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "steam_energy": 0.0,
            "lab_energy": 0.0,
            "electric_drill_energy": 0.0,
            "current_research": "",
            "research_progress": 0.0,
            "power_online": False,
            "query_ok": False,
            "query_error": "not queried",
        }
        for target in RESEARCH_TARGETS:
            state[f"researched__{target}"] = False
            state[f"available__{target}"] = True
        for entity in OPENENDED_TRACKED_ENTITIES:
            state[f"count__{entity}"] = 0.0
        return state

    def _write_openended_config(self) -> None:
        payload = {
            "environment": "FactorioOpenEndedEnv",
            "externally_bounded": True,
            "terminal_goal": None,
            "action_count": len(self.actions),
            "actions": [asdict(spec) for spec in self.actions],
            "observation_features": list(self.observation_feature_names),
            "tracked_items": list(OPENENDED_TRACKED_ITEMS),
            "tracked_entities": list(OPENENDED_TRACKED_ENTITIES),
            "research_targets": list(RESEARCH_TARGETS),
            "reward": {
                "automated_output_weights": AUTOMATED_OUTPUT_WEIGHTS,
                "automated_output_reward_scale": self.automated_output_reward_scale,
                "research_progress_reward": self.research_progress_reward,
                "research_completion_reward": self.research_completion_reward,
                "first_power_reward": self.first_power_reward,
                "infrastructure_growth_reward": self.infrastructure_growth_reward,
                "step_penalty": self.openended_step_penalty,
                "manual_harvest_reward": 0.0,
            },
        }
        (Path(self.run_dir) / "openended_environment_config.json").write_text(
            json.dumps(json_safe(payload), indent=2),
            encoding="utf-8",
        )

    def _collect_compact_state(self, *, include_production_stats: bool) -> None:
        raw_inventory = self.namespace.inspect_inventory()
        parsed_inventory = inventory_to_dict(raw_inventory)
        for item in OPENENDED_TRACKED_ITEMS:
            self.inventory[item] = float(parsed_inventory.get(item, 0.0))

        try:
            self.production_score = score_to_float(self.namespace.score())
        except Exception:
            pass

        if include_production_stats:
            try:
                self.last_production_stats = production_stats_summary(
                    self.namespace._get_production_stats()
                )
            except Exception:
                pass

    def _entity_records(self) -> List[Dict[str, Any]]:
        records = super()._entity_records()
        counts: Counter[str] = Counter(record.get("name", "unknown") for record in records)
        for name in OPENENDED_TRACKED_ENTITIES:
            self.entity_counts[name] = float(counts.get(name, 0))
        self.flags["drill_placed"] = bool(
            self.entity_counts.get("burner-mining-drill", 0)
            or self.entity_counts.get("electric-mining-drill", 0)
        )
        self.flags["furnace_placed"] = bool(self.entity_counts.get("stone-furnace", 0))
        self._last_entity_records = records
        return records

    @staticmethod
    def _parse_runtime_result(raw: str, marker: str) -> Dict[str, Any]:
        line = next((line for line in str(raw).splitlines() if marker in line), "")
        if not line:
            raise RuntimeError(
                f"runtime marker {marker!r} missing from RCON response: {str(raw)[:500]!r}"
            )
        payload = line.split(marker, 1)[1].lstrip(": |")
        result: Dict[str, Any] = {"raw": str(raw)}
        for token in payload.split("|"):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            result[key.strip()] = value.strip()
        if result.get("ok") != "1":
            failure = result.get("error") or result.get("reason") or "runtime operation failed"
            raise RuntimeError(
                # Keep the structured fields emitted by the Factorio command.
                # A short error code alone loses the technology/item that made
                # a scenario unrecoverable and caused several blind releases.
                f"{failure}: {result}"
            )
        return result

    def _runtime_place_near(
        self,
        *,
        entity_name: str,
        item_name: str,
        anchor_name: str,
        minimum_radius: int,
        maximum_radius: int,
    ) -> Dict[str, Any]:
        """Place one real entity near an anchor and consume one inventory item.

        This runtime fallback exists because FLE 0.4.3 can report a successful
        ``place_entity_next_to`` call for a small electric pole while no
        persistent pole appears on the surface. The command verifies the
        entity count after creation before reporting success.
        """
        marker = RUNTIME_BUILD_MARKER
        electric_options = ",auto_connect=true" if entity_name == "small-electric-pole" else ""
        command = (
            "/sc local marker=" + json.dumps(marker) + "; "
            "local function run() "
            "local p=nil; local controller=''; ""if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then ""p=storage.agent_characters[1]; controller='agent-character'; ""elseif player and player.valid then p=player; controller='global-character'; ""elseif game.get_player(1) then p=game.get_player(1); controller='lua-player-1'; ""elseif game.connected_players[1] then p=game.connected_players[1]; controller='connected-player'; end; "
            "if not p then rcon.print(marker..'|ok=0|error=no-player'); return end; "
            "local s=p.surface; local f=p.force; "
            "local anchors=s.find_entities_filtered{name=" + json.dumps(anchor_name) + ",force=f}; "
            "local a=anchors[1]; if not a then rcon.print(marker..'|ok=0|error=missing-anchor'); return end; "
            "local item=" + json.dumps(item_name) + "; "
            "local available=p.get_item_count(item); if available<1 then rcon.print(marker..'|ok=0|error=missing-item|controller='..controller..'|available='..available); return end; "
            "local bb=a.bounding_box; local cx=(bb.left_top.x+bb.right_bottom.x)/2; "
            "local cy=(bb.left_top.y+bb.right_bottom.y)/2; "
            "for r=" + str(int(minimum_radius)) + "," + str(int(maximum_radius)) + " do "
            "local c={{x=bb.left_top.x-r,y=cy},{x=bb.right_bottom.x+r,y=cy},"
            "{x=cx,y=bb.left_top.y-r},{x=cx,y=bb.right_bottom.y+r},"
            "{x=bb.left_top.x-r,y=bb.left_top.y-r},{x=bb.right_bottom.x+r,y=bb.left_top.y-r},"
            "{x=bb.left_top.x-r,y=bb.right_bottom.y+r},{x=bb.right_bottom.x+r,y=bb.right_bottom.y+r}}; "
            "for _,pos in ipairs(c) do "
            "if s.can_place_entity{name=" + json.dumps(entity_name) + ",position=pos,force=f} then "
            "local e=s.create_entity{name=" + json.dumps(entity_name) + ",position=pos,force=f,raise_built=true" + electric_options + "}; "
            "if e and e.valid then local removed=p.remove_item{name=item,count=1}; "
            "if removed==1 then local n=#s.find_entities_filtered{name=" + json.dumps(entity_name) + ",force=f}; "
            "rcon.print(marker..'|ok=1|entity='..e.name..'|x='..e.position.x..'|y='..e.position.y..'|count='..n..'|controller='..controller..'|remaining='..p.get_item_count(item)); return "
            "else e.destroy(); rcon.print(marker..'|ok=0|error=inventory-remove-failed'); return end; end; end; end; end; "
            "rcon.print(marker..'|ok=0|error=no-buildable-position'); end; "
            "local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_connect_lab_power(self) -> Dict[str, Any]:
        """Build a verified, resource-consuming pole route to the nearest lab."""
        marker = RUNTIME_POWER_MARKER
        command = (
            "/sc local marker=" + json.dumps(marker) + "; "
            "local function run() "
            "local p=nil; local controller=''; ""if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then ""p=storage.agent_characters[1]; controller='agent-character'; ""elseif player and player.valid then p=player; controller='global-character'; ""elseif game.get_player(1) then p=game.get_player(1); controller='lua-player-1'; ""elseif game.connected_players[1] then p=game.connected_players[1]; controller='connected-player'; end; "
            "if not p then rcon.print(marker..'|ok=0|error=no-player'); return end; "
            "local s=p.surface; local f=p.force; "
            "local engines=s.find_entities_filtered{name='steam-engine',force=f}; local engine=engines[1]; "
            "local labs=s.find_entities_filtered{name='lab',force=f}; local lab=labs[1]; "
            "if not engine then rcon.print(marker..'|ok=0|error=missing-steam-engine'); return end; "
            "if not lab then rcon.print(marker..'|ok=0|error=missing-lab'); return end; "
            "local ok0,c0=pcall(function() return lab.is_connected_to_electric_network() end); "
            "if ok0 and c0 then rcon.print(marker..'|ok=1|already=1|created=0|lab_connected=1'); return end; "
            "local created={}; "
            "local function d2(a,b) local dx=a.position.x-b.position.x; local dy=a.position.y-b.position.y; return dx*dx+dy*dy end; "
            "local function nearest_pole(a,maxd2) local best=nil; local bd=maxd2; "
            "for _,e in ipairs(s.find_entities_filtered{name='small-electric-pole',force=f}) do local v=d2(a,e); if v<bd then best=e; bd=v end end; return best end; "
            "local function consume_and_create(pos) "
            "local available=p.get_item_count('small-electric-pole'); if available<1 then return nil,'missing-small-electric-pole-'..controller..'-count-'..available end; "
            "if not s.can_place_entity{name='small-electric-pole',position=pos,force=f} then return nil,'blocked' end; "
            "local e=s.create_entity{name='small-electric-pole',position=pos,force=f,raise_built=true,auto_connect=true}; "
            "if not e then return nil,'create-failed' end; local removed=p.remove_item{name='small-electric-pole',count=1}; "
            "if removed~=1 then e.destroy(); return nil,'inventory-remove-failed' end; created[#created+1]=e; return e,nil end; "
            "local function build_near(a) local bb=a.bounding_box; local cx=(bb.left_top.x+bb.right_bottom.x)/2; local cy=(bb.left_top.y+bb.right_bottom.y)/2; "
            "for r=1,4 do local c={{x=bb.left_top.x-r,y=cy},{x=bb.right_bottom.x+r,y=cy},{x=cx,y=bb.left_top.y-r},{x=cx,y=bb.right_bottom.y+r},"
            "{x=bb.left_top.x-r,y=bb.left_top.y-r},{x=bb.right_bottom.x+r,y=bb.left_top.y-r},{x=bb.left_top.x-r,y=bb.right_bottom.y+r},{x=bb.right_bottom.x+r,y=bb.right_bottom.y+r}}; "
            "for _,pos in ipairs(c) do local e,why=consume_and_create(pos); if e then return e end end end; return nil end; "
            "local source=nearest_pole(engine,36) or build_near(engine); if not source then rcon.print(marker..'|ok=0|error=cannot-place-source-pole'); return end; "
            "local receiver=nearest_pole(lab,36) or build_near(lab); if not receiver then rcon.print(marker..'|ok=0|error=cannot-place-receiver-pole'); return end; "
            "local sx=source.position.x; local sy=source.position.y; local dx=receiver.position.x-sx; local dy=receiver.position.y-sy; "
            "local dist=math.sqrt(dx*dx+dy*dy); local segments=math.max(1,math.ceil(dist/6)); "
            "for i=1,segments-1 do local tx=sx+dx*i/segments; local ty=sy+dy*i/segments; local built=nil; "
            "local offsets={{0,0},{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}}; "
            "for _,o in ipairs(offsets) do local e,why=consume_and_create({x=tx+o[1],y=ty+o[2]}); if e then built=e; break end end; "
            "if not built then rcon.print(marker..'|ok=0|error=cannot-place-intermediate-pole'); return end; end; "
            "local okc,connected=pcall(function() return lab.is_connected_to_electric_network() end); "
            "if not (okc and connected) then rcon.print(marker..'|ok=0|error=lab-still-unpowered|created='..#created); return end; "
            "local total=#s.find_entities_filtered{name='small-electric-pole',force=f}; "
            "rcon.print(marker..'|ok=1|created='..#created..'|pole_count='..total..'|lab_connected=1|source_x='..source.position.x..'|source_y='..source.position.y..'|receiver_x='..receiver.position.x..'|receiver_y='..receiver.position.y); end; "
            "local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _queue_research_runtime(self, target: str) -> Dict[str, Any]:
        """Select research using Factorio's runtime API and verify the result.

        FLE 0.4.3's high-level ``set_research`` call can return without an
        execution error while leaving the force with no current technology.
        The runtime research queue is authoritative and can be checked in the
        same synchronized RCON command.
        """
        marker = json.dumps(RESEARCH_QUEUE_MARKER)
        target_lua = json.dumps(str(target))
        command = (
            "/sc local p=nil; local controller=''; ""if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then ""p=storage.agent_characters[1]; controller='agent-character'; ""elseif player and player.valid then p=player; controller='global-character'; ""elseif game.get_player(1) then p=game.get_player(1); controller='lua-player-1'; ""elseif game.connected_players[1] then p=game.connected_players[1]; controller='connected-player'; end; "
            "if not p then rcon.print(" + marker + "..'|ok=0|error=no-player') else "
            "local f=p.force; local target=" + target_lua + "; "
            "local t=f.technologies[target]; "
            "if not t then rcon.print(" + marker + "..'|ok=0|error=unknown-technology') "
            "elseif t.researched then rcon.print(" + marker + "..'|ok=1|already=1|current=') "
            "elseif not t.enabled then rcon.print(" + marker + "..'|ok=0|error=technology-disabled') "
            "else f.enable_research(); f.research_queue={target}; "
            "local cr=f.current_research; local current=cr and cr.name or ''; "
            "local ok=(current==target); "
            "rcon.print(" + marker + "..'|ok='..(ok and '1' or '0')..'|target='..target..'|current='..current); end; end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        line = next((line for line in raw.splitlines() if RESEARCH_QUEUE_MARKER in line), "")
        if not line:
            raise RuntimeError(f"research queue marker missing from RCON response: {raw[:400]!r}")
        payload = line.split(RESEARCH_QUEUE_MARKER, 1)[1].lstrip(": |")
        result: Dict[str, Any] = {"raw": raw, "target": str(target)}
        for token in payload.split("|"):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            result[key.strip()] = value.strip()
        if result.get("ok") != "1":
            raise RuntimeError(
                "Factorio did not select research "
                f"{target!r}: {result.get('error') or result.get('current') or raw!r}"
            )
        return result

    def _infrastructure_command(self) -> str:
        names = list(OPENENDED_TRACKED_ENTITIES)
        name_lua = "{" + ",".join(json.dumps(name) for name in names) + "}"
        tech_lua = "{" + ",".join(json.dumps(name) for name in RESEARCH_TARGETS) + "}"
        marker = json.dumps(INFRA_MARKER)
        # Use only Factorio runtime API fields so this query stays independent
        # of the Python-side FLE model classes.
        return (
            "/sc local p=nil; local controller=''; ""if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then ""p=storage.agent_characters[1]; controller='agent-character'; ""elseif player and player.valid then p=player; controller='global-character'; ""elseif game.get_player(1) then p=game.get_player(1); controller='lua-player-1'; ""elseif game.connected_players[1] then p=game.connected_players[1]; controller='connected-player'; end; "
            "if not p then rcon.print(" + marker + "..'|ERROR|no-player') else "
            "local s=p.surface; local f=p.force; local out={}; "
            "local names=" + name_lua + "; "
            "for _,n in ipairs(names) do "
            "local es=s.find_entities_filtered{name=n,force=f}; "
            "out[#out+1]='count__'..n..'='..#es; end; "
            "local function energy(n) local total=0; "
            "for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do "
            "total=total+(e.energy or 0); end; return total; end; "
            "out[#out+1]='steam_energy='..energy('steam-engine'); "
            "out[#out+1]='lab_energy='..energy('lab'); "
            "out[#out+1]='electric_drill_energy='..energy('electric-mining-drill'); "
            "local generated=0; local engine_connected=0; "
            "for _,e in ipairs(s.find_entities_filtered{name='steam-engine',force=f}) do "
            "local okg,g=pcall(function() return e.energy_generated_last_tick end); "
            "if okg and g then generated=generated+g end; "
            "local okc,c=pcall(function() return e.is_connected_to_electric_network() end); "
            "if okc and c then engine_connected=engine_connected+1 end; end; "
            "out[#out+1]='steam_generated_last_tick='..generated; "
            "out[#out+1]='steam_connected_count='..engine_connected; "
            "local lab_connected=0; local lab_science=0; local lab_working=0; "
            "for _,e in ipairs(s.find_entities_filtered{name='lab',force=f}) do "
            "local okc,c=pcall(function() return e.is_connected_to_electric_network() end); "
            "if okc and c then lab_connected=lab_connected+1 end; "
            "local oki,i=pcall(function() return e.get_item_count('automation-science-pack') end); "
            "if oki and i then lab_science=lab_science+i end; "
            "local oks,st=pcall(function() return e.status end); "
            "if oks and st==defines.entity_status.working then lab_working=lab_working+1 end; end; "
            "out[#out+1]='lab_connected_count='..lab_connected; "
            "out[#out+1]='lab_science_count='..lab_science; "
            "out[#out+1]='lab_working_count='..lab_working; "
            "local rq=f.research_queue or {}; out[#out+1]='research_queue_length='..#rq; "
            "local cr=f.current_research; "
            "out[#out+1]='current_research='..(cr and cr.name or ''); "
            "out[#out+1]='research_progress='..(f.research_progress or 0); "
            "local techs=" + tech_lua + "; "
            "for _,n in ipairs(techs) do local t=f.technologies[n]; "
            "local available=(t and t.enabled) and true or false; "
            "if available and t.prerequisites then for _,pre in pairs(t.prerequisites) do "
            "if not pre.researched then available=false; break; end; end; end; "
            "out[#out+1]='researched__'..n..'='..((t and t.researched) and '1' or '0'); "
            "out[#out+1]='available__'..n..'='..(available and '1' or '0'); end; "
            "rcon.print(" + marker + "..'|'..table.concat(out,'|')); end"
        )

    def _collect_infrastructure_state(self, *, force: bool = False) -> Dict[str, Any]:
        if not force and self.global_step % self.infrastructure_state_every != 0:
            return self.infrastructure_state

        state = dict(self.infrastructure_state)
        try:
            raw = str(self.instance.rcon_client.send_command(self._infrastructure_command()) or "")
            line = next((line for line in raw.splitlines() if INFRA_MARKER in line), "")
            if not line:
                raise RuntimeError(f"infrastructure marker missing from RCON response: {raw[:400]!r}")
            payload = line.split(INFRA_MARKER, 1)[1].lstrip(": |")
            if payload.startswith("ERROR|"):
                raise RuntimeError(payload)
            parsed: Dict[str, Any] = {}
            for token in payload.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.startswith("researched__") or key.startswith("available__"):
                    parsed[key] = value in {"1", "true", "True"}
                elif key == "current_research":
                    parsed[key] = value
                else:
                    parsed[key] = safe_float(value)
            state.update(parsed)
            state["query_ok"] = True
            state["query_error"] = ""
        except Exception as exc:
            state["query_ok"] = False
            state["query_error"] = f"{type(exc).__name__}: {exc}"

        for entity in OPENENDED_TRACKED_ENTITIES:
            key = f"count__{entity}"
            if key in state:
                self.entity_counts[entity] = float(state.get(key, 0.0))

        state["power_online"] = bool(
            safe_float(state.get("steam_generated_last_tick")) > 0.0
            or safe_float(state.get("steam_connected_count")) > 0.0
            or safe_float(state.get("lab_connected_count")) > 0.0
            or safe_float(state.get("steam_energy")) > 0.0
            or safe_float(state.get("lab_energy")) > 0.0
            or safe_float(state.get("electric_drill_energy")) > 0.0
        )
        self.flags["power_online"] = bool(state["power_online"])
        self.infrastructure_state = state
        return state

    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(len(self.actions), dtype=np.int8)
        for index, spec in enumerate(self.actions):
            for item, amount in spec.required_items.items():
                if self.inventory.get(item, 0.0) + 1e-9 < float(amount):
                    mask[index] = 0
                    break
            if not mask[index]:
                continue
            for entity, amount in spec.required_entities.items():
                if self.entity_counts.get(entity, 0.0) + 1e-9 < float(amount):
                    mask[index] = 0
                    break
            if not mask[index]:
                continue
            if spec.single_use_flag and self.flags.get(spec.single_use_flag, False):
                mask[index] = 0

        def set_valid(name: str, valid: bool) -> None:
            try:
                mask[self.action_index(name)] = int(bool(valid))
            except KeyError:
                pass

        # Extraction/transfer actions are enabled only after there has been
        # sufficient game time for output to exist. FLE still performs the
        # authoritative check and reports an action error when no output exists.
        set_valid(
            "collect_iron_plate",
            self.entity_counts.get("stone-furnace", 0.0) > 0
            and self.steps_since_transfer >= 3,
        )
        set_valid(
            "insert_hand_mined_ore",
            self.entity_counts.get("stone-furnace", 0.0) > 0
            and self.inventory.get("iron-ore", 0.0) >= 1,
        )
        set_valid(
            "transfer_chest_ore_to_furnace",
            self.entity_counts.get("wooden-chest", 0.0) > 0
            and self.entity_counts.get("stone-furnace", 0.0) > 0,
        )

        # Do not repeatedly select a technology that has already completed.
        for target in RESEARCH_TARGETS:
            set_valid(
                f"research_{target.replace('-', '_')}",
                self.entity_counts.get("lab", 0.0) > 0
                and bool(self.infrastructure_state.get(f"available__{target}", True))
                and not bool(self.infrastructure_state.get(f"researched__{target}", False)),
            )

        set_valid("noop", True)
        set_valid("wait", True)
        set_valid("wait_research", True)
        return mask

    def action_mask_diagnostics(self, action_name: str) -> Dict[str, Any]:
        """Explain why an action is currently masked."""
        index = self.action_index(action_name)
        spec = self.actions[index]
        reasons: List[str] = []
        missing_items: Dict[str, Dict[str, float]] = {}
        missing_entities: Dict[str, Dict[str, float]] = {}

        for item, required in spec.required_items.items():
            available = float(self.inventory.get(item, 0.0))
            required_f = float(required)
            if available + 1e-9 < required_f:
                missing_items[item] = {
                    "required": required_f,
                    "available": available,
                }
                reasons.append(
                    f"inventory {item}: requires {required_f:g}, has {available:g}"
                )

        for entity, required in spec.required_entities.items():
            available = float(self.entity_counts.get(entity, 0.0))
            required_f = float(required)
            if available + 1e-9 < required_f:
                missing_entities[entity] = {
                    "required": required_f,
                    "available": available,
                }
                reasons.append(
                    f"entity {entity}: requires {required_f:g}, has {available:g}"
                )

        if spec.single_use_flag and self.flags.get(spec.single_use_flag, False):
            reasons.append(f"single-use flag already set: {spec.single_use_flag}")

        mask = self.get_action_mask()
        valid = bool(mask[index])
        if not valid and not reasons:
            reasons.append("disabled by an action-specific mask rule")

        return {
            "action": action_name,
            "index": int(index),
            "valid": valid,
            "category": spec.category,
            "required_items": dict(spec.required_items),
            "required_entities": dict(spec.required_entities),
            "missing_items": missing_items,
            "missing_entities": missing_entities,
            "single_use_flag": spec.single_use_flag,
            "reasons": reasons,
            "inventory": dict(self.inventory),
            "entity_counts": dict(self.entity_counts),
            "flags": dict(self.flags),
        }

    @staticmethod
    def _weighted_output_delta(
        before: Mapping[str, float],
        after: Mapping[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        components: Dict[str, float] = {}
        total = 0.0
        for item, weight in AUTOMATED_OUTPUT_WEIGHTS.items():
            delta = max(0.0, safe_float(after.get(item)) - safe_float(before.get(item)))
            if delta > 0:
                components[item] = delta
                total += delta * weight
        return total, components

    @staticmethod
    def _researched_count(state: Mapping[str, Any]) -> int:
        return sum(bool(state.get(f"researched__{target}", False)) for target in RESEARCH_TARGETS)

    def _observation(self) -> np.ndarray:
        values: List[float] = []
        values.extend(
            min(10.0, math.log1p(max(0.0, self.inventory.get(item, 0.0))))
            for item in OPENENDED_TRACKED_ITEMS
        )
        values.extend(
            min(10.0, math.log1p(max(0.0, self.entity_counts.get(entity, 0.0))))
            for entity in OPENENDED_TRACKED_ENTITIES
        )
        flag_names = (
            "drill_placed",
            "furnace_placed",
            "drill_fuelled",
            "furnace_fuelled",
            "pump_boiler_connected",
            "boiler_engine_connected",
            "electric_drill_connected",
            "lab_connected",
            "power_online",
        )
        values.extend(float(bool(self.flags.get(name, False))) for name in flag_names)
        values.extend(
            (
                min(10.0, math.log1p(max(0.0, safe_float(self.infrastructure_state.get("steam_energy")))) / 10.0),
                min(10.0, math.log1p(max(0.0, safe_float(self.infrastructure_state.get("lab_energy")))) / 10.0),
                min(10.0, math.log1p(max(0.0, safe_float(self.infrastructure_state.get("electric_drill_energy")))) / 10.0),
                float(np.clip(safe_float(self.infrastructure_state.get("research_progress")), 0.0, 1.0)),
            )
        )
        values.extend(
            float(bool(self.infrastructure_state.get(f"researched__{target}", False)))
            for target in RESEARCH_TARGETS
        )
        values.extend(
            float(bool(self.infrastructure_state.get(f"available__{target}", True)))
            for target in RESEARCH_TARGETS
        )
        values.extend(
            (
                float(self._researched_count(self.infrastructure_state)) / max(1, len(RESEARCH_TARGETS)),
                min(10.0, math.log1p(max(0.0, self.automated_output_total)) / 5.0),
                float(np.clip(self.research_reward_total / 100.0, -10.0, 10.0)),
                float(np.clip(self.power_reward_total / 100.0, -10.0, 10.0)),
                float(self.get_action_mask().mean()),
                min(10.0, math.log1p(max(0, self.global_step)) / 10.0),
            )
        )
        result = np.asarray(values, dtype=np.float32)
        expected_shape = (
            int(getattr(self, "_base_observation_feature_count", len(values))),
        )
        if result.shape != expected_shape:
            raise RuntimeError(
                f"Base observation shape mismatch: got {result.shape}, expected {expected_shape}"
            )
        return result

    def behavior_descriptors(self) -> Dict[str, float]:
        return {
            "entity_count": float(sum(self.entity_counts.values())),
            "electric_entity_count": float(
                self.entity_counts.get("electric-mining-drill", 0.0)
                + self.entity_counts.get("steam-engine", 0.0)
                + self.entity_counts.get("lab", 0.0)
            ),
            "automated_output_total": float(self.automated_output_total),
            "researched_count": float(self._researched_count(self.infrastructure_state)),
            "power_online": float(bool(self.infrastructure_state.get("power_online", False))),
            "action_diversity": float(len(self.successful_action_counts)),
        }

    def benchmark_fitness(self) -> float:
        return float(
            self.automated_output_total
            + 25.0 * self._researched_count(self.infrastructure_state)
            + 10.0 * bool(self.infrastructure_state.get("power_online", False))
            + 0.5 * sum(self.entity_counts.values())
        )

    def _base_info(self) -> Dict[str, Any]:
        info = super()._base_info()
        info.update(
            {
                "openended": True,
                "terminal_goal": None,
                "infrastructure_state": json_safe(self.infrastructure_state),
                "automated_output_total": float(self.automated_output_total),
                "researched_count": int(self._researched_count(self.infrastructure_state)),
                "current_research": str(self.infrastructure_state.get("current_research", "")),
                "research_progress": float(safe_float(self.infrastructure_state.get("research_progress"))),
                "power_online": bool(self.infrastructure_state.get("power_online", False)),
                "openended_reward_parts": dict(self.last_openended_reward_parts),
                "episode_success": False,
            }
        )
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        options = dict(options or {})
        observation, info = super().reset(seed=seed, options=options)
        del observation, info

        # Reinitialise only when the caller requested a hard factory reset.
        if bool(options.get("hard_reset", self.hard_reset)):
            self.infrastructure_state = self._empty_infrastructure_state()
            self.previous_infrastructure_state = dict(self.infrastructure_state)
            self.automated_output_total = 0.0
            self.research_reward_total = 0.0
            self.power_reward_total = 0.0
            self.milestones_seen.clear()

        try:
            self._entity_records()
        except Exception:
            pass
        self._collect_infrastructure_state(force=True)
        self.previous_infrastructure_state = dict(self.infrastructure_state)
        observation = self._observation()
        info = self._base_info()
        info.update({"action_mask": self.action_masks(), "hard_reset": bool(options.get("hard_reset", self.hard_reset))})
        return observation, info

    def step(self, action: int):
        before_output = dict(self.last_production_stats.get("output", {}))
        before_infrastructure = dict(self.infrastructure_state)
        before_entities = dict(self.entity_counts)

        observation, base_reward, _terminated, _truncated, info = super().step(action)
        del observation

        action_name = str(info.get("action_name", ""))
        action_category = next(
            (spec.category for spec in self.actions if spec.name == action_name),
            "",
        )

        runtime_action: Dict[str, Any] = {}
        runtime_event = ""
        if not str(info.get("action_error", "")):
            try:
                if action_name == "place_small_electric_pole":
                    runtime_action = self._runtime_place_near(
                        entity_name="small-electric-pole",
                        item_name="small-electric-pole",
                        anchor_name="steam-engine",
                        minimum_radius=1,
                        maximum_radius=4,
                    )
                    runtime_event = "electric_pole_placed"
                elif action_name == "place_lab":
                    runtime_action = self._runtime_place_near(
                        entity_name="lab",
                        item_name="lab",
                        anchor_name="steam-engine",
                        minimum_radius=3,
                        maximum_radius=9,
                    )
                    runtime_event = "lab_placed"
                elif action_name == "connect_lab_to_power":
                    runtime_action = self._runtime_connect_lab_power()
                    runtime_event = "lab_connected"
            except Exception as exc:
                info["action_error"] = f"{type(exc).__name__}: {exc}"
                self.total_action_errors += 1
                base_reward -= self.execution_error_penalty

        if runtime_action:
            info["runtime_action"] = runtime_action
            # Runtime construction happens after the base transition has
            # sampled inventory, so refresh before masks/rewards are computed.
            self._collect_compact_state(include_production_stats=False)

        research_targets_by_action = {
            "research_automation": "automation",
            "research_logistics": "logistics",
            "research_steel_processing": "steel-processing",
            "research_electronics": "electronics",
        }
        research_selection: Dict[str, Any] = {}
        selected_research_event = ""
        target = research_targets_by_action.get(action_name)
        if target and not str(info.get("action_error", "")):
            try:
                research_selection = self._queue_research_runtime(target)
                selected_research_event = f"research_selected_{target.replace('-', '_')}"
            except Exception as exc:
                info["action_error"] = f"{type(exc).__name__}: {exc}"
                self.total_action_errors += 1
                base_reward -= self.execution_error_penalty

        events = set(info.get("action_events", []) or [])
        if selected_research_event:
            events.add(selected_research_event)
        if runtime_event:
            events.add(runtime_event)
        if selected_research_event or runtime_event:
            info["action_events"] = sorted(events)
        if research_selection:
            info["research_selection"] = research_selection
        event_flag_map = {
            "drill_placed": "drill_placed",
            "furnace_placed": "furnace_placed",
            "drill_fuelled": "drill_fuelled",
            "furnace_fuelled": "furnace_fuelled",
            "pump_boiler_connected": "pump_boiler_connected",
            "boiler_engine_connected": "boiler_engine_connected",
            "electric_drill_connected": "electric_drill_connected",
            "lab_connected": "lab_connected",
        }
        for event, flag in event_flag_map.items():
            if event in events:
                self.flags[flag] = True

        force_infra = action_category in {"power", "research"} or action_name in {"wait", "wait_research"}
        self._collect_infrastructure_state(force=force_infra)
        # The live network query is authoritative. Do not permanently mask a
        # retry merely because an action printed an optimistic event.
        self.flags["lab_connected"] = bool(
            safe_float(self.infrastructure_state.get("lab_connected_count")) > 0.0
        )

        after_output = dict(self.last_production_stats.get("output", {}))
        output_reward_raw, output_components = self._weighted_output_delta(before_output, after_output)
        output_reward = self.automated_output_reward_scale * output_reward_raw
        self.automated_output_total += output_reward_raw

        progress_before = safe_float(before_infrastructure.get("research_progress"))
        progress_after = safe_float(self.infrastructure_state.get("research_progress"))
        progress_delta = max(0.0, progress_after - progress_before)
        research_progress_component = self.research_progress_reward * progress_delta
        researched_before = self._researched_count(before_infrastructure)
        researched_after = self._researched_count(self.infrastructure_state)
        new_research_count = max(0, researched_after - researched_before)
        research_completion_component = self.research_completion_reward * new_research_count
        research_reward = research_progress_component + research_completion_component
        self.research_reward_total += research_reward

        power_before = bool(before_infrastructure.get("power_online", False))
        power_after = bool(self.infrastructure_state.get("power_online", False))
        power_component = 0.0
        if power_after and not power_before and "power_online" not in self.milestones_seen:
            power_component = self.first_power_reward
            self.milestones_seen.add("power_online")
        self.power_reward_total += power_component

        entity_growth_raw = 0.0
        for name in OPENENDED_TRACKED_ENTITIES:
            before = max(0.0, safe_float(before_entities.get(name)))
            after = max(0.0, safe_float(self.entity_counts.get(name)))
            if after > before:
                entity_growth_raw += math.log1p(after) - math.log1p(before)
        infrastructure_component = self.infrastructure_growth_reward * entity_growth_raw

        step_component = -self.openended_step_penalty
        extra_reward = (
            output_reward
            + research_reward
            + power_component
            + infrastructure_component
            + step_component
        )
        reward = float(base_reward + extra_reward)
        self.episode_return += extra_reward
        self.last_openended_reward_parts = {
            "base_error_and_invalid_reward": float(base_reward),
            "automated_output": float(output_reward),
            "research_progress": float(research_progress_component),
            "research_completion": float(research_completion_component),
            "power_online": float(power_component),
            "infrastructure_growth": float(infrastructure_component),
            "step_penalty": float(step_component),
            "total": float(reward),
        }

        final_info = dict(info)
        final_info.update(self._base_info())
        final_info.update(
            {
                "reward_parts": dict(self.last_openended_reward_parts),
                "automated_output_delta": output_components,
                "action_mask": self.action_masks(),
                "episode_success": False,
            }
        )
        observation = self._observation()

        # The trainer, not the factory state, controls run termination.
        return observation, reward, False, False, final_info


__all__ = [
    "AUTOMATED_OUTPUT_WEIGHTS",
    "DEFAULT_OPENENDED_INVENTORY",
    "FactorioOpenEndedEnv",
    "OPENENDED_ACTIONS",
    "OPENENDED_TRACKED_ENTITIES",
    "OPENENDED_TRACKED_ITEMS",
    "RESEARCH_TARGETS",
]
