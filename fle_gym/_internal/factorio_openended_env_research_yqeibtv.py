from __future__ import annotations

"""runtime expanded, algorithm-neutral Factorio environment.

The environment keeps a fixed ``Discrete`` action catalogue and a numeric
``Box`` observation so the exact same task can be used by MaskablePPO,
categorical/discretised SAC, and MAP-Elites.

runtime addresses the main runtime run findings:

* live entity counts are synchronised into the observation after every step;
* electrical network existence, generation capacity, active generation, and
  connected/unpowered consumers are represented separately;
* geometry masks and execution use the same deterministic layout searches;
* unreliable legacy power/furnace actions are redirected to the runtime runtime
  planners instead of the old FLE helpers;
* trigger-technology reward is separated from laboratory-research reward;
* additional crafting, automation, red/green science, and early/mid-game
  research actions are exposed to every algorithm;
* no curriculum hides actions: masks only represent live feasibility.

Training remains goal-free. ``training_horizon_steps`` is a time-limit
truncation only; set it to zero for one persistent evaluation factory.
"""

from collections import Counter
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from factorio_openended_env_research_ajweuzp import FactorioOpenEndedEnvAjweuzp
from factorio_openended_env_recipe import (
    FactorioOpenEndedEnvRuntime,
    MACHINE_SOURCE_KIND,
    PROGRESSION_MILESTONE_REWARDS,
)
from factorio_openended_env_research_ajweuzp import (
    MACHINE_OUTPUT_WEIGHTS,
    MANUAL_CRAFT_OUTPUTS,
    Runtime_ACTIONS,
)
from factorio_transition_env import ActionSpec, _dedent, json_safe, safe_float, spaces


Runtime_VERSION = "17.0"
Runtime_STATE_MARKER = "__FLE_RL_STATE_runtime__"
Runtime_RUNTIME_MARKER = "__FLE_RL_RUNTIME_runtime__"


# ---------------------------------------------------------------------------
# Expanded capability catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CraftSpec:
    action: str
    recipe: str
    craft_count: int
    output_item: str
    output_per_craft: float
    description: str


Runtime_CRAFT_SPECS: Tuple[CraftSpec, ...] = (
    CraftSpec("craft_copper_cables", "copper-cable", 10, "copper-cable", 2.0, "Queue ten copper-cable crafts."),
    CraftSpec("craft_pipes", "pipe", 10, "pipe", 1.0, "Queue ten pipe crafts."),
    CraftSpec("craft_underground_belts", "underground-belt", 2, "underground-belt", 2.0, "Queue two underground-belt crafts."),
    CraftSpec("craft_splitters", "splitter", 2, "splitter", 1.0, "Queue two splitter crafts."),
    CraftSpec("craft_long_handed_inserters", "long-handed-inserter", 4, "long-handed-inserter", 1.0, "Queue four long-handed inserters."),
    CraftSpec("craft_fast_inserters", "fast-inserter", 4, "fast-inserter", 1.0, "Queue four fast inserters."),
    CraftSpec("craft_steel_chests", "steel-chest", 2, "steel-chest", 1.0, "Queue two steel chests."),
    CraftSpec("craft_medium_electric_poles", "medium-electric-pole", 4, "medium-electric-pole", 1.0, "Queue four medium electric poles."),
    CraftSpec("craft_steel_furnace", "steel-furnace", 1, "steel-furnace", 1.0, "Queue one steel furnace."),
    CraftSpec("craft_assembling_machine_2", "assembling-machine-2", 1, "assembling-machine-2", 1.0, "Queue one assembling machine 2."),
    CraftSpec("craft_engine_units", "engine-unit", 5, "engine-unit", 1.0, "Queue five engine units."),
    # The vanilla logistic-science recipe produces two packs per craft.
    CraftSpec("craft_logistic_science", "logistic-science-pack", 5, "logistic-science-pack", 2.0, "Queue five logistic-science crafts (normally ten packs)."),
    CraftSpec("craft_military_science", "military-science-pack", 5, "military-science-pack", 2.0, "Queue five military-science crafts when unlocked."),
    CraftSpec("craft_storage_tank", "storage-tank", 1, "storage-tank", 1.0, "Queue one storage tank."),
    CraftSpec("craft_pumpjack", "pumpjack", 1, "pumpjack", 1.0, "Queue one pumpjack."),
    CraftSpec("craft_oil_refinery", "oil-refinery", 1, "oil-refinery", 1.0, "Queue one oil refinery."),
    CraftSpec("craft_chemical_plants", "chemical-plant", 2, "chemical-plant", 1.0, "Queue two chemical plants."),
    CraftSpec("craft_repair_packs", "repair-pack", 5, "repair-pack", 1.0, "Queue five repair packs."),
    CraftSpec("craft_lamps", "small-lamp", 4, "small-lamp", 1.0, "Queue four lamps."),
    CraftSpec("craft_firearm_magazines", "firearm-magazine", 10, "firearm-magazine", 1.0, "Queue ten firearm magazines."),
    CraftSpec("craft_piercing_magazines", "piercing-rounds-magazine", 5, "piercing-rounds-magazine", 1.0, "Queue five piercing magazines."),
    CraftSpec("craft_grenades", "grenade", 5, "grenade", 1.0, "Queue five grenades."),
    CraftSpec("craft_stone_walls", "stone-wall", 10, "stone-wall", 1.0, "Queue ten stone walls."),
    CraftSpec("craft_gun_turrets", "gun-turret", 2, "gun-turret", 1.0, "Queue two gun turrets."),
    CraftSpec("craft_radars", "radar", 2, "radar", 1.0, "Queue two radars."),
    CraftSpec("craft_train_stops", "train-stop", 2, "train-stop", 1.0, "Queue two train stops."),
    CraftSpec("craft_rail_signals", "rail-signal", 4, "rail-signal", 1.0, "Queue four rail signals."),
    CraftSpec("craft_rail_chain_signals", "rail-chain-signal", 4, "rail-chain-signal", 1.0, "Queue four rail chain signals."),
    CraftSpec("craft_locomotive", "locomotive", 1, "locomotive", 1.0, "Queue one locomotive."),
    CraftSpec("craft_cargo_wagons", "cargo-wagon", 2, "cargo-wagon", 1.0, "Queue two cargo wagons."),
)

CRAFT_BY_ACTION: Dict[str, CraftSpec] = {spec.action: spec for spec in Runtime_CRAFT_SPECS}

# Technologies are queried from the running game. Missing, disabled, trigger,
# already-researched, or prerequisite-blocked technologies are masked out.
Runtime_RESEARCH_TARGETS: Tuple[str, ...] = (
    "automation",
    "logistics",
    "steel-processing",
    "logistic-science-pack",
    "military",
    # Factorio 2 removed ``optics``.  ``lamp`` is its valid early-game,
    # automation-science successor in this fixed policy slot.
    "lamp",
    "stone-wall",
    "gun-turret",
    # ``steel-axe`` is a player-trigger technology, which cannot be queued by
    # headless FLE.  Keep the mining-progression decision as a real lab tech.
    "electric-mining-drill",
    "toolbelt",
    "automation-2",
    "logistics-2",
    "advanced-material-processing",
    "engine",
    "fluid-handling",
    "electric-energy-distribution-1",
    "fast-inserter",
    "railway",
    # ``oil-processing`` is a Factorio 2 trigger technology.  The advanced
    # oil decision is queueable and preserves an oil-production research slot.
    "advanced-oil-processing",
    "circuit-network",
    "military-science-pack",
    "physical-projectile-damage-1",
    "weapon-shooting-speed-1",
    "automated-rail-transportation",
    # Factorio 2 no longer exposes ``rail-signals`` as a technology.  Use a
    # rail-related, queueable technology in the same fixed policy slot.
    "braking-force-1",
    "flammables",
    "plastics",
    "sulfur-processing",
)

# Trigger technologies are observed dynamically. These names are useful for
# reward/milestone reporting but are never forced or queued by the environment.
Runtime_TRIGGER_TECHNOLOGIES: Tuple[str, ...] = (
    "electronics",
    "automation-science-pack",
)

# Generic assembly cells. Inputs are stocked from the player inventory and from
# any other runtime output chest. The policy chooses build and stock actions.
Runtime_CELL_INPUTS: Dict[str, Tuple[str, ...]] = {
    "gear": ("iron-plate",),
    "cable": ("copper-plate",),
    "circuit": ("iron-plate", "copper-cable"),
    "belt": ("iron-plate", "iron-gear-wheel"),
    "inserter": ("iron-plate", "iron-gear-wheel", "electronic-circuit"),
    "science": ("copper-plate", "iron-gear-wheel"),
    "logistic-science": ("transport-belt", "inserter"),
    "engine": ("steel-plate", "iron-gear-wheel", "pipe"),
    "military-science": ("piercing-rounds-magazine", "grenade", "stone-wall"),
}
Runtime_CELL_RECIPES: Dict[str, str] = {
    "gear": "iron-gear-wheel",
    "cable": "copper-cable",
    "circuit": "electronic-circuit",
    "belt": "transport-belt",
    "inserter": "inserter",
    "science": "automation-science-pack",
    "logistic-science": "logistic-science-pack",
    "engine": "engine-unit",
    "military-science": "military-science-pack",
}

Runtime_NEW_CELL_ACTIONS: Dict[str, str] = {
    "build_cable_assembly_cell": "cable",
    "build_circuit_assembly_cell": "circuit",
    "build_belt_assembly_cell": "belt",
    "build_inserter_assembly_cell": "inserter",
    "build_logistic_science_assembly_cell": "logistic-science",
    "build_engine_assembly_cell": "engine",
    "build_military_science_assembly_cell": "military-science",
}
Runtime_STOCK_ACTIONS: Dict[str, str] = {
    "stock_cable_assembler": "cable",
    "stock_circuit_assembler": "circuit",
    "stock_belt_assembler": "belt",
    "stock_inserter_assembler": "inserter",
    "stock_logistic_science_assembler": "logistic-science",
    "stock_engine_assembler": "engine",
    "stock_military_science_assembler": "military-science",
}

Runtime_TRACKED_ITEMS: Tuple[str, ...] = (
    "copper-cable",
    "underground-belt",
    "splitter",
    "long-handed-inserter",
    "fast-inserter",
    "steel-chest",
    "medium-electric-pole",
    "steel-furnace",
    "assembling-machine-2",
    "engine-unit",
    "logistic-science-pack",
    "military-science-pack",
    "storage-tank",
    "pumpjack",
    "oil-refinery",
    "chemical-plant",
    "repair-pack",
    "small-lamp",
    "firearm-magazine",
    "piercing-rounds-magazine",
    "grenade",
    "stone-wall",
    "gun-turret",
    "radar",
    "train-stop",
    "rail-signal",
    "rail-chain-signal",
    "locomotive",
    "cargo-wagon",
)

Runtime_LIVE_ENTITY_NAMES: Tuple[str, ...] = (
    "burner-mining-drill",
    "electric-mining-drill",
    "stone-furnace",
    "steel-furnace",
    "offshore-pump",
    "boiler",
    "steam-engine",
    "small-electric-pole",
    "medium-electric-pole",
    "lab",
    "wooden-chest",
    "steel-chest",
    "transport-belt",
    "underground-belt",
    "splitter",
    "inserter",
    "long-handed-inserter",
    "fast-inserter",
    "assembling-machine-1",
    "assembling-machine-2",
    "storage-tank",
    "pumpjack",
    "oil-refinery",
    "chemical-plant",
)

# Extend the strict production whitelist. Unknown products, fluids, transfers,
# and manual work still receive zero machine-production reward.
MACHINE_OUTPUT_WEIGHTS.update(
    {
        "copper-cable": 0.20,
        "transport-belt": 0.70,
        "inserter": 1.00,
        "engine-unit": 2.00,
        "logistic-science-pack": 4.00,
        "military-science-pack": 5.00,
    }
)
MACHINE_SOURCE_KIND.update(
    {
        "copper-cable": "cable-assembler",
        "transport-belt": "belt-assembler",
        "inserter": "inserter-assembler",
        "engine-unit": "engine-assembler",
        "logistic-science-pack": "logistic-science-assembler",
        "military-science-pack": "military-science-assembler",
    }
)

Runtime_PROGRESSION_REWARDS: Dict[str, float] = {
    **PROGRESSION_MILESTONE_REWARDS,
    "electronics_triggered": 2.0,
    "lab_gears_ready": 1.0,
    "lab_circuits_ready": 1.0,
    "lab_belts_ready": 1.0,
    "lab_all_ingredients_ready": 3.0,
    "copper_cable_available": 1.0,
    "circuit_cell_built": 4.0,
    "belt_cell_built": 4.0,
    "inserter_cell_built": 4.0,
    "logistic_science_unlocked": 6.0,
    "logistic_science_available": 6.0,
    "logistic_science_cell_built": 8.0,
    "military_science_unlocked": 7.0,
    "military_science_available": 7.0,
    "military_science_cell_built": 9.0,
    "green_science_research_started": 4.0,
    "automation_2_researched": 10.0,
    "logistics_2_researched": 10.0,
    "engine_researched": 8.0,
    "oil_processing_researched": 12.0,
}
# The inherited runtime step calculates milestones before runtime post-processing.
# Updating the shared mapping ensures new milestones receive their reward on
# either path instead of being marked seen with a zero value.
PROGRESSION_MILESTONE_REWARDS.update(Runtime_PROGRESSION_REWARDS)



def _marker_action(name: str, category: str, description: str) -> ActionSpec:
    return ActionSpec(
        name=name,
        category=category,
        code=f'print("__FLE_RL_EVENT__:runtime_{name}_requested")',
        description=description,
    )


def _build_actions_yqeibtv() -> Tuple[ActionSpec, ...]:
    actions: List[ActionSpec] = []
    for spec in Runtime_ACTIONS:
        # Redirect the unreliable legacy geometry helpers to runtime transactional
        # runtime planners while preserving stable action names.
        if spec.name in {"place_furnace_near_drill", "connect_electric_drill_to_power"}:
            spec = replace(
                spec,
                code=f'print("__FLE_RL_EVENT__:runtime_{spec.name}_requested")',
                single_use_flag=None,
            )
        actions.append(spec)

    existing = {spec.name for spec in actions}
    for craft in Runtime_CRAFT_SPECS:
        if craft.action not in existing:
            actions.append(_marker_action(craft.action, "craft", craft.description))

    for action, key in Runtime_NEW_CELL_ACTIONS.items():
        actions.append(
            _marker_action(
                action,
                "automation",
                f"Build one powered {Runtime_CELL_RECIPES[key]} assembly cell.",
            )
        )
    for action, key in Runtime_STOCK_ACTIONS.items():
        actions.append(
            _marker_action(
                action,
                "transfer",
                f"Stock the {Runtime_CELL_RECIPES[key]} assembly cell from available buffers.",
            )
        )
    actions.extend(
        (
            _marker_action(
                "build_steel_smelting_cell",
                "automation",
                "Build a fuelled iron-plate to steel-plate smelting cell.",
            ),
            _marker_action(
                "stock_steel_smelting_cell",
                "transfer",
                "Stock the steel smelting cell with iron plates and fuel.",
            ),
            _marker_action(
                "feed_labs_all_science",
                "research",
                "Move available automation and logistic science into laboratories.",
            ),
            ActionSpec(
                name="wait_crafting",
                category="wait",
                code="sleep(5)",
                description="Advance simulation time so queued hand crafts and machines can complete.",
            ),
        )
    )

    existing = {spec.name for spec in actions}
    for technology in Runtime_RESEARCH_TARGETS:
        action_name = f"research_{technology.replace('-', '_')}"
        if action_name not in existing:
            actions.append(
                _marker_action(
                    action_name,
                    "research",
                    f"Queue {technology} if it is a live, available laboratory technology.",
                )
            )
    return tuple(actions)


Runtime_ACTIONS: Tuple[ActionSpec, ...] = _build_actions_yqeibtv()


class FactorioOpenEndedEnvYqeibtv(FactorioOpenEndedEnvRuntime):
    """Expanded shared environment for PPO, discrete SAC, and MAP-Elites."""

    def __init__(
        self,
        *,
        trigger_technology_reward: float = 8.0,
        lab_research_completion_reward: float = 20.0,
        max_cells_per_recipe: int = 2,
        **kwargs: Any,
    ) -> None:
        # Long crafts can finish several transitions after the initiating action.
        kwargs.setdefault("manual_guard_ttl_steps", 64)
        super().__init__(**kwargs)
        self._runtime_feature_names = tuple(self.observation_feature_names)
        self.actions = Runtime_ACTIONS
        self.action_space = spaces.Discrete(len(self.actions))
        self.trigger_technology_reward = float(trigger_technology_reward)
        self.lab_research_completion_reward = float(lab_research_completion_reward)
        self.max_cells_per_recipe = max(1, int(max_cells_per_recipe))
        self.runtime_runtime_failures: Counter[str] = Counter()
        self.runtime_trigger_completions: Counter[str] = Counter()
        self.runtime_lab_research_completions: Counter[str] = Counter()

        extra_names: List[str] = []
        extra_names.extend(f"inventory_runtime__{item}" for item in Runtime_TRACKED_ITEMS)
        recipes = tuple(dict.fromkeys([s.recipe for s in Runtime_CRAFT_SPECS] + list(Runtime_CELL_RECIPES.values())))
        self.runtime_recipe_names = recipes
        extra_names.extend(f"recipe_runtime__{name}" for name in recipes)
        extra_names.extend(f"researched_runtime__{name}" for name in Runtime_RESEARCH_TARGETS)
        extra_names.extend(f"available_runtime__{name}" for name in Runtime_RESEARCH_TARGETS)
        extra_names.extend(
            (
                "power_runtime__network_exists",
                "power_runtime__generation_capacity",
                "power_runtime__active_generation",
                "power_runtime__connected_consumers",
                "power_runtime__unpowered_consumers",
                "power_runtime__steam_available",
                "science_runtime__red_in_labs",
                "science_runtime__green_in_labs",
                "science_runtime__military_in_labs",
                "science_runtime__red_buffered",
                "science_runtime__green_buffered",
                "science_runtime__military_buffered",
                "crafting_runtime__queue_length",
            )
        )
        extra_names.extend(f"entities_runtime__{name}" for name in Runtime_LIVE_ENTITY_NAMES)
        extra_names.extend(f"cell_runtime__{key}" for key in Runtime_CELL_RECIPES)
        extra_names.append("cell_runtime__steel")
        self.runtime_observation_feature_names = tuple(extra_names)
        self.observation_feature_names = self._runtime_feature_names + self.runtime_observation_feature_names
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )
        self._write_config_yqeibtv()

    # ------------------------------------------------------------------
    # Configuration and schema
    # ------------------------------------------------------------------
    def schema(self) -> Dict[str, Any]:
        schema = super().schema()
        schema.update(
            {
                "version": Runtime_VERSION,
                "environment": type(self).__name__,
                "research_targets": list(Runtime_RESEARCH_TARGETS),
                "trigger_technologies": list(Runtime_TRIGGER_TECHNOLOGIES),
                "expanded_crafting": [spec.__dict__ for spec in Runtime_CRAFT_SPECS],
                "assembly_cells": {
                    key: {"recipe": Runtime_CELL_RECIPES[key], "inputs": list(inputs)}
                    for key, inputs in Runtime_CELL_INPUTS.items()
                },
                "action_space": {
                    "type": "Discrete",
                    "count": int(self.action_space.n),
                    "actions": [
                        {
                            "index": index,
                            "name": spec.name,
                            "category": spec.category,
                            "description": spec.description,
                        }
                        for index, spec in enumerate(self.actions)
                    ],
                },
                "observation_space": {
                    "type": "Box",
                    "shape": list(self.observation_space.shape),
                    "features": list(self.observation_feature_names),
                },
                "compatibility": ["MaskablePPO", "CategoricalDiscreteSAC", "MAP-Elites"],
                "all_actions_exposed": True,
                "curriculum_action_hiding": False,
            }
        )
        return schema

    def _write_config_yqeibtv(self) -> None:
        payload = {
            "version": Runtime_VERSION,
            "action_count": int(self.action_space.n),
            "observation_shape": list(self.observation_space.shape),
            "research_targets": list(Runtime_RESEARCH_TARGETS),
            "craft_actions": [spec.__dict__ for spec in Runtime_CRAFT_SPECS],
            "cell_recipes": dict(Runtime_CELL_RECIPES),
            "cell_inputs": {key: list(value) for key, value in Runtime_CELL_INPUTS.items()},
            "reward": {
                "trigger_technology_reward": self.trigger_technology_reward,
                "lab_research_completion_reward": self.lab_research_completion_reward,
                "progression_milestones": Runtime_PROGRESSION_REWARDS,
            },
            "screenshots_optional": True,
            "terminal_goal": None,
            "training_horizon_steps": int(self.training_horizon_steps),
        }
        path = Path(self.run_dir) / "runtime_environment_config.json"
        path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        self.export_schema(Path(self.run_dir) / "environment_schema_runtime.json")

    # ------------------------------------------------------------------
    # Live state and authoritative entity/inventory synchronisation
    # ------------------------------------------------------------------
    def _state_command_yqeibtv(self) -> str:
        recipes = tuple(dict.fromkeys(self.runtime_recipe_names + ("lab", "automation-science-pack", "iron-gear-wheel", "transport-belt", "inserter", "assembling-machine-1")))
        recipe_lua = "{" + ",".join(json.dumps(name) for name in recipes) + "}"
        techs = tuple(dict.fromkeys(Runtime_TRIGGER_TECHNOLOGIES + Runtime_RESEARCH_TARGETS))
        tech_lua = "{" + ",".join(json.dumps(name) for name in techs) + "}"
        items = tuple(dict.fromkeys(Runtime_TRACKED_ITEMS + ("iron-plate", "copper-plate", "steel-plate", "iron-gear-wheel", "electronic-circuit", "automation-science-pack", "logistic-science-pack", "military-science-pack", "transport-belt", "inserter", "lab", "assembling-machine-1", "small-electric-pole", "coal")))
        item_lua = "{" + ",".join(json.dumps(name) for name in items) + "}"
        entities = "{" + ",".join(json.dumps(name) for name in Runtime_LIVE_ENTITY_NAMES) + "}"
        cell_recipes = "{" + ",".join(
            "{" + f"key={json.dumps(key)},recipe={json.dumps(recipe)}" + "}"
            for key, recipe in Runtime_CELL_RECIPES.items()
        ) + "}"
        marker = json.dumps(Runtime_STATE_MARKER)
        return (
            "/sc local marker=" + marker + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ERROR|no-agent'); return end; "
            "local s=p.surface; local f=p.force; local out={}; local function emit(k,v) out[#out+1]=k..'='..tostring(v) end; "
            "for _,n in ipairs(" + item_lua + ") do emit('inventory__'..n,p.get_item_count(n) or 0) end; "
            "for _,n in ipairs(" + entities + ") do emit('entity__'..n,s.count_entities_filtered{name=n,force=f}) end; "
            "for _,n in ipairs(" + recipe_lua + ") do local r=f.recipes[n]; emit('recipe_exists__'..n,r and 1 or 0); emit('recipe_enabled__'..n,(r and r.enabled) and 1 or 0); local ok,c=pcall(function() return p.get_craftable_count(n) end); emit('craftable__'..n,(ok and c) or 0) end; "
            "for _,n in ipairs(" + tech_lua + ") do local t=f.technologies[n]; local exists=t and 1 or 0; local trigger=(t and t.prototype and t.prototype.research_trigger) and 1 or 0; local researched=(t and t.researched) and 1 or 0; local prereq=1; if t then for _,q in pairs(t.prerequisites) do if not q.researched then prereq=0; break end end else prereq=0 end; local available=(t and t.enabled and researched==0 and trigger==0 and prereq==1) and 1 or 0; emit('tech_exists__'..n,exists); emit('tech_trigger__'..n,trigger); emit('tech_researched__'..n,researched); emit('tech_available__'..n,available); local red=0; local green=0; local military=0; local chemical=0; if t then for _,ing in ipairs(t.research_unit_ingredients) do local iname=ing.name or ing[1]; if iname=='automation-science-pack' then red=1 elseif iname=='logistic-science-pack' then green=1 elseif iname=='military-science-pack' then military=1 elseif iname=='chemical-science-pack' then chemical=1 end end end; emit('tech_red__'..n,red); emit('tech_green__'..n,green); emit('tech_military__'..n,military); emit('tech_chemical__'..n,chemical) end; "
            "local queue=0; local okq,q=pcall(function() return p.crafting_queue end); if okq and q then queue=#q end; emit('crafting_queue_length',queue); "
            "local redlab=0; local greenlab=0; local militarylab=0; local labworking=0; for _,e in ipairs(s.find_entities_filtered{name='lab',force=f}) do redlab=redlab+(e.get_item_count('automation-science-pack') or 0); greenlab=greenlab+(e.get_item_count('logistic-science-pack') or 0); militarylab=militarylab+(e.get_item_count('military-science-pack') or 0); if e.status==defines.entity_status.working then labworking=labworking+1 end end; emit('lab_red_science_count',redlab); emit('lab_green_science_count',greenlab); emit('lab_military_science_count',militarylab); emit('lab_working_count_runtime',labworking); "
            "local redbuf=0; local greenbuf=0; local militarybuf=0; local buffer_items={'iron-plate','copper-plate','steel-plate','iron-gear-wheel','electronic-circuit','copper-cable','transport-belt','inserter','pipe','engine-unit','automation-science-pack','logistic-science-pack','military-science-pack'}; local buffers={}; for _,i in ipairs(buffer_items) do buffers[i]=0 end; for _,n in ipairs({'wooden-chest','steel-chest'}) do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do redbuf=redbuf+(e.get_item_count('automation-science-pack') or 0); greenbuf=greenbuf+(e.get_item_count('logistic-science-pack') or 0); militarybuf=militarybuf+(e.get_item_count('military-science-pack') or 0); for _,i in ipairs(buffer_items) do buffers[i]=buffers[i]+(e.get_item_count(i) or 0) end end end; emit('buffered_automation_science_pack_runtime',redbuf); emit('buffered_logistic_science_pack_runtime',greenbuf); emit('buffered_military_science_pack_runtime',militarybuf); for _,i in ipairs(buffer_items) do emit('buffered_item__'..i,buffers[i]) end; local okci,ci=pcall(function() return p.can_insert{name='engine-unit',count=1} end); emit('can_collect_engine_unit',(okci and ci) and 1 or 0); "
            "for _,spec in ipairs(" + cell_recipes + ") do local c=0; local working=0; for _,n in ipairs({'assembling-machine-1','assembling-machine-2'}) do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,r=pcall(function() return e.get_recipe() end); if ok and r and r.name==spec.recipe then c=c+1; if e.status==defines.entity_status.working then working=working+1 end end end end; emit('cell_count__'..spec.key,c); emit('cell_working__'..spec.key,working) end; "
            "local steelcells=0; if storage and storage.fle_rl_runtime and storage.fle_rl_runtime.steel_cells then for _,cell in pairs(storage.fle_rl_runtime.steel_cells) do if cell and cell.furnace_position then local e=s.find_entity(cell.furnace_name or 'stone-furnace',cell.furnace_position); if e and e.valid then steelcells=steelcells+1 end end end end; emit('cell_count__steel',steelcells); "
            "local consumers=0; local connected=0; local unpowered=0; for _,n in ipairs({'electric-mining-drill','inserter','long-handed-inserter','fast-inserter','assembling-machine-1','assembling-machine-2','lab','pumpjack','oil-refinery','chemical-plant'}) do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do consumers=consumers+1; local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if ok and c then connected=connected+1 else unpowered=unpowered+1 end end end; "
            "local engines=s.find_entities_filtered{name='steam-engine',force=f}; local engine_connected=0; local active=0; local steam=0; for _,e in ipairs(engines) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if ok and c then engine_connected=engine_connected+1 end; if e.status==defines.entity_status.working or (e.energy or 0)>0 then active=active+1 end; local fb=e.fluidbox; if fb and fb[1] then steam=steam+(fb[1].amount or 0) end end; emit('power_network_exists',(engine_connected>0 and s.count_entities_filtered{name={'small-electric-pole','medium-electric-pole'},force=f}>0) and 1 or 0); emit('power_generation_capacity',#engines); emit('power_active_generation',active); emit('power_connected_consumers',connected); emit('power_unpowered_consumers',unpowered); emit('power_steam_available',steam); local function route_plan(names) local engine=engines[1]; if not engine then return false,0 end; local target=nil; for _,n in ipairs(names) do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not(ok and c) then target=e; break end end; if target then break end end; if not target then return false,0 end; local poles=s.find_entities_filtered{name={'small-electric-pole','medium-electric-pole'},force=f}; local function d2(a,b) local dx=a.x-b.x; local dy=a.y-b.y; return dx*dx+dy*dy end; local pole_name=(p.get_item_count('medium-electric-pole')>0 and 'medium-electric-pole') or 'small-electric-pole'; local spacing=(pole_name=='medium-electric-pole') and 9 or 6; local planned={}; local function pick(center,radius) local pos=s.find_non_colliding_position(pole_name,center,radius,0.5,true); if pos and s.can_place_entity{name=pole_name,position=pos,force=f} then return pos end end; local source=nil; for _,q in ipairs(poles) do if d2(q.position,engine.position)<=36 then source=q.position; break end end; if not source then source=pick(engine.position,7); if not source then return false,0 end; planned[#planned+1]=source end; local receiver=nil; for _,q in ipairs(poles) do if d2(q.position,target.position)<=25 then receiver=q.position; break end end; if not receiver then receiver=pick(target.position,6); if not receiver then return false,0 end; planned[#planned+1]=receiver end; local dx=receiver.x-source.x; local dy=receiver.y-source.y; local dist=math.sqrt(dx*dx+dy*dy); local segments=math.max(1,math.ceil(dist/spacing)); for i=1,segments-1 do local desired={x=source.x+dx*i/segments,y=source.y+dy*i/segments}; local covered=false; for _,q in ipairs(poles) do if d2(q.position,desired)<=spacing*spacing then covered=true; break end end; if not covered then for _,q in ipairs(planned) do if d2(q,desired)<=spacing*spacing then covered=true; break end end end; if not covered then local pos=pick(desired,3); if not pos then return false,0 end; planned[#planned+1]=pos end end; return p.get_item_count(pole_name)>=#planned,#planned end; local rp,rn=route_plan({'electric-mining-drill','inserter','long-handed-inserter','fast-inserter','assembling-machine-1','assembling-machine-2','lab','pumpjack','oil-refinery','chemical-plant'}); emit('power_route_possible',rp and 1 or 0); emit('power_route_poles_needed',rn); local dp,dn=route_plan({'electric-mining-drill'}); emit('drill_power_route_possible',dp and 1 or 0); emit('drill_power_route_poles_needed',dn); local lp,ln=route_plan({'lab'}); emit('lab_power_route_possible',lp and 1 or 0); emit('lab_power_route_poles_needed',ln); "
            "local function furnace_free() local drills=s.find_entities_filtered{name={'burner-mining-drill','electric-mining-drill'},force=f}; for _,d in ipairs(drills) do local pos=s.find_non_colliding_position('stone-furnace',d.position,18,0.5,true); if pos and s.can_place_entity{name='stone-furnace',position=pos,force=f} then return true end end; return false end; emit('can_place_furnace_runtime',furnace_free() and 1 or 0); local function can_line(resource) local patch=s.find_entities_filtered{name=resource}; local dirs={{1,0,defines.direction.east},{-1,0,defines.direction.west},{0,1,defines.direction.south},{0,-1,defines.direction.north}}; for pi=1,math.min(#patch,100) do local base=patch[pi].position; for _,d in ipairs(dirs) do local x=d[1]; local y=d[2]; local c={{'electric-mining-drill',{x=base.x,y=base.y},d[3]},{'transport-belt',{x=base.x+2*x,y=base.y+2*y},d[3]},{'transport-belt',{x=base.x+3*x,y=base.y+3*y},d[3]},{'transport-belt',{x=base.x+4*x,y=base.y+4*y},d[3]},{'inserter',{x=base.x+5*x,y=base.y+5*y},d[3]},{'stone-furnace',{x=base.x+7*x,y=base.y+7*y},d[3]},{'inserter',{x=base.x+9*x,y=base.y+9*y},d[3]},{'transport-belt',{x=base.x+10*x,y=base.y+10*y},d[3]},{'transport-belt',{x=base.x+11*x,y=base.y+11*y},d[3]},{'transport-belt',{x=base.x+12*x,y=base.y+12*y},d[3]},{'transport-belt',{x=base.x+13*x,y=base.y+13*y},d[3]},{'transport-belt',{x=base.x+14*x,y=base.y+14*y},d[3]},{'inserter',{x=base.x+15*x,y=base.y+15*y},d[3]},{'wooden-chest',{x=base.x+16*x,y=base.y+16*y},d[3]},{'wooden-chest',{x=base.x+7*x-y*3,y=base.y+7*y+x*3},d[3]},{'inserter',{x=base.x+7*x-y*2,y=base.y+7*y+x*2},d[3]},{'small-electric-pole',{x=base.x+2*x-y*2,y=base.y+2*y+x*2},d[3]},{'small-electric-pole',{x=base.x+8*x-y*2,y=base.y+8*y+x*2},d[3]}}; local ok=true; for _,q in ipairs(c) do if not s.can_place_entity{name=q[1],position=q[2],force=f,direction=q[3]} then ok=false; break end end; if ok then return true end end end; return false end; emit('can_build_iron_line_runtime',can_line('iron-ore') and 1 or 0); emit('can_build_copper_line_runtime',can_line('copper-ore') and 1 or 0); "
            "local function can_cell(inputs) local origin=p.position; for slot=0,80 do local col=slot%6; local row=math.floor(slot/6); local x=origin.x+18+col*16; local y=origin.y-40+row*14; local c={{'assembling-machine-1',{x=x,y=y}},{'wooden-chest',{x=x-4,y=y}},{'inserter',{x=x-2,y=y}},{'inserter',{x=x+2,y=y}},{'transport-belt',{x=x+3,y=y}},{'transport-belt',{x=x+4,y=y}},{'transport-belt',{x=x+5,y=y}},{'inserter',{x=x+6,y=y}},{'wooden-chest',{x=x+7,y=y}},{'small-electric-pole',{x=x+4,y=y+4}}}; if inputs>=2 then c[#c+1]={'wooden-chest',{x=x,y=y-4}}; c[#c+1]={'inserter',{x=x,y=y-2}} end; if inputs>=3 then c[#c+1]={'wooden-chest',{x=x,y=y+4}}; c[#c+1]={'inserter',{x=x,y=y+2}} end; local ok=true; for _,q in ipairs(c) do if not s.can_place_entity{name=q[1],position=q[2],force=f} then ok=false; break end end; if ok then return true end end; return false end; emit('can_build_cell_1',can_cell(1) and 1 or 0); emit('can_build_cell_2',can_cell(2) and 1 or 0); emit('can_build_cell_3',can_cell(3) and 1 or 0); "
            "rcon.print(marker..'|'..table.concat(out,'|')); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ERROR|'..tostring(err)) end"
        )

    def _collect_feasibility_state(self, *, force: bool = False) -> Dict[str, Any]:
        # Skip the runtime approximate geometry preflight. runtime queries authoritative
        # live state and uses matching layout searches in the runtime planners.
        state = FactorioOpenEndedEnvAjweuzp._collect_feasibility_state(self, force=force)
        if not force and self.global_step % self.feasibility_state_every != 0:
            return state
        try:
            raw = str(self.instance.rcon_client.send_command(self._state_command_yqeibtv()) or "")
            line = next((line for line in raw.splitlines() if Runtime_STATE_MARKER in line), "")
            if not line:
                raise RuntimeError(f"runtime state marker missing: {raw[:400]!r}")
            payload = line.split(Runtime_STATE_MARKER, 1)[1].lstrip(": |")
            if payload.startswith("ERROR|"):
                raise RuntimeError(payload)
            for token in payload.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                state[key.strip()] = safe_float(value.strip())
            state["runtime_query_ok"] = True
            state["runtime_query_error"] = ""
        except Exception as exc:
            state["runtime_query_ok"] = False
            state["runtime_query_error"] = f"{type(exc).__name__}: {exc}"
            state.setdefault("can_place_furnace_runtime", 0.0)
            state.setdefault("can_build_cell_1", 0.0)
            state.setdefault("can_build_cell_2", 0.0)
            state.setdefault("can_build_cell_3", 0.0)

        # Authoritative synchronisation fixes the runtime stale pole/entity issue.
        if state.get("runtime_query_ok"):
            for item in set(Runtime_TRACKED_ITEMS) | set(self.inventory):
                key = f"inventory__{item}"
                if key in state:
                    self.inventory[item] = safe_float(state[key])
            for entity in Runtime_LIVE_ENTITY_NAMES:
                key = f"entity__{entity}"
                if key in state:
                    self.entity_counts[entity] = safe_float(state[key])
            state["active_power"] = safe_float(state.get("power_active_generation")) > 0
            state["network_present"] = safe_float(state.get("power_network_exists")) > 0
            state["factory_unpowered_count"] = safe_float(state.get("power_unpowered_consumers"))
            state["transport_belt_count"] = safe_float(state.get("entity__transport-belt"))
            state["inserter_count"] = (
                safe_float(state.get("entity__inserter"))
                + safe_float(state.get("entity__long-handed-inserter"))
                + safe_float(state.get("entity__fast-inserter"))
            )
            state["assembling_machine_count"] = (
                safe_float(state.get("entity__assembling-machine-1"))
                + safe_float(state.get("entity__assembling-machine-2"))
            )
            state["lab_working_count"] = safe_float(state.get("lab_working_count_runtime"))
            state["buffered_automation_science_pack"] = safe_float(state.get("buffered_automation_science_pack_runtime"))
            state["buffered_logistic_science_pack"] = safe_float(state.get("buffered_logistic_science_pack_runtime"))
            state["buffered_military_science_pack"] = safe_float(state.get("buffered_military_science_pack_runtime"))
            # Preserve names expected by runtime machine-source verification.
            state["gear_assembler_count"] = safe_float(state.get("cell_count__gear"))
            state["science_assembler_count"] = safe_float(state.get("cell_count__science"))
            state["circuit_assembler_count"] = safe_float(state.get("cell_count__circuit"))
            state["logistic_science_assembler_count"] = safe_float(state.get("cell_count__logistic-science"))
            state["military_science_assembler_count"] = safe_float(state.get("cell_count__military-science"))
        self.feasibility_state = state
        return state

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        # Use runtime's live inventory/machine mask as the base, bypassing runtime's
        # approximate geometry preflight that caused the run-time mismatch.
        mask = FactorioOpenEndedEnvAjweuzp.get_action_mask(self).astype(np.int8)
        f = self.feasibility_state
        live = bool(f.get("runtime_query_ok", False))

        def set_valid(name: str, valid: bool) -> None:
            try:
                mask[self.action_index(name)] = int(bool(valid))
            except KeyError:
                pass

        def recipe_ready(recipe: str, minimum: float = 1.0) -> bool:
            return (
                live
                and safe_float(f.get(f"recipe_enabled__{recipe}")) > 0
                and safe_float(f.get(f"craftable__{recipe}")) >= minimum
                and safe_float(f.get("crafting_queue_length")) < 80
            )

        # Tighten all dependency-sensitive legacy craft actions.
        legacy_craft = {
            "craft_electronic_circuits": ("electronic-circuit", 10),
            "craft_transport_belts": ("transport-belt", 10),
            "craft_inserters": ("inserter", 4),
            "craft_lab": ("lab", 1),
            "craft_automation_science": ("automation-science-pack", 10),
            "craft_iron_gears": ("iron-gear-wheel", 10),
            "craft_assembling_machine": ("assembling-machine-1", 1),
        }
        for action, (recipe, count) in legacy_craft.items():
            valid = recipe_ready(recipe, count)
            if action == "craft_lab":
                valid = valid and self.inventory.get("lab", 0.0) < 1 and self.entity_counts.get("lab", 0.0) < 2
            set_valid(action, valid)

        for spec in Runtime_CRAFT_SPECS:
            set_valid(spec.action, recipe_ready(spec.recipe, 1))
        set_valid("wait_crafting", live and safe_float(f.get("crafting_queue_length")) > 0)

        # Runtime geometry aliases. Each action handles one feasible target or
        # searches the same layout pattern used by the state query.
        has_drill = (
            self.entity_counts.get("burner-mining-drill", 0.0)
            + self.entity_counts.get("electric-mining-drill", 0.0)
        ) > 0
        set_valid(
            "place_furnace_near_drill",
            live
            and has_drill
            and self.inventory.get("stone-furnace", 0.0) >= 1
            and safe_float(f.get("can_place_furnace_runtime")) > 0,
        )
        unpowered = safe_float(f.get("power_unpowered_consumers"))
        power_items = self.inventory.get("small-electric-pole", 0.0) + self.inventory.get("medium-electric-pole", 0.0)
        can_power = (
            live
            and self.entity_counts.get("steam-engine", 0.0) > 0
            and unpowered > 0
            and safe_float(f.get("power_route_possible")) > 0
            and power_items >= safe_float(f.get("power_route_poles_needed"))
        )
        set_valid("connect_factory_power", can_power)
        set_valid(
            "connect_electric_drill_to_power",
            live
            and safe_float(f.get("entity__electric-mining-drill")) > 0
            and safe_float(f.get("drill_power_route_possible")) > 0
            and power_items >= safe_float(f.get("drill_power_route_poles_needed")),
        )
        set_valid(
            "connect_lab_to_power",
            live
            and safe_float(f.get("entity__lab")) > 0
            and safe_float(f.get("lab_power_route_possible")) > 0
            and power_items >= safe_float(f.get("lab_power_route_poles_needed")),
        )

        # Existing macro cells use the runtime layout planner too.
        active_power = safe_float(f.get("power_network_exists")) > 0
        line_items = (
            self.inventory.get("electric-mining-drill", 0.0) >= 1
            and self.inventory.get("stone-furnace", 0.0) >= 1
            and self.inventory.get("transport-belt", 0.0) >= 8
            and self.inventory.get("inserter", 0.0) >= 4
            and self.inventory.get("wooden-chest", 0.0) >= 2
            and self.inventory.get("small-electric-pole", 0.0) >= 2
            and self.inventory.get("coal", 0.0) >= 10
        )
        set_valid("build_iron_smelting_line", live and active_power and line_items and safe_float(f.get("can_build_iron_line_runtime")) > 0)
        set_valid("build_copper_smelting_line", live and active_power and line_items and safe_float(f.get("can_build_copper_line_runtime")) > 0)

        def can_build_cell(key: str) -> bool:
            inputs = len(Runtime_CELL_INPUTS[key])
            machine_available = (
                self.inventory.get("assembling-machine-1", 0.0)
                + self.inventory.get("assembling-machine-2", 0.0)
            ) >= 1
            needed_inserters = inputs + 2
            needed_chests = inputs + 1
            return (
                live
                and active_power
                and safe_float(f.get(f"recipe_enabled__{Runtime_CELL_RECIPES[key]}")) > 0
                and safe_float(f.get(f"cell_count__{key}")) < self.max_cells_per_recipe
                and machine_available
                and self.inventory.get("transport-belt", 0.0) >= 3
                and self.inventory.get("inserter", 0.0) >= needed_inserters
                and self.inventory.get("wooden-chest", 0.0) >= needed_chests
                and self.inventory.get("small-electric-pole", 0.0) >= 1
                and safe_float(f.get(f"can_build_cell_{inputs}")) > 0
            )

        set_valid("build_gear_assembly_cell", can_build_cell("gear"))
        set_valid("build_science_assembly_cell", can_build_cell("science"))
        for action, key in Runtime_NEW_CELL_ACTIONS.items():
            set_valid(action, can_build_cell(key))
        for action, key in Runtime_STOCK_ACTIONS.items():
            available_input = all(
                self.inventory.get(item, 0.0) > 0
                or safe_float(f.get(f"buffered_item__{item}")) > 0
                for item in Runtime_CELL_INPUTS[key]
            )
            set_valid(action, live and safe_float(f.get(f"cell_count__{key}")) > 0 and available_input)

        steel_furnace_item = self.inventory.get("steel-furnace", 0.0) + self.inventory.get("stone-furnace", 0.0)
        set_valid(
            "build_steel_smelting_cell",
            live
            and bool(self.infrastructure_state.get("researched__steel-processing", False) or safe_float(f.get("tech_researched__steel-processing")) > 0)
            and steel_furnace_item >= 1
            and self.inventory.get("inserter", 0.0) >= 3
            and self.inventory.get("wooden-chest", 0.0) >= 3
            and self.inventory.get("small-electric-pole", 0.0) >= 1,
        )
        set_valid(
            "stock_steel_smelting_cell",
            live
            and safe_float(f.get("cell_count__steel")) > 0
            and self.inventory.get("iron-plate", 0.0) >= 5
            and self.inventory.get("coal", 0.0) >= 2,
        )

        science_available = (
            self.inventory.get("automation-science-pack", 0.0)
            + self.inventory.get("logistic-science-pack", 0.0)
            + safe_float(f.get("buffered_automation_science_pack_runtime"))
            + self.inventory.get("military-science-pack", 0.0)
            + safe_float(f.get("buffered_logistic_science_pack_runtime"))
            + safe_float(f.get("buffered_military_science_pack_runtime"))
        ) > 0
        set_valid("feed_labs_all_science", live and self.entity_counts.get("lab", 0.0) > 0 and science_available)

        current = str(self.infrastructure_state.get("current_research", "") or "")
        # Queuing a technology is a Force operation.  Factorio permits it
        # before a lab is powered; power/science determine progress, not
        # whether the selection action is legal.  Requiring a connected lab
        # here hid valid research actions from MaskablePPO and conflated
        # availability with eventual throughput.
        for technology in Runtime_RESEARCH_TARGETS:
            action_name = f"research_{technology.replace('-', '_')}"
            set_valid(
                action_name,
                live
                and not current
                and safe_float(f.get(f"tech_exists__{technology}")) > 0
                and safe_float(f.get(f"tech_trigger__{technology}")) <= 0
                and safe_float(f.get(f"tech_available__{technology}")) > 0
                and safe_float(f.get(f"tech_researched__{technology}")) <= 0,
            )
        set_valid("noop", True)
        return mask

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------
    def _runtime_begin_craft(self, spec: CraftSpec) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_CRAFT"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            "local recipe=" + json.dumps(spec.recipe) + "; local requested=" + str(int(spec.craft_count)) + "; local r=p.force.recipes[recipe]; if not r or not r.enabled then rcon.print(marker..'|ok=0|error=recipe-locked|recipe='..recipe); return end; local craftable=p.get_craftable_count(recipe); local count=math.min(requested,craftable); if count<1 then rcon.print(marker..'|ok=0|error=not-craftable|recipe='..recipe); return end; local started=p.begin_crafting{count=count,recipe=recipe,silent=true}; if started<1 then rcon.print(marker..'|ok=0|error=craft-not-started|recipe='..recipe); return end; rcon.print(marker..'|ok=1|recipe='..recipe..'|started='..started..'|controller='..controller); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_place_furnace_near_drill_yqeibtv(self) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_FURNACE"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; if p.get_item_count('stone-furnace')<1 then rcon.print(marker..'|ok=0|error=missing-stone-furnace'); return end; local s=p.surface; local f=p.force; local drills=s.find_entities_filtered{name={'burner-mining-drill','electric-mining-drill'},force=f}; for _,d in ipairs(drills) do local pos=s.find_non_colliding_position('stone-furnace',d.position,18,0.5,true); if pos and s.can_place_entity{name='stone-furnace',position=pos,force=f} then local e=s.create_entity{name='stone-furnace',position=pos,force=f,raise_built=true}; if e and p.remove_item{name='stone-furnace',count=1}==1 then rcon.print(marker..'|ok=1|x='..pos.x..'|y='..pos.y); return elseif e then e.destroy() end end end; rcon.print(marker..'|ok=0|error=no-buildable-position'); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    @staticmethod
    def _cell_layout_lua() -> str:
        return (
            "local function find_layout(inputs) local origin=p.position; for slot=0,120 do local col=slot%7; local row=math.floor(slot/7); local x=origin.x+18+col*16; local y=origin.y-56+row*14; local c={{'assembling-machine-1',{x=x,y=y},defines.direction.east},{'wooden-chest',{x=x-4,y=y},defines.direction.east},{'inserter',{x=x-2,y=y},defines.direction.east},{'inserter',{x=x+2,y=y},defines.direction.east},{'transport-belt',{x=x+3,y=y},defines.direction.east},{'transport-belt',{x=x+4,y=y},defines.direction.east},{'transport-belt',{x=x+5,y=y},defines.direction.east},{'inserter',{x=x+6,y=y},defines.direction.east},{'wooden-chest',{x=x+7,y=y},defines.direction.east},{'small-electric-pole',{x=x+4,y=y+4},defines.direction.east}}; if inputs>=2 then c[#c+1]={'wooden-chest',{x=x,y=y-4},defines.direction.east}; c[#c+1]={'inserter',{x=x,y=y-2},defines.direction.south} end; if inputs>=3 then c[#c+1]={'wooden-chest',{x=x,y=y+4},defines.direction.east}; c[#c+1]={'inserter',{x=x,y=y+2},defines.direction.north} end; local ok=true; for _,q in ipairs(c) do if not s.can_place_entity{name=q[1],position=q[2],force=f,direction=q[3]} then ok=false; break end end; if ok then return {x=x,y=y,candidates=c} end end; return nil end; "
        )

    def _runtime_create_cell_yqeibtv(self, key: str) -> Dict[str, Any]:
        recipe = Runtime_CELL_RECIPES[key]
        inputs = Runtime_CELL_INPUTS[key]
        marker = f"{Runtime_RUNTIME_MARKER}_CELL"
        input_lua = "{" + ",".join(json.dumps(item) for item in inputs) + "}"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local f=p.force; local key=" + json.dumps(key) + "; local recipe=" + json.dumps(recipe) + "; local input_items=" + input_lua + "; local inputs=#input_items; storage.fle_rl_runtime=storage.fle_rl_runtime or {cells={},steel_cells={}}; storage.fle_rl_runtime.cells=storage.fle_rl_runtime.cells or {}; local existing=0; for _,e in ipairs(s.find_entities_filtered{name={'assembling-machine-1','assembling-machine-2'},force=f}) do local ok,r=pcall(function() return e.get_recipe() end); if ok and r and r.name==recipe then existing=existing+1 end end; if existing>=" + str(self.max_cells_per_recipe) + " then rcon.print(marker..'|ok=0|error=cell-limit'); return end; local machine=(p.get_item_count('assembling-machine-1')>0 and 'assembling-machine-1') or (p.get_item_count('assembling-machine-2')>0 and 'assembling-machine-2') or nil; if not machine then rcon.print(marker..'|ok=0|error=missing-assembler'); return end; local need={['transport-belt']=3,['inserter']=inputs+2,['wooden-chest']=inputs+1,['small-electric-pole']=1}; need[machine]=1; for n,c in pairs(need) do if p.get_item_count(n)<c then rcon.print(marker..'|ok=0|error=missing-'..n); return end end; "
            + self._cell_layout_lua()
            + "local plan=find_layout(inputs); if not plan then rcon.print(marker..'|ok=0|error=no-buildable-layout'); return end; local made={}; local function create(n,pos,dir) local spec={name=n,position=pos,force=f,raise_built=true,direction=dir or defines.direction.east}; if n=='small-electric-pole' then spec.auto_connect=true end; local e=s.create_entity(spec); if e and e.valid then made[#made+1]=e; return e end; return nil end; local function rollback() for i=#made,1,-1 do if made[i] and made[i].valid then made[i].destroy() end end end; local asm=create(machine,{x=plan.x,y=plan.y},defines.direction.east); if not asm then rcon.print(marker..'|ok=0|error=create-assembler'); return end; local okr=pcall(function() asm.set_recipe(recipe) end); if not okr then rollback(); rcon.print(marker..'|ok=0|error=set-recipe'); return end; local input_positions={}; local ch=create('wooden-chest',{x=plan.x-4,y=plan.y}); create('inserter',{x=plan.x-2,y=plan.y},defines.direction.east); input_positions[1]=ch and ch.position or nil; if inputs>=2 then ch=create('wooden-chest',{x=plan.x,y=plan.y-4}); create('inserter',{x=plan.x,y=plan.y-2},defines.direction.south); input_positions[2]=ch and ch.position or nil end; if inputs>=3 then ch=create('wooden-chest',{x=plan.x,y=plan.y+4}); create('inserter',{x=plan.x,y=plan.y+2},defines.direction.north); input_positions[3]=ch and ch.position or nil end; create('inserter',{x=plan.x+2,y=plan.y},defines.direction.east); for i=3,5 do create('transport-belt',{x=plan.x+i,y=plan.y},defines.direction.east) end; create('inserter',{x=plan.x+6,y=plan.y},defines.direction.east); local output=create('wooden-chest',{x=plan.x+7,y=plan.y}); create('small-electric-pole',{x=plan.x+4,y=plan.y+4}); for _,e in ipairs(made) do if not e or not e.valid then rollback(); rcon.print(marker..'|ok=0|error=create-layout'); return end end; for n,c in pairs(need) do local removed=p.remove_item{name=n,count=c}; if removed<c then rollback(); rcon.print(marker..'|ok=0|error=inventory-remove-'..n); return end end; local id=key..'-'..game.tick; storage.fle_rl_runtime.cells[id]={key=key,recipe=recipe,assembler_position=asm.position,input_items=input_items,input_positions=input_positions,output_position=output.position}; rcon.print(marker..'|ok=1|key='..key..'|recipe='..recipe..'|created='..#made..'|id='..id); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    # Existing runtime actions are transparently upgraded to the runtime cell builder.
    def _runtime_create_assembly_cell(self, recipe_name: str, cell_key: str) -> Dict[str, Any]:
        key = "science" if recipe_name == "automation-science-pack" else "gear"
        return self._runtime_create_cell_yqeibtv(key)

    def _runtime_stock_cell_yqeibtv(self, key: str) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_STOCK"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local key=" + json.dumps(key) + "; local cells=storage and storage.fle_rl_runtime and storage.fle_rl_runtime.cells; if not cells then rcon.print(marker..'|ok=0|error=no-cells'); return end; local target=nil; for _,c in pairs(cells) do if c and c.key==key then target=c; break end end; if not target then rcon.print(marker..'|ok=0|error=missing-cell'); return end; local moved=0; local function take_from_sources(item,count,destination) local need=count; local have=p.get_item_count(item); if have>0 then local n=math.min(need,have); local r=p.remove_item{name=item,count=n}; local i=destination.insert{name=item,count=r}; if i<r then p.insert{name=item,count=r-i} end; moved=moved+i; need=need-i end; if need<=0 then return end; for _,chest_name in ipairs({'wooden-chest','steel-chest'}) do for _,src in ipairs(s.find_entities_filtered{name=chest_name,force=p.force}) do if src~=destination then local n=math.min(need,src.get_item_count(item)); if n>0 then local r=src.remove_item{name=item,count=n}; local i=destination.insert{name=item,count=r}; if i<r then src.insert{name=item,count=r-i} end; moved=moved+i; need=need-i; if need<=0 then return end end end end end end; for i,item in ipairs(target.input_items or {}) do local pos=target.input_positions and target.input_positions[i]; local chest=pos and (s.find_entity('wooden-chest',pos) or s.find_entity('steel-chest',pos)) or nil; if not chest then rcon.print(marker..'|ok=0|error=missing-input-chest'); return end; take_from_sources(item,40,chest) end; if moved<1 then rcon.print(marker..'|ok=0|error=no-input-items'); return end; rcon.print(marker..'|ok=1|key='..key..'|moved='..moved); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_create_steel_cell_yqeibtv(self) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_STEEL"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local f=p.force; storage.fle_rl_runtime=storage.fle_rl_runtime or {cells={},steel_cells={}}; storage.fle_rl_runtime.steel_cells=storage.fle_rl_runtime.steel_cells or {}; local furnace=(p.get_item_count('steel-furnace')>0 and 'steel-furnace') or (p.get_item_count('stone-furnace')>0 and 'stone-furnace') or nil; if not furnace then rcon.print(marker..'|ok=0|error=missing-furnace'); return end; local need={[furnace]=1,['inserter']=3,['wooden-chest']=3,['small-electric-pole']=1}; for n,c in pairs(need) do if p.get_item_count(n)<c then rcon.print(marker..'|ok=0|error=missing-'..n); return end end; local origin=p.position; local pos=nil; for slot=0,80 do local x=origin.x-18-(slot%6)*12; local y=origin.y-40+math.floor(slot/6)*12; local c={{furnace,{x=x,y=y}},{'wooden-chest',{x=x-4,y=y}},{'inserter',{x=x-2,y=y}},{'wooden-chest',{x=x,y=y-4}},{'inserter',{x=x,y=y-2}},{'inserter',{x=x+2,y=y}},{'wooden-chest',{x=x+4,y=y}},{'small-electric-pole',{x=x+4,y=y+4}}}; local ok=true; for _,q in ipairs(c) do if not s.can_place_entity{name=q[1],position=q[2],force=f} then ok=false; break end end; if ok then pos={x=x,y=y}; break end end; if not pos then rcon.print(marker..'|ok=0|error=no-buildable-layout'); return end; local made={}; local function c(n,p0,dir) local spec={name=n,position=p0,force=f,raise_built=true,direction=dir or defines.direction.east}; if n=='small-electric-pole' then spec.auto_connect=true end; local e=s.create_entity(spec); if e then made[#made+1]=e end; return e end; local fu=c(furnace,pos); local input=c('wooden-chest',{x=pos.x-4,y=pos.y}); c('inserter',{x=pos.x-2,y=pos.y},defines.direction.east); local fuel=c('wooden-chest',{x=pos.x,y=pos.y-4}); c('inserter',{x=pos.x,y=pos.y-2},defines.direction.south); c('inserter',{x=pos.x+2,y=pos.y},defines.direction.east); local output=c('wooden-chest',{x=pos.x+4,y=pos.y}); c('small-electric-pole',{x=pos.x+4,y=pos.y+4}); if #made<8 then for _,e in ipairs(made) do if e.valid then e.destroy() end end; rcon.print(marker..'|ok=0|error=create-layout'); return end; for n,count in pairs(need) do if p.remove_item{name=n,count=count}<count then for _,e in ipairs(made) do if e.valid then e.destroy() end end; rcon.print(marker..'|ok=0|error=inventory-remove-'..n); return end end; local id='steel-'..game.tick; storage.fle_rl_runtime.steel_cells[id]={furnace_name=furnace,furnace_position=fu.position,input_position=input.position,fuel_position=fuel.position,output_position=output.position}; rcon.print(marker..'|ok=1|id='..id..'|furnace='..furnace); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_stock_steel_yqeibtv(self) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_STEELSTOCK"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local cells=storage and storage.fle_rl_runtime and storage.fle_rl_runtime.steel_cells; if not cells then rcon.print(marker..'|ok=0|error=no-steel-cell'); return end; local cell=nil; for _,c in pairs(cells) do cell=c; break end; if not cell then rcon.print(marker..'|ok=0|error=no-steel-cell'); return end; local input=s.find_entity('wooden-chest',cell.input_position); local fuel=s.find_entity('wooden-chest',cell.fuel_position); if not input or not fuel then rcon.print(marker..'|ok=0|error=missing-chest'); return end; local pi=math.min(50,p.get_item_count('iron-plate')); local pc=math.min(20,p.get_item_count('coal')); local mi=0; local mc=0; if pi>0 then mi=input.insert{name='iron-plate',count=p.remove_item{name='iron-plate',count=pi}} end; if pc>0 then mc=fuel.insert{name='coal',count=p.remove_item{name='coal',count=pc}} end; if mi+mc<1 then rcon.print(marker..'|ok=0|error=no-input-items'); return end; rcon.print(marker..'|ok=1|iron='..mi..'|coal='..mc); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_feed_labs_yqeibtv(self) -> Dict[str, Any]:
        """Move available science into labs using Factorio's real input capacity.

        runtime still carried an old absolute ``50 - current_count`` target even
        though the mask used ``lab_input.can_insert``.  Once a lab already held
        50+ packs, the mask could correctly expose this action while the
        executor moved zero items and raised ``no-science``.  runtime removes that
        artificial fill level: each action may move up to 50 packs of each
        supported science type *per lab*, but only while the actual lab input
        inventory can accept the item.  ``lab.insert`` remains authoritative
        for the exact capacity and all uninserted items are returned.
        """
        marker = f"{Runtime_RUNTIME_MARKER}_LABFEED"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            + "local s=p.surface; local labs=s.find_entities_filtered{name='lab',force=p.force}; "
            + "if #labs<1 then rcon.print(marker..'|ok=0|error=no-lab'); return end; "
            + "local moved=0; "
            + "local function move_item(item) "
            + "for _,lab in ipairs(labs) do "
            + "local ok,inv=pcall(function() return lab.get_inventory(defines.inventory.lab_input) end); "
            + "if ok and inv and inv.valid then "
            + "local budget=50; "
            + "local function can_accept() local okc,c=pcall(function() return inv.can_insert{name=item,count=1} end); return okc and c end; "
            + "if budget>0 and can_accept() then "
            + "local have=p.get_item_count(item); "
            + "if have>0 then local n=math.min(budget,have); local r=p.remove_item{name=item,count=n}; local i=lab.insert{name=item,count=r}; if i<r then p.insert{name=item,count=r-i} end; moved=moved+i; budget=budget-i end; "
            + "end; "
            + "if budget>0 and can_accept() then "
            + "for _,chest_name in ipairs({'wooden-chest','steel-chest'}) do "
            + "for _,ch in ipairs(s.find_entities_filtered{name=chest_name,force=p.force}) do "
            + "if budget<=0 or not can_accept() then break end; "
            + "local n=math.min(budget,ch.get_item_count(item)); "
            + "if n>0 then local r=ch.remove_item{name=item,count=n}; local i=lab.insert{name=item,count=r}; if i<r then ch.insert{name=item,count=r-i} end; moved=moved+i; budget=budget-i end; "
            + "end; if budget<=0 or not can_accept() then break end; end; "
            + "end; "
            + "end; "
            + "end; "
            + "end; "
            + "move_item('automation-science-pack'); move_item('logistic-science-pack'); move_item('military-science-pack'); "
            + "if moved<1 then rcon.print(marker..'|ok=0|error=no-science'); return end; "
            + "rcon.print(marker..'|ok=1|moved='..moved); "
            + "end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_connect_power_yqeibtv(self, target_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_POWER"
        targets = tuple(target_names or ("electric-mining-drill", "inserter", "long-handed-inserter", "fast-inserter", "assembling-machine-1", "assembling-machine-2", "lab", "pumpjack", "oil-refinery", "chemical-plant"))
        target_lua = "{" + ",".join(json.dumps(name) for name in targets) + "}"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local f=p.force; local engines=s.find_entities_filtered{name='steam-engine',force=f}; local engine=engines[1]; if not engine then rcon.print(marker..'|ok=0|error=missing-steam-engine'); return end; local target=nil; for _,n in ipairs(" + target_lua + ") do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not (ok and c) then target=e; break end end; if target then break end end; if not target then rcon.print(marker..'|ok=0|error=no-unpowered-target'); return end; local poles=s.find_entities_filtered{name={'small-electric-pole','medium-electric-pole'},force=f}; local function d2pos(a,b) local dx=a.x-b.x; local dy=a.y-b.y; return dx*dx+dy*dy end; local source=nil; for _,pole in ipairs(poles) do if d2pos(pole.position,engine.position)<=36 then source=pole; break end end; local plan={}; local pole_name=(p.get_item_count('medium-electric-pole')>0 and 'medium-electric-pole') or 'small-electric-pole'; local spacing=(pole_name=='medium-electric-pole') and 9 or 6; local function planned_pos(center,radius) local pos=s.find_non_colliding_position(pole_name,center,radius,0.5,true); if pos and s.can_place_entity{name=pole_name,position=pos,force=f} then return pos end; return nil end; if not source then local pos=planned_pos(engine.position,7); if not pos then rcon.print(marker..'|ok=0|error=cannot-place-source-pole'); return end; plan[#plan+1]=pos; source={position=pos} end; local receiver=nil; for _,pole in ipairs(poles) do if d2pos(pole.position,target.position)<=25 then receiver=pole; break end end; if not receiver then local pos=planned_pos(target.position,6); if not pos then rcon.print(marker..'|ok=0|error=cannot-place-receiver-pole'); return end; receiver={position=pos}; plan[#plan+1]=pos end; local dx=receiver.position.x-source.position.x; local dy=receiver.position.y-source.position.y; local dist=math.sqrt(dx*dx+dy*dy); local segments=math.max(1,math.ceil(dist/spacing)); for i=1,segments-1 do local desired={x=source.position.x+dx*i/segments,y=source.position.y+dy*i/segments}; local covered=false; for _,pole in ipairs(poles) do if d2pos(pole.position,desired)<=spacing*spacing then covered=true; break end end; if not covered then for _,pos in ipairs(plan) do if d2pos(pos,desired)<=spacing*spacing then covered=true; break end end end; if not covered then local pos=planned_pos(desired,3); if not pos then rcon.print(marker..'|ok=0|error=cannot-route-poles'); return end; plan[#plan+1]=pos end end; if p.get_item_count(pole_name)<#plan then rcon.print(marker..'|ok=0|error=missing-poles|need='..#plan); return end; local made={}; for _,pos in ipairs(plan) do local e=s.create_entity{name=pole_name,position=pos,force=f,raise_built=true,auto_connect=true}; if not e then for _,q in ipairs(made) do if q.valid then q.destroy() end end; rcon.print(marker..'|ok=0|error=create-pole'); return end; made[#made+1]=e end; local removed=p.remove_item{name=pole_name,count=#made}; if removed<#made then for _,q in ipairs(made) do if q.valid then q.destroy() end end; rcon.print(marker..'|ok=0|error=inventory-remove-poles'); return end; local ok,c=pcall(function() return target.is_connected_to_electric_network() end); rcon.print(marker..'|ok=1|target='..target.name..'|created='..#made..'|connected='..((ok and c) and 1 or 0)); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    # Existing runtime factory-power action now connects one target atomically.
    def _runtime_connect_factory_power(self) -> Dict[str, Any]:
        return self._runtime_connect_power_yqeibtv()

    def _runtime_connect_lab_power(self) -> Dict[str, Any]:
        return self._runtime_connect_power_yqeibtv(("lab",))

    def _queue_research_runtime(self, target: str) -> Dict[str, Any]:
        if target not in Runtime_RESEARCH_TARGETS:
            raise RuntimeError(f"unsupported runtime research target: {target}")
        if safe_float(self.feasibility_state.get(f"tech_trigger__{target}")) > 0:
            raise RuntimeError(f"{target} is a trigger technology and cannot be queued")
        marker = f"{Runtime_RUNTIME_MARKER}_RESEARCH"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local target=" + json.dumps(target) + "; local f=p.force; local t=f.technologies[target]; if not t then rcon.print(marker..'|ok=0|error=missing-technology'); return end; if t.prototype and t.prototype.research_trigger then rcon.print(marker..'|ok=0|error=trigger-technology'); return end; if t.researched then rcon.print(marker..'|ok=0|error=already-researched'); return end; for _,q in pairs(t.prerequisites) do if not q.researched then rcon.print(marker..'|ok=0|error=missing-prerequisite|name='..q.name); return end end; if f.current_research then rcon.print(marker..'|ok=0|error=research-already-active|name='..f.current_research.name); return end; local ok=f.add_research(t); if not ok and not f.current_research then rcon.print(marker..'|ok=0|error=queue-failed'); return end; rcon.print(marker..'|ok=1|target='..target..'|current='..((f.current_research and f.current_research.name) or '')); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    # ------------------------------------------------------------------
    # Rewards, milestones, observations
    # ------------------------------------------------------------------
    def _has_machine_source(self, item: str) -> bool:
        kind = MACHINE_SOURCE_KIND.get(item)
        if kind and kind.endswith("-assembler"):
            key = {
                "gear-assembler": "gear",
                "cable-assembler": "cable",
                "circuit-assembler": "circuit",
                "belt-assembler": "belt",
                "inserter-assembler": "inserter",
                "science-assembler": "science",
                "logistic-science-assembler": "logistic-science",
                "engine-assembler": "engine",
            }.get(kind)
            if key:
                return safe_float(self.feasibility_state.get(f"cell_count__{key}")) > 0
        return super()._has_machine_source(item)

    def _trigger_researched_set(self) -> set[str]:
        return {
            name
            for name in Runtime_TRIGGER_TECHNOLOGIES
            if safe_float(self.feasibility_state.get(f"tech_researched__{name}")) > 0
        }

    def _lab_researched_set(self) -> set[str]:
        return {
            name
            for name in Runtime_RESEARCH_TARGETS
            if safe_float(self.feasibility_state.get(f"tech_researched__{name}")) > 0
            and safe_float(self.feasibility_state.get(f"tech_trigger__{name}")) <= 0
        }

    def _current_progress_milestones(self) -> set[str]:
        milestones = set(super()._current_progress_milestones())
        f = self.feasibility_state
        if safe_float(f.get("tech_researched__electronics")) > 0:
            milestones.add("electronics_triggered")
        if self.inventory.get("iron-gear-wheel", 0.0) >= 10:
            milestones.add("lab_gears_ready")
        if self.inventory.get("electronic-circuit", 0.0) >= 10:
            milestones.add("lab_circuits_ready")
        if self.inventory.get("transport-belt", 0.0) >= 4:
            milestones.add("lab_belts_ready")
        if all(
            (
                self.inventory.get("iron-gear-wheel", 0.0) >= 10,
                self.inventory.get("electronic-circuit", 0.0) >= 10,
                self.inventory.get("transport-belt", 0.0) >= 4,
            )
        ):
            milestones.add("lab_all_ingredients_ready")
        if self.inventory.get("copper-cable", 0.0) > 0:
            milestones.add("copper_cable_available")
        for key, milestone in (
            ("circuit", "circuit_cell_built"),
            ("belt", "belt_cell_built"),
            ("inserter", "inserter_cell_built"),
            ("logistic-science", "logistic_science_cell_built"),
            ("military-science", "military_science_cell_built"),
        ):
            if safe_float(f.get(f"cell_count__{key}")) > 0:
                milestones.add(milestone)
        if safe_float(f.get("recipe_enabled__logistic-science-pack")) > 0:
            milestones.add("logistic_science_unlocked")
        if (
            self.inventory.get("logistic-science-pack", 0.0)
            + safe_float(f.get("buffered_logistic_science_pack_runtime"))
            + safe_float(f.get("lab_green_science_count"))
        ) > 0:
            milestones.add("logistic_science_available")
        if safe_float(f.get("recipe_enabled__military-science-pack")) > 0:
            milestones.add("military_science_unlocked")
        if (
            self.inventory.get("military-science-pack", 0.0)
            + safe_float(f.get("buffered_military_science_pack_runtime"))
            + safe_float(f.get("lab_military_science_count"))
        ) > 0:
            milestones.add("military_science_available")
        current = str(self.infrastructure_state.get("current_research", "") or "")
        if current and safe_float(f.get(f"tech_green__{current}")) > 0:
            milestones.add("green_science_research_started")
        for tech, milestone in (
            ("automation-2", "automation_2_researched"),
            ("logistics-2", "logistics_2_researched"),
            ("engine", "engine_researched"),
            ("oil-processing", "oil_processing_researched"),
        ):
            if safe_float(f.get(f"tech_researched__{tech}")) > 0:
                milestones.add(milestone)
        return milestones

    def _observation(self) -> np.ndarray:
        final_names = self.observation_feature_names
        final_space = self.observation_space
        self.observation_feature_names = self._runtime_feature_names
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self._runtime_feature_names),),
            dtype=np.float32,
        )
        try:
            base = FactorioOpenEndedEnvAjweuzp._observation(self)
        finally:
            self.observation_feature_names = final_names
            self.observation_space = final_space

        f = self.feasibility_state
        values: List[float] = []
        values.extend(
            min(10.0, math.log1p(max(0.0, self.inventory.get(item, 0.0))))
            for item in Runtime_TRACKED_ITEMS
        )
        values.extend(float(safe_float(f.get(f"recipe_enabled__{name}")) > 0) for name in self.runtime_recipe_names)
        values.extend(float(safe_float(f.get(f"tech_researched__{name}")) > 0) for name in Runtime_RESEARCH_TARGETS)
        values.extend(float(safe_float(f.get(f"tech_available__{name}")) > 0) for name in Runtime_RESEARCH_TARGETS)
        values.extend(
            (
                float(safe_float(f.get("power_network_exists")) > 0),
                min(10.0, safe_float(f.get("power_generation_capacity"))),
                min(10.0, safe_float(f.get("power_active_generation"))),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("power_connected_consumers"))))),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("power_unpowered_consumers"))))),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("power_steam_available")))) / 5.0),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("lab_red_science_count")))) / 5.0),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("lab_green_science_count")))) / 5.0),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("lab_military_science_count")))) / 5.0),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("buffered_automation_science_pack_runtime")))) / 5.0),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("buffered_logistic_science_pack_runtime")))) / 5.0),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("buffered_military_science_pack_runtime")))) / 5.0),
                min(10.0, safe_float(f.get("crafting_queue_length"))),
            )
        )
        values.extend(
            min(10.0, math.log1p(max(0.0, safe_float(f.get(f"entity__{name}")))))
            for name in Runtime_LIVE_ENTITY_NAMES
        )
        values.extend(min(10.0, safe_float(f.get(f"cell_count__{key}"))) for key in Runtime_CELL_RECIPES)
        values.append(min(10.0, safe_float(f.get("cell_count__steel"))))
        result = np.concatenate([base, np.asarray(values, dtype=np.float32)])
        if result.shape != self.observation_space.shape:
            raise RuntimeError(
                f"runtime observation mismatch: {result.shape} != {self.observation_space.shape}"
            )
        return result.astype(np.float32)

    def behavior_descriptors(self) -> Dict[str, float]:
        d = dict(super().behavior_descriptors())
        f = self.feasibility_state
        advanced_cells = sum(safe_float(f.get(f"cell_count__{key}")) for key in Runtime_CELL_RECIPES)
        d.update(
            {
                "advanced_cell_count": advanced_cells,
                "green_science_available": float(
                    self.inventory.get("logistic-science-pack", 0.0)
                    + safe_float(f.get("buffered_logistic_science_pack_runtime"))
                    > 0
                ),
                "runtime_lab_researched_count": float(len(self._lab_researched_set())),
                "active_generation_count": safe_float(f.get("power_active_generation")),
            }
        )
        return d

    def benchmark_fitness(self) -> float:
        d = self.behavior_descriptors()
        return float(
            self.machine_output_total
            + 35.0 * d.get("runtime_lab_researched_count", 0.0)
            + 12.0 * d.get("green_science_available", 0.0)
            + 4.0 * d.get("advanced_cell_count", 0.0)
            + 0.5 * d.get("logistics_complexity", 0.0)
            + 0.1 * d.get("entity_count", 0.0)
        )

    def _base_info(self) -> Dict[str, Any]:
        info = super()._base_info()
        info.update(
            {
                "environment_version": Runtime_VERSION,
                "runtime_feasibility_state": json_safe(self.feasibility_state),
                "runtime_runtime_failures": dict(self.runtime_runtime_failures),
                "runtime_trigger_completions": dict(self.runtime_trigger_completions),
                "runtime_lab_research_completions": dict(self.runtime_lab_research_completions),
                "runtime_researched_count": len(self._lab_researched_set()),
                "runtime_trigger_count": len(self._trigger_researched_set()),
                "live_entity_counts": {
                    name: safe_float(self.feasibility_state.get(f"entity__{name}"))
                    for name in Runtime_LIVE_ENTITY_NAMES
                },
                "power_state_runtime": {
                    key: safe_float(self.feasibility_state.get(key))
                    for key in (
                        "power_network_exists",
                        "power_generation_capacity",
                        "power_active_generation",
                        "power_connected_consumers",
                        "power_unpowered_consumers",
                        "power_steam_available",
                    )
                },
            }
        )
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        observation, info = super().reset(seed=seed, options=options)
        hard = bool(dict(options or {}).get("hard_reset", self.hard_reset))
        if hard:
            self.runtime_runtime_failures.clear()
            try:
                self.instance.rcon_client.send_command(
                    "/sc storage.fle_rl_runtime={cells={},steel_cells={}}"
                )
            except Exception:
                pass
        self._collect_feasibility_state(force=True)
        final = {**dict(info), **self._base_info()}
        final["action_mask"] = self.action_masks()
        final["environment_schema_path"] = str(Path(self.run_dir) / "environment_schema_runtime.json")
        return self._observation(), final

    def _dispatch_runtime_yqeibtv(self, action_name: str) -> Optional[Dict[str, Any]]:
        if action_name == "place_furnace_near_drill":
            return self._runtime_place_furnace_near_drill_yqeibtv()
        if action_name == "connect_electric_drill_to_power":
            return self._runtime_connect_power_yqeibtv(("electric-mining-drill",))
        if action_name in CRAFT_BY_ACTION:
            return self._runtime_begin_craft(CRAFT_BY_ACTION[action_name])
        if action_name in Runtime_NEW_CELL_ACTIONS:
            return self._runtime_create_cell_yqeibtv(Runtime_NEW_CELL_ACTIONS[action_name])
        if action_name in Runtime_STOCK_ACTIONS:
            return self._runtime_stock_cell_yqeibtv(Runtime_STOCK_ACTIONS[action_name])
        if action_name == "build_steel_smelting_cell":
            return self._runtime_create_steel_cell_yqeibtv()
        if action_name == "stock_steel_smelting_cell":
            return self._runtime_stock_steel_yqeibtv()
        if action_name == "feed_labs_all_science":
            return self._runtime_feed_labs_yqeibtv()
        if action_name.startswith("research_"):
            target = action_name[len("research_"):].replace("_", "-")
            if target in Runtime_RESEARCH_TARGETS and target not in {"automation", "logistics", "steel-processing"}:
                return self._queue_research_runtime(target)
        return None

    def step(self, action: int):
        before_triggers = self._trigger_researched_set()
        before_lab = self._lab_researched_set()
        action_name = self.actions[int(action)].name if 0 <= int(action) < len(self.actions) else "invalid-index"

        observation, reward, terminated, truncated, info = super().step(action)
        del observation
        info = dict(info)
        runtime_result: Optional[Dict[str, Any]] = None
        runtime_error = ""

        # runtime/runtime already dispatch existing line, gear/science-cell,
        # connect_factory_power, and connect_lab_to_power actions. Dispatch only
        # runtime additions and redirected legacy aliases here.
        if not info.get("action_error"):
            try:
                runtime_result = self._dispatch_runtime_yqeibtv(action_name)
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
                self.runtime_runtime_failures[action_name] += 1

        if runtime_result is not None:
            info["runtime_action_extended"] = runtime_result
            if action_name in CRAFT_BY_ACTION:
                spec = CRAFT_BY_ACTION[action_name]
                started = safe_float(runtime_result.get("started", spec.craft_count))
                expected = max(0.0, started * spec.output_per_craft)
                if expected > 0:
                    self._add_manual_guard(spec.output_item, expected)
                    self.pending_manual_craft_output[spec.output_item] += expected

        if runtime_error:
            info["action_error"] = runtime_error
            self.total_action_errors += 1
            penalty = -abs(float(self.execution_error_penalty))
            reward += penalty
            self.episode_return += penalty
            self.lifetime_canonical_reward_components["error"] += penalty
            self.horizon_canonical_reward_components["error"] += penalty
            self.lifetime_canonical_reward_components["total"] += penalty
            self.horizon_canonical_reward_components["total"] += penalty
            self.lifetime_reward_components["error"] += penalty

        # runtime: runtime has already refreshed live state after the inherited action.
        # Only repeat the expensive state/RCON scan when runtime itself dispatched
        # an additional runtime mutation (craft/research/cell/lab action).
        # This preserves authoritative semantics while avoiding one full scan
        # on the majority of transitions.
        try:
            if runtime_result is not None or runtime_error:
                self._collect_compact_state(include_production_stats=True)
                self._entity_records()
                self._collect_infrastructure_state(force=True)
                self._collect_feasibility_state(force=True)
            elif hasattr(self, "_refresh_if_needed_research"):
                self._refresh_if_needed_research()
        except Exception as exc:
            info["runtime_refresh_error"] = f"{type(exc).__name__}: {exc}"

        parts = dict(info.get("reward_parts", {}))
        if runtime_error:
            parts["error"] = safe_float(parts.get("error")) - abs(float(self.execution_error_penalty))
        old_completion = safe_float(parts.get("research_completion"))
        after_triggers = self._trigger_researched_set()
        after_lab = self._lab_researched_set()
        new_triggers = sorted(after_triggers - before_triggers)
        new_lab = sorted(after_lab - before_lab)
        trigger_reward = len(new_triggers) * self.trigger_technology_reward
        lab_reward = len(new_lab) * self.lab_research_completion_reward
        replacement = trigger_reward + lab_reward
        delta = replacement - old_completion
        if abs(delta) > 1e-12:
            reward += delta
            self.episode_return += delta
            self.lifetime_canonical_reward_components["research_completion"] -= old_completion
            self.horizon_canonical_reward_components["research_completion"] -= old_completion
            self.lifetime_canonical_reward_components["trigger_technology"] += trigger_reward
            self.horizon_canonical_reward_components["trigger_technology"] += trigger_reward
            self.lifetime_canonical_reward_components["lab_research_completion"] += lab_reward
            self.horizon_canonical_reward_components["lab_research_completion"] += lab_reward
            self.lifetime_canonical_reward_components["total"] += delta
            self.horizon_canonical_reward_components["total"] += delta
            self.lifetime_reward_components["research_completion"] -= old_completion
            self.lifetime_reward_components["trigger_technology"] += trigger_reward
            self.lifetime_reward_components["lab_research_completion"] += lab_reward
        for name in new_triggers:
            self.runtime_trigger_completions[name] += 1
        for name in new_lab:
            self.runtime_lab_research_completions[name] += 1

        # Runtime actions occur after the inherited milestone calculation. Add
        # any milestones crossed by the runtime action itself.
        current_milestones = self._current_progress_milestones()
        post_runtime_milestones = sorted(current_milestones - self.progression_milestones_seen)
        post_progression_reward = sum(Runtime_PROGRESSION_REWARDS.get(name, 0.0) for name in post_runtime_milestones)
        if post_progression_reward:
            self.progression_milestones_seen.update(post_runtime_milestones)
            reward += post_progression_reward
            self.episode_return += post_progression_reward
            self.lifetime_canonical_reward_components["progression_milestone"] += post_progression_reward
            self.horizon_canonical_reward_components["progression_milestone"] += post_progression_reward
            self.lifetime_canonical_reward_components["total"] += post_progression_reward
            self.horizon_canonical_reward_components["total"] += post_progression_reward
            self.lifetime_reward_components["progression_milestone"] += post_progression_reward

        parts["research_completion"] = 0.0
        parts["trigger_technology"] = float(trigger_reward)
        parts["lab_research_completion"] = float(lab_reward)
        parts["progression_milestone"] = safe_float(parts.get("progression_milestone")) + post_progression_reward
        # Keep the decomposition exact after post-super adjustments.
        for key in ("total", "residual"):
            parts.pop(key, None)
        parts["residual"] = float(reward - sum(safe_float(v) for v in parts.values()))
        parts["total"] = float(reward)

        final_info = dict(info)
        final_info.update(self._base_info())
        final_info.update(
            {
                "reward": float(reward),
                "reward_parts": parts,
                "new_trigger_technologies": new_triggers,
                "new_lab_research": new_lab,
                "new_progression_milestones": sorted(
                    set(info.get("new_progression_milestones", [])) | set(post_runtime_milestones)
                ),
                "action_mask": self.action_masks(),
                "terminal_goal": None,
            }
        )
        return self._observation(), float(reward), bool(terminated), bool(truncated), final_info


# Compatibility alias used by public import and tools.
FactorioOpenEndedEnvRuntime = FactorioOpenEndedEnvYqeibtv


__all__ = [
    "CraftSpec",
    "FactorioOpenEndedEnvRuntime",
    "FactorioOpenEndedEnvYqeibtv",
    "Runtime_ACTIONS",
    "Runtime_CELL_INPUTS",
    "Runtime_CELL_RECIPES",
    "Runtime_CRAFT_SPECS",
    "Runtime_PROGRESSION_REWARDS",
    "Runtime_RESEARCH_TARGETS",
    "Runtime_TRACKED_ITEMS",
    "Runtime_VERSION",
]
