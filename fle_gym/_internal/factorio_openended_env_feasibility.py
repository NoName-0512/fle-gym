from __future__ import annotations

"""runtime open-ended Factorio environment.

This revision keeps PPO in control of every high-level decision while adding
real logistics and assembly macro-actions.  The environment does not execute a
scripted build order: MaskablePPO chooses among currently feasible actions.
Each selected macro performs deterministic geometry and API work, which is the
same abstraction already used for pump/boiler/pole placement.

Main corrections compared with runtime:

* trigger technologies are observed but never queued as lab research;
* the starting inventory no longer gifts labs, belts, inserters, assemblers,
  plates, circuits, gears, or science packs;
* belts, inserters, smelting lines, gear assembly, and red-science assembly are
  represented by explicit policy actions;
* machine-production reward uses a strict item whitelist and never rewards
  water, steam, manual harvesting, manual crafting, or item transfers;
* every reward component is exposed in ``info["reward_parts"]``;
* runtime feasibility is used to mask saturated or unavailable actions;
* fixed-horizon training remains goal-free, while evaluation can use one
  persistent factory by setting ``training_horizon_steps=0``.
"""

from collections import Counter
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from factorio_openended_env import FactorioOpenEndedEnv
from factorio_openended_env_observation import (
    FactorioOpenEndedEnvObservation,
    IMPROVED_ACTIONS,
    IMPROVED_DEFAULT_INVENTORY,
)
from factorio_transition_env import ActionSpec, _dedent, json_safe, safe_float, spaces


Runtime_FEASIBILITY_MARKER = "__FLE_RL_FEASIBILITY_runtime__"
Runtime_RUNTIME_MARKER = "__FLE_RL_RUNTIME_runtime__"

LAB_RESEARCH_TARGETS: Tuple[str, ...] = (
    "automation",
    "logistics",
    "steel-processing",
)
TRIGGER_TECHNOLOGIES: Tuple[str, ...] = (
    "electronics",
    "automation-science-pack",
)

# Only useful item production is rewarded.  Fluids, manual work, transfers,
# and unknown products receive exactly zero production reward.
MACHINE_OUTPUT_WEIGHTS: Dict[str, float] = {
    "iron-ore": 0.02,
    "copper-ore": 0.02,
    "iron-plate": 0.40,
    "copper-plate": 0.40,
    "steel-plate": 1.20,
    "iron-gear-wheel": 0.80,
    "electronic-circuit": 1.20,
    "automation-science-pack": 2.50,
    "logistic-science-pack": 4.00,
}

# These outputs are explicitly attributed to player/agent work.  The value is
# the requested item count and is used even when FLE's crafted/harvested bucket
# is delayed or incomplete.
MANUAL_CRAFT_OUTPUTS: Dict[str, Tuple[str, float]] = {
    "craft_stone_furnace": ("stone-furnace", 1.0),
    "craft_iron_gears": ("iron-gear-wheel", 10.0),
    "craft_electronic_circuits": ("electronic-circuit", 10.0),
    "craft_transport_belts": ("transport-belt", 10.0),
    "craft_inserters": ("inserter", 4.0),
    "craft_lab": ("lab", 1.0),
    "craft_automation_science": ("automation-science-pack", 10.0),
    "craft_assembling_machine": ("assembling-machine-1", 1.0),
}
MANUAL_HARVEST_OUTPUTS: Dict[str, Tuple[str, float]] = {
    "harvest_coal": ("coal", 10.0),
    "harvest_stone": ("stone", 10.0),
    "harvest_iron_ore": ("iron-ore", 10.0),
    "harvest_copper_ore": ("copper-ore", 10.0),
}

Runtime_DEFAULT_INVENTORY: Dict[str, int] = dict(IMPROVED_DEFAULT_INVENTORY)
Runtime_DEFAULT_INVENTORY.update(
    {
        # Raw-resource production and infrastructure are available, but the
        # actual research/logistics chain must be manufactured by the policy.
        "iron-plate": 0,
        "copper-plate": 0,
        "iron-gear-wheel": 0,
        "electronic-circuit": 0,
        "automation-science-pack": 0,
        "transport-belt": 0,
        "inserter": 0,
        "assembling-machine-1": 0,
        "lab": 0,
        # Enough bootstrap hardware to discover a valid power/smelting chain.
        "coal": 120,
        "burner-mining-drill": 2,
        "electric-mining-drill": 6,
        "stone-furnace": 4,
        "wooden-chest": 12,
        "offshore-pump": 1,
        "boiler": 2,
        "steam-engine": 2,
        "pipe": 120,
        "small-electric-pole": 60,
    }
)


def _marker_action(
    name: str,
    *,
    category: str,
    description: str,
    required_items: Optional[Mapping[str, float]] = None,
    required_entities: Optional[Mapping[str, float]] = None,
) -> ActionSpec:
    return ActionSpec(
        name=name,
        category=category,
        code=f'print("__FLE_RL_EVENT__:runtime_{name}_requested")',
        required_items=dict(required_items or {}),
        required_entities=dict(required_entities or {}),
        description=description,
    )


Runtime_EXTRA_ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec(
        name="craft_electronic_circuits",
        category="craft",
        code=_dedent(
            """
            result = craft_item(Prototype.ElectronicCircuit, quantity=10)
            print("__FLE_RL_EVENT__:electronic_circuits_crafted")
            print("craft_result=" + repr(result))
            """
        ),
        description="Manually craft ten electronic circuits after Electronics unlocks.",
    ),
    ActionSpec(
        name="craft_transport_belts",
        category="craft",
        code=_dedent(
            """
            result = craft_item(Prototype.TransportBelt, quantity=10)
            print("__FLE_RL_EVENT__:transport_belts_crafted")
            print("craft_result=" + repr(result))
            """
        ),
        description="Manually craft ten transport belts for logistics construction.",
    ),
    ActionSpec(
        name="craft_inserters",
        category="craft",
        code=_dedent(
            """
            result = craft_item(Prototype.Inserter, quantity=4)
            print("__FLE_RL_EVENT__:inserters_crafted")
            print("craft_result=" + repr(result))
            """
        ),
        description="Manually craft four electric inserters.",
    ),
    ActionSpec(
        name="craft_assembling_machine",
        category="craft",
        code=_dedent(
            """
            result = craft_item(Prototype.AssemblingMachine1, quantity=1)
            print("__FLE_RL_EVENT__:assembling_machine_crafted")
            print("craft_result=" + repr(result))
            """
        ),
        description="Manually craft one assembling machine 1 after Automation unlocks.",
    ),
    _marker_action(
        "build_iron_smelting_line",
        category="logistics",
        description=(
            "Build one powered iron mining/smelting line with belts, ore and fuel "
            "inserters, a furnace, and an output chest."
        ),
    ),
    _marker_action(
        "build_copper_smelting_line",
        category="logistics",
        description=(
            "Build one powered copper mining/smelting line with belts, ore and fuel "
            "inserters, a furnace, and an output chest."
        ),
    ),
    _marker_action(
        "build_gear_assembly_cell",
        category="automation",
        description=(
            "Place and power an iron-gear assembler with an input chest, output "
            "belts, inserters, and an output chest."
        ),
    ),
    _marker_action(
        "build_science_assembly_cell",
        category="automation",
        description=(
            "Place and power an automation-science assembler with two input "
            "chests, output belts, inserters, and an output chest."
        ),
    ),
    _marker_action(
        "stock_gear_assembler",
        category="transfer",
        description="Move available iron plates into the gear assembler input chest.",
    ),
    _marker_action(
        "stock_science_assembler",
        category="transfer",
        description=(
            "Move available copper plates and machine-produced gears into the "
            "automation-science assembler input chests."
        ),
    ),
    _marker_action(
        "feed_lab_from_science_output",
        category="research",
        description="Move machine-produced red science from its output chest into a lab.",
    ),
    _marker_action(
        "connect_factory_power",
        category="power",
        description=(
            "Connect all currently unpowered drills, inserters, assemblers, and "
            "labs to the existing steam network with verified pole routes."
        ),
    ),
)


def _actions_feasibility() -> Tuple[ActionSpec, ...]:
    actions: List[ActionSpec] = []
    for spec in IMPROVED_ACTIONS:
        # Electronics and Automation science pack are trigger technologies in
        # Factorio 2.x; they must never be queued as lab research.
        if spec.name == "research_electronics":
            continue
        if spec.name in {
            "craft_lab",
            "craft_automation_science",
            "craft_iron_gears",
        }:
            # Live get_craftable_count/recipe telemetry is authoritative.
            spec = replace(spec, required_items={})
        actions.append(spec)
    actions.extend(Runtime_EXTRA_ACTIONS)
    return tuple(actions)


Runtime_ACTIONS: Tuple[ActionSpec, ...] = _actions_feasibility()


class FactorioOpenEndedEnvFeasibility(FactorioOpenEndedEnvObservation):
    """Goal-free horizon environment with logistics and assembly automation."""

    def __init__(
        self,
        *,
        initial_inventory: Optional[Mapping[str, int]] = None,
        research_activity_reward: float = 0.02,
        first_smelting_line_reward: float = 3.0,
        first_assembler_reward: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            initial_inventory=initial_inventory or Runtime_DEFAULT_INVENTORY,
            **kwargs,
        )
        self.actions = Runtime_ACTIONS
        self.action_space = spaces.Discrete(len(self.actions))
        self.research_activity_reward = float(research_activity_reward)
        self.first_smelting_line_reward = float(first_smelting_line_reward)
        self.first_assembler_reward = float(first_assembler_reward)

        self.machine_output_units_total = 0.0
        self.lifetime_machine_output_units_total = 0.0
        self.manual_crafted_units_total = 0.0
        self.manual_harvested_units_total = 0.0
        self.lifetime_manual_crafted_units_total = 0.0
        self.lifetime_manual_harvested_units_total = 0.0
        self.lifetime_reward_components: Counter[str] = Counter()
        # Explicit manual operations can appear in FLE's production output a
        # few transitions later. These pending credits prevent delayed hand
        # work from being misclassified as machine production.
        self.pending_manual_craft_output: Counter[str] = Counter()
        self.pending_manual_harvest_output: Counter[str] = Counter()

        extra_features = (
            "trigger__electronics",
            "trigger__automation_science_pack",
            "recipe__electronic_circuit",
            "recipe__transport_belt",
            "recipe__inserter",
            "recipe__lab",
            "recipe__automation_science_pack",
            "recipe__assembling_machine_1",
            "entities__transport_belt_runtime",
            "entities__inserter_runtime",
            "entities__assembling_machine_runtime",
            "assembler__gear_cells",
            "assembler__science_cells",
            "assembler__working",
            "assembler__unpowered",
            "output__gears_buffered",
            "output__science_buffered",
            "lines__iron_smelting",
            "lines__copper_smelting",
            "factory__unpowered_consumers",
        )
        self.observation_feature_names = tuple(self.observation_feature_names) + extra_features
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )
        self._write_config_feasibility()

    @staticmethod
    def _empty_feasibility_state() -> Dict[str, Any]:
        state = FactorioOpenEndedEnvObservation._empty_feasibility_state()
        state.update(
            {
                "runtime_query_ok": False,
                "runtime_query_error": "not queried",
                "trigger_researched__electronics": False,
                "trigger_researched__automation-science-pack": False,
                "trigger_type__electronics": False,
                "trigger_type__automation-science-pack": False,
                "recipe_enabled__electronic-circuit": False,
                "recipe_enabled__transport-belt": False,
                "recipe_enabled__inserter": False,
                "recipe_enabled__lab": False,
                "recipe_enabled__automation-science-pack": False,
                "recipe_enabled__assembling-machine-1": False,
                "craftable__electronic-circuit": 0.0,
                "craftable__transport-belt": 0.0,
                "craftable__inserter": 0.0,
                "craftable__lab": 0.0,
                "craftable__automation-science-pack": 0.0,
                "craftable__assembling-machine-1": 0.0,
                "gear_assembler_count": 0.0,
                "science_assembler_count": 0.0,
                "assembler_working_count": 0.0,
                "assembler_unpowered_count": 0.0,
                "buffered_iron_plate": 0.0,
                "buffered_copper_plate": 0.0,
                "buffered_iron_gear_wheel": 0.0,
                "buffered_automation_science_pack": 0.0,
                "iron_smelting_line_count": 0.0,
                "copper_smelting_line_count": 0.0,
                "factory_unpowered_count": 0.0,
                "transport_belt_count": 0.0,
                "inserter_count": 0.0,
                "assembling_machine_count": 0.0,
            }
        )
        return state

    def _write_config_feasibility(self) -> None:
        payload = {
            "environment": "FactorioOpenEndedEnvFeasibility",
            "terminal_goal": None,
            "training_horizon_steps": self.training_horizon_steps,
            "action_count": len(self.actions),
            "actions": [asdict(spec) for spec in self.actions],
            "lab_research_targets": list(LAB_RESEARCH_TARGETS),
            "trigger_technologies": list(TRIGGER_TECHNOLOGIES),
            "machine_output_weights": dict(MACHINE_OUTPUT_WEIGHTS),
            "manual_output_actions": {
                "craft": MANUAL_CRAFT_OUTPUTS,
                "harvest": MANUAL_HARVEST_OUTPUTS,
            },
            "initial_inventory": dict(self.initial_inventory_config),
            "screenshots": {
                "recommended_resolution": [4096, 2304],
                "recommended_margin_tiles": 24,
            },
        }
        Path(self.run_dir, "runtime_environment_config.json").write_text(
            json.dumps(json_safe(payload), indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _strict_weight(items: Mapping[str, float]) -> float:
        return sum(
            max(0.0, float(amount)) * MACHINE_OUTPUT_WEIGHTS.get(str(item), 0.0)
            for item, amount in items.items()
        )

    @staticmethod
    def _raw_units(items: Mapping[str, float]) -> float:
        return sum(max(0.0, float(amount)) for amount in items.values())

    def _feasibility_command_feasibility(self) -> str:
        marker = json.dumps(Runtime_FEASIBILITY_MARKER)
        recipes = (
            "electronic-circuit",
            "transport-belt",
            "inserter",
            "lab",
            "automation-science-pack",
            "assembling-machine-1",
            "iron-gear-wheel",
        )
        recipe_lua = "{" + ",".join(json.dumps(name) for name in recipes) + "}"
        techs = TRIGGER_TECHNOLOGIES + LAB_RESEARCH_TARGETS
        tech_lua = "{" + ",".join(json.dumps(name) for name in techs) + "}"
        return (
            "/sc local p=nil; "
            "if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then p=storage.agent_characters[1] "
            "elseif player and player.valid then p=player elseif game.get_player(1) then p=game.get_player(1) end; "
            "if not p then rcon.print(" + marker + "..'|ERROR|no-agent') else "
            "local s=p.surface; local f=p.force; local out={}; "
            "local recipes=" + recipe_lua + "; for _,n in ipairs(recipes) do "
            "local r=f.recipes[n]; out[#out+1]='recipe_enabled__'..n..'='..((r and r.enabled) and '1' or '0'); "
            "local ok,c=pcall(function() return p.get_craftable_count(n) end); out[#out+1]='craftable__'..n..'='..((ok and c) or 0); end; "
            "local techs=" + tech_lua + "; for _,n in ipairs(techs) do local t=f.technologies[n]; "
            "local trigger=(t and t.prototype and t.prototype.research_trigger) and true or false; "
            "out[#out+1]='trigger_type__'..n..'='..(trigger and '1' or '0'); "
            "out[#out+1]='trigger_researched__'..n..'='..((t and t.researched) and '1' or '0'); end; "
            "local gear=0; local science=0; local awork=0; local aunpowered=0; "
            "for _,e in ipairs(s.find_entities_filtered{name='assembling-machine-1',force=f}) do "
            "local okr,r=pcall(function() return e.get_recipe() end); local rn=(okr and r and r.name) or ''; "
            "if rn=='iron-gear-wheel' then gear=gear+1 elseif rn=='automation-science-pack' then science=science+1 end; "
            "if e.status==defines.entity_status.working then awork=awork+1 end; "
            "local okc,c=pcall(function() return e.is_connected_to_electric_network() end); if not (okc and c) then aunpowered=aunpowered+1 end; end; "
            "out[#out+1]='gear_assembler_count='..gear; out[#out+1]='science_assembler_count='..science; "
            "out[#out+1]='assembler_working_count='..awork; out[#out+1]='assembler_unpowered_count='..aunpowered; "
            "local bip=0; local bcp=0; local big=0; local bsp=0; "
            "for _,e in ipairs(s.find_entities_filtered{name='wooden-chest',force=f}) do "
            "bip=bip+(e.get_item_count('iron-plate') or 0); bcp=bcp+(e.get_item_count('copper-plate') or 0); "
            "big=big+(e.get_item_count('iron-gear-wheel') or 0); bsp=bsp+(e.get_item_count('automation-science-pack') or 0); end; "
            "out[#out+1]='buffered_iron_plate='..bip; out[#out+1]='buffered_copper_plate='..bcp; "
            "out[#out+1]='buffered_iron_gear_wheel='..big; out[#out+1]='buffered_automation_science_pack='..bsp; "
            "local unpowered=0; for _,n in ipairs({'electric-mining-drill','inserter','assembling-machine-1','lab'}) do "
            "for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not (ok and c) then unpowered=unpowered+1 end end end; "
            "out[#out+1]='factory_unpowered_count='..unpowered; "
            "out[#out+1]='transport_belt_count='..#s.find_entities_filtered{name='transport-belt',force=f}; "
            "out[#out+1]='inserter_count='..#s.find_entities_filtered{name='inserter',force=f}; "
            "out[#out+1]='assembling_machine_count='..#s.find_entities_filtered{name='assembling-machine-1',force=f}; "
            "local il=0; local cl=0; if storage and storage.fle_rl_runtime and storage.fle_rl_runtime.lines then "
            "for _,cell in pairs(storage.fle_rl_runtime.lines) do if cell and cell.furnace_position then local e=s.find_entity('stone-furnace',cell.furnace_position); if e and e.valid then if cell.resource=='iron-ore' then il=il+1 elseif cell.resource=='copper-ore' then cl=cl+1 end end end end end; "
            "out[#out+1]='iron_smelting_line_count='..il; out[#out+1]='copper_smelting_line_count='..cl; "
            "rcon.print(" + marker + "..'|'..table.concat(out,'|')); end"
        )

    def _collect_feasibility_state(self, *, force: bool = False) -> Dict[str, Any]:
        state = super()._collect_feasibility_state(force=force)
        if not force and self.global_step % self.feasibility_state_every != 0:
            return state
        try:
            raw = str(self.instance.rcon_client.send_command(self._feasibility_command_feasibility()) or "")
            line = next((line for line in raw.splitlines() if Runtime_FEASIBILITY_MARKER in line), "")
            if not line:
                raise RuntimeError(f"runtime feasibility marker missing: {raw[:400]!r}")
            payload = line.split(Runtime_FEASIBILITY_MARKER, 1)[1].lstrip(": |")
            if payload.startswith("ERROR|"):
                raise RuntimeError(payload)
            for token in payload.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.startswith("recipe_enabled__") or key.startswith("trigger_"):
                    state[key] = value in {"1", "true", "True"}
                else:
                    state[key] = safe_float(value)
            state["runtime_query_ok"] = True
            state["runtime_query_error"] = ""
        except Exception as exc:
            state["runtime_query_ok"] = False
            state["runtime_query_error"] = f"{type(exc).__name__}: {exc}"
        self.feasibility_state = state
        return state

    def get_action_mask(self) -> np.ndarray:
        """Return a live-state feasibility mask for the runtime action set."""
        mask = super().get_action_mask().astype(np.int8)
        f = self.feasibility_state
        query_ok = bool(f.get("query_ok", False))
        runtime_ok = bool(f.get("runtime_query_ok", False))

        def set_valid(name: str, valid: bool) -> None:
            try:
                mask[self.action_index(name)] = int(bool(valid))
            except KeyError:
                pass

        def craftable(recipe: str, quantity: float = 1.0) -> bool:
            return (not runtime_ok) or safe_float(f.get(f"craftable__{recipe}")) >= quantity

        # Trigger technologies are observed, never queued. Craft actions are
        # enabled only after the live force recipe is available/craftable.
        set_valid(
            "craft_electronic_circuits",
            craftable("electronic-circuit", 10)
            and self.inventory.get("electronic-circuit", 0.0) < 30.0,
        )
        set_valid(
            "craft_transport_belts",
            craftable("transport-belt", 10)
            and self.inventory.get("transport-belt", 0.0) < 40.0,
        )
        set_valid(
            "craft_inserters",
            craftable("inserter", 4)
            and self.inventory.get("inserter", 0.0) < 16.0,
        )
        set_valid(
            "craft_lab",
            craftable("lab", 1)
            and self.entity_counts.get("lab", 0.0) < 2.0
            and self.inventory.get("lab", 0.0) < 1.0,
        )
        set_valid(
            "craft_automation_science",
            craftable("automation-science-pack", 10)
            and self.inventory.get("automation-science-pack", 0.0) < 20.0,
        )
        set_valid(
            "craft_iron_gears",
            craftable("iron-gear-wheel", 10)
            and self.inventory.get("iron-gear-wheel", 0.0) < 30.0,
        )
        set_valid(
            "craft_assembling_machine",
            craftable("assembling-machine-1", 1)
            and self.inventory.get("assembling-machine-1", 0.0) < 2.0,
        )

        active_power = bool(f.get("active_power", False))
        common_line_items = (
            self.inventory.get("electric-mining-drill", 0.0) >= 1
            and self.inventory.get("stone-furnace", 0.0) >= 1
            and self.inventory.get("transport-belt", 0.0) >= 8
            and self.inventory.get("inserter", 0.0) >= 4
            and self.inventory.get("wooden-chest", 0.0) >= 2
            and self.inventory.get("small-electric-pole", 0.0) >= 2
            and self.inventory.get("coal", 0.0) >= 10
        )
        set_valid(
            "build_iron_smelting_line",
            active_power
            and common_line_items
            and safe_float(f.get("iron_smelting_line_count")) < 2.0,
        )
        set_valid(
            "build_copper_smelting_line",
            active_power
            and common_line_items
            and safe_float(f.get("copper_smelting_line_count")) < 2.0,
        )

        common_cell_items = (
            active_power
            and self.inventory.get("assembling-machine-1", 0.0) >= 1
            and self.inventory.get("transport-belt", 0.0) >= 3
            and self.inventory.get("inserter", 0.0) >= 2
            and self.inventory.get("wooden-chest", 0.0) >= 2
            and self.inventory.get("small-electric-pole", 0.0) >= 1
        )
        set_valid(
            "build_gear_assembly_cell",
            common_cell_items
            and bool(f.get("recipe_enabled__iron-gear-wheel", True))
            and safe_float(f.get("gear_assembler_count")) < 1.0,
        )
        set_valid(
            "build_science_assembly_cell",
            common_cell_items
            and self.inventory.get("wooden-chest", 0.0) >= 3
            and self.inventory.get("inserter", 0.0) >= 3
            and bool(f.get("recipe_enabled__automation-science-pack", False))
            and safe_float(f.get("science_assembler_count")) < 1.0,
        )
        set_valid(
            "stock_gear_assembler",
            safe_float(f.get("gear_assembler_count")) > 0
            and (
                self.inventory.get("iron-plate", 0.0)
                + safe_float(f.get("buffered_iron_plate"))
            )
            >= 2,
        )
        set_valid(
            "stock_science_assembler",
            safe_float(f.get("science_assembler_count")) > 0
            and (
                self.inventory.get("copper-plate", 0.0)
                + safe_float(f.get("buffered_copper_plate"))
            )
            >= 1
            and (
                self.inventory.get("iron-gear-wheel", 0.0)
                + safe_float(f.get("buffered_iron_gear_wheel"))
            )
            >= 1,
        )
        set_valid(
            "feed_lab_from_science_output",
            self.entity_counts.get("lab", 0.0) > 0
            and safe_float(f.get("buffered_automation_science_pack")) > 0,
        )
        set_valid(
            "connect_factory_power",
            self.entity_counts.get("steam-engine", 0.0) > 0
            and safe_float(f.get("factory_unpowered_count")) > 0
            and self.inventory.get("small-electric-pole", 0.0) >= 1,
        )

        # Research selection is limited to genuine lab technologies. While one
        # technology is active, all competing selections are masked.
        current = str(self.infrastructure_state.get("current_research", "") or "")
        lab_connected = safe_float(self.infrastructure_state.get("lab_connected_count"))
        for target in LAB_RESEARCH_TARGETS:
            set_valid(
                f"research_{target.replace('-', '_')}",
                not current
                and lab_connected > 0
                and bool(self.infrastructure_state.get(f"available__{target}", False))
                and not bool(self.infrastructure_state.get(f"researched__{target}", False)),
            )
        set_valid(
            "insert_automation_science",
            bool(current)
            and self.inventory.get("automation-science-pack", 0.0) >= 1
            and (not query_ok or safe_float(f.get("lab_science_space")) > 0),
        )
        set_valid(
            "wait_research",
            bool(current)
            and (
                safe_float(f.get("lab_science_count")) > 0
                or safe_float(f.get("buffered_automation_science_pack")) > 0
            ),
        )

        # Tighten the most common failing legacy actions.
        if query_ok:
            set_valid(
                "place_furnace_near_drill",
                self.entity_counts.get("stone-furnace", 0.0) < 2.0
                and self.inventory.get("stone-furnace", 0.0) >= 1
                and safe_float(f.get("furnace_iron_input_space")) >= 0,
            )
            set_valid(
                "connect_electric_drill_to_power",
                safe_float(f.get("electric_unpowered_count")) > 0
                and self.entity_counts.get("steam-engine", 0.0) > 0
                and self.inventory.get("small-electric-pole", 0.0) >= 1,
            )
        set_valid("noop", True)
        return mask

    @staticmethod
    def _runtime_agent_prelude() -> str:
        return (
            "local p=nil; local controller=''; "
            "if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then p=storage.agent_characters[1]; controller='agent-character' "
            "elseif player and player.valid then p=player; controller='global-character' "
            "elseif game.get_player(1) then p=game.get_player(1); controller='lua-player-1' end; "
        )

    def _runtime_create_compact_line(self, resource_name: str) -> Dict[str, Any]:
        """Build a verified compact drill→belt→furnace→chest line.

        The policy chooses whether and when to build the line. The runtime
        operation only handles deterministic collision search and entity
        placement, then verifies all created entities before reporting success.
        """
        marker = f"{Runtime_RUNTIME_MARKER}_LINE"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            "local s=p.surface; local f=p.force; local resource=" + json.dumps(resource_name) + "; "
            "storage.fle_rl_runtime=storage.fle_rl_runtime or {lines={},cells={}}; storage.fle_rl_runtime.lines=storage.fle_rl_runtime.lines or {}; "
            "local need={['electric-mining-drill']=1,['stone-furnace']=1,['transport-belt']=8,['inserter']=4,['wooden-chest']=2,['small-electric-pole']=2,['coal']=10}; "
            "for n,c in pairs(need) do if p.get_item_count(n)<c then rcon.print(marker..'|ok=0|error=missing-'..n..'|have='..p.get_item_count(n)..'|need='..c); return end end; "
            "local patch=s.find_entities_filtered{name=resource}; if #patch<1 then rcon.print(marker..'|ok=0|error=missing-resource'); return end; "
            "local dirs={{1,0,defines.direction.east},{-1,0,defines.direction.west},{0,1,defines.direction.south},{0,-1,defines.direction.north}}; "
            "local made={}; local function create(n,pos,dir,extra) local spec={name=n,position=pos,force=f,raise_built=true,direction=dir}; if n=='small-electric-pole' then spec.auto_connect=true end; local e=s.create_entity(spec); if e and e.valid then made[#made+1]=e; return e end; return nil end; "
            "local function destroy_all() for i=#made,1,-1 do if made[i] and made[i].valid then made[i].destroy() end end end; "
            "local chosen=nil; for pi=1,math.min(#patch,100) do local base=patch[pi].position; for _,d in ipairs(dirs) do local x=d[1]; local y=d[2]; local candidates={"
            "{'electric-mining-drill',{x=base.x,y=base.y},d[3]},"
            "{'transport-belt',{x=base.x+2*x,y=base.y+2*y},d[3]},"
            "{'transport-belt',{x=base.x+3*x,y=base.y+3*y},d[3]},"
            "{'transport-belt',{x=base.x+4*x,y=base.y+4*y},d[3]},"
            "{'inserter',{x=base.x+5*x,y=base.y+5*y},d[3]},"
            "{'stone-furnace',{x=base.x+7*x,y=base.y+7*y},d[3]},"
            "{'inserter',{x=base.x+9*x,y=base.y+9*y},d[3]},"
            "{'transport-belt',{x=base.x+10*x,y=base.y+10*y},d[3]},"
            "{'transport-belt',{x=base.x+11*x,y=base.y+11*y},d[3]},"
            "{'transport-belt',{x=base.x+12*x,y=base.y+12*y},d[3]},"
            "{'transport-belt',{x=base.x+13*x,y=base.y+13*y},d[3]},"
            "{'transport-belt',{x=base.x+14*x,y=base.y+14*y},d[3]},"
            "{'inserter',{x=base.x+15*x,y=base.y+15*y},d[3]},"
            "{'wooden-chest',{x=base.x+16*x,y=base.y+16*y},d[3]},"
            "{'wooden-chest',{x=base.x+7*x-y*3,y=base.y+7*y+x*3},d[3]},"
            "{'inserter',{x=base.x+7*x-y*2,y=base.y+7*y+x*2},(x==1 and defines.direction.north or x==-1 and defines.direction.south or y==1 and defines.direction.east or defines.direction.west)},"
            "{'small-electric-pole',{x=base.x+2*x-y*2,y=base.y+2*y+x*2},d[3]},"
            "{'small-electric-pole',{x=base.x+8*x-y*2,y=base.y+8*y+x*2},d[3]}}; "
            "local ok=true; for _,c in ipairs(candidates) do if not s.can_place_entity{name=c[1],position=c[2],force=f,direction=c[3]} then ok=false; break end end; if ok then chosen={d=d,c=candidates}; break end end; if chosen then break end end; "
            "if not chosen then rcon.print(marker..'|ok=0|error=no-buildable-layout'); return end; "
            "for _,c in ipairs(chosen.c) do if not create(c[1],c[2],c[3]) then destroy_all(); rcon.print(marker..'|ok=0|error=create-'..c[1]); return end end; "
            "for n,c in pairs(need) do local removed=p.remove_item{name=n,count=c}; if removed<c then destroy_all(); for rn,rc in pairs(need) do p.insert{name=rn,count=rc} end; rcon.print(marker..'|ok=0|error=inventory-remove-'..n); return end end; "
            "local furnace=nil; local fuel=nil; local chest_index=0; for _,e in ipairs(made) do if e.name=='stone-furnace' then furnace=e elseif e.name=='wooden-chest' then chest_index=chest_index+1; if chest_index==2 then fuel=e end end end; if fuel then fuel.insert{name='coal',count=10} end; "
            "local key=resource..'-'..game.tick; storage.fle_rl_runtime.lines[key]={resource=resource,furnace_position=furnace and furnace.position or nil}; "
            "rcon.print(marker..'|ok=1|resource='..resource..'|created='..#made..'|controller='..controller..'|key='..key); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_create_assembly_cell(self, recipe_name: str, cell_key: str) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_CELL"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local f=p.force; "
            "local recipe=" + json.dumps(recipe_name) + "; local key=" + json.dumps(cell_key) + "; "
            "storage.fle_rl_runtime=storage.fle_rl_runtime or {lines={},cells={}}; storage.fle_rl_runtime.cells=storage.fle_rl_runtime.cells or {}; "
            "local inputs=(recipe=='automation-science-pack') and 2 or 1; local need={['assembling-machine-1']=1,['transport-belt']=3,['inserter']=inputs+2,['wooden-chest']=inputs+1,['small-electric-pole']=1}; "
            "for n,c in pairs(need) do if p.get_item_count(n)<c then rcon.print(marker..'|ok=0|error=missing-'..n..'|have='..p.get_item_count(n)); return end end; "
            "local origin=p.position; local dirs={{1,0,defines.direction.east},{-1,0,defines.direction.west},{0,1,defines.direction.south},{0,-1,defines.direction.north}}; local selected=nil; "
            "for radius=8,40,4 do for _,d in ipairs(dirs) do local x=origin.x+d[1]*radius; local y=origin.y+d[2]*radius; local pos={asm={x=x,y=y},in1={x=x-3,y=y},in2={x=x,y=y-3},out={x=x+7,y=y},pole={x=x,y=y+4}}; local ok=s.can_place_entity{name='assembling-machine-1',position=pos.asm,force=f}; if ok then selected=pos; break end end; if selected then break end end; "
            "if not selected then rcon.print(marker..'|ok=0|error=no-buildable-layout'); return end; local made={}; local function c(n,pos,dir) local spec={name=n,position=pos,force=f,raise_built=true,direction=dir or defines.direction.east}; if n=='small-electric-pole' then spec.auto_connect=true end; local e=s.create_entity(spec); if e and e.valid then made[#made+1]=e; return e end end; "
            "local asm=c('assembling-machine-1',selected.asm); if not asm then rcon.print(marker..'|ok=0|error=create-assembler'); return end; local okr=asm.set_recipe(recipe); if not okr then asm.destroy(); rcon.print(marker..'|ok=0|error=set-recipe'); return end; "
            "local in1=c('wooden-chest',selected.in1); local i1=c('inserter',{x=selected.asm.x-2,y=selected.asm.y},defines.direction.east); local in2=nil; if inputs==2 then in2=c('wooden-chest',selected.in2); c('inserter',{x=selected.asm.x,y=selected.asm.y-2},defines.direction.south) end; "
            "c('inserter',{x=selected.asm.x+2,y=selected.asm.y},defines.direction.east); for i=1,3 do c('transport-belt',{x=selected.asm.x+2+i,y=selected.asm.y},defines.direction.east) end; c('inserter',{x=selected.asm.x+6,y=selected.asm.y},defines.direction.east); local out=c('wooden-chest',selected.out); c('small-electric-pole',selected.pole); "
            "for n,count in pairs(need) do local removed=p.remove_item{name=n,count=count}; if removed<count then for _,e in ipairs(made) do if e.valid then e.destroy() end end; rcon.print(marker..'|ok=0|error=inventory-remove-'..n); return end end; storage.fle_rl_runtime.cells[key]={recipe=recipe,assembler_position=asm.position,input1_position=in1 and in1.position or nil,input2_position=in2 and in2.position or nil,output_position=out and out.position or nil}; "
            "rcon.print(marker..'|ok=1|recipe='..recipe..'|created='..#made..'|controller='..controller); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_stock_assembly_cell(self, cell_key: str) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_STOCK"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local key=" + json.dumps(cell_key) + "; local cell=storage and storage.fle_rl_runtime and storage.fle_rl_runtime.cells and storage.fle_rl_runtime.cells[key]; if not cell then rcon.print(marker..'|ok=0|error=missing-cell'); return end; "
            "local c1=cell.input1_position and s.find_entity('wooden-chest',cell.input1_position) or nil; local c2=cell.input2_position and s.find_entity('wooden-chest',cell.input2_position) or nil; if not c1 then rcon.print(marker..'|ok=0|error=missing-input-chest'); return end; "
            "local moved=0; local function move(item,chest,count) local have=p.get_item_count(item); local take=math.min(have,count); if take>0 then local r=p.remove_item{name=item,count=take}; local i=chest.insert{name=item,count=r}; if i<r then p.insert{name=item,count=r-i} end; moved=moved+i end end; "
            "if key=='gear' then move('iron-plate',c1,40) else move('copper-plate',c1,40); if not c2 then rcon.print(marker..'|ok=0|error=missing-second-input'); return end; move('iron-gear-wheel',c2,40) end; if moved<1 then rcon.print(marker..'|ok=0|error=no-input-items'); return end; rcon.print(marker..'|ok=1|moved='..moved); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_feed_lab(self) -> Dict[str, Any]:
        marker = f"{Runtime_RUNTIME_MARKER}_LABFEED"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local f=p.force; local cell=storage and storage.fle_rl_runtime and storage.fle_rl_runtime.cells and storage.fle_rl_runtime.cells.science; if not cell or not cell.output_position then rcon.print(marker..'|ok=0|error=missing-science-cell'); return end; local chest=s.find_entity('wooden-chest',cell.output_position); local labs=s.find_entities_filtered{name='lab',force=f}; local lab=labs[1]; if not chest or not lab then rcon.print(marker..'|ok=0|error=missing-chest-or-lab'); return end; local n=chest.get_item_count('automation-science-pack'); local take=math.min(n,20); if take<1 then rcon.print(marker..'|ok=0|error=no-science-output'); return end; local removed=chest.remove_item{name='automation-science-pack',count=take}; local inserted=lab.insert{name='automation-science-pack',count=removed}; if inserted<removed then chest.insert{name='automation-science-pack',count=removed-inserted} end; if inserted<1 then rcon.print(marker..'|ok=0|error=lab-full'); return end; rcon.print(marker..'|ok=1|moved='..inserted); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_connect_factory_power(self) -> Dict[str, Any]:
        """Build verified pole routes to every unpowered electric consumer."""
        marker = f"{Runtime_RUNTIME_MARKER}_POWER"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local f=p.force; local engines=s.find_entities_filtered{name='steam-engine',force=f}; local engine=engines[1]; if not engine then rcon.print(marker..'|ok=0|error=missing-steam-engine'); return end; "
            "local created={}; local connected=0; local function d2(a,b) local dx=a.position.x-b.position.x; local dy=a.position.y-b.position.y; return dx*dx+dy*dy end; "
            "local function nearest_pole(a,maxd2) local best=nil; local bd=maxd2; for _,e in ipairs(s.find_entities_filtered{name='small-electric-pole',force=f}) do local v=d2(a,e); if v<bd then best=e; bd=v end end; return best end; "
            "local function create_pole(pos) if p.get_item_count('small-electric-pole')<1 then return nil end; if not s.can_place_entity{name='small-electric-pole',position=pos,force=f} then return nil end; local e=s.create_entity{name='small-electric-pole',position=pos,force=f,raise_built=true,auto_connect=true}; if not e then return nil end; if p.remove_item{name='small-electric-pole',count=1}~=1 then e.destroy(); return nil end; created[#created+1]=e; return e end; "
            "local function build_near(a) local bb=a.bounding_box; local cx=(bb.left_top.x+bb.right_bottom.x)/2; local cy=(bb.left_top.y+bb.right_bottom.y)/2; for r=1,6 do local c={{x=bb.left_top.x-r,y=cy},{x=bb.right_bottom.x+r,y=cy},{x=cx,y=bb.left_top.y-r},{x=cx,y=bb.right_bottom.y+r},{x=bb.left_top.x-r,y=bb.left_top.y-r},{x=bb.right_bottom.x+r,y=bb.left_top.y-r},{x=bb.left_top.x-r,y=bb.right_bottom.y+r},{x=bb.right_bottom.x+r,y=bb.right_bottom.y+r}}; for _,pos in ipairs(c) do local e=create_pole(pos); if e then return e end end end; return nil end; "
            "local source=nearest_pole(engine,36) or build_near(engine); if not source then rcon.print(marker..'|ok=0|error=cannot-place-source-pole'); return end; "
            "local targets={}; for _,n in ipairs({'electric-mining-drill','inserter','assembling-machine-1','lab'}) do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not (ok and c) then targets[#targets+1]=e end end end; "
            "for _,target in ipairs(targets) do local receiver=nearest_pole(target,25) or build_near(target); if not receiver then rcon.print(marker..'|ok=0|error=cannot-place-receiver-pole|target='..target.name); return end; local sx=source.position.x; local sy=source.position.y; local dx=receiver.position.x-sx; local dy=receiver.position.y-sy; local dist=math.sqrt(dx*dx+dy*dy); local segments=math.max(1,math.ceil(dist/6)); for i=1,segments-1 do local tx=sx+dx*i/segments; local ty=sy+dy*i/segments; local built=nil; for _,o in ipairs({{0,0},{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}}) do built=create_pole({x=tx+o[1],y=ty+o[2]}); if built then break end end; if not built and not nearest_pole({position={x=tx,y=ty}},36) then rcon.print(marker..'|ok=0|error=cannot-route-poles'); return end end; local ok2,c2=pcall(function() return target.is_connected_to_electric_network() end); if ok2 and c2 then connected=connected+1 else rcon.print(marker..'|ok=0|error=target-still-unpowered|target='..target.name); return end end; rcon.print(marker..'|ok=1|connected='..connected..'|created='..#created..'|remaining='..p.get_item_count('small-electric-pole')); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _queue_research_runtime(self, target: str) -> Dict[str, Any]:
        if target in TRIGGER_TECHNOLOGIES:
            raise RuntimeError(
                f"{target} is a trigger technology and cannot be queued as lab research"
            )
        if target not in LAB_RESEARCH_TARGETS:
            raise RuntimeError(f"unsupported runtime lab research target: {target}")
        return super()._queue_research_runtime(target)

    def _clear_storage_feasibility(self) -> None:
        try:
            self.instance.rcon_client.send_command(
                "/sc storage.fle_rl_runtime={lines={},cells={}}"
            )
        except Exception:
            pass

    def _observation(self) -> np.ndarray:
        # runtime validates its 79-feature result against ``self.observation_space``.
        # This subclass extends that result by 20 features, so temporarily
        # expose the inherited shape while asking runtime to construct its vector.
        final_space = self.observation_space
        inherited_count = len(self.observation_feature_names) - 20
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(inherited_count,),
            dtype=np.float32,
        )
        try:
            base = FactorioOpenEndedEnvObservation._observation(self)
        finally:
            self.observation_space = final_space
        f = self.feasibility_state
        extras = np.asarray(
            [
                float(bool(f.get("trigger_researched__electronics", False))),
                float(bool(f.get("trigger_researched__automation-science-pack", False))),
                float(bool(f.get("recipe_enabled__electronic-circuit", False))),
                float(bool(f.get("recipe_enabled__transport-belt", False))),
                float(bool(f.get("recipe_enabled__inserter", False))),
                float(bool(f.get("recipe_enabled__lab", False))),
                float(bool(f.get("recipe_enabled__automation-science-pack", False))),
                float(bool(f.get("recipe_enabled__assembling-machine-1", False))),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("transport_belt_count"))))),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("inserter_count"))))),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("assembling_machine_count"))))),
                min(10.0, safe_float(f.get("gear_assembler_count"))),
                min(10.0, safe_float(f.get("science_assembler_count"))),
                min(10.0, safe_float(f.get("assembler_working_count"))),
                min(10.0, safe_float(f.get("assembler_unpowered_count"))),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("buffered_iron_gear_wheel")))) / 5.0),
                min(10.0, math.log1p(max(0.0, safe_float(f.get("buffered_automation_science_pack")))) / 5.0),
                min(10.0, safe_float(f.get("iron_smelting_line_count"))),
                min(10.0, safe_float(f.get("copper_smelting_line_count"))),
                min(10.0, safe_float(f.get("factory_unpowered_count"))),
            ],
            dtype=np.float32,
        )
        result = np.concatenate([base, extras])
        if result.shape != self.observation_space.shape:
            raise RuntimeError(
                f"runtime observation mismatch: {result.shape} != {self.observation_space.shape}"
            )
        return result.astype(np.float32)

    def _base_info(self) -> Dict[str, Any]:
        info = super()._base_info()
        info.update(
            {
                "runtime_feasibility_state": json_safe(self.feasibility_state),
                "machine_output_units_total": float(self.machine_output_units_total),
                "manual_crafted_units_total": float(self.manual_crafted_units_total),
                "manual_harvested_units_total": float(self.manual_harvested_units_total),
                "lifetime_machine_output_units_total": float(self.lifetime_machine_output_units_total),
                "lifetime_manual_crafted_units_total": float(self.lifetime_manual_crafted_units_total),
                "lifetime_manual_harvested_units_total": float(self.lifetime_manual_harvested_units_total),
                "lifetime_reward_components": dict(self.lifetime_reward_components),
                "trigger_technologies": {
                    name: bool(self.feasibility_state.get(f"trigger_researched__{name}", False))
                    for name in TRIGGER_TECHNOLOGIES
                },
                "logistics_state": {
                    key: safe_float(self.feasibility_state.get(key))
                    for key in (
                        "transport_belt_count",
                        "inserter_count",
                        "assembling_machine_count",
                        "iron_smelting_line_count",
                        "copper_smelting_line_count",
                        "gear_assembler_count",
                        "science_assembler_count",
                        "assembler_working_count",
                        "factory_unpowered_count",
                    )
                },
            }
        )
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        observation, info = super().reset(seed=seed, options=options)
        hard = bool(dict(options or {}).get("hard_reset", self.hard_reset))
        if hard:
            self._clear_storage_feasibility()
            self.machine_output_units_total = 0.0
            self.manual_crafted_units_total = 0.0
            self.manual_harvested_units_total = 0.0
            self.pending_manual_craft_output.clear()
            self.pending_manual_harvest_output.clear()
        self._collect_feasibility_state(force=True)
        return self._observation(), {
            **self._base_info(),
            "action_mask": self.action_masks(),
            "hard_reset": hard,
        }

    def step(self, action: int):
        before_stats = {
            bucket: dict(self.last_production_stats.get(bucket, {}))
            for bucket in ("output", "crafted", "harvested")
        }
        before_researched = self._researched_count(self.infrastructure_state)
        before_progress = safe_float(self.infrastructure_state.get("research_progress"))
        before_lines = safe_float(self.feasibility_state.get("iron_smelting_line_count")) + safe_float(self.feasibility_state.get("copper_smelting_line_count"))
        before_cells = safe_float(self.feasibility_state.get("gear_assembler_count")) + safe_float(self.feasibility_state.get("science_assembler_count"))
        # runtime updates these counters with a permissive default item weight. Keep
        # authoritative pre-step snapshots so runtime can replace, not merely add
        # to, those values after strict attribution.
        prior_machine_total = float(self.machine_output_total)
        prior_lifetime_machine_total = float(self.lifetime_machine_output_total)
        prior_manual_crafted_total = float(self.manual_crafted_total)
        prior_lifetime_manual_crafted_total = float(self.lifetime_manual_crafted_total)
        prior_manual_harvested_total = float(self.manual_harvested_total)
        prior_lifetime_manual_harvested_total = float(self.lifetime_manual_harvested_total)
        prior_machine_by_item = Counter(self.machine_output_by_item)
        prior_lifetime_machine_by_item = Counter(self.lifetime_machine_output_by_item)
        prior_manual_crafted_by_item = Counter(self.manual_crafted_by_item)
        prior_lifetime_manual_crafted_by_item = Counter(self.lifetime_manual_crafted_by_item)
        prior_manual_harvested_by_item = Counter(self.manual_harvested_by_item)
        prior_lifetime_manual_harvested_by_item = Counter(self.lifetime_manual_harvested_by_item)

        observation, reward, terminated, truncated, info = super().step(action)
        del observation
        action_name = str(info.get("action_name", ""))

        runtime_dispatch = {
            "build_iron_smelting_line": lambda: self._runtime_create_compact_line("iron-ore"),
            "build_copper_smelting_line": lambda: self._runtime_create_compact_line("copper-ore"),
            "build_gear_assembly_cell": lambda: self._runtime_create_assembly_cell("iron-gear-wheel", "gear"),
            "build_science_assembly_cell": lambda: self._runtime_create_assembly_cell("automation-science-pack", "science"),
            "stock_gear_assembler": lambda: self._runtime_stock_assembly_cell("gear"),
            "stock_science_assembler": lambda: self._runtime_stock_assembly_cell("science"),
            "feed_lab_from_science_output": self._runtime_feed_lab,
            "connect_factory_power": self._runtime_connect_factory_power,
        }
        runtime_call = runtime_dispatch.get(action_name)
        if runtime_call is not None and not info.get("action_error"):
            try:
                info["runtime_action_factory"] = runtime_call()
                # runtime: do not refresh here; the unconditional authoritative
                # refresh immediately below runs after this mutation and is
                # sufficient for reward/mask/state accounting.
            except Exception as exc:
                info["action_error"] = f"{type(exc).__name__}: {exc}"
                self.total_action_errors += 1
                reward -= self.execution_error_penalty
                self.episode_return -= self.execution_error_penalty

        # Force a production refresh after runtime macros and waits.
        self._collect_compact_state(include_production_stats=True)
        self._collect_infrastructure_state(force=True)
        self._collect_feasibility_state(force=True)
        after_stats = {
            bucket: dict(self.last_production_stats.get(bucket, {}))
            for bucket in ("output", "crafted", "harvested")
        }
        output_delta = self._bucket_delta(before_stats["output"], after_stats["output"])
        crafted_delta = self._bucket_delta(before_stats["crafted"], after_stats["crafted"])
        harvested_delta = self._bucket_delta(before_stats["harvested"], after_stats["harvested"])

        delayed_manual_craft: Dict[str, float] = {}
        delayed_manual_harvest: Dict[str, float] = {}
        for item, amount in output_delta.items():
            craft_credit = min(amount, self.pending_manual_craft_output.get(item, 0.0))
            if craft_credit > 0:
                delayed_manual_craft[item] = craft_credit
                self.pending_manual_craft_output[item] -= craft_credit
            remaining = max(0.0, amount - craft_credit)
            harvest_credit = min(remaining, self.pending_manual_harvest_output.get(item, 0.0))
            if harvest_credit > 0:
                delayed_manual_harvest[item] = harvest_credit
                self.pending_manual_harvest_output[item] -= harvest_credit

        # Explicit action attribution closes gaps in delayed FLE production
        # buckets. Only the selected manual action is attributed manually.
        explicit_craft: Dict[str, float] = {}
        explicit_harvest: Dict[str, float] = {}
        if not info.get("action_error") and action_name in MANUAL_CRAFT_OUTPUTS:
            item, amount = MANUAL_CRAFT_OUTPUTS[action_name]
            explicit_craft[item] = amount
        if not info.get("action_error") and action_name in MANUAL_HARVEST_OUTPUTS:
            item, amount = MANUAL_HARVEST_OUTPUTS[action_name]
            explicit_harvest[item] = amount
        for item, amount in explicit_craft.items():
            crafted_delta[item] = max(crafted_delta.get(item, 0.0), amount)
        for item, amount in explicit_harvest.items():
            harvested_delta[item] = max(harvested_delta.get(item, 0.0), amount)
        for item, amount in explicit_craft.items():
            uncovered = max(0.0, amount - output_delta.get(item, 0.0))
            if uncovered > 0:
                self.pending_manual_craft_output[item] += uncovered
        for item, amount in explicit_harvest.items():
            uncovered = max(0.0, amount - output_delta.get(item, 0.0))
            if uncovered > 0:
                self.pending_manual_harvest_output[item] += uncovered

        machine_delta: Dict[str, float] = {}
        for item, amount in output_delta.items():
            if item not in MACHINE_OUTPUT_WEIGHTS:
                continue
            residual = max(
                0.0,
                amount
                - crafted_delta.get(item, 0.0)
                - harvested_delta.get(item, 0.0)
                - delayed_manual_craft.get(item, 0.0)
                - delayed_manual_harvest.get(item, 0.0),
            )
            if residual > 0:
                machine_delta[item] = residual

        machine_weighted = self._strict_weight(machine_delta)
        crafted_weighted = self._strict_weight(crafted_delta)
        harvested_weighted = self._strict_weight(harvested_delta)
        machine_units = self._raw_units(machine_delta)
        crafted_units = self._raw_units(crafted_delta)
        harvested_units = self._raw_units(harvested_delta)

        # Replace all runtime per-step production accounting with strict runtime
        # attribution. Unknown outputs such as water and steam disappear, while
        # explicit manual craft/harvest outputs remain in their own counters.
        self.machine_output_total = prior_machine_total + machine_weighted
        self.lifetime_machine_output_total = prior_lifetime_machine_total + machine_weighted
        self.manual_crafted_total = prior_manual_crafted_total + crafted_weighted
        self.lifetime_manual_crafted_total = prior_lifetime_manual_crafted_total + crafted_weighted
        self.manual_harvested_total = prior_manual_harvested_total + harvested_weighted
        self.lifetime_manual_harvested_total = prior_lifetime_manual_harvested_total + harvested_weighted
        self.machine_output_by_item = Counter(prior_machine_by_item)
        self.machine_output_by_item.update(machine_delta)
        self.lifetime_machine_output_by_item = Counter(prior_lifetime_machine_by_item)
        self.lifetime_machine_output_by_item.update(machine_delta)
        self.manual_crafted_by_item = Counter(prior_manual_crafted_by_item)
        self.manual_crafted_by_item.update(crafted_delta)
        self.lifetime_manual_crafted_by_item = Counter(prior_lifetime_manual_crafted_by_item)
        self.lifetime_manual_crafted_by_item.update(crafted_delta)
        self.manual_harvested_by_item = Counter(prior_manual_harvested_by_item)
        self.manual_harvested_by_item.update(harvested_delta)
        self.lifetime_manual_harvested_by_item = Counter(prior_lifetime_manual_harvested_by_item)
        self.lifetime_manual_harvested_by_item.update(harvested_delta)
        self.automated_output_total = self.machine_output_total

        self.machine_output_units_total += machine_units
        self.manual_crafted_units_total += crafted_units
        self.manual_harvested_units_total += harvested_units
        self.lifetime_machine_output_units_total += machine_units
        self.lifetime_manual_crafted_units_total += crafted_units
        self.lifetime_manual_harvested_units_total += harvested_units

        # runtime already replaced the base output reward, but its default weighting
        # included unknown items. Replace that component a second time with the
        # strict whitelist value.
        old_parts = dict(info.get("reward_parts", {}))
        old_machine_reward = safe_float(old_parts.get("automated_output"))
        strict_machine_reward = self.automated_output_reward_scale * machine_weighted
        reward += strict_machine_reward - old_machine_reward
        self.episode_return += strict_machine_reward - old_machine_reward

        after_researched = self._researched_count(self.infrastructure_state)
        after_progress = safe_float(self.infrastructure_state.get("research_progress"))
        progress_delta = max(0.0, after_progress - before_progress)
        completion_delta = max(0, after_researched - before_researched)
        research_activity = (
            self.research_activity_reward
            if safe_float(self.feasibility_state.get("lab_working_count")) > 0
            else 0.0
        )
        reward += research_activity
        self.episode_return += research_activity

        after_lines = safe_float(self.feasibility_state.get("iron_smelting_line_count")) + safe_float(self.feasibility_state.get("copper_smelting_line_count"))
        after_cells = safe_float(self.feasibility_state.get("gear_assembler_count")) + safe_float(self.feasibility_state.get("science_assembler_count"))
        logistics_milestone = 0.0
        if before_lines <= 0 < after_lines:
            logistics_milestone += self.first_smelting_line_reward
        if before_cells <= 0 < after_cells:
            logistics_milestone += self.first_assembler_reward
        reward += logistics_milestone
        self.episode_return += logistics_milestone

        parts = dict(old_parts)
        parts.update(
            {
                "machine_production": float(strict_machine_reward),
                "automated_output": float(strict_machine_reward),
                "research_progress": float(
                    self.research_progress_reward * progress_delta
                ),
                "research_completion": float(
                    self.research_completion_reward * completion_delta
                ),
                "research_activity": float(research_activity),
                "logistics_milestone": float(logistics_milestone),
                "manual_craft_reward": 0.0,
                "manual_harvest_reward": 0.0,
                "total": float(reward),
            }
        )
        for key, value in parts.items():
            if key != "total":
                self.lifetime_reward_components[key] += safe_float(value)

        final_info = dict(info)
        final_info.update(self._base_info())
        final_info.update(
            {
                "reward": float(reward),
                "reward_parts": parts,
                "machine_output_delta": machine_delta,
                "machine_output_units_delta": float(machine_units),
                "manual_crafted_delta": crafted_delta,
                "manual_harvested_delta": harvested_delta,
                "delayed_manual_craft_attribution": delayed_manual_craft,
                "delayed_manual_harvest_attribution": delayed_manual_harvest,
                "pending_manual_craft_output": dict(self.pending_manual_craft_output),
                "pending_manual_harvest_output": dict(self.pending_manual_harvest_output),
                "action_mask": self.action_masks(),
                "terminal_goal": None,
            }
        )
        return self._observation(), float(reward), bool(terminated), bool(truncated), final_info


FactorioOpenEndedEnvImprovedRuntime = FactorioOpenEndedEnvFeasibility

__all__ = [
    "FactorioOpenEndedEnvImprovedRuntime",
    "FactorioOpenEndedEnvFeasibility",
    "LAB_RESEARCH_TARGETS",
    "MACHINE_OUTPUT_WEIGHTS",
    "TRIGGER_TECHNOLOGIES",
    "Runtime_ACTIONS",
    "Runtime_DEFAULT_INVENTORY",
]
