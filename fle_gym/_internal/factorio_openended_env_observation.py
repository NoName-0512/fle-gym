from __future__ import annotations

"""Improved open-ended Factorio environment for MaskablePPO.

This version keeps the task open-ended while making training episodic only by a
fixed *time horizon*. A horizon is not a production goal: it gives PPO fresh
factory attempts and a well-defined return. Set ``training_horizon_steps=0``
for a persistent evaluation factory.

The environment adds:

* resource-chain actions for copper plates, iron gears, and red science;
* live runtime feasibility telemetry used by the action mask;
* locked, prerequisite-aware research selection;
* active-power telemetry separate from mere network connectivity;
* separate machine-output, manual-crafting, and manual-harvest accounting;
* lifetime counters that remain valid across training-horizon resets.
"""

from collections import Counter
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from factorio_openended_env import (
    AUTOMATED_OUTPUT_WEIGHTS,
    DEFAULT_OPENENDED_INVENTORY,
    FactorioOpenEndedEnv,
    OPENENDED_ACTIONS,
    OPENENDED_TRACKED_ENTITIES,
    OPENENDED_TRACKED_ITEMS,
    RESEARCH_TARGETS,
)
from factorio_transition_env import ActionSpec, _dedent, json_safe, safe_float, spaces


FEASIBILITY_MARKER = "__FLE_RL_FEASIBILITY_runtime__"


IMPROVED_DEFAULT_INVENTORY: Dict[str, int] = dict(DEFAULT_OPENENDED_INVENTORY)
IMPROVED_DEFAULT_INVENTORY.update(
    {
        # The research chain is no longer gifted. The policy must smelt plates,
        # craft gears, and craft science packs itself.
        "iron-plate": 0,
        "copper-plate": 0,
        "iron-gear-wheel": 0,
        "automation-science-pack": 0,
        # Enough coal to bootstrap one useful power system without rewarding
        # endless manual coal harvesting.
        "coal": 100,
        # One coherent power chain is sufficient for the current curriculum.
        "offshore-pump": 1,
        "boiler": 1,
        "steam-engine": 1,
        "pipe": 80,
        "small-electric-pole": 40,
        "lab": 2,
        "electric-mining-drill": 5,
        "stone-furnace": 3,
        "wooden-chest": 8,
    }
)


EXTRA_ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec(
        name="place_electric_drill_on_copper",
        category="power",
        required_items={"electric-mining-drill": 1},
        produced_entity="electric-mining-drill",
        description="Place an electric mining drill on the nearest copper patch.",
        code=_dedent(
            """
            ore = nearest(Prototype.CopperOre)
            move_to(ore)
            drill = place_entity(
                Prototype.ElectricMiningDrill,
                position=ore,
                direction=Direction.DOWN,
                exact=False,
            )
            if drill is None:
                raise RuntimeError("electric copper drill placement returned None")
            print("__FLE_RL_EVENT__:electric_copper_drill_placed")
            """
        ),
    ),
    ActionSpec(
        name="harvest_copper_ore",
        category="harvest",
        description="Recovery action: manually harvest ten copper ore.",
        code=_dedent(
            """
            target = nearest(Prototype.CopperOre)
            move_to(target)
            result = harvest_resource(target, 10)
            print("__FLE_RL_EVENT__:copper_harvested")
            print("harvest_result=" + repr(result))
            """
        ),
    ),
    ActionSpec(
        name="insert_copper_ore",
        category="insert",
        required_items={"copper-ore": 1},
        required_entities={"stone-furnace": 1},
        description="Insert one copper ore into an available stone furnace.",
        code=_dedent(
            """
            furnace_position = nearest(Prototype.StoneFurnace)
            furnace = get_entity(Prototype.StoneFurnace, furnace_position)
            result = insert_item(Prototype.CopperOre, furnace, quantity=1)
            print("__FLE_RL_EVENT__:copper_ore_inserted")
            print("insert_result=" + repr(result))
            """
        ),
    ),
    ActionSpec(
        name="collect_copper_plate",
        category="extract",
        required_entities={"stone-furnace": 1},
        description="Collect one available copper plate from a stone furnace.",
        code=_dedent(
            """
            furnace_position = nearest(Prototype.StoneFurnace)
            furnace = get_entity(Prototype.StoneFurnace, furnace_position)
            extracted = extract_item(Prototype.CopperPlate, furnace, quantity=1)
            if extracted is None or extracted == 0:
                raise RuntimeError("no copper plate available in stone furnace")
            print("__FLE_RL_EVENT__:copper_plate_collected")
            print("extract_result=" + repr(extracted))
            """
        ),
    ),
    ActionSpec(
        name="craft_iron_gears",
        category="craft",
        required_items={"iron-plate": 20},
        description="Craft ten iron gear wheels from twenty iron plates.",
        code=_dedent(
            """
            result = craft_item(Prototype.IronGearWheel, quantity=10)
            print("__FLE_RL_EVENT__:iron_gears_crafted")
            print("craft_result=" + repr(result))
            """
        ),
    ),
    ActionSpec(
        name="transfer_copper_chest_ore_to_furnace",
        category="transfer",
        required_entities={"wooden-chest": 1, "stone-furnace": 1},
        description="Move copper ore from any output chest into a compatible furnace.",
        code='print("__FLE_RL_EVENT__:runtime_copper_transfer_requested")',
    ),
)


def _improved_actions() -> Tuple[ActionSpec, ...]:
    """Replace error-prone nearest-entity macros with verified runtime variants."""
    runtime_replacements: Dict[str, Dict[str, Any]] = {
        "fuel_drill": {
            "code": 'print("__FLE_RL_EVENT__:runtime_fuel_drill_requested")',
            "single_use_flag": None,
        },
        "fuel_furnace": {
            "code": 'print("__FLE_RL_EVENT__:runtime_fuel_furnace_requested")',
            "single_use_flag": None,
        },
        "fuel_boiler": {
            "code": 'print("__FLE_RL_EVENT__:runtime_fuel_boiler_requested")',
            "single_use_flag": None,
        },
        "insert_hand_mined_ore": {
            "code": 'print("__FLE_RL_EVENT__:runtime_iron_insert_requested")',
        },
        "collect_iron_plate": {
            "code": 'print("__FLE_RL_EVENT__:runtime_iron_collect_requested")',
        },
        "place_chest_at_burner_output": {
            "code": 'print("__FLE_RL_EVENT__:runtime_burner_chest_requested")',
        },
        "place_chest_at_electric_output": {
            "code": 'print("__FLE_RL_EVENT__:runtime_electric_chest_requested")',
        },
        "transfer_chest_ore_to_furnace": {
            "code": 'print("__FLE_RL_EVENT__:runtime_iron_transfer_requested")',
        },
        "connect_electric_drill_to_power": {
            "required_items": {"small-electric-pole": 2},
            "single_use_flag": None,
            "description": (
                "Connect every currently unpowered electric mining drill to "
                "the steam network using verified runtime pole routes."
            ),
            "code": 'print("__FLE_RL_EVENT__:runtime_electric_drill_power_requested")',
        },
    }
    actions: List[ActionSpec] = []
    for spec in OPENENDED_ACTIONS:
        updates = runtime_replacements.get(spec.name)
        actions.append(replace(spec, **updates) if updates else spec)

    for spec in EXTRA_ACTIONS:
        if spec.name == "insert_copper_ore":
            spec = replace(
                spec,
                code='print("__FLE_RL_EVENT__:runtime_copper_insert_requested")',
            )
        elif spec.name == "collect_copper_plate":
            spec = replace(
                spec,
                code='print("__FLE_RL_EVENT__:runtime_copper_collect_requested")',
            )
        actions.append(spec)
    return tuple(actions)


IMPROVED_ACTIONS: Tuple[ActionSpec, ...] = _improved_actions()


class FactorioOpenEndedEnvObservation(FactorioOpenEndedEnv):
    """Open-ended factory task with horizon training and live feasibility masks."""

    def __init__(
        self,
        *,
        training_horizon_steps: int = 512,
        initial_inventory: Optional[Mapping[str, int]] = None,
        feasibility_state_every: int = 1,
        idle_action_penalty: float = 0.003,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("initial_inventory", initial_inventory or IMPROVED_DEFAULT_INVENTORY)
        kwargs.setdefault("infrastructure_state_every", 1)
        kwargs.setdefault("production_stats_every", 1)
        super().__init__(**kwargs)

        self.actions = IMPROVED_ACTIONS
        self.action_space = spaces.Discrete(len(self.actions))
        self.training_horizon_steps = max(0, int(training_horizon_steps))
        self.feasibility_state_every = max(1, int(feasibility_state_every))
        self.idle_action_penalty = abs(float(idle_action_penalty))

        self.feasibility_state: Dict[str, Any] = self._empty_feasibility_state()
        self.machine_output_total = 0.0
        self.machine_output_by_item: Counter[str] = Counter()
        self.manual_crafted_total = 0.0
        self.manual_crafted_by_item: Counter[str] = Counter()
        self.manual_harvested_total = 0.0
        self.manual_harvested_by_item: Counter[str] = Counter()

        self.lifetime_machine_output_total = 0.0
        self.lifetime_manual_crafted_total = 0.0
        self.lifetime_manual_harvested_total = 0.0
        self.lifetime_machine_output_by_item: Counter[str] = Counter()
        self.lifetime_manual_crafted_by_item: Counter[str] = Counter()
        self.lifetime_manual_harvested_by_item: Counter[str] = Counter()
        self.lifetime_action_counts: Counter[str] = Counter()
        self.lifetime_successful_action_counts: Counter[str] = Counter()
        self.completed_horizons = 0

        extra_features = (
            "feasible__inventory_full",
            "feasible__furnace_iron_plate",
            "feasible__furnace_copper_plate",
            "feasible__chest_iron_ore",
            "feasible__chest_copper_ore",
            "feasible__lab_working",
            "feasible__active_power",
            *(f"current_research__{target}" for target in RESEARCH_TARGETS),
            "machine_output_total",
            "manual_crafted_total",
            "manual_harvested_total",
            "horizon_progress",
        )
        self.observation_feature_names = tuple(self.observation_feature_names) + extra_features
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )
        self._write_improved_config()

    @staticmethod
    def _empty_feasibility_state() -> Dict[str, Any]:
        return {
            "query_ok": False,
            "query_error": "not queried",
            "inventory_full": False,
            "burner_drill_fuel_space": 0.0,
            "boiler_fuel_space": 0.0,
            "furnace_fuel_space": 0.0,
            "furnace_iron_input_space": 0.0,
            "furnace_copper_input_space": 0.0,
            "furnace_iron_plate": 0.0,
            "furnace_copper_plate": 0.0,
            "chest_iron_ore": 0.0,
            "chest_copper_ore": 0.0,
            "lab_science_space": 0.0,
            "lab_science_count": 0.0,
            "lab_working_count": 0.0,
            "active_power": False,
            "electric_iron_drill_count": 0.0,
            "electric_copper_drill_count": 0.0,
            "electric_unpowered_count": 0.0,
            "electric_iron_unpowered_count": 0.0,
            "electric_copper_unpowered_count": 0.0,
            "burner_output_chest_placeable": 0.0,
            "electric_output_chest_placeable": 0.0,
        }

    def _write_improved_config(self) -> None:
        payload = {
            "environment": "FactorioOpenEndedEnvObservation",
            "training_horizon_steps": self.training_horizon_steps,
            "terminal_production_goal": None,
            "action_count": len(self.actions),
            "actions": [asdict(spec) for spec in self.actions],
            "initial_inventory": dict(self.initial_inventory_config),
            "metric_separation": {
                "machine_output": "output minus crafted minus harvested",
                "manual_crafted": "FLE crafted bucket",
                "manual_harvested": "FLE harvested bucket",
            },
        }
        Path(self.run_dir, "improved_environment_config.json").write_text(
            json.dumps(json_safe(payload), indent=2),
            encoding="utf-8",
        )

    def _feasibility_command(self) -> str:
        marker = json.dumps(FEASIBILITY_MARKER)
        insert_items = (
            "coal",
            "stone",
            "iron-ore",
            "copper-ore",
            "iron-plate",
            "copper-plate",
            "iron-gear-wheel",
            "automation-science-pack",
        )
        insert_lua = "{" + ",".join(json.dumps(name) for name in insert_items) + "}"
        return (
            "/sc local p=nil; "
            "if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then p=storage.agent_characters[1] "
            "elseif player and player.valid then p=player "
            "elseif game.get_player(1) then p=game.get_player(1) end; "
            "if not p then rcon.print(" + marker + "..'|ERROR|no-agent') else "
            "local s=p.surface; local f=p.force; local out={}; "
            "local main=p.get_main_inventory and p.get_main_inventory() or nil; "
            "out[#out+1]='inventory_full='..((main and main.is_full()) and '1' or '0'); "
            "local items=" + insert_lua + "; for _,n in ipairs(items) do "
            "local ok,c=pcall(function() return p.can_insert({name=n,count=1}) end); "
            "out[#out+1]='can_insert__'..n..'='..((ok and c) and '1' or '0'); end; "
            "local function fuel_space(name) local c=0; for _,e in ipairs(s.find_entities_filtered{name=name,force=f}) do "
            "local ok,inv=pcall(function() return e.get_fuel_inventory() end); "
            "if ok and inv and inv.can_insert({name='coal',count=1}) then c=c+1 end end return c end; "
            "out[#out+1]='burner_drill_fuel_space='..fuel_space('burner-mining-drill'); "
            "out[#out+1]='boiler_fuel_space='..fuel_space('boiler'); "
            "out[#out+1]='furnace_fuel_space='..fuel_space('stone-furnace'); "
            "local fi=0; local fc=0; local ip=0; local cp=0; "
            "for _,e in ipairs(s.find_entities_filtered{name='stone-furnace',force=f}) do "
            "local oki,ci=pcall(function() return e.can_insert({name='iron-ore',count=1}) end); if oki and ci then fi=fi+1 end; "
            "local okc,cc=pcall(function() return e.can_insert({name='copper-ore',count=1}) end); if okc and cc then fc=fc+1 end; "
            "local okip,nip=pcall(function() return e.get_item_count('iron-plate') end); if okip and nip then ip=ip+nip end; "
            "local okcp,ncp=pcall(function() return e.get_item_count('copper-plate') end); if okcp and ncp then cp=cp+ncp end; end; "
            "out[#out+1]='furnace_iron_input_space='..fi; out[#out+1]='furnace_copper_input_space='..fc; "
            "out[#out+1]='furnace_iron_plate='..ip; out[#out+1]='furnace_copper_plate='..cp; "
            "local cio=0; local cco=0; for _,e in ipairs(s.find_entities_filtered{name='wooden-chest',force=f}) do "
            "cio=cio+(e.get_item_count('iron-ore') or 0); cco=cco+(e.get_item_count('copper-ore') or 0); end; "
            "out[#out+1]='chest_iron_ore='..cio; out[#out+1]='chest_copper_ore='..cco; "
            "local lspace=0; local lscience=0; local lworking=0; local lnopower=0; "
            "for _,e in ipairs(s.find_entities_filtered{name='lab',force=f}) do "
            "local oki,ci=pcall(function() return e.can_insert({name='automation-science-pack',count=1}) end); if oki and ci then lspace=lspace+1 end; "
            "lscience=lscience+(e.get_item_count('automation-science-pack') or 0); "
            "local st=e.status; if st==defines.entity_status.working then lworking=lworking+1 end; "
            "if st==defines.entity_status.no_power or st==defines.entity_status.low_power or st==defines.entity_status.not_plugged_in_electric_network then lnopower=lnopower+1 end; end; "
            "out[#out+1]='lab_science_space='..lspace; out[#out+1]='lab_science_count='..lscience; "
            "out[#out+1]='lab_working_count='..lworking; out[#out+1]='lab_no_power_count='..lnopower; "
            "local iron_drills=0; local copper_drills=0; local electric_working=0; local electric_unpowered=0; local iron_unpowered=0; local copper_unpowered=0; "
            "for _,e in ipairs(s.find_entities_filtered{name='electric-mining-drill',force=f}) do "
            "local mt=e.mining_target; local n=(mt and mt.valid and mt.name) or ''; "
            "if n=='iron-ore' then iron_drills=iron_drills+1 elseif n=='copper-ore' then copper_drills=copper_drills+1 end; "
            "local oknet,connected=pcall(function() return e.is_connected_to_electric_network() end); if not (oknet and connected) then electric_unpowered=electric_unpowered+1; if n=='iron-ore' then iron_unpowered=iron_unpowered+1 elseif n=='copper-ore' then copper_unpowered=copper_unpowered+1 end end; "
            "if e.status==defines.entity_status.working then electric_working=electric_working+1 end; end; "
            "out[#out+1]='electric_iron_drill_count='..iron_drills; out[#out+1]='electric_copper_drill_count='..copper_drills; "
            "out[#out+1]='electric_unpowered_count='..electric_unpowered; out[#out+1]='electric_iron_unpowered_count='..iron_unpowered; out[#out+1]='electric_copper_unpowered_count='..copper_unpowered; "
            "local generated=0; for _,e in ipairs(s.find_entities_filtered{name='steam-engine',force=f}) do "
            "local ok,g=pcall(function() return e.energy_generated_last_tick end); if ok and g then generated=generated+g end end; "
            "local active=(generated>0 or lworking>0 or electric_working>0); out[#out+1]='active_power='..(active and '1' or '0'); "
            "local function chest_placeable(drill_name) local c=0; for _,e in ipairs(s.find_entities_filtered{name=drill_name,force=f}) do "
            "local target=e.drop_target; if not (target and target.valid and target.name=='wooden-chest') and s.can_place_entity{name='wooden-chest',position=e.drop_position,force=f} then c=c+1 end end return c end; "
            "out[#out+1]='burner_output_chest_placeable='..chest_placeable('burner-mining-drill'); "
            "out[#out+1]='electric_output_chest_placeable='..chest_placeable('electric-mining-drill'); "
            "local recipes={'stone-furnace','lab','iron-gear-wheel','automation-science-pack'}; for _,n in ipairs(recipes) do "
            "local ok,c=pcall(function() return p.get_craftable_count(n) end); out[#out+1]='craftable__'..n..'='..((ok and c) or 0); end; "
            "rcon.print(" + marker + "..'|'..table.concat(out,'|')); end"
        )

    def _collect_feasibility_state(self, *, force: bool = False) -> Dict[str, Any]:
        if not force and self.global_step % self.feasibility_state_every != 0:
            return self.feasibility_state
        state = dict(self.feasibility_state)
        try:
            raw = str(self.instance.rcon_client.send_command(self._feasibility_command()) or "")
            line = next((line for line in raw.splitlines() if FEASIBILITY_MARKER in line), "")
            if not line:
                raise RuntimeError(f"feasibility marker missing: {raw[:300]!r}")
            payload = line.split(FEASIBILITY_MARKER, 1)[1].lstrip(": |")
            if payload.startswith("ERROR|"):
                raise RuntimeError(payload)
            for token in payload.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                state[key.strip()] = safe_float(value.strip())
            state["inventory_full"] = bool(state.get("inventory_full", 0.0))
            state["active_power"] = bool(state.get("active_power", 0.0))
            state["query_ok"] = True
            state["query_error"] = ""
        except Exception as exc:
            state["query_ok"] = False
            state["query_error"] = f"{type(exc).__name__}: {exc}"
        self.feasibility_state = state
        return state

    @staticmethod
    def _bucket_delta(before: Mapping[str, float], after: Mapping[str, float]) -> Dict[str, float]:
        keys = set(before) | set(after)
        return {
            key: max(0.0, safe_float(after.get(key)) - safe_float(before.get(key)))
            for key in keys
            if safe_float(after.get(key)) > safe_float(before.get(key))
        }

    @staticmethod
    def _weighted(items: Mapping[str, float]) -> float:
        return sum(float(amount) * AUTOMATED_OUTPUT_WEIGHTS.get(item, 0.05) for item, amount in items.items())

    def get_action_mask(self) -> np.ndarray:
        mask = super().get_action_mask().astype(np.int8)
        f = self.feasibility_state
        query_ok = bool(f.get("query_ok", False))

        def set_valid(name: str, valid: bool) -> None:
            try:
                mask[self.action_index(name)] = int(bool(valid))
            except KeyError:
                pass

        def can_insert(item: str) -> bool:
            return not query_ok or bool(f.get(f"can_insert__{item}", 1.0))

        # Recovery actions are need-driven rather than permanent reward-farming actions.
        set_valid("harvest_coal", can_insert("coal") and self.inventory.get("coal", 0.0) < 20.0)
        set_valid(
            "harvest_stone",
            can_insert("stone")
            and self.inventory.get("stone", 0.0) < 5.0
            and self.inventory.get("stone-furnace", 0.0) < 1.0
            and self.entity_counts.get("stone-furnace", 0.0) < 2.0,
        )
        set_valid(
            "harvest_iron_ore",
            can_insert("iron-ore")
            and self.inventory.get("iron-ore", 0.0) < 10.0
            and (not query_ok or safe_float(f.get("furnace_iron_input_space")) > 0),
        )
        set_valid(
            "harvest_copper_ore",
            can_insert("copper-ore")
            and self.inventory.get("copper-ore", 0.0) < 10.0
            and (not query_ok or safe_float(f.get("furnace_copper_input_space")) > 0),
        )

        set_valid(
            "craft_stone_furnace",
            self.entity_counts.get("stone-furnace", 0.0) < 2.0
            and self.inventory.get("stone-furnace", 0.0) < 1.0
            and (not query_ok or safe_float(f.get("craftable__stone-furnace")) >= 1),
        )
        set_valid(
            "craft_lab",
            self.entity_counts.get("lab", 0.0) < 1.0
            and self.inventory.get("lab", 0.0) < 1.0
            and (not query_ok or safe_float(f.get("craftable__lab")) >= 1),
        )
        set_valid(
            "craft_iron_gears",
            self.inventory.get("iron-gear-wheel", 0.0) < 10.0
            and (not query_ok or safe_float(f.get("craftable__iron-gear-wheel")) >= 10),
        )
        set_valid(
            "craft_automation_science",
            self.inventory.get("automation-science-pack", 0.0) < 20.0
            and (not query_ok or safe_float(f.get("craftable__automation-science-pack")) >= 10),
        )

        # Bound placement macros to useful topology. The objective remains open-ended;
        # these limits prevent repeatedly trying an already saturated placement macro.
        set_valid("place_drill_on_iron", self.entity_counts.get("burner-mining-drill", 0.0) < 2.0 and self.inventory.get("burner-mining-drill", 0.0) >= 1)
        set_valid("place_furnace_near_drill", self.entity_counts.get("stone-furnace", 0.0) < 2.0 and self.inventory.get("stone-furnace", 0.0) >= 1)
        set_valid("place_electric_drill_on_iron", safe_float(f.get("electric_iron_drill_count")) < 3.0 and self.inventory.get("electric-mining-drill", 0.0) >= 1)
        set_valid("place_electric_drill_on_copper", safe_float(f.get("electric_copper_drill_count")) < 2.0 and self.inventory.get("electric-mining-drill", 0.0) >= 1)
        set_valid("place_offshore_pump", self.entity_counts.get("offshore-pump", 0.0) < 1.0 and self.inventory.get("offshore-pump", 0.0) >= 1)
        set_valid("place_boiler_near_pump", self.entity_counts.get("boiler", 0.0) < 1.0 and self.entity_counts.get("offshore-pump", 0.0) >= 1 and self.inventory.get("boiler", 0.0) >= 1)
        set_valid("place_steam_engine_near_boiler", self.entity_counts.get("steam-engine", 0.0) < 1.0 and self.entity_counts.get("boiler", 0.0) >= 1 and self.inventory.get("steam-engine", 0.0) >= 1)
        set_valid("place_small_electric_pole", self.entity_counts.get("small-electric-pole", 0.0) < 1.0 and self.entity_counts.get("steam-engine", 0.0) >= 1 and self.inventory.get("small-electric-pole", 0.0) >= 1)
        set_valid("place_lab", self.entity_counts.get("lab", 0.0) < 2.0 and self.entity_counts.get("steam-engine", 0.0) >= 1 and self.inventory.get("lab", 0.0) >= 1)

        if query_ok:
            set_valid("fuel_drill", safe_float(f.get("burner_drill_fuel_space")) > 0 and self.inventory.get("coal", 0.0) >= 5)
            set_valid("fuel_furnace", safe_float(f.get("furnace_fuel_space")) > 0 and self.inventory.get("coal", 0.0) >= 5)
            set_valid("fuel_boiler", safe_float(f.get("boiler_fuel_space")) > 0 and self.inventory.get("coal", 0.0) >= 10)
            set_valid("collect_iron_plate", safe_float(f.get("furnace_iron_plate")) > 0 and can_insert("iron-plate"))
            set_valid("collect_copper_plate", safe_float(f.get("furnace_copper_plate")) > 0 and can_insert("copper-plate"))
            set_valid("insert_hand_mined_ore", safe_float(f.get("furnace_iron_input_space")) > 0 and self.inventory.get("iron-ore", 0.0) >= 1)
            set_valid("insert_copper_ore", safe_float(f.get("furnace_copper_input_space")) > 0 and self.inventory.get("copper-ore", 0.0) >= 1)
            set_valid("transfer_chest_ore_to_furnace", safe_float(f.get("chest_iron_ore")) > 0 and safe_float(f.get("furnace_iron_input_space")) > 0)
            set_valid("transfer_copper_chest_ore_to_furnace", safe_float(f.get("chest_copper_ore")) > 0 and safe_float(f.get("furnace_copper_input_space")) > 0)
            set_valid("place_chest_at_burner_output", safe_float(f.get("burner_output_chest_placeable")) > 0 and self.inventory.get("wooden-chest", 0.0) >= 1)
            set_valid("place_chest_at_electric_output", safe_float(f.get("electric_output_chest_placeable")) > 0 and self.inventory.get("wooden-chest", 0.0) >= 1)

        set_valid(
            "connect_electric_drill_to_power",
            self.entity_counts.get("steam-engine", 0.0) >= 1
            and safe_float(f.get("electric_unpowered_count")) > 0
            and self.inventory.get("small-electric-pole", 0.0) >= 2,
        )

        lab_count = self.entity_counts.get("lab", 0.0)
        lab_connected = safe_float(self.infrastructure_state.get("lab_connected_count"))
        set_valid("connect_lab_to_power", lab_count > lab_connected and self.inventory.get("small-electric-pole", 0.0) >= 2)

        current = str(self.infrastructure_state.get("current_research", "") or "")
        for target in RESEARCH_TARGETS:
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
            and self.inventory.get("automation-science-pack", 0.0) >= 10
            and (not query_ok or safe_float(f.get("lab_science_space")) > 0),
        )
        set_valid("wait_research", bool(current))
        set_valid("noop", True)
        set_valid("wait", bool(sum(self.entity_counts.values()) > 0))
        return mask

    @staticmethod
    def _agent_lua() -> str:
        return (
            "local p=nil; "
            "if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then p=storage.agent_characters[1] "
            "elseif player and player.valid then p=player "
            "elseif game.get_player(1) then p=game.get_player(1) end; "
        )

    def _runtime_fuel(self, *, entity_name: str, quantity: int) -> Dict[str, Any]:
        marker = "__FLE_RL_FUEL_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._agent_lua()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            "local available=p.get_item_count('coal'); if available<" + str(int(quantity)) + " then rcon.print(marker..'|ok=0|error=missing-coal|available='..available); return end; "
            "local target=nil; for _,e in ipairs(p.surface.find_entities_filtered{name=" + json.dumps(entity_name) + ",force=p.force}) do local ok,inv=pcall(function() return e.get_fuel_inventory() end); if ok and inv and inv.can_insert({name='coal',count=1}) then target=e; break end end; "
            "if not target then rcon.print(marker..'|ok=0|error=no-fuel-space'); return end; "
            "local removed=p.remove_item{name='coal',count=" + str(int(quantity)) + "}; if removed<1 then rcon.print(marker..'|ok=0|error=remove-failed'); return end; "
            "local inserted=target.insert{name='coal',count=removed}; if inserted<removed then p.insert{name='coal',count=removed-inserted} end; "
            "if inserted<1 then rcon.print(marker..'|ok=0|error=insert-failed'); return end; rcon.print(marker..'|ok=1|inserted='..inserted..'|entity='..target.name..'|remaining='..p.get_item_count('coal')); end; "
            "local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_insert_furnace(self, *, item_name: str, quantity: int = 1) -> Dict[str, Any]:
        marker = "__FLE_RL_FURNACE_INSERT_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._agent_lua()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local item=" + json.dumps(item_name) + "; "
            "local available=p.get_item_count(item); if available<1 then rcon.print(marker..'|ok=0|error=missing-item'); return end; "
            "local target=nil; for _,e in ipairs(p.surface.find_entities_filtered{name='stone-furnace',force=p.force}) do local ok,c=pcall(function() return e.can_insert({name=item,count=1}) end); if ok and c then target=e; break end end; "
            "if not target then rcon.print(marker..'|ok=0|error=no-compatible-furnace'); return end; local amount=math.min(" + str(int(quantity)) + ",available); "
            "local removed=p.remove_item{name=item,count=amount}; local inserted=target.insert{name=item,count=removed}; if inserted<removed then p.insert{name=item,count=removed-inserted} end; "
            "if inserted<1 then rcon.print(marker..'|ok=0|error=insert-failed'); return end; rcon.print(marker..'|ok=1|item='..item..'|inserted='..inserted); end; "
            "local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_collect_furnace(self, *, item_name: str, quantity: int = 1) -> Dict[str, Any]:
        marker = "__FLE_RL_FURNACE_COLLECT_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._agent_lua()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local item=" + json.dumps(item_name) + "; "
            "if not p.can_insert({name=item,count=1}) then rcon.print(marker..'|ok=0|error=inventory-full'); return end; "
            "local target=nil; for _,e in ipairs(p.surface.find_entities_filtered{name='stone-furnace',force=p.force}) do if e.get_item_count(item)>0 then target=e; break end end; "
            "if not target then rcon.print(marker..'|ok=1|collected=0|stale=1|reason=no-output'); return end; local removed=target.remove_item{name=item,count=" + str(int(quantity)) + "}; local inserted=p.insert{name=item,count=removed}; "
            "if inserted<removed then target.insert{name=item,count=removed-inserted} end; if inserted<1 then rcon.print(marker..'|ok=0|error=collect-failed'); return end; rcon.print(marker..'|ok=1|item='..item..'|collected='..inserted); end; "
            "local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_transfer_chest_to_furnace(self, *, item_name: str, quantity: int = 5) -> Dict[str, Any]:
        marker = "__FLE_RL_CHEST_TRANSFER_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._agent_lua()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local item=" + json.dumps(item_name) + "; "
            "local chest=nil; for _,e in ipairs(p.surface.find_entities_filtered{name='wooden-chest',force=p.force}) do if e.get_item_count(item)>0 then chest=e; break end end; if not chest then rcon.print(marker..'|ok=0|error=no-chest-item'); return end; "
            "local furnace=nil; for _,e in ipairs(p.surface.find_entities_filtered{name='stone-furnace',force=p.force}) do local ok,c=pcall(function() return e.can_insert({name=item,count=1}) end); if ok and c then furnace=e; break end end; if not furnace then rcon.print(marker..'|ok=0|error=no-compatible-furnace'); return end; "
            "local removed=chest.remove_item{name=item,count=" + str(int(quantity)) + "}; local inserted=furnace.insert{name=item,count=removed}; if inserted<removed then chest.insert{name=item,count=removed-inserted} end; "
            "if inserted<1 then rcon.print(marker..'|ok=0|error=transfer-failed'); return end; rcon.print(marker..'|ok=1|item='..item..'|transferred='..inserted); end; "
            "local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_place_output_chest(self, *, drill_name: str) -> Dict[str, Any]:
        marker = "__FLE_RL_OUTPUT_CHEST_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._agent_lua()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; if p.get_item_count('wooden-chest')<1 then rcon.print(marker..'|ok=0|error=missing-chest'); return end; "
            "local s=p.surface; local target=nil; for _,e in ipairs(s.find_entities_filtered{name=" + json.dumps(drill_name) + ",force=p.force}) do local dt=e.drop_target; if not (dt and dt.valid and dt.name=='wooden-chest') and s.can_place_entity{name='wooden-chest',position=e.drop_position,force=p.force} then target=e; break end end; "
            "if not target then rcon.print(marker..'|ok=0|error=no-placeable-output'); return end; local chest=s.create_entity{name='wooden-chest',position=target.drop_position,force=p.force,raise_built=true}; "
            "if not chest then rcon.print(marker..'|ok=0|error=create-failed'); return end; if p.remove_item{name='wooden-chest',count=1}~=1 then chest.destroy(); rcon.print(marker..'|ok=0|error=inventory-remove-failed'); return end; "
            "rcon.print(marker..'|ok=1|x='..chest.position.x..'|y='..chest.position.y..'|drill='..target.name); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_connect_electric_drills_power(self) -> Dict[str, Any]:
        """Create verified pole routes to all currently unpowered electric drills."""
        marker = "__FLE_RL_ELECTRIC_DRILL_POWER_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; "
            "local function run() "
            "local p=nil; if storage and storage.agent_characters and storage.agent_characters[1] and storage.agent_characters[1].valid then p=storage.agent_characters[1] elseif player and player.valid then p=player elseif game.get_player(1) then p=game.get_player(1) end; "
            "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            "local s=p.surface; local f=p.force; local engines=s.find_entities_filtered{name='steam-engine',force=f}; local engine=engines[1]; "
            "if not engine then rcon.print(marker..'|ok=0|error=missing-steam-engine'); return end; "
            "local created={}; local connected_count=0; "
            "local function d2(a,b) local dx=a.position.x-b.position.x; local dy=a.position.y-b.position.y; return dx*dx+dy*dy end; "
            "local function nearest_pole(a,maxd2) local best=nil; local bd=maxd2; for _,e in ipairs(s.find_entities_filtered{name='small-electric-pole',force=f}) do local v=d2(a,e); if v<bd then best=e; bd=v end end; return best end; "
            "local function create_pole(pos) if p.get_item_count('small-electric-pole')<1 then return nil end; if not s.can_place_entity{name='small-electric-pole',position=pos,force=f} then return nil end; local e=s.create_entity{name='small-electric-pole',position=pos,force=f,raise_built=true,auto_connect=true}; if not e then return nil end; if p.remove_item{name='small-electric-pole',count=1}~=1 then e.destroy(); return nil end; created[#created+1]=e; return e end; "
            "local function build_near(a) local bb=a.bounding_box; local cx=(bb.left_top.x+bb.right_bottom.x)/2; local cy=(bb.left_top.y+bb.right_bottom.y)/2; for r=1,5 do local c={{x=bb.left_top.x-r,y=cy},{x=bb.right_bottom.x+r,y=cy},{x=cx,y=bb.left_top.y-r},{x=cx,y=bb.right_bottom.y+r},{x=bb.left_top.x-r,y=bb.left_top.y-r},{x=bb.right_bottom.x+r,y=bb.left_top.y-r},{x=bb.left_top.x-r,y=bb.right_bottom.y+r},{x=bb.right_bottom.x+r,y=bb.right_bottom.y+r}}; for _,pos in ipairs(c) do local e=create_pole(pos); if e then return e end end end; return nil end; "
            "local source=nearest_pole(engine,36) or build_near(engine); if not source then rcon.print(marker..'|ok=0|error=cannot-place-source-pole'); return end; "
            "for _,target in ipairs(s.find_entities_filtered{name='electric-mining-drill',force=f}) do local ok0,c0=pcall(function() return target.is_connected_to_electric_network() end); if not (ok0 and c0) then "
            "local receiver=nearest_pole(target,36) or build_near(target); if not receiver then rcon.print(marker..'|ok=0|error=cannot-place-receiver-pole'); return end; "
            "local sx=source.position.x; local sy=source.position.y; local dx=receiver.position.x-sx; local dy=receiver.position.y-sy; local dist=math.sqrt(dx*dx+dy*dy); local segments=math.max(1,math.ceil(dist/6)); "
            "for i=1,segments-1 do local tx=sx+dx*i/segments; local ty=sy+dy*i/segments; local built=nil; local offsets={{0,0},{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}}; for _,o in ipairs(offsets) do built=create_pole({x=tx+o[1],y=ty+o[2]}); if built then break end end; if not built then rcon.print(marker..'|ok=0|error=cannot-place-intermediate-pole'); return end end; "
            "local ok1,c1=pcall(function() return target.is_connected_to_electric_network() end); if ok1 and c1 then connected_count=connected_count+1 else rcon.print(marker..'|ok=0|error=drill-still-unpowered'); return end end end; "
            "rcon.print(marker..'|ok=1|connected='..connected_count..'|created='..#created..'|remaining='..p.get_item_count('small-electric-pole')); end; "
            "local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _observation(self) -> np.ndarray:
        base = super()._observation()
        f = self.feasibility_state
        current = str(self.infrastructure_state.get("current_research", "") or "")
        extras: List[float] = [
            float(bool(f.get("inventory_full", False))),
            min(10.0, math.log1p(max(0.0, safe_float(f.get("furnace_iron_plate"))))),
            min(10.0, math.log1p(max(0.0, safe_float(f.get("furnace_copper_plate"))))),
            min(10.0, math.log1p(max(0.0, safe_float(f.get("chest_iron_ore"))))),
            min(10.0, math.log1p(max(0.0, safe_float(f.get("chest_copper_ore"))))),
            min(10.0, safe_float(f.get("lab_working_count"))),
            float(bool(f.get("active_power", False))),
        ]
        extras.extend(float(current == target) for target in RESEARCH_TARGETS)
        extras.extend(
            [
                min(10.0, math.log1p(max(0.0, self.machine_output_total)) / 5.0),
                min(10.0, math.log1p(max(0.0, self.manual_crafted_total)) / 5.0),
                min(10.0, math.log1p(max(0.0, self.manual_harvested_total)) / 5.0),
                (
                    float(self.episode_step) / self.training_horizon_steps
                    if self.training_horizon_steps > 0
                    else 0.0
                ),
            ]
        )
        result = np.concatenate([base, np.asarray(extras, dtype=np.float32)])
        if result.shape != self.observation_space.shape:
            raise RuntimeError(f"runtime observation mismatch: {result.shape} != {self.observation_space.shape}")
        return result.astype(np.float32)

    def _base_info(self) -> Dict[str, Any]:
        info = super()._base_info()
        info.update(
            {
                "feasibility_state": json_safe(self.feasibility_state),
                "machine_output_total": float(self.machine_output_total),
                "machine_output_by_item": dict(self.machine_output_by_item),
                "manual_crafted_total": float(self.manual_crafted_total),
                "manual_crafted_by_item": dict(self.manual_crafted_by_item),
                "manual_harvested_total": float(self.manual_harvested_total),
                "manual_harvested_by_item": dict(self.manual_harvested_by_item),
                "lifetime_machine_output_total": float(self.lifetime_machine_output_total),
                "lifetime_manual_crafted_total": float(self.lifetime_manual_crafted_total),
                "lifetime_manual_harvested_total": float(self.lifetime_manual_harvested_total),
                "training_horizon_steps": int(self.training_horizon_steps),
                "completed_horizons": int(self.completed_horizons),
            }
        )
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if getattr(self, "episode_step", 0) > 0:
            self.completed_horizons += 1
        observation, info = super().reset(seed=seed, options=options)
        hard = bool(dict(options or {}).get("hard_reset", self.hard_reset))
        if hard:
            self.machine_output_total = 0.0
            self.machine_output_by_item.clear()
            self.manual_crafted_total = 0.0
            self.manual_crafted_by_item.clear()
            self.manual_harvested_total = 0.0
            self.manual_harvested_by_item.clear()
        self._collect_feasibility_state(force=True)
        observation = self._observation()
        info = self._base_info()
        info.update({"action_mask": self.action_masks(), "hard_reset": hard})
        return observation, info

    def step(self, action: int):
        before_stats = {
            bucket: dict(self.last_production_stats.get(bucket, {}))
            for bucket in ("output", "crafted", "harvested")
        }
        observation, reward, terminated, _truncated, info = super().step(action)
        del observation

        action_name = str(info.get("action_name", ""))
        runtime_dispatch = {
            "fuel_drill": lambda: self._runtime_fuel(entity_name="burner-mining-drill", quantity=5),
            "fuel_furnace": lambda: self._runtime_fuel(entity_name="stone-furnace", quantity=5),
            "fuel_boiler": lambda: self._runtime_fuel(entity_name="boiler", quantity=10),
            "insert_hand_mined_ore": lambda: self._runtime_insert_furnace(item_name="iron-ore", quantity=1),
            "insert_copper_ore": lambda: self._runtime_insert_furnace(item_name="copper-ore", quantity=1),
            "collect_iron_plate": lambda: self._runtime_collect_furnace(item_name="iron-plate", quantity=1),
            "collect_copper_plate": lambda: self._runtime_collect_furnace(item_name="copper-plate", quantity=1),
            "transfer_chest_ore_to_furnace": lambda: self._runtime_transfer_chest_to_furnace(item_name="iron-ore", quantity=5),
            "transfer_copper_chest_ore_to_furnace": lambda: self._runtime_transfer_chest_to_furnace(item_name="copper-ore", quantity=5),
            "place_chest_at_burner_output": lambda: self._runtime_place_output_chest(drill_name="burner-mining-drill"),
            "place_chest_at_electric_output": lambda: self._runtime_place_output_chest(drill_name="electric-mining-drill"),
            "connect_electric_drill_to_power": self._runtime_connect_electric_drills_power,
        }
        runtime_call = runtime_dispatch.get(action_name)
        if runtime_call is not None and not info.get("action_error"):
            try:
                runtime_result = runtime_call()
                info["runtime_action_inventory"] = runtime_result
                if str(runtime_result.get("stale", "0")) == "1":
                    # A live Factorio state can advance between the mask sample and
                    # execution.  A stale collection target is a benign no-op, not
                    # an execution fault.  Keep it observable for diagnostics.
                    info["runtime_stale_noop"] = True
                    info["runtime_stale_reason"] = str(runtime_result.get("reason", ""))
                if action_name == "fuel_drill":
                    self.flags["drill_fuelled"] = True
                elif action_name == "fuel_furnace":
                    self.flags["furnace_fuelled"] = True
                elif action_name == "connect_electric_drill_to_power":
                    self.flags["electric_drill_connected"] = True
                self._collect_compact_state(include_production_stats=False)
                self._collect_infrastructure_state(force=True)
            except Exception as exc:
                info["action_error"] = f"{type(exc).__name__}: {exc}"
                self.total_action_errors += 1
                reward -= self.execution_error_penalty
                self.episode_return -= self.execution_error_penalty

        self._collect_feasibility_state(force=True)
        self.lifetime_action_counts[action_name] += 1
        if (
            not info.get("action_error")
            and not info.get("invalid_action")
            and not info.get("runtime_stale_noop")
        ):
            self.lifetime_successful_action_counts[action_name] += 1

        after_stats = {
            bucket: dict(self.last_production_stats.get(bucket, {}))
            for bucket in ("output", "crafted", "harvested")
        }
        output_delta = self._bucket_delta(before_stats["output"], after_stats["output"])
        crafted_delta = self._bucket_delta(before_stats["crafted"], after_stats["crafted"])
        harvested_delta = self._bucket_delta(before_stats["harvested"], after_stats["harvested"])
        machine_delta: Dict[str, float] = {}
        for item, amount in output_delta.items():
            residual = max(0.0, amount - crafted_delta.get(item, 0.0) - harvested_delta.get(item, 0.0))
            if residual > 0:
                machine_delta[item] = residual

        machine_weighted = self._weighted(machine_delta)
        crafted_weighted = self._weighted(crafted_delta)
        harvested_weighted = self._weighted(harvested_delta)
        self.machine_output_total += machine_weighted
        self.manual_crafted_total += crafted_weighted
        self.manual_harvested_total += harvested_weighted
        self.lifetime_machine_output_total += machine_weighted
        self.lifetime_manual_crafted_total += crafted_weighted
        self.lifetime_manual_harvested_total += harvested_weighted
        self.machine_output_by_item.update(machine_delta)
        self.manual_crafted_by_item.update(crafted_delta)
        self.manual_harvested_by_item.update(harvested_delta)
        self.lifetime_machine_output_by_item.update(machine_delta)
        self.lifetime_manual_crafted_by_item.update(crafted_delta)
        self.lifetime_manual_harvested_by_item.update(harvested_delta)

        # Replace the old inclusive output reward with machine-only reward.
        old_output_reward = safe_float(dict(info.get("reward_parts", {})).get("automated_output"))
        reward = float(reward - old_output_reward + self.automated_output_reward_scale * machine_weighted)
        self.episode_return += self.automated_output_reward_scale * machine_weighted - old_output_reward
        self.automated_output_total = self.machine_output_total

        if action_name in {"noop", "wait", "wait_research"} and machine_weighted <= 0:
            progress = safe_float(dict(info.get("reward_parts", {})).get("research_progress"))
            if progress <= 0:
                reward -= self.idle_action_penalty
                self.episode_return -= self.idle_action_penalty

        self.flags["power_online"] = bool(self.feasibility_state.get("active_power", False))
        self.infrastructure_state["network_present"] = bool(
            safe_float(self.infrastructure_state.get("steam_connected_count")) > 0
            or safe_float(self.infrastructure_state.get("lab_connected_count")) > 0
        )
        self.infrastructure_state["active_power"] = bool(self.feasibility_state.get("active_power", False))
        self.infrastructure_state["power_online"] = bool(self.feasibility_state.get("active_power", False))

        truncated = bool(
            self.training_horizon_steps > 0
            and self.episode_step >= self.training_horizon_steps
        )
        final_info = dict(info)
        final_info.update(self._base_info())
        final_info.update(
            {
                "reward": float(reward),
                "machine_output_delta": machine_delta,
                "manual_crafted_delta": crafted_delta,
                "manual_harvested_delta": harvested_delta,
                "automated_output_delta": machine_delta,
                "action_mask": self.action_masks(),
                "time_limit_truncated": truncated,
                "terminal_goal": None,
            }
        )
        parts = dict(final_info.get("reward_parts", {}))
        parts["automated_output"] = float(self.automated_output_reward_scale * machine_weighted)
        parts["idle_penalty"] = float(-self.idle_action_penalty if action_name in {"noop", "wait", "wait_research"} and machine_weighted <= 0 else 0.0)
        parts["total"] = float(reward)
        final_info["reward_parts"] = parts
        return self._observation(), reward, bool(terminated), truncated, final_info


# Keep a conventional import name for the improved trainer.
FactorioOpenEndedEnvImproved = FactorioOpenEndedEnvObservation


__all__ = [
    "EXTRA_ACTIONS",
    "FEASIBILITY_MARKER",
    "FactorioOpenEndedEnvImproved",
    "FactorioOpenEndedEnvObservation",
    "IMPROVED_ACTIONS",
    "IMPROVED_DEFAULT_INVENTORY",
]
