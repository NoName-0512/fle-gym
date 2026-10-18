from __future__ import annotations

"""runtime research-oriented, time-normalised shared Factorio RL environment.

runtime keeps the exact runtime action and observation dimensions so MaskablePPO,
masked categorical SAC, and MAP-Elites can still be compared on one task.
The changes are semantic:

* small crafts become available as soon as at least one unit is craftable;
* hand-crafting macros still request a batch but start the currently craftable
  amount, allowing the lab dependency chain to grow incrementally;
* machine-production reward is normalised by simulated Factorio time;
* the five-second ``wait_crafting`` loophole is removed;
* verified laboratory milestones and research rewards are stronger;
* zero-pole power routes never call ``remove_item`` with count zero;
* authoritative live entity counts are exposed consistently.
"""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from factorio_openended_env_recipe import PROGRESSION_MILESTONE_REWARDS
from factorio_openended_env_research_yqeibtv import (
    CRAFT_BY_ACTION,
    CraftSpec,
    Runtime_ACTIONS,
    Runtime_CRAFT_SPECS,
    Runtime_LIVE_ENTITY_NAMES,
    Runtime_PROGRESSION_REWARDS,
    Runtime_RESEARCH_TARGETS,
    Runtime_TRACKED_ITEMS,
    FactorioOpenEndedEnvYqeibtv,
)
from factorio_transition_env import ActionSpec, json_safe, safe_float, spaces
from factorio_openended_env_research_ajweuzp import MANUAL_CRAFT_OUTPUTS


Runtime_VERSION = "18.0"
Runtime_TICK_MARKER = "__FLE_RL_TICK_runtime__"


# Research shaping is intentionally dense only along verified progression.
# Every milestone is awarded at most once per factory horizon.
Runtime_PROGRESSION_REWARDS: Dict[str, float] = {
    "electronics_triggered": 3.0,
    "first_gears_available": 2.0,
    "first_circuits_available": 2.0,
    "first_belts_available": 2.0,
    "lab_gears_ready": 2.0,
    "lab_circuits_ready": 2.0,
    "lab_belts_ready": 2.0,
    "lab_craftable": 5.0,
    "lab_all_ingredients_ready": 5.0,
    "lab_obtained": 8.0,
    "lab_placed": 4.0,
    "lab_powered": 6.0,
    "automation_science_unlocked": 4.0,
    "automation_science_available": 6.0,
    "science_inserted": 3.0,
    "research_started": 5.0,
    "research_progress_started": 5.0,
    "lab_working": 10.0,
    "automation_researched": 18.0,
    "logistics_researched": 14.0,
    "steel_processing_researched": 14.0,
    "logistic_science_unlocked": 8.0,
    "logistic_science_available": 8.0,
    "green_science_research_started": 6.0,
    "automation_2_researched": 16.0,
    "logistics_2_researched": 16.0,
    "engine_researched": 14.0,
    "oil_processing_researched": 20.0,
}

# The inherited runtime step looks these rewards up from the shared mappings.
# Updating both mappings keeps the one authoritative milestone ledger.
PROGRESSION_MILESTONE_REWARDS.update(Runtime_PROGRESSION_REWARDS)
Runtime_PROGRESSION_REWARDS.update(Runtime_PROGRESSION_REWARDS)


Runtime_LEGACY_CRAFT_SPECS: Mapping[str, CraftSpec] = {
    "craft_electronic_circuits": CraftSpec(
        "craft_electronic_circuits", "electronic-circuit", 10,
        "electronic-circuit", 1.0, "Craft up to ten electronic circuits."
    ),
    "craft_transport_belts": CraftSpec(
        "craft_transport_belts", "transport-belt", 5,
        "transport-belt", 2.0, "Craft up to ten transport belts."
    ),
    "craft_inserters": CraftSpec(
        "craft_inserters", "inserter", 4,
        "inserter", 1.0, "Craft up to four inserters."
    ),
    "craft_lab": CraftSpec(
        "craft_lab", "lab", 1, "lab", 1.0, "Craft one laboratory."
    ),
    "craft_automation_science": CraftSpec(
        "craft_automation_science", "automation-science-pack", 10,
        "automation-science-pack", 1.0,
        "Craft up to ten automation science packs."
    ),
    "craft_iron_gears": CraftSpec(
        "craft_iron_gears", "iron-gear-wheel", 10,
        "iron-gear-wheel", 1.0, "Craft up to ten iron gear wheels."
    ),
    "craft_assembling_machine": CraftSpec(
        "craft_assembling_machine", "assembling-machine-1", 1,
        "assembling-machine-1", 1.0, "Craft one assembling machine 1."
    ),
}

# runtime dispatches craft runtime helpers from this shared mapping. Extending it
# lets runtime redirect legacy craft actions to the same partial-batch-safe path.
CRAFT_BY_ACTION.update(Runtime_LEGACY_CRAFT_SPECS)
# Engine units use the ``crafting-with-fluid`` recipe category in Factorio 2
# and cannot be queued by the headless character.  runtime retains this fixed
# policy slot as a machine-output collection action instead.
CRAFT_BY_ACTION.pop("craft_engine_units", None)


def _build_actions_machine() -> Tuple[ActionSpec, ...]:
    actions = []
    for spec in Runtime_ACTIONS:
        if spec.name == "craft_engine_units":
            spec = replace(
                spec,
                name="collect_engine_units",
                category="transfer",
                code='print("__FLE_RL_EVENT__:collect_engine_units_requested")',
                required_items={},
                required_entities={},
                description=(
                    "Collect completed engine units from factory output "
                    "chests into the agent inventory."
                ),
            )
        elif spec.name in Runtime_LEGACY_CRAFT_SPECS:
            spec = replace(
                spec,
                code=f'print("__FLE_RL_EVENT__:runtime_{spec.name}_requested")',
                required_items={},
                description=Runtime_LEGACY_CRAFT_SPECS[spec.name].description,
            )
        elif spec.name == "wait_crafting":
            spec = replace(
                spec,
                code="sleep(1)",
                description=(
                    "Advance one simulated second while queued hand crafts and "
                    "machines progress. runtime removes the former five-second advantage."
                ),
            )
        actions.append(spec)
    return tuple(actions)


Runtime_ACTIONS: Tuple[ActionSpec, ...] = _build_actions_machine()


LEGACY_INCREMENTAL_CRAFTS: Mapping[str, str] = {
    "craft_electronic_circuits": "electronic-circuit",
    "craft_transport_belts": "transport-belt",
    "craft_inserters": "inserter",
    "craft_lab": "lab",
    "craft_automation_science": "automation-science-pack",
    "craft_iron_gears": "iron-gear-wheel",
    "craft_assembling_machine": "assembling-machine-1",
}


class FactorioOpenEndedEnvMachine(FactorioOpenEndedEnvYqeibtv):
    """runtime shared environment used unchanged by PPO, SAC, and MAP-Elites."""

    def __init__(
        self,
        *,
        normalize_machine_reward_by_sim_time: bool = True,
        machine_reward_reference_seconds: float = 1.0,
        automated_output_reward_scale: float = 0.50,
        research_progress_reward: float = 40.0,
        trigger_technology_reward: float = 3.0,
        lab_research_completion_reward: float = 60.0,
        research_activity_reward: float = 0.10,
        **kwargs: Any,
    ) -> None:
        # Set runtime fields before parent initialisation because runtime exports the
        # schema during its constructor and Python dispatches to this override.
        self.normalize_machine_reward_by_sim_time = bool(
            normalize_machine_reward_by_sim_time
        )
        self.machine_reward_reference_seconds = max(
            1e-6, float(machine_reward_reference_seconds)
        )
        self.trigger_technology_reward = float(trigger_technology_reward)
        self.lab_research_completion_reward = float(
            lab_research_completion_reward
        )
        self._last_game_tick_runtime = None
        self.lifetime_simulated_seconds = 0.0
        self.horizon_simulated_seconds = 0.0
        self.lifetime_raw_machine_reward = 0.0
        self.horizon_raw_machine_reward = 0.0
        self.lifetime_normalized_machine_reward = 0.0
        self.horizon_normalized_machine_reward = 0.0

        kwargs.setdefault("automated_output_reward_scale", automated_output_reward_scale)
        kwargs.setdefault("research_progress_reward", research_progress_reward)
        kwargs.setdefault("research_activity_reward", research_activity_reward)
        super().__init__(
            trigger_technology_reward=trigger_technology_reward,
            lab_research_completion_reward=lab_research_completion_reward,
            **kwargs,
        )
        self.actions = Runtime_ACTIONS
        self.action_space = spaces.Discrete(len(self.actions))
        self._write_config_machine()

    # ------------------------------------------------------------------
    # Configuration and authoritative state
    # ------------------------------------------------------------------
    def _write_config_machine(self) -> None:
        payload = {
            "version": Runtime_VERSION,
            "action_count": int(self.action_space.n),
            "observation_shape": list(self.observation_space.shape),
            "normalize_machine_reward_by_sim_time": self.normalize_machine_reward_by_sim_time,
            "machine_reward_reference_seconds": self.machine_reward_reference_seconds,
            "automated_output_reward_scale": float(getattr(self, "automated_output_reward_scale", 0.50)),
            "research_progress_reward": float(getattr(self, "research_progress_reward", 40.0)),
            "trigger_technology_reward": float(getattr(self, "trigger_technology_reward", 3.0)),
            "lab_research_completion_reward": float(getattr(self, "lab_research_completion_reward", 60.0)),
            "research_activity_reward": float(getattr(self, "research_activity_reward", 0.10)),
            "progression_rewards": Runtime_PROGRESSION_REWARDS,
        }
        path = Path(self.run_dir) / "environment_config_runtime.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        schema_path = Path(self.run_dir) / "environment_schema_runtime.json"
        schema_path.write_text(
            json.dumps(json_safe(self.schema()), indent=2),
            encoding="utf-8",
        )

    def schema(self) -> Dict[str, Any]:
        schema = dict(super().schema())
        schema["version"] = Runtime_VERSION
        schema["environment"] = type(self).__name__
        schema["reward_design_runtime"] = {
            "machine_production": "strict whitelist, verified machine source, normalised by simulated seconds",
            "research_progress_reward": float(getattr(self, "research_progress_reward", 40.0)),
            "trigger_technology_reward": float(getattr(self, "trigger_technology_reward", 3.0)),
            "lab_research_completion_reward": float(getattr(self, "lab_research_completion_reward", 60.0)),
            "research_activity_reward": float(getattr(self, "research_activity_reward", 0.10)),
            "progression_milestones": Runtime_PROGRESSION_REWARDS,
        }
        return schema

    def _sync_authoritative_entity_counts(self) -> None:
        state = getattr(self, "feasibility_state", {})
        for entity in Runtime_LIVE_ENTITY_NAMES:
            key = f"entity__{entity}"
            if key in state:
                self.entity_counts[entity] = safe_float(state[key])

    # ------------------------------------------------------------------
    # Incremental crafting and research reachability
    # ------------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        mask = super().get_action_mask().astype(np.int8)
        f = self.feasibility_state
        live = bool(f.get("runtime_query_ok", False))

        def set_valid(name: str, valid: bool) -> None:
            try:
                mask[self.action_index(name)] = int(bool(valid))
            except KeyError:
                pass

        def at_least_one(recipe: str) -> bool:
            return (
                live
                and safe_float(f.get(f"recipe_enabled__{recipe}")) > 0
                and safe_float(f.get(f"craftable__{recipe}")) >= 1
                and safe_float(f.get("crafting_queue_length")) < 80
            )

        for action_name, recipe in LEGACY_INCREMENTAL_CRAFTS.items():
            valid = at_least_one(recipe)
            if action_name == "craft_lab":
                valid = (
                    valid
                    and self.inventory.get("lab", 0.0) < 1
                    and self.entity_counts.get("lab", 0.0) < 2
                )
            set_valid(action_name, valid)

        # runtime craft macros already use a live one-item threshold. Re-assert it
        # here so every algorithm receives identical semantics.
        for spec in Runtime_CRAFT_SPECS:
            set_valid(spec.action, at_least_one(spec.recipe))

        set_valid(
            "wait_crafting",
            live and safe_float(f.get("crafting_queue_length")) > 0,
        )
        set_valid("noop", True)
        return mask

    def _current_progress_milestones(self) -> set[str]:
        milestones = set(super()._current_progress_milestones())
        f = self.feasibility_state
        infra = self.infrastructure_state

        if self.inventory.get("iron-gear-wheel", 0.0) > 0:
            milestones.add("first_gears_available")
        if self.inventory.get("electronic-circuit", 0.0) > 0:
            milestones.add("first_circuits_available")
        if self.inventory.get("transport-belt", 0.0) > 0:
            milestones.add("first_belts_available")
        if safe_float(f.get("craftable__lab")) >= 1:
            milestones.add("lab_craftable")
        if (
            self.entity_counts.get("lab", 0.0) > 0
            and safe_float(infra.get("lab_connected_count")) > 0
        ):
            milestones.add("lab_powered")
        if (
            safe_float(f.get("lab_red_science_count"))
            + safe_float(f.get("lab_green_science_count"))
            + safe_float(f.get("lab_military_science_count"))
        ) > 0:
            milestones.add("science_inserted")
        if str(infra.get("current_research", "") or ""):
            milestones.add("research_started")
        if safe_float(infra.get("research_progress")) > 0:
            milestones.add("research_progress_started")
        return milestones

    # ------------------------------------------------------------------
    # Simulated-time accounting
    # ------------------------------------------------------------------
    def _query_game_tick_machine(self) -> Optional[int]:
        command = (
            "/sc rcon.print(" + json.dumps(Runtime_TICK_MARKER + "|") + "..game.tick)"
        )
        try:
            raw = str(self.instance.rcon_client.send_command(command) or "")
            line = next(
                (line for line in raw.splitlines() if Runtime_TICK_MARKER in line),
                "",
            )
            if not line:
                return None
            tail = line.split(Runtime_TICK_MARKER, 1)[1].lstrip(": |")
            return int(float(tail.split("|", 1)[0]))
        except Exception:
            return None

    def _normalise_machine_reward(
        self, raw_machine_reward: float, elapsed_seconds: float
    ) -> float:
        raw = max(0.0, float(raw_machine_reward))
        if not self.normalize_machine_reward_by_sim_time or raw <= 0:
            return raw
        denominator = max(1.0, float(elapsed_seconds))
        return raw * self.machine_reward_reference_seconds / denominator

    # ------------------------------------------------------------------
    # Zero-count-safe power routing
    # ------------------------------------------------------------------
    def _runtime_connect_power_yqeibtv(
        self, target_names: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        marker = "__FLE_RL_RUNTIME_runtime___POWER"
        names = tuple(
            target_names
            or (
                "electric-mining-drill",
                "inserter",
                "long-handed-inserter",
                "fast-inserter",
                "assembling-machine-1",
                "assembling-machine-2",
                "lab",
                "pumpjack",
                "oil-refinery",
                "chemical-plant",
            )
        )
        target_lua = "{" + ",".join(json.dumps(n) for n in names) + "}"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            + "local s=p.surface; local f=p.force; local engines=s.find_entities_filtered{name='steam-engine',force=f}; local engine=engines[1]; if not engine then rcon.print(marker..'|ok=0|error=missing-steam-engine'); return end; "
            + "local target=nil; for _,n in ipairs(" + target_lua + ") do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not(ok and c) then target=e; break end end; if target then break end end; "
            + "if not target then rcon.print(marker..'|ok=1|already_connected=1|created=0|connected=1'); return end; "
            + "local poles=s.find_entities_filtered{name={'small-electric-pole','medium-electric-pole'},force=f}; local function d2(a,b) local dx=a.x-b.x; local dy=a.y-b.y; return dx*dx+dy*dy end; "
            + "local pole_name=(p.get_item_count('medium-electric-pole')>0 and 'medium-electric-pole') or 'small-electric-pole'; local spacing=(pole_name=='medium-electric-pole') and 9 or 6; local plan={}; "
            + "local function pick(center,radius) local pos=s.find_non_colliding_position(pole_name,center,radius,0.5,true); if pos and s.can_place_entity{name=pole_name,position=pos,force=f} then return pos end end; "
            + "local source=nil; for _,q in ipairs(poles) do if d2(q.position,engine.position)<=36 then source=q.position; break end end; if not source then source=pick(engine.position,7); if not source then rcon.print(marker..'|ok=0|error=cannot-place-source-pole'); return end; plan[#plan+1]=source end; "
            + "local receiver=nil; for _,q in ipairs(poles) do if d2(q.position,target.position)<=25 then receiver=q.position; break end end; if not receiver then receiver=pick(target.position,6); if not receiver then rcon.print(marker..'|ok=0|error=cannot-place-receiver-pole'); return end; plan[#plan+1]=receiver end; "
            + "local dx=receiver.x-source.x; local dy=receiver.y-source.y; local dist=math.sqrt(dx*dx+dy*dy); local segments=math.max(1,math.ceil(dist/spacing)); for i=1,segments-1 do local desired={x=source.x+dx*i/segments,y=source.y+dy*i/segments}; local covered=false; for _,q in ipairs(poles) do if d2(q.position,desired)<=spacing*spacing then covered=true; break end end; if not covered then for _,q in ipairs(plan) do if d2(q,desired)<=spacing*spacing then covered=true; break end end end; if not covered then local pos=pick(desired,3); if not pos then rcon.print(marker..'|ok=0|error=cannot-route-poles'); return end; plan[#plan+1]=pos end end; "
            + "if #plan>0 and p.get_item_count(pole_name)<#plan then rcon.print(marker..'|ok=0|error=missing-poles|need='..#plan); return end; "
            + "local made={}; for _,pos in ipairs(plan) do local e=s.create_entity{name=pole_name,position=pos,force=f,raise_built=true,auto_connect=true}; if not e then for _,q in ipairs(made) do if q.valid then q.destroy() end end; rcon.print(marker..'|ok=0|error=create-pole'); return end; made[#made+1]=e end; "
            + "if #made>0 then local removed=p.remove_item{name=pole_name,count=#made}; if removed<#made then for _,q in ipairs(made) do if q.valid then q.destroy() end end; rcon.print(marker..'|ok=0|error=inventory-remove-poles'); return end end; "
            + "local ok,c=pcall(function() return target.is_connected_to_electric_network() end); rcon.print(marker..'|ok=1|target='..target.name..'|created='..#made..'|connected='..((ok and c) and 1 or 0)); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        observation, info = super().reset(seed=seed, options=options)
        hard = bool(dict(options or {}).get("hard_reset", self.hard_reset))
        if hard:
            self.horizon_simulated_seconds = 0.0
            self.horizon_raw_machine_reward = 0.0
            self.horizon_normalized_machine_reward = 0.0
        self._sync_authoritative_entity_counts()
        self._last_game_tick_runtime = self._query_game_tick_machine()
        final = dict(info)
        final.update(self._base_info())
        final["environment_version"] = Runtime_VERSION
        final["environment_schema_path"] = str(
            Path(self.run_dir) / "environment_schema_runtime.json"
        )
        final["action_mask"] = self.action_masks()
        return self._observation(), final

    def step(self, action: int):
        before_tick = self._last_game_tick_runtime
        action_index = int(action)
        action_name = (
            self.actions[action_index].name
            if 0 <= action_index < len(self.actions)
            else "invalid-index"
        )
        # runtime's legacy attribution assumes a full fixed batch. runtime executes the
        # same actions through a partial-batch runtime helper, so suppress that
        # old static estimate and let runtime record the actual `started` amount.
        legacy_manual_spec = None
        if action_name in Runtime_LEGACY_CRAFT_SPECS:
            legacy_manual_spec = MANUAL_CRAFT_OUTPUTS.pop(action_name, None)
        try:
            observation, reward, terminated, truncated, info = super().step(action)
        finally:
            if legacy_manual_spec is not None:
                MANUAL_CRAFT_OUTPUTS[action_name] = legacy_manual_spec
        info = dict(info)
        after_tick = self._query_game_tick_machine()
        self._last_game_tick_runtime = after_tick

        elapsed_ticks = 0
        if before_tick is not None and after_tick is not None:
            elapsed_ticks = max(0, int(after_tick - before_tick))
        elapsed_seconds = max(1.0, elapsed_ticks / 60.0) if elapsed_ticks else 1.0
        self.lifetime_simulated_seconds += elapsed_seconds
        self.horizon_simulated_seconds += elapsed_seconds

        parts = dict(info.get("reward_parts", {}))
        raw_machine_reward = safe_float(parts.get("machine_production"))
        normalized_machine_reward = self._normalise_machine_reward(
            raw_machine_reward, elapsed_seconds
        )
        reward_delta = normalized_machine_reward - raw_machine_reward

        self.lifetime_raw_machine_reward += raw_machine_reward
        self.horizon_raw_machine_reward += raw_machine_reward
        self.lifetime_normalized_machine_reward += normalized_machine_reward
        self.horizon_normalized_machine_reward += normalized_machine_reward

        if abs(reward_delta) > 1e-12:
            reward = float(reward + reward_delta)
            self.episode_return += reward_delta
            self.lifetime_canonical_reward_components["machine_production"] += reward_delta
            self.horizon_canonical_reward_components["machine_production"] += reward_delta
            self.lifetime_canonical_reward_components["total"] += reward_delta
            self.horizon_canonical_reward_components["total"] += reward_delta
            self.lifetime_reward_components["machine_production"] += reward_delta
            if "automated_output" in self.lifetime_reward_components:
                self.lifetime_reward_components["automated_output"] += reward_delta

        parts["machine_production"] = float(normalized_machine_reward)
        parts.pop("automated_output", None)
        parts.pop("legacy_base", None)
        parts.pop("total", None)
        parts.pop("residual", None)
        parts["residual"] = float(
            reward - sum(safe_float(value) for value in parts.values())
        )
        parts["total"] = float(reward)

        self._sync_authoritative_entity_counts()
        final = dict(info)
        final.update(self._base_info())
        final.update(
            {
                "environment_version": Runtime_VERSION,
                "reward": float(reward),
                "reward_parts": parts,
                "machine_production_reward_raw": float(raw_machine_reward),
                "machine_production_reward_normalized": float(
                    normalized_machine_reward
                ),
                "simulated_ticks_elapsed": int(elapsed_ticks),
                "simulated_seconds_elapsed": float(elapsed_seconds),
                "lifetime_simulated_seconds": float(self.lifetime_simulated_seconds),
                "horizon_simulated_seconds": float(self.horizon_simulated_seconds),
                "lifetime_raw_machine_reward": float(
                    self.lifetime_raw_machine_reward
                ),
                "lifetime_normalized_machine_reward": float(
                    self.lifetime_normalized_machine_reward
                ),
                "authoritative_entity_counts": dict(self.entity_counts),
                "action_mask": self.action_masks(),
            }
        )
        return self._observation(), float(reward), terminated, truncated, final

    def _base_info(self) -> Dict[str, Any]:
        self._sync_authoritative_entity_counts()
        info = super()._base_info()
        info.update(
            {
                "environment_version": Runtime_VERSION,
                "normalize_machine_reward_by_sim_time": bool(
                    self.normalize_machine_reward_by_sim_time
                ),
                "machine_reward_reference_seconds": float(
                    self.machine_reward_reference_seconds
                ),
                "lifetime_simulated_seconds": float(self.lifetime_simulated_seconds),
                "horizon_simulated_seconds": float(self.horizon_simulated_seconds),
                "lifetime_raw_machine_reward": float(
                    self.lifetime_raw_machine_reward
                ),
                "horizon_raw_machine_reward": float(
                    self.horizon_raw_machine_reward
                ),
                "lifetime_normalized_machine_reward": float(
                    self.lifetime_normalized_machine_reward
                ),
                "horizon_normalized_machine_reward": float(
                    self.horizon_normalized_machine_reward
                ),
                "authoritative_entity_counts": dict(self.entity_counts),
            }
        )
        return info


# A stable descriptive alias for algorithm-neutral code.
FactorioOpenEndedEnvResearchBalanced = FactorioOpenEndedEnvMachine


__all__ = [
    "FactorioOpenEndedEnvResearchBalanced",
    "FactorioOpenEndedEnvMachine",
    "LEGACY_INCREMENTAL_CRAFTS",
    "Runtime_LEGACY_CRAFT_SPECS",
    "Runtime_ACTIONS",
    "Runtime_PROGRESSION_REWARDS",
    "Runtime_VERSION",
    "Runtime_RESEARCH_TARGETS",
    "Runtime_TRACKED_ITEMS",
]
