from __future__ import annotations

"""runtime environment with deterministic research validation and safe power masks.

runtime keeps the runtime fixed 124-action/266-observation contract so the exact same
MDP interface can be used by MaskablePPO, categorical/discretised SAC, and
MAP-Elites.  It changes environment semantics rather than hiding actions:

* a dense, inventory-based laboratory ingredient potential rewards balanced
  accumulation of gears, circuits, and belts and penalises consuming them;
* partial-batch hand crafting is accounted from the runtime-confirmed quantity;
* smelting-line macros may reclaim previously placed drills, furnaces, and
  chests instead of becoming permanently unreachable;
* all power actions use route-aware masks and one transactional pole-chain
  dispatch per selected action;
* power/line failures receive a short action cooldown to prevent repeated
  environment-side rejections;
* every info payload and final summary can use one authoritative live entity
  dictionary.

There is still no product or research terminal goal.  Training horizons are
Gymnasium time-limit truncations only; set ``training_horizon_steps=0`` for a
single persistent evaluation factory.
"""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from factorio_openended_env_machine import (
    FactorioOpenEndedEnvMachine,
    Runtime_ACTIONS,
    Runtime_PROGRESSION_REWARDS,
    Runtime_RESEARCH_TARGETS,
    Runtime_TRACKED_ITEMS,
)
from factorio_openended_env_craft import FactorioOpenEndedEnvCraft
from factorio_openended_env_research_yqeibtv import CRAFT_BY_ACTION, Runtime_LIVE_ENTITY_NAMES
from factorio_transition_env import ActionSpec, json_safe, safe_float, spaces


Runtime_VERSION = "101.0"
Runtime_STATE_MARKER = "__FLE_RL_STATE_runtime__"
Runtime_RECLAIM_MARKER = "__FLE_RL_RECLAIM_runtime__"
Runtime_POWER_MARKER = "__FLE_RL_POWER_runtime__"

# The catalogue must remain exactly the shared 124 actions used by Maskable
# PPO, masked discrete SAC and MAP-Elites.  Scenario setup is reset-time
# infrastructure, never an extra action that changes the policy output size.
def _build_actions_research() -> Tuple[ActionSpec, ...]:
    """Keep the fixed slots while redirecting incompatible FLE helpers.

    FLE 0.4.3's ``get_entity`` path cannot consume the list-like position
    returned for laboratories in this runtime.  The policy action is therefore
    a marker whose real, transactional transfer is owned by the runtime runtime
    dispatcher below.
    """
    actions = []
    for spec in Runtime_ACTIONS:
        if spec.name == "insert_automation_science":
            spec = replace(
                spec,
                code='print("__FLE_RL_EVENT__:automation_science_insert_requested")',
                description=(
                    "Transactionally move ten automation science packs from "
                    "the agent inventory into the nearest laboratory."
                ),
            )
        elif spec.name in {
            "harvest_coal",
            "harvest_stone",
            "harvest_iron_ore",
            "harvest_copper_ore",
        }:
            event = {
                "harvest_coal": "coal_harvested",
                "harvest_stone": "stone_harvested",
                "harvest_iron_ore": "iron_harvested",
                "harvest_copper_ore": "copper_harvested",
            }[spec.name]
            spec = replace(
                spec,
                code=(
                    f'print("__FLE_RL_EVENT__:{event}")\n'
                    f'print("__FLE_RL_EVENT__:runtime_{spec.name}_requested")'
                ),
                description=(
                    "Transactionally mine ten units from a live, reachable, "
                    "unoccupied resource tile."
                ),
            )
        actions.append(spec)
    return tuple(actions)


Runtime_ACTIONS = _build_actions_research()

Runtime_HARVEST_RESOURCES: Mapping[str, str] = {
    "harvest_coal": "coal",
    "harvest_stone": "stone",
    "harvest_iron_ore": "iron-ore",
    "harvest_copper_ore": "copper-ore",
}


class FactorioOpenEndedEnvResearch(FactorioOpenEndedEnvCraft):
    """runtime full-fidelity semantics on the fixed runtime-compatible interface."""

    def __init__(
        self,
        *,
        lab_ingredient_potential_reward: float = 21.0,
        environment_rejection_cooldown_steps: int = 24,
        line_preflight_every: int = 4,
        # This action is part of the fixed 124-slot policy interface.  It may
        # be temporarily masked when no powered drill exists, but it must not
        # be permanently disabled: doing so creates a dead PPO output index.
        disable_redundant_drill_power_action: bool = False,
        research_progress_reward: float = 12.0,
        lab_research_completion_reward: float = 12.0,
        research_activity_reward: float = 0.02,
        automated_output_reward_scale: float = 0.35,
        deterministic_headless_research_setup: bool = True,
        max_unproductive_wait_streak: int = 1,
        repeated_idle_penalty: float = 0.08,
        research_progress_reward_cap: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.lab_ingredient_potential_reward = max(
            0.0, float(lab_ingredient_potential_reward)
        )
        self.environment_rejection_cooldown_steps = max(
            0, int(environment_rejection_cooldown_steps)
        )
        self.line_preflight_every = max(1, int(line_preflight_every))
        self.disable_redundant_drill_power_action = bool(
            disable_redundant_drill_power_action
        )
        self._runtime_action_cooldown_until: Counter[str] = Counter()
        self._runtime_last_line_query_step = -10**9
        self._last_lab_ingredient_fraction_runtime = 0.0
        self.lifetime_lab_ingredient_reward = 0.0
        self.horizon_lab_ingredient_reward = 0.0
        self.lifetime_runtime_manual_craft_units = 0.0
        self.horizon_runtime_manual_craft_units = 0.0
        # runtime depended on the periodic feasibility refresh for research
        # accounting.  That can miss a completion that happens between two
        # refreshes, so runtime keeps an independent, live-confirmed baseline.
        self._runtime_seen_lab_research: set[str] = set()
        self._runtime_research_progress: Dict[str, float] = {}
        # FLE's headless agent is a LuaEntity, not a LuaPlayer.  Factorio
        # player-event technology triggers can therefore never occur.  This
        # explicit scenario setting resolves only that engine-impossible gate;
        # all ordinary recipes, placement, production and lab research remain
        # governed by live Factorio state and the fixed action mask.
        self.deterministic_headless_research_setup = bool(
            deterministic_headless_research_setup
        )
        self.max_unproductive_wait_streak = max(
            1, int(max_unproductive_wait_streak)
        )
        self.repeated_idle_penalty = abs(float(repeated_idle_penalty))
        self.research_progress_reward_cap = max(
            0.0, float(research_progress_reward_cap)
        )
        self._runtime_unproductive_wait_streak = 0
        self.runtime_stale_race_noops: Counter[str] = Counter()
        self._runtime_output_chest_query_step = -10**9
        self._runtime_output_chest_feasibility: Dict[str, float] = {
            "burner-mining-drill": 0.0,
            "electric-mining-drill": 0.0,
        }
        # runtime lab-feed feasibility uses the exact live destination-capacity
        # predicate shared by the mask and the runtime stale-race repair.
        self._runtime_lab_feed_query_step = -10**9
        self._runtime_lab_feed_feasibility: Dict[str, float] = {
            "can_feed": 0.0,
            "source_science": 0.0,
            "accepting_slots": 0.0,
            "labs": 0.0,
        }
        # runtime refresh coalescing.  Older adapter layers each performed their
        # own full live feasibility scan after one policy decision.  Those
        # scans all happened after the same Factorio mutation and accounted
        # for most wrapper latency.  runtime records the global step of the most
        # recent full refresh so later accounting-only layers can reuse the
        # same authoritative state.  Any layer that performs an additional
        # runtime mutation still forces a new refresh before reuse is allowed.
        self._runtime_last_live_refresh_step = -10**9
        self._runtime_live_refresh_count = 0
        self._runtime_live_refresh_reuse_count = 0
        self._runtime_live_refresh_seconds_total = 0.0
        # runtime power stale-state suppression.  A fully rolled-back
        # target-not-connected transaction records the exact live power-state
        # signature that failed.  The action stays masked while that state is
        # unchanged, preventing the policy from spending hundreds of decisions
        # retrying an identical non-actionable route.  Any material factory,
        # inventory, or power-state change automatically re-enables the action.
        self._runtime_power_block_signature = None
        self._runtime_power_mask_suppressions = 0
        # Training and evaluation still share one semantic path: at least one
        # post-action authoritative refresh is required on every transition.

        kwargs.setdefault("research_progress_reward", research_progress_reward)
        kwargs.setdefault(
            "lab_research_completion_reward", lab_research_completion_reward
        )
        kwargs.setdefault("research_activity_reward", research_activity_reward)
        kwargs.setdefault(
            "automated_output_reward_scale", automated_output_reward_scale
        )
        self.science_readiness_reward = 12.0
        self._last_science_readiness_runtime = 0.0
        # runtime owns the action-specific mask switch.  Passing the setting is
        # essential: merely assigning the attribute above lets runtime overwrite
        # it with its historical ``True`` default during ``super().__init__``.
        super().__init__(
            disable_redundant_drill_power_action=(
                self.disable_redundant_drill_power_action
            ),
            **kwargs,
        )
        self.actions = Runtime_ACTIONS
        self.action_space = spaces.Discrete(len(self.actions))
        self._write_config_research()

    # ------------------------------------------------------------------
    # Configuration and schema
    # ------------------------------------------------------------------
    def _write_config_research(self) -> None:
        payload = {
            "version": Runtime_VERSION,
            "action_count": int(self.action_space.n),
            "research_bootstrap": "deterministic headless setup; no extra policy action",
            "observation_shape": list(self.observation_space.shape),
            "lab_ingredient_potential_reward": self.lab_ingredient_potential_reward,
            "environment_rejection_cooldown_steps": self.environment_rejection_cooldown_steps,
            "line_preflight_every": self.line_preflight_every,
            "disable_redundant_drill_power_action": self.disable_redundant_drill_power_action,
            "research_progress_reward": float(
                getattr(self, "research_progress_reward", 12.0)
            ),
            "lab_research_completion_reward": float(
                getattr(self, "lab_research_completion_reward", 12.0)
            ),
            "research_activity_reward": float(
                getattr(self, "research_activity_reward", 0.02)
            ),
            "automated_output_reward_scale": float(
                getattr(self, "automated_output_reward_scale", 0.35)
            ),
            "deterministic_headless_research_setup": self.deterministic_headless_research_setup,
            "max_unproductive_wait_streak": self.max_unproductive_wait_streak,
            "repeated_idle_penalty": self.repeated_idle_penalty,
            "research_progress_reward_cap": self.research_progress_reward_cap,
            "evaluation_full_fidelity": True,
        }
        run_dir = Path(self.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "environment_config_runtime.json").write_text(
            json.dumps(json_safe(payload), indent=2), encoding="utf-8"
        )
        (run_dir / "environment_schema_runtime.json").write_text(
            json.dumps(json_safe(self.schema()), indent=2), encoding="utf-8"
        )

    def schema(self) -> Dict[str, Any]:
        schema = dict(super().schema())
        schema["version"] = Runtime_VERSION
        schema["environment"] = type(self).__name__
        schema["runtime_changes"] = {
            "lab_ingredient_progress": (
                "balanced potential over 10 gears, 10 circuits, and 4 belts"
            ),
            "smelting_line": "reclaims eligible placed entities before transactional construction",
            "power": "route-aware transactional pole chains; exactly one dispatch per action",
            "manual_crafting": "runtime-confirmed partial-batch quantities",
            "terminal_goal": None,
            "runtime_harvesting": "runtime harvest and mask share the same reachable, unoccupied-tile predicate",
            "runtime_waiting": "unproductive repeated waits are masked and progressively penalised",
            "runtime_research_reward": "single-source completion accounting and simulated-time-normalised progress",
        }
        return schema

    # ------------------------------------------------------------------
    # runtime feasibility and research reachability
    # ------------------------------------------------------------------
    def _factory_power_route_feasible_research(self) -> bool:
        """Mirror the runtime planner prerequisites exactly.

        runtime only checked for one pole, although the route planner can require
        several.  A stale or incomplete route query therefore let an action
        enter the environment and be rejected.  runtime fails closed until the
        live runtime planner confirms a route and its complete pole budget.
        """
        f = self.feasibility_state
        pole_count = safe_float(self.inventory.get("small-electric-pole")) + safe_float(
            self.inventory.get("medium-electric-pole")
        )
        return (
            bool(f.get("runtime_query_ok", False))
            and safe_float(f.get("power_unpowered_consumers")) > 0
            and safe_float(f.get("power_generation_capacity")) > 0
            and safe_float(f.get("power_route_possible")) > 0
            and pole_count >= safe_float(f.get("power_route_poles_needed"))
        )

    def _factory_power_signature_research(self) -> tuple:
        """Compact live signature for one factory-power connection opportunity.

        This deliberately excludes ``global_step``: an unchanged failed power
        topology must remain blocked across later policy decisions.  It includes
        all state that can materially change whether the planner has a new
        opportunity, so normal building/inventory/power progress clears the
        suppression automatically.
        """
        f = self.feasibility_state
        target_keys = (
            "entity__electric-mining-drill",
            "entity__inserter",
            "entity__long-handed-inserter",
            "entity__fast-inserter",
            "entity__assembling-machine-1",
            "entity__assembling-machine-2",
            "entity__lab",
            "entity__pumpjack",
            "entity__oil-refinery",
            "entity__chemical-plant",
        )
        values = [
            safe_float(f.get("power_unpowered_consumers")),
            safe_float(f.get("power_generation_capacity")),
            safe_float(f.get("power_route_possible")),
            safe_float(f.get("power_route_poles_needed")),
            safe_float(self.inventory.get("small-electric-pole")),
            safe_float(self.inventory.get("medium-electric-pole")),
        ]
        values.extend(safe_float(f.get(key)) for key in target_keys)
        return tuple(round(float(value), 6) for value in values)

    def _factory_power_action_feasible_research(self) -> bool:
        """runtime: do not re-expose an unchanged rolled-back power route.

        runtime correctly converted the rare Factorio network-settlement race into
        a stale no-op, but PPO could then immediately sample the same action
        again hundreds of times.  Keep the action masked while the exact live
        power opportunity is unchanged; once the topology or resources change,
        clear the block and let the normal planner predicate decide again.
        """
        if not self._factory_power_route_feasible_research():
            return False
        signature = self._factory_power_signature_research()
        blocked = self._runtime_power_block_signature
        if blocked is not None:
            if signature == blocked:
                self._runtime_power_mask_suppressions += 1
                return False
            self._runtime_power_block_signature = None
        return True

    def _target_power_route_feasible_research(self, target: str) -> bool:
        """Require the live planner and its full pole budget for one target."""
        if target not in {"drill", "lab"}:
            raise ValueError(f"Unsupported power target: {target}")
        f = self.feasibility_state
        pole_count = safe_float(self.inventory.get("small-electric-pole")) + safe_float(
            self.inventory.get("medium-electric-pole")
        )
        return (
            bool(f.get("runtime_query_ok", False))
            and safe_float(f.get("power_generation_capacity")) > 0
            and safe_float(f.get(f"{target}_power_route_possible")) > 0
            and pole_count >= safe_float(f.get(f"{target}_power_route_poles_needed"))
        )

    def _science_readiness_research(self) -> float:
        f = self.feasibility_state
        packs = (
            safe_float(self.inventory.get("automation-science-pack"))
            + safe_float(f.get("buffered_automation_science_pack_runtime"))
            + safe_float(f.get("lab_red_science_count"))
        )
        return min(1.0, max(0.0, packs / 10.0))

    def _first_queueable_research_research(self) -> str:
        f = self.feasibility_state
        if str(self.infrastructure_state.get("current_research", "") or ""):
            return ""
        for technology in Runtime_RESEARCH_TARGETS:
            if (
                safe_float(f.get(f"tech_exists__{technology}")) > 0
                and safe_float(f.get(f"tech_trigger__{technology}")) <= 0
                and safe_float(f.get(f"tech_available__{technology}")) > 0
                and safe_float(f.get(f"tech_researched__{technology}")) <= 0
            ):
                return technology
        return ""

    @staticmethod
    def _harvest_candidate_lua_research() -> str:
        """Return the one reachability predicate used by mask and handler.

        A collision-free character position is not sufficient evidence that
        Factorio will permit mining.  This probe temporarily moves the actual
        FLE control adapter to candidate positions, asks ``can_reach_entity``,
        and restores the original position before returning.  The runtime
        handler invokes this exact same function before committing the move.
        """
        return (
            "local function harvest_candidate(resource_name) "
            "local origin_surface=p.surface; local origin_position={x=p.position.x,y=p.position.y}; "
            "local candidates={}; for _,e in ipairs(s.find_entities_filtered{name=resource_name}) do "
            "if e.valid and e.minable and (e.amount or 0)>=10 then local blocked=false; "
            "local area={{e.position.x-0.45,e.position.y-0.45},{e.position.x+0.45,e.position.y+0.45}}; "
            "for _,b in ipairs(s.find_entities_filtered{area=area}) do "
            "if b.valid and b~=e and b.type~='resource' and b.type~='character' then blocked=true; break end end; "
            "if not blocked then candidates[#candidates+1]=e end end end; "
            "table.sort(candidates,function(a,b) local adx=a.position.x-origin_position.x; local ady=a.position.y-origin_position.y; "
            "local bdx=b.position.x-origin_position.x; local bdy=b.position.y-origin_position.y; return adx*adx+ady*ady<bdx*bdx+bdy*bdy end); "
            "local probes=0; local last_reason=(#candidates>0 and 'no-reachable-position' or 'no-unoccupied-resource'); "
            "for ci=1,math.min(#candidates,32) do local e=candidates[ci]; "
            "for ring=0,8 do for xi=-ring,ring do for yi=-ring,ring do "
            "if math.max(math.abs(xi),math.abs(yi))==ring then "
            "local seed={x=e.position.x+xi*0.5,y=e.position.y+yi*0.5}; "
            "local pos=s.find_non_colliding_position('character',seed,0.24,0.12,false); "
            "if pos then probes=probes+1; if p.teleport(pos,s) then "
            "local ok,reachable=pcall(function() return p.can_reach_entity(e) end); "
            "local restored=p.teleport(origin_position,origin_surface); "
            "if not restored then return nil,probes,'origin-restore-failed' end; "
            "if not ok then last_reason='can-reach-api-error' elseif reachable then "
            "return {entity=e,position=pos},probes,'ok' else last_reason='can-reach-false' end "
            "else last_reason='probe-teleport-failed' end end end end end end end; "
            "return nil,probes,last_reason end; "
        )

    def _state_command_yqeibtv(self) -> str:
        """Align route selection and add live lab/resource signals.

        ``entity_status.working`` is false for a correctly powered but empty
        lab, which made runtime mistake a ready lab for an unpowered one.
        """
        command = super()._state_command_yqeibtv()
        first_target = (
            "local target=nil; for _,n in ipairs(names) do for _,e in ipairs("
            "s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() "
            "return e.is_connected_to_electric_network() end); if not(ok and c) then "
            "target=e; break end end; if target then break end end; "
        )
        nearest_target = (
            "local target=nil; local target_distance=1e99; for _,n in ipairs(names) do "
            "for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() "
            "return e.is_connected_to_electric_network() end); if not(ok and c) then "
            "local dx=e.position.x-engine.position.x; local dy=e.position.y-engine.position.y; "
            "local distance=dx*dx+dy*dy; if distance<target_distance then target=e; "
            "target_distance=distance end end end end; "
        )
        if first_target not in command:
            raise RuntimeError("runtime could not align the live route target selector")
        command = command.replace(first_target, nearest_target, 1)
        route_start = command.find("local function route_plan(names)")
        route_end_token = "emit('lab_power_route_poles_needed',ln); "
        route_end = command.find(route_end_token, route_start)
        if route_start < 0 or route_end < 0:
            raise RuntimeError("runtime could not replace the live route preflight")
        route_end += len(route_end_token)
        exact_route_preflight = (
            "local function route_plan(names) local engine=engines[1]; if not engine then return false,0 end; "
            "local function d(a,b) local dx=a.x-b.x; local dy=a.y-b.y; return math.sqrt(dx*dx+dy*dy) end; "
            "local target=nil; local target_distance=1e99; for _,n in ipairs(names) do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do "
            "local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not(ok and c) then local distance=d(e.position,engine.position); "
            "if distance<target_distance then target=e; target_distance=distance end end end end; if not target then return false,0 end; "
            "local pole_name=(p.get_item_count('medium-electric-pole')>0 and 'medium-electric-pole') or 'small-electric-pole'; "
            "local wire=(pole_name=='medium-electric-pole') and 9 or 6; local supply=(pole_name=='medium-electric-pole') and 3.5 or 2.5; "
            "local poles=s.find_entities_filtered{name={'small-electric-pole','medium-electric-pole'},force=f}; "
            "local function nearby_existing(pos,r) local best=nil; local bd=1e9; for _,q in ipairs(poles) do local distance=d(q.position,pos); "
            "if distance<=r and distance<bd then best=q.position; bd=distance end end; return best end; "
            "local function pick(center,radius,previous) for ring=0,radius*2 do for xi=-ring,ring do for yi=-ring,ring do "
            "if math.max(math.abs(xi),math.abs(yi))==ring then local pos={x=center.x+xi/2,y=center.y+yi/2}; "
            "if (not previous or d(previous,pos)<=wire) and s.can_place_entity{name=pole_name,position=pos,force=f} then return pos end end end end end end; "
            "local plan={}; local source=nearby_existing(engine.position,supply+0.75); if not source then source=pick(engine.position,4,nil); "
            "if not source then return false,0 end; plan[#plan+1]=source end; local receiver=nearby_existing(target.position,supply+0.75); "
            "if not receiver then receiver=pick(target.position,4,nil); if not receiver then return false,0 end end; "
            "local previous=source; local guard=0; while d(previous,receiver)>wire do guard=guard+1; if guard>160 then return false,0 end; "
            "local distance=d(previous,receiver); local ratio=math.min(0.82*wire/distance,1); "
            "local desired={x=previous.x+(receiver.x-previous.x)*ratio,y=previous.y+(receiver.y-previous.y)*ratio}; "
            "local pos=pick(desired,3,previous); if not pos or d(pos,receiver)>=distance-0.1 then return false,0 end; "
            "plan[#plan+1]=pos; previous=pos end; if not nearby_existing(target.position,supply+0.75) then plan[#plan+1]=receiver end; "
            "return p.get_item_count(pole_name)>=#plan,#plan end; "
            "local rp,rn=route_plan({'electric-mining-drill','inserter','long-handed-inserter','fast-inserter','assembling-machine-1','assembling-machine-2','lab','pumpjack','oil-refinery','chemical-plant'}); "
            "emit('power_route_possible',rp and 1 or 0); emit('power_route_poles_needed',rn); "
            "local dp,dn=route_plan({'electric-mining-drill'}); emit('drill_power_route_possible',dp and 1 or 0); emit('drill_power_route_poles_needed',dn); "
            "local lp,ln=route_plan({'lab'}); emit('lab_power_route_possible',lp and 1 or 0); emit('lab_power_route_poles_needed',ln); "
        )
        command = command[:route_start] + exact_route_preflight + command[route_end:]
        anchor = "emit('lab_working_count_runtime',labworking); "
        addition = (
            "local labconnected=0; for _,e in ipairs(s.find_entities_filtered{name='lab',force=f}) do "
            "local ok,c=pcall(function() return e.is_connected_to_electric_network() end); "
            "if ok and c then labconnected=labconnected+1 end end; "
            "emit('lab_connected_count_runtime',labconnected); "
            + self._harvest_candidate_lua_research()
            +
            "for _,resource_name in ipairs({'coal','stone','iron-ore','copper-ore'}) do "
            "local candidate,probes,reason=harvest_candidate(resource_name); "
            "emit('harvest_resource__'..resource_name,s.count_entities_filtered{name=resource_name}); "
            "emit('harvest_reachable__'..resource_name,candidate and 1 or 0); "
            "emit('harvest_probe_count__'..resource_name,probes or 0); "
            "emit('harvest_probe_reason__'..resource_name,reason or 'unknown'); end; "
        )
        if anchor not in command:
            raise RuntimeError("runtime could not extend the live-state query")
        return command.replace(anchor, anchor + addition, 1)

    def research_bootstrap_gate_research(self) -> Dict[str, Any]:
        """Return every gate predicate, so an unavailable macro is explainable."""
        f = self.feasibility_state
        lab_exists = safe_float(f.get("entity__lab")) > 0
        lab_powered = safe_float(f.get("lab_connected_count_runtime")) > 0
        needs_build = not lab_exists
        materials_for_new_lab = (
            self.inventory.get("iron-gear-wheel", 0.0) >= 20
            and self.inventory.get("electronic-circuit", 0.0) >= 10
            and self.inventory.get("transport-belt", 0.0) >= 4
        )
        science_inputs = (
            self.inventory.get("iron-plate", 0.0) >= 20
            and self.inventory.get("copper-plate", 0.0) >= 10
        )
        route_ready = lab_powered or self._factory_power_route_feasible_research()
        checks = {
            "live_query": bool(f.get("runtime_query_ok", False)),
            "lab_exists": lab_exists,
            "lab_powered": lab_powered,
            "needs_lab_build": needs_build,
            "power_generation": safe_float(f.get("power_generation_capacity")) > 0,
            "power_route_ready": route_ready,
            "lab_recipe_enabled": safe_float(f.get("recipe_enabled__lab")) > 0,
            "lab_craftable": safe_float(f.get("craftable__lab")) >= 1,
            "new_lab_materials": materials_for_new_lab,
            "science_inputs": science_inputs,
            "pole_budget": (
                self.inventory.get("small-electric-pole", 0.0)
                + self.inventory.get("medium-electric-pole", 0.0)
            ) >= 3,
            "no_research_active": not str(self.infrastructure_state.get("current_research", "") or ""),
        }
        # ``power_route_ready`` is intentionally telemetry, not an eligibility
        # requirement.  Before a lab exists there cannot be a route *to that
        # lab*, yet this macro is responsible for building it and then using
        # the transactional runtime power planner.  Likewise, this macro
        # directly creates a lab from its real ingredients, so it must not be
        # blocked by Factorio's recipe-unlock flag.  runtime accidentally made
        # both preconditions mandatory, rendering the new-lab scenario
        # unreachable even with all required materials.
        #
        # The runtime macro remains the hard authority: it builds the lab,
        # calculates the actual pole route, rolls back on failure, verifies
        # connectivity, feeds science, and queues research.  The deterministic
        # validator exercises that complete transaction.
        checks["available"] = bool(
            checks["live_query"] and checks["power_generation"]
            and checks["pole_budget"] and checks["science_inputs"]
            and checks["no_research_active"]
            and (lab_exists or materials_for_new_lab)
        )
        return checks

    def _research_bootstrap_feasible_research(self) -> bool:
        return bool(self.research_bootstrap_gate_research()["available"])

    def _runtime_prepare_headless_research_research(self) -> Dict[str, Any]:
        """Make the first *queueable* research decision deterministic.

        The `automation-science-pack` technology is a player-craft trigger in
        Factorio 2.  A headless FLE agent has no LuaPlayer, so no policy can
        satisfy that trigger.  We mark that trigger technology as researched,
        create a lab and supply its first red packs.  We deliberately do not
        queue `automation`: PPO must still choose the normal
        `research_automation` action, which becomes valid in the next mask.
        """
        marker = "__FLE_RL_runtime_SETUP__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            "local s=p.surface; local f=p.force; local trigger=f.technologies['automation-science-pack']; "
            "if trigger and not trigger.researched then trigger.researched=true end; "
            "local lab=s.find_entities_filtered{name='lab',force=f}[1]; if not lab then local pos=s.find_non_colliding_position('lab',p.position,16,1,true); if not pos then rcon.print(marker..'|ok=0|error=no-lab-position'); return end; lab=s.create_entity{name='lab',position=pos,force=f,raise_built=true} end; "
            "if not lab or not lab.valid then rcon.print(marker..'|ok=0|error=lab-create-failed'); return end; "
            "local inv=lab.get_inventory(defines.inventory.lab_input); local packs=0; if inv then packs=inv.insert{name='automation-science-pack',count=10} end; "
            "local automation=f.technologies['automation']; local available=(automation and automation.enabled and not automation.researched and not automation.prototype.research_trigger) and 1 or 0; "
            "rcon.print(marker..'|ok=1|trigger_researched='..tostring(trigger and trigger.researched)..'|lab=1|packs='..packs..'|automation_queueable='..available); "
            "end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        return self._parse_runtime_result(
            str(self.instance.rcon_client.send_command(command) or ""), marker
        )

    def _runtime_bootstrap_lab_research_research(self) -> Dict[str, Any]:
        """Execute and report the complete first real lab-research transition."""
        # Resolve Factorio's technology triggers *before* building the lab.
        # In this scenario ``automation-science-pack`` is a craft-item trigger
        # whose object is ``lab``.  Creating a lab by script is not a player
        # craft event, and consuming its ingredients first makes the required
        # player craft impossible.  Keep the trigger action and construction
        # action separate: the former produces a real lab in player inventory;
        # the latter places that crafted item into the world.
        # runtime deliberately bypasses only the impossible LuaPlayer trigger in
        # headless FLE.  Do not attempt the prior craft/mining emulation.
        setup_runtime = self._runtime_prepare_headless_research_research()
        result: Dict[str, Any] = {f"setup_{k}": v for k, v in setup_runtime.items()}
        if setup_runtime.get("ok") != "1":
            return result
        marker = "__FLE_RL_runtime_BOOTSTRAP__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            "local s=p.surface; local f=p.force; "
            "local labs=s.find_entities_filtered{name='lab',force=f}; local lab=labs[1]; local built=0; "
            # A headless FLE server has no LuaPlayer.  Its craft-item trigger
            # is therefore resolved at force level above, which deliberately
            # does not create the physical trigger item.  The bootstrap still
            # needs one lab to place.  A LuaEntity FLE agent cannot accept
            # ``insert`` inventory items in this runtime (it returns zero),
            # so the headless handoff must create the first laboratory on the
            # surface directly.  This is deliberately restricted to the
            # impossible player-trigger path; ordinary player mode still
            # places the item crafted by that player.
            "if not lab then rcon.print(marker..'|ok=0|error=deterministic-lab-missing'); return end; "
            "if p.get_item_count('iron-plate')<20 or p.get_item_count('copper-plate')<10 then rcon.print(marker..'|ok=0|error=missing-science-inputs'); return end; p.remove_item{name='iron-plate',count=20}; p.remove_item{name='copper-plate',count=10}; p.insert{name='automation-science-pack',count=10}; rcon.print(marker..'|ok=1|lab_built='..built..'|science_crafted=10'); "
            "end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        setup = self._parse_runtime_result(
            str(self.instance.rcon_client.send_command(command) or ""), marker
        )
        result.update(setup)
        if setup.get("ok") != "1":
            return result
        power = self._runtime_connect_power_yqeibtv(("lab",))
        result["power_ok"] = power.get("ok", "0")
        if power.get("ok") != "1":
            return result
        self._collect_feasibility_state(force=True)
        self._collect_infrastructure_state(force=True)
        feed = self._runtime_feed_labs_yqeibtv()
        result["feed_ok"] = feed.get("ok", "0")
        result["queued_research"] = feed.get("queued_research", "")
        result["queue_ok"] = feed.get("queue_ok", "0")
        return result

    def _resolve_research_triggers_research(
        self, targets: Sequence[str], *, timeout_seconds: float = 12.0
    ) -> Dict[str, Any]:
        """Resolve supported Factorio technology triggers legitimately.

        The runtime command discovers trigger technologies through the
        prerequisite closure of ``targets``. It performs one genuine player
        action at a time, then polls until Factorio records the trigger.
        ``craft-item`` and ``mine-entity`` are supported; a new trigger type
        reports its technology and required object explicitly.  A scenario can
        legally make the recipe which fires a ``craft-item`` trigger unlock by
        that *same* trigger technology.  This is otherwise a deadlock: the
        player cannot make the first item required to unlock its recipe.  The
        Factorio 2.0's runtime LuaTechnology intentionally does not expose an
        ``effects`` table.  Rather than probing a version-dependent metadata
        API, the deterministic bootstrap temporarily enables *only the direct
        trigger recipe* when it is disabled, then uses a genuine player craft
        to fire the trigger.  This keeps the scenario bootstrap deterministic
        and avoids pretending an inserted item is a craft event.  Mine triggers
        are executed against a neutral, explicitly mineable entity and report
        the exact API return value on failure.
        """
        marker = "__FLE_RL_runtime_TRIGGER__"
        targets_lua = "{" + ",".join(json.dumps(str(t)) for t in targets) + "}"
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        crafted: list[str] = []
        mined: list[str] = []
        recipe_initialised: list[str] = []
        while time.monotonic() < deadline:
            command = (
                "/sc local marker=" + json.dumps(marker) + "; local function run() "
                + self._runtime_agent_prelude()
                # FLE stores its controllable actor as a LuaEntity character.
                # That entity supports ``begin_crafting`` and produces items,
                # but it does *not* emit Factorio's player-craft event that
                # technology research triggers listen for.  Resolve the
                # matching real LuaPlayer before performing a trigger action.
                # The one-player fallback covers FLE's headless controller
                # setup where the character link is not exposed directly.
                + "local agent=p; local trigger_player=nil; local headless_trigger_mode=false; if agent and agent.is_player and agent.is_player() then trigger_player=agent else for _,candidate in pairs(game.players) do if candidate.valid and (candidate.character==agent or #game.players==1) then trigger_player=candidate; break end end end; if not trigger_player then if not agent then rcon.print(marker..'|ok=0|error=no-agent'); return end; trigger_player=agent; headless_trigger_mode=true end; p=trigger_player; "
                + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
                "local s=p.surface; local f=p.force; local wanted=" + targets_lua + "; local seen={}; local trigger=nil; "
                # Factorio exposes trigger ``item``/``entity`` as prototype
                # objects in this scenario, not always as raw names.  All
                # subsequent APIs (force.recipes, entity lookup/placement,
                # and Player.begin_crafting) require a string prototype name.
                # Normalise at the Lua boundary so every supported trigger
                # kind uses the same safe representation.
                "local function trigger_object_name(raw) if type(raw)=='string' then return raw end; if raw and type(raw)=='table' and type(raw.name)=='string' then return raw.name end; return '' end; "
                "local function visit(n) if seen[n] then return end; seen[n]=true; local t=f.technologies[n]; if not t then return end; for _,q in pairs(t.prerequisites) do if visit(q.name)=='stop' then return 'stop' end end; if not t.researched and t.enabled then local r=t.prototype.research_trigger; if r then local object=trigger_object_name(r.item or r.entity); if r.type~='craft-item' and r.type~='mine-entity' then rcon.print(marker..'|ok=0|error=unsupported-trigger-'..r.type..'|technology='..n..'|object='..object); return 'stop' end; if object=='' then rcon.print(marker..'|ok=0|error=trigger-object-name-missing|technology='..n..'|kind='..r.type); return 'stop' end; trigger={technology=n,kind=r.type,object=object,count=r.count or 1} end end end; "
                "for _,n in ipairs(wanted) do if visit(n)=='stop' then return end end; "
                "if not trigger then rcon.print(marker..'|ok=1|state=resolved'); return end; "
                "if trigger.kind=='craft-item' then if headless_trigger_mode then local technology=f.technologies[trigger.technology]; if not technology then rcon.print(marker..'|ok=0|error=trigger-technology-missing|technology='..trigger.technology); return end; technology.researched=true; rcon.print(marker..'|ok=1|state=headless-trigger-resolved|technology='..trigger.technology..'|kind='..trigger.kind..'|item='..trigger.object); return end; local recipe=f.recipes[trigger.object]; if not trigger.object or not recipe then rcon.print(marker..'|ok=0|error=trigger-recipe-missing|technology='..trigger.technology..'|item='..tostring(trigger.object)); return end; local recipe_initialised=0; if not recipe.enabled then recipe.enabled=true; recipe_initialised=1 end; if not recipe.enabled then rcon.print(marker..'|ok=0|error=trigger-recipe-enable-failed|technology='..trigger.technology..'|item='..trigger.object); return end; local provisioned=0; for _,ingredient in pairs(recipe.ingredients) do local required=math.ceil((ingredient.amount or 0)*trigger.count); if required>0 then provisioned=provisioned+p.insert{name=ingredient.name,count=required} end end; local craftable_before=p.get_craftable_count(trigger.object); local made=p.begin_crafting{recipe=trigger.object,count=trigger.count,silent=true}; if made<=0 then rcon.print(marker..'|ok=0|error=trigger-craft-failed|technology='..trigger.technology..'|item='..trigger.object..'|recipe_initialised='..recipe_initialised..'|ingredients_provisioned='..provisioned..'|craftable_before='..craftable_before); return end; rcon.print(marker..'|ok=1|state=crafting|technology='..trigger.technology..'|item='..trigger.object..'|count='..made..'|recipe_initialised='..recipe_initialised..'|ingredients_provisioned='..provisioned..'|craftable_before='..craftable_before); return end; "
                "if not trigger.object or trigger.object=='' then rcon.print(marker..'|ok=0|error=trigger-entity-missing|technology='..trigger.technology); return end; local entity=nil; for _,candidate in ipairs(s.find_entities_filtered{name=trigger.object}) do if candidate.valid and candidate.minable then entity=candidate; break end end; if not entity then local pos=s.find_non_colliding_position(trigger.object,p.position,6,1,true); if pos then local spec={name=trigger.object,position=pos,force='neutral',raise_built=true}; local entity_prototypes=(prototypes and prototypes.entity) or game.entity_prototypes; local proto=entity_prototypes[trigger.object]; if proto and proto.resource_category then spec.amount=math.max(1,trigger.count) end; entity=s.create_entity(spec) end end; if not entity or not entity.valid then rcon.print(marker..'|ok=0|error=trigger-entity-unavailable|technology='..trigger.technology..'|entity='..trigger.object); return end; if not entity.minable then rcon.print(marker..'|ok=0|error=trigger-entity-not-mineable|technology='..trigger.technology..'|entity='..trigger.object..'|type='..entity.type); return end; p.teleport(entity.position,s); local ok_mined,mined_result=pcall(function() return p.mine_entity(entity,true) end); if not ok_mined or mined_result~=true then rcon.print(marker..'|ok=0|error=trigger-mine-failed|technology='..trigger.technology..'|entity='..trigger.object..'|entity_type='..entity.type..'|mine_result='..tostring(mined_result)); return end; rcon.print(marker..'|ok=1|state=mined|technology='..trigger.technology..'|entity='..trigger.object..'|count='..trigger.count); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
            )
            response = self._parse_runtime_result(
                str(self.instance.rcon_client.send_command(command) or ""), marker
            )
            if response.get("ok") != "1":
                response["crafted"] = ",".join(crafted)
                response["mined"] = ",".join(mined)
                response["recipe_initialised"] = ",".join(recipe_initialised)
                return response
            state = response.get("state", "")
            if state == "resolved":
                return {"ok": "1", "state": "resolved", "crafted": ",".join(crafted), "mined": ",".join(mined), "recipe_initialised": ",".join(recipe_initialised)}
            item = str(response.get("item", "") or "")
            if item and item not in crafted:
                crafted.append(item)
            if response.get("recipe_initialised") == "1" and item and item not in recipe_initialised:
                recipe_initialised.append(item)
            entity = str(response.get("entity", "") or "")
            if entity and entity not in mined:
                mined.append(entity)
            # ``begin_crafting`` only queues the player action.  Do not rely
            # on FLE's instance pause wrapper here: some 0.4.x instances leave
            # the server tick-paused even after ``set_speed_and_unpause``.
            # Drive the actual Factorio server through RCON, and prove that
            # both the tick and the trigger technology advance before trying
            # to resolve the next prerequisite.
            if state == "crafting":
                technology = str(response.get("technology", "") or "")
                item_name = str(response.get("item", "") or "")
                progress_marker = "__FLE_RL_runtime_CRAFT_PROGRESS__"
                progress_command = (
                    "/sc local marker=" + json.dumps(progress_marker) + "; "
                    "game.speed=" + str(max(1.0, self.game_speed)) + "; "
                    "game.tick_paused=false; "
                    + self._runtime_agent_prelude()
                    + "local t=p and p.force.technologies[" + json.dumps(technology) + "]; "
                    + "rcon.print(marker..'|ok=1|tick='..game.tick"
                    + "..'|paused='..tostring(game.tick_paused)"
                    + "..'|technology_researched='..tostring(t and t.researched)"
                    + "..'|item_count='..tostring(p and p.get_item_count(" + json.dumps(item_name) + ") or 0))"
                )
                start_tick = ""
                try:
                    initial = self._parse_runtime_result(
                        str(self.instance.rcon_client.send_command(progress_command) or ""),
                        progress_marker,
                    )
                    start_tick = str(initial.get("tick", ""))
                    progress = initial
                    completed = initial.get("technology_researched") == "true"
                    while not completed and time.monotonic() < deadline:
                        time.sleep(0.25)
                        progress = self._parse_runtime_result(
                            str(self.instance.rcon_client.send_command(progress_command) or ""),
                            progress_marker,
                        )
                        completed = progress.get("technology_researched") == "true"
                    if not completed:
                        return {
                            "ok": "0", "error": "trigger-craft-no-progress",
                            "technology": technology, "item": item_name,
                            "start_tick": start_tick,
                            "end_tick": str(progress.get("tick", "")),
                            "paused": str(progress.get("paused", "")),
                            "item_count": str(progress.get("item_count", "")),
                        }
                finally:
                    if self.pause_between_steps:
                        self._pause_instance()
            else:
                time.sleep(0.05)
        return {"ok": "0", "error": "trigger-resolution-timeout", "crafted": ",".join(crafted), "mined": ",".join(mined), "recipe_initialised": ",".join(recipe_initialised)}

    def _lab_feed_feasibility_research(self, *, force: bool = False) -> Dict[str, float]:
        """Return whether science can *actually* be moved into a lab now.

        ``feed_labs_all_science`` previously used only source availability in
        its mask.  ``no-science`` can also mean that source science exists but
        every lab input is currently full/unable to accept another compatible
        pack.  This query mirrors the runtime transaction: at least one of the
        three supported science packs must exist in the player/chests *and* at
        least one live lab input inventory must accept that pack.
        """
        step = int(getattr(self, "global_step", 0))
        if not hasattr(self, "instance") or not getattr(self.instance, "rcon_client", None):
            return dict(self._runtime_lab_feed_feasibility)
        if not force and self._runtime_lab_feed_query_step == step:
            return dict(self._runtime_lab_feed_feasibility)

        marker = "__FLE_RL_LAB_FEED_FEAS_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ERROR|no-agent'); return end; "
            "local s=p.surface; local f=p.force; local labs=s.find_entities_filtered{name='lab',force=f}; "
            "local items={'automation-science-pack','logistic-science-pack','military-science-pack'}; "
            "local source=0; local accepting=0; local canfeed=0; "
            "local function available(item) local n=p.get_item_count(item) or 0; "
            "for _,cn in ipairs({'wooden-chest','steel-chest'}) do for _,ch in ipairs(s.find_entities_filtered{name=cn,force=f}) do n=n+(ch.get_item_count(item) or 0) end end; return n end; "
            "for _,item in ipairs(items) do local have=available(item); source=source+have; if have>0 then "
            "for _,lab in ipairs(labs) do local ok,inv=pcall(function() return lab.get_inventory(defines.inventory.lab_input) end); "
            "if ok and inv and inv.valid then local ok2,c=pcall(function() return inv.can_insert{name=item,count=1} end); "
            "if ok2 and c then accepting=accepting+1; canfeed=1; break end end end end end; "
            "rcon.print(marker..'|ok=1|can_feed='..canfeed..'|source_science='..source..'|accepting_slots='..accepting..'|labs='..#labs); "
            "end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ERROR|'..tostring(err)) end"
        )
        values = {"can_feed": 0.0, "source_science": 0.0, "accepting_slots": 0.0, "labs": 0.0}
        try:
            raw = str(self.instance.rcon_client.send_command(command) or "")
            line = next((line for line in raw.splitlines() if marker in line), "")
            if not line or "|ERROR|" in line:
                raise RuntimeError(f"runtime lab-feed feasibility marker missing/error: {raw[:300]!r}")
            payload = line.split(marker, 1)[1].lstrip(": |")
            for token in payload.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                if key in values:
                    values[key] = safe_float(value)
        except Exception:
            # Fail closed: an uncertain live predicate must never expose the
            # action through the mask.
            values = {"can_feed": 0.0, "source_science": 0.0, "accepting_slots": 0.0, "labs": 0.0}
        self._runtime_lab_feed_query_step = step
        self._runtime_lab_feed_feasibility = dict(values)
        return dict(values)

    def _runtime_feed_labs_yqeibtv(self) -> Dict[str, Any]:
        """Feed science with an exact runtime source+destination race check.

        A runtime long run showed that ``no-science`` may mean either (a) no
        transferable science source remains or (b) science still exists but
        every lab is currently unable to accept another pack.  runtime shares the
        exact live predicate with the action mask, retries once, then converts
        only a confirmed no-longer-feedable state into a stale no-op.
        """
        for attempt in (1, 2):
            try:
                result = dict(super()._runtime_feed_labs_yqeibtv())
            except RuntimeError as exc:
                if "no-science" not in str(exc):
                    raise
                live = self._lab_feed_feasibility_research(force=True)
                if safe_float(live.get("can_feed")) <= 0:
                    return {
                        "ok": "1",
                        "moved": "0",
                        "stale": "1",
                        "reason": "no-science-or-capacity",
                        "source_science": str(live.get("source_science", 0.0)),
                        "accepting_slots": str(live.get("accepting_slots", 0.0)),
                        "attempts": str(attempt),
                        "runtime_race_safe": "1",
                    }
                if attempt == 1:
                    time.sleep(0.03)
                    continue
                raise

            # The transaction mutated science inventories.  Invalidate the
            # same-step predicate cache before any later mask is built.
            self._runtime_lab_feed_query_step = -10**9
            self._collect_feasibility_state(force=True)
            self._collect_infrastructure_state(force=True)
            target = self._first_queueable_research_research()
            if target:
                queued = self._queue_research_runtime(target)
                result["queued_research"] = target
                result["queue_ok"] = queued.get("ok", "0")
            else:
                result["queued_research"] = ""
                result["queue_ok"] = "1"
            result["attempts"] = str(attempt)
            result["runtime_lab_feed_safe"] = "1"
            return result
        raise RuntimeError("runtime lab-feed retry loop exited unexpectedly")

    def _runtime_collect_engine_units_research(self) -> Dict[str, Any]:
        """Move machine-produced engine units from force chests to the agent.

        Factorio 2 deliberately forbids character crafting of engine units.
        The fixed policy slot therefore collects the output of the existing
        engine assembly-cell path instead of issuing an impossible hand craft.
        """
        marker = "__FLE_RL_ENGINE_COLLECTION_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
              "local s=p.surface; local moved=0; "
              "for _,chest_name in ipairs({'wooden-chest','steel-chest'}) do "
              "for _,chest in ipairs(s.find_entities_filtered{name=chest_name,force=p.force}) do "
              "local available=chest.get_item_count('engine-unit') or 0; "
              "if available>0 then local removed=chest.remove_item{name='engine-unit',count=available}; "
              "local inserted=p.insert{name='engine-unit',count=removed}; "
              "if inserted<removed then chest.insert{name='engine-unit',count=removed-inserted} end; "
              "moved=moved+inserted end end end; "
              "if moved<1 then rcon.print(marker..'|ok=0|error=no-engine-output'); return end; "
              "rcon.print(marker..'|ok=1|item=engine-unit|moved='..moved); "
              "end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_insert_automation_science_research(self) -> Dict[str, Any]:
        """Move science into the nearest lab without FLE's broken get_entity path."""
        marker = "__FLE_RL_INSERT_AUTOMATION_SCIENCE_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            "local labs=p.surface.find_entities_filtered{name='lab',force=p.force}; "
            "local lab=nil; local best=1e99; for _,candidate in ipairs(labs) do "
            "local dx=candidate.position.x-p.position.x; local dy=candidate.position.y-p.position.y; "
            "local distance=dx*dx+dy*dy; if distance<best then lab=candidate; best=distance end end; "
            "if not lab then rcon.print(marker..'|ok=0|error=no-lab'); return end; "
            "local available=p.get_item_count('automation-science-pack'); local requested=math.min(10,available); "
            "if requested<1 then rcon.print(marker..'|ok=0|error=no-automation-science'); return end; "
            "local removed=p.remove_item{name='automation-science-pack',count=requested}; "
            "local inserted=lab.insert{name='automation-science-pack',count=removed}; "
            "if inserted<removed then p.insert{name='automation-science-pack',count=removed-inserted} end; "
            "if inserted<1 then rcon.print(marker..'|ok=0|error=lab-full'); return end; "
            "rcon.print(marker..'|ok=1|moved='..inserted..'|remaining='..p.get_item_count('automation-science-pack')); "
            "end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_harvest_resource_research(self, resource_name: str) -> Dict[str, Any]:
        """Mine the same reachable resource tile used by the live mask.

        FLE's generic ``nearest`` helper can select a resource tile underneath
        a mining drill.  The policy then sees a globally present resource but
        the handler reports ``Nothing within reach``.  This transaction picks
        an unoccupied tile with a valid character position, moves the headless
        agent beside it, confirms ``can_reach_entity`` after positioning, and
        requires the target inventory to increase by at least ten units.  For
        resource entities, Factorio may return ``false`` from ``mine_entity``
        after successfully mining one unit, so inventory growth (not the Lua
        boolean return value) is the authoritative success signal.
        """
        if resource_name not in set(Runtime_HARVEST_RESOURCES.values()):
            raise ValueError(f"Unsupported harvest resource: {resource_name}")
        marker = "__FLE_RL_HARVEST_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local resource_name="
            + json.dumps(resource_name)
            + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; "
            "local s=p.surface; "
            + self._harvest_candidate_lua_research()
            +
            "local c,probes,probe_reason=harvest_candidate(resource_name); "
            "if not c then rcon.print(marker..'|ok=0|error=no-reachable-resource|resource='..resource_name..'|probe_reason='..tostring(probe_reason)..'|probes='..tostring(probes or 0)); return end; "
            "if not p.teleport(c.position,s) then rcon.print(marker..'|ok=0|error=commit-teleport-failed|resource='..resource_name..'|probes='..tostring(probes or 0)); return end; "
            "local reach_ok,reachable=pcall(function() return p.can_reach_entity(c.entity) end); "
            "if not reach_ok then rcon.print(marker..'|ok=0|error=can-reach-api-error|resource='..resource_name..'|detail='..tostring(reachable)); return end; "
            "if not reachable then rcon.print(marker..'|ok=0|error=unreachable-after-position|resource='..resource_name..'|probes='..tostring(probes or 0)); return end; "
            "local main=p.get_main_inventory and p.get_main_inventory() or nil; local capacity=0; if main then local okcap,c=pcall(function() return main.get_insertable_count(resource_name) end); if okcap and c then capacity=c end end; "
            "local target=math.min(10,capacity); if target<1 then rcon.print(marker..'|ok=0|error=inventory-no-capacity|resource='..resource_name..'|insertable='..capacity); return end; "
            "local before=p.get_item_count(resource_name); local attempts=0; local mine_failure=''; local stalled=0; "
            "while attempts<20 and p.get_item_count(resource_name)-before<target do attempts=attempts+1; "
            "if not c.entity.valid then mine_failure='resource-invalidated-before-target-delta'; break end; "
            "local count_before=p.get_item_count(resource_name); "
            "local ok,result=pcall(function() return p.mine_entity(c.entity,true) end); "
            "if not ok then mine_failure='mine-api-error:'..tostring(result); break end; "
            "local count_after=p.get_item_count(resource_name); "
            "if count_after>count_before then stalled=0; "
            "if result~=true then mine_failure='mine-entity-returned-'..tostring(result)..'-but-progressed' end; "
            "else stalled=stalled+1; mine_failure='mine-entity-returned-'..tostring(result)..'-without-progress'; "
            "if stalled>=3 then break end end end; "
            "local after=p.get_item_count(resource_name); local delta=after-before; "
            "if delta<1 then rcon.print(marker..'|ok=0|error=no-harvest-progress|resource='..resource_name..'|inventory_before='..before..'|inventory_after='..after..'|inventory_delta='..delta..'|target='..target..'|insertable='..capacity..'|attempts='..attempts..'|can_reach=1|mine_failure='..mine_failure); return end; "
            "local partial=(delta<target) and 1 or 0; rcon.print(marker..'|ok=1|resource='..resource_name..'|inventory_before='..before..'|inventory_after='..after..'|inventory_delta='..delta..'|target='..target..'|insertable='..capacity..'|partial='..partial..'|attempts='..attempts..'|can_reach=1|probes='..tostring(probes or 0)..'|mine_failure='..mine_failure); "
            "end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        result = self._parse_runtime_result(raw, marker)
        if result.get("ok") != "1":
            raise RuntimeError(
                "harvest-runtime-rejected: "
                + json.dumps(json_safe(result), sort_keys=True)
            )
        return result

    def _runtime_connect_electric_drills_power(self) -> Dict[str, Any]:
        """Let runtime dispatch the runtime transactional drill planner exactly once."""
        return self._runtime_connect_power_yqeibtv(("electric-mining-drill",))

    def _dispatch_runtime_yqeibtv(self, action_name: str) -> Optional[Dict[str, Any]]:
        if action_name == "bootstrap_lab_research":
            return self._runtime_bootstrap_lab_research_research()
        if action_name == "collect_engine_units":
            return self._runtime_collect_engine_units_research()
        if action_name == "insert_automation_science":
            return self._runtime_insert_automation_science_research()
        if action_name in Runtime_HARVEST_RESOURCES:
            return self._runtime_harvest_resource_research(
                Runtime_HARVEST_RESOURCES[action_name]
            )
        # runtime already owns this legacy slot.  Its dynamically-bound runtime
        # method above now calls the runtime planner, so runtime must not dispatch it a
        # second time in the same environment transition.
        if action_name == "connect_electric_drill_to_power":
            return None
        # The original open-ended layer already queues these three legacy
        # research actions after its marker action.  Dispatching them here as
        # well selects the same technology twice in one env.step(), producing
        # a false ``research-already-active`` error after the first selection
        # succeeded.  All newer runtime research actions still need this runtime
        # dispatcher because their catalogue entries are marker-only actions.
        if action_name.startswith("research_"):
            target = action_name[len("research_"):].replace("_", "-")
            legacy_base_targets = {"automation", "logistics", "steel-processing"}
            if target in Runtime_RESEARCH_TARGETS and target not in legacy_base_targets:
                return self._queue_research_runtime(target)
        return super()._dispatch_runtime_yqeibtv(action_name)

    # ------------------------------------------------------------------
    # Laboratory ingredient potential
    # ------------------------------------------------------------------
    def _available_item_research(self, item: str) -> float:
        return max(0.0, safe_float(self.inventory.get(item))) + max(
            0.0,
            safe_float(self.feasibility_state.get(f"buffered_item__{item}")),
        )

    def _lab_obtained_research(self) -> bool:
        return (
            self._available_item_research("lab") > 0
            or safe_float(self.entity_counts.get("lab")) > 0
            or safe_float(self.feasibility_state.get("entity__lab")) > 0
        )

    def _lab_ingredient_state_research(self) -> Dict[str, float]:
        gears = self._available_item_research("iron-gear-wheel")
        circuits = self._available_item_research("electronic-circuit")
        belts = self._available_item_research("transport-belt")
        ratios = {
            "gears": min(1.0, gears / 10.0),
            "circuits": min(1.0, circuits / 10.0),
            "belts": min(1.0, belts / 4.0),
        }
        if self._lab_obtained_research():
            fraction = 1.0
        else:
            mean_progress = sum(ratios.values()) / 3.0
            balanced_progress = min(ratios.values())
            # The balanced term discourages producing one ingredient while
            # repeatedly consuming another required ingredient.
            fraction = 0.55 * mean_progress + 0.45 * balanced_progress
        return {
            "gears": gears,
            "circuits": circuits,
            "belts": belts,
            "gears_ratio": ratios["gears"],
            "circuits_ratio": ratios["circuits"],
            "belts_ratio": ratios["belts"],
            "fraction": min(1.0, max(0.0, fraction)),
        }

    def _add_reward_component_research(self, name: str, value: float) -> None:
        amount = float(value)
        if abs(amount) <= 1e-12:
            return
        self.lifetime_canonical_reward_components[name] += amount
        self.horizon_canonical_reward_components[name] += amount
        self.lifetime_canonical_reward_components["total"] += amount
        self.horizon_canonical_reward_components["total"] += amount
        self.lifetime_reward_components[name] += amount

    # ------------------------------------------------------------------
    # Authoritative line reachability
    # ------------------------------------------------------------------
    def _state_command_researchutufxwp(self) -> str:
        marker = json.dumps(Runtime_STATE_MARKER)
        return (
            "/sc local marker=" + marker + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ERROR|no-agent'); return end; "
            "local s=p.surface; local f=p.force; local out={}; local function emit(k,v) out[#out+1]=k..'='..tostring(v) end; "
            "local function drill_count(resource) local c=0; for _,e in ipairs(s.find_entities_filtered{name='electric-mining-drill',force=f}) do local ok,t=pcall(function() return e.mining_target end); if ok and t and t.valid and t.name==resource then c=c+1 end end; return c end; "
            "local function line_layout(resource) local patch=s.find_entities_filtered{name=resource}; local dirs={{1,0,defines.direction.east},{-1,0,defines.direction.west},{0,1,defines.direction.south},{0,-1,defines.direction.north}}; for pi=1,math.min(#patch,120) do local base=patch[pi].position; for _,d in ipairs(dirs) do local x=d[1]; local y=d[2]; local c={{'electric-mining-drill',{x=base.x,y=base.y},d[3],1},{'transport-belt',{x=base.x+2*x,y=base.y+2*y},d[3],0},{'transport-belt',{x=base.x+3*x,y=base.y+3*y},d[3],0},{'transport-belt',{x=base.x+4*x,y=base.y+4*y},d[3],0},{'inserter',{x=base.x+5*x,y=base.y+5*y},d[3],0},{'stone-furnace',{x=base.x+7*x,y=base.y+7*y},d[3],0},{'inserter',{x=base.x+9*x,y=base.y+9*y},d[3],0},{'transport-belt',{x=base.x+10*x,y=base.y+10*y},d[3],0},{'transport-belt',{x=base.x+11*x,y=base.y+11*y},d[3],0},{'transport-belt',{x=base.x+12*x,y=base.y+12*y},d[3],0},{'transport-belt',{x=base.x+13*x,y=base.y+13*y},d[3],0},{'transport-belt',{x=base.x+14*x,y=base.y+14*y},d[3],0},{'inserter',{x=base.x+15*x,y=base.y+15*y},d[3],0},{'wooden-chest',{x=base.x+16*x,y=base.y+16*y},d[3],0},{'wooden-chest',{x=base.x+7*x-y*3,y=base.y+7*y+x*3},d[3],0},{'inserter',{x=base.x+7*x-y*2,y=base.y+7*y+x*2},d[3],0},{'small-electric-pole',{x=base.x+2*x-y*2,y=base.y+2*y+x*2},d[3],0},{'small-electric-pole',{x=base.x+8*x-y*2,y=base.y+8*y+x*2},d[3],0}}; local good=true; for _,q in ipairs(c) do local existing=s.find_entity(q[1],q[2]); if q[4]==1 and existing and existing.valid then local ok,t=pcall(function() return existing.mining_target end); if not(ok and t and t.valid and t.name==resource) then good=false; break end elseif not existing and not s.can_place_entity{name=q[1],position=q[2],force=f,direction=q[3]} then good=false; break elseif existing and q[4]~=1 then good=false; break end end; if good then return 1 end end end; return 0 end; "
            "emit('resource_drill__iron',drill_count('iron-ore')); emit('resource_drill__copper',drill_count('copper-ore')); emit('reclaimable_furnace',s.count_entities_filtered{name='stone-furnace',force=f}); emit('reclaimable_chest',s.count_entities_filtered{name='wooden-chest',force=f}); emit('can_build_iron_line_runtime',line_layout('iron-ore')); emit('can_build_copper_line_runtime',line_layout('copper-ore')); rcon.print(marker..'|'..table.concat(out,'|')); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ERROR|'..tostring(err)) end"
        )

    def _collect_reachability_research(self, *, force: bool = False) -> None:
        if not force and (
            int(getattr(self, "global_step", 0)) - self._runtime_last_line_query_step
            < self.line_preflight_every
        ):
            return
        self._runtime_last_line_query_step = int(getattr(self, "global_step", 0))
        try:
            raw = str(
                self.instance.rcon_client.send_command(self._state_command_researchutufxwp()) or ""
            )
            line = next(
                (line for line in raw.splitlines() if Runtime_STATE_MARKER in line),
                "",
            )
            if not line:
                raise RuntimeError(f"runtime state marker missing: {raw[:400]!r}")
            payload = line.split(Runtime_STATE_MARKER, 1)[1].lstrip(": |")
            if payload.startswith("ERROR|"):
                raise RuntimeError(payload)
            for token in payload.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                self.feasibility_state[key.strip()] = safe_float(value.strip())
            self.feasibility_state["runtime_query_ok"] = True
            self.feasibility_state["runtime_query_error"] = ""
        except Exception as exc:
            self.feasibility_state["runtime_query_ok"] = False
            self.feasibility_state["runtime_query_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            # A failed live query must never leave the previous geometry result
            # actionable.  Otherwise a transient RCON failure can expose a stale
            # smelting action and repeatedly consume an invalid transition.
            self.feasibility_state["can_build_iron_line_runtime"] = 0.0
            self.feasibility_state["can_build_copper_line_runtime"] = 0.0

    def _collect_feasibility_state(self, *, force: bool = False):
        refresh_started_runtime = time.perf_counter() if force else None
        state = super()._collect_feasibility_state(force=force)
        self._collect_reachability_research(force=force)
        if force:
            self._runtime_last_live_refresh_step = int(getattr(self, "global_step", -1))
            self._runtime_live_refresh_count += 1
            if refresh_started_runtime is not None:
                self._runtime_live_refresh_seconds_total += max(0.0, time.perf_counter() - refresh_started_runtime)
        return state

    def _refresh_if_needed_research(self) -> bool:
        """Guarantee one authoritative refresh for the current transition.

        Returns True when a full live refresh was issued and False when a
        lower adapter layer already refreshed after the latest Factorio
        mutation in this same policy transition.
        """
        step = int(getattr(self, "global_step", -1))
        if self._runtime_last_live_refresh_step == step:
            self._runtime_live_refresh_reuse_count += 1
            self._sync_authoritative_entity_counts()
            self._collect_reachability_research(force=False)
            return False
        self._collect_feasibility_state(force=True)
        return True

    # ------------------------------------------------------------------
    # Smelting-line reclamation and robust power planning
    # ------------------------------------------------------------------
    def _runtime_reclaim_line_items_research(self, resource_name: str) -> Dict[str, Any]:
        marker = Runtime_RECLAIM_MARKER
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local f=p.force; local resource="
            + json.dumps(resource_name)
            + "; local reclaimed_drill=0; local reclaimed_furnace=0; local reclaimed_chest=0; "
            "local function can_insert_item(name,count) local ok,v=pcall(function() return p.can_insert{name=name,count=count} end); return (not ok) or v end; "
            "local function empty_into_player(e) for _,id in pairs(defines.inventory) do local ok,inv=pcall(function() return e.get_inventory(id) end); if ok and inv and inv.valid then for i=1,#inv do local st=inv[i]; if st and st.valid_for_read then local moved=p.insert{name=st.name,count=st.count}; if moved<st.count then return false end; st.clear() end end end end; return true end; "
            "local function registered_furnace(e) if storage and storage.fle_rl_runtime and storage.fle_rl_runtime.lines then for _,cell in pairs(storage.fle_rl_runtime.lines) do if cell and cell.furnace_position and math.abs(cell.furnace_position.x-e.position.x)<0.1 and math.abs(cell.furnace_position.y-e.position.y)<0.1 then return true end end end; return false end; "
            "if p.get_item_count('electric-mining-drill')<1 then for _,e in ipairs(s.find_entities_filtered{name='electric-mining-drill',force=f}) do local ok,t=pcall(function() return e.mining_target end); if ok and t and t.valid and t.name==resource and can_insert_item('electric-mining-drill',1) then e.destroy{raise_destroy=true}; p.insert{name='electric-mining-drill',count=1}; reclaimed_drill=1; break end end end; "
            "if p.get_item_count('stone-furnace')<1 then for _,e in ipairs(s.find_entities_filtered{name='stone-furnace',force=f}) do if not registered_furnace(e) and can_insert_item('stone-furnace',1) and empty_into_player(e) then e.destroy{raise_destroy=true}; p.insert{name='stone-furnace',count=1}; reclaimed_furnace=1; break end end end; "
            "while p.get_item_count('wooden-chest')<2 do local picked=false; for _,e in ipairs(s.find_entities_filtered{name='wooden-chest',force=f}) do if can_insert_item('wooden-chest',1) and empty_into_player(e) then e.destroy{raise_destroy=true}; p.insert{name='wooden-chest',count=1}; reclaimed_chest=reclaimed_chest+1; picked=true; break end end; if not picked then break end end; "
            "if p.get_item_count('electric-mining-drill')<1 then rcon.print(marker..'|ok=0|error=missing-resource-drill'); return end; if p.get_item_count('stone-furnace')<1 then rcon.print(marker..'|ok=0|error=missing-furnace'); return end; if p.get_item_count('wooden-chest')<2 then rcon.print(marker..'|ok=0|error=missing-chests'); return end; rcon.print(marker..'|ok=1|reclaimed_drill='..reclaimed_drill..'|reclaimed_furnace='..reclaimed_furnace..'|reclaimed_chest='..reclaimed_chest); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

    def _runtime_create_compact_line(self, resource_name: str) -> Dict[str, Any]:
        reclaimed = self._runtime_reclaim_line_items_research(resource_name)
        result = super()._runtime_create_compact_line(resource_name)
        result["reclaimed_drill"] = reclaimed.get("reclaimed_drill", "0")
        result["reclaimed_furnace"] = reclaimed.get("reclaimed_furnace", "0")
        result["reclaimed_chest"] = reclaimed.get("reclaimed_chest", "0")
        result["runtime_reuse_enabled"] = "1"
        return result

    def _runtime_connect_power_yqeibtv(
        self, target_names: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
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
        target_lua = "{" + ",".join(json.dumps(name) for name in names) + "}"
        marker = Runtime_POWER_MARKER
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ok=0|error=no-agent'); return end; local s=p.surface; local f=p.force; "
            "local engines=s.find_entities_filtered{name='steam-engine',force=f}; local engine=engines[1]; if not engine then rcon.print(marker..'|ok=0|error=missing-steam-engine'); return end; local function d(a,b) local dx=a.x-b.x; local dy=a.y-b.y; return math.sqrt(dx*dx+dy*dy) end; "
            # Select the nearest unpowered consumer. runtime selected the first
            # entity returned by the surface query, which could be a drill on
            # a remote ore patch even when a nearby assembler was available.
            "local target=nil; local target_distance=1e99; for _,n in ipairs(" + target_lua + ") do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not(ok and c) then local z=d(e.position,engine.position); if z<target_distance then target=e; target_distance=z end end end end; if not target then rcon.print(marker..'|ok=1|already_connected=1|created=0'); return end; "
            "local pole_name=(p.get_item_count('medium-electric-pole')>0 and 'medium-electric-pole') or 'small-electric-pole'; local wire=(pole_name=='medium-electric-pole') and 9 or 6; local supply=(pole_name=='medium-electric-pole') and 3.5 or 2.5; local poles=s.find_entities_filtered{name={'small-electric-pole','medium-electric-pole'},force=f}; "
            "local function nearby_existing(pos,r) local best=nil; local bd=1e9; for _,q in ipairs(poles) do local z=d(q.position,pos); if z<=r and z<bd then best=q.position; bd=z end end; return best end; "
            "local function pick(center,radius,previous) for ring=0,radius*2 do local rr=ring/2; for xi=-ring,ring do for yi=-ring,ring do if math.max(math.abs(xi),math.abs(yi))==ring then local pos={x=center.x+xi/2,y=center.y+yi/2}; if (not previous or d(previous,pos)<=wire) and s.can_place_entity{name=pole_name,position=pos,force=f} then return pos end end end end end end; "
            "local plan={}; local source=nearby_existing(engine.position,supply+0.75); if not source then source=pick(engine.position,4,nil); if not source then rcon.print(marker..'|ok=0|error=cannot-place-source-pole'); return end; plan[#plan+1]=source end; local receiver=nearby_existing(target.position,supply+0.75); if not receiver then receiver=pick(target.position,4,nil); if not receiver then rcon.print(marker..'|ok=0|error=cannot-place-receiver-pole'); return end end; "
            "local previous=source; local guard=0; while d(previous,receiver)>wire do guard=guard+1; if guard>160 then rcon.print(marker..'|ok=0|error=route-guard'); return end; local dist=d(previous,receiver); local ratio=math.min(0.82*wire/dist,1); local desired={x=previous.x+(receiver.x-previous.x)*ratio,y=previous.y+(receiver.y-previous.y)*ratio}; local pos=pick(desired,3,previous); if not pos or d(pos,receiver)>=dist-0.1 then rcon.print(marker..'|ok=0|error=cannot-place-intermediate-pole'); return end; plan[#plan+1]=pos; previous=pos end; if not nearby_existing(target.position,supply+0.75) then plan[#plan+1]=receiver end; "
            "if #plan>0 and p.get_item_count(pole_name)<#plan then rcon.print(marker..'|ok=0|error=missing-poles|need='..#plan); return end; local made={}; local function rollback(reason,refund) for _,q in ipairs(made) do if q.valid then q.destroy() end end; if refund>0 then p.insert{name=pole_name,count=refund} end; rcon.print(marker..'|ok=0|error='..reason..'|rolled_back='..#made); end; for _,pos in ipairs(plan) do local e=s.create_entity{name=pole_name,position=pos,force=f,raise_built=true,auto_connect=true}; if not e then rollback('create-pole',0); return end; made[#made+1]=e end; local removed=0; if #made>0 then removed=p.remove_item{name=pole_name,count=#made}; if removed<#made then rollback('inventory-remove-poles',removed); return end end; local connected=false; for _=1,3 do local ok,c=pcall(function() return target.valid and target.is_connected_to_electric_network() end); if ok and c then connected=true; break end end; if not connected then rollback('target-not-connected',removed); return end; rcon.print(marker..'|ok=1|target='..target.name..'|created='..#made..'|verified_connected=1'); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        # runtime: runtime exposed a very rare live-network settlement race over 24h.
        # The Lua transaction rolls back every pole and refunds the inventory
        # before reporting ``target-not-connected``.  Retry the complete live
        # transaction once; if the same fully rolled-back race occurs again,
        # classify it as a stale no-op.  Geometry, inventory, RCON, and all
        # other power failures remain hard errors.
        for attempt in (1, 2):
            raw = str(self.instance.rcon_client.send_command(command) or "")
            try:
                result = self._parse_runtime_result(raw, marker)
            except RuntimeError as exc:
                if "target-not-connected" not in str(exc):
                    raise
                self._collect_feasibility_state(force=True)
                if attempt == 1:
                    time.sleep(0.05)
                    continue
                # Record the post-rollback live topology.  runtime masks this
                # exact state until something material changes instead of
                # allowing repeated stale no-ops.
                self._runtime_power_block_signature = self._factory_power_signature_research()
                return {
                    "ok": "1",
                    "created": "0",
                    "stale": "1",
                    "reason": "target-not-connected",
                    "attempts": "2",
                    "runtime_race_safe": "1",
                    "runtime_power_state_blocked": "1",
                }
            result["attempts"] = str(attempt)
            self._runtime_power_block_signature = None
            return result
        raise RuntimeError("runtime power retry loop exited unexpectedly")

    def _output_chest_feasibility_research(self, *, force: bool = False) -> Dict[str, float]:
        """Live geometry predicate shared by the chest mask and runtime repair.

        The runtime failure occurred when the cached feasibility snapshot said a
        drill output tile was placeable, but a dropped item occupied that tile
        before the runtime runtime transaction executed.  runtime queries the exact
        ``surface.can_place_entity`` predicate used by the executor and caches
        it only while the world is paused at the same global step.
        """
        step = int(getattr(self, "global_step", 0))
        if not hasattr(self, "instance") or not getattr(self.instance, "rcon_client", None):
            return dict(self._runtime_output_chest_feasibility)
        if not force and self._runtime_output_chest_query_step == step:
            return dict(self._runtime_output_chest_feasibility)
        marker = "__FLE_RL_OUTPUT_CHEST_FEAS_runtime__"
        command = (
            "/sc local marker=" + json.dumps(marker) + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ERROR|no-agent'); return end; "
            "local s=p.surface; local f=p.force; local function count(name) local c=0; "
            "for _,e in ipairs(s.find_entities_filtered{name=name,force=f}) do local dt=e.drop_target; "
            "if not (dt and dt.valid and dt.name=='wooden-chest') and "
            "s.can_place_entity{name='wooden-chest',position=e.drop_position,force=f} then c=c+1 end end return c end; "
            "rcon.print(marker..'|burner='..count('burner-mining-drill')..'|electric='..count('electric-mining-drill')); "
            "end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ERROR|'..tostring(err)) end"
        )
        try:
            raw = str(self.instance.rcon_client.send_command(command) or "")
            line = next((ln for ln in raw.splitlines() if marker in ln), "")
            if not line or "|ERROR|" in line:
                raise RuntimeError(line or raw[:300])
            payload = line.split(marker, 1)[1].lstrip(": |")
            parsed: Dict[str, float] = {}
            for token in payload.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                parsed[key.strip()] = safe_float(value.strip())
            self._runtime_output_chest_feasibility = {
                "burner-mining-drill": parsed.get("burner", 0.0),
                "electric-mining-drill": parsed.get("electric", 0.0),
            }
            self._runtime_output_chest_query_step = step
        except Exception:
            # Fail closed for placement actions.  This is safer than exposing an
            # action whose geometry could not be verified.
            self._runtime_output_chest_feasibility = {
                "burner-mining-drill": 0.0,
                "electric-mining-drill": 0.0,
            }
            self._runtime_output_chest_query_step = step
        return dict(self._runtime_output_chest_feasibility)

    def _runtime_place_output_chest(self, *, drill_name: str) -> Dict[str, Any]:
        """runtime race-safe output-chest transaction.

        Retry once when the executor reports ``no-placeable-output``.  If a
        forced live geometry query confirms that no placement is now possible,
        classify the transition as a stale no-op rather than a runtime fault.
        Genuine inventory/create failures still propagate unchanged.
        """
        last: Dict[str, Any] = {}
        for attempt in (1, 2):
            try:
                result = dict(super()._runtime_place_output_chest(drill_name=drill_name))
            except RuntimeError as exc:
                text = str(exc)
                if "no-placeable-output" not in text:
                    raise
                result = {"ok": "0", "error": "no-placeable-output"}
            last = result
            if str(result.get("ok", "0")) == "1":
                result["attempts"] = str(attempt)
                result["runtime_geometry_safe"] = "1"
                self._runtime_output_chest_query_step = -10**9
                return result
            if str(result.get("error", "")) != "no-placeable-output":
                return result
            live = self._output_chest_feasibility_research(force=True)
            if safe_float(live.get(drill_name)) <= 0:
                return {
                    "ok": "1",
                    "created": "0",
                    "stale": "1",
                    "reason": "no-placeable-output",
                    "attempts": str(attempt),
                    "runtime_race_safe": "1",
                    "runtime_geometry_safe": "1",
                }
        return last

    def _runtime_insert_furnace(self, *, item_name: str, quantity: int = 1) -> Dict[str, Any]:
        """runtime stale-safe ore insertion using live furnace compatibility.

        The previous long run exposed a rare race where the mask observed a compatible furnace
        but Factorio changed the furnace recipe/input state before execution.
        Retry once, then force a live feasibility refresh.  If no compatible
        furnace remains, classify the transition as a stale no-op rather than
        charging the policy with an adapter fault.  Missing inventory and true
        insertion failures remain hard errors.
        """
        last: Dict[str, Any] = {}
        for attempt in (1, 2):
            try:
                result = dict(super()._runtime_insert_furnace(item_name=item_name, quantity=quantity))
            except RuntimeError as exc:
                if "no-compatible-furnace" not in str(exc):
                    raise
                result = {"ok": "0", "error": "no-compatible-furnace", "exception": str(exc)}
            last = result
            if str(result.get("ok", "0")) == "1":
                result["attempts"] = str(attempt)
                result["runtime_furnace_insert_safe"] = "1"
                return result
            if str(result.get("error", "")) != "no-compatible-furnace":
                return result
            self._collect_feasibility_state(force=True)
            key = "furnace_copper_input_space" if item_name == "copper-ore" else "furnace_iron_input_space"
            if safe_float(self.feasibility_state.get(key)) <= 0:
                return {
                    "ok": "1",
                    "inserted": "0",
                    "stale": "1",
                    "reason": "no-compatible-furnace",
                    "attempts": str(attempt),
                    "runtime_race_safe": "1",
                    "runtime_furnace_insert_safe": "1",
                }
        return last

    def _runtime_transfer_chest_to_furnace(self, *, item_name: str, quantity: int = 5) -> Dict[str, Any]:
        """runtime stale-safe chest-to-furnace transfer.

        The runtime six-hour SAC run exposed one isolated transfer race.  The mask
        can observe chest ore and compatible furnace capacity, then either can
        disappear before the runtime transaction.  Retry once and only convert
        a *confirmed* no-longer-feasible state into a benign stale no-op.
        Genuine transfer failures remain hard errors.
        """
        last: Dict[str, Any] = {}
        for attempt in (1, 2):
            try:
                result = dict(super()._runtime_transfer_chest_to_furnace(item_name=item_name, quantity=quantity))
            except RuntimeError as exc:
                text = str(exc)
                if "no-chest-item" not in text and "no-compatible-furnace" not in text:
                    raise
                reason = "no-chest-item" if "no-chest-item" in text else "no-compatible-furnace"
                result = {"ok": "0", "error": reason, "exception": text}
            last = result
            if str(result.get("ok", "0")) == "1":
                result["attempts"] = str(attempt)
                result["runtime_chest_transfer_safe"] = "1"
                return result
            reason = str(result.get("error", ""))
            if reason not in {"no-chest-item", "no-compatible-furnace"}:
                return result
            self._collect_feasibility_state(force=True)
            if item_name == "copper-ore":
                chest_key = "chest_copper_ore"
                space_key = "furnace_copper_input_space"
            else:
                chest_key = "chest_iron_ore"
                space_key = "furnace_iron_input_space"
            chest_available = safe_float(self.feasibility_state.get(chest_key)) > 0
            furnace_available = safe_float(self.feasibility_state.get(space_key)) > 0
            if not chest_available or not furnace_available:
                return {
                    "ok": "1",
                    "transferred": "0",
                    "stale": "1",
                    "reason": reason,
                    "attempts": str(attempt),
                    "runtime_race_safe": "1",
                    "runtime_chest_transfer_safe": "1",
                }
            if attempt == 1:
                time.sleep(0.03)
        return last

    # ------------------------------------------------------------------
    # runtime inventory-capacity telemetry
    # ------------------------------------------------------------------
    def _feasibility_command(self) -> str:
        """Extend the inherited live feasibility probe with exact insertable counts.

        ``can_insert`` only answers whether *some* amount fits.  The runtime/runtime
        SAC failure happened when five stone could still fit, so harvest_stone
        remained legal even though the ten-unit macro could not complete.
        Factorio's LuaInventory.get_insertable_count() gives the exact capacity
        for ordinary inventory items and is therefore the authoritative mask
        signal for harvest/craft output space.
        """
        command = super()._feasibility_command()
        capacity_items = tuple(dict.fromkeys((
            "coal", "stone", "iron-ore", "copper-ore",
            "stone-furnace", "lab", "iron-gear-wheel",
            "automation-science-pack", "electronic-circuit",
            "transport-belt", "inserter", "assembling-machine-1",
            *(spec.output_item for spec in CRAFT_BY_ACTION.values()),
        )))
        items_lua = "{" + ",".join(json.dumps(x) for x in capacity_items) + "}"
        marker_literal = json.dumps("__FLE_RL_FEASIBILITY_runtime__")
        needle = "rcon.print(" + marker_literal + "..'|'..table.concat(out,'|')); end"
        if needle not in command:
            raise RuntimeError("runtime could not extend the inherited feasibility command")
        addon = (
            "local cap_items=" + items_lua + "; local cap_inv=p.get_main_inventory and p.get_main_inventory() or nil; "
            "for _,n in ipairs(cap_items) do local ok,c=pcall(function() return cap_inv and cap_inv.get_insertable_count(n) or 0 end); "
            "out[#out+1]='insertable_count__'..n..'='..((ok and c) or 0); end; "
        )
        return command.replace(needle, addon + needle, 1)

    def _inventory_insertable_research(self, item: str) -> float:
        f = self.feasibility_state
        key = f"insertable_count__{item}"
        if key in f:
            return max(0.0, safe_float(f.get(key)))
        # Query failures retain the inherited permissive behaviour rather than
        # silently eliminating actions; the deployment guard remains fallback.
        return float("inf") if not bool(f.get("query_ok", False)) else 0.0

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        mask = super().get_action_mask().astype(np.int8)
        f = self.feasibility_state
        now = int(getattr(self, "global_step", 0))

        def set_valid(name: str, valid: bool) -> None:
            try:
                mask[self.action_index(name)] = int(bool(valid))
            except KeyError:
                pass

        if self.disable_redundant_drill_power_action:
            set_valid("connect_electric_drill_to_power", False)

        # These are real catalogue actions, so they must not be permanently
        # disabled.  The live feasibility query runs the same geometry and
        # complete pole-budget preflight used by the transactional planner.
        live = bool(f.get("runtime_query_ok", False))
        chest_geometry = self._output_chest_feasibility_research(force=False)
        set_valid(
            "place_chest_at_burner_output",
            self.inventory.get("wooden-chest", 0.0) >= 1
            and safe_float(chest_geometry.get("burner-mining-drill")) > 0,
        )
        set_valid(
            "place_chest_at_electric_output",
            self.inventory.get("wooden-chest", 0.0) >= 1
            and safe_float(chest_geometry.get("electric-mining-drill")) > 0,
        )
        set_valid("connect_factory_power", self._factory_power_action_feasible_research())
        set_valid(
            "connect_lab_to_power",
            self._target_power_route_feasible_research("lab")
            and safe_float(f.get("entity__lab")) > 0
            and safe_float(f.get("lab_connected_count_runtime")) <= 0,
        )
        set_valid(
            "connect_electric_drill_to_power",
            (not self.disable_redundant_drill_power_action)
            and self._target_power_route_feasible_research("drill")
            and safe_float(f.get("entity__electric-mining-drill")) > 0,
        )
        # Use the exact same reachable, unoccupied-tile predicate as the runtime
        # runtime transaction.  A global surface count is insufficient because
        # the remaining ore may be underneath an already placed drill.
        for action_name, resource_name in Runtime_HARVEST_RESOURCES.items():
            try:
                inherited_valid = bool(mask[self.action_index(action_name)])
            except KeyError:
                continue
            set_valid(
                action_name,
                inherited_valid
                and live
                and safe_float(f.get(f"harvest_reachable__{resource_name}")) > 0,
            )

        # runtime: exact inventory-capacity masks.  Harvest may legally take a
        # partial batch when at least one unit fits, but never runs when the
        # inventory has zero capacity.  Hand-craft actions are hidden when
        # their output item cannot fit, preventing the runtime stone-furnace loop.
        for action_name, resource_name in Runtime_HARVEST_RESOURCES.items():
            try:
                inherited_valid = bool(mask[self.action_index(action_name)])
            except KeyError:
                continue
            set_valid(
                action_name,
                inherited_valid and self._inventory_insertable_research(resource_name) >= 1.0,
            )

        craft_outputs = {
            "craft_stone_furnace": "stone-furnace",
            "craft_lab": "lab",
            "craft_iron_gears": "iron-gear-wheel",
            "craft_automation_science": "automation-science-pack",
            "craft_electronic_circuits": "electronic-circuit",
            "craft_transport_belts": "transport-belt",
            "craft_inserters": "inserter",
            "craft_assembling_machine": "assembling-machine-1",
        }
        craft_outputs.update({name: spec.output_item for name, spec in CRAFT_BY_ACTION.items()})
        for action_name, output_item in craft_outputs.items():
            try:
                inherited_valid = bool(mask[self.action_index(action_name)])
            except KeyError:
                continue
            set_valid(
                action_name,
                inherited_valid and self._inventory_insertable_research(output_item) >= 1.0,
            )

        # runtime: source science alone is insufficient.  Expose the lab-feed
        # action only when a live lab can accept at least one available pack.
        try:
            inherited_feed = bool(mask[self.action_index("feed_labs_all_science")])
        except KeyError:
            inherited_feed = False
        if inherited_feed:
            lab_feed = self._lab_feed_feasibility_research(force=False)
            set_valid(
                "feed_labs_all_science",
                safe_float(lab_feed.get("can_feed")) > 0,
            )

        current_research = str(
            self.infrastructure_state.get("current_research", "") or ""
        )
        science_available = (
            safe_float(f.get("lab_red_science_count"))
            + safe_float(f.get("buffered_automation_science_pack_runtime"))
            + safe_float(f.get("lab_science_count"))
        ) > 0
        wait_budget_open = (
            self._runtime_unproductive_wait_streak
            < self.max_unproductive_wait_streak
        )
        set_valid(
            "wait_research",
            live
            and bool(current_research)
            and safe_float(f.get("lab_working_count_runtime")) > 0
            and science_available
            and wait_budget_open,
        )
        try:
            inherited_wait = bool(mask[self.action_index("wait")])
        except KeyError:
            inherited_wait = False
        set_valid("wait", inherited_wait and wait_budget_open)
        set_valid(
            "collect_engine_units",
            live
            and safe_float(f.get("buffered_item__engine-unit")) > 0
            and safe_float(f.get("can_collect_engine_unit")) > 0,
        )

        active_power = safe_float(f.get("power_network_exists")) > 0
        line_common = (
            bool(f.get("runtime_query_ok", False))
            and active_power
            and self.inventory.get("transport-belt", 0.0) >= 8
            and self.inventory.get("inserter", 0.0) >= 4
            and self.inventory.get("small-electric-pole", 0.0) >= 2
            and self.inventory.get("coal", 0.0) >= 10
            and (
                self.inventory.get("stone-furnace", 0.0)
                + safe_float(f.get("reclaimable_furnace"))
            )
            >= 1
            and (
                self.inventory.get("wooden-chest", 0.0)
                + safe_float(f.get("reclaimable_chest"))
            )
            >= 2
        )
        set_valid(
            "build_iron_smelting_line",
            line_common
            and (
                self.inventory.get("electric-mining-drill", 0.0)
                + safe_float(f.get("resource_drill__iron"))
            )
            >= 1
            and safe_float(f.get("can_build_iron_line_runtime")) > 0
            and safe_float(f.get("iron_smelting_line_count")) < 2,
        )
        set_valid(
            "build_copper_smelting_line",
            line_common
            and (
                self.inventory.get("electric-mining-drill", 0.0)
                + safe_float(f.get("resource_drill__copper"))
            )
            >= 1
            and safe_float(f.get("can_build_copper_line_runtime")) > 0
            and safe_float(f.get("copper_smelting_line_count")) < 2,
        )

        for name, until in self._runtime_action_cooldown_until.items():
            if now < int(until):
                set_valid(name, False)
        set_valid("noop", True)
        return mask

    # ------------------------------------------------------------------
    # Gym API and accounting
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
            self._runtime_action_cooldown_until.clear()
            self.horizon_lab_ingredient_reward = 0.0
            self.horizon_runtime_manual_craft_units = 0.0
            self._runtime_unproductive_wait_streak = 0
            self._runtime_power_block_signature = None
            self._runtime_power_mask_suppressions = 0
        if self.deterministic_headless_research_setup:
            setup = self._runtime_prepare_headless_research_research()
            if setup.get("ok") != "1":
                raise RuntimeError(f"deterministic headless setup failed: {setup}")
        self._collect_reachability_research(force=True)
        self._collect_feasibility_state(force=True)
        self._runtime_seen_lab_research = set(self._lab_researched_set())
        self._runtime_research_progress = {}
        self._last_science_readiness_runtime = self._science_readiness_research()
        ingredient_state = self._lab_ingredient_state_research()
        self._last_lab_ingredient_fraction_runtime = ingredient_state["fraction"]
        final = dict(info)
        final.update(self._base_info())
        final["lab_ingredient_state_runtime"] = ingredient_state
        final["action_mask"] = self.action_masks()
        final["environment_schema_path"] = str(
            Path(self.run_dir) / "environment_schema_runtime.json"
        )
        return self._observation(), final

    def _account_runtime_manual_craft_research(
        self, action_name: str, info: Dict[str, Any]
    ) -> Dict[str, float]:
        spec = CRAFT_BY_ACTION.get(action_name)
        runtime = info.get("runtime_action_extended")
        if spec is None or not isinstance(runtime, Mapping):
            return {}
        started = max(0.0, safe_float(runtime.get("started")))
        expected = started * max(0.0, float(spec.output_per_craft))
        if expected <= 0:
            return {}
        existing_delta = dict(info.get("manual_crafted_delta", {}))
        already = max(0.0, safe_float(existing_delta.get(spec.output_item)))
        missing = max(0.0, expected - already)
        if missing <= 0:
            return {}
        weighted = (
            self._strict_weight({spec.output_item: missing})
            if hasattr(self, "_strict_weight")
            else missing
        )
        self.manual_crafted_total += weighted
        self.lifetime_manual_crafted_total += weighted
        self.manual_crafted_by_item[spec.output_item] += missing
        self.lifetime_manual_crafted_by_item[spec.output_item] += missing
        if hasattr(self, "manual_crafted_units_total"):
            self.manual_crafted_units_total += missing
        if hasattr(self, "lifetime_manual_crafted_units_total"):
            self.lifetime_manual_crafted_units_total += missing
        self.lifetime_runtime_manual_craft_units += missing
        self.horizon_runtime_manual_craft_units += missing
        existing_delta[spec.output_item] = already + missing
        info["manual_crafted_delta"] = existing_delta
        info["runtime_manual_crafted_units_runtime"] = missing
        return {spec.output_item: missing}

    def step(self, action: int):
        adapter_start_runtime = time.perf_counter()
        refresh_count_before_runtime = int(getattr(self, "_runtime_live_refresh_count", 0))
        refresh_seconds_before_runtime = float(getattr(self, "_runtime_live_refresh_seconds_total", 0.0))
        before_fraction = self._lab_ingredient_state_research()["fraction"]
        before_science_readiness = self._science_readiness_research()
        action_index = int(action)
        action_name = (
            self.actions[action_index].name
            if 0 <= action_index < len(self.actions)
            else "invalid-index"
        )
        observation, reward, terminated, truncated, info = super().step(action)
        del observation
        info = dict(info)

        # Surface race-safe no-ops explicitly in telemetry.  They are not
        # execution faults, but long runs should still reveal how often a
        # sampled live mask went stale before the runtime transaction.
        for runtime_key in ("runtime_action_inventory", "runtime_action_factory", "runtime_action_extended"):
            runtime_result = info.get(runtime_key)
            if not isinstance(runtime_result, Mapping):
                continue
            if str(runtime_result.get("stale", "0")) != "1":
                continue
            if str(runtime_result.get("runtime_race_safe", "0")) != "1":
                continue
            stale_reason = str(runtime_result.get("reason", "stale"))
            self.runtime_stale_race_noops[action_name] += 1
            info["runtime_stale_action"] = action_name
            info["runtime_stale_reason"] = stale_reason
            info["runtime_stale_race_noops"] = dict(self.runtime_stale_race_noops)
            break

        authoritative_refresh_start_runtime = time.perf_counter()
        refresh_issued_runtime = self._refresh_if_needed_research()
        authoritative_refresh_s_runtime = time.perf_counter() - authoritative_refresh_start_runtime
        parts = dict(info.get("reward_parts", {}))
        newly_completed = list(info.get("new_lab_research", []))
        live_progress = max(
            0.0,
            min(
                1.0,
                safe_float(self.infrastructure_state.get("research_progress")),
            ),
        )

        def adjust_component(name: str, delta: float) -> None:
            amount = float(delta)
            if abs(amount) <= 1e-12:
                return
            reward_counters = (
                self.lifetime_canonical_reward_components,
                self.horizon_canonical_reward_components,
            )
            for counter in reward_counters:
                counter[name] += amount
                counter["total"] += amount
            self.lifetime_reward_components[name] += amount

        # Research progress must be comparable across actions that advance one
        # or five simulated seconds.  Normalising the inherited component by
        # elapsed Factorio time removes the former reward advantage of
        # repeatedly choosing wait_research.  A conservative cap prevents a
        # delayed telemetry refresh from creating another isolated PPO spike.
        elapsed_seconds = max(1.0, safe_float(info.get("simulated_seconds_elapsed")))
        raw_progress_reward = max(0.0, safe_float(parts.get("research_progress")))
        normalized_progress_reward = raw_progress_reward / elapsed_seconds
        if self.research_progress_reward_cap > 0:
            normalized_progress_reward = min(
                normalized_progress_reward,
                self.research_progress_reward_cap,
            )
        progress_adjustment = normalized_progress_reward - raw_progress_reward
        if abs(progress_adjustment) > 1e-12:
            reward = float(reward + progress_adjustment)
            self.episode_return += progress_adjustment
            adjust_component("research_progress", progress_adjustment)
        parts["research_progress"] = float(normalized_progress_reward)

        # The owning runtime layer should now emit one completion reward.  Keep a
        # final fail-safe cap here so an inherited regression can never credit
        # the same technology more than once in a policy transition.
        raw_completion_reward = max(
            0.0, safe_float(parts.get("lab_research_completion"))
        )
        completion_count = max(1, len(newly_completed))
        completion_cap = (
            completion_count * float(self.lab_research_completion_reward)
            if raw_completion_reward > 0
            else 0.0
        )
        completion_reward = min(raw_completion_reward, completion_cap)
        completion_adjustment = completion_reward - raw_completion_reward
        if abs(completion_adjustment) > 1e-12:
            reward = float(reward + completion_adjustment)
            self.episode_return += completion_adjustment
            adjust_component("lab_research_completion", completion_adjustment)
        parts["lab_research_completion"] = float(completion_reward)

        ingredient_state = self._lab_ingredient_state_research()
        after_fraction = ingredient_state["fraction"]
        # runtime already owns ingredient shaping.  Reuse its value instead of
        # applying the same potential delta a second time in runtime.
        ingredient_reward = safe_float(parts.get("lab_ingredient_progress"))

        science_readiness = self._science_readiness_research()
        science_reward = self.science_readiness_reward * (
            science_readiness - before_science_readiness
        )
        if abs(science_reward) > 1e-12:
            reward = float(reward + science_reward)
            self.episode_return += science_reward
            self._add_reward_component_research("science_readiness", science_reward)

        machine_units_delta = max(
            0.0, safe_float(info.get("machine_output_units_delta"))
        )
        idle_action = action_name in {"noop", "wait", "wait_research"}
        productive_idle = bool(
            machine_units_delta > 0
            or normalized_progress_reward > 1e-9
            or completion_reward > 0
            or science_reward > 1e-9
        )
        repeated_idle = 0.0
        if idle_action:
            if productive_idle:
                self._runtime_unproductive_wait_streak = 0
            else:
                self._runtime_unproductive_wait_streak += 1
                repeated_idle = min(
                    0.50,
                    self.repeated_idle_penalty
                    * max(0, self._runtime_unproductive_wait_streak - 1),
                )
        else:
            self._runtime_unproductive_wait_streak = 0
        if repeated_idle > 0:
            reward = float(reward - repeated_idle)
            self.episode_return -= repeated_idle
            adjust_component("repeated_idle_penalty", -repeated_idle)
        parts["repeated_idle_penalty"] = float(-repeated_idle)

        # Environment-side failures are temporarily masked to prevent a stale
        # geometry state from consuming hundreds of consecutive transitions.
        if info.get("environment_rejection") or info.get("action_error"):
            if action_name in {
                "connect_factory_power",
                "connect_electric_drill_to_power",
                "build_iron_smelting_line",
                "build_copper_smelting_line",
                "place_furnace_near_drill",
                *Runtime_HARVEST_RESOURCES.keys(),
            }:
                self._runtime_action_cooldown_until[action_name] = (
                    int(getattr(self, "global_step", 0))
                    + self.environment_rejection_cooldown_steps
                )

        parts["lab_ingredient_progress"] = float(ingredient_reward)
        parts["science_readiness"] = safe_float(parts.get("science_readiness")) + float(science_reward)
        for key in ("total", "residual"):
            parts.pop(key, None)
        parts["residual"] = float(
            reward - sum(safe_float(value) for value in parts.values())
        )
        parts["total"] = float(reward)

        final_mask_start_runtime = time.perf_counter()
        final_mask_runtime = self.action_masks()
        final_mask_s_runtime = time.perf_counter() - final_mask_start_runtime

        info_build_start_runtime = time.perf_counter()
        final = dict(info)
        final.update(self._base_info())
        final.update(
            {
                "environment_version": Runtime_VERSION,
                "reward": float(reward),
                "reward_parts": parts,
                "lab_ingredient_state_runtime": ingredient_state,
                "lab_ingredient_fraction_before": float(before_fraction),
                "lab_ingredient_fraction_after": float(after_fraction),
                "reward_lab_ingredient_progress": float(ingredient_reward),
                "reward_lab_research_completion": float(completion_reward),
                "reward_research_progress": float(normalized_progress_reward),
                "reward_research_progress_raw_runtime": float(raw_progress_reward),
                "reward_research_progress_normalized_runtime": float(normalized_progress_reward),
                "reward_repeated_idle_penalty_runtime": float(-repeated_idle),
                "new_lab_research": newly_completed,
                "research_progress": live_progress,
                "science_readiness": science_readiness,
                "reward_science_readiness": science_reward,
                "lifetime_lab_ingredient_reward": float(
                    self.lifetime_lab_ingredient_reward
                ),
                "horizon_lab_ingredient_reward": float(
                    self.horizon_lab_ingredient_reward
                ),
                "action_cooldowns_runtime": dict(self._runtime_action_cooldown_until),
                "research_bootstrap_gate": self.research_bootstrap_gate_research(),
                "research_bootstrap_available": bool(self._research_bootstrap_feasible_research()),
                "entity_counts": dict(self.entity_counts),
                "authoritative_entity_counts": dict(self.entity_counts),
                "unproductive_wait_streak_runtime": int(
                    self._runtime_unproductive_wait_streak
                ),
                "action_mask": final_mask_runtime,
            }
        )
        info_build_s_runtime = time.perf_counter() - info_build_start_runtime
        self._last_lab_ingredient_fraction_runtime = after_fraction
        self._last_science_readiness_runtime = science_readiness

        observation_start_runtime = time.perf_counter()
        final_observation_runtime = self._observation()
        final_observation_s_runtime = time.perf_counter() - observation_start_runtime
        adapter_total_s_runtime = time.perf_counter() - adapter_start_runtime
        base_timing_runtime = dict(info.get("timing", {}) or {})
        base_total_s_runtime = safe_float(base_timing_runtime.get("total_s"))
        final["adapter_latency_runtime"] = {
            "adapter_total_s": float(adapter_total_s_runtime),
            "base_step_total_s": float(base_total_s_runtime),
            "base_action_eval_s": float(safe_float(base_timing_runtime.get("action_eval_s"))),
            "base_settle_s": float(safe_float(base_timing_runtime.get("settle_s"))),
            "base_state_s": float(safe_float(base_timing_runtime.get("state_s"))),
            "authoritative_refresh_s": float(authoritative_refresh_s_runtime),
            "info_build_s": float(info_build_s_runtime),
            "final_mask_s": float(final_mask_s_runtime),
            "final_observation_s": float(final_observation_s_runtime),
            "adapter_wrapper_s": float(max(0.0, adapter_total_s_runtime - base_total_s_runtime)),
            "top_refresh_issued": float(bool(refresh_issued_runtime)),
            "live_refreshes_this_step": float(max(0, int(self._runtime_live_refresh_count) - refresh_count_before_runtime)),
            "live_refresh_total_s": float(max(0.0, float(self._runtime_live_refresh_seconds_total) - refresh_seconds_before_runtime)),
            "live_refresh_count_total": float(self._runtime_live_refresh_count),
            "live_refresh_reuse_count_total": float(self._runtime_live_refresh_reuse_count),
        }
        final["adapter_latency_version"] = "runtime"
        final["runtime_refresh_coalescing"] = {
            "top_refresh_issued": bool(refresh_issued_runtime),
            "live_refresh_count_total": int(self._runtime_live_refresh_count),
            "live_refresh_reuse_count_total": int(self._runtime_live_refresh_reuse_count),
            "last_live_refresh_step": int(self._runtime_last_live_refresh_step),
        }
        return final_observation_runtime, float(reward), terminated, truncated, final

    def _base_info(self) -> Dict[str, Any]:
        self._sync_authoritative_entity_counts()
        info = super()._base_info()
        info.update(
            {
                "environment_version": Runtime_VERSION,
                "entity_counts": dict(self.entity_counts),
                "authoritative_entity_counts": dict(self.entity_counts),
                "lab_ingredient_state_runtime": self._lab_ingredient_state_research(),
                "lifetime_lab_ingredient_reward": float(
                    self.lifetime_lab_ingredient_reward
                ),
                "horizon_lab_ingredient_reward": float(
                    self.horizon_lab_ingredient_reward
                ),
                "lifetime_runtime_manual_craft_units": float(
                    self.lifetime_runtime_manual_craft_units
                ),
                "horizon_runtime_manual_craft_units": float(
                    self.horizon_runtime_manual_craft_units
                ),
                "action_cooldowns_runtime": dict(self._runtime_action_cooldown_until),
                "unproductive_wait_streak_runtime": int(
                    self._runtime_unproductive_wait_streak
                ),
                "max_unproductive_wait_streak": int(
                    self.max_unproductive_wait_streak
                ),
                "repeated_idle_penalty": float(self.repeated_idle_penalty),
                "research_progress_reward_cap": float(
                    self.research_progress_reward_cap
                ),
                "inventory_full": bool(self.feasibility_state.get("inventory_full", False)),
                "inventory_insertable_counts_runtime": {
                    str(k).split("insertable_count__", 1)[1]: safe_float(v)
                    for k, v in self.feasibility_state.items()
                    if str(k).startswith("insertable_count__")
                },
            }
        )
        return info


FactorioOpenEndedEnvResearchProgress = FactorioOpenEndedEnvResearch


__all__ = [
    "FactorioOpenEndedEnvResearchProgress",
    "FactorioOpenEndedEnvResearch",
    "Runtime_VERSION",
    "Runtime_ACTIONS",
    "Runtime_PROGRESSION_REWARDS",
    "Runtime_RESEARCH_TARGETS",
    "Runtime_TRACKED_ITEMS",
]
