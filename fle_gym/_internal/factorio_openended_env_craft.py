from __future__ import annotations

"""runtime shared Factorio environment with research reachability and robust macros.

runtime keeps the runtime fixed 124-action/266-observation contract so the exact same
MDP interface can be used by MaskablePPO, categorical/discretised SAC, and
MAP-Elites.  It changes environment semantics rather than hiding actions:

* a dense, inventory-based laboratory ingredient potential rewards balanced
  accumulation of gears, circuits, and belts and penalises consuming them;
* partial-batch hand crafting is accounted from the runtime-confirmed quantity;
* smelting-line macros may reclaim previously placed drills, furnaces, and
  chests instead of becoming permanently unreachable;
* the redundant drill-only power action is masked and the factory-wide power
  action uses a transactional pole-chain planner;
* power/line failures receive a short action cooldown to prevent repeated
  environment-side rejections;
* every info payload and final summary can use one authoritative live entity
  dictionary.

There is still no product or research terminal goal.  Training horizons are
Gymnasium time-limit truncations only; set ``training_horizon_steps=0`` for a
single persistent evaluation factory.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from factorio_openended_env_machine import (
    FactorioOpenEndedEnvMachine,
    Runtime_ACTIONS,
    Runtime_PROGRESSION_REWARDS,
    Runtime_RESEARCH_TARGETS,
    Runtime_TRACKED_ITEMS,
)
from factorio_openended_env_research_yqeibtv import CRAFT_BY_ACTION, Runtime_LIVE_ENTITY_NAMES
from factorio_transition_env import json_safe, safe_float, spaces


Runtime_VERSION = "21.0"
Runtime_STATE_MARKER = "__FLE_RL_STATE_runtime__"
Runtime_RECLAIM_MARKER = "__FLE_RL_RECLAIM_runtime__"
Runtime_POWER_MARKER = "__FLE_RL_POWER_runtime__"


class FactorioOpenEndedEnvCraft(FactorioOpenEndedEnvMachine):
    """Algorithm-neutral runtime environment."""

    def __init__(
        self,
        *,
        lab_ingredient_potential_reward: float = 21.0,
        environment_rejection_cooldown_steps: int = 24,
        line_preflight_every: int = 4,
        disable_redundant_drill_power_action: bool = True,
        research_progress_reward: float = 80.0,
        lab_research_completion_reward: float = 121.0,
        research_activity_reward: float = 0.20,
        automated_output_reward_scale: float = 0.35,
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

        kwargs.setdefault("research_progress_reward", research_progress_reward)
        kwargs.setdefault(
            "lab_research_completion_reward", lab_research_completion_reward
        )
        kwargs.setdefault("research_activity_reward", research_activity_reward)
        kwargs.setdefault(
            "automated_output_reward_scale", automated_output_reward_scale
        )
        super().__init__(**kwargs)
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
            "observation_shape": list(self.observation_space.shape),
            "lab_ingredient_potential_reward": self.lab_ingredient_potential_reward,
            "environment_rejection_cooldown_steps": self.environment_rejection_cooldown_steps,
            "line_preflight_every": self.line_preflight_every,
            "disable_redundant_drill_power_action": self.disable_redundant_drill_power_action,
            "research_progress_reward": float(
                getattr(self, "research_progress_reward", 80.0)
            ),
            "lab_research_completion_reward": float(
                getattr(self, "lab_research_completion_reward", 121.0)
            ),
            "research_activity_reward": float(
                getattr(self, "research_activity_reward", 0.20)
            ),
            "automated_output_reward_scale": float(
                getattr(self, "automated_output_reward_scale", 0.35)
            ),
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
            "power": "factory-wide transactional pole chain; redundant drill-only action masked",
            "manual_crafting": "runtime-confirmed partial-batch quantities",
            "terminal_goal": None,
        }
        return schema

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
        state = super()._collect_feasibility_state(force=force)
        self._collect_reachability_research(force=force)
        return state

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
            "local target=nil; for _,n in ipairs(" + target_lua + ") do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not(ok and c) then target=e; break end end; if target then break end end; if not target then rcon.print(marker..'|ok=1|already_connected=1|created=0'); return end; "
            "local engines=s.find_entities_filtered{name='steam-engine',force=f}; local engine=engines[1]; if not engine then rcon.print(marker..'|ok=0|error=missing-steam-engine'); return end; local pole_name=(p.get_item_count('medium-electric-pole')>0 and 'medium-electric-pole') or 'small-electric-pole'; local wire=(pole_name=='medium-electric-pole') and 9 or 6; local supply=(pole_name=='medium-electric-pole') and 3.5 or 2.5; local poles=s.find_entities_filtered{name={'small-electric-pole','medium-electric-pole'},force=f}; local function d(a,b) local dx=a.x-b.x; local dy=a.y-b.y; return math.sqrt(dx*dx+dy*dy) end; "
            "local function nearby_existing(pos,r) local best=nil; local bd=1e9; for _,q in ipairs(poles) do local z=d(q.position,pos); if z<=r and z<bd then best=q.position; bd=z end end; return best end; "
            "local function pick(center,radius,previous) for ring=0,radius*2 do local rr=ring/2; for xi=-ring,ring do for yi=-ring,ring do if math.max(math.abs(xi),math.abs(yi))==ring then local pos={x=center.x+xi/2,y=center.y+yi/2}; if (not previous or d(previous,pos)<=wire) and s.can_place_entity{name=pole_name,position=pos,force=f} then return pos end end end end end end; "
            "local plan={}; local source=nearby_existing(engine.position,supply+0.75); if not source then source=pick(engine.position,4,nil); if not source then rcon.print(marker..'|ok=0|error=cannot-place-source-pole'); return end; plan[#plan+1]=source end; local receiver=nearby_existing(target.position,supply+0.75); if not receiver then receiver=pick(target.position,4,nil); if not receiver then rcon.print(marker..'|ok=0|error=cannot-place-receiver-pole'); return end end; "
            "local previous=source; local guard=0; while d(previous,receiver)>wire do guard=guard+1; if guard>160 then rcon.print(marker..'|ok=0|error=route-guard'); return end; local dist=d(previous,receiver); local ratio=math.min(0.82*wire/dist,1); local desired={x=previous.x+(receiver.x-previous.x)*ratio,y=previous.y+(receiver.y-previous.y)*ratio}; local pos=pick(desired,3,previous); if not pos or d(pos,receiver)>=dist-0.1 then rcon.print(marker..'|ok=0|error=cannot-place-intermediate-pole'); return end; plan[#plan+1]=pos; previous=pos end; if not nearby_existing(target.position,supply+0.75) then plan[#plan+1]=receiver end; "
            "if #plan>0 and p.get_item_count(pole_name)<#plan then rcon.print(marker..'|ok=0|error=missing-poles|need='..#plan); return end; local made={}; local function rollback(reason,refund) for _,q in ipairs(made) do if q.valid then q.destroy() end end; if refund>0 then p.insert{name=pole_name,count=refund} end; rcon.print(marker..'|ok=0|error='..reason..'|rolled_back='..#made); end; for _,pos in ipairs(plan) do local e=s.create_entity{name=pole_name,position=pos,force=f,raise_built=true,auto_connect=true}; if not e then rollback('create-pole',0); return end; made[#made+1]=e end; local removed=0; if #made>0 then removed=p.remove_item{name=pole_name,count=#made}; if removed<#made then rollback('inventory-remove-poles',removed); return end end; local connected=false; for _=1,3 do local ok,c=pcall(function() return target.valid and target.is_connected_to_electric_network() end); if ok and c then connected=true; break end end; if not connected then rollback('target-not-connected',removed); return end; rcon.print(marker..'|ok=1|target='..target.name..'|created='..#made..'|verified_connected=1'); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ok=0|error='..tostring(err)) end"
        )
        raw = str(self.instance.rcon_client.send_command(command) or "")
        return self._parse_runtime_result(raw, marker)

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

        # The factory-wide action is the single power-routing capability.
        set_valid(
            "connect_factory_power",
            bool(f.get("runtime_query_ok", False))
            and safe_float(f.get("power_unpowered_consumers")) > 0
            and safe_float(f.get("power_generation_capacity")) > 0
            and (
                self.inventory.get("small-electric-pole", 0.0)
                + self.inventory.get("medium-electric-pole", 0.0)
            )
            > 0,
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
        self._collect_reachability_research(force=True)
        self._collect_feasibility_state(force=True)
        self._runtime_seen_lab_research = set(self._lab_researched_set())
        self._runtime_research_progress = {}
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
        before_fraction = self._lab_ingredient_state_research()["fraction"]
        action_index = int(action)
        action_name = (
            self.actions[action_index].name
            if 0 <= action_index < len(self.actions)
            else "invalid-index"
        )
        observation, reward, terminated, truncated, info = super().step(action)
        del observation
        info = dict(info)

        self._account_runtime_manual_craft_research(action_name, info)
        self._sync_authoritative_entity_counts()
        # runtime: runtime/runtime already performs an authoritative post-action refresh.
        # Reuse it when it belongs to this same global step; forcing another
        # complete RCON state scan here was redundant and dominated latency.
        # If a lower layer failed to refresh, fail safe by issuing one now.
        if hasattr(self, "_refresh_if_needed_research"):
            self._refresh_if_needed_research()
        else:
            self._collect_feasibility_state(force=True)
            self._collect_reachability_research(force=False)

        parts = dict(info.get("reward_parts", {}))
        live_lab = set(self._lab_researched_set())
        newly_completed = sorted(live_lab - self._runtime_seen_lab_research)
        inherited_completion_reward = safe_float(
            parts.get("lab_research_completion")
        )
        expected_completion_reward = (
            len(newly_completed) * float(self.lab_research_completion_reward)
        )
        # runtime already replaces the legacy completion reward with the canonical
        # lab-research component when it observes the completion in the same
        # transition.  runtime is a live-refresh fallback for completions that the
        # inherited periodic query misses; it must add only the missing amount.
        # The former unconditional addition credited every ordinary completion
        # twice and produced 240-point PPO reward spikes.
        completion_reward = 0.0
        if expected_completion_reward > inherited_completion_reward:
            completion_reward = (
                expected_completion_reward - inherited_completion_reward
            )
            reward = float(reward + completion_reward)
            self.episode_return += completion_reward
            self._add_reward_component_research("lab_research_completion", completion_reward)
        self._runtime_seen_lab_research.update(live_lab)

        current_research = str(self.infrastructure_state.get("current_research", "") or "")
        live_progress = max(0.0, min(1.0, safe_float(self.infrastructure_state.get("research_progress"))))
        previous_progress = self._runtime_research_progress.get(current_research, 0.0)
        inherited_progress_reward = max(
            0.0, safe_float(parts.get("research_progress"))
        )
        expected_progress_reward = 0.0
        if current_research and live_progress > previous_progress:
            expected_progress_reward = (
                (live_progress - previous_progress)
                * float(self.research_progress_reward)
            )
        # As with completion rewards, inherited live accounting owns the
        # normal case.  Add only a genuinely missing asynchronous delta.
        progress_reward = 0.0
        if expected_progress_reward > inherited_progress_reward:
            progress_reward = expected_progress_reward - inherited_progress_reward
            reward = float(reward + progress_reward)
            self.episode_return += progress_reward
            self._add_reward_component_research("research_progress", progress_reward)
        if current_research:
            self._runtime_research_progress[current_research] = live_progress

        ingredient_state = self._lab_ingredient_state_research()
        after_fraction = ingredient_state["fraction"]
        ingredient_reward = 0.0
        if self.entity_counts.get("lab", 0.0) > 0:
            ingredient_reward = self.lab_ingredient_potential_reward * (
                after_fraction - before_fraction
            )
        if abs(ingredient_reward) > 1e-12:
            reward = float(reward + ingredient_reward)
            self.episode_return += ingredient_reward
            self._add_reward_component_research(
                "lab_ingredient_progress", ingredient_reward
            )
            self.lifetime_lab_ingredient_reward += ingredient_reward
            self.horizon_lab_ingredient_reward += ingredient_reward

        # Environment-side failures are temporarily masked to prevent a stale
        # geometry state from consuming hundreds of consecutive transitions.
        if info.get("environment_rejection") or info.get("action_error"):
            if action_name in {
                "connect_factory_power",
                "connect_electric_drill_to_power",
                "build_iron_smelting_line",
                "build_copper_smelting_line",
                "place_furnace_near_drill",
            }:
                self._runtime_action_cooldown_until[action_name] = (
                    int(getattr(self, "global_step", 0))
                    + self.environment_rejection_cooldown_steps
                )

        parts["lab_ingredient_progress"] = float(ingredient_reward)
        parts["lab_research_completion"] = float(parts.get("lab_research_completion", 0.0)) + completion_reward
        parts["research_progress"] = float(parts.get("research_progress", 0.0)) + progress_reward
        for key in ("total", "residual"):
            parts.pop(key, None)
        parts["residual"] = float(
            reward - sum(safe_float(value) for value in parts.values())
        )
        parts["total"] = float(reward)

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
                "reward_lab_research_completion": float(parts["lab_research_completion"]),
                "reward_research_progress": float(parts["research_progress"]),
                "new_lab_research": newly_completed,
                "research_progress": live_progress,
                "lifetime_lab_ingredient_reward": float(
                    self.lifetime_lab_ingredient_reward
                ),
                "horizon_lab_ingredient_reward": float(
                    self.horizon_lab_ingredient_reward
                ),
                "action_cooldowns_runtime": dict(self._runtime_action_cooldown_until),
                "entity_counts": dict(self.entity_counts),
                "authoritative_entity_counts": dict(self.entity_counts),
                "action_mask": self.action_masks(),
            }
        )
        self._last_lab_ingredient_fraction_runtime = after_fraction
        return self._observation(), float(reward), terminated, truncated, final

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
            }
        )
        return info


FactorioOpenEndedEnvResearchProgress = FactorioOpenEndedEnvCraft


__all__ = [
    "FactorioOpenEndedEnvResearchProgress",
    "FactorioOpenEndedEnvCraft",
    "Runtime_VERSION",
    "Runtime_ACTIONS",
    "Runtime_PROGRESSION_REWARDS",
    "Runtime_RESEARCH_TARGETS",
    "Runtime_TRACKED_ITEMS",
]
