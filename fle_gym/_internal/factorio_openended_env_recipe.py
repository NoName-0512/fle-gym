from __future__ import annotations

"""runtime algorithm-neutral open-ended Factorio environment.

runtime keeps the same fixed Discrete action space and numeric observation vector as
runtime so MaskablePPO, categorical/discretised SAC, and MAP-Elites can consume the
same environment.  It corrects reward accounting, fails closed when live
feasibility telemetry is unavailable, records action opportunities, and treats
geometry failures caused by a stale feasibility query as environment
rejections rather than policy mistakes.

The environment remains goal-free.  During training, ``training_horizon_steps``
causes a Gymnasium time-limit truncation.  Setting it to zero creates a
persistent factory for evaluation.
"""

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from factorio_openended_env_feasibility import (
    FactorioOpenEndedEnvFeasibility,
    LAB_RESEARCH_TARGETS,
    MACHINE_OUTPUT_WEIGHTS,
    MANUAL_CRAFT_OUTPUTS,
    MANUAL_HARVEST_OUTPUTS,
    TRIGGER_TECHNOLOGIES,
    Runtime_ACTIONS,
    Runtime_DEFAULT_INVENTORY,
)
from factorio_transition_env import json_safe, safe_float


Runtime_PREFLIGHT_MARKER = "__FLE_RL_PREFLIGHT_runtime__"
Runtime_VERSION = "16.0"

# Actions where a collision/pathfinding failure is an environment-side
# feasibility miss.  These are refunded rather than teaching the policy that a
# valid action is intrinsically bad.
ENVIRONMENT_GEOMETRY_ACTIONS = {
    "place_furnace_near_drill",
    "connect_electric_drill_to_power",
    "connect_lab_to_power",
    "connect_factory_power",
    "build_iron_smelting_line",
    "build_copper_smelting_line",
    "build_gear_assembly_cell",
    "build_science_assembly_cell",
}
ENVIRONMENT_GEOMETRY_ERRORS = (
    "no-buildable-layout",
    "cannot-place",
    "cannot-route",
    "still-unpowered",
    "create-",
    "missing-resource",
    "no compatible position",
    "could not place",
    "no buildable position",
)

# Milestones make the long dependency chain learnable without prescribing an
# action order.  Every algorithm receives exactly the same reward function.
PROGRESSION_MILESTONE_REWARDS: Dict[str, float] = {
    "lab_obtained": 2.0,
    "lab_placed": 3.0,
    "automation_science_unlocked": 4.0,
    "automation_science_available": 4.0,
    "lab_working": 6.0,
    "automation_researched": 12.0,
    "logistics_researched": 8.0,
    "steel_processing_researched": 8.0,
}

# Items may only be classified as machine output if a machine capable of
# producing that item exists.  This prevents delayed manual crafting statistics
# from being mistaken for automated production.
MACHINE_SOURCE_KIND: Dict[str, str] = {
    "iron-ore": "miner",
    "copper-ore": "miner",
    "iron-plate": "furnace",
    "copper-plate": "furnace",
    "steel-plate": "furnace",
    "iron-gear-wheel": "gear-assembler",
    "electronic-circuit": "circuit-assembler",
    "automation-science-pack": "science-assembler",
    "logistic-science-pack": "logistic-science-assembler",
}


class FactorioOpenEndedEnvRecipe(FactorioOpenEndedEnvFeasibility):
    """Shared runtime environment for PPO, discrete SAC, and MAP-Elites."""

    def __init__(
        self,
        *,
        progression_shaping: bool = True,
        manual_guard_ttl_steps: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Preserve the exact runtime action/observation dimensions.  All algorithms
        # use this same schema; runtime changes semantics and diagnostics, not shape.
        self.actions = Runtime_ACTIONS
        self.progression_shaping = bool(progression_shaping)
        self.manual_guard_ttl_steps = max(1, int(manual_guard_ttl_steps))

        self.lifetime_mask_samples = 0
        self.horizon_mask_samples = 0
        self.lifetime_action_available_counts: Counter[str] = Counter()
        self.horizon_action_available_counts: Counter[str] = Counter()
        self.lifetime_environment_rejections: Counter[str] = Counter()
        self.horizon_environment_rejections: Counter[str] = Counter()
        self.lifetime_unattributed_output_by_item: Counter[str] = Counter()
        self.horizon_unattributed_output_by_item: Counter[str] = Counter()
        self.lifetime_canonical_reward_components: Counter[str] = Counter()
        self.horizon_canonical_reward_components: Counter[str] = Counter()
        self.progression_milestones_seen: set[str] = set()
        # item -> list of [remaining amount, expiry global step]
        self._manual_guard: Dict[str, List[List[float]]] = defaultdict(list)
        self._write_config_recipe()

    # ------------------------------------------------------------------
    # Schema and configuration
    # ------------------------------------------------------------------
    def schema(self) -> Dict[str, Any]:
        return {
            "version": Runtime_VERSION,
            "environment": type(self).__name__,
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
            "mask_method": "action_masks",
            "terminal_goal": None,
            "training_horizon_steps": int(self.training_horizon_steps),
            "algorithms": ["MaskablePPO", "CategoricalDiscreteSAC", "MAP-Elites"],
            "machine_output_weights": dict(MACHINE_OUTPUT_WEIGHTS),
            "progression_milestone_rewards": dict(PROGRESSION_MILESTONE_REWARDS),
        }

    def export_schema(self, destination: Optional[str | Path] = None) -> str:
        path = Path(destination) if destination else Path(self.run_dir) / "environment_schema_runtime.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(self.schema()), indent=2), encoding="utf-8")
        return str(path)

    def _write_config_recipe(self) -> None:
        payload = {
            "version": Runtime_VERSION,
            "progression_shaping": self.progression_shaping,
            "manual_guard_ttl_steps": self.manual_guard_ttl_steps,
            "geometry_actions": sorted(ENVIRONMENT_GEOMETRY_ACTIONS),
            "machine_source_kind": dict(MACHINE_SOURCE_KIND),
            "screenshot_requirement": "optional; no graphical client is required when disabled",
        }
        Path(self.run_dir, "runtime_environment_config.json").write_text(
            json.dumps(json_safe(payload), indent=2), encoding="utf-8"
        )
        self.export_schema()

    # ------------------------------------------------------------------
    # Live preflight query and stricter masks
    # ------------------------------------------------------------------
    @staticmethod
    def _bool_token(value: str) -> bool:
        return value.strip() in {"1", "true", "True"}

    def _preflight_command_recipe(self) -> str:
        marker = json.dumps(Runtime_PREFLIGHT_MARKER)
        return (
            "/sc local marker=" + marker + "; local function run() "
            + self._runtime_agent_prelude()
            + "if not p then rcon.print(marker..'|ERROR|no-agent'); return end; "
            "local s=p.surface; local f=p.force; local out={}; "
            "local function emit(k,v) out[#out+1]=k..'='..(v and '1' or '0') end; "
            "local function pole_near(a) if not a or not a.valid then return false end; "
            "for _,e in ipairs(s.find_entities_filtered{name='small-electric-pole',force=f}) do local dx=e.position.x-a.position.x; local dy=e.position.y-a.position.y; if dx*dx+dy*dy<=36 then return true end end; "
            "local bb=a.bounding_box; local cx=(bb.left_top.x+bb.right_bottom.x)/2; local cy=(bb.left_top.y+bb.right_bottom.y)/2; "
            "for r=1,6 do local c={{x=bb.left_top.x-r,y=cy},{x=bb.right_bottom.x+r,y=cy},{x=cx,y=bb.left_top.y-r},{x=cx,y=bb.right_bottom.y+r},{x=bb.left_top.x-r,y=bb.left_top.y-r},{x=bb.right_bottom.x+r,y=bb.left_top.y-r},{x=bb.left_top.x-r,y=bb.right_bottom.y+r},{x=bb.right_bottom.x+r,y=bb.right_bottom.y+r}}; for _,pos in ipairs(c) do if s.can_place_entity{name='small-electric-pole',position=pos,force=f} then return true end end end; return false end; "
            "local function furnace_near_drill() for _,dn in ipairs({'burner-mining-drill','electric-mining-drill'}) do for _,d in ipairs(s.find_entities_filtered{name=dn,force=f}) do local b=d.position; for r=3,14 do for _,o in ipairs({{r,0},{-r,0},{0,r},{0,-r},{r,r},{r,-r},{-r,r},{-r,-r}}) do if s.can_place_entity{name='stone-furnace',position={x=b.x+o[1],y=b.y+o[2]},force=f} then return true end end end end end; return false end; "
            "local function line(resource) local patches=s.find_entities_filtered{name=resource}; local dirs={{1,0,defines.direction.east},{-1,0,defines.direction.west},{0,1,defines.direction.south},{0,-1,defines.direction.north}}; "
            "for pi=1,math.min(#patches,100) do local base=patches[pi].position; for _,d in ipairs(dirs) do local x=d[1]; local y=d[2]; local c={{'electric-mining-drill',{x=base.x,y=base.y},d[3]},{'transport-belt',{x=base.x+2*x,y=base.y+2*y},d[3]},{'transport-belt',{x=base.x+3*x,y=base.y+3*y},d[3]},{'transport-belt',{x=base.x+4*x,y=base.y+4*y},d[3]},{'inserter',{x=base.x+5*x,y=base.y+5*y},d[3]},{'stone-furnace',{x=base.x+7*x,y=base.y+7*y},d[3]},{'inserter',{x=base.x+9*x,y=base.y+9*y},d[3]},{'transport-belt',{x=base.x+10*x,y=base.y+10*y},d[3]},{'transport-belt',{x=base.x+11*x,y=base.y+11*y},d[3]},{'transport-belt',{x=base.x+12*x,y=base.y+12*y},d[3]},{'transport-belt',{x=base.x+13*x,y=base.y+13*y},d[3]},{'transport-belt',{x=base.x+14*x,y=base.y+14*y},d[3]},{'inserter',{x=base.x+15*x,y=base.y+15*y},d[3]},{'wooden-chest',{x=base.x+16*x,y=base.y+16*y},d[3]},{'wooden-chest',{x=base.x+7*x-y*3,y=base.y+7*y+x*3},d[3]},{'inserter',{x=base.x+7*x-y*2,y=base.y+7*y+x*2},(x==1 and defines.direction.north or x==-1 and defines.direction.south or y==1 and defines.direction.east or defines.direction.west)},{'small-electric-pole',{x=base.x+2*x-y*2,y=base.y+2*y+x*2},d[3]},{'small-electric-pole',{x=base.x+8*x-y*2,y=base.y+8*y+x*2},d[3]}}; local ok=true; for _,q in ipairs(c) do if not s.can_place_entity{name=q[1],position=q[2],force=f,direction=q[3]} then ok=false; break end end; if ok then return true end end end; return false end; "
            "local function cell_layout(inputs) local origin=p.position; local dirs={{1,0},{-1,0},{0,1},{0,-1}}; for radius=8,40,4 do for _,d in ipairs(dirs) do local x=origin.x+d[1]*radius; local y=origin.y+d[2]*radius; local c={{'assembling-machine-1',{x=x,y=y},defines.direction.east},{'wooden-chest',{x=x-3,y=y},defines.direction.east},{'inserter',{x=x-2,y=y},defines.direction.east},{'inserter',{x=x+2,y=y},defines.direction.east},{'transport-belt',{x=x+3,y=y},defines.direction.east},{'transport-belt',{x=x+4,y=y},defines.direction.east},{'transport-belt',{x=x+5,y=y},defines.direction.east},{'inserter',{x=x+6,y=y},defines.direction.east},{'wooden-chest',{x=x+7,y=y},defines.direction.east},{'small-electric-pole',{x=x,y=y+4},defines.direction.east}}; if inputs==2 then c[#c+1]={'wooden-chest',{x=x,y=y-3},defines.direction.east}; c[#c+1]={'inserter',{x=x,y=y-2},defines.direction.south} end; local ok=true; for _,q in ipairs(c) do if not s.can_place_entity{name=q[1],position=q[2],force=f,direction=q[3]} then ok=false; break end end; if ok then return true end end end; return false end; "
            "local engines=s.find_entities_filtered{name='steam-engine',force=f}; local route=(#engines>0 and pole_near(engines[1])); local any_unpowered=false; if route then for _,n in ipairs({'electric-mining-drill','inserter','assembling-machine-1','lab'}) do for _,e in ipairs(s.find_entities_filtered{name=n,force=f}) do local ok,c=pcall(function() return e.is_connected_to_electric_network() end); if not (ok and c) then any_unpowered=true; if not pole_near(e) then route=false; break end end end; if not route then break end end end; "
            "emit('can_place_furnace_near_drill',furnace_near_drill()); emit('can_build_iron_line',line('iron-ore')); emit('can_build_copper_line',line('copper-ore')); emit('can_build_gear_cell',cell_layout(1)); emit('can_build_science_cell',cell_layout(2)); emit('can_connect_factory_power',route and any_unpowered); "
            "local circuit=0; for _,e in ipairs(s.find_entities_filtered{name='assembling-machine-1',force=f}) do local ok,r=pcall(function() return e.get_recipe() end); if ok and r and r.name=='electronic-circuit' then circuit=circuit+1 end end; out[#out+1]='circuit_assembler_count='..circuit; "
            "rcon.print(marker..'|'..table.concat(out,'|')); end; local ok,err=pcall(run); if not ok then rcon.print(marker..'|ERROR|'..tostring(err)) end"
        )

    def _collect_feasibility_state(self, *, force: bool = False) -> Dict[str, Any]:
        state = super()._collect_feasibility_state(force=force)
        if not force and self.global_step % self.feasibility_state_every != 0:
            return state
        try:
            raw = str(self.instance.rcon_client.send_command(self._preflight_command_recipe()) or "")
            line = next((part for part in raw.splitlines() if Runtime_PREFLIGHT_MARKER in part), "")
            if not line:
                raise RuntimeError(f"runtime preflight marker missing: {raw[:400]!r}")
            payload = line.split(Runtime_PREFLIGHT_MARKER, 1)[1].lstrip(": |")
            if payload.startswith("ERROR|"):
                raise RuntimeError(payload)
            for token in payload.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                if key.startswith("can_"):
                    state[key] = self._bool_token(value)
                else:
                    state[key] = safe_float(value)
            state["runtime_preflight_ok"] = True
            state["runtime_preflight_error"] = ""
        except Exception as exc:
            state.update(
                {
                    "runtime_preflight_ok": False,
                    "runtime_preflight_error": f"{type(exc).__name__}: {exc}",
                    "can_place_furnace_near_drill": False,
                    "can_build_iron_line": False,
                    "can_build_copper_line": False,
                    "can_build_gear_cell": False,
                    "can_build_science_cell": False,
                    "can_connect_factory_power": False,
                    "circuit_assembler_count": safe_float(state.get("circuit_assembler_count")),
                }
            )
        self.feasibility_state = state
        return state

    def get_action_mask(self) -> np.ndarray:
        mask = super().get_action_mask().astype(np.int8)
        f = self.feasibility_state
        runtime_ok = bool(f.get("runtime_query_ok", False))
        runtime_ok = bool(f.get("runtime_preflight_ok", False))

        def set_valid(name: str, valid: bool) -> None:
            try:
                mask[self.action_index(name)] = int(bool(valid))
            except KeyError:
                return

        def recipe_ready(recipe: str, quantity: float = 1.0) -> bool:
            return (
                runtime_ok
                and bool(f.get(f"recipe_enabled__{recipe}", False))
                and safe_float(f.get(f"craftable__{recipe}")) >= quantity
            )

        # Fail closed for dependency-sensitive crafting.  A failed telemetry
        # query must not expose actions that are known to be locked.
        set_valid("craft_electronic_circuits", recipe_ready("electronic-circuit", 10))
        set_valid("craft_transport_belts", recipe_ready("transport-belt", 10))
        set_valid("craft_inserters", recipe_ready("inserter", 4))
        set_valid(
            "craft_lab",
            recipe_ready("lab", 1)
            and self.entity_counts.get("lab", 0.0) < 2.0
            and self.inventory.get("lab", 0.0) < 1.0,
        )
        set_valid(
            "craft_automation_science",
            recipe_ready("automation-science-pack", 10)
            and bool(f.get("trigger_researched__automation-science-pack", False)),
        )
        set_valid("craft_iron_gears", recipe_ready("iron-gear-wheel", 10))
        set_valid(
            "craft_assembling_machine",
            recipe_ready("assembling-machine-1", 1)
            and bool(self.infrastructure_state.get("researched__automation", False)),
        )

        # Exact geometry preflights match the deterministic construction macros.
        set_valid(
            "place_furnace_near_drill",
            runtime_ok
            and bool(f.get("can_place_furnace_near_drill", False))
            and self.inventory.get("stone-furnace", 0.0) >= 1,
        )
        set_valid(
            "build_iron_smelting_line",
            bool(mask[self.action_index("build_iron_smelting_line")])
            and runtime_ok
            and bool(f.get("can_build_iron_line", False)),
        )
        set_valid(
            "build_copper_smelting_line",
            bool(mask[self.action_index("build_copper_smelting_line")])
            and runtime_ok
            and bool(f.get("can_build_copper_line", False)),
        )
        set_valid(
            "build_gear_assembly_cell",
            bool(mask[self.action_index("build_gear_assembly_cell")])
            and runtime_ok
            and bool(f.get("can_build_gear_cell", False)),
        )
        set_valid(
            "build_science_assembly_cell",
            bool(mask[self.action_index("build_science_assembly_cell")])
            and runtime_ok
            and bool(f.get("can_build_science_cell", False)),
        )
        route_possible = runtime_ok and bool(f.get("can_connect_factory_power", False))
        set_valid(
            "connect_factory_power",
            bool(mask[self.action_index("connect_factory_power")]) and route_possible,
        )
        set_valid(
            "connect_electric_drill_to_power",
            bool(mask[self.action_index("connect_electric_drill_to_power")]) and route_possible,
        )
        set_valid(
            "connect_lab_to_power",
            bool(mask[self.action_index("connect_lab_to_power")]) and route_possible,
        )

        # A lab research action is only exposed when the live technology query
        # confirms it is available and no other research is active.
        current = str(self.infrastructure_state.get("current_research", "") or "")
        for target in LAB_RESEARCH_TARGETS:
            set_valid(
                f"research_{target.replace('-', '_')}",
                runtime_ok
                and not current
                and safe_float(self.infrastructure_state.get("lab_connected_count")) > 0
                and bool(self.infrastructure_state.get(f"available__{target}", False))
                and not bool(self.infrastructure_state.get(f"researched__{target}", False)),
            )
        set_valid("noop", True)
        return mask

    # ------------------------------------------------------------------
    # Output attribution and reward accounting
    # ------------------------------------------------------------------
    def _has_machine_source(self, item: str) -> bool:
        kind = MACHINE_SOURCE_KIND.get(item)
        if kind is None:
            return False
        if kind == "miner":
            return (
                self.entity_counts.get("burner-mining-drill", 0.0)
                + self.entity_counts.get("electric-mining-drill", 0.0)
            ) > 0
        if kind == "furnace":
            return self.entity_counts.get("stone-furnace", 0.0) > 0
        if kind == "gear-assembler":
            return safe_float(self.feasibility_state.get("gear_assembler_count")) > 0
        if kind == "circuit-assembler":
            return safe_float(self.feasibility_state.get("circuit_assembler_count")) > 0
        if kind == "science-assembler":
            return safe_float(self.feasibility_state.get("science_assembler_count")) > 0
        if kind == "logistic-science-assembler":
            return safe_float(self.feasibility_state.get("logistic_science_assembler_count")) > 0
        return False

    def _expire_manual_guard(self) -> None:
        for item in list(self._manual_guard):
            remaining = [entry for entry in self._manual_guard[item] if entry[0] > 1e-9 and entry[1] >= self.global_step]
            if remaining:
                self._manual_guard[item] = remaining
            else:
                del self._manual_guard[item]

    def _add_manual_guard(self, item: str, amount: float) -> None:
        if amount > 0:
            self._manual_guard[item].append(
                [float(amount), float(self.global_step + self.manual_guard_ttl_steps)]
            )

    def _consume_manual_guard(self, item: str, amount: float) -> float:
        remaining = max(0.0, float(amount))
        consumed = 0.0
        queue = self._manual_guard.get(item, [])
        for entry in queue:
            if remaining <= 0:
                break
            take = min(remaining, max(0.0, entry[0]))
            entry[0] -= take
            remaining -= take
            consumed += take
        self._expire_manual_guard()
        return consumed

    @staticmethod
    def _counter_subtract(counter: Counter[str], item: str, amount: float) -> None:
        if amount <= 0:
            return
        counter[item] -= amount
        if counter[item] <= 1e-9:
            counter.pop(item, None)

    def _current_progress_milestones(self) -> set[str]:
        f = self.feasibility_state
        infra = self.infrastructure_state
        milestones: set[str] = set()
        if self.inventory.get("lab", 0.0) > 0:
            milestones.add("lab_obtained")
        if self.entity_counts.get("lab", 0.0) > 0:
            milestones.add("lab_placed")
        if bool(f.get("trigger_researched__automation-science-pack", False)):
            milestones.add("automation_science_unlocked")
        if (
            self.inventory.get("automation-science-pack", 0.0)
            + safe_float(f.get("buffered_automation_science_pack"))
            + safe_float(f.get("lab_science_count"))
        ) > 0:
            milestones.add("automation_science_available")
        if safe_float(f.get("lab_working_count")) > 0:
            milestones.add("lab_working")
        if bool(infra.get("researched__automation", False)):
            milestones.add("automation_researched")
        if bool(infra.get("researched__logistics", False)):
            milestones.add("logistics_researched")
        if bool(infra.get("researched__steel-processing", False)):
            milestones.add("steel_processing_researched")
        return milestones

    def _canonical_reward_parts(
        self,
        *,
        reward: float,
        legacy_parts: Mapping[str, Any],
        action_error: str,
        invalid_action: bool,
        progression_reward: float,
        environment_rejection: str,
    ) -> Dict[str, float]:
        error = -self.execution_error_penalty if action_error and not invalid_action else 0.0
        invalid = -self.invalid_action_penalty if invalid_action else 0.0
        base = safe_float(legacy_parts.get("base_error_and_invalid_reward"))
        known_base = error + invalid
        legacy_base = base - known_base
        parts = {
            "machine_production": safe_float(
                legacy_parts.get("machine_production", legacy_parts.get("automated_output"))
            ),
            "research_progress": safe_float(legacy_parts.get("research_progress")),
            "research_completion": safe_float(legacy_parts.get("research_completion")),
            "research_activity": safe_float(legacy_parts.get("research_activity")),
            "power_milestone": safe_float(legacy_parts.get("power_online")),
            "infrastructure": safe_float(legacy_parts.get("infrastructure_growth")),
            "logistics_milestone": safe_float(legacy_parts.get("logistics_milestone")),
            "progression_milestone": float(progression_reward),
            "idle": safe_float(legacy_parts.get("idle_penalty")),
            "step": safe_float(legacy_parts.get("step_penalty")),
            "error": float(error),
            "invalid": float(invalid),
            "legacy_base": float(legacy_base),
            "environment_rejection": 0.0 if environment_rejection else 0.0,
        }
        subtotal = sum(parts.values())
        parts["residual"] = float(reward - subtotal)
        parts["total"] = float(reward)
        return parts

    def behavior_descriptors(self) -> Dict[str, float]:
        f = self.feasibility_state
        entity_total = float(sum(self.entity_counts.values()))
        belts = safe_float(f.get("transport_belt_count"))
        inserters = safe_float(f.get("inserter_count"))
        assemblers = safe_float(f.get("assembling_machine_count"))
        lines = safe_float(f.get("iron_smelting_line_count")) + safe_float(f.get("copper_smelting_line_count"))
        researched = float(self._researched_count(self.infrastructure_state))
        return {
            "entity_count": entity_total,
            "logistics_complexity": belts + 2.0 * inserters + 4.0 * assemblers + 8.0 * lines,
            "machine_output_total": float(self.machine_output_total),
            "researched_count": researched,
            "power_online": float(bool(self.infrastructure_state.get("active_power", False))),
            "action_diversity": float(len(self.successful_action_counts)),
        }

    def benchmark_fitness(self) -> float:
        descriptors = self.behavior_descriptors()
        return float(
            self.machine_output_total
            + 30.0 * descriptors["researched_count"]
            + 10.0 * descriptors["power_online"]
            + 0.5 * descriptors["logistics_complexity"]
            + 0.1 * descriptors["entity_count"]
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def _base_info(self) -> Dict[str, Any]:
        info = super()._base_info()
        info.update(
            {
                "environment_version": Runtime_VERSION,
                "lifetime_mask_samples": int(self.lifetime_mask_samples),
                "horizon_mask_samples": int(self.horizon_mask_samples),
                "lifetime_action_available_counts": dict(self.lifetime_action_available_counts),
                "horizon_action_available_counts": dict(self.horizon_action_available_counts),
                "lifetime_environment_rejections": dict(self.lifetime_environment_rejections),
                "horizon_environment_rejections": dict(self.horizon_environment_rejections),
                "lifetime_unattributed_output_by_item": dict(self.lifetime_unattributed_output_by_item),
                "horizon_unattributed_output_by_item": dict(self.horizon_unattributed_output_by_item),
                "lifetime_canonical_reward_components": dict(self.lifetime_canonical_reward_components),
                "horizon_canonical_reward_components": dict(self.horizon_canonical_reward_components),
                "progression_milestones_seen": sorted(self.progression_milestones_seen),
            }
        )
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        observation, info = super().reset(seed=seed, options=options)
        hard = bool(dict(options or {}).get("hard_reset", self.hard_reset))
        if hard:
            self.horizon_mask_samples = 0
            self.horizon_action_available_counts.clear()
            self.horizon_environment_rejections.clear()
            self.horizon_unattributed_output_by_item.clear()
            self.horizon_canonical_reward_components.clear()
            self.progression_milestones_seen.clear()
            self._manual_guard.clear()
        self._collect_feasibility_state(force=True)
        observation = self._observation()
        final = {**self._base_info(), **dict(info)}
        final["action_mask"] = self.action_masks()
        final["environment_schema_path"] = str(Path(self.run_dir) / "environment_schema_runtime.json")
        return observation, final

    def step(self, action: int):
        pre_mask = np.asarray(self.action_masks(), dtype=bool).copy()
        self.lifetime_mask_samples += 1
        self.horizon_mask_samples += 1
        for index in np.flatnonzero(pre_mask):
            name = self.actions[int(index)].name
            self.lifetime_action_available_counts[name] += 1
            self.horizon_action_available_counts[name] += 1

        action_index = int(action)
        action_name = self.actions[action_index].name if 0 <= action_index < len(self.actions) else "invalid-index"
        observation, reward, terminated, truncated, info = super().step(action_index)
        info = dict(info)
        legacy_parts = dict(info.get("reward_parts", {}))

        # Register explicit manual operations in an independent delayed-output
        # guard.  This is deliberately conservative: false machine production
        # is worse than a small temporary undercount around hand crafting.
        if not info.get("action_error") and not info.get("invalid_action"):
            if action_name in MANUAL_CRAFT_OUTPUTS:
                item, amount = MANUAL_CRAFT_OUTPUTS[action_name]
                self._add_manual_guard(item, amount)
            if action_name in MANUAL_HARVEST_OUTPUTS:
                item, amount = MANUAL_HARVEST_OUTPUTS[action_name]
                self._add_manual_guard(item, amount)
        self._expire_manual_guard()

        reported_machine = {
            str(item): max(0.0, safe_float(amount))
            for item, amount in dict(info.get("machine_output_delta", {})).items()
        }
        verified_machine: Dict[str, float] = {}
        unattributed: Dict[str, float] = {}
        for item, amount in reported_machine.items():
            guarded = self._consume_manual_guard(item, amount)
            residual = max(0.0, amount - guarded)
            if residual > 0 and self._has_machine_source(item):
                verified_machine[item] = residual
            rejected = amount - verified_machine.get(item, 0.0)
            if rejected > 0:
                unattributed[item] = rejected

        leaked_weight = self._strict_weight(unattributed)
        leaked_units = self._raw_units(unattributed)
        if leaked_weight or leaked_units:
            self.machine_output_total = max(0.0, self.machine_output_total - leaked_weight)
            self.lifetime_machine_output_total = max(
                0.0, self.lifetime_machine_output_total - leaked_weight
            )
            self.machine_output_units_total = max(
                0.0, self.machine_output_units_total - leaked_units
            )
            self.lifetime_machine_output_units_total = max(
                0.0, self.lifetime_machine_output_units_total - leaked_units
            )
            for item, amount in unattributed.items():
                self._counter_subtract(self.machine_output_by_item, item, amount)
                self._counter_subtract(self.lifetime_machine_output_by_item, item, amount)
                self.horizon_unattributed_output_by_item[item] += amount
                self.lifetime_unattributed_output_by_item[item] += amount
            correction = self.automated_output_reward_scale * leaked_weight
            reward -= correction
            self.episode_return -= correction
            self.automated_output_total = self.machine_output_total
            legacy_parts["machine_production"] = max(
                0.0, safe_float(legacy_parts.get("machine_production")) - correction
            )
            legacy_parts["automated_output"] = max(
                0.0, safe_float(legacy_parts.get("automated_output")) - correction
            )
            self.lifetime_reward_components["machine_production"] -= correction
            self.lifetime_reward_components["automated_output"] -= correction

        # Refund known geometry/pathfinding misses.  The action was valid under
        # the last live mask; the policy should not learn from an environment
        # implementation failure.
        environment_rejection = ""
        action_error = str(info.get("action_error", "") or "")
        if (
            action_name in ENVIRONMENT_GEOMETRY_ACTIONS
            and action_error
            and any(token in action_error.lower() for token in ENVIRONMENT_GEOMETRY_ERRORS)
        ):
            environment_rejection = action_error
            reward += self.execution_error_penalty
            self.episode_return += self.execution_error_penalty
            self.total_action_errors = max(0, int(self.total_action_errors) - 1)
            self.lifetime_environment_rejections[action_name] += 1
            self.horizon_environment_rejections[action_name] += 1
            info["action_error"] = ""
            info["environment_rejection"] = environment_rejection
            legacy_parts["base_error_and_invalid_reward"] = (
                safe_float(legacy_parts.get("base_error_and_invalid_reward"))
                + self.execution_error_penalty
            )
            action_error = ""

        progression_reward = 0.0
        new_milestones: List[str] = []
        if self.progression_shaping:
            current = self._current_progress_milestones()
            new_milestones = sorted(current - self.progression_milestones_seen)
            for milestone in new_milestones:
                progression_reward += PROGRESSION_MILESTONE_REWARDS.get(milestone, 0.0)
            self.progression_milestones_seen.update(new_milestones)
            reward += progression_reward
            self.episode_return += progression_reward
            self.lifetime_reward_components["progression_milestone"] += progression_reward

        canonical = self._canonical_reward_parts(
            reward=float(reward),
            legacy_parts=legacy_parts,
            action_error=action_error,
            invalid_action=bool(info.get("invalid_action", False)),
            progression_reward=progression_reward,
            environment_rejection=environment_rejection,
        )

        for component, value in canonical.items():
            if component != "total":
                self.lifetime_canonical_reward_components[component] += float(value)
                self.horizon_canonical_reward_components[component] += float(value)
        self.lifetime_canonical_reward_components["total"] += float(reward)
        self.horizon_canonical_reward_components["total"] += float(reward)

        final_info = dict(info)
        final_info.update(self._base_info())
        final_info.update(
            {
                "reward": float(reward),
                "reward_parts": canonical,
                "legacy_reward_parts": legacy_parts,
                "machine_output_delta": verified_machine,
                "reported_machine_output_delta": reported_machine,
                "unattributed_output_delta": unattributed,
                "pre_action_mask": pre_mask.astype(np.int8),
                "valid_action_count_before": int(pre_mask.sum()),
                "selected_action_was_available": bool(
                    0 <= action_index < pre_mask.size and pre_mask[action_index]
                ),
                "new_progression_milestones": new_milestones,
                "environment_rejection": environment_rejection,
                "action_mask": self.action_masks(),
                "terminal_goal": None,
            }
        )
        return self._observation(), float(reward), bool(terminated), bool(truncated), final_info


FactorioOpenEndedEnvRuntime = FactorioOpenEndedEnvRecipe

__all__ = [
    "ENVIRONMENT_GEOMETRY_ACTIONS",
    "FactorioOpenEndedEnvRecipe",
    "FactorioOpenEndedEnvRuntime",
    "MACHINE_OUTPUT_WEIGHTS",
    "PROGRESSION_MILESTONE_REWARDS",
    "Runtime_VERSION",
]
