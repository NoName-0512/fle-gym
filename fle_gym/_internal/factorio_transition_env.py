from __future__ import annotations

"""Lean discrete-action transition layer for Factorio Learning Environment.

The environment connects directly to :class:`fle.env.instance.FactorioInstance`.
It deliberately avoids FLE's heavyweight Gym observation builder during normal
steps. Expensive full-world queries are performed only for periodic factory
snapshots or explicit debugging.

The same interface is suitable for:

* sb3-contrib ``MaskablePPO`` through :meth:`action_masks`;
* categorical/discrete SAC through current and next action masks;
* MAP-Elites through deterministic integer action sequences.

The initial action catalogue is a deliberately small iron-smelting smoke task.
It is intended to validate throughput and algorithm integration before the
action catalogue is expanded.
"""

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
import base64
import json
import math
from io import BytesIO
from pathlib import Path
import re
import textwrap
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Gymnasium is required for real training. The minimal fallback exists only so
# the offline smoke test can run in stripped-down development environments.
try:  # pragma: no cover - exercised on the user's machine
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:
        class _MinimalEnv:
            metadata: Dict[str, Any] = {}

            def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
                del options
                if seed is not None:
                    np.random.seed(seed)
                return None

        class _Discrete:
            def __init__(self, n: int) -> None:
                self.n = int(n)

            def contains(self, x: Any) -> bool:
                try:
                    value = int(x)
                except Exception:
                    return False
                return 0 <= value < self.n

            def sample(self, mask: Optional[np.ndarray] = None) -> int:
                if mask is None:
                    return int(np.random.randint(self.n))
                valid = np.flatnonzero(np.asarray(mask, dtype=bool))
                if valid.size == 0:
                    return 0
                return int(np.random.choice(valid))

        class _Box:
            def __init__(self, low: Any, high: Any, shape: Tuple[int, ...], dtype: Any) -> None:
                self.low = low
                self.high = high
                self.shape = tuple(shape)
                self.dtype = dtype

            def contains(self, x: Any) -> bool:
                arr = np.asarray(x)
                return arr.shape == self.shape and np.isfinite(arr).all()

        class _Spaces:
            Discrete = _Discrete
            Box = _Box

        class _Gym:
            Env = _MinimalEnv

        gym = _Gym()
        spaces = _Spaces()


ACTION_OK = "__FLE_RL_ACTION_OK__"
ACTION_ERROR = "__FLE_RL_ACTION_ERROR__"
ACTION_EVENT = "__FLE_RL_EVENT__"
ACTION_NAME = "__FLE_RL_ACTION__"


DEFAULT_STARTER_INVENTORY: Dict[str, int] = {
    "burner-mining-drill": 1,
    "stone-furnace": 1,
    "coal": 30,
}

TRACKED_ITEMS: Tuple[str, ...] = (
    "wood",
    "coal",
    "stone",
    "iron-ore",
    "iron-plate",
    "stone-furnace",
    "burner-mining-drill",
)

TRACKED_ENTITIES: Tuple[str, ...] = (
    "burner-mining-drill",
    "stone-furnace",
)

PROTOTYPE_ENUM_NAMES: Dict[str, str] = {
    "wood": "Wood",
    "coal": "Coal",
    "stone": "Stone",
    "iron-ore": "IronOre",
    "iron-plate": "IronPlate",
    "stone-furnace": "StoneFurnace",
    "burner-mining-drill": "BurnerMiningDrill",
}


@dataclass(frozen=True)
class ActionSpec:
    name: str
    category: str
    code: str
    required_items: Mapping[str, float] = field(default_factory=dict)
    required_entities: Mapping[str, float] = field(default_factory=dict)
    event_bonus: float = 0.0
    settle_multiplier: float = 1.0
    produced_entity: Optional[str] = None
    single_use_flag: Optional[str] = None
    description: str = ""


@dataclass
class StepTiming:
    decode_s: float = 0.0
    action_eval_s: float = 0.0
    settle_s: float = 0.0
    state_s: float = 0.0
    snapshot_s: float = 0.0
    total_s: float = 0.0


@dataclass
class RewardParts:
    score_delta: float = 0.0
    iron_plate_delta: float = 0.0
    event_bonus: float = 0.0
    invalid_penalty: float = 0.0
    error_penalty: float = 0.0
    step_penalty: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(asdict(self).values()))


def _dedent(code: str) -> str:
    return textwrap.dedent(code).strip()


def _action_specs() -> Tuple[ActionSpec, ...]:
    return (
        ActionSpec(
            name="noop",
            category="noop",
            code="pass",
            description="Do nothing except let the configured settling interval elapse.",
        ),
        ActionSpec(
            name="place_drill_on_iron",
            category="place",
            required_items={"burner-mining-drill": 1},
            event_bonus=4.0,
            produced_entity="burner-mining-drill",
            single_use_flag="drill_placed",
            description="Place one burner mining drill on the nearest iron patch.",
            code=_dedent(
                """
                ore = nearest(Prototype.IronOre)
                move_to(ore)
                placed = place_entity(
                    Prototype.BurnerMiningDrill,
                    position=ore,
                    direction=Direction.DOWN,
                    exact=False,
                )
                if placed is None:
                    raise RuntimeError("burner drill placement returned None")
                print("__FLE_RL_EVENT__:drill_placed")
                """
            ),
        ),
        ActionSpec(
            name="place_furnace_near_drill",
            category="place",
            required_items={"stone-furnace": 1},
            required_entities={"burner-mining-drill": 1},
            event_bonus=4.0,
            produced_entity="stone-furnace",
            single_use_flag="furnace_placed",
            description="Place one stone furnace next to the existing burner drill.",
            code=_dedent(
                """
                # FLE 0.4.3 nearest(...) returns a Position for both
                # resources and placed entities. Use it directly rather than
                # attempting to access a nonexistent .position attribute.
                drill_position = nearest(Prototype.BurnerMiningDrill)
                move_to(drill_position)
                # FLE 0.4.3 PlaceEntityNextTo does not accept the
                # place_entity-only `exact` keyword.
                placed = place_entity_next_to(
                    Prototype.StoneFurnace,
                    reference_position=drill_position,
                )
                if placed is None:
                    raise RuntimeError("stone furnace placement returned None")
                print("__FLE_RL_EVENT__:furnace_placed")
                """
            ),
        ),
        ActionSpec(
            name="fuel_drill",
            category="insert",
            required_items={"coal": 5},
            required_entities={"burner-mining-drill": 1},
            event_bonus=2.0,
            single_use_flag="drill_fuelled",
            description="Insert five coal into the nearest burner mining drill.",
            code=_dedent(
                """
                drill_position = nearest(Prototype.BurnerMiningDrill)
                drill = get_entity(
                    Prototype.BurnerMiningDrill,
                    drill_position,
                )
                result = insert_item(
                    Prototype.Coal,
                    drill,
                    quantity=5,
                )
                print("__FLE_RL_EVENT__:drill_fuelled")
                print("fuel_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="fuel_furnace",
            category="insert",
            required_items={"coal": 5},
            required_entities={"stone-furnace": 1},
            event_bonus=2.0,
            single_use_flag="furnace_fuelled",
            description="Insert five coal into the nearest stone furnace.",
            code=_dedent(
                """
                furnace_position = nearest(Prototype.StoneFurnace)
                furnace = get_entity(
                    Prototype.StoneFurnace,
                    furnace_position,
                )
                result = insert_item(
                    Prototype.Coal,
                    furnace,
                    quantity=5,
                )
                print("__FLE_RL_EVENT__:furnace_fuelled")
                print("fuel_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="transfer_drill_ore_to_furnace",
            category="transfer",
            required_entities={"burner-mining-drill": 1, "stone-furnace": 1},
            event_bonus=1.0,
            description="Move available iron ore from the drill into the furnace.",
            code=_dedent(
                """
                drill_position = nearest(Prototype.BurnerMiningDrill)
                furnace_position = nearest(Prototype.StoneFurnace)
                drill = get_entity(
                    Prototype.BurnerMiningDrill,
                    drill_position,
                )
                furnace = get_entity(
                    Prototype.StoneFurnace,
                    furnace_position,
                )
                moved = extract_item(
                    Prototype.IronOre,
                    drill,
                    quantity=5,
                )
                if moved is None or moved == 0:
                    raise RuntimeError("no iron ore available in burner drill")
                try:
                    amount = int(moved)
                except Exception:
                    amount = 1
                amount = max(1, min(5, amount))
                inserted = insert_item(
                    Prototype.IronOre,
                    furnace,
                    quantity=amount,
                )
                print("__FLE_RL_EVENT__:ore_transferred")
                print("transfer_result=" + repr((moved, inserted)))
                """
            ),
        ),
        ActionSpec(
            name="collect_iron_plate",
            category="extract",
            required_entities={"stone-furnace": 1},
            event_bonus=3.0,
            description="Collect an available iron plate from the furnace.",
            code=_dedent(
                """
                furnace_position = nearest(Prototype.StoneFurnace)
                furnace = get_entity(
                    Prototype.StoneFurnace,
                    furnace_position,
                )
                extracted = extract_item(
                    Prototype.IronPlate,
                    furnace,
                    quantity=1,
                )
                if extracted is None or extracted == 0:
                    raise RuntimeError("no iron plate available in stone furnace")
                print("__FLE_RL_EVENT__:plate_collected")
                print("extract_result=" + repr(extracted))
                """
            ),
        ),
        ActionSpec(
            name="wait",
            category="wait",
            code="sleep(1)",
            settle_multiplier=0.0,
            description="Advance one second of Factorio production time.",
        ),
        ActionSpec(
            name="harvest_coal",
            category="harvest",
            description="Recovery action: harvest ten coal.",
            code=_dedent(
                """
                target = nearest(Prototype.Coal)
                move_to(target)
                result = harvest_resource(target, 10)
                print("__FLE_RL_EVENT__:coal_harvested")
                print("harvest_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="harvest_stone",
            category="harvest",
            description="Recovery action: harvest ten stone.",
            code=_dedent(
                """
                target = nearest(Prototype.Stone)
                move_to(target)
                result = harvest_resource(target, 10)
                print("__FLE_RL_EVENT__:stone_harvested")
                print("harvest_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="craft_stone_furnace",
            category="craft",
            required_items={"stone": 5},
            description="Recovery action: craft one stone furnace.",
            code=_dedent(
                """
                result = craft_item(Prototype.StoneFurnace, quantity=1)
                print("__FLE_RL_EVENT__:furnace_crafted")
                print("craft_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="harvest_iron_ore",
            category="harvest",
            description="Recovery action: harvest ten iron ore.",
            code=_dedent(
                """
                target = nearest(Prototype.IronOre)
                move_to(target)
                result = harvest_resource(target, 10)
                print("__FLE_RL_EVENT__:iron_harvested")
                print("harvest_result=" + repr(result))
                """
            ),
        ),
        ActionSpec(
            name="insert_hand_mined_ore",
            category="insert",
            required_items={"iron-ore": 1},
            required_entities={"stone-furnace": 1},
            description="Recovery action: insert one hand-mined iron ore into the furnace.",
            code=_dedent(
                """
                furnace_position = nearest(Prototype.StoneFurnace)
                furnace = get_entity(
                    Prototype.StoneFurnace,
                    furnace_position,
                )
                result = insert_item(
                    Prototype.IronOre,
                    furnace,
                    quantity=1,
                )
                print("__FLE_RL_EVENT__:hand_ore_inserted")
                print("insert_result=" + repr(result))
                """
            ),
        ),
    )


ACTIONS: Tuple[ActionSpec, ...] = _action_specs()


def normalise_name(value: Any) -> str:
    text = str(value).strip().strip("'\"").lower()
    if "." in text:
        text = text.split(".")[-1]
    text = text.replace("_", "-").replace(" ", "-")
    aliases = {
        "ironore": "iron-ore",
        "ironplate": "iron-plate",
        "stonfurnace": "stone-furnace",
        "stonefurnace": "stone-furnace",
        "burnerminingdrill": "burner-mining-drill",
    }
    return aliases.get(text.replace("-", ""), text)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except Exception:
        return float(default)


def json_safe(value: Any, *, max_string: int = 4000) -> Any:
    # NumPy scalar and array values commonly appear in Gymnasium observations,
    # action masks and callback metadata. Convert them before the ordinary
    # Python primitive checks so json.dumps receives only native values.
    if isinstance(value, np.generic):
        return json_safe(value.item(), max_string=max_string)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist(), max_string=max_string)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + "...<truncated>"
    if isinstance(value, Mapping):
        return {str(key): json_safe(item, max_string=max_string) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item, max_string=max_string) for item in value]
    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value), max_string=max_string)
        except Exception:
            pass
    text = repr(value)
    return text if len(text) <= max_string else text[:max_string] + "...<truncated>"


def inventory_to_dict(raw: Any) -> Dict[str, float]:
    result: Dict[str, float] = {}

    if raw is None:
        return result

    if isinstance(raw, Mapping):
        iterator: Iterable[Tuple[Any, Any]] = raw.items()
    elif hasattr(raw, "items"):
        try:
            iterator = raw.items()
        except Exception:
            iterator = []
    elif isinstance(raw, (list, tuple)):
        pairs: List[Tuple[Any, Any]] = []
        for entry in raw:
            if isinstance(entry, Mapping):
                name = entry.get("type", entry.get("name"))
                amount = entry.get("quantity", entry.get("count", entry.get("amount", 0)))
                pairs.append((name, amount))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                pairs.append((entry[0], entry[1]))
            else:
                name = getattr(entry, "type", getattr(entry, "name", None))
                amount = getattr(entry, "quantity", getattr(entry, "count", 0))
                if name is not None:
                    pairs.append((name, amount))
        iterator = pairs
    elif hasattr(raw, "__dict__"):
        iterator = vars(raw).items()
    else:
        iterator = []

    for key, value in iterator:
        name = normalise_name(key)
        if name:
            result[name] = safe_float(value)

    # Last-resort parsing for custom Inventory reprs.
    if not result:
        for raw_name, raw_value in re.findall(
            r"([A-Za-z][A-Za-z0-9_.-]*)\s*=\s*([-+]?\d+(?:\.\d+)?)",
            repr(raw),
        ):
            result[normalise_name(raw_name)] = safe_float(raw_value)

    return result


def score_to_float(raw: Any) -> float:
    if isinstance(raw, (int, float, np.number)):
        return float(raw)
    for attr in ("score", "value", "total", "production_score"):
        if hasattr(raw, attr):
            value = getattr(raw, attr)
            if isinstance(value, (int, float, np.number)):
                return float(value)
    if isinstance(raw, Mapping):
        for key in ("score", "value", "total", "production_score"):
            if key in raw:
                return safe_float(raw[key])
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", repr(raw))
    return safe_float(numbers[0]) if numbers else 0.0


def production_stats_summary(raw: Any) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for field_name in ("input", "output", "crafted", "harvested"):
        value = None
        if isinstance(raw, Mapping):
            value = raw.get(field_name)
        else:
            value = getattr(raw, field_name, None)
        summary[field_name] = inventory_to_dict(value)
    return summary


def _extract_event_names(output: str) -> List[str]:
    return re.findall(rf"{re.escape(ACTION_EVENT)}:([A-Za-z0-9_.-]+)", output or "")


def _extract_error(output: str) -> str:
    if ACTION_ERROR not in (output or ""):
        return ""
    index = output.rfind(ACTION_ERROR)
    line = output[index:].splitlines()[0]
    return line.split(":", 1)[1].strip() if ":" in line else line


class FactorioTransitionEnv(gym.Env):
    """Fast shared Factorio transition environment.

    Parameters use one real Factorio worker. Parallel training requires one
    independent worker/container and RCON port per environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        address: str = "localhost",
        tcp_port: int = 27000,
        game_speed: float = 10.0,
        max_steps: int = 128,
        settle_seconds: float = 0.04,
        pause_between_steps: bool = True,
        initial_inventory: Optional[Mapping[str, int]] = None,
        all_technologies_researched: bool = True,
        clear_entities_on_reset: bool = True,
        hard_reset: bool = True,
        snapshot_every: int = 100,
        snapshot_on_reset: bool = True,
        snapshot_on_episode_end: bool = True,
        run_dir: str | Path = "runs/fle_rl_transition",
        run_name: Optional[str] = None,
        debug: bool = True,
        debug_flush_every: int = 100,
        include_raw_output_in_info: bool = False,
        raw_output_limit: int = 4000,
        production_stats_every: int = 5,
        score_reward_scale: float = 0.01,
        iron_plate_reward: float = 10.0,
        invalid_action_penalty: float = 0.5,
        execution_error_penalty: float = 1.0,
        step_penalty: float = 0.005,
        reward_clip: Optional[float] = 20.0,
        terminate_on_plate: bool = True,
        eval_timeout: float = 30.0,
        instance: Optional[Any] = None,
        instance_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__()

        if game_speed <= 0:
            raise ValueError("game_speed must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if settle_seconds < 0:
            raise ValueError("settle_seconds cannot be negative")

        self.address = str(address)
        self.tcp_port = int(tcp_port)
        self.game_speed = float(game_speed)
        self.max_steps = int(max_steps)
        self.settle_seconds = float(settle_seconds)
        self.pause_between_steps = bool(pause_between_steps)
        self.initial_inventory_config = dict(initial_inventory or DEFAULT_STARTER_INVENTORY)
        self.all_technologies_researched = bool(all_technologies_researched)
        self.clear_entities_on_reset = bool(clear_entities_on_reset)
        self.hard_reset = bool(hard_reset)
        self.snapshot_every = max(0, int(snapshot_every))
        self.snapshot_on_reset = bool(snapshot_on_reset)
        self.snapshot_on_episode_end = bool(snapshot_on_episode_end)
        self.debug = bool(debug)
        self.debug_flush_every = max(1, int(debug_flush_every))
        self.include_raw_output_in_info = bool(include_raw_output_in_info)
        self.raw_output_limit = max(100, int(raw_output_limit))
        self.production_stats_every = max(0, int(production_stats_every))
        self.score_reward_scale = float(score_reward_scale)
        self.iron_plate_reward = float(iron_plate_reward)
        self.invalid_action_penalty = abs(float(invalid_action_penalty))
        self.execution_error_penalty = abs(float(execution_error_penalty))
        self.step_penalty = abs(float(step_penalty))
        self.reward_clip = None if reward_clip is None else abs(float(reward_clip))
        self.terminate_on_plate = bool(terminate_on_plate)
        self.eval_timeout = float(eval_timeout)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_name = run_name or f"iron_smoke_{timestamp}"
        self.run_dir = Path(run_dir) / resolved_name
        self.snapshot_dir = self.run_dir / "snapshots"
        self.debug_dir = self.run_dir / "debug"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.debug_dir / "events.jsonl"
        self.config_path = self.run_dir / "environment_config.json"

        self.actions: Tuple[ActionSpec, ...] = ACTIONS
        self.action_space = spaces.Discrete(len(self.actions))

        # Inventory, entity counts, stage flags, score metrics, and progress.
        self.observation_feature_names: Tuple[str, ...] = (
            *(f"inventory__{item}" for item in TRACKED_ITEMS),
            *(f"entities__{entity}" for entity in TRACKED_ENTITIES),
            "flag__drill_placed",
            "flag__furnace_placed",
            "flag__drill_fuelled",
            "flag__furnace_fuelled",
            "transfers",
            "plates_collected",
            "score",
            "score_delta",
            "iron_plate_delta",
            "valid_action_fraction",
            "episode_progress",
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )

        self._instance_factory = instance_factory
        self.instance = instance if instance is not None else self._create_instance()
        self.namespace = self.instance.namespace
        self._owns_instance = instance is None

        self.episode_index = 0
        self.episode_step = 0
        self.global_step = 0
        self.episode_return = 0.0
        self.total_action_errors = 0
        self.total_invalid_actions = 0
        self.last_raw_output = ""
        self.last_snapshot_path: Optional[str] = None
        self.last_production_stats: Dict[str, Dict[str, float]] = {
            "input": {}, "output": {}, "crafted": {}, "harvested": {}
        }
        self.inventory: Dict[str, float] = {item: 0.0 for item in TRACKED_ITEMS}
        self.entity_counts: Dict[str, float] = {name: 0.0 for name in TRACKED_ENTITIES}
        self.flags: Dict[str, bool] = {
            "drill_placed": False,
            "furnace_placed": False,
            "drill_fuelled": False,
            "furnace_fuelled": False,
        }
        self.transfer_count = 0
        self.plate_collect_count = 0
        self.steps_since_transfer = 10_000
        self.production_score = 0.0
        self.previous_score = 0.0
        self.last_score_delta = 0.0
        self.last_iron_plate_delta = 0.0
        self.action_counts: Counter[str] = Counter()
        self.successful_action_counts: Counter[str] = Counter()
        self.recent_events: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._debug_buffer: List[Dict[str, Any]] = []
        self._closed = False

        self._set_speed_and_unpause()
        if self.pause_between_steps:
            self._pause_instance()

        self._write_config()

    # ------------------------------------------------------------------
    # Construction and configuration
    # ------------------------------------------------------------------

    def _create_instance(self) -> Any:
        inventory = self._convert_initial_inventory(self.initial_inventory_config)
        kwargs = dict(
            address=self.address,
            tcp_port=self.tcp_port,
            fast=True,
            inventory=inventory,
            all_technologies_researched=self.all_technologies_researched,
            clear_entities=self.clear_entities_on_reset,
            reset_speed=self.game_speed,
            reset_paused=self.pause_between_steps,
        )

        if self._instance_factory is not None:
            return self._instance_factory(**kwargs)

        try:
            from fle.env.instance import FactorioInstance
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "FLE is not installed. Activate the project virtual environment and "
                "install factorio-learning-environment==0.4.3."
            ) from exc

        # Filter unsupported constructor arguments before constructing the
        # instance. Do not catch TypeError raised from inside FLE initialisation,
        # because doing so hides the actual error and opens a second connection.
        try:
            import inspect

            parameters = inspect.signature(FactorioInstance).parameters
            supported_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in parameters
            }
        except (TypeError, ValueError):
            supported_kwargs = kwargs

        return FactorioInstance(**supported_kwargs)

    @staticmethod
    def _convert_initial_inventory(inventory: Mapping[str, int]) -> Dict[str, int]:
        """Return the JSON-safe inventory mapping required by FLE 0.4.3.

        FLE serialises this mapping directly through json.dumps(), so keys must
        remain ordinary item-name strings rather than Prototype enum members.
        """
        converted: Dict[str, int] = {}

        for raw_name, amount in inventory.items():
            name = normalise_name(raw_name)

            if name not in PROTOTYPE_ENUM_NAMES:
                raise ValueError(
                    f"No supported initial inventory mapping exists for {raw_name!r}"
                )

            converted[name] = int(amount)

        return converted

    def _write_config(self) -> None:
        payload = {
            "address": self.address,
            "tcp_port": self.tcp_port,
            "game_speed": self.game_speed,
            "max_steps": self.max_steps,
            "settle_seconds": self.settle_seconds,
            "pause_between_steps": self.pause_between_steps,
            "initial_inventory": self.initial_inventory_config,
            "snapshot_every": self.snapshot_every,
            "terminate_on_plate": self.terminate_on_plate,
            "observation_features": list(self.observation_feature_names),
            "actions": [asdict(spec) for spec in self.actions],
        }
        self.config_path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public interfaces shared by PPO, discrete SAC, and MAP-Elites
    # ------------------------------------------------------------------

    def action_name(self, action: int) -> str:
        return self.actions[int(action)].name

    def action_index(self, name: str) -> int:
        for index, spec in enumerate(self.actions):
            if spec.name == name:
                return index
        raise KeyError(f"Unknown action name: {name}")

    def action_masks(self) -> np.ndarray:
        """Return ``True`` for valid actions, as expected by MaskablePPO."""
        return self.get_action_mask().astype(bool)

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

        # Stage-specific constraints narrow the smoke task and reduce wasted calls.
        # Disabled for this first live smoke environment. A burner mining
        # drill emits ore at its drop_position rather than storing it in an
        # extractable output inventory. A later automation action should align
        # the furnace with that output or route it through belts/chests.
        mask[self.action_index("transfer_drill_ore_to_furnace")] = 0
        mask[self.action_index("collect_iron_plate")] = int(
            self.flags["furnace_placed"]
            and self.transfer_count > self.plate_collect_count
            and self.steps_since_transfer >= 2
        )
        mask[self.action_index("insert_hand_mined_ore")] = int(
            self.flags["furnace_placed"] and self.inventory.get("iron-ore", 0.0) >= 1.0
        )

        mask[self.action_index("noop")] = 1
        mask[self.action_index("wait")] = 1
        return mask

    def behavior_descriptors(self) -> Dict[str, float]:
        total_entities = float(sum(self.entity_counts.values()))
        automated_components = float(
            self.flags["drill_placed"]
            + self.flags["furnace_placed"]
            + self.flags["drill_fuelled"]
            + self.flags["furnace_fuelled"]
        )
        return {
            "occupied_entity_count": total_entities,
            "automation_stage": automated_components,
            "iron_plate_inventory": float(self.inventory.get("iron-plate", 0.0)),
            "production_score": float(self.production_score),
            "action_diversity": float(len(self.successful_action_counts)),
        }

    def benchmark_fitness(self) -> float:
        return float(
            self.production_score
            + 25.0 * self.inventory.get("iron-plate", 0.0)
            + 5.0 * sum(self.flags.values())
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        try:
            super().reset(seed=seed)
        except TypeError:
            super().reset()

        options = options or {}
        reset_start = time.perf_counter()
        self.episode_index += 1
        self.episode_step = 0
        self.episode_return = 0.0
        self.last_score_delta = 0.0
        self.last_iron_plate_delta = 0.0
        self.steps_since_transfer = 10_000
        self.transfer_count = 0
        self.plate_collect_count = 0
        self.flags = {key: False for key in self.flags}
        self.entity_counts = {name: 0.0 for name in TRACKED_ENTITIES}
        self.action_counts.clear()
        self.successful_action_counts.clear()

        should_reset = bool(options.get("hard_reset", self.hard_reset))
        if should_reset:
            self._reset_instance()

        self._set_speed_and_unpause()
        self._collect_compact_state(include_production_stats=True)
        self.previous_score = self.production_score
        if self.pause_between_steps:
            self._pause_instance()

        snapshot_path = None
        if self.snapshot_on_reset:
            snapshot_path = self.save_factory_snapshot(f"episode_{self.episode_index:04d}_reset")

        obs = self._observation()
        info = self._base_info()
        info.update(
            {
                "reset_seconds": time.perf_counter() - reset_start,
                "snapshot_path": snapshot_path,
                "action_mask": self.action_masks(),
            }
        )
        self._record_event({"kind": "reset", **self._info_for_log(info)})
        return obs, info

    def step(self, action: int):
        timing = StepTiming()
        total_start = time.perf_counter()
        decode_start = total_start

        try:
            action = int(action)
        except Exception as exc:
            raise ValueError(f"Action must be integer-like, received {action!r}") from exc
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} outside Discrete({self.action_space.n})")

        spec = self.actions[action]
        mask_before = self.get_action_mask()
        invalid = not bool(mask_before[action])
        timing.decode_s = time.perf_counter() - decode_start

        before_inventory = dict(self.inventory)
        before_score = float(self.production_score)
        raw_output = ""
        action_error = ""
        events: List[str] = []

        try:
            if self.pause_between_steps:
                self._set_speed_and_unpause()

            eval_start = time.perf_counter()
            if invalid:
                raw_output = f"{ACTION_ERROR}:masked action selected"
            else:
                raw_output, eval_duration = self._execute_action(spec)
                timing.action_eval_s = max(time.perf_counter() - eval_start, eval_duration)
            action_error = _extract_error(raw_output)
            events = _extract_event_names(raw_output)

            settle_start = time.perf_counter()
            resolved_settle = self.settle_seconds * max(0.0, float(spec.settle_multiplier))
            if resolved_settle > 0:
                time.sleep(resolved_settle)
            timing.settle_s = time.perf_counter() - settle_start

            state_start = time.perf_counter()
            include_stats = bool(
                self.production_stats_every
                and (self.global_step + 1) % self.production_stats_every == 0
            )
            self._collect_compact_state(include_production_stats=include_stats)
            timing.state_s = time.perf_counter() - state_start

            if self.pause_between_steps:
                self._pause_instance()
        except Exception as exc:
            action_error = f"{type(exc).__name__}: {exc}"
            raw_output = raw_output or action_error
            self.total_action_errors += 1
            self.save_debug_bundle("step_exception", exception=exc)
            try:
                if self.pause_between_steps:
                    self._pause_instance()
            except Exception:
                pass

        action_ok = not invalid and not action_error
        self.last_raw_output = raw_output
        self.action_counts[spec.name] += 1

        if invalid:
            self.total_invalid_actions += 1
        if action_error:
            self.total_action_errors += int(not invalid)

        if action_ok:
            self.successful_action_counts[spec.name] += 1
            if spec.produced_entity:
                self.entity_counts[spec.produced_entity] = self.entity_counts.get(spec.produced_entity, 0.0) + 1.0
            if spec.single_use_flag:
                self.flags[spec.single_use_flag] = True
            if (
                "ore_transferred" in events
                or "hand_ore_inserted" in events
            ):
                self.transfer_count += 1
                self.steps_since_transfer = 0
            if "plate_collected" in events:
                self.plate_collect_count += 1

        self.steps_since_transfer += 1
        self.episode_step += 1
        self.global_step += 1

        self.last_score_delta = float(self.production_score - before_score)
        self.last_iron_plate_delta = float(
            self.inventory.get("iron-plate", 0.0) - before_inventory.get("iron-plate", 0.0)
        )

        reward_parts = RewardParts(
            score_delta=self.score_reward_scale * max(-100.0, min(100.0, self.last_score_delta)),
            iron_plate_delta=self.iron_plate_reward * max(0.0, self.last_iron_plate_delta),
            event_bonus=spec.event_bonus if action_ok else 0.0,
            invalid_penalty=-self.invalid_action_penalty if invalid else 0.0,
            error_penalty=-self.execution_error_penalty if action_error and not invalid else 0.0,
            step_penalty=-self.step_penalty,
        )
        reward = reward_parts.total
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        self.episode_return += reward

        snapshot_path: Optional[str] = None
        if self.snapshot_every and self.global_step % self.snapshot_every == 0:
            snapshot_start = time.perf_counter()
            snapshot_path = self.save_factory_snapshot(f"step_{self.global_step:07d}")
            timing.snapshot_s = time.perf_counter() - snapshot_start

        episode_success = bool(self.inventory.get("iron-plate", 0.0) >= 1.0)
        terminated = bool(self.terminate_on_plate and episode_success)
        truncated = bool(not terminated and self.episode_step >= self.max_steps)
        if truncated and self.snapshot_on_episode_end:
            snapshot_start = time.perf_counter()
            snapshot_path = self.save_factory_snapshot(
                f"episode_{self.episode_index:04d}_final_step_{self.global_step:07d}"
            )
            timing.snapshot_s += time.perf_counter() - snapshot_start

        timing.total_s = time.perf_counter() - total_start
        obs = self._observation()
        info = self._base_info()
        info.update(
            {
                "action": action,
                "action_name": spec.name,
                "action_description": spec.description,
                "action_mask": self.action_masks(),
                "invalid_action": invalid,
                "action_error": action_error,
                "action_events": events,
                "reward_parts": asdict(reward_parts),
                "snapshot_path": snapshot_path,
                "timing": asdict(timing),
                "episode_success": episode_success,
            }
        )
        if self.include_raw_output_in_info:
            info["raw_output"] = raw_output[-self.raw_output_limit :]

        self._record_event(
            {
                "kind": "step",
                "episode": self.episode_index,
                "episode_step": self.episode_step,
                "global_step": self.global_step,
                "action": spec.name,
                "reward": reward,
                "reward_parts": asdict(reward_parts),
                "invalid": invalid,
                "error": action_error,
                "events": events,
                "inventory": dict(self.inventory),
                "entities": dict(self.entity_counts),
                "flags": dict(self.flags),
                "score": self.production_score,
                "timing": asdict(timing),
                "raw_output": raw_output[-self.raw_output_limit :],
                "snapshot_path": snapshot_path,
            }
        )

        if truncated:
            self.flush_debug()

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._closed:
            return
        self.flush_debug()
        self._closed = True
        if self._owns_instance:
            try:
                self.instance.cleanup()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Execution, state collection, and reward observation
    # ------------------------------------------------------------------

    def _execute_action(self, spec: ActionSpec) -> Tuple[str, float]:
        # Construct the REPL program line-by-line. Avoid embedding an already
        # indented multi-line block inside a triple-quoted f-string: subsequent
        # interpolated lines would otherwise lose the surrounding indentation.
        indented = textwrap.indent(spec.code or "pass", "    ")
        program = (
            f'print("{ACTION_NAME}:{spec.name}")\n'
            'try:\n'
            f'{indented}\n'
            f'    print("{ACTION_OK}")\n'
            'except Exception as _fle_rl_exc:\n'
            f'    print("{ACTION_ERROR}:" + type(_fle_rl_exc).__name__ + ":" + str(_fle_rl_exc))\n'
        )
        # This catches indentation/syntax regressions before sending work over
        # RCON. Names such as Prototype are resolved only inside FLE.
        compile(program, "<fle_rl_action>", "exec")
        _reward, duration, result = self.instance.eval(
            program,
            agent_idx=0,
            timeout=self.eval_timeout,
        )
        return str(result or ""), safe_float(duration)

    def _collect_compact_state(self, *, include_production_stats: bool) -> None:
        raw_inventory = self.namespace.inspect_inventory()
        parsed_inventory = inventory_to_dict(raw_inventory)
        for item in TRACKED_ITEMS:
            self.inventory[item] = float(parsed_inventory.get(item, 0.0))

        try:
            self.production_score = score_to_float(self.namespace.score())
        except Exception:
            # Score failure is non-fatal; inventory and milestone rewards remain valid.
            pass

        if include_production_stats:
            try:
                self.last_production_stats = production_stats_summary(
                    self.namespace._get_production_stats()
                )
            except Exception:
                pass

    def _observation(self) -> np.ndarray:
        values: List[float] = []
        values.extend(
            min(10.0, math.log1p(max(0.0, self.inventory.get(item, 0.0))))
            for item in TRACKED_ITEMS
        )
        values.extend(
            min(10.0, math.log1p(max(0.0, self.entity_counts.get(entity, 0.0))))
            for entity in TRACKED_ENTITIES
        )
        values.extend(float(self.flags[key]) for key in (
            "drill_placed", "furnace_placed", "drill_fuelled", "furnace_fuelled"
        ))
        valid_fraction = float(self.get_action_mask().mean())
        values.extend(
            (
                min(10.0, self.transfer_count / 10.0),
                min(10.0, self.plate_collect_count / 10.0),
                float(np.clip(self.production_score / 1000.0, -10.0, 10.0)),
                float(np.clip(self.last_score_delta / 100.0, -10.0, 10.0)),
                float(np.clip(self.last_iron_plate_delta / 10.0, -10.0, 10.0)),
                valid_fraction,
                float(self.episode_step / self.max_steps),
            )
        )
        result = np.asarray(values, dtype=np.float32)
        if result.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Observation shape mismatch: got {result.shape}, expected {self.observation_space.shape}"
            )
        return result

    def _base_info(self) -> Dict[str, Any]:
        return {
            # ``episode`` is reserved by Stable-Baselines3's Monitor wrapper.
            # Returning a scalar under that key causes SB3 to append integers
            # to ep_info_buffer and later fail while dumping rollout logs.
            "environment_episode": self.episode_index,
            "episode_step": self.episode_step,
            "global_step": self.global_step,
            "episode_return": float(self.episode_return),
            "inventory": dict(self.inventory),
            "entity_counts": dict(self.entity_counts),
            "flags": dict(self.flags),
            "production_score": float(self.production_score),
            "production_stats": json_safe(self.last_production_stats),
            "behavior_descriptors": self.behavior_descriptors(),
            "benchmark_fitness": self.benchmark_fitness(),
            "run_dir": str(self.run_dir),
            "last_snapshot_path": self.last_snapshot_path,
            "episode_success": bool(self.inventory.get("iron-plate", 0.0) >= 1.0),
            "total_action_errors": int(self.total_action_errors),
            "total_invalid_actions": int(self.total_invalid_actions),
        }

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    def _set_speed_and_unpause(self) -> None:
        try:
            self.instance.set_speed_and_unpause(self.game_speed)
            return
        except AttributeError:
            pass
        try:
            self.instance.set_speed(self.game_speed)
        except AttributeError:
            pass
        try:
            self.instance.unpause()
        except AttributeError:
            pass

    def _pause_instance(self) -> None:
        try:
            self.instance.pause()
        except AttributeError:
            pass

    def _reset_instance(self) -> None:
        kwargs = dict(
            reset_position=True,
            all_technologies_researched=self.all_technologies_researched,
            clear_entities=self.clear_entities_on_reset,
        )
        try:
            self.instance.reset(**kwargs)
        except TypeError:
            kwargs.pop("reset_position", None)
            try:
                self.instance.reset(**kwargs)
            except TypeError:
                self.instance.reset()

    # ------------------------------------------------------------------
    # Factory snapshots
    # ------------------------------------------------------------------

    def _entity_records(self) -> List[Dict[str, Any]]:
        raw_entities = self.namespace.get_entities()
        records: List[Dict[str, Any]] = []
        counts: Counter[str] = Counter()

        for index, entity in enumerate(raw_entities or []):
            name = normalise_name(
                getattr(entity, "name", None)
                or getattr(entity, "prototype", None)
                or getattr(entity, "type", None)
                or (entity.get("name") if isinstance(entity, Mapping) else "unknown")
            )

            position = getattr(entity, "position", None)
            if position is None and isinstance(entity, Mapping):
                position = entity.get("position")

            if isinstance(position, Mapping):
                x = safe_float(position.get("x"))
                y = safe_float(position.get("y"))
            else:
                x = safe_float(getattr(position, "x", getattr(entity, "x", 0.0)))
                y = safe_float(getattr(position, "y", getattr(entity, "y", 0.0)))

            record = {
                "index": index,
                "name": name,
                "x": x,
                "y": y,
                "direction": str(getattr(entity, "direction", "")),
                "status": str(getattr(entity, "status", "")),
                "warnings": json_safe(getattr(entity, "warnings", [])),
            }
            records.append(record)
            counts[name] += 1

        for tracked in TRACKED_ENTITIES:
            if counts.get(tracked, 0):
                self.entity_counts[tracked] = float(counts[tracked])
                if tracked == "burner-mining-drill":
                    self.flags["drill_placed"] = True
                elif tracked == "stone-furnace":
                    self.flags["furnace_placed"] = True

        return records

    @staticmethod
    def _decode_image(candidate: Any) -> Optional[Any]:
        try:
            from PIL import Image
        except ImportError:
            return None

        try:
            if candidate is None:
                return None
            if hasattr(candidate, "save") and hasattr(candidate, "size"):
                return candidate.convert("RGB")
            if isinstance(candidate, np.ndarray):
                return Image.fromarray(candidate.astype(np.uint8)).convert("RGB")
            if isinstance(candidate, bytes):
                return Image.open(BytesIO(candidate)).convert("RGB")
            if isinstance(candidate, str):
                text = candidate.strip()
                if not text:
                    return None
                possible_path = Path(text)
                if possible_path.exists():
                    return Image.open(possible_path).convert("RGB")
                if ",base64," in text:
                    text = text.split(",base64,", 1)[1]
                return Image.open(BytesIO(base64.b64decode(text))).convert("RGB")
        except Exception:
            return None
        return None

    def _try_native_render(self) -> Tuple[Optional[Any], List[str]]:
        errors: List[str] = []
        candidates: List[Tuple[str, Any]] = []
        for owner_name, owner in (("namespace", self.namespace), ("instance", self.instance)):
            for method_name in ("render", "get_map_image", "map_image"):
                try:
                    method = getattr(owner, method_name)
                except Exception:
                    continue
                if callable(method):
                    candidates.append((f"{owner_name}.{method_name}", method))

        for name, method in candidates:
            try:
                decoded = self._decode_image(method())
                if decoded is not None:
                    return decoded, errors
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return None, errors

    def save_factory_snapshot(self, label: Optional[str] = None) -> Optional[str]:
        """Save a real-state factory image and sidecar JSON.

        A native FLE map image is preferred when exposed by the installed FLE
        version. Otherwise, the method performs one expensive ``get_entities``
        call and generates a schematic from the actual entity positions. This
        fallback works on a headless Factorio server.
        """
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label or f"step_{self.global_step:07d}")
        png_path = self.snapshot_dir / f"{safe_label}.png"
        json_path = self.snapshot_dir / f"{safe_label}.json"
        render_errors: List[str] = []

        try:
            image, render_errors = self._try_native_render()
            records = self._entity_records()

            if image is not None:
                image.save(png_path)
            else:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 9))
                if records:
                    grouped: Dict[str, List[Dict[str, Any]]] = {}
                    for record in records:
                        grouped.setdefault(record["name"], []).append(record)
                    for entity_name, group in grouped.items():
                        xs = [row["x"] for row in group]
                        ys = [row["y"] for row in group]
                        ax.scatter(xs, ys, s=90, label=entity_name, alpha=0.85)
                        for row in group:
                            if entity_name in TRACKED_ENTITIES:
                                ax.annotate(entity_name, (row["x"], row["y"]), fontsize=8)
                    ax.legend(loc="best", fontsize=8)
                    ax.set_xlabel("Factorio x")
                    ax.set_ylabel("Factorio y")
                    ax.grid(True, alpha=0.25)
                    ax.set_aspect("equal", adjustable="datalim")
                else:
                    ax.text(0.5, 0.5, "No placed entities", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()

                ax.set_title(
                    f"Factory snapshot | global step {self.global_step} | "
                    f"score {self.production_score:.2f} | return {self.episode_return:.2f}"
                )
                inventory_text = "\n".join(
                    f"{key}: {value:g}" for key, value in self.inventory.items() if value
                ) or "inventory empty"
                fig.text(0.01, 0.01, inventory_text, fontsize=8, va="bottom")
                fig.tight_layout(rect=(0, 0.08, 1, 1))
                fig.savefig(png_path, dpi=160)
                plt.close(fig)

            sidecar = {
                "label": safe_label,
                "episode": self.episode_index,
                "episode_step": self.episode_step,
                "global_step": self.global_step,
                "inventory": dict(self.inventory),
                "entity_counts": dict(self.entity_counts),
                "flags": dict(self.flags),
                "production_score": self.production_score,
                "episode_return": self.episode_return,
                "entities": records,
                "native_render_errors": render_errors,
                "png_path": str(png_path),
            }
            json_path.write_text(json.dumps(json_safe(sidecar), indent=2), encoding="utf-8")
            self.last_snapshot_path = str(png_path)
            return str(png_path)
        except Exception as exc:
            self._record_event(
                {
                    "kind": "snapshot_error",
                    "global_step": self.global_step,
                    "label": safe_label,
                    "error": f"{type(exc).__name__}: {exc}",
                    "native_render_errors": render_errors,
                }
            )
            return None

    # ------------------------------------------------------------------
    # Diagnostics and debugging
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        start = time.perf_counter()
        result: Dict[str, Any] = {
            "address": self.address,
            "port": self.tcp_port,
            "run_dir": str(self.run_dir),
            "action_count": self.action_space.n,
            "observation_shape": self.observation_space.shape,
        }
        try:
            raw = self.instance.rcon_client.send_command("/sc rcon.print(game.tick)")
            result["rcon_tick"] = str(raw).strip()
            result["rcon_ok"] = True
        except Exception as exc:
            result["rcon_ok"] = False
            result["rcon_error"] = f"{type(exc).__name__}: {exc}"

        try:
            obs = self._observation()
            mask = self.action_masks()
            result["observation_valid"] = bool(self.observation_space.contains(obs))
            result["mask_shape"] = list(mask.shape)
            result["valid_action_count"] = int(mask.sum())
            result["noop_valid"] = bool(mask[self.action_index("noop")])
        except Exception as exc:
            result["state_validation_error"] = f"{type(exc).__name__}: {exc}"

        result["health_check_seconds"] = time.perf_counter() - start
        return result

    def save_debug_bundle(self, reason: str, *, exception: Optional[BaseException] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.debug_dir / f"debug_{reason}_{timestamp}.json"
        payload = {
            "reason": reason,
            "exception": None if exception is None else f"{type(exception).__name__}: {exception}",
            "episode": self.episode_index,
            "episode_step": self.episode_step,
            "global_step": self.global_step,
            "inventory": self.inventory,
            "entity_counts": self.entity_counts,
            "flags": self.flags,
            "production_score": self.production_score,
            "last_raw_output": self.last_raw_output[-self.raw_output_limit :],
            "recent_events": list(self.recent_events),
            "health": self.health_check(),
        }
        path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        return str(path)

    def _record_event(self, event: Dict[str, Any]) -> None:
        event = {"timestamp": datetime.now().isoformat(timespec="milliseconds"), **event}
        self.recent_events.append(event)
        if not self.debug:
            return
        self._debug_buffer.append(event)
        if len(self._debug_buffer) >= self.debug_flush_every:
            self.flush_debug()

    def flush_debug(self) -> None:
        if not self._debug_buffer:
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            for event in self._debug_buffer:
                handle.write(json.dumps(json_safe(event), ensure_ascii=False) + "\n")
        self._debug_buffer.clear()

    @staticmethod
    def _info_for_log(info: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "environment_episode": info.get("environment_episode"),
            "monitor_episode": info.get("episode"),
            "global_step": info.get("global_step"),
            "inventory": info.get("inventory"),
            "entity_counts": info.get("entity_counts"),
            "flags": info.get("flags"),
            "production_score": info.get("production_score"),
            "snapshot_path": info.get("snapshot_path"),
        }


__all__ = [
    "ACTIONS",
    "ActionSpec",
    "DEFAULT_STARTER_INVENTORY",
    "FactorioTransitionEnv",
    "RewardParts",
    "StepTiming",
    "TRACKED_ENTITIES",
    "TRACKED_ITEMS",
    "inventory_to_dict",
    "production_stats_summary",
    "score_to_float",
]
