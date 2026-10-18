from __future__ import annotations

"""Real Factorio screenshots through a connected graphical client.

The FLE worker is headless and cannot render. A normal Factorio multiplayer
client receives ``game.take_screenshot`` requests and writes PNG files to its
``script-output`` directory. WSL then copies those files into the run folder.

Two capture modes are supported:

* ``screen``: 1920x1080 by default, includes the complete Factorio GUI and is
  centred on the selected player's current view position.
* ``factory``: GUI-free overview dynamically framed around every placed entity.
* ``both``: saves one image of each type at every screenshot interval.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    try:
        import gym
    except ImportError:  # pragma: no cover
        class _MinimalWrapper:
            def __init__(self, env: Any) -> None:
                self.env = env

            def reset(self, **kwargs: Any):
                return self.env.reset(**kwargs)

            def step(self, action: Any):
                return self.env.step(action)

            def close(self) -> None:
                return self.env.close()

            def __getattr__(self, name: str) -> Any:
                return getattr(self.env, name)

        class _MinimalGym:
            Wrapper = _MinimalWrapper

        gym = _MinimalGym()


_SCREENSHOT_QUEUED = "__FLE_CLIENT_SCREENSHOT_QUEUED__"
_SCREENSHOT_ERROR = "__FLE_CLIENT_SCREENSHOT_ERROR__"
_PLAYER_PREFIX = "__FLE_CONNECTED_PLAYER__"
_TECH_SCREENSHOT_QUEUED = "__FLE_TECH_SCREENSHOT_QUEUED__"


def _lua_string(value: str) -> str:
    return (
        '"'
        + str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        + '"'
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "snapshot"


def _normalise_entity_record(entity: Any, index: int) -> Dict[str, Any]:
    if isinstance(entity, Mapping):
        name = entity.get("name") or entity.get("prototype") or entity.get("type") or "unknown"
        position = entity.get("position")
    else:
        name = (
            getattr(entity, "name", None)
            or getattr(entity, "prototype", None)
            or getattr(entity, "type", None)
            or "unknown"
        )
        position = getattr(entity, "position", None)

    if isinstance(position, Mapping):
        x = position.get("x", 0.0)
        y = position.get("y", 0.0)
    else:
        x = getattr(position, "x", getattr(entity, "x", 0.0))
        y = getattr(position, "y", getattr(entity, "y", 0.0))

    try:
        x = float(x)
    except Exception:
        x = 0.0
    try:
        y = float(y)
    except Exception:
        y = 0.0

    raw_name = str(name)
    if "." in raw_name:
        raw_name = raw_name.rsplit(".", 1)[-1]
    raw_name = raw_name.replace("_", "-").lower()
    return {"index": int(index), "name": raw_name, "x": x, "y": y}


@dataclass
class ScreenshotResult:
    requested: bool
    success: bool
    label: str
    mode: str
    player_name: str
    relative_client_path: str
    client_source_path: Optional[str]
    copied_path: Optional[str]
    center_x: Optional[float]
    center_y: Optional[float]
    zoom: Optional[float]
    width: int
    height: int
    show_gui: bool
    rcon_response: str
    error: str
    elapsed_seconds: float
    connected_players: List[Dict[str, Any]]
    entity_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FactorioClientScreenshotManager:
    """Request real screenshots from a connected graphical Factorio client."""

    def __init__(
        self,
        *,
        rcon_client: Any,
        player_name: str,
        client_script_output_dir: Path | str,
        destination_dir: Path | str,
        run_name: str,
        width: int = 4096,
        height: int = 2304,
        fixed_zoom: Optional[float] = None,
        screen_zoom: Optional[float] = None,
        min_zoom: float = 0.03,
        max_zoom: float = 1.0,
        margin_tiles: float = 24.0,
        wait_timeout_seconds: float = 10.0,
        poll_seconds: float = 0.10,
        show_entity_info: bool = True,
        factory_daytime: Optional[float] = 0.0,
        strict: bool = False,
        screen_frame_factory: bool = True,
    ) -> None:
        if not player_name.strip():
            raise ValueError("player_name cannot be blank")
        if width <= 0 or height <= 0:
            raise ValueError("Screenshot dimensions must be positive")
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be positive")

        self.rcon_client = rcon_client
        self.player_name = player_name.strip()
        self.client_script_output_dir = Path(client_script_output_dir)
        self.destination_dir = Path(destination_dir)
        self.run_name = _safe_name(run_name)
        self.width = int(width)
        self.height = int(height)
        self.fixed_zoom = fixed_zoom
        self.screen_zoom = screen_zoom
        self.min_zoom = float(min_zoom)
        self.max_zoom = float(max_zoom)
        self.margin_tiles = float(margin_tiles)
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self.poll_seconds = max(0.02, float(poll_seconds))
        self.show_entity_info = bool(show_entity_info)
        self.factory_daytime = None if factory_daytime is None else float(factory_daytime)
        self.strict = bool(strict)
        self.screen_frame_factory = bool(screen_frame_factory)

        self.destination_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.destination_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def list_connected_players(self) -> List[Dict[str, Any]]:
        command = (
            "/sc for _,p in pairs(game.connected_players) do "
            f"rcon.print({_lua_string(_PLAYER_PREFIX)}.."
            "p.index..'|'..p.name..'|'..p.position.x..'|'..p.position.y) end"
        )
        raw = str(self.rcon_client.send_command(command) or "")
        players: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            marker = line.find(_PLAYER_PREFIX)
            if marker < 0:
                continue
            payload = line[marker + len(_PLAYER_PREFIX):].lstrip(": ")
            parts = payload.split("|")
            if len(parts) < 4:
                continue
            try:
                players.append(
                    {
                        "index": int(float(parts[0])),
                        "name": parts[1],
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                    }
                )
            except Exception:
                continue
        return players

    def player_is_connected(self) -> bool:
        return any(row.get("name") == self.player_name for row in self.list_connected_players())

    def _frame(
        self,
        entity_records: Sequence[Mapping[str, Any]],
        fallback_position: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float]:
        xs: List[float] = []
        ys: List[float] = []
        for record in entity_records:
            try:
                xs.append(float(record["x"]))
                ys.append(float(record["y"]))
            except Exception:
                continue

        if xs and ys:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            extent_x = max_x - min_x + 2.0 * self.margin_tiles
            extent_y = max_y - min_y + 2.0 * self.margin_tiles
        elif fallback_position is not None:
            center_x, center_y = fallback_position
            extent_x = extent_y = 32.0
        else:
            center_x = center_y = 0.0
            extent_x = extent_y = 32.0

        if self.fixed_zoom is not None:
            zoom = float(self.fixed_zoom)
        else:
            zoom_x = self.width / max(32.0 * extent_x, 1.0)
            zoom_y = self.height / max(32.0 * extent_y, 1.0)
            zoom = max(self.min_zoom, min(self.max_zoom, min(zoom_x, zoom_y)))
        return float(center_x), float(center_y), float(zoom)

    def _wait_and_copy(self, source_path: Path, copied_path: Path) -> None:
        deadline = time.perf_counter() + self.wait_timeout_seconds
        while time.perf_counter() < deadline:
            if source_path.exists() and source_path.stat().st_size > 0:
                copied_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, copied_path)
                return
            time.sleep(self.poll_seconds)
        raise TimeoutError(
            "Screenshot request was queued but the PNG did not appear at "
            f"{source_path}. Keep the graphical client connected and not minimised."
        )

    def capture(
        self,
        label: str,
        *,
        mode: str = "factory",
        entity_records: Optional[Sequence[Mapping[str, Any]]] = None,
        fallback_position: Optional[Tuple[float, float]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> ScreenshotResult:
        mode = str(mode).strip().lower()
        if mode not in {"screen", "factory"}:
            raise ValueError("mode must be 'screen' or 'factory'")

        start = time.perf_counter()
        safe_label = _safe_name(label)
        records = list(entity_records or [])
        center_x: Optional[float] = None
        center_y: Optional[float] = None
        zoom: Optional[float] = None
        show_gui = mode == "screen"
        if mode == "factory" or (mode == "screen" and self.screen_frame_factory and records):
            center_x, center_y, zoom = self._frame(records, fallback_position)

        relative = Path("fle_rl") / self.run_name / f"{safe_label}.png"
        relative_posix = relative.as_posix()
        source_path = self.client_script_output_dir / relative
        copied_path = self.destination_dir / f"{safe_label}.png"
        sidecar_path = self.metadata_dir / f"{safe_label}.json"

        players: List[Dict[str, Any]] = []
        response = ""
        result: ScreenshotResult
        try:
            players = self.list_connected_players()
            player = next((row for row in players if row.get("name") == self.player_name), None)
            if player is None:
                available = ", ".join(row.get("name", "?") for row in players) or "none"
                raise RuntimeError(
                    f"Graphical screenshot player {self.player_name!r} is not connected. "
                    f"Connected players: {available}"
                )

            if (mode == "factory" or self.screen_frame_factory) and not records:
                center_x, center_y, zoom = self._frame(
                    [],
                    (float(player.get("x", 0.0)), float(player.get("y", 0.0))),
                )

            source_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.exists():
                source_path.unlink()
            if copied_path.exists():
                copied_path.unlink()

            options = [
                "player=p",
                "by_player=p",
                "surface=p.surface",
                f"resolution={{x={self.width},y={self.height}}}",
                f"path={_lua_string(relative_posix)}",
                f"show_gui={'true' if show_gui else 'false'}",
                f"show_entity_info={'true' if self.show_entity_info else 'false'}",
                "show_cursor_building_preview=false",
                "anti_alias=false",
                "hide_clouds=true",
                "hide_fog=true",
                "force_render=true",
            ]
            if mode == "factory":
                options.extend(
                    [
                        f"position={{x={center_x:.6f},y={center_y:.6f}}}",
                        f"zoom={zoom:.6f}",
                    ]
                )
                if self.factory_daytime is not None:
                    options.append(f"daytime={self.factory_daytime:.6f}")
            elif center_x is not None and center_y is not None and zoom is not None:
                # Frame the factory instead of the spectator's remote camera.
                # show_gui=true still requests the complete client interface.
                options.extend(
                    [
                        f"position={{x={center_x:.6f},y={center_y:.6f}}}",
                        f"zoom={zoom:.6f}",
                    ]
                )
            elif self.screen_zoom is not None:
                options.append(f"zoom={float(self.screen_zoom):.6f}")

            command = (
                "/sc "
                f"local p=game.get_player({_lua_string(self.player_name)}); "
                "if not p or not p.connected then "
                f"rcon.print({_lua_string(_SCREENSHOT_ERROR + ':player-not-connected')}); "
                "else game.take_screenshot{" + ",".join(options) + "}; "
                "game.set_wait_for_screenshots_to_finish(); "
                f"rcon.print({_lua_string(_SCREENSHOT_QUEUED + ':')}..{_lua_string(relative_posix)}); end"
            )
            response = str(self.rcon_client.send_command(command) or "")
            if _SCREENSHOT_ERROR in response:
                raise RuntimeError(response.strip())
            self._wait_and_copy(source_path, copied_path)

            result = ScreenshotResult(
                requested=True,
                success=True,
                label=safe_label,
                mode=mode,
                player_name=self.player_name,
                relative_client_path=relative_posix,
                client_source_path=str(source_path),
                copied_path=str(copied_path),
                center_x=center_x,
                center_y=center_y,
                zoom=zoom,
                width=self.width,
                height=self.height,
                show_gui=show_gui,
                rcon_response=response,
                error="",
                elapsed_seconds=time.perf_counter() - start,
                connected_players=players,
                entity_count=len(records),
            )
        except Exception as exc:
            result = ScreenshotResult(
                requested=True,
                success=False,
                label=safe_label,
                mode=mode,
                player_name=self.player_name,
                relative_client_path=relative_posix,
                client_source_path=str(source_path),
                copied_path=None,
                center_x=center_x,
                center_y=center_y,
                zoom=zoom,
                width=self.width,
                height=self.height,
                show_gui=show_gui,
                rcon_response=response,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.perf_counter() - start,
                connected_players=players,
                entity_count=len(records),
            )
            if self.strict:
                payload = result.to_dict()
                payload["extra_metadata"] = dict(extra_metadata or {})
                sidecar_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
                raise

        payload = result.to_dict()
        payload["extra_metadata"] = dict(extra_metadata or {})
        sidecar_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return result

    def capture_technology_tree(
        self,
        label: str = "technology_tree",
        *,
        selected_technology: Optional[str] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> ScreenshotResult:
        """Capture Factorio's technology graph for the selected player."""
        start = time.perf_counter()
        safe_label = _safe_name(label)
        relative = Path("fle_rl") / self.run_name / f"{safe_label}.png"
        source_path = self.client_script_output_dir / relative
        copied_path = self.destination_dir / f"{safe_label}.png"
        sidecar_path = self.metadata_dir / f"{safe_label}.json"
        players: List[Dict[str, Any]] = []
        response = ""
        try:
            players = self.list_connected_players()
            if not any(row.get("name") == self.player_name for row in players):
                raise RuntimeError(f"Screenshot player {self.player_name!r} is not connected")
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.unlink(missing_ok=True)
            copied_path.unlink(missing_ok=True)
            options = [
                f"player=game.get_player({_lua_string(self.player_name)})",
                f"path={_lua_string(relative.as_posix())}",
            ]
            if selected_technology:
                options.append(f"selected_technology={_lua_string(selected_technology)}")
            command = (
                "/sc game.take_technology_screenshot{" + ",".join(options) + "}; "
                "game.set_wait_for_screenshots_to_finish(); "
                f"rcon.print({_lua_string(_TECH_SCREENSHOT_QUEUED)})"
            )
            response = str(self.rcon_client.send_command(command) or "")
            self._wait_and_copy(source_path, copied_path)
            result = ScreenshotResult(
                True, True, safe_label, "technology", self.player_name,
                relative.as_posix(), str(source_path), str(copied_path),
                None, None, None, self.width, self.height, False,
                response, "", time.perf_counter() - start, players, 0,
            )
        except Exception as exc:
            result = ScreenshotResult(
                True, False, safe_label, "technology", self.player_name,
                relative.as_posix(), str(source_path), None,
                None, None, None, self.width, self.height, False,
                response, f"{type(exc).__name__}: {exc}",
                time.perf_counter() - start, players, 0,
            )
            if self.strict:
                raise
        payload = result.to_dict()
        payload["extra_metadata"] = dict(extra_metadata or {})
        sidecar_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return result


def _unwrap(env: Any) -> Any:
    current = env
    seen = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return getattr(current, "unwrapped", current)


class ActualFactorioScreenshotWrapper(gym.Wrapper):
    """Capture client-screen and/or full-factory screenshots periodically."""

    def __init__(
        self,
        env: Any,
        *,
        player_name: str,
        client_script_output_dir: Path | str,
        screenshot_every: int = 100,
        screenshot_every_episodes: int = 0,
        capture_mode: str = "factory",
        width: int = 4096,
        height: int = 2304,
        zoom: Optional[float] = None,
        screen_zoom: Optional[float] = None,
        wait_timeout_seconds: float = 10.0,
        strict: bool = False,
        screen_frame_factory: bool = True,
    ) -> None:
        super().__init__(env)
        capture_mode = str(capture_mode).strip().lower()
        if capture_mode not in {"screen", "factory", "both"}:
            raise ValueError("capture_mode must be screen, factory, or both")
        self.base_env = _unwrap(env)
        self.screenshot_every = max(0, int(screenshot_every))
        self.screenshot_every_episodes = max(0, int(screenshot_every_episodes))
        self.capture_mode = capture_mode
        self.completed_episodes = 0

        base_run_dir = Path(getattr(self.base_env, "run_dir", "runs/fle_rl"))
        self.run_dir = base_run_dir
        self.snapshot_dir = base_run_dir / "actual_screenshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_manager = FactorioClientScreenshotManager(
            rcon_client=self.base_env.instance.rcon_client,
            player_name=player_name,
            client_script_output_dir=client_script_output_dir,
            destination_dir=self.snapshot_dir,
            run_name=base_run_dir.name,
            width=width,
            height=height,
            fixed_zoom=zoom,
            screen_zoom=screen_zoom,
            wait_timeout_seconds=wait_timeout_seconds,
            strict=strict,
            screen_frame_factory=screen_frame_factory,
        )
        self.last_actual_screenshot: Optional[Dict[str, Any]] = None
        self.last_actual_screenshots: List[Dict[str, Any]] = []

    def action_masks(self):
        return self.env.action_masks()

    def get_action_mask(self):
        if hasattr(self.env, "get_action_mask"):
            return self.env.get_action_mask()
        return self.env.action_masks()

    def _entity_records(self) -> List[Dict[str, Any]]:
        if hasattr(self.base_env, "_entity_records"):
            try:
                return list(self.base_env._entity_records())
            except Exception:
                pass
        namespace = getattr(self.base_env, "namespace", None)
        if namespace is None:
            return []
        try:
            raw_entities = namespace.get_entities()
        except Exception:
            return []
        return [_normalise_entity_record(entity, index) for index, entity in enumerate(raw_entities or [])]

    def _snapshot_metadata(self) -> Dict[str, Any]:
        return {
            "global_step": int(getattr(self.base_env, "global_step", 0)),
            "environment_episode": int(getattr(self.base_env, "episode_index", 0)),
            "episode_step": int(getattr(self.base_env, "episode_step", 0)),
            "inventory": dict(getattr(self.base_env, "inventory", {})),
            "entity_counts": dict(getattr(self.base_env, "entity_counts", {})),
            "production_score": float(getattr(self.base_env, "production_score", 0.0)),
            "episode_return": float(getattr(self.base_env, "episode_return", 0.0)),
            "infrastructure_state": dict(getattr(self.base_env, "infrastructure_state", {})),
        }

    def save_factory_snapshot(self, label: Optional[str] = None) -> Optional[str]:
        global_step = int(getattr(self.base_env, "global_step", 0))
        base_label = label or f"step_{global_step:07d}"
        records = self._entity_records()
        metadata = self._snapshot_metadata()
        modes = [self.capture_mode] if self.capture_mode != "both" else ["screen", "factory"]
        results: List[ScreenshotResult] = []
        for mode in modes:
            result_label = base_label if len(modes) == 1 else f"{base_label}_{mode}"
            results.append(
                self.screenshot_manager.capture(
                    result_label,
                    mode=mode,
                    entity_records=records,
                    extra_metadata=metadata,
                )
            )

        self.last_actual_screenshots = [result.to_dict() for result in results]
        successful = [result for result in results if result.success and result.copied_path]
        self.last_actual_screenshot = successful[-1].to_dict() if successful else results[-1].to_dict()
        primary = next((result for result in successful if result.mode == "factory"), None)
        primary = primary or (successful[0] if successful else None)
        path = None if primary is None else primary.copied_path
        if hasattr(self.base_env, "last_snapshot_path"):
            self.base_env.last_snapshot_path = path
        return path

    def save_technology_snapshot(
        self,
        label: str = "technology_tree",
        *,
        selected_technology: Optional[str] = None,
    ) -> Optional[str]:
        result = self.screenshot_manager.capture_technology_tree(
            label,
            selected_technology=selected_technology,
            extra_metadata=self._snapshot_metadata(),
        )
        return result.copied_path

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def reset_screenshot_episode_counter(self) -> None:
        self.completed_episodes = 0

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        global_step = int(info.get("global_step", getattr(self.base_env, "global_step", 0)))
        done = bool(terminated or truncated)
        path: Optional[str] = None

        if done:
            self.completed_episodes += 1
            info["completed_screenshot_episodes"] = self.completed_episodes
            if self.screenshot_every_episodes and self.completed_episodes % self.screenshot_every_episodes == 0:
                path = self.save_factory_snapshot(
                    f"episode_{self.completed_episodes:04d}_final_step_{global_step:07d}"
                )
                info["episode_screenshot_path"] = path

        if path is None and self.screenshot_every and global_step % self.screenshot_every == 0:
            path = self.save_factory_snapshot(f"step_{global_step:07d}")

        # Include both successful captures and non-fatal failed-capture
        # metadata in the transition info.  This makes a lost graphical client
        # auditable without turning an otherwise healthy PPO run into a failed
        # experiment.
        if self.last_actual_screenshots:
            paths = [
                row.get("copied_path")
                for row in self.last_actual_screenshots
                if row.get("success") and row.get("copied_path")
            ]
            failed = [row for row in self.last_actual_screenshots if not row.get("success")]
            if path is not None:
                info["snapshot_path"] = path
                info["actual_screenshot_path"] = path
            info["actual_screenshot_paths"] = paths
            info["actual_screenshots"] = list(self.last_actual_screenshots)
            info["actual_screenshot"] = self.last_actual_screenshot
            info["actual_screenshot_failed"] = bool(failed)
            info["actual_screenshot_errors"] = [str(row.get("error", "")) for row in failed]

        return observation, reward, terminated, truncated, info

    def list_connected_players(self) -> List[Dict[str, Any]]:
        return self.screenshot_manager.list_connected_players()

    def screenshot_health_check(self) -> Dict[str, Any]:
        players = self.list_connected_players()
        target = self.screenshot_manager.player_name
        return {
            "target_player": target,
            "target_connected": any(row.get("name") == target for row in players),
            "connected_players": players,
            "client_script_output_dir": str(self.screenshot_manager.client_script_output_dir),
            "destination_dir": str(self.snapshot_dir),
            "capture_mode": self.capture_mode,
            "resolution": [self.screenshot_manager.width, self.screenshot_manager.height],
            "screenshot_every_steps": self.screenshot_every,
            "screenshot_every_episodes": self.screenshot_every_episodes,
        }

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.env, name)
