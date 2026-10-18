from __future__ import annotations

"""Single runtime environment factory shared by PPO, discrete SAC, and MAP-Elites."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from factorio_client_screenshots import ActualFactorioScreenshotWrapper
from factorio_openended_env_research import FactorioOpenEndedEnvResearch


@dataclass(frozen=True)
class ScreenshotConfig:
    enabled: bool = False
    player_name: str = ""
    client_script_output_dir: str = ""
    screenshot_every: int = 1000
    screenshot_every_horizons: int = 5
    capture_mode: str = "both"
    width: int = 4096
    height: int = 2304
    strict: bool = False
    screen_frame_factory: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.player_name.strip():
            raise ValueError("Screenshot capture requires player_name")
        if not self.client_script_output_dir.strip():
            raise ValueError("Screenshot capture requires client_script_output_dir")
        if self.capture_mode not in {"screen", "factory", "both"}:
            raise ValueError("capture_mode must be screen, factory, or both")


def make_env_factory(
    *, screenshot_config: Optional[ScreenshotConfig] = None, **env_kwargs: Any
) -> Any:
    base_env = FactorioOpenEndedEnvResearch(**env_kwargs)
    config = screenshot_config or ScreenshotConfig(enabled=False)
    config.validate()
    if not config.enabled:
        return base_env
    return ActualFactorioScreenshotWrapper(
        base_env,
        player_name=config.player_name,
        client_script_output_dir=Path(config.client_script_output_dir),
        screenshot_every=max(0, int(config.screenshot_every)),
        screenshot_every_episodes=max(0, int(config.screenshot_every_horizons)),
        capture_mode=config.capture_mode,
        width=int(config.width),
        height=int(config.height),
        strict=bool(config.strict),
        screen_frame_factory=bool(config.screen_frame_factory),
    )


def unwrap_env_factory(env: Any) -> FactorioOpenEndedEnvResearch:
    current = env
    for _ in range(16):
        if isinstance(current, FactorioOpenEndedEnvResearch):
            return current
        if hasattr(current, "base_env"):
            current = current.base_env
            continue
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    raise TypeError(f"Could not unwrap runtime environment from {type(env).__name__}")


def screenshots_enabled(env: Any) -> bool:
    return isinstance(env, ActualFactorioScreenshotWrapper)


__all__ = ["ScreenshotConfig", "make_env_factory", "screenshots_enabled", "unwrap_env_factory"]
