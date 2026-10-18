from __future__ import annotations

"""Public Gymnasium-compatible API for the Factorio transition environment."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    try:
        import gym
    except ImportError:  # lightweight import-time fallback for static/package checks
        class _Wrapper:
            def __init__(self, env):
                self.env = env
                self.action_space = env.action_space
                self.observation_space = env.observation_space
            def __getattr__(self, name):
                return getattr(self.env, name)
            def close(self):
                return self.env.close()
        class _Gym:
            Wrapper = _Wrapper
        gym = _Gym()

from ._bootstrap import activate_internal_path

activate_internal_path()
from env_factory import ScreenshotConfig, make_env_factory, unwrap_env_factory  # type: ignore  # noqa: E402
from environment_contract import SEMANTIC_ARGUMENT_DEFAULTS, build_environment_kwargs_contract  # type: ignore  # noqa: E402


class FactorioEnv(gym.Wrapper):
    """Thin public wrapper over the validated Factorio environment.

    The Factorio server must already be running.  ``scenario`` is recorded for
    clarity; scenario creation is intentionally owned by the orchestration
    layer because one Gym environment maps to one Factorio/RCON server.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        rcon_port: int = 27000,
        address: str = "localhost",
        scenario: str = "default_lab_scenario",
        seed: Optional[int] = None,
        run_dir: str | Path = "runs/fle_gym",
        run_name: str = "external_agent",
        semantic_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        args: Dict[str, Any] = dict(SEMANTIC_ARGUMENT_DEFAULTS)
        if semantic_overrides:
            unknown = sorted(set(semantic_overrides) - set(args))
            if unknown:
                raise KeyError(f"Unknown semantic override(s): {unknown}")
            args.update(semantic_overrides)
        kwargs = build_environment_kwargs_contract(
            args,
            run_dir=str(Path(run_dir)),
            run_name=str(run_name),
        )
        env = make_env_factory(
            screenshot_config=ScreenshotConfig(enabled=False),
            address=address,
            tcp_port=int(rcon_port),
            **kwargs,
        )
        super().__init__(env)
        self.scenario = str(scenario)
        self.initial_seed = seed
        self._base = unwrap_env_factory(env)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = self.initial_seed
        opts = {"hard_reset": True}
        if options:
            opts.update(options)
        return self.env.reset(seed=seed, options=opts)

    def action_masks(self) -> np.ndarray:
        return np.asarray(self._base.action_masks(), dtype=bool)

    @property
    def action_registry(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self._base.actions]

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._base.actions)

    @property
    def observation_schema(self) -> tuple[str, ...]:
        fields = getattr(self._base, "observation_feature_names", None) or getattr(self._base, "observation_fields", None)
        if fields:
            return tuple(str(x) for x in fields)
        # The validated contract is vector based.  When an older internal class
        # does not expose names, return stable positional labels rather than
        # pretending semantic names are known.
        n = int(self.observation_space.shape[0])
        return tuple(f"feature_{i:03d}" for i in range(n))

    def action_name(self, index: int) -> str:
        return str(self._base.action_name(int(index)))

    def action_index(self, name: str) -> int:
        return int(self._base.action_index(str(name)))

    def get_metrics(self) -> dict[str, Any]:
        self._base._collect_compact_state(include_production_stats=True)
        self._base._collect_feasibility_state(force=True)
        self._base._collect_infrastructure_state(force=True)
        return dict(self._base._base_info())

    @property
    def base_env(self) -> Any:
        """Access the validated internal environment for advanced adapters."""
        return self._base


class FactorioLabEnv(FactorioEnv):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("scenario", "default_lab_scenario")
        super().__init__(**kwargs)


class FactorioOpenPlayEnv(FactorioEnv):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("scenario", "open_world")
        super().__init__(**kwargs)


def make_env(*, scenario: str = "default_lab_scenario", **kwargs: Any) -> FactorioEnv:
    if scenario not in ("default_lab_scenario", "open_world"):
        raise ValueError(f"Unsupported Factorio scenario: {scenario!r}. "
                         "Expected default_lab_scenario or open_world.")
    cls = FactorioOpenPlayEnv if scenario == "open_world" else FactorioLabEnv
    return cls(scenario=scenario, **kwargs)


__all__ = [
    "FactorioEnv",
    "FactorioLabEnv",
    "FactorioOpenPlayEnv",
    "make_env",
]


# Explicit full-game save API. The running environment must still be open.
def _save_world(self, name, output_path, *, container=None, rcon_port=None, scrub_storage=True):
    from fle_gym.world_save import save_factorio_world
    return save_factorio_world(self, name, output_path, container=container,
                               rcon_port=rcon_port, scrub_storage=scrub_storage)

FactorioEnv.save_world = _save_world
