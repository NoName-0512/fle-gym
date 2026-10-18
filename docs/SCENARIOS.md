# Lab Play and Open Play: current public API

Both scenario classes are exposed in `fle_gym.api`, but the `scenario` value is **metadata identifying the expected server scenario**. This wrapper connects over RCON to a Factorio server **already started** with that scenario. It does not start/change a Docker container's scenario.

```python
from fle_gym import FactorioLabEnv, FactorioOpenPlayEnv

# Connect each class to a server already configured for its corresponding scenario.
lab = FactorioLabEnv(rcon_port=27000)
# open_play = FactorioOpenPlayEnv(rcon_port=27001)  # only if such a server actually exists
```

The current CLI accepts `fle cluster start -n 1 -s default_lab_scenario` or `fle cluster start -n 1 -s open_world`, but starting it again may **recreate/replace the existing cluster**. In the user's tests, both scenarios were tested sequentially on the same host RCON port `27000` and container `cluster-factorio_0-1`.

**Caution:** `FactorioEnv.reset()` defaults to `hard_reset=True`, which clears/reinitializes the connected factory. Treat the connected scenario as a disposable research environment. Final `save_world()` normally scrubs Lua storage and should be used near the end of a run.

**Known API limitation:** `make_env(scenario=...)` silently selects the Lab class for any value other than the exact string `open_world`; this should reject unknown scenario names before public release. Passing `scenario='open_world'` to the Lab class (or vice versa) is currently possible and does not verify the underlying server.

**Verified scenario tests:** Lab Play 4096 steps + save export + manual Factorio load; Open Play 4096 steps + save export + successful Factorio load shown by user. These are **protocol/integration tests with a random policy**, not evidence of competent trained-agent factory performance.
