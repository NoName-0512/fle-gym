# FLE-Gym validation evidence — 18 October 2026

This document reports **user-executed live tests**. It is not a claim that the source audit container reran Factorio; archive-integrity and load observations were provided by the researcher.

| Test | Lab | Open Play |
|---|---|---|
| Server scenario | `default_lab_scenario` | `open_world` (confirmed in container command) |
| Host RCON port | `27000` | `27000` (same container was reconfigured sequentially) |
| 256-step test | PASS | PASS |
| 4096-step test | PASS; `truncated=True` at horizon | PASS; `truncated=True` at horizon |
| Native world save | PASS (`lab_full_test.zip`) | PASS (`open_play_4096.zip`) |
| `unzip -t` | Earlier Lab ZIP verified; full Lab archive integrity not explicitly shown in supplied final log | PASS; no ZIP errors |
| Opens in desktop Factorio | PASS, user-confirmed | PASS, user showed loaded map screenshot |
| Factory production | Not separately tested in a Lab-specific controlled production test | PASS: controlled iron-production path in Open Play |

Other supplied evidence: Open Play 256-step save (`open_play_256.zip`) exported as a valid ZIP. A controlled sequence placed a burner drill and furnace, fueled both, harvested and inserted iron ore, advanced game time, and collected iron plates. Machine output observation increased and iron-plate inventory became `1.791759` (`log1p(5)`). The test does **not** prove a fully automated continuously running mining/smelting/logistics loop.

**Cautions:** Random valid-action rollouts are API stability tests, not trained-policy benchmarks. Reset is destructive to the current connected factory. `save_world()` may scrub Lua storage and is best used as a terminal test step. Macro-actions and research completion should receive separate live integration tests.
