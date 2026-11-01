# Validation evidence and limits

This document records integration and package checks performed on 18 October 2026. Results are grouped by the implementation tested; results from earlier versions are retained as historical evidence.

## Live integration — completed on the validated pre-refactor implementation

| Test | Lab Play | Open Play |
|---|---|---|
| Server scenario | `default_lab_scenario` | `open_world` |
| Host TCP RCON | `27000` | `27000` (same container, reconfigured sequentially) |
| 256-decision run | PASS | PASS |
| 4,096-decision run | PASS; time-limit truncation | PASS; time-limit truncation |
| Native world export | PASS (`lab_full_test.zip`) | PASS (`open_play_4096.zip`) |
| Archive integrity | An earlier Lab ZIP verified; full-run ZIP integrity not explicitly captured in supplied final log | PASS (`unzip -t`) |
| Opened in desktop Factorio | PASS, researcher-confirmed | PASS, researcher-confirmed |
| Controlled iron-plate production | Not separately run in Lab | PASS — burner drill/furnace, fueled; harvested ore inserted, plates collected |

The controlled Open Play test collected **five iron plates**, with inventory feature `1.791759` corresponding to `log1p(5)` and positive machine-production telemetry. It **does not demonstrate** a fully automated, continuously running ore-to-plate factory. Random actions test API stability, not learned-policy performance.

## Additional completed checks for rc5

- Distinct telemetry keys `runtime_action_inventory`, `runtime_action_factory`, and `runtime_action_extended` were checked. A subsequent live regression verified fuel insertion telemetry (`inserted=5` for drill and furnace) and no action errors in that test.
- All **nine** offline regression tests passed.
- `pip check` reported no broken package requirements in the test environment.
- An `rc5` wheel and source distribution were built successfully.
- An `rc5` wheel was force-installed and imported from a separate `site-packages` directory; Lab, Open Play and `make_env` imports passed.

## rc6 integration verification — 18 October 2026

These are **user-executed tests of the renamed rc6 package**. The tests used
Factorio `factoriotools/factorio:2.0.73` on the existing `open_world` server
(`cluster-factorio_0-1`, host TCP RCON `27000`). The running world was approved
for resetting; previously saved ZIPs were not overwritten.

| rc6 check | Observed result |
|---|---|
| Offline regressions | PASS — 13 tests, including the internal-import consistency test |
| Compilation and dependencies | PASS — `compileall`, `pip check`, and Git whitespace checks |
| Packaging | PASS — built rc6 wheel and source distribution |
| Wheel installation | PASS — imported rc6 from `site-packages` after installing the wheel and running from `/tmp`; editable source installation subsequently restored |
| Initial factory progression probe | PARTIAL (exit 2) — four successful actions placed a burner drill and stone furnace and fueled both; harvesting and ore-transfer actions were masked, and machine output remained zero |
| Live API diagnostic | PASS (exit 0) — returned observation `(266,)`, schema length 266, declared observation space `(266,)`, `contains(obs)=True`, 124 actions/registry entries, finite values |
| Four-step Open Play action sequence | PASS — drill placement, furnace placement, drill fueling, furnace fueling; no action errors or invalid actions |
| Runtime telemetry | PASS — `runtime_action_extended` furnace placement `ok=1`; `runtime_action_inventory` reported `ok=1`, `inserted=5` for each fueled machine |

The first attempt at the telemetry script stopped on an assertion before printing
dimensions. The subsequent diagnostic reset returned the documented `(266,)`
observation shape and completed all four operations successfully. No observation
contract change was established by these tests.

These checks establish **rc6 Open Play construction/telemetry integration**.
They do **not** establish rc6 iron-plate production, full factory automation,
or 4,096-step horizon behavior. Separate rc6 Lab Play and save-export checks
are documented below. The prior rc5 iron-plate production and Lab save tests remain historical
evidence for the validated earlier implementation, not independent rc6 results.

## Additional rc6 verification — 18 October 2026

- Ordered rc5-to-rc6 compatibility check: PASS for all 124 action names and 266 observation labels; dimensions and documentation links also passed. This does not establish numerical equivalence of all observations or rewards.
- Dedicated Lab Play server `fle-gym-lab-1`, host RCON `27001`: PASS for reset, 20 masked-action transitions, observation dimensions, finite rewards and Gymnasium return contract.
- Lab save `rc6_lab_smoke_20261018.zip`: PASS for native export, ZIP integrity and researcher-confirmed loading in desktop Factorio.
- Existing Open Play server on RCON `27000` remained running during the separate Lab test.
- GitHub Actions static package checks: PASS on release-preparation commit `7d41a4b`.
- Source distribution contents: PASS for documentation, scripts, notices and manifest. Wheel contents: PASS for renamed internal modules.
- Fresh Python 3.12 virtual environment: PASS for wheel installation, `pip check`, version verification and importing the public API from `site-packages`.
- Initial upstream comparison: no exact matches of 12 consecutive nonblank/non-comment lines across 18 local Python files against 347 installed FLE Python files. This is not proof of independent authorship.

## Commands and reproducibility

```bash
python -m unittest discover -s tests -v
python -m compileall -q fle_gym tests
python -m pip check
python -m build
```

Live tests require a **disposable** server and an active RCON connection:

```bash
python scripts/smoke_test.py --rcon-port 27000 --steps 10
# A longer scenario-specific run (resets the connected server!):
python scripts/validate_scenarios.py --scenario open_world --rcon-port 27000 \
  --steps 4096 --save-world open_play_validation_new
```

Run the `open_world` command only when the server was actually launched with `open_world`. Do not re-use an output save name: `save_world` refuses to overwrite existing archives. Preserve the exact package commit, dependency versions, world configuration, logs and final action counts with each validation result.

## Interpreting validation results

A valid masked action can still fail due to changing game state or runtime rejection; inspect `info` for the actual result. Research readiness and completion need dedicated end-to-end validation. Reward aggregation includes inherited and normalized components. See [Rewards](REWARDS.md) for reward settings and diagnostic semantics.
