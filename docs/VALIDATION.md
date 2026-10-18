# Validation evidence and limits

This record separates **previously completed live validation**, **package checks**, and **work still required after the rc6 internal-module rename**. Evidence summarized here is from user-run tests on 18 October 2026; it is not an independent reproduction on a new machine.

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
Lab Play integration, 4,096-step horizon behavior, or post-refactor native save
export. The prior rc5 iron-plate production and Lab save tests remain historical
evidence for the validated earlier implementation, not independent rc6 results.

## rc6 refactor: remaining release gates

The internal renaming has passed offline, packaging, installed-wheel and a short
live Open Play action/telemetry test. The following remain before the final
release is claimed to be fully validated:

1. On a disposable **Lab Play** server, verify a fresh reset, observations,
   action masks, steps, rewards and diagnostics on rc6.
2. Compare the **ordered** action names and observation feature names (not just
   their lengths) against the validated rc5 contract; inspect normalization
   and effective reward semantics for unintended changes.
3. If further code changes affect saving, rerun native save archive integrity
   and desktop-game loading. Existing pre-refactor save evidence does not
   establish a post-refactor test.
4. Run GitHub Actions after pushing the release-preparation branch; audit
   distribution contents and ideally install in a **fresh** virtual environment.
5. Complete per-file upstream/code-provenance and license/ownership review
   before any public publication. Test research completion and long-run
   reliability separately where those capabilities are claimed.

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

## Scientific interpretation

A valid masked action can still fail due to changing game state or runtime rejection; inspect `info` for the actual result. Research readiness and completion need dedicated end-to-end validation. Reward aggregation contains inherited and normalized terms; see [Rewards](REWARDS.md) before claiming a fully verified mathematical reward decomposition.
