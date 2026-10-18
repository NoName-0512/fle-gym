# Reward design and diagnostic semantics

This is a **source-derived implementation reference**, not a formally audited proof that every component is free of double counting. The public wrapper constructs a `FactorioOpenEndedEnvResearch` adapter on top of multiple environment layers, so the **returned `reward` and `info['reward_parts']`** are authoritative for each step. Intermediate ancestors' reward values may be replaced or adjusted by descendants.

## Default semantic parameters

The following are configured by `fle_gym/_internal/environment_contract.py` and converted into environment keyword arguments. Values can be changed using `semantic_overrides` where supported.

| Parameter | Default | Meaning / caveat |
|---|---:|---|
| `speed` | 10.0 | Factorio game-speed setting |
| `settle_seconds` | 0.04 | Post-action settling interval (scaled by action) |
| `training_horizon` | 4096 | Time-limit truncation; not an endogenous terminal goal |
| `production_reward_scale` | 0.35 | Weighted machine-output reward coefficient; downstream normalization/correction applies |
| `research_progress_reward` | 12.0 | Research-progress reward coefficient before final time normalization and cap |
| `research_progress_reward_cap` | 1.0 | Caps final research-progress component per transition |
| `trigger_technology_reward` | 3.0 | A newly completed trigger technology in a dedicated runtime layer |
| `lab_research_completion_reward` | 12.0 | Newly completed laboratory technology, subject to duplicate prevention |
| `research_activity_reward` | 0.02 | Small research-activity shaping term |
| `lab_ingredient_potential_reward` | 20.0 | Changes in laboratory-ingredient potential; may be negative if inventory is consumed |
| `idle_action_penalty` | 0.10 | Idle actions with no credited production/research progress in one runtime layer |
| `repeated_idle_penalty` | 0.08 | Applied for consecutive unproductive idle actions, with per-step cap 0.50 |
| `machine_reward_reference_seconds` | 1.0 | Target duration used for machine reward normalization |
| `normalize_machine_reward_by_sim_time` | True | Machine reward is divided by `max(1, simulated_seconds_elapsed)` and multiplied by reference time |
| `environment_rejection_cooldown_steps` | 24 | Masks known environment-side rejection actions for a period |
| `production_stats_every` | 3 | Production state refresh cadence; some descendants force additional refreshes |
| `infrastructure_state_every` | 3 | Infrastructure refresh cadence |
| `feasibility_state_every` | 2 | Feasibility refresh cadence |

Other inherited terms are configured at environment layer level, notably `openended_step_penalty=0.001`, `first_power_reward=10.0`, `infrastructure_growth_reward=0.25`, and `science_readiness_reward=12.0`. Inspect the final class MRO and `info['reward_parts']` to confirm effective values and contributions.

## Canonical reward diagnostics

The last canonical aggregator (`factorio_openended_env_recipe.py`, `_canonical_reward_parts`) groups rewards into: `machine_production`, `research_progress`, `research_completion`, `research_activity`, `power_milestone`, `infrastructure`, `logistics_milestone`, `progression_milestone`, `idle`, `step`, `error`, `invalid`, `legacy_base`, `environment_rejection`, `residual`, `total`. Descendant runtime layers also account for `lab_ingredient_progress`, `science_readiness`, `repeated_idle_penalty`, and research-completion adjustments.

The `residual` is deliberately computed as `returned_reward - sum(named_parts)` in the canonical aggregator. Its existence means the named list must not automatically be interpreted as a complete independent additive scientific model until the final returned diagnostics are reconciled. Compare `reward`, `info['reward_parts']['total']`, and the sum of individual components on real transitions; audit duplicate/overwritten telemetry keys first.

## Machine production and time normalization

`factorio_openended_env_machine.py` normalizes positive machine rewards as approximately:

\[
 r_{\mathrm{machine,norm}} = r_{\mathrm{machine,raw}} \times \frac{t_{\mathrm{ref}}}{\max(1,t_{\mathrm{sim}})}
\]

where `t_ref` defaults to one second. Further strict attribution in `factorio_openended_env_recipe.py` removes output wrongly attributed to machines when it could be from manual craft/harvest actions.

## Observations are **not** raw item counts

Several inventory/entity features use `min(10, log1p(max(0,count)))`. Thus the observed iron-plate inventory value `1.791759` is consistent with `log(1+5)` = five iron plates—not 1.791759 physical plates.

Several cumulative output features are scaled with `log1p(value)/5`, so the observed `machine_output_total=0.389753` is a normalized measurement, not a count or reward of 0.389753 units. Flags and technology researched/available indicators are Boolean, whereas power/production and compactness descriptors use different normalizations. Use `docs/OBSERVATIONS.md` for feature order and **audit the corresponding `_observation()` implementations** for exact scaling by group.

## End-of-episode semantics

The public configuration sets a 4096-step horizon. `truncated=True` on step 4096 is a time limit, not successful research or game completion. The source explicitly says there is no product/research terminal goal; `terminated=False` is normally expected.

## Scientific reporting and limitations

Provide an exact, source-reviewed expression for each final reward component, verify no duplicate credit between runtime layers, and include how weighted item production, scientific technology milestones, handcraft exclusion, action masks and simulated-time normalization affect policy incentives. Cite the exact source commit and release tag used for the experiment.

## Reproducibility

Report the FLE-Gym commit/tag, upstream FLE version, scenario, reward overrides, horizon, policy/action-masking method, and the exact interpretation of `info["reward_parts"]`. The rc6 changes rename internal Python identifiers only; no reward redesign is intended. This should be checked with a live reward/telemetry comparison before publication.
