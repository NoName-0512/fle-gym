# Reward design and diagnostic semantics

FLE-Gym combines machine-production rewards, research and factory-progression rewards, and penalties for unproductive or unsuccessful actions.

The public environment returns the final scalar reward from `env.step()`, together with a component breakdown in `info["reward_parts"]`. Internal environment layers calculate and adjust these components before returning the transition.

## Default parameters

The public defaults are defined in `fle_gym/_internal/environment_contract.py`. Supported settings can be changed through `semantic_overrides`.

| Parameter | Default | Description |
|---|---:|---|
| `speed` | 10.0 | Factorio game speed |
| `settle_seconds` | 0.04 | Post-action settling interval, scaled by action |
| `training_horizon` | 4096 | Maximum environment decisions before time-limit truncation |
| `production_reward_scale` | 0.35 | Coefficient applied to weighted machine production |
| `research_progress_reward` | 12.0 | Research-progress coefficient before time normalization and capping |
| `research_progress_reward_cap` | 1.0 | Maximum normalized research-progress reward per transition |
| `trigger_technology_reward` | 3.0 | Reward for a newly completed trigger technology |
| `lab_research_completion_reward` | 12.0 | Reward for newly completed laboratory research, with duplicate prevention |
| `research_activity_reward` | 0.02 | Research-activity shaping coefficient |
| `lab_ingredient_potential_reward` | 20.0 | Coefficient for changes in laboratory-ingredient potential |
| `idle_action_penalty` | 0.10 | Penalty for idle actions without credited production or research progress |
| `repeated_idle_penalty` | 0.08 | Additional penalty for consecutive unproductive idle actions, capped at 0.50 per step |
| `machine_reward_reference_seconds` | 1.0 | Reference duration for machine-production normalization |
| `normalize_machine_reward_by_sim_time` | True | Enables normalization by elapsed simulated time |
| `environment_rejection_cooldown_steps` | 24 | Number of steps for which known environment-side rejection actions are masked |
| `production_stats_every` | 3 | Production-state refresh interval in environment steps |
| `infrastructure_state_every` | 3 | Infrastructure-state refresh interval in environment steps |
| `feasibility_state_every` | 2 | Feasibility-state refresh interval in environment steps |

Additional terms are defined in the internal environment layers, including:

| Parameter | Default | Description |
|---|---:|---|
| `openended_step_penalty` | 0.001 | Per-step penalty |
| `first_power_reward` | 10.0 | First-power milestone reward |
| `infrastructure_growth_reward` | 0.25 | Infrastructure-growth shaping coefficient |
| `science_readiness_reward` | 12.0 | Science-readiness shaping coefficient |

Refresh intervals control periodic state collection. Some actions and environment layers request additional state refreshes.

## Machine-production rewards

Machine production uses an item-specific weight table. Credited output is multiplied by the item weight and the production coefficient.

The effective table contains 14 unique items:

| Item | Weight |
|---|---:|
| `iron-ore` | 0.02 |
| `copper-ore` | 0.02 |
| `iron-plate` | 0.40 |
| `copper-plate` | 0.40 |
| `steel-plate` | 1.20 |
| `iron-gear-wheel` | 0.80 |
| `electronic-circuit` | 1.20 |
| `copper-cable` | 0.20 |
| `transport-belt` | 0.70 |
| `inserter` | 1.00 |
| `engine-unit` | 2.00 |
| `automation-science-pack` | 2.50 |
| `logistic-science-pack` | 4.00 |
| `military-science-pack` | 5.00 |

The initial table is defined in `fle_gym/_internal/factorio_openended_env_feasibility.py` and extended in `fle_gym/_internal/factorio_openended_env_extended.py`.

Manual crafting and harvesting are excluded from machine-production credit. Attribution also accounts for delayed manual-output records. Unknown products, fluids, and item transfers do not receive machine-production credit.

Before time normalization, the machine-production component is:

$$
r_{\mathrm{machine,raw}}
=
c_{\mathrm{production}}
\sum_i w_i\,\Delta n_i^{\mathrm{machine}}
$$

where:

- \(c_{\mathrm{production}}\) is `production_reward_scale`, defaulting to 0.35.
- \(w_i\) is the weight of item \(i\).
- \(\Delta n_i^{\mathrm{machine}}\) is the credited machine-produced quantity after attribution corrections.

The public contract maps `production_reward_scale` to the internal `automated_output_reward_scale` argument. Although the internal machine-layer constructor declares a default of 0.50, the public environment supplies 0.35.

### Simulated-time normalization

With `normalize_machine_reward_by_sim_time=True`, positive machine-production rewards are normalized as:

$$
r_{\mathrm{machine}}
=
r_{\mathrm{machine,raw}}
\frac{t_{\mathrm{ref}}}
{\max(1.0,\Delta \mathrm{ticks}/60)}
$$

Here, \(t_{\mathrm{ref}}\) is `machine_reward_reference_seconds`, defaulting to 1.0 second. Elapsed time is measured from Factorio game ticks at 60 ticks per simulated second.

The denominator has a minimum of one second. If tick measurements are unavailable, the environment uses a one-second fallback. Disabling normalization leaves the machine-production component unnormalized.

For example, ten credited iron plates have a weighted output of \(10 \times 0.40 = 4.0\). With the default coefficient, their raw production reward is \(4.0 \times 0.35 = 1.4\). Over two simulated seconds, the normalized machine-production reward is 0.7.

This example describes the machine-production component. Other rewards and penalties also contribute to the transition's total reward.

## Research and factory progression

Research and progression shaping covers technology triggers, laboratory activity, research progress and completion, ingredient readiness, power, infrastructure, and factory milestones.

Research-progress rewards are normalized by elapsed simulated time and capped per transition. Completion and milestone tracking prevents repeated credit where the corresponding reward ledger applies.

Laboratory-ingredient potential rewards depend on changes in ingredient availability. These contributions can be negative when ingredients are consumed.

## Reward diagnostics

The final component breakdown is available in `info["reward_parts"]`. Depending on the transition, it can include:

| Component | Description |
|---|---|
| `machine_production` | Credited machine-output reward after attribution and time normalization |
| `research_progress` | Research-progress contribution |
| `research_completion` | Technology-completion contribution |
| `research_activity` | Research-activity shaping |
| `lab_ingredient_progress` | Change in laboratory-ingredient potential |
| `science_readiness` | Science-readiness shaping |
| `power_milestone` | Power milestone contribution |
| `infrastructure` | Infrastructure shaping |
| `logistics_milestone` | Logistics milestone contribution |
| `progression_milestone` | Factory-progression milestone contribution |
| `idle` | Idle-action contribution |
| `repeated_idle_penalty` | Additional consecutive-idle penalty |
| `step` | Per-step contribution |
| `error` | Action-error contribution |
| `invalid` | Invalid-action contribution |
| `environment_rejection` | Environment-side rejection contribution |
| `residual` | Difference between the returned reward and the named contributions at the aggregation stage |
| `total` | Total recorded by the reward breakdown |

Diagnostic keys can be added, replaced, or removed as internal layers assemble the final transition. Intermediate `automated_output` values may represent the same production contribution later reported as `machine_production`; they should not be added as separate rewards.

Use the scalar `reward` returned by `env.step()` as the transition reward. Use the final `info["reward_parts"]` to inspect its contributions. A `residual` value is an accounting remainder, rather than a separate gameplay objective.

Additional machine-production diagnostics include:

- `machine_production_reward_raw`
- `machine_production_reward_normalized`
- `simulated_seconds_elapsed`
- `lifetime_simulated_seconds`
- `horizon_simulated_seconds`

The elapsed-time diagnostic uses the same one-second floor and fallback as machine-production normalization.

## Observation values and reward values

Observation entries can contain transformed measurements. They should be interpreted according to their feature definitions.

Several inventory and entity features use:

```text
min(10, log1p(max(0, count)))
```

For example, an inventory feature value of approximately `1.791759` corresponds to five items because `log1p(5)` is approximately `1.791759`.

Several cumulative output features use:

```text
log1p(value) / 5
```

An observation named `machine_output_total` therefore does not directly represent a physical item count or the scalar production reward.

Boolean flags, research indicators, power measurements, and factory descriptors use feature-specific representations. See [Observations](OBSERVATIONS.md) for the ordered observation schema.

## Episode time limits

The default horizon is 4,096 environment decisions. Reaching this limit returns `truncated=True`.

Time-limit truncation does not indicate successful research or factory completion. The environment normally continues without a product or research terminal goal, with `terminated=False`.

See [API](API.md) for the reset and step interfaces, and [Validation](VALIDATION.md) for recorded test coverage.
