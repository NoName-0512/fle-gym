# Action catalogue and action masking

This catalogue lists action IDs, names, categories, descriptions and declared prerequisites exposed by `FactorioOpenPlayEnv.action_registry`.

**Contract:** `Discrete(124)`; indices 0–123. An action's `required_items` and `required_entities` reflect metadata, **not a complete feasibility guarantee**. Check `env.action_masks()` each step, and verify changes in Factorio state after execution.

**Important:** Several action `code` fields contain `runtime_*_requested` event markers. These are requests handled by the underlying runtime, not proof of execution or no-op instructions. This table documents declared intent; physical effects require live integration tests.

The 124 IDs are policy-compatible across Lab Play and Open Play; action feasibility is **state dependent**, and the server must be running the selected scenario. Both the action mask and execution can reject actions because of game conditions, so check `info` after each transition.

Category breakdown: automation=10, craft=35, extract=2, harvest=4, insert=4, logistics=4, noop=1, place=2, power=11, research=36, transfer=13, wait=2.

| ID | Name | Category | Required inventory | Required entities | Description |
|---:|---|---|---|---|---|
| 0 | `noop` | `noop` | — | — | Do nothing except let the configured settling interval elapse. |
| 1 | `wait` | `wait` | — | — | Advance one second of Factorio production time. |
| 2 | `place_drill_on_iron` | `place` | `burner-mining-drill` ×1 | — | Place another burner drill on the nearest iron patch. |
| 3 | `place_furnace_near_drill` | `place` | `stone-furnace` ×1 | `burner-mining-drill` ×1 | Place another stone furnace near a burner drill. |
| 4 | `fuel_drill` | `insert` | `coal` ×5 | `burner-mining-drill` ×1 | Insert five coal into the nearest burner mining drill. |
| 5 | `fuel_furnace` | `insert` | `coal` ×5 | `stone-furnace` ×1 | Insert five coal into the nearest stone furnace. |
| 6 | `collect_iron_plate` | `extract` | — | `stone-furnace` ×1 | Collect an available iron plate from the furnace. |
| 7 | `harvest_coal` | `harvest` | — | — | Transactionally mine ten units from a live, reachable, unoccupied resource tile. |
| 8 | `harvest_stone` | `harvest` | — | — | Transactionally mine ten units from a live, reachable, unoccupied resource tile. |
| 9 | `harvest_iron_ore` | `harvest` | — | — | Transactionally mine ten units from a live, reachable, unoccupied resource tile. |
| 10 | `craft_stone_furnace` | `craft` | `stone` ×5 | — | Recovery action: craft one stone furnace. |
| 11 | `insert_hand_mined_ore` | `insert` | `iron-ore` ×1 | `stone-furnace` ×1 | Recovery action: insert one hand-mined iron ore into the furnace. |
| 12 | `place_chest_at_burner_output` | `logistics` | `wooden-chest` ×1 | `burner-mining-drill` ×1 | Place a chest at the nearest burner drill's drop position. |
| 13 | `place_chest_at_electric_output` | `logistics` | `wooden-chest` ×1 | `electric-mining-drill` ×1 | Place a chest at the nearest electric drill's drop position. |
| 14 | `transfer_chest_ore_to_furnace` | `transfer` | — | `wooden-chest` ×1, `stone-furnace` ×1 | Move available iron ore from the nearest output chest to a furnace. |
| 15 | `place_offshore_pump` | `power` | `offshore-pump` ×1 | — | Place an offshore pump at the nearest water tile. |
| 16 | `place_boiler_near_pump` | `power` | `boiler` ×1 | `offshore-pump` ×1 | Place a boiler in buildable space near an offshore pump. |
| 17 | `connect_pump_to_boiler` | `power` | `pipe` ×1 | `offshore-pump` ×1, `boiler` ×1 | Connect the nearest offshore pump to the nearest boiler with pipes. |
| 18 | `fuel_boiler` | `power` | `coal` ×10 | `boiler` ×1 | Insert ten coal into the nearest boiler. |
| 19 | `place_steam_engine_near_boiler` | `power` | `steam-engine` ×1 | `boiler` ×1 | Place a steam engine in buildable space near the boiler. |
| 20 | `connect_boiler_to_engine` | `power` | `pipe` ×1 | `boiler` ×1, `steam-engine` ×1 | Connect the nearest boiler to the nearest steam engine with pipes. |
| 21 | `place_small_electric_pole` | `power` | `small-electric-pole` ×1 | `steam-engine` ×1 | Place a persistent source pole beside the steam engine using the verified Factorio runtime builder. |
| 22 | `place_electric_drill_on_iron` | `power` | `electric-mining-drill` ×1 | — | Place an electric mining drill on the nearest iron patch. |
| 23 | `connect_electric_drill_to_power` | `power` | `small-electric-pole` ×2 | `steam-engine` ×1, `electric-mining-drill` ×1 | Connect every currently unpowered electric mining drill to the steam network using verified runtime pole routes. |
| 24 | `craft_lab` | `research` | — | — | Craft one laboratory. |
| 25 | `craft_automation_science` | `research` | — | — | Craft up to ten automation science packs. |
| 26 | `place_lab` | `research` | `lab` ×1 | `steam-engine` ×1 | Place a laboratory near the steam engine without requiring a pre-existing pole. The power-connection action is atomic and creates any required poles afterwards. |
| 27 | `connect_lab_to_power` | `research` | `small-electric-pole` ×2 | `steam-engine` ×1, `lab` ×1 | Atomically create and auto-connect a verified electric-pole route from the steam engine to the laboratory. |
| 28 | `insert_automation_science` | `research` | `automation-science-pack` ×10 | `lab` ×1 | Transactionally move ten automation science packs from the agent inventory into the nearest laboratory. |
| 29 | `research_automation` | `research` | — | `lab` ×1 | Select automation as the current research target. |
| 30 | `research_logistics` | `research` | — | `lab` ×1 | Select logistics as the current research target. |
| 31 | `research_steel_processing` | `research` | — | `lab` ×1 | Select steel processing as the current research target. |
| 32 | `wait_research` | `research` | — | — | Advance five seconds of game time for power, production, and research. |
| 33 | `place_electric_drill_on_copper` | `power` | `electric-mining-drill` ×1 | — | Place an electric mining drill on the nearest copper patch. |
| 34 | `harvest_copper_ore` | `harvest` | — | — | Transactionally mine ten units from a live, reachable, unoccupied resource tile. |
| 35 | `insert_copper_ore` | `insert` | `copper-ore` ×1 | `stone-furnace` ×1 | Insert one copper ore into an available stone furnace. |
| 36 | `collect_copper_plate` | `extract` | — | `stone-furnace` ×1 | Collect one available copper plate from a stone furnace. |
| 37 | `craft_iron_gears` | `craft` | — | — | Craft up to ten iron gear wheels. |
| 38 | `transfer_copper_chest_ore_to_furnace` | `transfer` | — | `wooden-chest` ×1, `stone-furnace` ×1 | Move copper ore from any output chest into a compatible furnace. |
| 39 | `craft_electronic_circuits` | `craft` | — | — | Craft up to ten electronic circuits. |
| 40 | `craft_transport_belts` | `craft` | — | — | Craft up to ten transport belts. |
| 41 | `craft_inserters` | `craft` | — | — | Craft up to four inserters. |
| 42 | `craft_assembling_machine` | `craft` | — | — | Craft one assembling machine 1. |
| 43 | `build_iron_smelting_line` | `logistics` | — | — | Build one powered iron mining/smelting line with belts, ore and fuel inserters, a furnace, and an output chest. |
| 44 | `build_copper_smelting_line` | `logistics` | — | — | Build one powered copper mining/smelting line with belts, ore and fuel inserters, a furnace, and an output chest. |
| 45 | `build_gear_assembly_cell` | `automation` | — | — | Place and power an iron-gear assembler with an input chest, output belts, inserters, and an output chest. |
| 46 | `build_science_assembly_cell` | `automation` | — | — | Place and power an automation-science assembler with two input chests, output belts, inserters, and an output chest. |
| 47 | `stock_gear_assembler` | `transfer` | — | — | Move available iron plates into the gear assembler input chest. |
| 48 | `stock_science_assembler` | `transfer` | — | — | Move available copper plates and machine-produced gears into the automation-science assembler input chests. |
| 49 | `feed_lab_from_science_output` | `research` | — | — | Move machine-produced red science from its output chest into a lab. |
| 50 | `connect_factory_power` | `power` | — | — | Connect all currently unpowered drills, inserters, assemblers, and labs to the existing steam network with verified pole routes. |
| 51 | `craft_copper_cables` | `craft` | — | — | Queue ten copper-cable crafts. |
| 52 | `craft_pipes` | `craft` | — | — | Queue ten pipe crafts. |
| 53 | `craft_underground_belts` | `craft` | — | — | Queue two underground-belt crafts. |
| 54 | `craft_splitters` | `craft` | — | — | Queue two splitter crafts. |
| 55 | `craft_long_handed_inserters` | `craft` | — | — | Queue four long-handed inserters. |
| 56 | `craft_fast_inserters` | `craft` | — | — | Queue four fast inserters. |
| 57 | `craft_steel_chests` | `craft` | — | — | Queue two steel chests. |
| 58 | `craft_medium_electric_poles` | `craft` | — | — | Queue four medium electric poles. |
| 59 | `craft_steel_furnace` | `craft` | — | — | Queue one steel furnace. |
| 60 | `craft_assembling_machine_2` | `craft` | — | — | Queue one assembling machine 2. |
| 61 | `collect_engine_units` | `transfer` | — | — | Collect completed engine units from factory output chests into the agent inventory. |
| 62 | `craft_logistic_science` | `craft` | — | — | Queue five logistic-science crafts (normally ten packs). |
| 63 | `craft_military_science` | `craft` | — | — | Queue five military-science crafts when unlocked. |
| 64 | `craft_storage_tank` | `craft` | — | — | Queue one storage tank. |
| 65 | `craft_pumpjack` | `craft` | — | — | Queue one pumpjack. |
| 66 | `craft_oil_refinery` | `craft` | — | — | Queue one oil refinery. |
| 67 | `craft_chemical_plants` | `craft` | — | — | Queue two chemical plants. |
| 68 | `craft_repair_packs` | `craft` | — | — | Queue five repair packs. |
| 69 | `craft_lamps` | `craft` | — | — | Queue four lamps. |
| 70 | `craft_firearm_magazines` | `craft` | — | — | Queue ten firearm magazines. |
| 71 | `craft_piercing_magazines` | `craft` | — | — | Queue five piercing magazines. |
| 72 | `craft_grenades` | `craft` | — | — | Queue five grenades. |
| 73 | `craft_stone_walls` | `craft` | — | — | Queue ten stone walls. |
| 74 | `craft_gun_turrets` | `craft` | — | — | Queue two gun turrets. |
| 75 | `craft_radars` | `craft` | — | — | Queue two radars. |
| 76 | `craft_train_stops` | `craft` | — | — | Queue two train stops. |
| 77 | `craft_rail_signals` | `craft` | — | — | Queue four rail signals. |
| 78 | `craft_rail_chain_signals` | `craft` | — | — | Queue four rail chain signals. |
| 79 | `craft_locomotive` | `craft` | — | — | Queue one locomotive. |
| 80 | `craft_cargo_wagons` | `craft` | — | — | Queue two cargo wagons. |
| 81 | `build_cable_assembly_cell` | `automation` | — | — | Build one powered copper-cable assembly cell. |
| 82 | `build_circuit_assembly_cell` | `automation` | — | — | Build one powered electronic-circuit assembly cell. |
| 83 | `build_belt_assembly_cell` | `automation` | — | — | Build one powered transport-belt assembly cell. |
| 84 | `build_inserter_assembly_cell` | `automation` | — | — | Build one powered inserter assembly cell. |
| 85 | `build_logistic_science_assembly_cell` | `automation` | — | — | Build one powered logistic-science-pack assembly cell. |
| 86 | `build_engine_assembly_cell` | `automation` | — | — | Build one powered engine-unit assembly cell. |
| 87 | `build_military_science_assembly_cell` | `automation` | — | — | Build one powered military-science-pack assembly cell. |
| 88 | `stock_cable_assembler` | `transfer` | — | — | Stock the copper-cable assembly cell from available buffers. |
| 89 | `stock_circuit_assembler` | `transfer` | — | — | Stock the electronic-circuit assembly cell from available buffers. |
| 90 | `stock_belt_assembler` | `transfer` | — | — | Stock the transport-belt assembly cell from available buffers. |
| 91 | `stock_inserter_assembler` | `transfer` | — | — | Stock the inserter assembly cell from available buffers. |
| 92 | `stock_logistic_science_assembler` | `transfer` | — | — | Stock the logistic-science-pack assembly cell from available buffers. |
| 93 | `stock_engine_assembler` | `transfer` | — | — | Stock the engine-unit assembly cell from available buffers. |
| 94 | `stock_military_science_assembler` | `transfer` | — | — | Stock the military-science-pack assembly cell from available buffers. |
| 95 | `build_steel_smelting_cell` | `automation` | — | — | Build a fuelled iron-plate to steel-plate smelting cell. |
| 96 | `stock_steel_smelting_cell` | `transfer` | — | — | Stock the steel smelting cell with iron plates and fuel. |
| 97 | `feed_labs_all_science` | `research` | — | — | Move available automation and logistic science into laboratories. |
| 98 | `wait_crafting` | `wait` | — | — | Advance one simulated second while queued hand crafts and machines progress. runtime removes the former five-second advantage. |
| 99 | `research_logistic_science_pack` | `research` | — | — | Queue logistic-science-pack if it is a live, available laboratory technology. |
| 100 | `research_military` | `research` | — | — | Queue military if it is a live, available laboratory technology. |
| 101 | `research_lamp` | `research` | — | — | Queue lamp if it is a live, available laboratory technology. |
| 102 | `research_stone_wall` | `research` | — | — | Queue stone-wall if it is a live, available laboratory technology. |
| 103 | `research_gun_turret` | `research` | — | — | Queue gun-turret if it is a live, available laboratory technology. |
| 104 | `research_electric_mining_drill` | `research` | — | — | Queue electric-mining-drill if it is a live, available laboratory technology. |
| 105 | `research_toolbelt` | `research` | — | — | Queue toolbelt if it is a live, available laboratory technology. |
| 106 | `research_automation_2` | `research` | — | — | Queue automation-2 if it is a live, available laboratory technology. |
| 107 | `research_logistics_2` | `research` | — | — | Queue logistics-2 if it is a live, available laboratory technology. |
| 108 | `research_advanced_material_processing` | `research` | — | — | Queue advanced-material-processing if it is a live, available laboratory technology. |
| 109 | `research_engine` | `research` | — | — | Queue engine if it is a live, available laboratory technology. |
| 110 | `research_fluid_handling` | `research` | — | — | Queue fluid-handling if it is a live, available laboratory technology. |
| 111 | `research_electric_energy_distribution_1` | `research` | — | — | Queue electric-energy-distribution-1 if it is a live, available laboratory technology. |
| 112 | `research_fast_inserter` | `research` | — | — | Queue fast-inserter if it is a live, available laboratory technology. |
| 113 | `research_railway` | `research` | — | — | Queue railway if it is a live, available laboratory technology. |
| 114 | `research_advanced_oil_processing` | `research` | — | — | Queue advanced-oil-processing if it is a live, available laboratory technology. |
| 115 | `research_circuit_network` | `research` | — | — | Queue circuit-network if it is a live, available laboratory technology. |
| 116 | `research_military_science_pack` | `research` | — | — | Queue military-science-pack if it is a live, available laboratory technology. |
| 117 | `research_physical_projectile_damage_1` | `research` | — | — | Queue physical-projectile-damage-1 if it is a live, available laboratory technology. |
| 118 | `research_weapon_shooting_speed_1` | `research` | — | — | Queue weapon-shooting-speed-1 if it is a live, available laboratory technology. |
| 119 | `research_automated_rail_transportation` | `research` | — | — | Queue automated-rail-transportation if it is a live, available laboratory technology. |
| 120 | `research_braking_force_1` | `research` | — | — | Queue braking-force-1 if it is a live, available laboratory technology. |
| 121 | `research_flammables` | `research` | — | — | Queue flammables if it is a live, available laboratory technology. |
| 122 | `research_plastics` | `research` | — | — | Queue plastics if it is a live, available laboratory technology. |
| 123 | `research_sulfur_processing` | `research` | — | — | Queue sulfur-processing if it is a live, available laboratory technology. |

## How to verify against an installed release

```python
from fle_gym import FactorioOpenPlayEnv
env = FactorioOpenPlayEnv(rcon_port=27000)
try:
    assert len(env.action_registry) == 124
    assert all(env.action_name(i) == action['name']
               for i, action in enumerate(env.action_registry))
finally:
    env.close()
```

## Programmatic introspection

```python
from fle_gym import action_names, actions

assert len(action_names()) == 124
assert len(actions()) == 124
for i, spec in enumerate(actions()):
    assert action_names()[i] == spec["name"]
```

With a running server, `env.action_masks()` returns a Boolean `(124,)` array. `env.action_index(name)` and `env.action_name(index)` translate between names and IDs. This table describes *declared metadata*; masks and effects must be evaluated in live Factorio states.
