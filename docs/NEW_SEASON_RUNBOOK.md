# New-season Kickoff runbook

Goal: go from “official Competition Manual just published” to “a working, if rough, preset in TalonGym” **within days**, without a core-engine pull request.

BIOBUZZ™ presented by RTX (Kickoff 2026-09-12, 12:00 pm ET) is the concrete acceptance test for this runbook. Competition Manual **V1** is the current source. Encode gameplay as *data*, not engine assumptions:

- Scoring elements: **POLLEN** (2.8 in yellow, §9.8) and **NECTAR** (3.6 in red/blue)
- Source: [ftc.game](https://ftc-resources.firstinspires.org/ftc/game) / Competition Manual V1

## Day 0 (Kickoff afternoon): freeze provenance

1. Download the Competition Manual PDF/HTML and record:
   - `manualRevision` (e.g. `V1` or first Team Update id)
   - section versions for ARENA and Game Details
   - source URL and file hash (`sha256`)
   - `effectiveDate`
2. Download field CAD (Onshape / STEP) from the Playing Field Resources page.
3. Create `presets/seasons/biobuzz/` with empty-but-valid JSON files that already pass schema validation:
   - `field.json`, `objects` may live inside field.json
   - `scoring.json`
   - `capabilities.json` (or the `requiredCapabilities` array on the field/scoring presets)
4. Copy `presets/robots/mecanum_meepmeep_defaults.json` unless Kickoff robot defaults change.

Do **not** invent scoring numbers. Leave `verifyAgainstManual: true` on every point-value node until a second person checks Table 10-x.

## Day 1: geometry from CAD, not from memory

1. Extract the 12×12 ft (nominal 144 in) perimeter and origin at field center using the [official FTC coordinate system](https://ftc-docs.firstinspires.org/en/latest/game_specific_resources/field_coordinate_system/field-coordinate-system.html).
2. Place static elements (goals, ramps, gates, spike marks, walls, AprilTags, occluders) as `FieldElement` records with:
   - `collisionShape`, `isOccluder`, `isTrigger`, `tags`
3. Place scoring-element spawns as `GamePieceSpawn` records (type, pose list, physical properties).
4. Add alliance start slots. Use MeepMeep-sane Cartesian poses in inches.
5. Import an official field image as `backgroundAsset` for the 3D viewer (render-only).

If CAD extraction is slow, a tape-measure practice field is acceptable for a *rough* preset; mark `geometryProvenance: "practice-field-approximate"` so the UI shows a stale/approximate banner.

## Day 1–2: rule graph (no engine PR)

Encode scoring as a `ScoringRulesPreset` DAG. Typical Kickoff mapping:

| Manual language | Engine primitive |
|-----------------|------------------|
| “when object of type T enters zone Z” | `trigger: volumeEnter` + `condition: pieceTypeIs` + `action: addScore` |
| “once per robot per period” | `accumulator` with `maxFires: 1` scoped to `actor` + `phase: AUTO` |
| “assessed at end of AUTO” | `trigger: phaseEnd` with `phase: AUTO` |
| “randomized motif / start configuration” | `matchVariables` with `observeVia` sensor, **never** leaked into obs |
| “pattern matches randomized target” | `condition: sequenceEquals` against `matchVariables.motif` |
| “RP if count ≥ threshold” | `action: setFlag` + `rankingPointThresholds` (AUTO-only is a *proxy*) |
| “both alliance robots must …” | `scope: alliance` aggregator |

Keep season nouns (POLLEN, hives, whatever Kickoff names) **only** in the preset JSON `type` strings and display labels.

## Day 2: robot + observation wiring

1. Confirm drivetrain preset (mecanum default).
2. List sensors the AUTO actually needs (AprilTag FOV, intake beam-break, etc.).
3. Map randomized match variables to `observeVia` so the policy sees a `not_yet_observed` sentinel until the sensor would fire.
4. Run `talongym preset lint presets/seasons/biobuzz` — this checks schema, capability matrix, and “privileged leak” tests.

## Day 3: playable rough cut

1. Single-robot AUTO, scripted opponents off, high-level waypoint actions.
2. Watch 20 random-seed replays. If robots drive through scoring elements, fix collision shapes. If scores disagree with a hand-scored video, fix the rule graph — not the physics step.
3. Only then start a short PPO run (lightweight mode) to confirm the env does not NaN / leak motif.

## When you *do* need an engine primitive

Do this **only** if lint reports `unsupportedCapability` and you cannot encode the rule with the existing trigger/condition/accumulator set.

1. Write a failing fixture in `tests/engine/test_capability_matrix.py` that states the new primitive in season-agnostic language (e.g. `requires_mid_episode_geometry_swap`), not “requires BIOBUZZ hive open.”
2. Implement the primitive in the engine with a capability flag.
3. Add the flag to DECODE / INTO THE DEEP / CENTERSTAGE presets as `false` so old presets stay valid.
4. Bump `engineCapabilityVersion`.
5. Re-run the historical capability matrix. A BIOBUZZ PR that changes DECODE scores is a failed gate.

## Stale-preset hygiene during the season

Every Thursday Team Update:

1. Diff the manual. If Table 10-x or ARENA geometry changed, clone the preset to `manualRevision: TUnn` rather than silently editing.
2. Set `supersedes: <old id>` so the UI can banner “this preset may be stale.”
3. Do not mutate a published evaluation report’s preset hash.
