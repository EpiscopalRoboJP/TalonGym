# Presets

A **bundle** is field + robot + scoring, usually pinned by a training-run JSON. Seasons are data. Engine identifiers are capabilities (`planar_drive`, `retained_queue`, …), not ARTIFACT / POLLEN / PIXEL.

## Layout

```
presets/
  defaults.json                 # repo default bundle ids
  robots/
    mecanum_meepmeep_defaults.json
    mecanum_biobuzz_4cap.json
  training/
    decode_auto_lightweight.json
    into_the_deep_auto_lightweight.json
    centerstage_auto_lightweight.json
    biobuzz_auto_lightweight.json
  seasons/
    decode_2025/                # field.json, scoring.json
    into_the_deep_2024/
    centerstage_2023/
    biobuzz_2026/               # plus season README
schemas/                        # Draft 2020-12, normative
  field-preset.schema.json
  robot-preset.schema.json
  scoring-rules-preset.schema.json
  training-run-config.schema.json
```

Kind is inferred from JSON shape (`fieldSizeIn`, `drivetrain`, scoring `nodes`+`scoreChannels`, training `algorithm`). Ids are the document `id` field, not the filename.

## Switch the active bundle

```bash
python -m talongym defaults --training decode_auto_lightweight
python -m talongym defaults --training into_the_deep_auto_lightweight
python -m talongym defaults --training centerstage_auto_lightweight
python -m talongym defaults --training biobuzz_auto_lightweight
python -m talongym defaults
```

Resolution: `presets/defaults.json`, then overlay `var/defaults.json`. Lab **Set as default** is the same PUT `/defaults`.

Scoring documents may declare `fieldPresetId`. `defaults` rejects a mismatch.

## Seasons shipped

| Season | Field id | Scoring id | Manual |
|--------|----------|------------|--------|
| DECODE™ presented by RTX | `decode_2025_field_tu32` | `decode_2025_scoring_tu32` | TU32 |
| INTO THE DEEP℠ | `into_the_deep_2024_field` | `into_the_deep_2024_scoring` | archive |
| CENTERSTAGE℠ | `centerstage_2023_field` | `centerstage_2023_scoring` | archive |
| BIOBUZZ™ presented by RTX | `biobuzz_2026_field_v1` | `biobuzz_2026_scoring_v1` | V1 |

BIOBUZZ notes and placeholders: [presets/seasons/biobuzz_2026/README.md](../presets/seasons/biobuzz_2026/README.md). Kickoff process: [NEW_SEASON_RUNBOOK.md](NEW_SEASON_RUNBOOK.md).

`LATEST_KNOWN_MANUAL` in the loader flags stale `provenance.manualRevision`. Team Updates: clone the preset with a new revision (`TUnn`) rather than silently editing a hash that evaluations already stored.

## Lint

```bash
python -m talongym preset
```

Validates every on-disk preset against schema and `requiredCapabilities`. Unknown capability ids fail as `unsupportedCapabilities`.

Lab Field builder also POSTs `/presets/validate`. After a Kickoff week, a new season directory must lint with **zero** engine/Python sim identifier leaks (`tests/presets/test_no_season_leak.py`).

## Training JSON (what trainers actually edit)

Normative schema: [`schemas/training-run-config.schema.json`](../schemas/training-run-config.schema.json).

Important keys:

- `presets.fieldId` / `robotId` / `scoringId`
- `presets.teammatePolicy`: `none` \| `scripted` \| `shared_reward` \| `independent`
- `presets.opponentPolicy`: `none` \| `static` \| `scripted` \| `frozen_policy` (loads `var/ckpts/recurrent_ppo.zip` if present)
- `algorithm.name`: `recurrent_ppo` (Lab/CLI default), `ppo`, `rllib_ppo`
- `actionTier`: `high_level_waypoint` or `low_level_velocity`
- `computeProfile`: `lightweight_cpu` \| `workstation` \| `cloud`
- `nEnvs`, `episode.controlHz` (25), `episode.durationS` (30)
- `domainRandomization` + `curriculum` unlocks — [TRAINING.md](TRAINING.md)#curriculum
- `budget.totalEnvSteps`, `wallClockLimitS`, `earlyStopNoImproveSteps`
- `evaluation.nTrials` (500 for a label-eligible report), `heldOutSeedStart`
- `objective`: `mean_true_score` \| `p10_true_score` \| `lcb_true_score`

Point values and geometry belong in scoring/field JSON, not here.

## Coordinate system

Official FTC field coords, inches: origin at field center, +X right when standing at the Red Wall looking at center, +Y away from the Red Wall, +heading CCW. Viewer is MeepMeep-style non-rotated Cartesian. Exports use the same units.

## New season without an engine PR

Follow [NEW_SEASON_RUNBOOK.md](NEW_SEASON_RUNBOOK.md): provenance → CAD geometry → rule DAG → robot sensors → scripted replays → short PPO. Only add an engine capability if lint reports `unsupportedCapability` and the rule cannot be encoded with existing triggers/accumulators.
