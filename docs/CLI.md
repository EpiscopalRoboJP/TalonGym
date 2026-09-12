# CLI reference

```bash
python -m talongym --help
talongym --help
```

All commands load the active bundle from [PRESETS.md](PRESETS.md) (`presets/defaults.json` plus `var/defaults.json`) unless noted.

## `lab`

```bash
python -m talongym lab
python -m talongym lab --host 127.0.0.1 --port 8765
```

Serves FastAPI (`/api/v1/…`) and, if `web/dist` exists, the Lab SPA. See [LAB.md](LAB.md).

## `train`

```bash
python -m talongym train
python -m talongym train --steps 8192 --n-envs 4
python -m talongym train --allow-scripted
python -m talongym train --algo rllib_ppo --steps 2048
```

| Option | Type | Default |
|--------|------|---------|
| `--steps` | int | 8192 |
| `--n-envs` | int | min(training preset `nEnvs` or 4, 8) |
| `--allow-scripted` | flag | false |
| `--algo` | `recurrent_ppo` \| `rllib_ppo` | `recurrent_ppo` |

Prints `algo=… steps=… ckpt=…`. How-to: [TRAINING.md](TRAINING.md).

## `evaluate`

```bash
python -m talongym evaluate
python -m talongym evaluate --trials 32
```

Runs the **scripted** AUTO on held-out seeds starting at 10_000_000. Prints mean, 95% bootstrap CI, p10, n, and `bestLabelEligible` (true only for n ≥ 500). How-to: [EVALUATION.md](EVALUATION.md).

## `replay`

```bash
python -m talongym replay
python -m talongym replay --seed 0
```

Rolls the scripted policy, writes Road Runner 1.0 Java to `var/last_replay.java`, prints `frames=… trueScore=… export=…`. How-to: [EXPORT.md](EXPORT.md).

## `defaults`

```bash
python -m talongym defaults
python -m talongym defaults --training biobuzz_auto_lightweight
python -m talongym defaults --field decode_2025_field_tu32 --robot mecanum_meepmeep_defaults --scoring decode_2025_scoring_tu32
```

No flags: print the resolved bundle JSON (`fieldId`, `robotId`, `scoringId`, `trainingId`, `season`). With flags: write `var/defaults.json` after validating ids and that scoring’s `fieldPresetId` matches the field.

## `preset`

```bash
python -m talongym preset
```

Loads every field, robot, scoring, and training document on disk. Prints `ok <kind> <id>` or raises on schema/capability errors.

## `calibrate`

```bash
python -m talongym calibrate path/to/poses.json
python -m talongym calibrate path/to/poses.json --out var/robot_overlay.json
```

Fits `maxVel` / `maxAccel` / odometry noise / intake cycle from a JSON pose log onto `mecanum_meepmeep_defaults`. Default output: `var/robot_overlay.json`. See [EXPORT.md](EXPORT.md)#calibration.

## `validate-3d`

```bash
python -m pip install -e ".[mujoco]"
python -m talongym validate-3d
python -m talongym validate-3d --steps 40
```

Compares planar 2D vs MuJoCo on a short chassis motion; prints pose RMSE. Not used for bulk RL.

## `distill`

```bash
python -m talongym distill
python -m talongym distill --steps 256
```

Clones scripted waypoint targets into a feed-forward ONNX (or `.npz` if torch is missing). Demo only — not LSTM, not on-robot inference. Output under `var/ckpts/ff_distill.onnx`.
