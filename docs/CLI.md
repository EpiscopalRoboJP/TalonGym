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
python -m talongym train --easy
python -m talongym train --training biobuzz_auto_workstation --steps 8192 --n-envs 16
python -m talongym train --allow-scripted
python -m talongym train --algo rllib_ppo --steps 2048
```

| Option | Type | Default |
|--------|------|---------|
| `--easy` | flag | false |
| `--training` | id | active default |
| `--steps` | int | 8192, or the easy preset budget with `--easy` |
| `--n-envs` | int | training preset `nEnvs` (easy: autodetected) |
| `--allow-scripted` | flag | false |
| `--algo` | `recurrent_ppo` \| `rllib_ppo` | from the training preset |

Prints `compute=… nEnvs=… training=… steps=…` then `algo=… steps=… ckpt=…`. How-to: [TRAINING.md](TRAINING.md).

## `detect`

```bash
python -m talongym detect
```

Prints CPU count, RAM, CUDA/Metal availability, the resolved `lightweight_cpu` / `workstation` / `cloud` profile, recommended `nEnvs`, the matching `*_easy` training id, and the `torchDevice` (`cuda`, `mps`, or `cpu`) that training will actually run on. Override the profile with `TALONGYM_COMPUTE_PROFILE`, or the device with `TALONGYM_TORCH_DEVICE`.

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

Rolls the scripted policy, writes Road Runner 1.0 Java to `var/last_replay.java`, prints `frames=… trueScore=… export=…`. Paste that file into an AUTO OpMode; the Control Hub runs it. How-to: [EXPORT.md](EXPORT.md).

## `defaults`

```bash
python -m talongym defaults
python -m talongym defaults --training biobuzz_auto_lightweight
python -m talongym defaults --field biobuzz_2026_field_v1 --robot mecanum_biobuzz_4cap --scoring biobuzz_2026_scoring_v1
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

Compares planar 2D vs MuJoCo on a short chassis motion; prints pose RMSE. The chassis-only adapter is not the BIOBUZZ train backend (`MujocoFieldBackend` + `collisionAsset`).

## `import-field-cad`

```bash
python -m talongym import-field-cad
python -m talongym import-field-cad --page https://ftc-resources.firstinspires.org/ftc/archive/2027/field
python -m talongym import-field-cad --step "C:\Users\242440\Downloads\am-5850 BIOBUZZ.step"
```

Writes `assets/seasons/<slug>/field.glb` (Lab) and `field_mjcf.xml` (MuJoCo AABBs). `--step` tessellates a local official STEP with `trimesh`/`cascadio` (`pip install -e ".[cad]"`). Without `--step`, it tries the archive page; HubSpot often hides the file, in which case field preset AABBs are tessellated. Raw STEP is copied to `var/cad/` and is not committed. Physics stays AABB even when the Lab mesh is official CAD, so robots can still drive under the hive.

## `import-robot-cad`

```bash
python -m talongym import-robot-cad --robot team_hood_v1 --file path/to/robot.stl
```

Accepts GLB, glTF, STL, OBJ, or STEP (`pip install -e ".[cad]"`; STEP needs cascadio). Writes `var/assets/robots/<id>/visual.glb` and `collision.stl`, prints bbox + footprint JSON. Paste those paths onto the robot preset (`visualAsset`, `collisionAsset`, `chassis.footprint`). Lab **Robot** does the same via `POST /api/v1/presets/robot/{id}/model`.

## `distill`

```bash
python -m talongym distill
python -m talongym distill --steps 256
```

Clones scripted waypoint targets into a feed-forward ONNX (or `.npz` if torch is missing). Demo only — not LSTM, not a field path. AUTO on the field is a pasted Road Runner OpMode. Output under `var/ckpts/ff_distill.onnx`.
