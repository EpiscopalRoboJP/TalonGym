# Export, distill, and calibration

Deployment artifact is a **Road Runner 1.0 Actions** snippet (inches, official FTC coordinates). Neural-net export is a demo, not on-robot inference.

## Road Runner from the CLI

```bash
python -m talongym replay --seed 0
```

Writes `var/last_replay.java` from a scripted episode on the active bundle. Open the file and paste into `MeepMeepTesting` / an OpMode.

Dialects (API body `dialect`, Python `to_roadrunner_java`):

| Dialect | Output |
|---------|--------|
| `rr1_actions` (default) | `Actions.runBlocking(drive.actionBuilder(new Pose2d(…)).splineTo(…))` |
| `rr05_trajectory_sequence` | Legacy `trajectorySequenceBuilder` / `lineToLinearHeading` |

Waypoints are taken from the first robot pose each frame and **decimated** (~8 in minimum spacing) so the snippet is a polyline, not 25 Hz chatter.

## Road Runner from the Lab

- Replay → **Export** → **Road Runner 1.0**
- Compare → per evaluation **Road Runner 1.0** (uses that eval’s best replay)

Both call `POST /api/v1/replays/{id}/export/roadrunner`. Training jobs also store a `roadrunner` artifact on the run.

This is still a simulated path. Calibrate the robot preset before treating timings as real ([#calibration](#calibration)). Intended use: paste into **your** autonomous; do not stream actions during a MATCH.

## Distill (feed-forward demo)

LSTM ONNX is not a supported deployment path (`POST /runs/{id}/export/onnx` returns **501** unless `distill=true`).

```bash
python -m talongym distill --steps 256
```

Collects scripted `target_pose` labels, fits a small MLP (torch) or a linear map, writes `var/ckpts/ff_distill.onnx` or `.npz`. Use it to prove the export pipeline, not to run AUTO on the robot.

## Calibration

Fit kinematics from a JSON pose log (FTC Dashboard / WPILOG-shaped samples) onto the default mecanum preset:

```bash
python -m talongym calibrate path/to/log.json --out var/robot_overlay.json
```

Accepted JSON: a list of samples, or `{ "samples": [ … ] }` / `{ "poses": [ … ] }`. Each sample:

| Field | Also accepted |
|-------|----------------|
| `t` | `time` |
| `x` | `x_in` |
| `y` | `y_in` |
| `headingDeg` | `heading_deg` |
| `event`: `"intake"` | used for cycle-time median |

The overlay sets `constraints.maxVelInPerS`, `constraints.maxAccelInPerS2`, `odometry.positionNoiseStdIn`, `mechanisms.intakeCycleTimeS`, and `calibration.rmse`. Merge it into a robot preset (Lab robot builder or PUT `/presets/robot/{id}`) rather than forking season scoring.

## MuJoCo check (not export)

```bash
python -m pip install -e ".[mujoco]"
python -m talongym validate-3d --steps 40
```

Reports planar vs MuJoCo pose RMSE. Slow; not the RL backend.
