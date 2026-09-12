# Export, distill, and calibration

**Always and only this path to the field:** train offline → export a trajectory → paste into an AUTO OpMode → the Control Hub runs that OpMode.

The match-bound artifact is a **Road Runner 1.0 Actions** snippet (inches, official FTC coordinates). Neural-net export (`distill`) is a pipeline demo, not on-robot inference and not a deployment path. TalonGym does not stream actions during a MATCH.

## Road Runner from the CLI

```bash
python -m talongym replay --seed 0
```

Writes `var/last_replay.java` from a scripted episode on the active bundle. Paste that snippet into an AUTO OpMode (MeepMeep is fine for rehearsal). The Control Hub, not TalonGym, runs it in a MATCH.

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

This is still a simulated path. Calibrate the robot preset before treating timings as real ([#calibration](#calibration)). Then paste into **your** AUTO OpMode so the Control Hub runs it. Do not stream actions during a MATCH.

## Distill (feed-forward demo)

LSTM ONNX is not a supported deployment path (`POST /runs/{id}/export/onnx` returns **501** unless `distill=true`).

```bash
python -m talongym distill --steps 256
```

Collects scripted `target_pose` labels, fits a small MLP (torch) or a linear map, writes `var/ckpts/ff_distill.onnx` or `.npz`. Use it to prove the export pipeline. Do not load it on the Control Hub; AUTO on the field is the pasted Road Runner OpMode only.

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
