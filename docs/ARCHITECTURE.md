# TalonGym Architecture and Implementation Plan

**Status:** buildable specification for a greenfield rewrite.  
**Flagship season:** BIOBUZZ™ presented by RTX, 2026–2027, Competition Manual **V1**.  
**This document is the source of truth for implementation.** JSON Schema files in [`../schemas/`](../schemas/) are normative for preset files.

**Deployment path (always and only):** train offline → export a trajectory → paste into an AUTO OpMode → Control Hub runs that OpMode. Intended-use boundary, hardware modes, and official source links: [README.md](../README.md). Kickoff procedure: [NEW_SEASON_RUNBOOK.md](NEW_SEASON_RUNBOOK.md).

---

## 1. Executive summary

TalonGym is a **season-plugin, Gymnasium-first** FTC Autonomous trainer with a React/Three.js viewer that never runs physics. The engine default is **2.5D** planar contacts. BIOBUZZ AUTO opts into a **MuJoCo mesh field** because launches into elevated CELLs must arc. A team loads a field + robot + scoring-rules preset, trains a policy offline, inspects rollouts, and exports Road Runner 1.0 Actions to paste into an AUTO OpMode. The Control Hub runs that OpMode; TalonGym does not command a robot during a MATCH. “Best” is a **pre-registered statistical objective** over ≥500 held-out trials, never a single lucky episode.

### Decisions (not a menu)

| Decision | Choice | Why |
|----------|--------|-----|
| Physics | **2.5D default** via Python `Planar2DBackend` (`engine: planar2d`). The Rapier crate is a **stub** (`rapier2d-stub`); Box2D is **not built**. BIOBUZZ sets `mesh_field_collision` and runs **MuJoCo 3D** (Y-up inches) against CAD-derived colliders. | BIOBUZZ AUTO is launch-into-cell; robots must drive under the hive. Missing MuJoCo **refuses** a mesh-field training run — no silent planar fallback (Lab scoring/API tests may planar-fallback; training does not). |
| Rapier vs Box2D | **Not shipped.** Health reports `planar2d` while the Rapier crate is a stub. There is no Box2D worker pool. | A batched Rapier `step(n_worlds)` remains the Phase 0 spike; until it meets the gate, Python planar contacts are the 2.5D path. |
| 3D physics | **Opt-in `MujocoFieldBackend`** when the field declares `collisionAsset` / `mesh_field_collision`. The old chassis-only adapter remains for `talongym validate-3d`. | Required for ballistic CELL launches. Viewer still does not run physics. |
| Action space MVP | **High-level waypoint / spline** executed by a Road Runner-like follower | Trains faster; maps onto exportable RR 1.0 segments. `physical_actuators` is a legal training `actionTier` (robot builder). Low-level `(vx, vy, ω)` is Phase 5 transfer/validation. |
| Algorithm | **BIOBUZZ:** `bc_then_ppo` (scripted clone, then asymmetric-critic PPO). Actor stays encoder-only. | Scratch LSTM PPO does not learn a 30s +20 TIP. `grpo` and `rllib_ppo` exist as experimental unused code (no product preset). Not a VLA. |
| Frontend | React + react-three-fiber; FastAPI BFF; **no localStorage** | Matches the product requirement. Persistence is SQLite (MVP) → Postgres (Phase 5). |
| Multi-agent | Gymnasium single-agent MVP; **PettingZoo** is a required dependency wrapping an unused Phase 4 module (`python/talongym/env/petting.py`) | Two robots per alliance is real; it is not the first learning problem. |
| Export | **Road Runner 1.0 `TrajectoryActionBuilder` / Actions** default; 0.5.x `TrajectorySequence` as compatibility | RR 1.0 replaced trajectory sequences. Emitting deprecated APIs as the primary path would strand teams. |
| POLLEN size | **2.8 in** (Competition Manual V1 §9.8) | Pre-Kickoff planning notes said ~3 in; the manual is 2.8 in. |

### What 2.5D means, precisely

- **Simulated in Planar2D (Rapier crate is a stub):** chassis vs walls, chassis vs chassis, floor pieces vs floor pieces, floor pieces vs chassis (pushing). Gravity is off in-plane; friction and restitution are data.
- **Not simulated as rigid-body flight by default:** a piece “in the goal” can be an FSM (`held → launched → in_goal_volume → scored`) with timers. BIOBUZZ mesh seasons keep pieces as free rigid bodies with physical mechanism launch; `retained_queue` remains an engine capability for seasons that need an ordered overflow buffer.
- **Visualizer:** extrudes `WorldState` (2D poses + FSM + queue) into meshes. It is a renderer.

---

## 2. Repository / folder structure

Greenfield layout. Do not preserve a zone-based kinematic lab as the physics core.

```
talongym/
  README.md
  docs/
    ARCHITECTURE.md          # this file
    NEW_SEASON_RUNBOOK.md
  schemas/                   # Draft 2020-12 JSON Schema (normative)
    field-preset.schema.json
    robot-preset.schema.json
    scoring-rules-preset.schema.json
    training-run-config.schema.json
    api/                     # OpenAPI fragments live here when generated
  presets/
    seasons/
      biobuzz_2026/          # V1 flagship
    robots/
      mecanum_meepmeep_defaults.json
      mecanum_biobuzz_4cap.json
    training/
      biobuzz_auto_{lightweight,workstation,cloud,easy}.json
  assets/
    seasons/biobuzz_2026/    # shipped tessellation: field.glb, collision/*.stl, pieces/, cad_manifest.json; rebuild via import-field-cad
  engine/                    # Rust crate: Rapier2D batched worlds
    Cargo.toml
    src/lib.rs
  python/
    talongym/
      sim/                   # WorldState, PhysicsBackend protocol; Planar2D default; Rapier stub; no Box2D
      rules/                 # compile ScoringRulesPreset → DAG, evaluate per tick
      robot/                 # drivetrain FK/IK, motor model, sensors, mechanism FSMs
      env/                   # FTCAutoEnv, wrappers, PettingZoo
      training/              # RecurrentPPO loop, vectorization, curriculum, ONNX export
      eval/                  # statistical harness, reports
      export/                # RR 1.0 Actions + optional 0.5.x sequence
      api/                   # FastAPI app, jobs, sqlite
      calibrate/             # real-log kinematics fit
  web/                       # React + Vite + R3F
    src/
      routes/                # replay, train, field-builder, robot-builder, compare
      scene/                 # Three.js field; no physics
      api/
  tests/
    engine/
    env/
    rules/
    api/
    presets/
  ftc/                       # optional on-robot notes; exported Java/Kotlin snippets are generated, not hand-maintained
```

### Layer contracts (pure, no cycles)

```mermaid
flowchart LR
  presets[Preset JSON] --> compiler[Preset compiler]
  compiler --> world[WorldState plus FSMs]
  compiler --> dag[Compiled rule DAG]
  world --> planar[Planar2D / MuJoCo mesh]
  planar --> world
  world --> sensors[Sensor models]
  sensors --> obs[Policy observation]
  world --> dag
  dag --> channels[trueScore shaping rpProxy]
  obs --> env[FTCAutoEnv]
  channels --> env
  env --> train[RecurrentPPO]
  world --> replay[Replay records]
  replay --> api[FastAPI]
  api --> web[React renderer]
```

- `WorldState` is the simulation source of truth.
- The web client **never** integrates physics or scoring.
- Ground-truth motif / randomized match variables are **not** fields on the observation dict unless the matching `observeVia` sensor has fired.

### Engine primitives (strict superset target)

The engine implements only these capabilities. Seasons declare which they need. Adding a season must not add a BIOBUZZ-shaped code path.

| Capability id | Meaning |
|---------------|---------|
| `planar_drive` | Mecanum / tank / swerve on a flat field |
| `static_colliders` | Walls and field elements |
| `dynamic_game_pieces` | Disk/circle bodies with type ids |
| `trigger_volumes` | Enter/exit events, scoring and mechanism |
| `occluders` | Block vision rays (OBELISK, truss, submersible, …) |
| `apriltag_vision` | FOV cone, range, occlusion, tag id |
| `randomized_match_variable` | Per-reset enum/int drawn from preset domain |
| `partial_observability_sentinel` | `not_yet_observed` until `observeVia` |
| `n_stage_scoring` | Accumulators + sequenced conditions |
| `end_of_phase_scoring` | `phaseEnd` trigger |
| `alliance_aggregate_scoring` | `allAlliance` / `anyAlliance` |
| `ranking_point_thresholds` | Threshold tables; UI must say AUTO proxy |
| `mid_episode_geometry_change` | Lock/swap colliders (rare; Centerstage door analog) |
| `retained_queue` | Ordered slots with overflow |
| `scripted_mechanism_fsm` | Named FSMs parameterized by robot/field data |
| `multi_robot_collision` | 2–4 chassis contacts + penalty hooks |
| `phase_clock` | AUTO / TRANSITION / TELEOP timers |
| `mesh_field_collision` | 3D CAD/MJCF colliders + ballistic pieces (BIOBUZZ). Requires MuJoCo. |

---

## 3. Preset schemas

Normative JSON Schema:

- [`schemas/field-preset.schema.json`](../schemas/field-preset.schema.json)
- [`schemas/cad-manifest.schema.json`](../schemas/cad-manifest.schema.json)
- [`schemas/robot-preset.schema.json`](../schemas/robot-preset.schema.json)
- [`schemas/scoring-rules-preset.schema.json`](../schemas/scoring-rules-preset.schema.json)
- [`schemas/training-run-config.schema.json`](../schemas/training-run-config.schema.json)

Instance data lives under `presets/`. Example instances:

- [`presets/seasons/biobuzz_2026/field.json`](../presets/seasons/biobuzz_2026/field.json)
- [`presets/seasons/biobuzz_2026/scoring.json`](../presets/seasons/biobuzz_2026/scoring.json)
- [`presets/robots/mecanum_biobuzz_4cap.json`](../presets/robots/mecanum_biobuzz_4cap.json)
- [`presets/training/biobuzz_auto_lightweight.json`](../presets/training/biobuzz_auto_lightweight.json) (workstation / cloud / easy siblings)

**Rule:** season nouns (ARTIFACT, MOTIF, POLLEN, SAMPLE, PIXEL) appear only in instance `type` / `id` / `explain` strings, never as Python/Rust identifiers in `engine/` or `python/talongym/sim/`.

### Coordinate system (locked)

Use the official FTC field coordinate system:

- Origin: field center, tile surface.
- +X: right when standing at the Red Wall looking at field center.
- +Y: away from the Red Wall.
- +heading: counter-clockwise.
- Units in presets and exports: **inches**.
- Viewer convention: MeepMeep-style **non-rotated** Cartesian. Do not store audience-rotated poses.

The Red Wall definition still holds for BIOBUZZ. Document alliance-specific notes in the field preset `coordinateSystem.notes`.

### Official CAD pipeline (field and pieces)

BIOBUZZ physical geometry is built from official STEP, not schematic boxes.

- Raw STEP stays in `var/cad/` (gitignored). Derived GLB/STL, `cad_manifest.json`, and `field_mjcf.xml` for BIOBUZZ are committed under `assets/seasons/biobuzz_2026/` so a clone can run Lab and MuJoCo without the `[cad]` extra. Rebuild with `python -m talongym import-field-cad` after an official STEP change.
- Official field binary endpoint: `https://ftc-resources.firstinspires.org/ftc/archive/2027/field/field-cad-step` (handles `Content-Disposition`). POLLEN / red NECTAR / blue NECTAR STEP files are the AndyMark exports pinned in `python/talongym/assets/cad_sources.py`.
- Coordinates in derived assets: FTC inches, Y-up (`x = ftc.x`, `y = height`, `z = -ftc.y`). Piece GLBs are centered at the geometric origin for later free joints.
- **Collision:** one convex hull per CAD solid under `collision/`, plus four continuous STEP-bound glass proxies that close panel seams. Never a single concave field mesh — MuJoCo convexifies it and would seal HIVE/CELL openings. Piece hulls are one convex STL per type. The red and blue HIVE basket assemblies are grouped into separate hinge bodies; their A-frame and pivots remain static.
- **Semantics stay in the preset:** zones, triggers, start poses, and scoring volumes are `field.json` data. CAD does not assign game-rule meaning.
- A `mesh_field_collision` season fails closed if the pinned SHA-256 is missing or stale (`python -m talongym import-field-cad --verify`). Do not AABB-fallback.
- Handoff: backend loads the local CAD `collisionAsset` MJCF assembled from `cadManifest` convex parts (not `field.glb`) and per-`typeId` piece hulls. HIVE hinge angles are reported in each replay frame, and the frontend applies them to matching pivot-centered mechanism GLBs. The frontend also instances piece `visualAsset` GLBs at simulated pose/orientation and cache-busts derived assets with source hash plus generator version.

### BIOBUZZ scoring (preset data; verify)

From Competition Manual V1 Table 10-2. **Re-verify at implementation time.** Details: [presets/seasons/biobuzz_2026/README.md](../presets/seasons/biobuzz_2026/README.md).

| Achievement | AUTO points | Notes |
|-------------|-------------|--------|
| LEAVE | 3 | Assessed at end of AUTO if the ROBOT is in `leave_interior` |
| PARK | 5 | At least partially in the LOADING ZONE |
| HIVE TIP | 20 | Placeholder threshold: 7 in the upward CELL (3 staged NECTAR + 4 launched) |

CELL / FLOWER / GARDEN points are end-of-match only — model in the graph as `enabledPhases: ["TELEOP"]` but do not train on them in MVP.

RP thresholds (Table 10-3) are **match-wide**. AUTO-only training may optimize a **proxy** (LEAVE+PARK, tip count). The UI must never label that proxy “Ranking Points earned.”

POLLEN is 2.8 in yellow; NECTAR is 3.6 in red/blue. Four POLLEN are physically staged in each FLOWER and on each GARDEN line, three NECTAR are physical bodies in each upward CELL, and every enabled ROBOT must preload exactly four POLLEN. The five off-field NECTAR per alliance are not instantiated during AUTO; `World.spawn_nectar_in_garden` is the validated future TELEOP entry point.

AprilTags are 36h11 clusters on CELL bottoms. Manual extract confirmed ids 0, 7, and 38–41; 1–3, 4–6, and 42–45 are inferred.

---

## 4. Gymnasium environment interface

### `FTCAutoEnv`

```python
class FTCAutoEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "none"], "render_fps": 25}

    def reset(self, *, seed=None, options=None) -> tuple[ObsDict, InfoDict]: ...
    def step(self, action) -> tuple[ObsDict, float, bool, bool, InfoDict]: ...
```

`options` may set `match_setup` (enabled robots, official same-alliance `startSlotId`, and offsets inside that slot’s `legalRegion`). Runtime still enforces G304: own alliance half, touching the perimeter, and not in a LOADING ZONE or FLOWER. LEAVE is awarded only after the chassis leaves the wall. `opponent_mode`, `action_tier`, `full_noise`, and `ballistic_launch` are also legal `options`. Run/evaluation APIs use the camel-case `matchSetup` equivalent.

**Control rate:** 25 Hz default (preset `episode.controlHz`; legal 20–50). Physics substeps: 2 unless the Phase 0 bench says otherwise. Episode length: **30.0 s AUTO**, then truncation. `World.phase` is always `"AUTO"`; TRANSITION/TELEOP scoring nodes never run. TRANSITION (8 s official; 15 s at FIRST Championship) is **not** simulated; end-of-AUTO scoring uses the AUTO `phaseEnd` hook with a configurable `settle_time_s` (default 0.5 s) to stand in for “come to rest.”

### Observation space (`Dict`) — policy-visible only

All arrays `float32` unless noted. Missing detections are zero with a validity mask. **No pose ground truth. No unread match variables.**

| Key | Shape | Content |
|-----|-------|---------|
| `pose_noisy` | (3,) | Encoder/odometry `x_in, y_in, heading_rad` with configured Gaussian + slip. |
| `vel_noisy` | (3,) | `vx, vy, omega` from encoders/IMU. |
| `time_remaining_s` | (1,) | AUTO clock. |
| `held_count` | (1,) | Integers stored as float. |
| `held_colors` | (C,) | One-hot-ish slots; C = robot `mechanisms.capacity`. Unknown color = zeros. |
| `nearest_pieces` | (K, 6) | Up to K pieces in radius: `rel_x, rel_y, type_id_norm, color_id, valid, held_by_other`. |
| `triggers` | (T,) | Occupancy of named trigger volumes the robot can sense (bumpers, intake). |
| `vision_tags` | (M, 5) | `tag_id_norm, bearing, range, visible, occluded`. |
| `match_var_obs` | (V, E+1) | For each observable match variable: one-hot over domain **plus** sentinel channel `not_yet_observed`. Ground truth is **not** copied here at t=0. |
| `teammate_pose_noisy` | (3,) | Zeros in single-agent mode; still present so the space is stable. |
| `collision` | (1,) | 1 if in contact with wall/robot this step. |

`K, T, M, V, C, E` are compiled from the active presets so a BIOBUZZ preset can change them without env class edits. `observation_space` is rebuilt at `reset` only if presets change (training run pins presets).

**Privileged info** (eval, shaping, replay overlays) lives in `info["privileged"]` and is stripped by `EncoderOnlyObsAssertWrapper` before `model.learn`. Tests fail the suite if `match_var_obs` sentinel is false while the vision model reports the motif tag not visible.

### Action spaces

**Tier A — MVP `high_level_waypoint`**

`Dict`:

- `target_pose`: `Box` shape (3,) in inches / radians, clipped to field + margin.
- `speed_frac`: `Box` (1,) in [0.2, 1.0].
- `mechanism`: `Discrete(N)` compiled from robot FSM verbs (`idle`, `intake`, `score`, `open_gate`, `stow`, …).

A follower (pure pursuit + heading PID, constraints from the robot preset) consumes `target_pose` until the next env step **or** until the waypoint is reached, whichever first. The policy therefore emits a moving target at 25 Hz, which is enough to represent splines after export decimation.

**Tier B — `low_level_velocity` (Phase 5)**

`Box(3,)` = `(vx, vy, omega)` in in/s and rad/s, clipped by motor model (torque-speed curve + current limit). Mecanum inverse kinematics inside the drivetrain plugin. Tank ignores `vy`. Swerve is clipped as holonomic (same as mecanum); module-level IK is not solved.

**Tier C — `physical_actuators`**

Legal on the robot preset (`defaultActionTier`) and training `actionTier`. PPO uses the robot default when the training JSON omits `actionTier`.

### Reward signature

```python
@dataclass
class RewardBreakdown:
    true_score_delta: float      # official points this step (usually 0 until events)
    shaping: float               # progress, time cost, collision — NOT on the leaderboard
    objective: float             # what PPO actually maximizes (configurable mix)
```

`step` returns `reward = breakdown.objective`. `info["true_score"]` is the running official AUTO score. `info["shaping"]` is logged and plotted on a **separate** chart. Leaderboard and “statistically best” use **only** `true_score` (or the pre-registered objective of true_score / LCB / RP-proxy).

Default shaping (coefficients in the training config, not the scoring preset):

- +progress toward nearest useful piece / score volume (potential difference).
- −0.01 per second (time cost).
- −0.5 on wall contact impulse above threshold; −2.0 on robot-robot contact.
- 0 motif bonus in shaping (avoid leaking). Pattern points arrive only via the rule DAG at `phaseEnd`.

### `reset` / `step` semantics

**reset**

1. Seed NumPy + the physics backend from `seed`.
2. Draw match variables (motif, etc.).
3. Spawn pieces, robots (G304-legal start slots only; jitter is resampled inside the slot’s legal region), randomize build tolerance.
4. Zero all sensors’ observed-match-var flags.
5. Evaluate rule DAG `alwaysTick` once (should be no points).
6. Return obs with sentinel on unread variables.

**step**

1. Clip action; follower or IK → wheel forces / velocities.
2. Physics backend `step` × substeps (`planar2d` or MuJoCo mesh).
3. Advance mechanism FSMs (timers, trigger volumes).
4. Sensor models (vision last, using updated poses + occluders).
5. Rule DAG: triggers this tick → conditions → actions → score channels.
6. `terminated = False` for AUTO (no absorbing win). `truncated = time >= 30s` after `phaseEnd` scoring.
7. Illegal action (NaN) → `terminated=True`, true_score unchanged, objective −10 (training only).

### Recurrent state

`RecurrentPPO.predict(obs, state=lstm_states, episode_start=done)`. VecEnv must set `episode_start` on truncate. Do not carry LSTM across episodes.

### PettingZoo (Phase 4, unused)

`python/talongym/env/petting.py` is a Phase 4 wrapper (`FTCAutoParallelEnv`). It is not on the Lab or CLI train path. Single-agent Gym is `FTCAutoEnv` with teammate/opponents from `TrainingRunConfig.presets.teammatePolicy` / `opponentPolicy`. `frozen_policy` loads a checkpoint path or run artifact; if missing, the opponent is skipped (logged), not a hardcoded zip.

---

## 5. API contract

Base URL: `/api/v1`. Auth: none for local MVP; later a team token. **No browser storage.** All presets, runs, and reports are SQLite rows.

Typical error body: `{ "error": { "code": "SCHEMA"|"NOT_FOUND"|…, "message": "..." } }`.

HTTP today: 400 validation, 404 missing, 409 preset in use, 422 schema, 501 LSTM ONNX, 503 mesh/CAD extras. There is **no** `PRESET_STALE` code and **no** worker-busy 503. Stale presets are a boolean on list rows (`stale`), not an error envelope.

Training jobs are **in-process threads**. The SQLite `jobs` table is a write-only status log (`enqueue_job`); no worker reads it.

### REST

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | `{ "ok", "engine": "planar2d" (Rapier stub fallback), "fieldEngine", "db", "capabilityVersion": 1, "computeProfile", "nEnvs" }` |
| GET | `/compute` | Detected profile and recommended nEnvs |
| GET/PUT | `/defaults` | Active field/robot/scoring/training ids |
| GET | `/presets/{kind}` | List (`field` \| `robot` \| `scoring` \| `training`) |
| POST | `/presets/{kind}` | Create; 201 |
| GET | `/presets/{kind}/{id}` | Full document |
| PUT | `/presets/{kind}/{id}` | Replace |
| DELETE | `/presets/{kind}/{id}` | 204 if unused; 409 if referenced by a run |
| POST | `/presets/validate` | `{ "kind", "document" }` → `{ "ok", "errors", "unsupportedCapabilities" }` |
| GET | `/runs` | Training runs |
| POST | `/runs` | Starts an in-process thread; 202 `{ "runId" }` |
| GET | `/runs/{id}` | Status, config, metrics, `artifactIds`, `artifacts` |
| GET | `/runs/{id}/artifacts` | Artifact rows (checkpoint, roadrunner, onnx_ff, …) |
| POST | `/runs/{id}/cancel` | Cooperative cancel: `request_cancel` + state `cancelling` until the worker acknowledges `cancelled` |
| POST | `/runs/{id}/export/onnx` | **501** unless `distill=true`, which distills the **scripted** AUTO (not the run checkpoint) into feed-forward ONNX |
| GET | `/replays` | Replay list |
| GET | `/replays/{id}` | Metadata |
| GET | `/replays/{id}/chunks?fromStep=&limit=` | Frames (max 500/request) |
| POST | `/replays/{id}/export/roadrunner` | `{ "dialect": "rr1_actions" }` → text/plain Java |
| POST | `/replays/demo` | Scripted demo replay; 201 |
| POST | `/evaluations` | **200**, synchronous. `{ "nTrials", "objective", "policy", "runId", "checkpoint", "matchSetup" }` |
| GET | `/evaluations` | List |
| GET | `/evaluations/{id}` | Report with CIs; `bestLabelEligible` |
| GET | `/comparisons/latest` | Ranked evaluations; no “best” if overlapping CIs or n&lt;500. There is no `GET /comparisons/{id}`. |
| POST | `/presets/robot/{id}/model` | Upload robot CAD |
| GET | `/field-assets/{path}` | Season meshes |
| GET | `/robot-assets/{path}` | Robot meshes |

Layout/field-builder autosave uses PUT on the field preset. `EvalBody.objective` (`mean_true_score` \| `p10_true_score` \| `lcb_true_score`) is stored as `report.objective` / `objectiveValue`. There is no `rp_proxy_probability` objective.

### WebSocket

`GET /api/v1/ws/runs/{runId}` (upgrade).

The server **pushes** JSON text frames (`v`, `seq`, `type`, `payload`). Client text is **read and discarded**. There is no `ack`, no `BACKPRESSURE`, and no `?afterSeq=` resume.

- `status`: `{ "state": "queued|running|cancelling|succeeded|failed|cancelled", … }`
- `metrics`: `{ "envSteps", "trueScoreMean", "shapingMean", … }` — **true vs shaping always separate keys**
- `rollout`: downsampled `WorldState` for the live 3D view
- `log` / `error`

### Frontend routes (React Router)

| Route | View |
|-------|------|
| `/replay/:replayId?` | 3D field, robots, pieces, FSM overlays, scrub, play/pause/speed. FOV cones and planned path are best-effort overlays, not a physics planner. No reward heatmap. |
| `/train/:runId?` | Sparklines (true score vs shaping) and live scene. No score histogram, success-rate plot, or hyperparam editor panel. |
| `/build/field` | Pose + JSON field editor (not a free-form grid CAD tool) |
| `/build/robot` | Catalog 3D assembly builder: goBILDA/REV recipes, snap, drafts; `defaultActionTier` including `physical_actuators` after confirmation |
| `/compare` | Leaderboard of evaluations with CI whiskers; Export RR per row. No scoring graph editor. |

Stale preset: list rows include `stale` when `provenance.manualRevision` differs from `LATEST_KNOWN_MANUAL[season]`. There is no blocking `PRESET_STALE` API error.

---

## 6. Phased roadmap

### Phase 0 — Foundation benchmark (done when)

- Schemas frozen at `1.0.0`; compiler rejects unknown capabilities.
- 2.5D path is Python `Planar2DBackend`. Rapier crate is a stub; Box2D is not built.
- **Throughput gate (measure, then lock):**
  - Lightweight laptop: **≥ 5×10³** control-steps/s at 8 envs, BIOBUZZ single robot, no viewer.
  - Workstation 16-core: **≥ 5×10⁴** control-steps/s at 256 envs.
  - If Rapier spike misses 50% of workstation gate after two weeks, keep Planar2D and do not block Phase 1. Do not invent a third engine.
- Deterministic replay: same seed + action log → bit-stable poses at 1e-4 in.
- BIOBUZZ field + scoring presets compile; provenance fields populated; `verifyAgainstManual` still true until mentor sign-off.
- No frontend physics.

### Phase 1 — Single-robot vertical slice (MVP product)

- BIOBUZZ AUTO, high-level actions, mesh field, LEAVE / PARK / HIVE TIP.
- RecurrentPPO, lightweight + workstation profiles.
- SQLite jobs, WS telemetry, `/replay` + `/train`.
- Evaluation report: n=500, bootstrap 95% CI, objective `mean_true_score`.
- RR 1.0 Actions export from a decimated waypoint log.
- **Done:** a student can train overnight on a desktop, scrub the best mean-score rollout, and paste Java into an AUTO OpMode for the Control Hub.

Time-to-useful-policy is a **measured** number after Phase 1, not a promise. Planning target to hold the design honest: workstation reaches mean AUTO score **strictly above a scripted “leave + park” baseline** within **4 hours** wall clock at the workstation env count. If missed, cut shaping bugs and n_envs before adding more 3D physics.

### Phase 2 — Season tooling

- `/build/field` and `/build/robot`; scoring graph editor is Phase 2 and **not shipped**.
- Stale-version banners.
- Season-agnostic regression: new games as volumes + end-of-AUTO scoring; **no season nouns in engine identifiers**.
- Randomization seasons use `matchVariables` + `sequenceEquals` when needed.
- Runbook executed dry on a fake “season X” fixture in CI.

### Phase 3 — Multi-agent rigor

- PettingZoo cooperative red alliance; scripted then frozen-policy opponents.
- Collision / right-of-way metrics in the report (contact-seconds, first-contact time, G402-style “entered opponent side” flag as a **data rule**, not a hardcoded season name).
- Paired CRN comparisons; “best” suppressed when CIs overlap.

### Phase 4 — (renumbered in exec summary as Phase 4/5) Transfer and scale

Keep the plan-file numbering mapped as:

- Plan-file Phase 4 ≈ this Phase 3 (multi-agent).
- Plan-file Phase 5 ≈ this Phase 4: low-level velocity, real-log calibration, ONNX demo for feed-forward clone of the follower targets (LSTM ONNX remains 501), experimental Ray/RLlib toy, self-play, Postgres + job queue, optional MuJoCo validation mode.

---

## 7. Risks and mitigations

1. **Rapier Python ABI slips.** Mitigation: Phase 0 time-box; ship Python `Planar2DBackend`. Box2D is not implemented. Keep `PhysicsBackend` protocol tiny (`reset_batch`, `step_batch`, `contacts`).
2. **Policy cheats motif.** Mitigation: unit tests that mutate privileged motif without moving the robot/camera and assert obs unchanged; curriculum that starts with `motif_known_at_t0` then disables it.
3. **Shaping optimizes the wrong game.** Mitigation: leaderboard binds to `true_score`; freeze shaping coefficients in the run config hash. Automatic abort when true_score and objective diverge is **not implemented**.
4. **Sim-to-real gap on launch.** Mitigation: ballistic launches use the mesh field when unlocked; launch success can also be a Bernoulli + heading/range model fit from team logs (`calibrate/`). MuJoCo is required for BIOBUZZ training — no silent planar fallback.
5. **LSTM ONNX / in-browser demo is lossy.** Mitigation: the only field path is (a) follower waypoint sequence pasted into an AUTO OpMode the Control Hub runs; (b) `distill=true` writes a feed-forward clone of the **scripted** policy, not the run. Do not claim on-robot NN inference.
6. **Multi-robot collisions ignored in MVP then surprise in quals.** Mitigation: even single-agent MVP spawns a **static** teammate bounding box by default (configurable off) and penalizes contact.
7. **Mid-season Team Updates silently desync.** Mitigation: `manualRevision` + hash on every preset; UI stale banner; evaluations store preset hash immutably.
8. **Laptop cannot train.** Mitigation: lightweight profile + “import a run the mentor trained” via SQLite file copy; cloud path documented, not required.

---

## 8. Open questions for an FTC mentor (BIOBUZZ freeze)

Do not treat the following as engine work. They are **preset sign-off** questions. Until answered, keep `verifyAgainstManual: true`. Full checklist: [MENTOR_SIGNOFF.md](MENTOR_SIGNOFF.md).

1. Confirm Table 10-2 AUTO values still 3 / 5 / 20 for LEAVE / PARK / HIVE TIP.
2. Confirm HIVE TIP threshold (preset uses 7 in the upward CELL) against CAD / Field Setup Guide.
3. Confirm AprilTag cluster ids against official artwork (0, 7, 38–41 confirmed; 1–3, 4–6, 42–45 inferred).
4. G402 (no AUTO opponent-side interference): **encoded as a scoring/foul node.** Restricted-volume entry `addScore`s −15 on `trueScore` (training proxy for opponent Major Foul; VERIFY AGAINST MANUAL). The flag still feeds `restrictedEntryRate`. Replay highlights the foul; wall/robot contact stays shaping-only.
5. Championship 15 s transition vs 8 s: ignore for AUTO policy training?
6. Which optimization objective should the default leaderboard use for *your* team: mean match points, 10th percentile (robust), or RP-proxy probability (SWARM ≥ 16, POLLINATOR 1/2 — **verify Table 10-3**)?
7. Legal start poses for your region’s typical field build: which start slots are actually used in AUTO?

---

## Sim-to-real, collisions, statistics, performance (must-not-overlook)

### Simulated vs approximated

| Phenomenon | Modeled as |
|------------|------------|
| Planar holonomic / tank motion | Motor-limited 2D rigid body |
| Wall / robot / piece push | Planar2D contacts (Rapier stub unused) |
| Intake capture | Volume + cycle timer + capacity |
| Launch / score | Trigger path + optional ballistic mesh |
| Vision / match vars | FOV, range, occluder rays, false-negative rate |
| Launch ballistics | BIOBUZZ: 3D free-body flight vs mesh field; physical mechanism contact sets launch state |
| Foam tile compliance | Optional friction DR, not FEM |
| Battery sag | Optional; off in lightweight |

**Calibration workflow:** ingest FTC Dashboard / WPILOG pose streams from a few real AUTOs; fit `maxVel`, `maxAccel`, slip noise, and intake cycle time so simulated encoder traces match within a reported RMSE. Store the fit as a robot-preset overlay, not a fork of BIOBUZZ rules.

### Multi-robot right-of-way

Contact events increment `collision_time_s` and apply shaping penalty. Eval report includes collision rate. Cooperative Phase 3 adds a shared “who claimed this spike mark” accumulator teams can put in the rule graph if they want explicit coordination rewards — still data, not engine.

### Statistical rigor

- Pre-register `objective` on the training run.
- ≥500 held-out seeds, disjoint from training.
- Pair candidates with common random numbers (same motif, same piece jitter) when comparing.
- Bootstrap 95% CI on the objective.
- Holm correction if ranking >2 policies.
- `bestLabelEligible: false` when CIs overlap or nTrials < 500.
- Report mean, median, p10, p90, min of **true_score**.

### Performance budget (acceptance, not folklore)

| Profile | n_envs | Control Hz | Gate |
|---------|--------|------------|------|
| lightweight_cpu | 8 | 25 | ≥5e3 steps/s; 30 s AUTO replay export ≤ 2 s |
| workstation | 256 | 25 | ≥5e4 steps/s; 4 h to beat scripted baseline (Phase 1 target) |
| Memory | — | — | ≤ 8 GB laptop / 32 GB workstation for the above n_envs |

Final numbers are **locked after Phase 0 measurement**.

### Accessibility

Lightweight CPU path is first-class. Cloud is optional. UI copy must not assume a GPU.

### Extensibility self-check

Engine identifiers are capability names and geometry kinds (`aabb`, `volumeEnter`, `retained_queue`). If a grep of `engine/` plus `python/talongym/sim` plus `python/talongym/rules` for `artifact|obelisk|motif|pollen|specimen|pixel` returns hits outside tests/comments, the PR fails CI.

---

## Drivetrain, motors, sensors (robot model)

- Plugins: `mecanum` (default), `tank`, `swerve`.
- Forward/inverse kinematics in `python/talongym/robot/`. Swerve chassis clip is holonomic (same as mecanum); module states are not solved.
- DC motor: linear torque-speed, clip to `currentLimitA`. No infinite acceleration.
- Encoders: Gaussian noise + slip; separate `ground_truth` pose for rewards/eval only.
- AprilTag camera: pinhole FOV cone, max range, ray vs `isOccluder` elements (BIOBUZZ hive frame + CELL tag clusters).

---

## Road Runner export (required, not stretch)

This is the only match-bound artifact. Teams paste the Java into an AUTO OpMode; the Control Hub runs it. Do not stream the policy or load a neural net onto the robot.

From a rollout of high-level targets, decimate to a polyline with heading, then emit Java:

```java
Actions.runBlocking(
    drive.actionBuilder(new Pose2d(startX, startY, startHeading))
        .splineTo(new Vector2d(x1, y1), tangent1)
        // ...
        .build());
```

Dialect `rr05_trajectory_sequence` emits legacy `trajectorySequenceBuilder` for teams not on 1.0. Units: inches. Coordinate system: same as MeepMeep / official FTC.

---

## Capability regression matrix

| Need | BIOBUZZ V1 |
|------|------------|
| Planar drive + walls | yes |
| Game pieces | POLLEN 2.8 in, NECTAR 3.6 in |
| Randomized match var + sentinel | none (AUTO) |
| Occluders + vision | hive frame + cells (AprilTag clusters) |
| N-stage + pattern/derived | launch count then HIVE TIP |
| End-of-AUTO scoring | LEAVE, PARK, HIVE TIP |
| Retained queue | no |
| Alliance aggregate | SWARM RP (LEAVE+PARK) |
| Mid-episode geometry | red/blue HIVE hinge bodies tip from physical CELL occupancy |
| Climb / vertical FSM | no AUTO |

BIOBUZZ is the only shipped season. A new season is data under `presets/seasons/` following the runbook; CI capability lint must pass with **zero files under `engine/` changed** unless a new primitive is required.

---

## Implementation notes for the first coding agent

1. Implement `PhysicsBackend` and `Planar2DBackend`. The Rapier crate is a stub until it meets the Phase 0 gate; production 2.5D default is `planar2d`.
2. Compile scoring JSON to a contiguous struct-of-arrays DAG; evaluate in Rust or Cython-free Python first, profile, then move hot loops.
3. Write `tests/presets/test_no_season_leak.py` and `tests/env/test_motif_not_leaked.py` before any PPO run.
4. Pin `sb3-contrib` RecurrentPPO; flatten only the boxes SB3 cannot digest, keep Dict via `MultiInputLstmPolicy`. BIOBUZZ uses `AsymmetricLstmPolicy` so the critic may read `_privileged` while the actor cannot.
5. SQLite schema: `presets`, `runs`, `evaluations`, `artifacts`, `replays`, `jobs` — WAL mode, files under `var/talongym.db`. Training is in-process threads; `jobs` is a status log.
