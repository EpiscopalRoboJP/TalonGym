# Lab UI

The Lab is a React + Three.js viewer. It never integrates physics or scoring. Persistence is the API database, not `localStorage`.

## Start

```bash
cd web && npm install && npm run build && cd ..
python -m talongym lab
```

http://127.0.0.1:8765

Vite (API still on 8765):

```bash
python -m talongym lab
cd web && npm run dev
```

http://127.0.0.1:5173 — Vite proxies `/api` and `/api/v1/ws/…`.

The header status chip polls `GET /api/v1/health`. Hover it for compute profile, engine, and DB. If it says **API offline**, the UI cannot train or load presets. Success and error messages appear as toasts in the bottom-right corner (Esc dismisses them).

## Routes

| Path | Nav label | Page |
|------|-----------|------|
| `/` → `/train`, `/train/:runId?` | Train | Start and watch runs: live rollout, true-score vs shaping curves |
| `/compare` | Evaluate | Run evaluations; leaderboard with CI whiskers |
| `/replay/:replayId?` | Replay | 3D replay, score events, Road Runner export |
| `/build/robot` | Robot | Top-view mechanism editor, 3D preview, component inspector |
| `/build/field` | Field | Field map / 3D, element poses, field and scoring JSON |

## Train

See [TRAINING.md](TRAINING.md)#3-train-from-the-lab.

- **New training run** panel: Demo / Short / Easy / Preset budget, then training, robot, field, and scoring presets. Easy autodetects laptop vs workstation vs cloud (`GET /compute`) and loads `*_easy`.
- **Save as default** writes all four ids to `var/defaults.json` (`PUT /defaults`).
- **Runs** lists every run; the most recent opens automatically. Selecting one reconnects its WebSocket.
- The run header shows state, progress, true score, held-out eval, env steps, curriculum stage, and entropy / KL, with **Open replay**, **Evaluate**, and **Cancel** (`POST /runs/{id}/cancel`).
- Learning curves keep true score, held-out eval, and shaping separate. Shaping is labeled as never ranked.

## Evaluate

[EVALUATION.md](EVALUATION.md) for the statistics.

- Pick **Trials** (24, 100, or 500), then **Evaluate** the scripted baseline or a finished run's checkpoint. The request runs synchronously; 500 trials can take several minutes.
- The leaderboard ranks by mean true score with bootstrap 95% CI whiskers on a shared axis, plus P10, trials, restricted-entry rate, and contact time.
- A row is labeled **Best** only when every row has n ≥ 500 and the top two intervals do not overlap.
- Each row links to its best-episode replay and opens the Road Runner export dialog.

## Replay

- Pick a replay from the **Replays** list (source · id · true score). **Record demo** POSTs `/replays/demo` and selects the result.
- Transport bar: play / pause, timeline with foul (red) and contact (amber) marks, ¼× / ½× / 1× / 2× speed. Perspective or top camera and a camera-FOV toggle sit in the viewport header.
- Keys: Space play/pause, arrows step, Home/End jump (when focus is not in an input).
- **Score** panel: running `trueScore` (net of fouls), penalties and scoring events, contact flags, pieces held, and robot pose.
- **Export** opens Road Runner 1.0 Actions from the decimated path with a Copy button. Paste into an AUTO OpMode; the Control Hub runs it. TalonGym does not stream this during a MATCH.

## Field builder

Toolbar: field preset, **Layout / Field JSON / Scoring rules** tabs, and Save. Verification pills show `verifyAgainstManual` and stale `manualRevision`.

- **Layout**: 2D map (click to select) or the 3D CAD view. The **Elements** list filters by id, type, or tag; the selected element's X / Y / heading edit below it.
- **Field JSON** and **Scoring rules**: full-document editors with **Validate** (`POST /presets/validate`). Switching from Field JSON back to Layout applies the edited JSON.

Do not invent point values; keep `verifyAgainstManual: true` until a second person checks the manual. Saves go to the API/SQLite row, not a silent browser blob.

## Robot builder

Three panes: a top view (robot frame, inches, +x forward, +y left), a 3D preview, and a **Components** inspector.

- **+ Intake** / **+ Launcher** add mechanisms; drag them on the top view or type exact poses. **Remove** deletes the selected one.
- **Chassis & drivetrain**: display name, size, mass, capacity, drivetrain type, track width, and MeepMeep velocity / acceleration limits. Chassis length/width/height drive both the canvas and replay collision box.
- **Camera**: AprilTag camera FOV and range.
- **3D model**: upload STL, OBJ, GLB, glTF, or STEP. The API converts to a light GLB (viewer) and a convex hull (optional physics) under `var/assets/robots/<id>/`. **Fit chassis** copies the mesh bounds into length/width/height. Leave **Use mesh for collision** off to keep AABB physics; turn it on for a 2D hull on planar fields and a MuJoCo mesh geom on BIOBUZZ (slower; fidelity, not overnight PPO). Offset X/Y/Z and yaw line the CAD up with chassis center.

**Save** validates (`POST /presets/validate`) then PUTs `/presets/robot/{id}`. **Save as new** POSTs a copy under the typed id so shipped presets stay intact. Fit real logs with CLI `calibrate` and merge the overlay ([EXPORT.md](EXPORT.md)) rather than guessing constraints.

## API surface the UI uses

Base: `/api/v1`. No auth on local MVP. Error envelope: `{ "error": { "code", "message", "details" } }`.

Common calls: `/health`, `/compute`, `/defaults`, `/presets/{kind}`, `POST /runs`, `GET /runs/{id}`, WebSocket `/ws/runs/{runId}`, `/replays`, `/evaluations`, `/comparisons/latest`, `POST /replays/{id}/export/roadrunner`.

Full contract: [ARCHITECTURE.md](ARCHITECTURE.md) §5. LSTM ONNX is HTTP 501 unless `distill=true` (feed-forward demo).
