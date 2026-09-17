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

The top bar health chip is `GET /api/v1/health` (`engine`, `db`, ok). If it says **API offline**, the UI cannot train or load presets.

## Routes

| Path | Page |
|------|------|
| `/` and `/replay/:replayId?` | 3D replay, score inspect, drive-only Road Runner export |
| `/train/:runId?` | Start/watch RecurrentPPO, true vs shaping charts |
| `/build/field` | Field + scoring JSON editor, lint, save |
| `/build/robot` | Catalog-driven 3D robot builder: guided goBILDA/REV drivebases, snap assembly, working drafts |
| `/compare` | Evaluation leaderboard with CI whiskers |
| any other path | Not-found page with a link back to Replay |

## Replay

- **Record scripted AUTO** — POST `/replays/demo`, then select the new id.
- Play / pause, 0.25× / 1× / 2×, 3/4 or top camera, scrub slider.
- **FOV** checkbox appears when the robot design has a camera sensor. The overlay is a wedge from robot pose using that sensor’s `fovDeg` / `rangeIn` (not a fixed disc).
- Keys: Space play/pause, arrows step, Home/End jump (when focus is not in an input).
- **Score** tab: running `trueScore` (net of fouls), sticky penalties, signed explains. Restricted-entry fouls and wall/robot contact are marked on the scrubber and flashed on the robot.
- **Field state** tab: named ramp queues, gate, privileged vs observed match variables (inspection only).
- **Export** tab: drive-only Road Runner 1.0 from executed (decimated) poses. Mechanism timeline is comments (`t` / verb / actuator), not a working AUTO — intake and launch still need team OpMode code. TalonGym does not stream this during a MATCH.

Training and evaluation jobs also store replays; pick them from the dropdown (id · source · true score).

## Train

See [TRAINING.md](TRAINING.md)#3-train-from-the-lab.

- Choose **Robot** and **Field** in the new-run form or on the 3D display. The viewport shows a setup preview of that pair immediately. A live rollout only replaces the preview when the selected run used the same robot and field.
- Demo / Short / Easy / Preset budgets. Demo is a short scripted fallback if RL extras are missing, not a scaled trainer. Easy autodetects laptop vs workstation vs cloud hardware (`GET /compute`) and loads `*_easy`. The cloud preset is a larger RecurrentPPO config, not a Ray/RLlib cluster.
- **Configure robot starts** to pick official `startSlotId` values and legal along-wall offsets. Illegal G304 poses never reach training.
- Live rollout of the current run (downsampled WebSocket frames).
- **Save as default** writes the selected training / field / robot / scoring ids for CLI jobs and builders.
- Cancel requests `POST /runs/{id}/cancel`.
- After success, links open the stored replay and Compare.

True-score and shaping charts are separate. Copy on the page states that shaping is never ranked.

## Field builder

Load a field preset, edit elements (pose, shape, tags), lint against the JSON Schema, PUT to replace. Scoring tab edits the paired scoring document. Stale `manualRevision` vs `LATEST_KNOWN_MANUAL` should show in list metadata. Do not invent point values; keep `verifyAgainstManual: true` until a second person checks the manual.

Saves go to the API/SQLite row, not a silent browser blob. Invalid JSON in the advanced editor is shown as an error; save and validate do not throw in the click handler.

## Robot builder

Catalog assembly editor: 3D viewport with mount-first hole-pattern mating, a compatibility-filtered parts catalog (goBILDA and REV), an assembly tree, numeric inspector, and four guided drivebase recipes (goBILDA/REV × mecanum/tank). Shipped scoring starters load with intake, conveyor, flywheel, hood, and gate catalog parts already bound. Select a part and then one of its visible mount nodes; the drawer shows only compatible parts and creates a preview before confirmation. Connected children move through their mate definitions. Translate/rotate gizmos are limited to the root and detached/free-mounted parts. The wizard instantiates real rails, brackets, motors, and wheels as a schema 1.2 assembly. Inference proposes drivetrain, transmissions, and part roles; **Confirm inference** is required before a competitive save can enable `physical_actuators`. Unconfirmed robots stay on `high_level_waypoint`. Scoring topology is compiled only when the assembly declares it (`includeScoringTopology` or intake/flywheel parts); drivebases no longer inherit a hidden 4-cap mechanism.

Builder controls: left-click selects or confirms, right-drag orbits, the wheel zooms, `M` starts mount selection, `Q`/`E` cycle valid 90-degree mate orientations, `Enter` confirms, `Escape` cancels one interaction level, `Delete` removes the selected subtree, `F` frames the assembly, and `Ctrl`/`Cmd` + `Z` / `Shift+Z` undo and redo. The top-right orientation control snaps to top, front, right, and isometric views. Camera position is stored per working draft outside the robot preset.

Working copies autosave as API drafts (`PUT /api/v1/presets/robot/{id}/draft`) and never replace shipped presets. Final **Save** / **Save as** runs schema validation and the confirmation gate. Shipped presets stay immutable.

Official manufacturer CAD downloads on demand into `var/assets/robot_parts/` (not git). Each successful conversion also generates a deterministic isometric thumbnail from the official cached mesh. Cache is `ready` only when the GLB and thumbnail both exist and load. Committed viewport parts render that GLB (collision proxy only while loading or after a load error). **Cache required CAD** lists missing SKUs on the current assembly, queues them, and refreshes those parts when the batch finishes. **Cache all CAD** queues every downloadable uncached SKU through bounded background jobs (`POST /api/v1/catalog/cache/all`) with progress, cancel, retry-failed, and thumbnail generation; it never blocks the catalog UI. Without `[cad]`, or when a SKU has download disabled, each part remains clearly labeled **Proxy** with its authored collision geometry — never a single chassis lump when catalog parts exist. Per-part GLB upload remains `POST /api/v1/presets/robot/{id}/parts/{partId}/model`.

Save a compiled robot with PUT `/presets/robot/{id}` after `POST /presets/validate`. **Save as** POSTs a copy so shipped presets stay intact. Fit real logs with CLI `calibrate` and merge the overlay ([EXPORT.md](EXPORT.md)) rather than guessing constraints.

## Compare

[EVALUATION.md](EVALUATION.md) for the statistics.

- **Run 24-trial evaluation (scripted)** — stores a row; n&lt;500 so it cannot earn “best”.
- **500-trial eval** — posts `nTrials: 500` (API clamp). That size can earn “best” if CIs do not overlap. The eval runs inline, so the Lab waits.
- **Optional checkpoint eval** — 24- or 500-trial against the selected run’s checkpoint artifact.
- Empty leaderboard explains that there are no rows yet. Collision rate is shown when the report includes it.
- Whiskers are bootstrap 95% CIs on mean true score.
- Per-row export is drive-only Road Runner 1.0; mechanism timeline is comments, not a complete AUTO.

## API surface the UI uses

Base: `/api/v1`. No auth on local MVP. Error envelope: `{ "error": { "code", "message", "details" } }`.

Common calls: `/health`, `/compute`, `/defaults`, `/presets/{kind}`, `POST /runs`, `GET /runs/{id}`, WebSocket `/ws/runs/{runId}`, `/replays`, `/evaluations`, `/comparisons/latest`, `POST /replays/{id}/export/roadrunner`.

Full contract: [ARCHITECTURE.md](ARCHITECTURE.md) §5. LSTM ONNX is HTTP 501 unless `distill=true` (feed-forward demo).
