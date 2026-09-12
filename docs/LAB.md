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
| `/` and `/replay/:replayId?` | 3D replay, score inspect, Road Runner export |
| `/train/:runId?` | Start/watch RecurrentPPO, true vs shaping charts |
| `/build/field` | Field + scoring JSON editor, lint, save |
| `/build/robot` | Drivetrain / constraint / mechanism sliders |
| `/compare` | Evaluation leaderboard with CI whiskers |

## Replay

- **Record scripted AUTO** — POST `/replays/demo`, then select the new id.
- Play / pause, 0.25× / 1× / 2×, 3/4 or top camera, FOV cones, scrub slider.
- Keys: Space play/pause, arrows step, Home/End jump (when focus is not in an input).
- **Score** tab: running `trueScore` and per-event explains.
- **Field state** tab: ramp queue, gate, privileged vs observed match variables (inspection only).
- **Export** tab: Road Runner 1.0 Actions from the decimated path. Paste into an AUTO OpMode; the Control Hub runs it. TalonGym does not stream this during a MATCH.

Training and evaluation jobs also store replays; pick them from the dropdown (id · source · true score).

## Train

See [TRAINING.md](TRAINING.md)#3-train-from-the-lab.

- Demo / Short / Easy / Preset budgets. Easy autodetects laptop vs workstation vs cloud (`GET /compute`) and loads `*_easy`.
- Live rollout of the current run (downsampled WebSocket frames).
- **Set as default** writes the training preset bundle; **Use current selection** writes the three ids without requiring a training preset.
- Cancel requests `POST /runs/{id}/cancel`.
- After success, links open the stored replay and Compare.

True-score and shaping charts are separate. Copy on the page states that shaping is never ranked.

## Field builder

Load a field preset, edit elements (pose, shape, tags), lint against the JSON Schema, PUT to replace. Scoring tab edits the paired scoring document. Stale `manualRevision` vs `LATEST_KNOWN_MANUAL` should show in list metadata. Do not invent point values; keep `verifyAgainstManual: true` until a second person checks the manual.

Saves go to the API/SQLite row, not a silent browser blob.

## Robot builder

Visual design editor: top-down robot-frame inch grid (+x forward, +y left) plus a 3D preview. Place intake mouths and launcher muzzles, then set cycle times, muzzle speed, hood pitch/yaw, and MeepMeep vel/accel. Chassis length/width/height drive both the canvas and replay meshes.

Save with PUT `/presets/robot/{id}` after `POST /presets/validate`. **Save as new** POSTs a copy so shipped presets stay intact. Fit real logs with CLI `calibrate` and merge the overlay ([EXPORT.md](EXPORT.md)) rather than guessing constraints.

## Compare

[EVALUATION.md](EVALUATION.md) for the statistics.

- **Run 24-trial evaluation (scripted)** — stores a row; n&lt;500 so it stays unlabeled.
- **Optional checkpoint eval** — loads the selected run’s checkpoint artifact.
- Whiskers are bootstrap 95% CIs on mean true score.
- Banner: no strategy is labeled statistically best if n&lt;500 or intervals overlap.
- Per-row Road Runner export from that eval’s best replay.

## API surface the UI uses

Base: `/api/v1`. No auth on local MVP. Error envelope: `{ "error": { "code", "message", "details" } }`.

Common calls: `/health`, `/compute`, `/defaults`, `/presets/{kind}`, `POST /runs`, `GET /runs/{id}`, WebSocket `/ws/runs/{runId}`, `/replays`, `/evaluations`, `/comparisons/latest`, `POST /replays/{id}/export/roadrunner`.

Full contract: [ARCHITECTURE.md](ARCHITECTURE.md) §5. LSTM ONNX is HTTP 501 unless `distill=true` (feed-forward demo).
