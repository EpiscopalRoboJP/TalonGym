# TalonGym

A **training and strategy-discovery aid** for FIRST Tech Challenge (FTC) autonomous periods. Teams configure a field, robot, and scoring-rules preset; train a control policy in a fast 2.5D simulator; inspect *why* a strategy scored; and export a Road Runner 1.0 Actions snippet they can paste into competition code.

This is **not** a mechanism for real-time in-match robot control. Using it during a MATCH to drive or retarget a robot would violate the spirit of FTC Autonomous (zero driver input) and is outside the intended-use boundary of this product.

## What it does

1. Simulates the 30-second FTC Autonomous Period with enough physical and rules fidelity that a policy trained here can discover strategies meaningfully close to deployable on a real robot.
2. Trains a robot control policy via reinforcement learning and ranks candidates only after a statistical evaluation harness (hundreds of randomized trials, confidence intervals).
3. Exposes a browser 3D visualizer (MeepMeep-style replay, live training, preset builders, leaderboard).
4. Treats the current game as a **preset**, not an engine shape. DECODE™ presented by RTX (2025–2026, Competition Manual **TU32**) ships as the flagship preset. INTO THE DEEP℠ (2024–2025) ships as the season-agnostic regression proof. CENTERSTAGE℠ (2023–2024) is the third regression preset. BIOBUZZ™ (2026–2027) is the Kickoff-week acceptance test: a team must stand it up from data + a small rule graph, without a core-engine PR.

## Intended-use boundary (spirit of competition)

- **Allowed:** off-field strategy discovery, path rehearsal, sim-to-real calibration against *your own* logs, exporting a trajectory into *your* robot code that then runs autonomously on the field.
- **Not allowed / not supported:** streaming policy actions to a robot during a MATCH; leaking motif/randomization from the event system into the robot before the robot would perceive it; treating a single lucky sim rollout as “the auto.”

## Hardware modes

| Mode | Audience | Parallel envs (starting point) | Notes |
|------|----------|--------------------------------|-------|
| Lightweight / local | Typical FTC laptop, CPU-only | 8–32 | Reduced randomization; still reports true score + CIs |
| Workstation | Mentor / school desktop | 128–512 | Default training path |
| Cloud (optional, Phase 5) | Club with a rented GPU/CPU box | 1024+ via Ray | Documented upgrade; not required |

Throughput and time-to-policy numbers are **Phase 0 benchmark gates**, not claimed facts. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Official sources

- DECODE Competition Manual TU32: https://ftc-resources.firstinspires.org/ftc/archive/2026/game/manual
- DECODE Game Details (HTML): https://ftc-resources.firstinspires.org/ftc/game/manual-10
- DECODE field CAD / STEP: https://ftc-resources.firstinspires.org/ftc/archive/2026/field
- FTC field coordinate system: https://ftc-docs.firstinspires.org/en/latest/game_specific_resources/field_coordinate_system/field-coordinate-system.html
- Road Runner 1.0 Actions: https://rr.brott.dev/docs/v1-0/actions/
- MeepMeep: https://github.com/acmerobotics/meepmeep

Point values and geometry in presets are **data**. If a Team Update changes them, bump the preset’s `manualRevision` rather than editing engine code.

## Run the Lab

```bash
python -m pip install -e ".[dev,rl]"
python -m talongym lab
```

Optional extras: `[scale]` Ray/RLlib, `[mujoco]` 3D validation only, `[postgres]` when `TALONGYM_DATABASE_URL` is set. LSTM ONNX export is 501; `python -m talongym distill` writes a feed-forward demo. Deployment export remains Road Runner 1.0.

Open http://127.0.0.1:8765 after `cd web && npm install && npm run build`, or use Vite (`npm run dev` in `web/`, API on :8765). State lives in `var/talongym.db` — not in the browser.

Train, evaluate, replay, and export (full flags and Lab budgets): **[docs/TRAINING.md](docs/TRAINING.md)** and **[docs/README.md](docs/README.md)**.

```bash
python -m talongym defaults --training decode_auto_lightweight
python -m talongym train --steps 8192
python -m talongym replay
python -m talongym evaluate --trials 32
python -m talongym distill
python -m pytest
```

## Docs

How-to index: **[docs/README.md](docs/README.md)**.

| Doc | Contents |
|-----|----------|
| [docs/INSTALL.md](docs/INSTALL.md) | Python extras, Lab build, SQLite / Postgres |
| [docs/TRAINING.md](docs/TRAINING.md) | RecurrentPPO from CLI and `/train` |
| [docs/CLI.md](docs/CLI.md) | Every `python -m talongym` command |
| [docs/LAB.md](docs/LAB.md) | Replay, Train, Field, Robot, Compare |
| [docs/PRESETS.md](docs/PRESETS.md) | Season bundles, lint, training JSON |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Held-out trials, CIs, “best” rules |
| [docs/EXPORT.md](docs/EXPORT.md) | Road Runner, distill ONNX, log calibration |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Gymnasium contract, API, roadmap, risks |
| [docs/NEW_SEASON_RUNBOOK.md](docs/NEW_SEASON_RUNBOOK.md) | Kickoff preset without an engine PR |
| [docs/MENTOR_SIGNOFF.md](docs/MENTOR_SIGNOFF.md) | DECODE TU32 freeze questions |

JSON Schema files live in [`schemas/`](schemas/).
