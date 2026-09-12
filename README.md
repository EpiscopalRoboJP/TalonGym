# TalonGym

A **training and strategy-discovery aid** for FIRST Tech Challenge (FTC) autonomous periods. Teams configure a field, robot, and scoring-rules preset; train a control policy in a 2.5D simulator (DECODE) or a MuJoCo 3D mesh field (BIOBUZZ); inspect *why* a strategy scored; and export a Road Runner 1.0 Actions snippet they paste into an AUTO OpMode.

**Always and only this deployment path:** train offline → export a trajectory → paste into an AUTO OpMode → the Control Hub runs that OpMode. TalonGym never talks to a robot during a MATCH.

## What it does

1. Simulates the 30-second FTC Autonomous Period with enough physical and rules fidelity that a policy trained here can discover strategies meaningfully close to deployable on a real robot.
2. Trains a robot control policy via reinforcement learning and ranks candidates only after a statistical evaluation harness (hundreds of randomized trials, confidence intervals).
3. Exposes a browser 3D visualizer (MeepMeep-style replay, live training, preset builders, leaderboard).
4. Treats the current game as a **preset**, not an engine shape. DECODE™ presented by RTX (2025–2026, Competition Manual **TU32**) ships as the flagship preset. INTO THE DEEP℠ (2024–2025) ships as the season-agnostic regression proof. CENTERSTAGE℠ (2023–2024) is the third regression preset. BIOBUZZ™ (2026–2027) is the Kickoff-week acceptance test: a team must stand it up from data + a small rule graph, without a core-engine PR.

## Deployment path (always and only)

1. **Train offline** in TalonGym (CLI or Lab). The policy never leaves the simulator.
2. **Export a trajectory** as Road Runner 1.0 Actions (inches, official FTC coordinates).
3. **Paste that snippet into an AUTO OpMode** in *your* robot code.
4. **The Control Hub runs that OpMode** on the field. No laptop, no live policy, no retargeting.

That is the only supported way a TalonGym result may be used at an event. It matches FTC Autonomous: the robot runs pre-programmed instructions with zero driver input (G401).

**Not supported (and illegal play if used in a MATCH):** streaming policy actions to a robot; running the neural net on the field; using TalonGym or any other PC as a live controller; leaking motif/randomization into the robot before onboard sensors would perceive it; treating a single lucky sim rollout as “the auto.”

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
- BIOBUZZ field CAD / STEP: https://ftc-resources.firstinspires.org/ftc/archive/2027/field
- FTC field coordinate system: https://ftc-docs.firstinspires.org/en/latest/game_specific_resources/field_coordinate_system/field-coordinate-system.html
- Road Runner 1.0 Actions: https://rr.brott.dev/docs/v1-0/actions/
- MeepMeep: https://github.com/acmerobotics/meepmeep

Point values and geometry in presets are **data**. If a Team Update changes them, bump the preset’s `manualRevision` rather than editing engine code.

## Run the Lab

```bash
python -m pip install -e ".[dev,rl]"
python -m talongym lab
```

Optional extras: `[scale]` Ray/RLlib, `[mujoco]` required for BIOBUZZ 3D mesh physics (DECODE stays planar), `[cad]` official STEP tessellation, `[postgres]` when `TALONGYM_DATABASE_URL` is set. LSTM ONNX export is 501; `python -m talongym distill` writes a feed-forward demo and is not a deployment path. The only match-bound artifact is a Road Runner 1.0 snippet pasted into an AUTO OpMode.

Open http://127.0.0.1:8765 after `cd web && npm install && npm run build`, or use Vite (`npm run dev` in `web/`, API on :8765). State lives in `var/talongym.db` — not in the browser.

Train, evaluate, replay, and export (full flags and Lab budgets): **[docs/TRAINING.md](docs/TRAINING.md)** and **[docs/README.md](docs/README.md)**.

```bash
python -m talongym defaults --training decode_auto_lightweight
python -m talongym train --steps 8192
python -m talongym replay
python -m talongym evaluate --trials 32
python -m talongym distill
python -m talongym import-field-cad
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
