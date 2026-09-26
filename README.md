# TalonGym

A **training and strategy-discovery aid** for FIRST Tech Challenge (FTC) autonomous periods. Teams configure a field, robot, and scoring-rules preset; train a control policy on the BIOBUZZ™ MuJoCo mesh field; inspect *why* a strategy scored; and export a Road Runner 1.0 Actions snippet they paste into an AUTO OpMode.

**Status:** 0.5.16 alpha. APIs and presets still move. Licensed **GPL-3.0-or-later**. TalonGym is not affiliated with, endorsed by, or sponsored by FIRST.

TalonGym is built by **FTC Team 17986 904 Robo Eagles** and **FTC Team 27268 Talon Strike**.

**Always and only this deployment path:** train offline → export a trajectory → paste into an AUTO OpMode → the Control Hub runs that OpMode. TalonGym never talks to a robot during a MATCH.

## What it does

1. Simulates the 30-second FTC Autonomous Period with enough physical and rules fidelity that a policy trained here can discover strategies meaningfully close to deployable on a real robot.
2. Trains a robot control policy via reinforcement learning and ranks candidates only after a statistical evaluation harness (hundreds of randomized trials, confidence intervals).
3. Exposes a browser 3D visualizer (MeepMeep-style replay, live training, preset builders, leaderboard).
4. Treats the current game as a **preset**, not an engine shape. BIOBUZZ™ presented by RTX (2026–2027, Competition Manual **V1**) is the shipped season. Earlier games are not included — only this season’s field and scoring are encoded accurately enough to train against.

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
| Workstation | Mentor / school desktop | 128–512 | Default overnight training path |
| Cloud (optional) | Club with a rented GPU/CPU box | 1024 RecurrentPPO envs | Same `recurrent_ppo` as workstation; `[scale]` RLlib is a toy extra |

`python -m talongym detect` prints which mode this machine is. `python -m talongym train --easy` loads that season’s easy run config and fills `nEnvs` from the detection (override with `TALONGYM_COMPUTE_PROFILE=lightweight_cpu|workstation|cloud`).

Throughput and time-to-policy numbers are **Phase 0 benchmark gates**, not claimed facts. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Official sources

- BIOBUZZ Competition Manual / Game Details: https://ftc-resources.firstinspires.org/ftc/game/manual-10
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

Optional extras: `[scale]` Ray/RLlib, `[mujoco]` required for BIOBUZZ 3D mesh physics, `[cad]` official STEP tessellation, `[postgres]` when `TALONGYM_DATABASE_URL` is set. LSTM ONNX export is 501; `python -m talongym distill` writes a feed-forward demo and is not a deployment path. The only match-bound artifact is a Road Runner 1.0 snippet pasted into an AUTO OpMode.

Open http://127.0.0.1:8765 after `cd web && npm install && npm run build`, or use Vite (`npm run dev` in `web/`, API on :8765). State lives in `var/talongym.db` — not in the browser.

Train, evaluate, replay, and export (full flags and Lab budgets): **[docs/TRAINING.md](docs/TRAINING.md)** and **[docs/README.md](docs/README.md)**.

```bash
python -m talongym detect
python -m talongym defaults --training biobuzz_auto_easy
python -m talongym train --easy
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
| [docs/MENTOR_SIGNOFF.md](docs/MENTOR_SIGNOFF.md) | BIOBUZZ V1 freeze questions |

JSON Schema files live in [`schemas/`](schemas/).

## Contributing

Setup, PR checklist, and the steps to flip the GitHub repo from private to public: **[CONTRIBUTING.md](CONTRIBUTING.md)**. People who contributed substantial work: **[CONTRIBUTORS.md](CONTRIBUTORS.md)**. Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security reports: [SECURITY.md](SECURITY.md).

Do not commit `var/` (logs, SQLite, checkpoints), virtualenvs, official FIRST STEP/CAD, or team robot CAD. BIOBUZZ derived tessellation (`assets/seasons/biobuzz_2026/`: `field.glb`, `collision/`, `pieces/`, `mechanisms/`, `cad_manifest.json`, `field_mjcf.xml`) is committed so a clone can train and open Lab without regenerating meshes. Rebuild from a new official STEP with `python -m talongym import-field-cad`. That command downloads the official binary STEP endpoint, verifies hashes, and refuses AABB fallback on `mesh_field_collision` seasons. Raw STEP stays under `var/cad/` (gitignored).

## License

Copyright (C) 2026 TalonGym contributors. Built by FTC Team 17986 904 Robo Eagles and Team 27268 Talon Strike.

TalonGym is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License](LICENSE) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

**Additional permission (export snippets):** Java emitted by `python -m talongym` export / Lab export is generated output for pasting into *your* AUTO OpMode. Those snippets are dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Your robot code does not become GPL just because you pasted a trajectory.

FIRST®, FIRST® Tech Challenge, FTC®, BIOBUZZ™, and FIRST CANOPY™ are trademarks or service marks of FIRST. Road Runner and MeepMeep are third-party projects; we are not affiliated with them. Point values and field facts in presets come from public Competition Manuals — if a Team Update changes them, bump `manualRevision` rather than treating this repo as rules authority.
