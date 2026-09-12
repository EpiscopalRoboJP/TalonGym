# TalonGym docs

How-to pages for running TalonGym today. The architecture spec, Kickoff runbook, and mentor sign-off stay in the files listed at the bottom.

## First hour

1. Install Python extras and build the Lab: [INSTALL.md](INSTALL.md)
2. Pick a season bundle (`defaults --training …`): [PRESETS.md](PRESETS.md)
3. Train RecurrentPPO from the CLI or `/train`: [TRAINING.md](TRAINING.md)
4. Rank candidates with held-out trials and CIs: [EVALUATION.md](EVALUATION.md)
5. Scrub a replay in the Lab: [LAB.md](LAB.md)
6. Export Road Runner 1.0 Actions and paste into an AUTO OpMode: [EXPORT.md](EXPORT.md)

Every CLI command, including calibrate and distill: [CLI.md](CLI.md).

## Operator how-tos

| Doc | Use it when you want to |
|-----|-------------------------|
| [INSTALL.md](INSTALL.md) | Set up Python, RL extras, the web UI, and storage |
| [TRAINING.md](TRAINING.md) | Train a policy (CLI + Lab budgets, checkpoints, curriculum) |
| [CLI.md](CLI.md) | Look up a command and its flags |
| [LAB.md](LAB.md) | Use Replay, Train, Field, Robot, and Compare in the browser |
| [PRESETS.md](PRESETS.md) | Switch seasons, lint JSON, and keep nouns in data |
| [EVALUATION.md](EVALUATION.md) | Run the statistical harness and read the leaderboard |
| [EXPORT.md](EXPORT.md) | Export Road Runner Java to paste into an AUTO OpMode; distill ONNX (demo); fit a robot overlay |

## Spec and season process

| Doc | Role |
|-----|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Gymnasium contract, API, roadmap, risks (implementation spec) |
| [NEW_SEASON_RUNBOOK.md](NEW_SEASON_RUNBOOK.md) | Stand up a new game from the Competition Manual without an engine PR |
| [MENTOR_SIGNOFF.md](MENTOR_SIGNOFF.md) | DECODE TU32 freeze questions |
| [../presets/seasons/biobuzz_2026/README.md](../presets/seasons/biobuzz_2026/README.md) | BIOBUZZ V1 encoded facts and placeholders |

JSON Schema for presets lives in [`../schemas/`](../schemas/).

## Deployment path (always and only)

Train offline → export a trajectory → paste into an AUTO OpMode → Control Hub runs that OpMode.

TalonGym is off-field only. The Control Hub runs the pasted OpMode; TalonGym does not talk to the robot during a MATCH. Full boundary: [../README.md](../README.md).
