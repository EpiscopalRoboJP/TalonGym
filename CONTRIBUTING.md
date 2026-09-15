# Contributing

TalonGym is a training aid for FTC Autonomous, licensed **GPL-3.0-or-later**. PRs that keep that boundary — train offline, export a trajectory, paste into an AUTO OpMode — are welcome. By contributing, you license your work under the same terms. Exported Road Runner snippets stay CC0 (see the README) so team OpModes are not copylefted.

Named contributors for substantial work are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Setup

```bash
python -m pip install -e ".[dev,rl]"
cd web && npm install && npm run build && cd ..
python -m pytest
```

Optional extras (`[mujoco]`, `[cad]`, `[scale]`, `[postgres]`): [docs/INSTALL.md](docs/INSTALL.md).

## What to send

- Bug fixes and tests that fail without the fix
- Docs that match current CLI / Lab behavior
- Season preset data with a Competition Manual citation (`manualRevision`, `verifyAgainstManual` when the figure is still a guess)

## What not to send

- Live robot control, match-time policy streaming, or anything that would be illegal play
- Official FIRST STEP/CAD files, team robot CAD, raw logs, checkpoints, SQLite DBs, or `.env` files (`var/` stays gitignored). Shipped BIOBUZZ tessellation under `assets/seasons/biobuzz_2026/` is already in git — do not add extra seasons of generated meshes unless they are the playable field for a clone.
- Invented scoring or geometry with no manual citation

## PR checklist

- [ ] `python -m pytest` passes (skip notes in the PR if an extra was not installed)
- [ ] New preset numbers cite the manual revision
- [ ] No secrets, venv, `node_modules`, `var/`, or official STEP/CAD

## Conduct

Be kind. This is a student-and-mentor project. Harassment or FIRST-rule-evasion discussion is not accepted.

## Making the GitHub repo public

The remote can stay private until you are ready. Before flipping visibility:

1. Confirm `LICENSE`, `SECURITY.md`, and this file are on `main`
2. Confirm `git ls-files` has no `.env`, `*.db`, `*.log`, `.venv`, official STEP/CAD, or checkpoint zips
3. GitHub → Settings → Danger Zone → Change repository visibility → Public
4. Enable private vulnerability reporting (Settings → Code security)
5. Add topics such as `ftc`, `first-tech-challenge`, `reinforcement-learning`
