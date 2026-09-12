# Install

TalonGym is a Python 3.12+ package plus a React Lab. Physics and scoring run in Python. The browser only renders.

## Python

From the repository root:

```bash
python -m pip install -e ".[dev,rl]"
```

That extra set is the laptop/workstation path: tests (`dev`) plus RecurrentPPO (`rl`: torch, stable-baselines3, sb3-contrib).

| Extra | Install when | Provides |
|-------|----------------|----------|
| `dev` | Always for contributors | pytest |
| `rl` | Training a policy | RecurrentPPO |
| `scale` | Optional Ray trainer | `python -m talongym train --algo rllib_ppo` |
| `mujoco` | 3D pose-check only | `python -m talongym validate-3d` |
| `postgres` | Shared DB | used if `TALONGYM_DATABASE_URL` starts with `postgres` |

Entry points after install: `talongym` and `python -m talongym`.

Without `[rl]`, CLI `train` fails unless you pass `--allow-scripted`. Lab **Demo** runs also allow that fallback; **Short** and **Preset** do not.

## Lab UI

Build once and let `python -m talongym lab` serve the SPA from `web/dist`:

```bash
cd web
npm install
npm run build
cd ..
python -m talongym lab
```

Open http://127.0.0.1:8765.

For a hot-reload frontend, keep the API on 8765 and Vite on 5173 (it proxies `/api` and WebSockets):

```bash
python -m talongym lab
# another terminal
cd web && npm install && npm run dev
```

Open http://127.0.0.1:5173. Details: [LAB.md](LAB.md).

## Storage

| Location | Contents |
|----------|----------|
| `var/talongym.db` | SQLite: presets, runs, evaluations, artifacts, replays (WAL) |
| `var/defaults.json` | Your active field/robot/scoring/training ids |
| `var/ckpts/` | RecurrentPPO zips (`latest.zip`, `best.zip`, per-run folders) |
| `var/last_replay.java` | CLI `replay` export |

`var/` is gitignored. Copy the SQLite file to share a mentor-trained run with a laptop that cannot train.

Set `TALONGYM_DATABASE_URL` to a `postgres://…` URL and install `[postgres]` to use Postgres instead of SQLite. If the URL is unset, Lab uses `var/talongym.db`.

## Tests

```bash
python -m pytest
```

`[rl]` is required for PPO smoke tests that import sb3-contrib. MuJoCo tests skip unless `[mujoco]` is installed.

## Hardware modes

Starting points from the training presets (`nEnvs`). CLI `train` caps parallel envs at 8 unless you pass `--n-envs`.

| Profile | Typical n_envs | Notes |
|---------|----------------|-------|
| Lightweight / local | 8 | CPU laptop; still reports true score |
| Workstation | 128–512 in the spec; shipped presets use 8 | Raise `nEnvs` in the training JSON |
| Cloud | Ray via `[scale]` | Optional; laptop default stays RecurrentPPO |

The 2.5D engine is the production path. Health may report `planar2d` when the Rapier crate is a stub.
