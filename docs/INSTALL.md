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
| `scale` | Optional experimental Ray extra | one-shot `train --algo rllib_ppo` toy; not a production scale path |
| `mujoco` | BIOBUZZ 3D mesh physics (and `validate-3d`) | `MujocoFieldBackend` |
| `cad` | Official field/piece STEP *and* catalog/team robot CAD (STL/OBJ/GLB/STEP) | trimesh + cascadio + fast-simplification |
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
| `var/talongym.db` | SQLite: presets, robot drafts, runs, evaluations, artifacts, replays, jobs (WAL). `jobs` is a status log; training runs in-process. |
| `var/defaults.json` | Your active field/robot/scoring/training ids |
| `var/ckpts/` | RecurrentPPO zips (`latest.zip`, `best.zip`, per-run folders) |
| `var/last_replay.java` | CLI `replay` export |
| `var/assets/robot_parts/` | On-demand manufacturer CAD cache (GLB/collision). Not in git. |

`var/` is gitignored. Copy the SQLite file to share a mentor-trained run with a laptop that cannot train. BIOBUZZ tessellation under `assets/seasons/biobuzz_2026/` (`field.glb`, `collision/`, `pieces/`, `mechanisms/`, `cad_manifest.json`, `field_mjcf.xml`) is committed with the repo. Rebuild from a new official STEP with `python -m talongym import-field-cad`.

Set `TALONGYM_DATABASE_URL` to a `postgres://…` URL and install `[postgres]` to use Postgres instead of SQLite. If the URL is unset, Lab uses `var/talongym.db`.

## Tests

```bash
python -m pytest
cd web && npm test
```

`[rl]` is required for PPO smoke tests that import sb3-contrib. MuJoCo tests skip unless `[mujoco]` is installed. `npm test` runs the self-contained TypeScript files under `web/src`.

## Hardware modes

`python -m talongym detect` classifies this machine. Starting points also live on the training JSON (`nEnvs`). CLI `train --easy` uses the detected count; an explicit preset keeps its JSON `nEnvs` unless you pass `--n-envs`.

| Profile | Typical n_envs | Notes |
|---------|----------------|-------|
| Lightweight / local | 8 | CPU laptop; still reports true score |
| Workstation | 256 in the spec; autodetect scales with cores | Overnight desktop path |
| Cloud | 1024 RecurrentPPO envs (`*_cloud`) | Optional; same algorithm as workstation, larger `nEnvs`. `[scale]` RLlib is a toy extra. |

The 2.5D engine is the production path. Health may report `planar2d` when the Rapier crate is a stub.
