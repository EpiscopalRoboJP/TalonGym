# Train a policy

TalonGym trains an LSTM policy on a 30-second AUTO episode. BIOBUZZ uses **`bc_then_ppo`**: clone `scripted_biobuzz`, then asymmetric-critic PPO (actor encoder-only, critic sees privileged 3D state). Optional **`grpo`** is a value-free group baseline for sparse HIVE TIP. The leaderboard uses **true score** only.

Algorithm and observation contract: [ARCHITECTURE.md](ARCHITECTURE.md) §4. This page is the operator path.

## 1. Pick the season bundle

Repo default in [`presets/defaults.json`](../presets/defaults.json) is BIOBUZZ. Override without editing the repo:

```bash
python -m talongym defaults --training biobuzz_auto_lightweight
python -m talongym defaults
```

`--training` copies that preset’s `fieldId` / `robotId` / `scoringId` into `var/defaults.json`. CLI `train` / `evaluate` / `replay` and Lab jobs that omit ids all load this bundle.

Shipped training ids (four compute variants):

| Family | lightweight | workstation | cloud (Ray) | easy (autodetect) |
|--------|-------------|-------------|-------------|-------------------|
| BIOBUZZ V1 | `biobuzz_auto_lightweight` | `biobuzz_auto_workstation` | `biobuzz_auto_cloud` | `biobuzz_auto_easy` |

BIOBUZZ lightweight/workstation/easy keep `bc_then_ppo`. Cloud presets set `rllib_ppo` (Lab/CLI fall back to RecurrentPPO if `[scale]` is missing).

`computeProfile: "auto"` (the `*_easy` files) resolves at train time from CPU count, RAM, and an accelerator (CUDA/ROCm or Apple Metal). Override with `TALONGYM_COMPUTE_PROFILE=lightweight_cpu|workstation|cloud`. Print the detection:

```bash
python -m talongym detect
python -m talongym train --easy
```

More: [PRESETS.md](PRESETS.md).

### Hardware acceleration

RecurrentPPO's torch device is picked automatically, independent of `computeProfile`: NVIDIA CUDA (or an AMD ROCm build of torch, which reports through the same `torch.cuda` API) if present, else Apple Silicon Metal (`mps`), else CPU. `python -m talongym detect` prints the resolved device as `torchDevice`. Force a specific device with `TALONGYM_TORCH_DEVICE=cuda|mps|cpu` (or a specific index, e.g. `cuda:1`) — useful on a multi-GPU box or to force CPU for a reproducible run. Ray/RLlib (`rllib_ppo`, the cloud profile) only supports CUDA GPUs; it stays on CPU elsewhere.

## 2. Train from the CLI

```bash
python -m pip install -e ".[rl,mujoco]"
python -m talongym detect
python -m talongym train --easy
python -m talongym train --steps 8192
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--easy` | off | Load the season `*_easy` preset and autodetect `nEnvs` |
| `--training` | active default | Training-run id for this invocation |
| `--steps` | 8192 (easy: preset budget) | Total env steps this invocation |
| `--n-envs` | preset `nEnvs` (easy: detected) | Parallel `DummyVecEnv` workers |
| `--algo` | from the training preset | `rllib_ppo` needs `pip install -e ".[scale]"` |
| `--allow-scripted` | off | If sb3 is missing, run the scripted AUTO instead of failing |

The loop:

1. Builds `FTCAutoEnv` with Dict observations and `AsymmetricLstmPolicy` (actor drops `_privileged`).
2. Optional BC warmup (`algorithm.bcWarmupSteps`) from the scripted AUTO.
3. Wraps with `EncoderOnlyObsAssertWrapper` so privileged motif/match vars cannot leak into the actor.
4. Applies curriculum unlocks on each reset (`scripted_launch` / `ballistic_launch` / `full_noise`).
5. Saves `var/ckpts/latest.zip` every chunk and `var/ckpts/best.zip` when the held-out **objective** improves.
6. Prints `true=` (episode true-score mean) and `eval=` (held-out true-score mean). Use `eval`, not shaping.

A short run is a smoke test. Lightweight presets declare `budget.totalEnvSteps` of 5e6 and a 4-hour wall-clock cap; pass a larger `--steps` for an overnight CLI job.

## 3. Train from the Lab

1. `python -m talongym lab` and open `/train` (see [LAB.md](LAB.md)).
2. Choose field, robot, scoring, and training presets (or **Set as default**).
3. Pick a budget, then **Start run**.

| Budget | Env steps | n_envs | Scripted fallback if `[rl]` missing |
|--------|-----------|--------|-------------------------------------|
| Demo | 4096 | 2 | Yes |
| Short | 16384 | 4 | No |
| Easy | From `*_easy` JSON (250k) | Autodetected | No |
| Preset | From JSON (`totalEnvSteps` / `nEnvs`) | From JSON | No |

The dashboard shows:

- **True score** — episode mean; this is the leaderboard series
- **Held-out eval true score** — small eval on seeds from `evaluation.heldOutSeedStart`
- **Shaping** — labeled not-leaderboard
- Live downsampled rollout, curriculum stage, entropy, approx KL, FPS
- Cancel — cooperative stop; partial checkpoint may still be on disk

Runs persist in SQLite. Open a past run from the list to reconnect the WebSocket.

## Checkpoints and resume

| Path | What |
|------|------|
| `var/ckpts/latest.zip` | CLI train latest |
| `var/ckpts/best.zip` | CLI train best held-out objective |
| `var/ckpts/<runId>/latest.zip` | Lab/API run latest |
| `var/ckpts/<runId>/best.zip` | Lab/API run best |

`train_ppo(..., resume=True)` reloads `latest.zip` in that `save_dir`. The Lab start button does not pass `resume`; POST `/api/v1/runs` with `"resume": true` continues a matching save dir. Evaluate a zip from the Lab **Evaluate** page (*Trained checkpoint*) or `POST /evaluations` with `"policy": "checkpoint"` and `runId`.

## What “best” means during training

The training JSON `objective` is one of:

- `mean_true_score` (default)
- `p10_true_score`
- `lcb_true_score` (bootstrap lower bound)

In-loop eval uses a handful of held-out episodes (1–4 unless you change `eval_episodes`). That is **not** the 500-trial report. After training, run [EVALUATION.md](EVALUATION.md). `bestLabelEligible` is false until n ≥ 500 and CIs do not overlap.

Early stop: `budget.earlyStopNoImproveSteps` (1e6 in the shipped presets). Wall clock: `budget.wallClockLimitS` (14400 s).

## Curriculum

Unlocks come from the training preset, not engine code. BIOBUZZ lightweight:

```json
"curriculum": [
  { "untilFrac": 0.25, "unlock": ["scripted_launch"] },
  { "untilFrac": 0.7, "unlock": ["ballistic_launch"] },
  { "untilFrac": 1.0, "unlock": ["ballistic_launch", "full_noise"] }
]
```

| Unlock | Effect |
|--------|--------|
| `scripted_launch` | Teleport / auto-aim into the CELL (early BC) |
| `ballistic_launch` | 3D muzzle velocity vs the mesh field |
| `full_noise` | Full domain randomization scales |
| `scripted_teammate` / `scripted_opponent` | Override teammate/opponent policy for that stage |

BIOBUZZ AUTO has no motif. `motif_known_at_t0` remains a generic unlock for a future season that needs a match variable.

## Action tier

Shipped presets use `high_level_waypoint`: target pose (inches / rad), speed fraction, discrete mechanism. That maps onto Road Runner export — the only field path (paste into an AUTO OpMode; Control Hub runs it). `low_level_velocity` exists on the env (`vx, vy, ω`) for transfer work; do not use it if you need a pasteable AUTO.

## Scripted baseline

[`python/talongym/training/policies.py`](../python/talongym/training/policies.py) is LEAVE + intake + one score, per season slug. Use it to confirm the env scores, not as “the auto.” Compare trained checkpoints against it on [EVALUATION.md](EVALUATION.md).

## Optional RLlib

```bash
python -m pip install -e ".[scale]"
python -m talongym train --algo rllib_ppo --steps 2048
```

Laptop default remains RecurrentPPO. RLlib is a scale extra, not the Lab path.
