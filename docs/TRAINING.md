# Train a policy

TalonGym trains a **RecurrentPPO** (LSTM) policy on a 30-second AUTO episode. The policy sees noisy sensors only. The leaderboard uses **true score** (official AUTO points). Shaping is plotted separately and never ranked.

Algorithm and observation contract: [ARCHITECTURE.md](ARCHITECTURE.md) §4. This page is the operator path.

## 1. Pick the season bundle

Repo default in [`presets/defaults.json`](../presets/defaults.json) is DECODE. Override without editing the repo:

```bash
python -m talongym defaults --training biobuzz_auto_lightweight
python -m talongym defaults
```

`--training` copies that preset’s `fieldId` / `robotId` / `scoringId` into `var/defaults.json`. CLI `train` / `evaluate` / `replay` and Lab jobs that omit ids all load this bundle.

Shipped training ids:

| Training preset | Field / scoring |
|-----------------|-----------------|
| `decode_auto_lightweight` | DECODE TU32 |
| `into_the_deep_auto_lightweight` | INTO THE DEEP |
| `centerstage_auto_lightweight` | CENTERSTAGE |
| `biobuzz_auto_lightweight` | BIOBUZZ V1 (`mecanum_biobuzz_4cap`) |

More: [PRESETS.md](PRESETS.md).

## 2. Train from the CLI

```bash
python -m pip install -e ".[rl]"
python -m talongym train --steps 8192
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--steps` | 8192 | Total env steps this invocation |
| `--n-envs` | `min(preset nEnvs or 4, 8)` | Parallel `DummyVecEnv` workers |
| `--algo` | `recurrent_ppo` | `rllib_ppo` needs `pip install -e ".[scale]"` |
| `--allow-scripted` | off | If sb3 is missing, run the scripted AUTO instead of failing |

The loop:

1. Builds `FTCAutoEnv` with Dict observations and an LSTM (`MultiInputLstmPolicy`).
2. Wraps with `EncoderOnlyObsAssertWrapper` so privileged motif/match vars cannot leak into `learn`.
3. Applies curriculum unlocks on each reset from `domainRandomization.curriculum`.
4. Saves `var/ckpts/latest.zip` every chunk and `var/ckpts/best.zip` when the held-out **objective** improves.
5. Prints `true=` (episode true-score mean) and `eval=` (held-out true-score mean). Use `eval`, not shaping.

A short run is a smoke test. Lightweight presets declare `budget.totalEnvSteps` of 5e6 and a 4-hour wall-clock cap; pass a larger `--steps` for an overnight CLI job.

## 3. Train from the Lab

1. `python -m talongym lab` and open `/train` (see [LAB.md](LAB.md)).
2. Choose field, robot, scoring, and training presets (or **Set as default**).
3. Pick a budget, then **Start run**.

| Budget | Env steps | n_envs | Scripted fallback if `[rl]` missing |
|--------|-----------|--------|-------------------------------------|
| Demo | 4096 | 2 | Yes |
| Short | 16384 | 4 | No |
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

`train_ppo(..., resume=True)` reloads `latest.zip` in that `save_dir`. The Lab start button does not pass `resume`; POST `/api/v1/runs` with `"resume": true` continues a matching save dir. Evaluate a zip from Compare (**Optional checkpoint eval**) or `POST /evaluations` with `"policy": "checkpoint"` and `runId`.

## What “best” means during training

The training JSON `objective` is one of:

- `mean_true_score` (default)
- `p10_true_score`
- `lcb_true_score` (bootstrap lower bound)

In-loop eval uses a handful of held-out episodes (1–4 unless you change `eval_episodes`). That is **not** the 500-trial report. After training, run [EVALUATION.md](EVALUATION.md). `bestLabelEligible` is false until n ≥ 500 and CIs do not overlap.

Early stop: `budget.earlyStopNoImproveSteps` (1e6 in the shipped presets). Wall clock: `budget.wallClockLimitS` (14400 s).

## Curriculum

Unlocks come from the training preset, not engine code. DECODE lightweight:

```json
"curriculum": [
  { "untilFrac": 0.2, "unlock": ["motif_known_at_t0"] },
  { "untilFrac": 1.0, "unlock": ["motif_must_sense", "full_noise", "scripted_teammate"] }
]
```

| Unlock | Effect |
|--------|--------|
| `motif_known_at_t0` | Match variable visible at reset (curriculum only) |
| `motif_must_sense` | Policy must see the motif via sensors (`observeVia`) |
| `full_noise` | Full domain randomization scales |
| `scripted_teammate` / `scripted_opponent` | Override teammate/opponent policy for that stage |

BIOBUZZ lightweight unlocks `full_noise` for the whole run (no motif in AUTO).

## Action tier

Shipped presets use `high_level_waypoint`: target pose (inches / rad), speed fraction, discrete mechanism. That maps onto Road Runner export. `low_level_velocity` exists on the env (`vx, vy, ω`) for transfer work; do not use it if you need a pasteable AUTO.

## Scripted baseline

[`python/talongym/training/policies.py`](../python/talongym/training/policies.py) is LEAVE + intake + one score, per season slug. Use it to confirm the env scores, not as “the auto.” Compare trained checkpoints against it on [EVALUATION.md](EVALUATION.md).

## Optional RLlib

```bash
python -m pip install -e ".[scale]"
python -m talongym train --algo rllib_ppo --steps 2048
```

Laptop default remains RecurrentPPO. RLlib is a scale extra, not the Lab path.
