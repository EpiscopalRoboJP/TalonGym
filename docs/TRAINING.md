# Train a policy

TalonGym trains an LSTM policy on a 30-second AUTO episode. BIOBUZZ uses **`bc_then_ppo`**: clone `scripted_biobuzz`, then asymmetric-critic PPO (actor encoder-only, critic sees privileged 3D state). `grpo` in `python/talongym/training/grpo.py` is experimental unused code (no shipped preset). The leaderboard uses **true score** only.

Algorithm and observation contract: [ARCHITECTURE.md](ARCHITECTURE.md) §4. This page is the operator path.

## 1. Pick the season bundle

Repo default in [`presets/defaults.json`](../presets/defaults.json) is BIOBUZZ. Override without editing the repo:

```bash
python -m talongym defaults --training biobuzz_auto_lightweight
python -m talongym defaults
```

`--training` copies that preset’s `fieldId` / `robotId` / `scoringId` into `var/defaults.json`. CLI `train` / `evaluate` / `replay` and Lab jobs that omit ids all load this bundle.

Shipped training ids (four compute variants):

| Family | lightweight | workstation | cloud (nEnvs) | easy (autodetect) |
|--------|-------------|-------------|-------------|-------------------|
| BIOBUZZ V1 | `biobuzz_auto_lightweight` | `biobuzz_auto_workstation` | `biobuzz_auto_cloud` | `biobuzz_auto_easy` |

BIOBUZZ training presets keep `bc_then_ppo` / RecurrentPPO, including cloud (larger `nEnvs`). `rllib_ppo` is an experimental one-shot toy in `training/rllib.py`; Lab/CLI fall back to RecurrentPPO. Do not treat it as a production scale path.

`computeProfile: "auto"` (the `*_easy` files) resolves at train time from CPU count, RAM, and CUDA. Override with `TALONGYM_COMPUTE_PROFILE=lightweight_cpu|workstation|cloud`. Print the detection:

```bash
python -m talongym detect
python -m talongym train --easy
```

More: [PRESETS.md](PRESETS.md).

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
| `--algo` | from the training preset | RecurrentPPO / `bc_then_ppo`. `rllib_ppo` is a toy one-shot (`[scale]`) |
| `--allow-scripted` | off | If sb3 is missing, run the scripted AUTO instead of failing |

The loop:

1. Builds `FTCAutoEnv` with Dict observations and `AsymmetricLstmPolicy` (actor drops `_privileged`).
2. Optional BC warmup (`algorithm.bcWarmupSteps`) from the scripted AUTO. Demonstrations mix legal-spawn AUTO with launch-pose `mechanism_ready` episodes and supply **actions only**; they never inject launches or points. Launch-pose-only clones park for LEAVE without firing.
3. Asserts the scripted baseline can physically launch and score from the launch pose before long runs (`total_steps >= 2048`).
4. Wraps with `EncoderOnlyObsAssertWrapper` so privileged motif/match vars cannot leak into the actor.
5. Applies curriculum unlocks on each reset (`spawn_at_launch`, `spawn_approach`, then legal spawn, then `full_noise`).
6. Saves `var/ckpts/latest.zip` every chunk and `var/ckpts/best.zip` only when the held-out **objective** improves **and** the eval is healthy (at least one physical launch, wall contact within 8 s).
7. Prints `true=` (episode true-score mean) and `eval=` (held-out true-score mean), plus launch count and wall-contact time. Use `eval`, not shaping.

A short run is a smoke test. Lightweight presets declare `budget.totalEnvSteps` of 5e6 and a 4-hour wall-clock cap; pass a larger `--steps` for an overnight CLI job.

## 3. Train from the Lab

1. `python -m talongym lab` and open `/train` (see [LAB.md](LAB.md)).
2. Choose field, robot, scoring, and training presets. The 3D display previews that robot on that field; **Save as default** stores the same ids for CLI jobs.
3. Pick a budget, then **Start run**.
4. To change AUTO start pose: open **Advanced presets**, check **Configure robot starts**, pick an official same-alliance slot, and slide along the wall. Off-wall, opponent-side, LOADING ZONE, and FLOWER poses are rejected.

| Budget | Env steps | n_envs | Scripted fallback if `[rl]` missing |
|--------|-----------|--------|-------------------------------------|
| Demo | 4096 | 2 | Yes |
| Short | 16384 | 4 | No |
| Easy | From `*_easy` JSON (250k) | Autodetected | No |
| Preset | From JSON (`totalEnvSteps` / `nEnvs`) | From JSON | No |

The dashboard shows:

- **True score** — episode mean; this is the leaderboard series
- **Held-out eval true score** — small eval on seeds from `evaluation.heldOutSeedStart`
- **Launches / wall** — physical launch count and wall-contact seconds on the eval episode. A zero-launch or high-wall policy cannot become `best.zip`
- **Shaping** — labeled not-leaderboard
- Live downsampled rollout, curriculum stage, entropy, approx KL, FPS
- Cancel — cooperative stop (`cancelling` until the worker acknowledges `cancelled`); partial checkpoint may still be on disk

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

Unlocks come from the training preset, not engine code. BIOBUZZ stages are distinct physical scaffolds. Demonstrations clone actions only; scoring still requires flywheel/contact launch in the sim. BC warmup always includes at least one legal-spawn episode so `best.zip` can pass held-out eval, which is the full AUTO (not the launch-pose scaffold).

```json
"curriculum": [
  { "untilFrac": 0.15, "unlock": ["spawn_at_launch", "mechanism_ready"] },
  { "untilFrac": 0.4, "unlock": ["spawn_approach"] },
  { "untilFrac": 0.75, "unlock": [] },
  { "untilFrac": 1.0, "unlock": ["full_noise"] }
]
```

| Unlock | Effect |
|--------|--------|
| `spawn_at_launch` + `mechanism_ready` | Training-only spawn at the launch pose with the flywheel already at ready RPM and the hood aimed. The policy still has to command fire; this does not inject launches or points |
| `spawn_approach` | Training-only spawn halfway between the legal start and the launch pose |
| (none) | Legal G304 spawn; preloaded score-and-park |
| `full_noise` | Full domain randomization on the complete AUTO task |
| `scripted_teammate` / `scripted_opponent` | Override teammate/opponent policy for that stage |

Curriculum spawn never changes scoring physics or injects points. Mesh BIOBUZZ runs do not unlock `scripted_launch` or `ballistic_launch`. A launched piece leaves the magazine only after flywheel RPM and gate opening; MuJoCo then integrates the shot. The hive CAD currently intercepts many up-CELL trajectories, so the fail-closed baseline requires at least three physical launches plus AUTO LEAVE, not a silent zero-launch park.

BIOBUZZ AUTO has no motif. `motif_known_at_t0` remains a generic unlock for a future season that needs a match variable.

Default robot is `gobilda_mecanum_starter` (schema 1.2 catalog scoring assembly). Sister starters: `gobilda_tank_starter`, `rev_mecanum_starter`, `rev_tank_starter`. Drivebase recipes stay chassis-only until you add intake/flywheel parts and confirm scoring topology.

## Action tier

Shipped presets use `high_level_waypoint`: target pose (inches / rad), speed fraction, discrete mechanism. That maps onto Road Runner export — the only field path (paste into an AUTO OpMode; Control Hub runs it). If a training JSON omits `actionTier`, PPO uses `robot.defaultActionTier`. `physical_actuators` is a legal tier (Lab robot builder). `low_level_velocity` exists on the env (`vx, vy, ω`) for transfer work; do not use it if you need a pasteable AUTO.

## Scripted baseline

[`python/talongym/training/policies.py`](../python/talongym/training/policies.py) is HIVE TIP + LEAVE + PARK for BIOBUZZ. Use it to confirm the env can physically launch and score, not as “the auto.” Long training runs fail closed if that baseline launches 0 pieces. Compare trained checkpoints against it on [EVALUATION.md](EVALUATION.md).

A zero-launch `best.zip` is a simulator or robot-contract defect, not a PPO hyperparameter issue. Lab run `1d66d5d4b63e` used a catalog robot (`test1`) that compiled without a 4-piece magazine, so eval true score 3.0 was LEAVE from parking into walls with no flywheel fire (0 launches, 4 held at park, 0.68 s wall). Replayed on `mecanum_biobuzz_4cap` it stays unhealthy. The repaired default `gobilda_mecanum_starter` scripted baseline launches 4 pieces (wall 0.12 s). A 2048-step `bc_then_ppo` smoke on that starter selected a healthy `best.zip` (4 physical launches, 1.0 s wall, under the 8 s gate). Hive CAD still intercepts many up-CELL shots, so LEAVE 3.0 with launches is success; 3.0 with zero launches is not. `var/defaults.json` overlaying `mecanum_biobuzz_4cap` also hid the shipped starter — reset Lab defaults to `gobilda_mecanum_starter` for CLI jobs.

## Experimental RLlib (not a scale path)

```bash
python -m pip install -e ".[scale]"
python -m talongym train --algo rllib_ppo --steps 2048
```

That runs a **one-shot toy** `algo.train()` against a flattened env. Laptop/cloud product path remains RecurrentPPO. Lab Start run falls back to RecurrentPPO if RLlib is missing.
