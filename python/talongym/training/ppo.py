from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from talongym import paths
from talongym.env.ftc_auto import (
    BoxActionDictObsEnv,
    EncoderOnlyObsAssertWrapper,
    FTCAutoEnv,
    flatten_obs,
)
from talongym.eval.harness import bootstrap_ci
from talongym.presets.loader import LoadedPresets, load_bundle
from talongym.training.curriculum import (
    full_noise,
    motif_known_at_t0,
    objective_value,
    opponent_for,
    stage_info,
    teammate_for,
)
from talongym.training.policies import scripted_auto


def record_policy_episode(policy, bundle: LoadedPresets | None = None, seed: int = 0) -> list[dict[str, Any]]:
    if hasattr(policy, "reset_lstm"):
        policy.reset_lstm()
    env = FTCAutoEnv(bundle=bundle or load_bundle(), record=True)
    obs, info = env.reset(seed=seed)
    term = trunc = False
    while not term and not trunc:
        obs, _, term, trunc, info = env.step(policy(obs, info))
    frames = list(env.frames)
    env.close()
    return frames


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = max(1, total)
        self.steps = 0

    @property
    def frac(self) -> float:
        return min(1.0, self.steps / self.total)


class CurriculumEnv(gym.Wrapper):
    """Re-apply curriculum unlocks on every reset from shared training progress."""

    def __init__(self, env: gym.Env, progress: _Progress, training: dict[str, Any] | None) -> None:
        super().__init__(env)
        self._progress = progress
        self._training = training

    def reset(self, **kwargs):
        frac = self._progress.frac
        opts = dict(kwargs.get("options") or {})
        opts["motif_known_at_t0"] = motif_known_at_t0(self._training, frac)
        opts["teammate_policy"] = teammate_for(self._training, frac)
        opts["opponent_policy"] = opponent_for(self._training, frac)
        opts["full_noise"] = full_noise(self._training, frac)
        kwargs["options"] = opts
        return super().reset(**kwargs)


def _eval_true_scores(
    adapter: "RecurrentPolicyAdapter",
    bundle: LoadedPresets,
    seeds: list[int],
    record_first: bool = False,
) -> tuple[list[float], list[dict[str, Any]]]:
    scores: list[float] = []
    frames: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        adapter.reset_lstm()
        env = FTCAutoEnv(bundle=bundle, record=record_first and i == 0)
        obs, info = env.reset(seed=int(seed))
        term = trunc = False
        while not term and not trunc:
            obs, _, term, trunc, info = env.step(adapter(obs, info))
        scores.append(float(info.get("true_score") or 0.0))
        if record_first and i == 0:
            frames = list(env.frames)
        env.close()
    return scores, frames


def train_ppo(
    bundle: LoadedPresets | None = None,
    total_steps: int = 50_000,
    n_envs: int = 8,
    log: Callable[[str], None] | None = None,
    on_metrics: Callable[[dict[str, Any]], None] | None = None,
    on_rollout: Callable[[list[dict[str, Any]]], None] | None = None,
    allow_scripted: bool = False,
    save_dir: Path | None = None,
    frozen_policy: Any = None,
    should_stop: Callable[[], bool] | None = None,
    resume: bool = False,
    demo: bool = False,
    eval_episodes: int | None = None,
) -> dict[str, Any]:
    """Train RecurrentPPO with Dict observations and a live curriculum."""
    emit = log or (lambda m: None)
    bundle = bundle or load_bundle()
    total_steps = max(1, int(total_steps))
    progress = _Progress(total_steps)
    training = bundle.training
    algo_cfg = (training or {}).get("algorithm") or {}
    budget_cfg = (training or {}).get("budget") or {}
    eval_cfg = (training or {}).get("evaluation") or {}
    objective_name = str((training or {}).get("objective") or "mean_true_score")
    action_tier = str((training or {}).get("actionTier") or "high_level_waypoint")
    n_envs = max(1, int(n_envs))
    eval_n = int(eval_episodes if eval_episodes is not None else (1 if total_steps < 2048 else 4))
    held0 = int(eval_cfg.get("heldOutSeedStart") or 10_000_000)
    wall_limit = budget_cfg.get("wallClockLimitS")
    early_stop = budget_cfg.get("earlyStopNoImproveSteps")
    t0 = time.monotonic()

    def make_env():
        def _init():
            env = FTCAutoEnv(
                bundle=bundle,
                record=False,
                motif_known_at_t0=motif_known_at_t0(training, progress.frac),
                teammate_policy=teammate_for(training, progress.frac),
                opponent_policy=opponent_for(training, progress.frac),
                action_tier=action_tier,
                frozen_policy=frozen_policy,
            )
            wrapped = EncoderOnlyObsAssertWrapper(env)
            boxed = BoxActionDictObsEnv(wrapped)
            return CurriculumEnv(boxed, progress, training)

        return _init

    try:
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        if allow_scripted:
            emit("stable-baselines3 not installed; using scripted AUTO baseline")
            frames = record_policy_episode(scripted_auto, bundle=bundle, seed=1)
            metrics = {
                "envSteps": 0,
                "trueScoreMean": float(frames[-1].get("trueScore") or 0) if frames else None,
                "objectiveMean": float(frames[-1].get("trueScore") or 0) if frames else None,
                "shapingMean": 0.0,
                "evalTrueScoreMean": float(frames[-1].get("trueScore") or 0) if frames else None,
                "algo": "scripted",
                "nEnvs": 1,
                "progressFrac": 1.0,
                "curriculumStage": "scripted",
            }
            if on_metrics:
                on_metrics(metrics)
            if on_rollout and frames:
                on_rollout(frames)
            return {"algo": "scripted", "frames": frames, "steps": 0, "metrics": metrics}
        raise RuntimeError(
            "RecurrentPPO requires pip install -e '.[rl]' (torch, stable-baselines3, sb3-contrib)"
        ) from exc

    venv = DummyVecEnv([make_env() for _ in range(n_envs)])
    n_steps = int(algo_cfg.get("nSteps") or 128)
    rollout_len = n_steps * n_envs
    if total_steps < rollout_len:
        n_envs = 1
        venv.close()
        venv = DummyVecEnv([make_env()])
        n_steps = min(n_steps, max(16, total_steps))
        rollout_len = n_steps * n_envs
    batch = int(algo_cfg.get("batchSize") or 256)
    batch = min(batch, rollout_len)
    while batch > 1 and rollout_len % batch != 0:
        batch -= 1
    lstm_hidden = int(algo_cfg.get("lstmHiddenSize") or 256)
    policy_kwargs = {"lstm_hidden_size": lstm_hidden}

    save_dir = Path(save_dir) if save_dir is not None else (paths.VAR_DIR / "ckpts")
    save_dir.mkdir(parents=True, exist_ok=True)
    latest = save_dir / "latest.zip"
    best_path = save_dir / "best.zip"

    loaded = False
    if resume and latest.exists():
        try:
            model = RecurrentPPO.load(str(latest), env=venv)
            loaded = True
            emit(f"Resumed RecurrentPPO from {latest}")
        except Exception as exc:
            emit(f"Resume failed ({exc}); starting fresh")

    if not loaded:
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            venv,
            verbose=0,
            n_steps=n_steps,
            batch_size=batch,
            learning_rate=float(algo_cfg.get("learningRate") or 3e-4),
            gamma=float(algo_cfg.get("gamma") or 0.99),
            gae_lambda=float(algo_cfg.get("gaeLambda") or 0.95),
            clip_range=float(algo_cfg.get("clipRange") or 0.2),
            n_epochs=int(algo_cfg.get("nEpochs") or 10),
            ent_coef=float(algo_cfg.get("entCoef") or 0.01),
            policy_kwargs=policy_kwargs,
        )
    emit("Using sb3-contrib RecurrentPPO (MultiInputLstmPolicy)")

    true_hist: list[float] = []
    obj_hist: list[float] = []
    shape_hist: list[float] = []
    best_metric = float("-inf")
    last_improve_at = 0
    last_metrics: dict[str, Any] = {}

    class MetricsCb(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            dones = self.locals.get("dones")
            rewards = self.locals.get("rewards")
            if rewards is not None:
                obj_hist.extend(float(r) for r in np.asarray(rewards).ravel())
            for i, info in enumerate(infos):
                if not info:
                    continue
                if "shaping" in info:
                    shape_hist.append(float(info["shaping"]))
                done = bool(dones[i]) if dones is not None and i < len(dones) else False
                if done and "true_score" in info:
                    true_hist.append(float(info["true_score"]))
            if should_stop and should_stop():
                return False
            return True

    chunk = max(rollout_len, min(total_steps, 8192))
    chunk = max(rollout_len, (chunk // rollout_len) * rollout_len)
    done = int(getattr(model, "num_timesteps", 0) or 0) if loaded else 0
    progress.steps = done
    cancelled = False

    def _logger_metrics() -> dict[str, Any]:
        lv = getattr(model.logger, "name_to_value", {}) or {}
        return {
            "entropy": _finite(lv.get("train/entropy_loss")),
            "approxKl": _finite(lv.get("train/approx_kl")),
            "explainedVariance": _finite(lv.get("train/explained_variance")),
            "clipFraction": _finite(lv.get("train/clip_fraction")),
            "fps": _finite(lv.get("time/fps")),
        }

    while done < total_steps:
        if should_stop and should_stop():
            cancelled = True
            break
        if wall_limit and (time.monotonic() - t0) >= float(wall_limit):
            emit("wall-clock limit reached")
            break
        step = min(chunk, total_steps - done)
        step = max(rollout_len, (step // rollout_len) * rollout_len) if step >= rollout_len else step
        if step <= 0:
            break
        model.learn(total_timesteps=step, reset_num_timesteps=done == 0, callback=MetricsCb())
        done = int(model.num_timesteps)
        progress.steps = done
        stage = stage_info(training, progress.frac)
        metrics: dict[str, Any] = {
            "envSteps": done,
            "nEnvs": n_envs,
            "objectiveMean": float(np.mean(obj_hist[-200:])) if obj_hist else None,
            "trueScoreMean": float(np.mean(true_hist[-32:])) if true_hist else None,
            "shapingMean": float(np.mean(shape_hist[-200:])) if shape_hist else None,
            "algo": "recurrent_ppo",
            "progressFrac": progress.frac,
            "curriculumStage": stage.get("index"),
            "curriculumUnlock": stage.get("unlock"),
            **_logger_metrics(),
        }
        eval_frames: list[dict[str, Any]] = []
        try:
            model.save(str(latest))
            if eval_n > 0:
                adapter = RecurrentPolicyAdapter(model)
                seeds = [held0 + i + done for i in range(eval_n)]
                scores, eval_frames = _eval_true_scores(adapter, bundle, seeds, record_first=True)
                report = bootstrap_ci(scores, n_boot=min(400, max(40, 20 * len(scores))))
                metrics["evalTrueScoreMean"] = report["mean"]
                metrics["evalTrueScoreLo"] = report["lo"]
                metrics["evalTrueScoreHi"] = report["hi"]
                metrics["evalP10"] = report["p10"]
                metric = objective_value(report, objective_name)
                metrics["bestMetric"] = metric
                if metric >= best_metric:
                    best_metric = metric
                    last_improve_at = done
                    model.save(str(best_path))
                    metrics["bestCheckpoint"] = str(best_path)
            else:
                last_improve_at = done
                model.save(str(best_path))
        except Exception as exc:
            emit(f"eval-in-loop failed: {exc}")
        last_metrics = metrics
        if on_metrics:
            on_metrics(metrics)
        if on_rollout and done < total_steps:
            frames_out = eval_frames
            if not frames_out:
                try:
                    frames_out = record_policy_episode(RecurrentPolicyAdapter(model), bundle=bundle, seed=int(held0 + done))
                except Exception:
                    frames_out = []
            if frames_out:
                on_rollout(frames_out)
        emit(
            f"trained {done}/{total_steps} steps (recurrent_ppo) "
            f"true={metrics.get('trueScoreMean')} eval={metrics.get('evalTrueScoreMean')}"
        )
        if early_stop and (done - last_improve_at) >= int(early_stop) and last_improve_at > 0:
            emit("early stop: no eval improvement")
            break
        if should_stop and should_stop():
            cancelled = True
            break

    ckpt = best_path if best_path.exists() else latest
    if not ckpt.exists():
        model.save(str(latest))
        ckpt = latest

    if cancelled:
        venv.close()
        return {
            "algo": "recurrent_ppo",
            "frames": [],
            "steps": done,
            "cancelled": True,
            "checkpoint": str(ckpt) if ckpt.exists() else None,
            "metrics": last_metrics,
        }

    adapter = RecurrentPolicyAdapter(model)
    try:
        adapter.model = RecurrentPPO.load(str(ckpt))
    except Exception:
        pass
    frames = record_policy_episode(adapter, bundle=bundle, seed=7)
    venv.close()
    return {
        "algo": "recurrent_ppo",
        "frames": frames,
        "steps": done,
        "model": model,
        "checkpoint": str(ckpt),
        "metrics": last_metrics,
    }


def _finite(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


class RecurrentPolicyAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)
        obs_space = getattr(model, "observation_space", None)
        self._dict_obs = isinstance(obs_space, spaces.Dict)

    def reset_lstm(self) -> None:
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def __call__(self, obs, info=None):
        if self._dict_obs and isinstance(obs, dict):
            inp = obs
        else:
            inp = flatten_obs(obs) if isinstance(obs, dict) else np.asarray(obs, dtype=np.float32)
        action, self.state = self.model.predict(
            inp,
            state=self.state,
            episode_start=self.episode_start,
            deterministic=True,
        )
        self.episode_start = np.zeros((1,), dtype=bool)
        return np.asarray(action, dtype=np.float32)


def load_trained_policy(path: str | Path) -> RecurrentPolicyAdapter:
    from sb3_contrib import RecurrentPPO

    model = RecurrentPPO.load(str(path))
    return RecurrentPolicyAdapter(model)
