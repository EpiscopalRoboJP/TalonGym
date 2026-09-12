from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from talongym.env.ftc_auto import EncoderOnlyObsAssertWrapper, FTCAutoEnv, FlatBoxEnv, flatten_obs
from talongym.paths import VAR_DIR
from talongym.presets.loader import LoadedPresets, load_bundle
from talongym.training.policies import scripted_auto


def record_policy_episode(policy, bundle: LoadedPresets | None = None, seed: int = 0) -> list[dict[str, Any]]:
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


def _curriculum_known(progress: _Progress, training: dict[str, Any] | None) -> bool:
    dr = (training or {}).get("domainRandomization") or {}
    stages = dr.get("curriculum") or []
    frac = progress.frac
    unlocks: list[str] = []
    for stage in stages:
        if frac <= float(stage.get("untilFrac", 1.0)):
            unlocks = list(stage.get("unlock") or [])
            break
    else:
        if stages:
            unlocks = list(stages[-1].get("unlock") or [])
    if "motif_must_sense" in unlocks:
        return False
    return "motif_known_at_t0" in unlocks


def _true_score_mean(infos: list[dict]) -> float | None:
    scores = [float(i.get("true_score", 0)) for i in infos if i]
    if not scores:
        return None
    return float(np.mean(scores))


def train_ppo(
    bundle: LoadedPresets | None = None,
    total_steps: int = 50_000,
    n_envs: int = 8,
    log: Callable[[str], None] | None = None,
    on_metrics: Callable[[dict[str, Any]], None] | None = None,
    allow_scripted: bool = False,
    save_dir: Path | None = None,
    frozen_policy: Any = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Train RecurrentPPO. Raises if RL extras are missing unless allow_scripted."""
    emit = log or (lambda m: None)
    bundle = bundle or load_bundle()
    progress = _Progress(total_steps)
    training = bundle.training
    algo_cfg = (training or {}).get("algorithm") or {}
    presets_cfg = (training or {}).get("presets") or {}
    action_tier = str((training or {}).get("actionTier") or "high_level_waypoint")
    teammate_pol = str(presets_cfg.get("teammatePolicy") or "none")
    opponent_pol = str(presets_cfg.get("opponentPolicy") or "none")

    def make_env():
        def _init():
            unlocks = []
            dr = (training or {}).get("domainRandomization") or {}
            for stage in dr.get("curriculum") or []:
                if progress.frac <= float(stage.get("untilFrac", 1.0)):
                    unlocks = list(stage.get("unlock") or [])
                    break
            teammate = "scripted" if "scripted_teammate" in unlocks else teammate_pol
            opponent = "scripted" if "scripted_opponent" in unlocks else opponent_pol
            env = FTCAutoEnv(
                bundle=bundle,
                record=False,
                motif_known_at_t0=_curriculum_known(progress, training),
                teammate_policy=teammate,
                opponent_policy=opponent,
                action_tier=action_tier,
                frozen_policy=frozen_policy,
            )
            wrapped = EncoderOnlyObsAssertWrapper(env)

            class _Curric(FlatBoxEnv):
                def reset(self, **kwargs):
                    opts = dict(kwargs.get("options") or {})
                    opts["motif_known_at_t0"] = _curriculum_known(progress, training)
                    kwargs["options"] = opts
                    return super().reset(**kwargs)

            return _Curric(wrapped)

        return _init

    try:
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        if allow_scripted:
            emit("stable-baselines3 not installed; using scripted AUTO baseline")
            frames = record_policy_episode(scripted_auto, bundle=bundle, seed=1)
            if on_metrics:
                true = float(frames[-1].get("trueScore") or 0) if frames else None
                on_metrics({"envSteps": 0, "trueScoreMean": true, "objectiveMean": true, "shapingMean": 0.0, "algo": "scripted"})
            return {"algo": "scripted", "frames": frames, "steps": 0}
        raise RuntimeError(
            "RecurrentPPO requires pip install -e '.[rl]' (torch, stable-baselines3, sb3-contrib)"
        ) from exc

    venv = DummyVecEnv([make_env() for _ in range(max(1, n_envs))])
    n_steps = int(algo_cfg.get("nSteps") or 128)
    batch = int(algo_cfg.get("batchSize") or 256)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        verbose=0,
        n_steps=n_steps,
        batch_size=min(batch, n_steps * max(1, n_envs)),
        learning_rate=float(algo_cfg.get("learningRate") or 3e-4),
        gamma=float(algo_cfg.get("gamma") or 0.99),
        gae_lambda=float(algo_cfg.get("gaeLambda") or 0.95),
        clip_range=float(algo_cfg.get("clipRange") or 0.2),
        n_epochs=int(algo_cfg.get("nEpochs") or 10),
        ent_coef=float(algo_cfg.get("entCoef") or 0.01),
    )
    emit("Using sb3-contrib RecurrentPPO")

    true_hist: list[float] = []
    obj_hist: list[float] = []
    shape_hist: list[float] = []

    class MetricsCb(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            for info in infos:
                if not info:
                    continue
                if "true_score" in info:
                    true_hist.append(float(info["true_score"]))
                if "shaping" in info:
                    shape_hist.append(float(info["shaping"]))
            rewards = self.locals.get("rewards")
            if rewards is not None:
                obj_hist.extend(float(r) for r in np.asarray(rewards).ravel())
            if should_stop and should_stop():
                return False
            return True

    chunk = max(2048, min(total_steps, 8192))
    done = 0
    while done < total_steps:
        if should_stop and should_stop():
            return {"algo": "recurrent_ppo", "frames": [], "steps": done, "cancelled": True}
        step = min(chunk, total_steps - done)
        model.learn(total_timesteps=step, reset_num_timesteps=done == 0, callback=MetricsCb())
        done += step
        progress.steps = done
        metrics = {
            "envSteps": done,
            "objectiveMean": float(np.mean(obj_hist[-200:])) if obj_hist else None,
            "trueScoreMean": float(np.mean(true_hist[-200:])) if true_hist else None,
            "shapingMean": float(np.mean(shape_hist[-200:])) if shape_hist else None,
            "entropy": None,
            "approxKl": None,
            "algo": "recurrent_ppo",
        }
        if on_metrics:
            on_metrics(metrics)
        emit(f"trained {done}/{total_steps} steps (recurrent_ppo) true={metrics['trueScoreMean']}")

    save_dir = save_dir or (VAR_DIR / "ckpts")
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_dir / "recurrent_ppo.zip"
    model.save(str(ckpt))

    adapter = RecurrentPolicyAdapter(model)
    frames = record_policy_episode(adapter, bundle=bundle, seed=7)
    venv.close()
    return {
        "algo": "recurrent_ppo",
        "frames": frames,
        "steps": done,
        "model": model,
        "checkpoint": str(ckpt),
    }


class RecurrentPolicyAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def __call__(self, obs, info=None):
        flat = flatten_obs(obs) if isinstance(obs, dict) else np.asarray(obs, dtype=np.float32)
        action, self.state = self.model.predict(
            flat,
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
