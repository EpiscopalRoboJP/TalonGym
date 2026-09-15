"""Experimental GRPO loop. No shipped preset; not a product training path."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from talongym import paths
from talongym.env.ftc_auto import FTCAutoEnv
from talongym.eval.harness import bootstrap_ci
from talongym.presets.loader import LoadedPresets, load_bundle
from talongym.training.curriculum import objective_value, resolved_action_tier
from talongym.training.ppo import RecurrentPolicyAdapter, record_policy_episode


def _episode(env: FTCAutoEnv, policy, seed: int) -> tuple[float, list[dict], list[np.ndarray], list]:
    if hasattr(policy, "reset_lstm"):
        policy.reset_lstm()
    obs, info = env.reset(seed=seed)
    traj_obs = []
    traj_act = []
    term = trunc = False
    while not term and not trunc:
        act = policy(obs, info)
        traj_obs.append({k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()})
        traj_act.append(np.asarray(act, dtype=np.float32).copy())
        obs, _, term, trunc, info = env.step(act)
    return float(info.get("true_score") or 0.0), traj_obs, traj_act, info


def train_grpo(
    bundle: LoadedPresets | None = None,
    total_steps: int = 8_000,
    group_size: int = 4,
    log: Callable[[str], None] | None = None,
    on_metrics: Callable[[dict[str, Any]], None] | None = None,
    on_rollout: Callable[[list[dict[str, Any]]], None] | None = None,
    save_dir: Path | None = None,
    should_stop: Callable[[], bool] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """K rollouts per seed; advantage = true_score - group mean. No value net."""
    emit = log or (lambda m: None)
    bundle = bundle or load_bundle()
    algo_cfg = (bundle.training or {}).get("algorithm") or {}
    group_size = max(2, int(algo_cfg.get("groupSize") or group_size))
    eval_cfg = (bundle.training or {}).get("evaluation") or {}
    held0 = int(eval_cfg.get("heldOutSeedStart") or 10_000_000)
    action_tier = resolved_action_tier(bundle.training, bundle.robot)
    match_setup = _kwargs.get("match_setup")

    try:
        import torch
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from talongym.env.ftc_auto import BoxActionDictObsEnv, EncoderOnlyObsAssertWrapper
        from talongym.training.asymmetric import AsymmetricLstmPolicy
        from talongym.training.privileged import PrivilegedObsWrapper
    except ImportError as exc:
        raise RuntimeError(
            "GRPO requires `.venv/bin/python -m pip install -e '.[rl]'` "
            f"in {sys.executable}: {exc}"
        ) from exc

    def make_env():
        def _init():
            env = FTCAutoEnv(
                bundle=bundle,
                record=False,
                action_tier=action_tier,
                match_setup=match_setup,
            )
            return BoxActionDictObsEnv(PrivilegedObsWrapper(EncoderOnlyObsAssertWrapper(env)))

        return _init

    venv = DummyVecEnv([make_env()])
    n_steps = int(algo_cfg.get("nSteps") or 128)
    model = RecurrentPPO(
        AsymmetricLstmPolicy,
        venv,
        verbose=0,
        n_steps=n_steps,
        batch_size=min(256, n_steps),
        learning_rate=float(algo_cfg.get("learningRate") or 3e-4),
        gamma=float(algo_cfg.get("gamma") or 0.99),
        n_epochs=1,
        ent_coef=float(algo_cfg.get("entCoef") or 0.01),
        policy_kwargs={"lstm_hidden_size": int(algo_cfg.get("lstmHiddenSize") or 256)},
    )
    save_dir = Path(save_dir) if save_dir is not None else (paths.VAR_DIR / "ckpts")
    save_dir.mkdir(parents=True, exist_ok=True)
    latest = save_dir / "latest.zip"
    best_path = save_dir / "best.zip"
    adapter = RecurrentPolicyAdapter(model)
    env = FTCAutoEnv(
        bundle=bundle,
        record=False,
        action_tier=action_tier,
        match_setup=match_setup,
    )
    boxed = BoxActionDictObsEnv(PrivilegedObsWrapper(EncoderOnlyObsAssertWrapper(env)))

    opt = torch.optim.Adam(model.policy.parameters(), lr=float(algo_cfg.get("learningRate") or 3e-4))
    done = 0
    best_metric = float("-inf")
    last_metrics: dict[str, Any] = {}
    seed0 = 1
    from stable_baselines3.common.utils import obs_as_tensor
    from talongym.training.privileged import PRIV_DIM, PRIV_KEY

    while done < total_steps:
        if should_stop and should_stop():
            break
        scores: list[float] = []
        logps: list[torch.Tensor] = []
        for k in range(group_size):
            score, traj_obs, traj_act, _info = _episode(boxed, adapter, seed0)
            seed0 += 1
            done += max(1, len(traj_act))
            scores.append(score)
            if not traj_act:
                logps.append(torch.zeros(1, device=model.policy.device))
                continue
            batch = {}
            keys = list(model.observation_space.spaces.keys())
            for key in keys:
                if key == PRIV_KEY and key not in traj_obs[0]:
                    batch[key] = np.zeros((len(traj_obs), PRIV_DIM), dtype=np.float32)
                    continue
                batch[key] = np.stack([row.get(key, np.zeros(model.observation_space.spaces[key].shape, np.float32)) for row in traj_obs])
            obs_t = obs_as_tensor(batch, model.policy.device)
            acts_t = torch.as_tensor(np.stack(traj_act), device=model.policy.device)
            dist = model.policy.get_distribution(obs_t)
            lp = dist.log_prob(acts_t)
            logps.append(lp.sum())
        mean = float(np.mean(scores))
        adv = [s - mean for s in scores]
        loss = torch.zeros((), device=model.policy.device)
        for a, lp in zip(adv, logps):
            loss = loss + (-float(a) * lp)
        loss = loss / max(1, group_size)
        opt.zero_grad()
        loss.backward()
        opt.step()
        adapter = RecurrentPolicyAdapter(model)
        metrics = {
            "envSteps": done,
            "trueScoreMean": mean,
            "evalTrueScoreMean": mean,
            "algo": "grpo",
            "groupSize": group_size,
        }
        if len(scores) >= 2:
            report = bootstrap_ci(scores, n_boot=40)
            metric = objective_value(report, str((bundle.training or {}).get("objective") or "mean_true_score"))
            metrics["bestMetric"] = metric
            if metric >= best_metric:
                best_metric = metric
                model.save(str(best_path))
        last_metrics = metrics
        if on_metrics:
            on_metrics(metrics)
        emit(f"grpo steps={done} group_mean={mean:.2f}")
        if on_rollout and done < total_steps:
            frames = record_policy_episode(adapter, bundle=bundle, seed=int(held0 + done))
            on_rollout(frames)
        model.save(str(latest))
        if done >= total_steps:
            break

    boxed.close()
    venv.close()
    ckpt = best_path if best_path.exists() else latest
    try:
        adapter.model = RecurrentPPO.load(str(ckpt))
    except Exception:
        pass
    frames = record_policy_episode(adapter, bundle=bundle, seed=7)
    return {"algo": "grpo", "frames": frames, "steps": done, "checkpoint": str(ckpt), "metrics": last_metrics, "model": model}
