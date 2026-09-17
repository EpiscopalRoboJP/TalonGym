"""Distill a feed-forward waypoint clone and export ONNX (demo, not LSTM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from talongym.env.ftc_auto import FTCAutoEnv, flatten_obs
from talongym.paths import VAR_DIR
from talongym.presets.loader import LoadedPresets, load_bundle
from talongym.training.policies import scripted_auto


def collect_pairs(n_steps: int = 256, bundle: LoadedPresets | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    env = FTCAutoEnv(bundle=bundle or load_bundle(), record=False)
    obs, info = env.reset(seed=seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    term = trunc = False
    steps = 0
    while steps < n_steps and not term and not trunc:
        act = scripted_auto(obs, info)
        xs.append(flatten_obs(obs))
        ys.append(np.asarray(act["target_pose"], dtype=np.float32).reshape(3))
        obs, _, term, trunc, info = env.step(act)
        steps += 1
        if term or trunc:
            obs, info = env.reset(seed=seed + steps)
            term = trunc = False
    env.close()
    return np.stack(xs), np.stack(ys)


def _box_action(act: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(act.get("target_pose", [0, 0, 0]), dtype=np.float32).reshape(3)
    speed = float(np.asarray(act.get("speed_frac", 0.8)).reshape(-1)[0])
    mech = float(act.get("mechanism", 0))
    return np.concatenate([pose, np.array([speed, mech], dtype=np.float32)])


DEMO_PACE_RANGE = (0.35, 1.0)
DEMO_MAX_PAUSE_S = 3.0
DEMO_TARGET_NOISE = (3.0, 3.0, 0.1)
DEMO_NOISE_HOLD_S = 0.5


def collect_episodes(
    n_episodes: int,
    bundle: LoadedPresets | None = None,
    seed: int = 0,
    gamma: float = 0.99,
    options: dict[str, Any] | None = None,
    perturb: bool = True,
    episode_options: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray, int]:
    """Whole scripted AUTO episodes, concatenated episode by episode and trimmed to one common length."""
    from talongym.training.privileged import PRIV_KEY, privileged_vector

    env = FTCAutoEnv(bundle=bundle or load_bundle(), record=False)
    rng = np.random.default_rng(seed)
    hold_steps = max(1, round(DEMO_NOISE_HOLD_S * env.control_hz))
    episodes: list[tuple[list[dict[str, np.ndarray]], list[np.ndarray], list[float]]] = []
    for ep in range(max(1, n_episodes)):
        opts = episode_options[ep] if episode_options and ep < len(episode_options) else options
        obs, info = env.reset(seed=seed + ep, options=dict(opts or {}))
        varied = perturb and ep > 0
        pace = float(rng.uniform(*DEMO_PACE_RANGE)) if varied else 1.0
        pause_steps = round(float(rng.uniform(0.0, DEMO_MAX_PAUSE_S)) * env.control_hz) if varied else 0
        noise = np.zeros(3, dtype=np.float32)
        xs: list[dict[str, np.ndarray]] = []
        ys: list[np.ndarray] = []
        rewards: list[float] = []
        term = trunc = False
        step = 0
        while not term and not trunc:
            act = scripted_auto(obs, info)
            row = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
            row[PRIV_KEY] = privileged_vector(info)
            xs.append(row)
            ys.append(_box_action(act))
            executed = dict(act)
            if varied:
                if step % hold_steps == 0:
                    noise = (rng.normal(0.0, 1.0, 3) * np.asarray(DEMO_TARGET_NOISE)).astype(np.float32)
                if step < pause_steps:
                    body = env.world.actor().body
                    executed = {"target_pose": np.array([body.x, body.y, body.heading], dtype=np.float32), "mechanism": 0}
                else:
                    executed["target_pose"] = np.asarray(act["target_pose"], dtype=np.float32) + noise
                executed["speed_frac"] = np.array([pace * float(np.asarray(act["speed_frac"]).reshape(-1)[0])], dtype=np.float32)
            obs, reward, term, trunc, info = env.step(executed)
            rewards.append(float(reward))
            step += 1
        episodes.append((xs, ys, rewards))
    env.close()
    length = min(len(xs) for xs, _, _ in episodes)
    obs_rows: list[dict[str, np.ndarray]] = []
    acts: list[np.ndarray] = []
    returns: list[float] = []
    for xs, ys, rewards in episodes:
        obs_rows.extend(xs[:length])
        acts.extend(ys[:length])
        ret = 0.0
        ep_returns = []
        for r in reversed(rewards[:length]):
            ret = r + gamma * ret
            ep_returns.append(ret)
        returns.extend(reversed(ep_returns))
    return obs_rows, np.stack(acts), np.asarray(returns, dtype=np.float32), length


def collect_full_actions(
    n_steps: int = 512,
    bundle: LoadedPresets | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """Clone scripted policy into (dict obs, box action) pairs for LSTM BC."""
    env = FTCAutoEnv(bundle=bundle or load_bundle(), record=False)
    obs, info = env.reset(seed=seed)
    xs: list[dict[str, np.ndarray]] = []
    ys: list[np.ndarray] = []
    term = trunc = False
    steps = 0
    while steps < n_steps:
        act = scripted_auto(obs, info)
        xs.append({k: np.asarray(v, dtype=np.float32) for k, v in obs.items()})
        ys.append(_box_action(act))
        obs, _, term, trunc, info = env.step(act)
        steps += 1
        if term or trunc:
            obs, info = env.reset(seed=seed + steps)
            term = trunc = False
    env.close()
    return xs, np.stack(ys)


BC_ACTION_STD = (4.0, 4.0, 0.2, 0.1, 0.3)
BC_LR = 3e-3
BC_TARGET_LOSS = 0.0002


def bc_warmup(model: Any, bundle: LoadedPresets | None, n_steps: int, log=lambda m: None, epochs: int = 2500) -> float:
    """Clone scripted AUTO into a RecurrentPPO actor through its LSTM, and fit the critic to returns."""
    if n_steps <= 0:
        return 0.0
    try:
        import torch
        from stable_baselines3.common.utils import obs_as_tensor
    except ImportError as exc:
        raise RuntimeError("BC warmup needs torch + stable-baselines3") from exc

    policy = model.policy
    bundle = bundle or load_bundle()
    ep_cfg = (bundle.training or {}).get("episode") or {}
    steps_per_episode = int(float(ep_cfg.get("durationS") or 30) * float(ep_cfg.get("controlHz") or 25))
    n_episodes = max(2, -(-int(n_steps) // steps_per_episode))
    from talongym.training.curriculum import curriculum_spawn, full_noise, mechanism_ready

    stage0 = {
        "full_noise": False,
        "curriculum_spawn": curriculum_spawn(bundle.training, 0.0),
        "mechanism_ready": mechanism_ready(bundle.training, 0.0),
        "static_teammate": False,
    }
    legal = {
        "full_noise": False,
        "curriculum_spawn": "legal",
        "mechanism_ready": False,
        "static_teammate": False,
    }
    # Held-out eval is the full AUTO from a legal spawn. Launch-pose-only demos clone a
    # fire-in-place policy that parks for LEAVE without launching.
    episode_options: list[dict[str, Any]] = []
    for i in range(n_episodes):
        if i == 0:
            episode_options.append(legal)
        elif i == 1:
            episode_options.append(stage0)
        elif i == 2:
            episode_options.append({**legal, "full_noise": bool(full_noise(bundle.training, 1.0))})
        else:
            episode_options.append(legal if i % 2 == 0 else stage0)
    obs_list, acts, returns, length = collect_episodes(
        n_episodes,
        bundle=bundle,
        gamma=float(getattr(model, "gamma", 0.99)),
        episode_options=episode_options,
    )
    space = getattr(model, "observation_space", None)
    obs_space = space if isinstance(space, spaces.Dict) else None
    keys = list(obs_space.spaces.keys()) if obs_space is not None else sorted(obs_list[0])
    batch: dict[str, np.ndarray] = {}
    n = len(obs_list)
    for key in keys:
        sample = obs_list[0].get(key)
        if sample is None:
            shp: tuple[int, ...] = (obs_space.spaces[key].shape or (1,)) if obs_space is not None else (1,)
            batch[key] = np.zeros((n, *shp), dtype=np.float32)
            continue
        batch[key] = np.stack([np.asarray(row.get(key, np.zeros_like(sample)), dtype=np.float32) for row in obs_list])
    device = policy.device
    obs_t = obs_as_tensor(batch, device)
    act_t = torch.as_tensor(acts, device=device)
    action_space = model.action_space
    half_range = np.maximum((action_space.high - action_space.low) / 2.0, 1e-3).astype(np.float32)
    scale = torch.as_tensor(half_range, device=device)
    lstm = policy.lstm_actor
    state_shape = (lstm.num_layers, n_episodes, lstm.hidden_size)
    lstm_states = (torch.zeros(state_shape, device=device), torch.zeros(state_shape, device=device))
    episode_starts = torch.zeros(n, device=device)
    torch.nn.init.orthogonal_(policy.action_net.weight, gain=1.0)
    torch.nn.init.zeros_(policy.action_net.bias)
    opt = torch.optim.Adam(policy.parameters(), lr=BC_LR)
    last = 0.0
    epochs_run = 0
    for _ in range(max(1, epochs)):
        epochs_run += 1
        dist, _ = policy.get_distribution(obs_t, lstm_states, episode_starts)
        mean = dist.distribution.mean.reshape(act_t.shape)
        loss = (((mean - act_t) / scale) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
        if last < BC_TARGET_LOSS:
            break
    if hasattr(policy, "log_std"):
        with torch.no_grad():
            std = torch.as_tensor(BC_ACTION_STD[: policy.log_std.shape[0]], device=device)
            policy.log_std.copy_(torch.log(std))
    value_loss = _fit_critic(policy, obs_t, torch.as_tensor(returns, device=device).reshape(-1, 1), n_episodes, episode_starts)
    log(f"BC warmup mse={last:.4f} epochs={epochs_run} episodes={n_episodes} steps={n} value_loss={value_loss:.4f}")
    return last


CRITIC_EPOCHS = 400


def _fit_critic(policy: Any, obs_t: Any, returns_t: Any, n_seq: int, episode_starts: Any) -> float:
    import torch

    lstm = policy.lstm_critic
    if lstm is None:
        return float("nan")
    state_shape = (lstm.num_layers, n_seq, lstm.hidden_size)
    states = (torch.zeros(state_shape, device=returns_t.device), torch.zeros(state_shape, device=returns_t.device))
    var = float(returns_t.var().clamp_min(1e-6))
    critic_params = [p for name, p in policy.named_parameters() if "vf" in name or "critic" in name or "value" in name]
    opt = torch.optim.Adam(critic_params, lr=BC_LR)
    rel = float("nan")
    for _ in range(CRITIC_EPOCHS):
        values = policy.predict_values(obs_t, states, episode_starts)
        loss = ((values.reshape(returns_t.shape) - returns_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rel = float(loss.detach().cpu()) / var
        if rel < 0.01:
            break
    return rel


def distill_mlp(x: np.ndarray, y: np.ndarray, out_path: Path | None = None) -> Path:
    out_path = out_path or (VAR_DIR / "ckpts" / "ff_distill.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from torch import nn

        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        model = nn.Sequential(nn.Linear(xt.shape[1], 64), nn.Tanh(), nn.Linear(64, yt.shape[1]))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(80):
            pred = model(xt)
            loss = ((pred - yt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        dummy = torch.zeros(1, xt.shape[1])
        torch.onnx.export(model, (dummy,), str(out_path), input_names=["obs"], output_names=["target_pose"], opset_version=17)
        return out_path
    except Exception:
        xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        w, *_ = np.linalg.lstsq(xb, y, rcond=None)
        npz = out_path.with_suffix(".npz")
        np.savez(npz, weights=w)
        return npz


def distill_from_scripted(out_path: Path | None = None, n_steps: int = 256) -> dict[str, Any]:
    x, y = collect_pairs(n_steps=n_steps)
    path = distill_mlp(x, y, out_path)
    return {"path": str(path), "n": int(x.shape[0]), "kind": "feedforward_demo"}
