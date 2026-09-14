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
    extra = _log_snippets(n_max=max(0, n_steps // 8))
    if extra:
        xs.extend(extra[0])
        ys.extend(extra[1])
    return xs, np.stack(ys)


def _log_snippets(n_max: int) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]] | None:
    """Optional RL-Co mix-in from var/calibrate_logs/*.json (not required)."""
    if n_max <= 0:
        return None
    root = VAR_DIR / "calibrate_logs"
    if not root.is_dir():
        return None
    xs: list[dict[str, np.ndarray]] = []
    ys: list[np.ndarray] = []
    for path in sorted(root.glob("*.json"))[:4]:
        try:
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        samples = raw if isinstance(raw, list) else raw.get("samples") or []
        for row in samples:
            if len(ys) >= n_max:
                break
            pose = np.array(
                [float(row.get("x", 0)), float(row.get("y", 0)), float(np.deg2rad(float(row.get("headingDeg", 0))))],
                dtype=np.float32,
            )
            obs = {"pose_noisy": pose}
            act = _box_action(
                {
                    "target_pose": pose,
                    "speed_frac": row.get("speed_frac", 0.8),
                    "mechanism": row.get("mechanism", 0),
                }
            )
            xs.append(obs)
            ys.append(act)
    if not ys:
        return None
    return xs, ys


def bc_warmup(model: Any, bundle: LoadedPresets | None, n_steps: int, log=lambda m: None) -> float:
    """Supervised clone of scripted AUTO into a RecurrentPPO actor (encoder keys only)."""
    if n_steps <= 0:
        return 0.0
    try:
        import torch
        from stable_baselines3.common.utils import obs_as_tensor
    except ImportError as exc:
        raise RuntimeError("BC warmup needs torch + stable-baselines3") from exc

    from talongym.training.privileged import PRIV_DIM, PRIV_KEY

    obs_list, acts = collect_full_actions(n_steps=n_steps, bundle=bundle)
    space = getattr(model, "observation_space", None)
    obs_space = space if isinstance(space, spaces.Dict) else None
    keys = list(obs_space.spaces.keys()) if obs_space is not None else sorted(obs_list[0])
    batch: dict[str, np.ndarray] = {}
    n = len(obs_list)
    for key in keys:
        if key == PRIV_KEY:
            batch[key] = np.zeros((n, PRIV_DIM), dtype=np.float32)
            continue
        sample = obs_list[0].get(key)
        if sample is None:
            shp: tuple[int, ...] = (obs_space.spaces[key].shape or (1,)) if obs_space is not None else (1,)
            batch[key] = np.zeros((n, *shp), dtype=np.float32)
            continue
        batch[key] = np.stack(
            [np.asarray(row.get(key, np.zeros_like(sample)), dtype=np.float32) for row in obs_list]
        )
    device = model.policy.device
    obs_t = obs_as_tensor(batch, device)
    act_t = torch.as_tensor(acts, device=device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    last = 0.0
    epochs = 8
    pi_ext = getattr(model.policy, "pi_features_extractor", None) or getattr(model.policy, "features_extractor", None)
    for _ in range(epochs):
        feat = model.policy.extract_features(obs_t, pi_ext) if pi_ext is not None else model.policy.extract_features(obs_t)
        if isinstance(feat, tuple):
            feat = feat[0]
        latent_pi, _latent_vf = model.policy.mlp_extractor(feat)
        mean = model.policy.action_net(latent_pi)
        if mean.shape != act_t.shape:
            mean = mean.reshape(act_t.shape[0], -1)[:, : act_t.shape[1]]
        loss = ((mean - act_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
    log(f"BC warmup mse={last:.4f} n={n}")
    return last


def distill_mlp(x: np.ndarray, y: np.ndarray, out_path: Path | None = None) -> Path:
    """Train a tiny MLP (torch if present) or a linear map; write ONNX or .npz."""
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
