"""Distill a feed-forward waypoint clone and export ONNX (demo, not LSTM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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
        torch.onnx.export(model, dummy, str(out_path), input_names=["obs"], output_names=["target_pose"], opset_version=17)
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
