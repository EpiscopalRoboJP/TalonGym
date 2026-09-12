"""Privileged critic vector (never on the actor observation)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

PRIV_DIM = 64
PRIV_KEY = "_privileged"


def privileged_vector(info: dict[str, Any] | None) -> np.ndarray:
    vec = np.zeros(PRIV_DIM, dtype=np.float32)
    priv = (info or {}).get("privileged") or {}
    pose = list(priv.get("pose3") or priv.get("pose") or [0, 0, 0])
    for i, v in enumerate(pose[:4]):
        vec[i] = float(v)
    vec[4] = float(len(priv.get("held") or []))
    vec[5] = float(priv.get("timeS") or 0.0)
    vec[6] = float(priv.get("trueScore") or 0.0)
    vec[7] = float(priv.get("cellLoad") or 0.0)
    pieces = list(priv.get("pieces") or [])
    for i, row in enumerate(pieces[:12]):
        base = 8 + i * 4
        for j, val in enumerate(list(row)[:4]):
            vec[base + j] = float(val)
    return vec


class PrivilegedObsWrapper(gym.Wrapper):
    """Attach a critic-only vector. Actor extractors must drop this key."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base = env.observation_space
        if not isinstance(base, spaces.Dict):
            raise TypeError("PrivilegedObsWrapper requires Dict observations")
        self.observation_space = spaces.Dict(
            {
                **base.spaces,
                PRIV_KEY: spaces.Box(-1e4, 1e4, shape=(PRIV_DIM,), dtype=np.float32),
            }
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = dict(obs)
        obs[PRIV_KEY] = privileged_vector(info)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        obs = dict(obs)
        obs[PRIV_KEY] = privileged_vector(info)
        return obs, rew, term, trunc, info
