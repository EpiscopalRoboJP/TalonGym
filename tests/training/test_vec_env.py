import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from talongym.training.vec_env import (
    ChunkedSubprocVecEnv,
    make_train_vec_env,
    slim_info,
    split_chunk_sizes,
)


def test_split_chunk_sizes_spreads_remainder():
    assert split_chunk_sizes(1, 8) == [1]
    assert split_chunk_sizes(10, 4) == [3, 3, 2, 2]
    assert split_chunk_sizes(8, 8) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert split_chunk_sizes(5, 2) == [3, 2]


class TinyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-10.0, 10.0, (2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.array([self.t, float(np.asarray(action).reshape(-1)[0])], dtype=np.float32)
        terminated = False
        truncated = self.t >= 4
        return obs, 1.0, terminated, truncated, {}


def _tiny_env() -> gym.Env:
    return TinyEnv()


def test_make_train_vec_env_workers_one_is_dummy(monkeypatch):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    monkeypatch.setenv("TALONGYM_SIM_WORKERS", "1")
    spec = {
        "field": {"id": "x"},
        "robot": {"id": "y"},
        "scoring": {"id": "z"},
        "training": None,
        "progress": type("P", (), {"frac": 0.0})(),
    }

    def _boom(*_a, **_k):
        raise AssertionError("must not start subprocesses when workers==1")

    monkeypatch.setattr("talongym.training.vec_env.worker_env_from_spec", lambda spec: _tiny_env())
    venv, workers = make_train_vec_env(4, spec, n_workers=1)
    try:
        assert workers == 1
        assert isinstance(venv, DummyVecEnv)
        assert venv.num_envs == 4
        obs = venv.reset()
        assert obs.shape == (4, 2)
    finally:
        venv.close()


def test_chunked_subproc_vec_env_reset_and_step():
    pytest.importorskip("stable_baselines3")
    venv = ChunkedSubprocVecEnv(4, 2, env_fns=[_tiny_env, _tiny_env, _tiny_env, _tiny_env])
    try:
        obs = venv.reset()
        assert obs.shape == (4, 2)
        actions = np.zeros((4, 1), dtype=np.float32)
        obs, rewards, dones, infos = venv.step(actions)
        assert obs.shape == (4, 2)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4
        assert venv.has_attr("t")
        assert venv.get_attr("t") == [1, 1, 1, 1]
    finally:
        venv.close()


class BoomEnv(TinyEnv):
    def step(self, action):
        raise RuntimeError("boom-step")


def test_chunked_worker_forwards_crash():
    pytest.importorskip("stable_baselines3")
    venv = ChunkedSubprocVecEnv(2, 2, env_fns=[BoomEnv, BoomEnv])
    try:
        venv.reset()
        with pytest.raises(RuntimeError, match="sim worker crashed"):
            venv.step(np.zeros((2, 1), dtype=np.float32))
    finally:
        venv.close()


def test_slim_info_keeps_true_score_and_drops_privileged():
    slim = slim_info(
        {
            "true_score": 3.0,
            "shaping": -0.02,
            "privileged": {"parts": [{"mesh": "x" * 1000}]},
            "waypoint_log": [[0.0, 0.0, 0.0]] * 800,
        }
    )
    assert slim == {"true_score": 3.0, "shaping": -0.02}


class FatInfoEnv(TinyEnv):
    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        info["true_score"] = 3.0
        info["privileged"] = {"blob": "x" * 200_000, "parts": [{"m": list(range(200))}]}
        info["waypoint_log"] = [[0.0, 0.0, 0.0]] * 800
        return obs, rew, term, trunc, info


def test_chunked_worker_slims_privileged_info():
    pytest.importorskip("stable_baselines3")
    venv = ChunkedSubprocVecEnv(2, 2, env_fns=[FatInfoEnv, FatInfoEnv])
    try:
        venv.reset()
        _obs, _rewards, _dones, infos = venv.step(np.zeros((2, 1), dtype=np.float32))
        assert len(infos) == 2
        assert infos[0]["true_score"] == 3.0
        assert "privileged" not in infos[0]
        assert "waypoint_log" not in infos[0]
    finally:
        venv.close()
