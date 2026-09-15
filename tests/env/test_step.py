import pytest
import numpy as np

from talongym.env.ftc_auto import FTCAutoEnv
from talongym.training.policies import scripted_auto


def test_reset_step_truncates_at_auto():
    env = FTCAutoEnv(record=False, static_teammate=False)
    obs, info = env.reset(seed=0)
    assert "true_score" in info
    steps = 0
    truncated = False
    while steps < 900:
        pose = obs["pose_noisy"]
        action = {
            "target_pose": np.array([pose[0], pose[1] + 8.0, pose[2]], dtype=np.float32),
            "speed_frac": np.array([1.0], dtype=np.float32),
            "mechanism": 0,
        }
        obs, rew, term, truncated, info = env.step(action)
        steps += 1
        if truncated:
            break
    assert truncated
    assert steps == 25 * 30
    assert info["true_score"] >= 0
    env.close()


def test_scripted_episode_runs():
    env = FTCAutoEnv(record=True)
    obs, info = env.reset(seed=2)
    term = trunc = False
    while not term and not trunc:
        obs, _, term, trunc, info = env.step(scripted_auto(obs, info))
    assert len(env.frames) > 10
    env.close()


@pytest.mark.require_mesh
def test_scripted_auto_scores_leave():
    env = FTCAutoEnv(record=False)
    obs, info = env.reset(seed=4)
    term = trunc = False
    while not term and not trunc:
        obs, _, term, trunc, info = env.step(scripted_auto(obs, info))
    assert info["true_score"] >= 3
    env.close()
