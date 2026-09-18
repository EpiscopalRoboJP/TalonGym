import numpy as np
import pytest

from talongym.env.ftc_auto import FTCAutoEnv
from talongym.presets.loader import load_bundle


def test_mid_episode_reward_equals_true_delta_when_only_true_score():
    bundle = load_bundle()
    bundle.training = dict(bundle.training or {})
    bundle.training["reward"] = {"terms": [{"kind": "trueScoreDelta"}]}
    env = FTCAutoEnv(bundle=bundle, record=False, static_teammate=False)
    obs, _info = env.reset(seed=0, options={"full_noise": False, "static_teammate": False})
    pose = obs["pose_noisy"]
    action = {
        "target_pose": np.asarray(pose, dtype=np.float32),
        "speed_frac": np.array([0.2], dtype=np.float32),
        "mechanism": 0,
    }
    _obs, reward, _term, trunc, info = env.step(action)
    env.close()
    assert trunc is False
    assert reward == pytest.approx(float(info["true_score_delta"]))
    assert float(info["shaping"]) == 0.0
