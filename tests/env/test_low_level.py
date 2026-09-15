import numpy as np

from talongym.env.ftc_auto import FTCAutoEnv


def test_low_level_velocity_steps():
    env = FTCAutoEnv(record=False, static_teammate=False, action_tier="low_level_velocity")
    obs, info = env.reset(seed=0)
    action = {"velocity": np.array([20.0, 0.0, 0.0], dtype=np.float32), "mechanism": 0}
    obs2, rew, term, trunc, info = env.step(action)
    assert np.isfinite(rew)
    assert not term
    assert obs2["pose_noisy"].shape == (3,)
    env.close()
