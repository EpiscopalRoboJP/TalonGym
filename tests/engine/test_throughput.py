import time

from talongym.env.ftc_auto import FTCAutoEnv, FlatBoxEnv
import numpy as np


def test_throughput_smoke(capsys):
    env = FlatBoxEnv(FTCAutoEnv(record=False, static_teammate=False))
    obs, _ = env.reset(seed=0)
    n = 400
    t0 = time.perf_counter()
    for _ in range(n):
        action = env.action_space.sample()
        action = np.asarray(action, dtype=np.float32)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - t0
    rate = n / max(elapsed, 1e-6)
    print(f"control_steps_per_s={rate:.1f}")
    env.close()
    assert rate > 50
