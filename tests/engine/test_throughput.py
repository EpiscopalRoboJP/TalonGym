import time

import numpy as np

from talongym.env.ftc_auto import FlatBoxEnv, FTCAutoEnv
from talongym.presets.loader import load_bundle


def test_throughput_smoke(capsys):
    # Pin the engine smoke robot so catalog scoring starters do not hide a physics-loop regression.
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    env = FlatBoxEnv(FTCAutoEnv(bundle=bundle, record=False, static_teammate=False))
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
