import numpy as np
import pytest

from talongym.env.petting import FTCAutoParallelEnv
from talongym.presets.loader import load_bundle
from talongym.sim.world import World


@pytest.mark.require_mesh
def test_two_reds_move_and_contact():
    pytest.importorskip("pettingzoo")
    env = FTCAutoParallelEnv(opponent_mode="none")
    obs, _ = env.reset(seed=1, options={"opponent_mode": "none", "live_teammate": True, "teammate_policy": "independent"})
    assert "red_0" in obs and "red_1" in obs
    w = env._gym.world
    start0 = (w.robots["red_0"].body.x, w.robots["red_0"].body.y)
    start1 = (w.robots["red_1"].body.x, w.robots["red_1"].body.y)
    for _ in range(100):
        a0 = w.robots["red_0"].body
        a1 = w.robots["red_1"].body
        actions = {
            "red_0": {
                "target_pose": np.array([a1.x, a1.y, 0.0], dtype=np.float32),
                "speed_frac": np.array([1.0], dtype=np.float32),
                "mechanism": 0,
            },
            "red_1": {
                "target_pose": np.array([a0.x, a0.y, 0.0], dtype=np.float32),
                "speed_frac": np.array([1.0], dtype=np.float32),
                "mechanism": 0,
            },
        }
        env.step(actions)
    end0 = (w.robots["red_0"].body.x, w.robots["red_0"].body.y)
    assert end0 != start0 or (w.robots["red_1"].body.x, w.robots["red_1"].body.y) != start1
    assert w.robots["red_0"].collision_time_s > 0 or w.robots["red_1"].collision_time_s > 0
    env.close()


def test_restricted_volume_sets_flag():
    world = World(load_bundle(), control_hz=25, allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False)
    rs = world.actor()
    rs.body.x, rs.body.y = 36.0, 0.0
    world.step(np.array([36.0, 0.0, 0.0]), 0.2, 0)
    rs = world.actor()
    assert rs.entered_restricted or world.accumulators.get("restricted_entry")
