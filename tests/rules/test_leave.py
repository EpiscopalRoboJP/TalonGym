from talongym.presets.loader import load_bundle
from talongym.sim.world import World
from talongym.training.policies import scripted_auto
from talongym.env.ftc_auto import FTCAutoEnv
import numpy as np


def test_leave_scores_at_phase_end():
    bundle = load_bundle()
    world = World(bundle, seed=0)
    world.reset(seed=0, static_teammate=False)
    rs = world.actor()
    # Drive toward the goal side past the start zone.
    for _ in range(80):
        world.step(np.array([rs.body.x, 20.0, 1.57]), 1.0, 0, end_phase=False)
        rs = world.actor()
    world.step(np.array([rs.body.x, rs.body.y, rs.body.heading]), 0.2, 0, end_phase=True)
    assert world.accumulators.get("left_launch_line") is True
    assert world.true_score >= 3
