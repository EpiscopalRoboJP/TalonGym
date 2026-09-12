import numpy as np

from talongym.presets.loader import load_bundle
from talongym.sim.world import World


def test_same_seed_replays_to_1e4_in():
    bundle = load_bundle()
    poses = []
    for _ in range(2):
        world = World(bundle, seed=11)
        world.reset(seed=11, static_teammate=True)
        log = []
        rs = world.actor()
        for i in range(80):
            target = np.array([rs.body.x + 4.0, rs.body.y + 2.0, 1.2], dtype=np.float64)
            world.step(target, 0.8, 0)
            rs = world.actor()
            log.append((rs.body.x, rs.body.y, rs.body.heading))
        poses.append(log)
    for a, b in zip(poses[0], poses[1]):
        assert abs(a[0] - b[0]) < 1e-4
        assert abs(a[1] - b[1]) < 1e-4
        assert abs(a[2] - b[2]) < 1e-4
