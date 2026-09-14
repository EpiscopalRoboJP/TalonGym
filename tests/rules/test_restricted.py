import numpy as np

from talongym.presets.loader import load_bundle
from talongym.sim.world import World


def test_restricted_entry_is_a_foul():
    bundle = load_bundle()
    world = World(bundle, seed=0, allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False)
    rs = world.actor()
    rs.body.x, rs.body.y = 20.0, 0.0
    world.step(np.array([20.0, 0.0, 0.0]), 0.2, 0, end_phase=False)
    assert world.accumulators.get("restricted_entry") is True
    assert world.true_score == -15
    assert any(float(e.get("points") or 0) < 0 for e in world.explains)
    snap = world.snapshot()
    assert snap["penalties"]
    assert any(float(e.get("points") or 0) < 0 for e in snap["stepExplains"])
    assert snap["trueScore"] == -15
