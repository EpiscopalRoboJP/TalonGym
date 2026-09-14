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
    assert world.true_score == -20
    assert any(float(e.get("points") or 0) < 0 for e in world.explains)
    snap = world.snapshot()
    assert snap["penalties"]
    assert any(float(e.get("points") or 0) < 0 for e in snap["stepExplains"])
    assert snap["trueScore"] == -20


def test_opponent_in_its_own_territory_is_not_a_red_foul():
    """restricted_for_red covers blue's home area; a blue robot entering it
    is not committing a G402 foul and must not dock the (red) trueScore."""
    bundle = load_bundle()
    world = World(bundle, seed=0, allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False, opponent_mode="scripted", live_teammate=False)
    blue = world.robots["blue_0"]
    blue.body.x, blue.body.y = 40.0, 0.0
    # Simulate blue freshly entering the zone this tick (it already starts inside its own territory).
    world.prev_occupancy.setdefault("restricted_for_red", set()).discard(blue.body.id)
    world.step(
        actions={
            "red_0": {"target_pose": np.array([-63.0, -24.0, 0.0]), "speed_frac": 0.2, "mechanism": 0},
            "blue_0": {"target_pose": np.array([40.0, 0.0, 0.0]), "speed_frac": 0.2, "mechanism": 0},
        }
    )
    assert world.true_score == 0
    assert world.accumulators.get("restricted_entry") is not True
