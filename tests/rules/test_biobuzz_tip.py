import numpy as np

from talongym.presets.loader import load_bundle
from talongym.sim.world import World


def test_four_launches_tip_once():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=1, allow_missing_mesh=True)
    world.reset(seed=1, static_teammate=False)
    rs = world.actor()
    rs.body.x, rs.body.y = -9.4, 9.4
    pollen = [p for p in world.pieces.values() if p.type_id == "pollen"][:4]
    assert len(pollen) == 4
    for piece in pollen:
        piece.held_by = rs.body.id
        piece.x, piece.y = rs.body.x, rs.body.y
        rs.held.append(piece.id)
    for _ in range(100):
        world.step(np.array([-9.4, 9.4, 1.57]), 0.2, 2)
        rs = world.actor()
        if int(world.accumulators.get("launched_count") or 0) >= 4:
            break
    assert int(world.accumulators.get("launched_count") or 0) == 4
    assert int(world.accumulators.get("tip_count") or 0) == 1
    assert world.true_score == 20
