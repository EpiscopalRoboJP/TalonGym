import numpy as np

from talongym.presets.loader import load_bundle
from talongym.sim.world import World


def test_four_launches_tip_once():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=1, allow_missing_mesh=True)
    world.reset(seed=1, static_teammate=False)
    rs = world.actor()
    cell = next(el for el in world.elements if el["id"] == "red_cell_up")
    cx, cy = float(cell["pose"]["x"]), float(cell["pose"]["y"])
    cz = float(cell["pose"].get("z") or 8.0)
    pollen = [p for p in world.pieces.values() if p.type_id == "pollen"][:4]
    assert len(pollen) == 4
    for piece in pollen:
        if piece.id in rs.held:
            rs.held.remove(piece.id)
        piece.held_by = None
        piece.scored = False
        piece.in_flight = False
        piece.x, piece.y, piece.z = cx, cy, cz
        piece.vx = piece.vy = piece.vz = 0.0
    pose = np.array([rs.body.x, rs.body.y, rs.body.heading], dtype=np.float64)
    for _ in range(8):
        world.step(pose, 0.2, 0)
        if int(world.accumulators.get("launched_count") or 0) >= 4:
            break
    assert int(world.accumulators.get("launched_count") or 0) == 4
    assert int(world.accumulators.get("tip_count") or 0) == 1
    assert world.true_score == 20
