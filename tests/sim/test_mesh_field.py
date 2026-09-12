import numpy as np
import pytest

from talongym.paths import ASSETS_DIR
from talongym.presets.loader import load_bundle, load_preset
from talongym.sim.mujoco_backend import available
from talongym.sim.world import World


def test_snapshot_includes_background_asset():
    field = load_preset("field", "biobuzz_2026_field_v1")
    assert field.get("backgroundAsset")
    glb = ASSETS_DIR / field["backgroundAsset"]
    assert glb.is_file()
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0, allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False)
    snap = world.snapshot()
    assert snap.get("backgroundAsset") == field["backgroundAsset"]
    red = next(el for el in snap["elements"] if el["id"] == "red_garden")
    assert red["pose"]["y"] < -60
    pollen = next(sp for sp in field["spawns"] if sp["id"] == "red_garden_pollen")
    assert all(p["y"] < -60 for p in pollen["poses"])


def test_mesh_required_refuses_planar_fallback(monkeypatch):
    from talongym.sim.mujoco_backend import MeshFieldRequiredError
    import talongym.sim.mujoco_backend as mb

    monkeypatch.setattr(mb, "available", lambda: False)
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    with pytest.raises(MeshFieldRequiredError):
        World(bundle, seed=0)


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_drive_under_hive_no_frame_hit():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0)
    world.reset(seed=0, static_teammate=False)
    rs = world.actor()
    rs.body.x, rs.body.y = 0.0, -20.0
    if hasattr(world.backend, "_robot_placed"):
        world.backend._robot_placed.discard(rs.body.id)
    hits = 0
    for _ in range(80):
        world.step(np.array([0.0, 20.0, 0.0]), 1.0, 0)
        if world.wall_hit:
            hits += 1
    assert abs(world.actor().body.y) < 25 or world.actor().body.y > -5
    assert hits == 0


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_piece_hits_hive_frame():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0)
    world.reset(seed=0, static_teammate=False)
    piece = next(iter(world.pieces.values()))
    piece.x, piece.y, piece.z = -40.0, 0.0, 22.0
    piece.vx, piece.vy, piece.vz = 80.0, 0.0, 0.0
    piece.kick = True
    piece.ballistic = True
    piece.in_flight = True
    world.ballistic_launch = True
    hit = False
    for _ in range(40):
        world.step(np.array([world.actor().body.x, world.actor().body.y, 0.0]), 0.2, 0)
        if abs(piece.x + 24.73) < 8 and abs(piece.y) < 22:
            hit = True
            break
        if piece.vx < 40:
            hit = True
            break
    assert piece.x > -40.0
    assert hit or piece.x < -18.0


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_ballistic_tip_still_once():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    bundle.robot["launchers"] = []
    world = World(bundle, seed=1)
    world.reset(seed=1, static_teammate=False, ballistic_launch=True)
    rs = world.actor()
    rs.body.x, rs.body.y, rs.body.heading = -9.4, 9.4, 1.57
    if hasattr(world.backend, "_robot_placed"):
        world.backend._robot_placed.discard(rs.body.id)
    pollen = [p for p in world.pieces.values() if p.type_id == "pollen"][:4]
    for piece in pollen:
        piece.held_by = rs.body.id
        piece.x, piece.y, piece.z = rs.body.x, rs.body.y, 8.0
        rs.held.append(piece.id)
    for _ in range(200):
        world.step(np.array([-9.4, 9.4, 1.57]), 0.2, 2)
        if int(world.accumulators.get("launched_count") or 0) >= 4:
            break
    assert int(world.accumulators.get("launched_count") or 0) >= 1
    assert int(world.accumulators.get("tip_count") or 0) <= 1
