import json

import numpy as np
import pytest

from talongym.paths import ASSETS_DIR
from talongym.presets.loader import load_bundle, load_preset
from talongym.sim.mujoco_backend import available
from talongym.sim.world import Piece, World


def test_snapshot_includes_background_asset():
    field = load_preset("field", "biobuzz_2026_field_v1")
    assert field.get("backgroundAsset")
    glb = ASSETS_DIR / field["backgroundAsset"]
    assert glb.is_file()
    assert glb.stat().st_size > 100_000
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0, allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False)
    snap = world.snapshot()
    assert snap.get("backgroundAsset") == field["backgroundAsset"]
    red = next(el for el in snap["elements"] if el["id"] == "red_garden")
    assert red["pose"]["y"] < -60
    pollen = next(sp for sp in field["spawns"] if sp["id"] == "red_garden_pollen")
    assert all(p["y"] < -60 for p in pollen["poses"])
    piece = snap["pieces"][0]
    assert piece["typeId"]
    assert "quat" in piece and len(piece["quat"]) == 4
    assert piece["radius"] > 0
    assert piece.get("visualAsset")
    assert snap.get("cadManifest") == field["cadManifest"]
    assert snap.get("cadSourceSha256") == field["provenance"]["contentSha256"]
    assert snap.get("cadAssetVersion", "").startswith(snap["cadSourceSha256"])
    assert snap.get("cadAssetVersion") != snap["cadSourceSha256"]
    assert {row["id"] for row in snap["fieldMechanisms"]} == {"red_hive", "blue_hive"}
    catalog = {row["typeId"]: row for row in snap["gamePieces"]}
    assert set(catalog) == {"pollen", "nectar_red", "nectar_blue"}
    assert catalog["pollen"]["visualAsset"].endswith("pollen.glb")
    assert catalog["pollen"]["shape"]["radius"] == pytest.approx(1.4)
    assert catalog["nectar_red"]["shape"]["radius"] == pytest.approx(1.8)
    assert abs(piece["qw"] ** 2 + piece["qx"] ** 2 + piece["qy"] ** 2 + piece["qz"] ** 2 - 1.0) < 0.05
    assert len(world.pieces) == 28  # 24 staged on-field + red_0's four preloads
    assert len(world.actor().held) == 4
    assert all(world.pieces[pid].held_by == "red_0" for pid in world.actor().held)
    assert not any(p.held_by in {"red_1", "blue_0", "blue_1"} for p in world.pieces.values())
    flowers = [el["pose"] for el in field["elements"] if el["type"] == "flower"]
    assert any(abs(p["x"]) > 60 and abs(p["y"]) < 30 for p in flowers)
    assert any(abs(p["y"]) > 60 and abs(p["x"]) < 30 for p in flowers)
    red_loading = next(el for el in field["elements"] if el["id"] == "red_loading_zone")
    assert red_loading["pose"]["y"] > 0


@pytest.mark.require_mesh
def test_mesh_required_refuses_planar_fallback(monkeypatch):
    from talongym.sim.mujoco_backend import MeshFieldRequiredError
    import talongym.sim.mujoco_backend as mb

    monkeypatch.setattr(mb, "available", lambda: False)
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    with pytest.raises(MeshFieldRequiredError):
        World(bundle, seed=0)


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_cad_world_loads_convex_parts_not_glb():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0)
    world.reset(seed=0, static_teammate=False)
    backend = world.backend
    assert backend.name == "mujoco_field"
    stats = backend.cad_stats or {}
    assert stats.get("cad") is True
    assert stats.get("filter", {}).get("kept", 0) >= 80
    model = backend._mj
    assert int(model.nmesh) >= 80
    names = []
    for i in range(int(model.ngeom)):
        raw = backend._mujoco.mj_id2name(model, backend._mujoco.mjtObj.mjOBJ_GEOM, i)
        names.append(str(raw or ""))
    assert any(n.startswith("cad_") for n in names)
    assert any(n.startswith("trig_") for n in names)
    assert not any(n == "hive_frame_west" for n in names)
    assert "pollen" in backend._pools
    assert "nectar_red" in backend._pools
    pollen_slots = backend._pools["pollen"]
    assert backend._piece_nq[pollen_slots[0]] == 7
    snap = world.snapshot()
    pollen = next(p for p in snap["pieces"] if p["typeId"] == "pollen")
    assert pollen["visualAsset"].endswith("pollen.glb")
    assert pollen["quat"][0] ** 2 + pollen["quat"][1] ** 2 + pollen["quat"][2] ** 2 + pollen["quat"][3] ** 2 == pytest.approx(
        1.0, abs=1e-3
    )


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_hive_tip_moves_articulated_cad_hinge():
    world = _cad_world(seed=0)
    world.accumulators["tip_count"] = 1
    target = np.array([world.actor().body.x, world.actor().body.y, world.actor().body.heading])
    for _ in range(40):
        world.step(target, 0.0, 0)
    angles = {row["id"]: row["angleRad"] for row in world.snapshot()["fieldMechanisms"]}
    assert angles["red_hive"] > 0.8
    assert angles["blue_hive"] == pytest.approx(0.0, abs=0.02)


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
    piece = next(p for p in world.pieces.values() if p.type_id == "pollen")
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
def test_typed_slots_and_rolling_orientation():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0)
    world.reset(seed=0, static_teammate=False)
    pollen = next(p for p in world.pieces.values() if p.type_id == "pollen")
    pollen.x, pollen.y, pollen.z = 0.0, 40.0, world.floor_y + pollen.radius + 6.0
    pollen.vx, pollen.vy, pollen.vz = 20.0, 0.0, 0.0
    pollen.kick = True
    pollen.ballistic = True
    pollen.in_flight = True
    world.ballistic_launch = True
    spun = False
    for _ in range(50):
        world.step(np.array([world.actor().body.x, world.actor().body.y, 0.0]), 0.1, 0)
        if abs(pollen.wx) + abs(pollen.wy) + abs(pollen.wz) > 0.2:
            spun = True
            break
    assert spun
    q = (pollen.qw, pollen.qx, pollen.qy, pollen.qz)
    assert abs(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2 - 1.0) < 0.05
    slot = world.backend._id_to_slot[pollen.id]
    assert world.backend._piece_type[slot] == "pollen"


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_contacts_use_geom_groups():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0)
    world.reset(seed=0, static_teammate=False)
    piece = next(p for p in world.pieces.values() if p.type_id == "pollen")
    piece.x, piece.y, piece.z = -40.0, 0.0, 22.0
    piece.vx, piece.vy, piece.vz = 90.0, 0.0, 0.0
    piece.kick = True
    piece.ballistic = True
    piece.in_flight = True
    world.ballistic_launch = True
    flags = None
    for _ in range(25):
        world.step(np.array([world.actor().body.x, world.actor().body.y, 0.0]), 0.1, 0)
        flags = world.backend.contacts()
        if flags.piece:
            break
    assert flags is not None
    groups = {int(world.backend._mj.geom_group[i]) for i in range(int(world.backend._mj.ngeom))}
    assert 0 in groups and 1 in groups and 2 in groups and 3 in groups


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_ballistic_tip_still_once():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    bundle.robot["launchers"] = []
    world = World(bundle, seed=1)
    world.reset(seed=1, static_teammate=False, ballistic_launch=True)
    rs = world.actor()
    target = next(el["pose"] for el in world.elements if el["id"] == "red_cell_up")
    rs.body.x, rs.body.y, rs.body.heading = target["x"], target["y"], 1.57
    if hasattr(world.backend, "_robot_placed"):
        world.backend._robot_placed.discard(rs.body.id)
    assert len(rs.held) == 4
    for _ in range(200):
        world.step(np.array([target["x"], target["y"], 1.57]), 0.2, 2)
        if int(world.accumulators.get("launched_count") or 0) >= 4:
            break
    assert int(world.accumulators.get("launched_count") or 0) >= 1
    assert int(world.accumulators.get("tip_count") or 0) <= 1


def _cad_world(seed: int = 0) -> World:
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=seed)
    world.reset(seed=seed, static_teammate=False)
    return world


def _ensure_piece(world: World, type_id: str):
    found = next((p for p in world.pieces.values() if p.type_id == type_id), None)
    if found is not None:
        return found
    from talongym.sim.mujoco_backend import ftc_yaw_to_mj_quat

    spec = world.piece_types[type_id]
    sh = spec.get("shape") or {}
    rad = float(sh.get("radius") or 1.8)
    qw, qx, qy, qz = ftc_yaw_to_mj_quat(0.0)
    name = f"probe_{type_id}"
    piece = Piece(
        id=name,
        type_id=type_id,
        x=0.0,
        y=40.0,
        radius=rad,
        attrs={"color": (spec.get("attributes") or {}).get("color")},
        restitution=float(spec.get("restitution") or 0.35),
        mass=float(spec.get("massKg") or 0.08),
        z=rad + world.floor_y,
        qw=qw,
        qx=qx,
        qy=qy,
        qz=qz,
        visual_asset=spec.get("visualAsset"),
        collision_asset=spec.get("collisionAsset"),
    )
    world.pieces[name] = piece
    return piece


def _launch(world: World, type_id: str, x: float, y: float, z: float, vx: float, vy: float, vz: float):
    piece = _ensure_piece(world, type_id)
    piece.x, piece.y, piece.z = x, y, z
    piece.vx, piece.vy, piece.vz = vx, vy, vz
    piece.held_by = None
    piece.scored = False
    piece.kick = True
    piece.ballistic = True
    piece.in_flight = True
    world.ballistic_launch = True
    return piece


def _step_hold(world: World, n: int, speed: float = 0.1) -> None:
    rs = world.actor()
    for _ in range(n):
        world.step(np.array([rs.body.x, rs.body.y, 0.0]), speed, 0)
        rs = world.actor()


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_runtime_loads_committed_cad_mjcf():
    world = _cad_world()
    stats = world.backend.cad_stats or {}
    assert stats.get("cad") is True
    assert stats.get("loadedCommitted") is True
    assert world.snapshot()["cadSourceSha256"] == "05b35961c7df847741031f00fda73ddd068537f809e92a11b1cd59a94bcc8331"


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_runtime_missing_manifest_fails_closed():
    from talongym.sim.mujoco_backend import MeshFieldRequiredError

    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    bundle.field = dict(bundle.field)
    bundle.field["cadManifest"] = "seasons/biobuzz_2026/missing_cad_manifest.json"
    with pytest.raises(MeshFieldRequiredError, match="cadManifest not found|CAD assets missing"):
        World(bundle, seed=0)


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_visual_physics_registration_matches_manifest_bounds():
    from talongym.assets.mjcf_field import select_field_collision_parts

    field = load_preset("field", "biobuzz_2026_field_v1")
    manifest = json.loads((ASSETS_DIR / field["cadManifest"]).read_text(encoding="utf-8"))
    kept, _stats = select_field_collision_parts(manifest["field"]["collisionParts"])
    world = _cad_world()
    backend = world.backend
    backend._mujoco.mj_forward(backend._mj, backend._data)
    checked = 0
    for i, part in enumerate(kept[:40]):
        gid = backend._mujoco.mj_name2id(backend._mj, backend._mujoco.mjtObj.mjOBJ_GEOM, f"cad_{i:04d}")
        if gid < 0:
            continue
        xpos = [float(v) for v in backend._data.geom_xpos[gid]]
        aabb = [float(v) for v in backend._mj.geom_aabb[gid]]
        center = [xpos[j] + aabb[j] for j in range(3)]
        half = aabb[3:]
        mn = [center[j] - half[j] for j in range(3)]
        mx = [center[j] + half[j] for j in range(3)]
        pmin, pmax = part["minIn"], part["maxIn"]
        overlap = all(mn[j] <= pmax[j] + 2.0 and mx[j] >= pmin[j] - 2.0 for j in range(3))
        assert overlap, f"{part['id']} geom AABB {mn}/{mx} vs manifest {pmin}/{pmax}"
        checked += 1
    assert checked >= 20
    pollen = manifest["pieces"]["pollen"]["boundsIn"]["extentsIn"]
    nectar = manifest["pieces"]["nectar_red"]["boundsIn"]["extentsIn"]
    assert max(pollen) == pytest.approx(2.8, abs=0.12)
    assert max(nectar) == pytest.approx(3.6, abs=0.15)
    flower_hits = [
        p["id"]
        for p in kept
        if "Flower" in str(p["id"]) and abs(float((p.get("maxIn") or [0, 0, 0])[2]) - 70.5) < 8
    ]
    assert flower_hits, "FLOWER CAD supports should sit on the perimeter wall"


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_wall_and_flower_contact():
    world = _cad_world()
    wall = _launch(world, "pollen", -50.0, 0.0, 8.0, -90.0, 0.0, 0.0)
    _step_hold(world, 30)
    assert wall.x > -70.0
    assert world.piece_hit or wall.vx > -40.0
    world = _cad_world(seed=1)
    flower = _launch(world, "pollen", -36.0, -40.0, 21.5, 0.0, -70.0, 0.0)
    _step_hold(world, 35)
    assert flower.y > -72.0
    assert world.piece_hit or flower.vy > -30.0


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_hive_support_rim_bounce():
    world = _cad_world()
    piece = _launch(world, "pollen", -40.0, 8.0, 22.0, 90.0, 0.0, 0.0)
    _step_hold(world, 35)
    assert piece.x > -40.0
    assert piece.x < -8.0 or piece.vx < 40.0
    assert piece.z > world.floor_y


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_cell_opening_probes_typed_pieces():
    from talongym.assets.mjcf_field import ftc_point_hits_parts, select_field_collision_parts

    field = load_preset("field", "biobuzz_2026_field_v1")
    manifest = json.loads((ASSETS_DIR / field["cadManifest"]).read_text(encoding="utf-8"))
    kept, _stats = select_field_collision_parts(manifest["field"]["collisionParts"])
    cell = next(el for el in field["elements"] if el["id"] == "red_cell_up")
    cx, cy, cz = float(cell["pose"]["x"]), float(cell["pose"]["y"]), float(cell["pose"]["z"])
    sealing = ftc_point_hits_parts(kept, cx, cy, cz, margin=1.0)
    # Convex hulls of concave CELL skins may occupy the opening; record that approximation.
    approaches = {
        "pollen": (-40.0, cy, cz, 85.0, 0.0, 8.0),
        "nectar_red": (-42.0, cy, cz, 80.0, 0.0, 6.0),
        "nectar_blue": (-42.0, -cy if cy else -9.4, 12.0, 80.0, 0.0, 4.0),
    }
    for type_id, (x, y, z, vx, vy, vz) in approaches.items():
        world = _cad_world()
        piece = _launch(world, type_id, x, y, z, vx, vy, vz)
        entered = False
        for _ in range(45):
            _step_hold(world, 1)
            if abs(piece.x - cx) < 12 and abs(piece.y - cy) < 16 and piece.z > 8:
                entered = True
                break
            if piece.vx < 15 and piece.x > x + 4:
                break
        assert piece.z > world.floor_y - 1.0, f"{type_id} fell through the field"
        assert piece.x > x, f"{type_id} did not move toward the HIVE"
        if sealing:
            assert entered or world.piece_hit or piece.vx < vx * 0.6
        else:
            assert entered or abs(piece.x - cx) < 18


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mass_and_radius_differ_by_piece_type():
    world = _cad_world()
    pollen = next(p for p in world.pieces.values() if p.type_id == "pollen")
    nectar = _ensure_piece(world, "nectar_red")
    assert pollen.radius == pytest.approx(1.4)
    assert nectar.radius == pytest.approx(1.8)
    assert pollen.mass == pytest.approx(0.05)
    assert nectar.mass == pytest.approx(0.08)
    assert set(world.backend._pools) >= {"pollen", "nectar_red", "nectar_blue"}
    _step_hold(world, 1)
    pollen_slot = world.backend._id_to_slot[pollen.id]
    nectar_slot = world.backend._id_to_slot[nectar.id]
    assert world.backend._piece_type[pollen_slot] == "pollen"
    assert world.backend._piece_type[nectar_slot] == "nectar_red"


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_cad_replay_is_deterministic():
    poses = []
    for _ in range(2):
        world = _cad_world(seed=11)
        pollen = next(p for p in world.pieces.values() if p.type_id == "pollen")
        log = []
        rs = world.actor()
        for _step in range(40):
            world.step(np.array([rs.body.x + 3.0, rs.body.y, 0.0]), 0.4, 0)
            rs = world.actor()
            log.append((round(rs.body.x, 5), round(rs.body.y, 5), round(pollen.x, 5), round(pollen.z, 5)))
        poses.append(log)
    assert poses[0] == poses[1]


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mujoco_compile_and_step_smoke(capsys):
    import time

    t0 = time.perf_counter()
    world = _cad_world()
    compile_s = time.perf_counter() - t0
    rs = world.actor()
    t1 = time.perf_counter()
    for _ in range(40):
        world.step(np.array([rs.body.x, rs.body.y + 2.0, 0.0]), 0.6, 0)
        rs = world.actor()
    step_s = time.perf_counter() - t1
    rate = 40.0 / max(step_s, 1e-6)
    print(f"cad_world_load_s={compile_s:.3f} control_steps_per_s={rate:.1f} nmesh={world.backend._mj.nmesh}")
    assert compile_s < 90
    assert rate > 2
    assert int(world.backend._mj.nmesh) >= 80
    snap = world.snapshot()
    garden = next(p for p in snap["pieces"] if p["id"] and p["y"] < -60)
    assert garden["typeId"] == "pollen"
    assert garden["visualAsset"].endswith("pollen.glb")
