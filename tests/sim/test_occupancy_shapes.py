import numpy as np

from talongym.env.ftc_auto import FTCAutoEnv
from talongym.presets.loader import load_bundle
from talongym.sim.geometry import AABB, Circle, point_in_shape, point_in_volume, shape_from_element, tag_set
from talongym.sim.world import World


def _teleport_actor(env: FTCAutoEnv, x: float, y: float | None = None, vx: float = 0.0, vy: float = 0.0) -> None:
    rs = env.world.actor()
    rs.body.x = float(x)
    if y is not None:
        rs.body.y = float(y)
    rs.body.vx = float(vx)
    rs.body.vy = float(vy)
    backend = env.world.backend
    placed = getattr(backend, "_robot_placed", None)
    if isinstance(placed, set):
        placed.discard(rs.body.id)
    write = getattr(backend, "_write_robot", None)
    if callable(write):
        write(rs.body)
        mj = getattr(backend, "_mujoco", None)
        if mj is not None:
            mj.mj_forward(backend._mj, backend._data)


def _world() -> World:
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    return World(bundle, seed=0, allow_missing_mesh=True)


def test_point_in_shape_ignores_non_geometry():
    assert point_in_shape(AABB(0, 0, 2, 2), 0.0, 0.0)
    assert point_in_shape(Circle(0, 0, 2), 0.0, 0.0)
    assert not point_in_shape(None, 0.0, 0.0)
    assert not point_in_shape(iter({"a": 1}.items()), 0.0, 0.0)
    assert not point_in_shape({"kind": "aabb", "width": 4, "depth": 4}, 0.0, 0.0)


def test_shape_from_element_tolerates_garbage():
    assert shape_from_element("not-a-dict") is None  # type: ignore[arg-type]
    assert shape_from_element({"id": "x", "shape": iter({"kind": "aabb"}.items())}) is None
    assert tag_set(iter({"flower": True}.items())) == set()
    flower = {"id": "flower_1", "type": "flower", "tags": ["flower"], "pose": {"x": 0, "y": 0, "z": 0}, "shape": {"kind": "aabb", "width": 4, "depth": 4, "height": 8}}
    sh = shape_from_element(flower)
    flower["shape"] = iter({"height": 8}.items())
    flower["tags"] = iter({"flower": True}.items())
    assert point_in_volume(flower, sh, 0.0, 0.0, 0.0, 0.0)


def test_occupancy_survives_corrupt_live_lookups():
    world = _world()
    world.reset(seed=0, static_teammate=True, full_noise=False)
    key = next(iter(world.element_shapes))
    world.element_shapes[key] = iter({"x": 1}.items())
    piece = next(iter(world.pieces.values()))
    world.triggers = piece  # type: ignore[assignment]
    occ = world._occupancy()
    assert occ
    assert all(isinstance(ids, set) for ids in occ.values())


def test_start_pose_is_not_a_flower_ram():
    world = _world()
    world.reset(seed=0, static_teammate=True, full_noise=False)
    assert not world._chassis_hits_tagged_fixture(world.actor(), {"flower"})
    world.wall_hit = False
    world.piece_hit = False
    world._apply_flower_ram(world._occupancy())
    assert world.wall_hit is False
    assert world.piece_hit is False


def test_chassis_on_flower_is_a_wall_and_ball_ram():
    world = _world()
    world.reset(seed=0, static_teammate=True, full_noise=False)
    pose = next(el for el in world.elements if el.get("id") == "flower_1")["pose"]
    rs = world.actor()
    rs.body.x = float(pose["x"])
    rs.body.y = float(pose["y"])
    assert world._chassis_hits_tagged_fixture(rs, {"flower"})
    world.wall_hit = False
    world.piece_hit = False
    world._apply_flower_ram(world._occupancy())
    assert world.wall_hit is True
    assert world.piece_hit is True


def test_flower_overlap_does_not_count_as_leave():
    world = _world()
    world.reset(seed=0, static_teammate=True, full_noise=False)
    flower = next(el for el in world.elements if el.get("id") == "flower_4")
    fy = float(flower["pose"]["y"])
    rs = world.actor()
    # Center stays inside leave_interior (|x|<61) while the chassis still overlaps the cup.
    rs.body.x = -60.5
    rs.body.y = fy
    rs.body.heading = 0.0
    assert abs(rs.body.x) < 61.0
    assert not world._touching_perimeter(rs.body.x, rs.body.y, rs.body.heading)
    assert world._chassis_hits_tagged_fixture(rs, {"flower"})
    occ = world._occupancy()
    assert "red_0" not in occ.get("leave_interior", set())


def test_flower_ram_shapes_before_leaving_start_wall():
    env = FTCAutoEnv(record=False, static_teammate=True)
    env.reset(seed=0, options={"teammate_policy": "none", "opponent_policy": "none"})
    rs = env.world.actor()
    assert env.world._touching_perimeter(rs.body.x, rs.body.y, rs.body.heading)
    assert env._left_start_wall is False
    flower = next(el for el in env.world.elements if el.get("id") == "flower_4")
    _teleport_actor(env, -60.5, float(flower["pose"]["y"]))
    _obs, _reward, _term, _trunc, info = env.step(
        {
            "target_pose": np.array([rs.body.x, rs.body.y, rs.body.heading], dtype=np.float32),
            "speed_frac": np.array([0.2], dtype=np.float32),
            "mechanism": 0,
        }
    )
    assert env._left_start_wall is True
    assert env.world.wall_hit or env.world.piece_hit
    assert info["shaping"] <= -0.02
    env.close()
