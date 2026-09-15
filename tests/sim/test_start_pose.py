import pytest

from talongym.presets.loader import load_bundle
from talongym.sim.world import World


def _world(seed: int = 0) -> World:
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=seed, allow_missing_mesh=True)
    return world


def test_default_starts_are_legal_and_unscored():
    world = _world()
    world.reset(seed=0, static_teammate=True, full_noise=False)
    park_shape = world.element_shapes["red_park"]
    for rs in world.robots.values():
        assert world._start_pose_error(rs.body.x, rs.body.y, rs.body.heading, rs.body.alliance) is None
        assert world._touching_perimeter(rs.body.x, rs.body.y, rs.body.heading)
        assert not world._robot_aabb(rs.body.x, rs.body.y, rs.body.heading).overlaps_aabb(park_shape)


def test_inward_offset_is_rejected():
    world = _world()
    with pytest.raises(ValueError, match="outside start slot"):
        world.reset(
            seed=0,
            static_teammate=False,
            full_noise=False,
            match_setup={
                "robots": [
                    {
                        "id": "red_0",
                        "enabled": True,
                        "startSlotId": "red_0",
                        "offset": {"x": 8.0, "y": 0.0, "headingDeg": 0.0},
                    }
                ]
            },
        )


def test_loading_zone_start_is_rejected():
    world = _world()
    world.reset(seed=0, static_teammate=False, full_noise=False)
    error = world._start_pose_error(-61.6, 35.25, 0.0, "red")
    assert error is not None
    assert "LOADING ZONE" in error


def test_interior_start_is_rejected():
    world = _world()
    world.reset(seed=0, static_teammate=False, full_noise=False)
    error = world._start_pose_error(-40.0, 0.0, 0.0, "red")
    assert error is not None
    assert "perimeter wall" in error


def test_cross_alliance_slot_is_rejected():
    world = _world()
    with pytest.raises(ValueError, match="not valid for red"):
        world.reset(
            seed=0,
            static_teammate=False,
            full_noise=False,
            match_setup={
                "robots": [
                    {
                        "id": "red_0",
                        "enabled": True,
                        "startSlotId": "blue_0",
                        "offset": {"x": 0.0, "y": 0.0, "headingDeg": 0.0},
                    }
                ]
            },
        )


def test_legal_along_wall_offset_is_kept():
    world = _world()
    world.reset(
        seed=0,
        static_teammate=False,
        full_noise=False,
        match_setup={
            "robots": [
                {
                    "id": "red_0",
                    "enabled": True,
                    "startSlotId": "red_1",
                    "offset": {"x": 0.0, "y": 3.0, "headingDeg": 0.0},
                }
            ]
        },
    )
    assert world.actor().body.y == pytest.approx(3.0, abs=0.2)
    assert world._start_pose_error(
        world.actor().body.x,
        world.actor().body.y,
        world.actor().body.heading,
        "red",
    ) is None


def test_start_jitter_stays_legal():
    world = _world(seed=3)
    park_shape = world.element_shapes["red_park"]
    for seed in range(12):
        world.reset(seed=seed, static_teammate=False, full_noise=True)
        body = world.actor().body
        assert world._start_pose_error(body.x, body.y, body.heading, "red") is None
        assert world._touching_perimeter(body.x, body.y, body.heading)
        assert not world._robot_aabb(body.x, body.y, body.heading).overlaps_aabb(park_shape)


def test_unmoved_wall_start_does_not_leave():
    import math

    world = _world()
    world.reset(seed=0, static_teammate=False, full_noise=False)
    assert "red_0" not in world.prev_occupancy.get("leave_interior", set())
    assert "red_0" not in world.prev_occupancy.get("red_park", set())
    heading = math.radians(20.0)
    x, y = world._snap_touching_wall("neg_x", -61.6, -48.0, heading)
    world.actor().body.x = x
    world.actor().body.y = y
    world.actor().body.heading = heading
    assert world._touching_perimeter(x, y, heading)
    occ = world._occupancy()
    assert "red_0" not in occ.get("leave_interior", set())
