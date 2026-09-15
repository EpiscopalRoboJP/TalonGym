import math

import numpy as np

from talongym.env.ftc_auto import FTCAutoEnv
from talongym.presets.loader import load_bundle
from talongym.sim.geometry import AABB, detour_waypoints, path_length, segment_crosses_aabb
from talongym.sim.world import PERIMETER_FACE_INSET_IN, World


def test_default_env_fields_four_robots_holding_preloads():
    env = FTCAutoEnv(record=True)
    env.reset(seed=3)
    world = env.world
    assert set(world.robots) == {"red_0", "red_1", "blue_0", "blue_1"}
    for rid, rs in world.robots.items():
        assert len(rs.held) == world.capacity, rid
        assert all(world.pieces[pid].held_by == rid for pid in rs.held)
    env.close()


def test_spawns_stay_inside_perimeter():
    bundle = load_bundle()
    half = float(bundle.field["fieldSizeIn"]["width"]) / 2.0 - PERIMETER_FACE_INSET_IN
    world = World(bundle, allow_missing_mesh=True)
    for seed in range(20):
        world.reset(seed=seed, opponent_mode="static")
        for p in world.pieces.values():
            assert abs(p.x) + p.radius <= half + 1e-6 and abs(p.y) + p.radius <= half + 1e-6, (seed, p.id, p.x, p.y)
        for rid, rs in world.robots.items():
            reach = max(world.robot_hx, world.robot_hy)
            assert abs(rs.body.x) + reach <= half + 1e-6 and abs(rs.body.y) + reach <= half + 1e-6, (seed, rid)


def test_missing_robot_drops_its_preload():
    world = World(load_bundle(), allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False)
    assert "red_1" not in world.robots
    assert not any(p.held_by == "red_1" for p in world.pieces.values())
    assert sum(1 for p in world.pieces.values() if p.held_by) == len(world.actor().held)


def test_detour_skirts_blocking_box():
    box = AABB(0.0, 0.0, 2.0, 20.0)
    route = detour_waypoints(-30.0, 0.0, 30.0, 0.0, [box], inflate=9.0)
    grown = AABB(0.0, 0.0, 11.0, 29.0)
    pts = [(-30.0, 0.0), *route]
    assert len(route) == 3
    assert not any(segment_crosses_aabb(*pts[i], *pts[i + 1], grown) for i in range(len(pts) - 1))
    assert path_length(-30.0, 0.0, route) > 60.0
    assert detour_waypoints(-30.0, 40.0, 30.0, 40.0, [box], inflate=9.0) == [(30.0, 40.0)]
    assert math.isclose(path_length(0.0, 0.0, [(3.0, 4.0)]), 5.0)


def test_launch_that_scores_nothing_counts_as_a_miss():
    from talongym.sim.world import LAUNCH_SCORE_WINDOW_S

    world = World(load_bundle(), allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False, ballistic_launch=True)
    rs = world.actor()
    # Facing the audience wall from the start slot: nothing the launcher reaches is a goal.
    rs.body.heading = -math.pi / 2
    if hasattr(world.backend, "_robot_placed"):
        world.backend._robot_placed.discard("red_0")
    misses = 0
    for _ in range(int((LAUNCH_SCORE_WINDOW_S + 1.5) * world.control_hz)):
        world.step(np.array([rs.body.x, rs.body.y, -math.pi / 2]), 0.2, 2 if len(rs.held) == 4 else 0)
        misses += world.missed_launches.get("red_0", 0)
    assert len(rs.held) == 3
    assert misses == 1


def test_shaping_potential_is_continuous_across_launch_at_spot():
    env = FTCAutoEnv(record=False)
    env.reset(seed=0)
    world = env.world
    spot = next(e for e in world.elements if "launch_spot" in (e.get("tags") or []) and e.get("alliance") == "red")
    rs = world.actor()
    rs.body.x, rs.body.y = float(spot["pose"]["x"]), float(spot["pose"]["y"])
    loaded = env._potential()
    held, rs.held = list(rs.held), []
    empty = env._potential()
    rs.held = held
    assert loaded < 0 and math.isclose(loaded, empty, abs_tol=1e-6)
    env.close()
