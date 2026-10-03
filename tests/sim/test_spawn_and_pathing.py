"""Loose-piece spawn clamp, A* follower brake-to-stop, launch spots, 4-robot eval."""

from __future__ import annotations

import math

import numpy as np

from talongym.env.ftc_auto import FTCAutoEnv
from talongym.presets.loader import load_bundle
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


def _world(seed: int = 0) -> World:
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=seed, allow_missing_mesh=True)
    return world


def test_launch_spots_are_sim_targets_not_colliders():
    world = _world()
    world.reset(seed=0, static_teammate=False, full_noise=False)
    red = next(el for el in world.elements if el["id"] == "red_launch_spot")
    blue = next(el for el in world.elements if el["id"] == "blue_launch_spot")
    assert "launch_spot" in red["tags"]
    assert red["isCollider"] is False
    assert blue["pose"]["x"] == -red["pose"]["x"]
    assert blue["pose"]["y"] == -red["pose"]["y"]
    nav_ids = {(box.cx, box.cy) for box in world.nav_obstacles}
    assert (red["pose"]["x"], red["pose"]["y"]) not in nav_ids


def test_loose_garden_pollen_jitters_as_a_group():
    world = _world()
    world.reset(seed=3, static_teammate=False, full_noise=True)
    garden = next(el for el in world.elements if el["id"] == "red_garden")
    gx, gy = float(garden["pose"]["x"]), float(garden["pose"]["y"])
    pollen = [
        p
        for p in world.pieces.values()
        if p.type_id == "pollen"
        and not p.held_by
        and math.hypot(p.x - gx, p.y - gy) < 20.0
    ]
    assert len(pollen) >= 3
    xs = [p.x for p in pollen]
    assert max(xs) - min(xs) > 2.0
    half = float(world.field["fieldSizeIn"]["width"]) / 2.0
    for piece in pollen:
        assert abs(piece.x) < half - piece.radius
        assert abs(piece.y) < half - piece.radius


def test_flower_stacks_do_not_take_garden_jitter():
    world = _world()
    world.reset(seed=3, static_teammate=False, full_noise=True)
    flower = next(el for el in world.elements if el["id"] == "flower_1")
    fx, fy = float(flower["pose"]["x"]), float(flower["pose"]["y"])
    stacked = [
        p
        for p in world.pieces.values()
        if p.type_id == "pollen" and math.hypot(p.x - fx, p.y - fy) < 1.0
    ]
    assert len(stacked) == 4
    assert max(p.x for p in stacked) - min(p.x for p in stacked) < 1e-6
    assert max(p.y for p in stacked) - min(p.y for p in stacked) < 1e-6


def test_follower_brakes_to_a_stop_on_target():
    world = _world()
    world.reset(seed=0, static_teammate=False, full_noise=False)
    rs = world.actor()
    target = np.array([rs.body.x + 18.0, rs.body.y, rs.body.heading], dtype=np.float64)
    for _ in range(80):
        world.step(target, 1.0, 0)
    dist = math.hypot(rs.body.x - target[0], rs.body.y - target[1])
    speed = math.hypot(rs.body.vx, rs.body.vy)
    assert dist < 4.0
    assert speed < 4.0


def test_recorded_eval_episode_fields_four_robots():
    env = FTCAutoEnv(record=True)
    env.reset(seed=0, options={"static_teammate": True, "opponent_policy": "static"})
    assert set(env.world.robots) == {"red_0", "red_1", "blue_0", "blue_1"}
    snap = env.world.snapshot()
    assert {row["id"] for row in snap["robots"]} == {"red_0", "red_1", "blue_0", "blue_1"}
    env.close()


def test_teammate_none_does_not_spawn_static_red_1():
    env = FTCAutoEnv(record=False)
    env.reset(seed=0, options={"teammate_policy": "none"})
    assert "red_0" in env.world.robots
    assert "red_1" not in env.world.robots
    env.close()


def test_clipped_waypoint_does_not_farm_restricted_wall_still_shapes():
    env = FTCAutoEnv(record=False, static_teammate=False)
    env.reset(seed=2, options={"teammate_policy": "none", "opponent_policy": "none"})
    parsed = env._parse_action(
        {
            "target_pose": np.array([40.0, 0.0, 0.0], dtype=np.float32),
            "speed_frac": np.array([1.0], dtype=np.float32),
            "mechanism": 0,
        }
    )
    assert parsed is not None
    assert parsed["target_pose"][0] <= -float(env.world.robot_hx)
    _obs, _reward, _term, _trunc, info = env.step(
        {
            "target_pose": np.array([40.0, 0.0, 0.0], dtype=np.float32),
            "speed_frac": np.array([1.0], dtype=np.float32),
            "mechanism": 0,
        }
    )
    assert env.world.accumulators.get("restricted_entry") is not True
    assert info["true_score"] > -20

    rs = env.world.actor()
    _teleport_actor(env, -40.0, vx=0.0)
    _obs, _reward, _term, _trunc, _left_info = env.step(
        {
            "target_pose": np.array([rs.body.x, rs.body.y, rs.body.heading], dtype=np.float32),
            "speed_frac": np.array([0.2], dtype=np.float32),
            "mechanism": 0,
        }
    )
    rs = env.world.actor()
    assert not env.world._touching_perimeter(rs.body.x, rs.body.y, rs.body.heading)
    assert env._left_start_wall is True

    half = float(env.world.field["fieldSizeIn"]["width"]) / 2.0
    rs = env.world.actor()
    _teleport_actor(env, -half + 0.2, vx=-40.0)
    _obs, _reward, _term, _trunc, wall_info = env.step(
        {
            "target_pose": np.array([rs.body.x, rs.body.y, rs.body.heading], dtype=np.float32),
            "speed_frac": np.array([1.0], dtype=np.float32),
            "mechanism": 0,
        }
    )
    if env.world.wall_hit:
        assert wall_info["shaping"] <= -0.02

    rs = env.world.actor()
    rs.body.x, rs.body.y = 20.0, 0.0
    env.world.step(np.array([20.0, 0.0, 0.0]), 0.2, 0, end_phase=False)
    assert env.world.accumulators.get("restricted_entry") is True
    assert env.world.true_score <= -20
    env.close()
