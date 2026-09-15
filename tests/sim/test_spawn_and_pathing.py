"""Loose-piece spawn clamp, A* follower brake-to-stop, launch spots, 4-robot eval."""

from __future__ import annotations

import math

import numpy as np

from talongym.env.ftc_auto import FTCAutoEnv
from talongym.presets.loader import load_bundle
from talongym.sim.world import World


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
    env.reset(seed=0)
    assert set(env.world.robots) == {"red_0", "red_1", "blue_0", "blue_1"}
    snap = env.world.snapshot()
    assert {row["id"] for row in snap["robots"]} == {"red_0", "red_1", "blue_0", "blue_1"}
    env.close()
