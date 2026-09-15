"""Physical flower cups retain stacked POLLEN without skip-physics staging."""

from __future__ import annotations

import math

import numpy as np
import pytest

from talongym.assets.mjcf_field import _flower_cup_proxy_geoms, apply_flower_cup_proxies
from talongym.presets.loader import load_bundle, load_preset
from talongym.sim.geometry import point_in_volume
from talongym.sim.world import World


def _world(seed: int = 0) -> World:
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=seed, allow_missing_mesh=True)
    world.reset(seed=seed, static_teammate=False, full_noise=False)
    return world


def _flowers(world: World):
    return [el for el in world.elements if el.get("type") == "flower" or "flower" in (el.get("tags") or [])]


def _flower_pollen(world: World) -> list:
    flowers = _flowers(world)
    out = []
    for piece in world.pieces.values():
        if piece.type_id != "pollen" or piece.held_by:
            continue
        if any(
            point_in_volume(el, world.element_shapes[el["id"]], piece.x, piece.y, piece.z, piece.radius)
            or math.hypot(piece.x - float(el["pose"]["x"]), piece.y - float(el["pose"]["y"])) < 4.0
            for el in flowers
        ):
            out.append(piece)
    return out


def test_flower_cup_proxies_have_floor_walls_and_one_gap():
    field = load_preset("field", "biobuzz_2026_field_v1")
    geoms = _flower_cup_proxy_geoms(field)
    joined = "\n".join(geoms)
    assert joined.count("_cup_floor") == 4
    assert "flower_1_cup_y+_sill" in joined
    assert "flower_2_cup_x-_sill" in joined
    assert "flower_3_cup_y-_sill" in joined
    assert "flower_4_cup_x+_sill" in joined
    for fid in ("flower_1", "flower_2", "flower_3", "flower_4"):
        assert f"{fid}_cup_floor" in joined
        closed = sum(1 for key in ("x-", "x+", "y-", "y+") if f'name="{fid}_cup_{key}" ' in joined)
        assert closed == 3
    patched = apply_flower_cup_proxies("<mujoco><worldbody></worldbody></mujoco>", field)
    assert "flower_1_cup_floor" in patched
    assert apply_flower_cup_proxies(patched, field) == patched


@pytest.mark.require_mesh
def test_flower_pollen_stays_stacked_when_idle():
    world = _world()
    rs = world.actor()
    hold = np.array([rs.body.x, rs.body.y, rs.body.heading], dtype=np.float64)
    before = {p.id: (p.x, p.y, p.z) for p in _flower_pollen(world)}
    assert len(before) == 16
    flowers = _flowers(world)
    for _ in range(40):
        world.step(hold, 0.0, 0)
    stacked_z = []
    for piece in world.pieces.values():
        if piece.id not in before:
            continue
        flower = min(
            flowers,
            key=lambda el: math.hypot(piece.x - float(el["pose"]["x"]), piece.y - float(el["pose"]["y"])),
        )
        shape = world.element_shapes[flower["id"]]
        assert point_in_volume(flower, shape, piece.x, piece.y, piece.z, piece.radius) or math.hypot(
            piece.x - float(flower["pose"]["x"]),
            piece.y - float(flower["pose"]["y"]),
        ) < 3.5
        stacked_z.append(piece.z)
        assert piece.z > 0.8
    assert max(stacked_z) > 7.0
    assert min(stacked_z) < 3.0


@pytest.mark.require_mesh
def test_intake_still_removes_flower_pollen():
    world = _world()
    rs = world.actor()
    for pid in list(rs.held):
        piece = world.pieces[pid]
        piece.held_by = None
        piece.x = rs.body.x - 24.0
        piece.y = rs.body.y
        rs.held.remove(pid)
    pollen = min(_flower_pollen(world), key=lambda p: p.z)
    heading = rs.body.heading
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    pollen.x = rs.body.x + cos_h * 5.0
    pollen.y = rs.body.y + sin_h * 5.0
    pollen.z = 2.0
    pollen.vx = pollen.vy = pollen.vz = 0.0
    pollen.held_by = None
    pose = np.array([rs.body.x, rs.body.y, heading], dtype=np.float64)
    for _ in range(40):
        world.step(pose, 0.2, 1)
    assert pollen.id in rs.held
    assert world.pieces[pollen.id].held_by == rs.body.id
