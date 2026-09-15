"""Physical fidelity: four-piece preload, 2 ms MJCF, articulated robot bodies."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from talongym.assets.mjcf_field import build_field_mjcf
from talongym.paths import ASSETS_DIR
from talongym.presets.loader import LoadedPresets, load_bundle, load_preset
from talongym.robot.contract import compile_robot_preset
from talongym.sim.mujoco_backend import available
from talongym.sim.world import World
from talongym.training.curriculum import curriculum_unlocks


def _bundle():
    return load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")


def _idle_world(seed: int = 0, **reset_kw) -> World:
    world = World(_bundle(), seed=seed, allow_missing_mesh=True)
    world.reset(seed=seed, static_teammate=False, full_noise=False, **reset_kw)
    return world


def test_training_bundle_compiles_physical_robot():
    bundle = _bundle()
    compiled = compile_robot_preset(bundle.robot, competitive=True)
    assert compiled.physical is True
    assert len(compiled.actuator_ids) >= 4
    assert int((bundle.robot.get("mechanisms") or {}).get("capacity") or 0) == 4
    assert int(bundle.field.get("requiredPreloadCount") or 0) == 4


def test_competitive_biobuzz_curriculum_does_not_unlock_scripted_launch():
    for name in (
        "biobuzz_auto_lightweight",
        "biobuzz_auto_easy",
        "biobuzz_auto_workstation",
        "biobuzz_auto_cloud",
    ):
        training = load_preset("training", name)
        for frac in (0.0, 0.3, 0.6, 0.99):
            unlocks = curriculum_unlocks(training, frac)
            assert "scripted_launch" not in unlocks, (name, frac, unlocks)


def test_four_piece_preload_is_physically_active_in_storage_slots():
    world = _idle_world()
    rs = world.actor()
    slots = list((world.robot.get("piecePath") or {}).get("storageSlots") or [])
    assert len(rs.held) == 4
    assert len(slots) >= 4
    held = [world.pieces[pid] for pid in rs.held]
    assert all(piece.held_by == rs.body.id for piece in held)
    assert all(piece.type_id == "pollen" for piece in held)
    assert all(not piece.scored and not piece.in_flight for piece in held)
    xs = [piece.x for piece in held]
    ys = [piece.y for piece in held]
    assert max(xs) - min(xs) > 4.0
    assert max(ys) - min(ys) < 3.0
    cos_h, sin_h = math.cos(rs.body.heading), math.sin(rs.body.heading)
    for piece, slot in zip(held, slots[:4], strict=True):
        local_x = cos_h * (piece.x - rs.body.x) + sin_h * (piece.y - rs.body.y)
        local_y = -sin_h * (piece.x - rs.body.x) + cos_h * (piece.y - rs.body.y)
        assert local_x == pytest.approx(float(slot["x"]), abs=0.6)
        assert local_y == pytest.approx(float(slot["y"]), abs=0.6)
        assert piece.z == pytest.approx(float(slot["z"]), abs=0.6)


def test_four_piece_preload_stays_held_through_idle_steps():
    world = _idle_world()
    rs = world.actor()
    held0 = list(rs.held)
    pose = np.array([rs.body.x, rs.body.y, rs.body.heading], dtype=np.float64)
    for _ in range(8):
        world.step(pose, 0.0, 0)
    assert list(world.actor().held) == held0
    assert all(not world.pieces[pid].scored for pid in world.actor().held)


def test_required_preload_count_is_enforced():
    bundle = _bundle()
    field = copy.deepcopy(bundle.field)
    field["requiredPreloadCount"] = 3
    broken = LoadedPresets(field=field, robot=bundle.robot, scoring=bundle.scoring, training=bundle.training)
    world = World(broken, seed=0, allow_missing_mesh=True)
    with pytest.raises(ValueError, match="requires exactly 3 preload pieces"):
        world.reset(seed=0, static_teammate=False)
    robot = copy.deepcopy(bundle.robot)
    robot["mechanisms"] = dict(robot.get("mechanisms") or {})
    robot["mechanisms"]["capacity"] = 3
    small = LoadedPresets(field=bundle.field, robot=robot, scoring=bundle.scoring, training=bundle.training)
    world = World(small, seed=0, allow_missing_mesh=True)
    with pytest.raises(ValueError, match="cannot hold required 4-piece preload"):
        world.reset(seed=0, static_teammate=False)


def test_mjcf_timestep_is_two_milliseconds():
    field = load_preset("field", "biobuzz_2026_field_v1")
    robot = load_preset("robot", "mecanum_biobuzz_4cap")
    xml = build_field_mjcf(field, robot=robot, n_robots=1).xml
    assert 'timestep="0.002"' in xml
    committed = (ASSETS_DIR / field["collisionAsset"]).read_text(encoding="utf-8")
    assert 'timestep="0.002"' in committed


def test_mjcf_emits_articulated_robot_bodies_and_joints():
    field = load_preset("field", "biobuzz_2026_field_v1")
    robot = load_preset("robot", "mecanum_biobuzz_4cap")
    xml = build_field_mjcf(field, robot=robot, n_robots=1).xml
    for part in ("intake_roller", "conveyor_roller", "flywheel", "hood", "release_gate"):
        assert f'red_0_part_{part}' in xml
    for joint in ("intake_joint", "conveyor_joint", "flywheel_joint", "hood_joint", "gate_joint"):
        assert f'red_0_joint_{joint}' in xml
    assert 'name="red_0_sx"' in xml
    assert xml.count("<joint") >= 8


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mujoco_runtime_uses_2ms_and_articulated_dofs():
    world = World(_bundle(), seed=0)
    world.reset(seed=0, static_teammate=False, full_noise=False)
    assert world.backend.name == "mujoco_field"
    assert float(world.backend._mj.opt.timestep) == pytest.approx(0.002, abs=1e-9)
    joints = world.backend.robot_mechanism_joint_states()
    assert "red_0" in joints
    names = set(joints["red_0"])
    assert "flywheel_joint" in names
    assert "hood_joint" in names
    assert "gate_joint" in names
    snap = world.snapshot()
    assert snap["physicalPieces"] is True
    actor_row = next(robot for robot in snap["robots"] if robot["id"] == "red_0")
    assert actor_row["parts"]
    assert actor_row.get("batteryVoltageV", 0) > 0
    assert "flywheel" in (actor_row.get("actuators") or {})
    held = [piece for piece in snap["pieces"] if piece.get("heldBy") == "red_0"]
    assert len(held) == 4
    assert all(isinstance(piece.get("storedSlot"), int) for piece in held)
    assert "mechanismCommands" in snap
    rs = world.actor()
    assert rs.mechanism is not None
    pose = np.array([rs.body.x, rs.body.y, rs.body.heading], dtype=np.float64)
    held0 = list(rs.held)
    for _ in range(6):
        world.step(pose, 0.0, 0)
    assert list(world.actor().held) == held0
    assert all(world.pieces[pid].held_by == "red_0" for pid in held0)
