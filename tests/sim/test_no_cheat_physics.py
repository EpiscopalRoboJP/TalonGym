"""Anti-cheat: mesh seasons score only by physical occupancy, not teleports."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from talongym.presets.loader import load_bundle, load_preset
from talongym.robot.mechanisms import muzzle_velocity
from talongym.sim.geometry import deg_to_rad, point_in_volume
from talongym.sim.world import World


def _world(seed: int = 0) -> World:
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=seed, allow_missing_mesh=True)
    world.reset(seed=seed, static_teammate=False, full_noise=False)
    return world


def _cell(world: World):
    return next(el for el in world.elements if el["id"] == "red_cell_up")


def _in_red_cell(world: World, piece) -> bool:
    el = _cell(world)
    return point_in_volume(el, world.element_shapes[el["id"]], piece.x, piece.y, piece.z, piece.radius)


def _park_away_from_hive(world: World) -> np.ndarray:
    rs = world.actor()
    rs.body.x, rs.body.y, rs.body.heading = -50.0, -50.0, 0.0
    rs.body.vx = rs.body.vy = rs.body.omega = 0.0
    slots = list((world.robot.get("piecePath") or {}).get("storageSlots") or [])
    for index, pid in enumerate(rs.held):
        slot = slots[index] if index < len(slots) else {"x": 0.0, "y": 0.0, "z": 4.0}
        piece = world.pieces[pid]
        piece.x = rs.body.x + float(slot.get("x") or 0.0)
        piece.y = rs.body.y + float(slot.get("y") or 0.0)
        piece.z = float(slot.get("z") or 4.0)
        piece.vx = piece.vy = piece.vz = 0.0
        piece.held_by = rs.body.id
        piece.in_flight = False
    return np.array([rs.body.x, rs.body.y, rs.body.heading], dtype=np.float64)


def test_scoring_preset_has_no_synthetic_piece_ops():
    scoring = load_preset("scoring", "biobuzz_2026_scoring_v1")
    kinds = [action.get("kind") for node in scoring.get("nodes") or [] for action in node.get("actions") or []]
    assert "transferPiece" not in kinds
    assert "despawnPiece" not in kinds
    assert not any("nearest" in str(action.get("kind") or "").lower() for node in scoring.get("nodes") or [] for action in node.get("actions") or [])


def test_mesh_season_cannot_disable_ballistic_or_use_transfer():
    world = _world()
    assert world.needs_mesh() is True
    world.reset(seed=0, static_teammate=False, ballistic_launch=False)
    assert world.ballistic_launch is True
    piece = next(p for p in world.pieces.values() if not p.held_by and p.type_id == "pollen")
    world.pending_piece_ops.append(("transfer", piece.id, "red_cell_up"))
    with pytest.raises(RuntimeError, match="transferPiece is prohibited"):
        world._apply_piece_ops()


def test_score_verb_does_not_teleport_nearest_piece_into_volume():
    world = _world()
    pose = _park_away_from_hive(world)
    cell = _cell(world)
    piece = next(p for p in world.pieces.values() if p.type_id == "pollen" and not p.held_by)
    piece.held_by = None
    piece.scored = False
    piece.in_flight = False
    piece.x = float(cell["pose"]["x"])
    piece.y = float(cell["pose"]["y"])
    piece.z = 4.0
    piece.vx = piece.vy = piece.vz = 0.0
    assert not _in_red_cell(world, piece)
    launched0 = int(world.accumulators.get("launched_count") or 0)
    for _ in range(20):
        world.step(pose, 0.2, 2)
    moved = world.pieces[piece.id]
    assert not _in_red_cell(world, moved)
    assert abs(moved.z - 4.0) < 3.0
    assert int(world.accumulators.get("launched_count") or 0) == launched0


def test_waypoint_to_hive_does_not_synthesize_a_score():
    world = _world()
    rs = world.actor()
    for pid in list(rs.held):
        piece = world.pieces[pid]
        piece.held_by = None
        rs.held.remove(pid)
    target = np.array(
        [float(_cell(world)["pose"]["x"]), float(_cell(world)["pose"]["y"]), math.pi / 2],
        dtype=np.float64,
    )
    launched0 = int(world.accumulators.get("launched_count") or 0)
    score0 = float(world.true_score)
    for _ in range(25):
        world.step(target, 1.0, 2)
    assert int(world.accumulators.get("launched_count") or 0) == launched0
    assert float(world.true_score) == score0


def test_empty_magazine_cannot_score():
    world = _world()
    pose = _park_away_from_hive(world)
    rs = world.actor()
    for pid in list(rs.held):
        world.pieces[pid].held_by = None
        world.pieces[pid].x = rs.body.x - 20.0
        world.pieces[pid].y = rs.body.y
        rs.held.remove(pid)
    assert rs.held == []
    launched0 = int(world.accumulators.get("launched_count") or 0)
    for _ in range(30):
        world.step(pose, 0.2, 2)
    assert rs.held == []
    assert int(world.accumulators.get("launched_count") or 0) == launched0
    assert not any(p.in_flight for p in world.pieces.values() if p.type_id == "pollen")


def test_below_speed_does_not_open_gate_or_fsm_launch():
    world = _world()
    pose = _park_away_from_hive(world)
    rs = world.actor()
    assert rs.mechanism is not None
    flywheel = rs.mechanism.actuators["flywheel"]
    target = float(flywheel.config["targetRpm"])
    held0 = list(rs.held)
    assert held0
    commands = world._mechanism_commands(rs, "score", None)
    assert commands[str(world.robot["piecePath"]["gateActuatorId"])] == pytest.approx(-1.0)
    for _ in range(8):
        world.step(pose, 0.2, 2)
    rpm = abs(flywheel.state.velocity_rad_s) * 60.0 / (2.0 * math.pi)
    assert rpm < 0.9 * target
    assert list(rs.held) == held0
    assert all(not world.pieces[pid].in_flight for pid in held0)
    assert all(not _in_red_cell(world, world.pieces[pid]) for pid in held0)
    gate = rs.mechanism.actuators["gate"]
    assert gate.state.command < 0.0


def test_blocked_gate_keeps_preload_in_envelope_until_flywheel_is_up():
    world = _world()
    pose = _park_away_from_hive(world)
    rs = world.actor()
    assert rs.mechanism is not None
    gate_id = str(world.robot["piecePath"]["gateActuatorId"])
    flywheel = rs.mechanism.actuators["flywheel"]
    cold = world._mechanism_commands(rs, "score", None)
    assert cold[gate_id] == pytest.approx(-1.0)
    flywheel.state.velocity_rad_s = float(flywheel.config["targetRpm"]) * (2.0 * math.pi / 60.0)
    hot = world._mechanism_commands(rs, "score", None)
    assert hot[gate_id] > 0.5
    flywheel.state.velocity_rad_s = 0.0
    held0 = list(rs.held)
    launched0 = int(world.accumulators.get("launched_count") or 0)
    for _ in range(10):
        world.step(pose, 0.0, 2)
    assert list(rs.held) == held0
    assert rs.mechanism.actuators["gate"].state.command < 0.0
    assert int(world.accumulators.get("launched_count") or 0) == launched0
    for pid in held0:
        piece = world.pieces[pid]
        assert piece.held_by == rs.body.id
        assert not piece.in_flight
        assert not _in_red_cell(world, piece)


def test_wrong_aim_launch_does_not_enter_up_cell():
    world = _world()
    pose = _park_away_from_hive(world)
    rs = world.actor()
    pid = rs.held[0]
    piece = world.pieces[pid]
    rs.held.remove(pid)
    piece.held_by = None
    launcher = copy.deepcopy(world.launchers[0])
    launcher["poseOnRobot"] = {
        **dict(launcher.get("poseOnRobot") or {}),
        "headingDeg": 180.0,
        "pitchDeg": 0.0,
    }
    world._launch_ballistic(rs, piece, launcher)
    yaw = rs.body.heading + deg_to_rad(180.0)
    vx, vy, vz = muzzle_velocity(float(launcher["muzzleSpeedInPerS"]), yaw, 0.0)
    assert piece.vx == pytest.approx(vx + rs.body.vx, abs=1e-6)
    assert piece.vy == pytest.approx(vy + rs.body.vy, abs=1e-6)
    assert piece.vz == pytest.approx(vz, abs=1e-6)
    assert piece.vx < 0.0
    for _ in range(15):
        world.step(pose, 0.0, 0)
    assert not _in_red_cell(world, world.pieces[pid])
    assert int(world.accumulators.get("launched_count") or 0) == 0


def test_physical_score_path_does_not_pop_held_pieces_like_legacy_fsm():
    world = _world()
    pose = _park_away_from_hive(world)
    rs = world.actor()
    assert rs.mechanism is not None
    flywheel = rs.mechanism.actuators["flywheel"]
    flywheel.state.velocity_rad_s = float(flywheel.config["targetRpm"]) * (2.0 * math.pi / 60.0)
    held0 = list(rs.held)
    world._mechanisms(rs, "score", 0.05, [])
    assert list(rs.held) == held0
    assert all(not world.pieces[pid].in_flight for pid in held0)
    world.step(pose, 0.0, 2)
    assert not any(_in_red_cell(world, world.pieces[pid]) for pid in held0)
