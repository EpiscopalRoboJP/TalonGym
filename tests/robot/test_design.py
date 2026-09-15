import math

from talongym.presets.loader import load_bundle, load_preset, validate_document
from talongym.robot.mechanisms import muzzle_velocity, piece_in_intake
from talongym.sim.geometry import deg_to_rad
from talongym.sim.world import World


def test_biobuzz_robot_schema_includes_intakes_and_launchers():
    robot = load_preset("robot", "mecanum_biobuzz_4cap")
    assert validate_document("robot", robot) == []
    assert robot["intakes"][0]["id"] == "front_intake"
    assert robot["launchers"][0]["muzzleSpeedInPerS"] == 220


def test_intake_misses_beside_mouth():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0, allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False)
    rs = world.actor()
    rs.body.x, rs.body.y, rs.body.heading = -50.0, -50.0, 0.0
    rs.body.vx = rs.body.vy = 0.0
    piece = next(p for p in world.pieces.values() if not p.scored)
    for other in world.pieces.values():
        if other.id == piece.id:
            continue
        other.scored = True
        other.held_by = None
    piece.held_by = None
    rs.held.clear()
    piece.x, piece.y, piece.z = -50.0, -36.0, 1.4
    intake = bundle.robot["intakes"][0]
    assert not piece_in_intake(piece.x, piece.y, piece.z, piece.radius, rs.body.x, rs.body.y, 0.0, intake)
    events: list = []
    for _ in range(20):
        world._mechanisms(rs, "intake", 0.05, events)
    assert piece.id not in rs.held
    piece.x, piece.y, piece.z = -38.0, -50.0, 1.4
    piece.vz = 0.0
    assert piece_in_intake(piece.x, piece.y, piece.z, piece.radius, -50.0, -50.0, 0.0, intake)
    for _ in range(20):
        world._mechanisms(rs, "intake", 0.05, events)
    assert piece.id not in rs.held


def test_hood_muzzle_helper_matches_trig():
    vx, vy, vz = muzzle_velocity(220.0, 0.0, deg_to_rad(52.0))
    assert math.isclose(vx, 220.0 * math.cos(deg_to_rad(52.0)), rel_tol=1e-9)
    assert math.isclose(vy, 0.0, abs_tol=1e-9)
    assert math.isclose(vz, 220.0 * math.sin(deg_to_rad(52.0)), rel_tol=1e-9)


def test_muzzle_velocity_matches_hood():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    launcher = dict(bundle.robot["launchers"][0])
    launcher["spinupTimeS"] = 0.0
    launcher["cycleTimeS"] = 0.05
    bundle.robot["launchers"] = [launcher]
    world = World(bundle, seed=0, allow_missing_mesh=True)
    world.reset(seed=0, static_teammate=False, ballistic_launch=True)
    rs = world.actor()
    rs.body.x, rs.body.y, rs.body.heading = -40.0, 0.0, 0.0
    rs.body.vx = rs.body.vy = 0.0
    rs.held.clear()
    piece = next(p for p in world.pieces.values() if p.type_id == "pollen")
    for other in world.pieces.values():
        if other.id != piece.id:
            other.scored = True
            other.held_by = None
    piece.held_by = rs.body.id
    rs.held.append(piece.id)
    world._launch_ballistic(rs, piece, launcher)
    pose = launcher["poseOnRobot"]
    speed = float(launcher["muzzleSpeedInPerS"])
    yaw = rs.body.heading + deg_to_rad(float(pose["headingDeg"]))
    pitch = deg_to_rad(float(pose["pitchDeg"]))
    evx, evy, evz = muzzle_velocity(speed, yaw, pitch)
    assert piece.ballistic
    assert math.isclose(piece.vx, evx, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(piece.vy, evy, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(piece.vz, evz, rel_tol=1e-6, abs_tol=1e-6)
    assert piece.x > rs.body.x
