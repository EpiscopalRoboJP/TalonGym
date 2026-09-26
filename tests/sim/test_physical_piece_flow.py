"""Causal intake/conveyor/flywheel forces, aero, and no-teleport MuJoCo flow."""

from __future__ import annotations

import math

import pytest

from talongym.sim.mujoco_backend import MujocoFieldBackend, available
from talongym.sim.physics import (
    Body,
    WorldStep,
    aerodynamic_wrench,
    gate_open_fraction,
    mechanism_piece_force,
    perimeter_walls,
)


def _path(**overrides):
    row = {
        "intakeActuatorId": "intake",
        "conveyorActuatorId": "conveyor",
        "flywheelActuatorId": "flywheel",
        "hoodActuatorId": "hood",
        "gateActuatorId": "gate",
        "storageSlots": [
            {"x": -4.5, "y": 0.0, "z": 4.0},
            {"x": 1.5, "y": 0.0, "z": 4.0},
        ],
        "intakePose": {"x": 8.5, "y": 0.0, "z": 2.0},
        "muzzlePose": {"x": 10.25, "y": 0.0, "z": 12.0, "pitchDeg": 52.0, "yawDeg": 0.0},
        "wheelRadiusIn": 2.0,
        "launchEfficiency": 0.235,
        "compressionIn": 0.25,
        "dragCoefficient": 0.47,
        "magnusCoefficient": 0.12,
    }
    row.update(overrides)
    return row


def _mech(actuators: dict, **overrides):
    row = {
        "piecePath": _path(),
        "chassis": {"lengthIn": 18.0, "widthIn": 18.0, "heightIn": 14.0},
        "actuators": actuators,
    }
    row.update(overrides)
    return row


def _flywheel_local():
    pitch = math.radians(52.0)
    dx, dz = math.cos(pitch), math.sin(pitch)
    return (
        10.25 - dx * 3.0,
        0.0,
        12.0 - dz * 3.0,
    )


def test_drag_opposes_velocity_and_magnus_lifts_backspin():
    fx, fy, fz, *_ = aerodynamic_wrench(180.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, drag_coefficient=0.47, magnus_coefficient=0.12)
    assert fx < 0.0
    assert abs(fy) < 1e-9
    lift_x, lift_y, lift_z, *_ = aerodynamic_wrench(
        180.0,
        0.0,
        0.0,
        0.0,
        -40.0,
        0.0,
        1.4,
        drag_coefficient=0.0,
        magnus_coefficient=0.12,
    )
    assert lift_z > 0.0
    assert abs(lift_x) < abs(lift_z)


def test_closed_gate_produces_no_launch_force():
    fx, fy, fz = _flywheel_local()
    closed = mechanism_piece_force(
        piece_local=(fx, fy, fz),
        piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4,
        piece_mass=0.1,
        mechanism=_mech(
            {
                "flywheel": {"rpm": 4500.0, "targetRpm": 4500.0, "jointId": "flywheel_joint"},
                "gate": {"position": 0.0, "command": -1.0, "travelLimit": [0.0, 75.0]},
                "hood": {"position": 52.0},
            }
        ),
    )
    along = closed.fx * math.cos(math.radians(52.0)) + closed.fz * math.sin(math.radians(52.0))
    assert along <= 1.0
    assert gate_open_fraction({"position": 0.0, "travelLimit": [0.0, 75.0]}) == 0.0


def test_open_gate_rpm_and_aim_control_launch_force():
    loc = _flywheel_local()
    high = mechanism_piece_force(
        piece_local=loc,
        piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4,
        piece_mass=0.1,
        mechanism=_mech(
            {
                "flywheel": {"rpm": 4500.0, "targetRpm": 4500.0},
                "gate": {"position": 75.0, "travelLimit": [0.0, 75.0]},
                "hood": {"position": 52.0},
            }
        ),
    )
    low = mechanism_piece_force(
        piece_local=loc,
        piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4,
        piece_mass=0.1,
        mechanism=_mech(
            {
                "flywheel": {"rpm": 400.0, "targetRpm": 4500.0},
                "gate": {"position": 75.0, "travelLimit": [0.0, 75.0]},
                "hood": {"position": 52.0},
            }
        ),
    )
    steep = mechanism_piece_force(
        piece_local=loc,
        piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4,
        piece_mass=0.1,
        mechanism=_mech(
            {
                "flywheel": {"rpm": 4500.0, "targetRpm": 4500.0},
                "gate": {"position": 75.0, "travelLimit": [0.0, 75.0]},
                "hood": {"position": 70.0},
            }
        ),
    )
    assert high.fx > 5.0
    assert high.fz > 0.0
    assert high.fx > low.fx
    assert high.fz > low.fz
    assert steep.fz / max(steep.fx, 1e-6) > high.fz / max(high.fx, 1e-6)
    assert high.flywheel_load_inch < 0.0


def test_flywheel_contact_uses_assembled_wheel_pose():
    actuators = {
        "flywheel": {"rpm": 4500.0, "targetRpm": 4500.0},
        "gate": {"position": 75.0, "travelLimit": [0.0, 75.0]},
        "hood": {"position": 52.0},
    }
    assembled = mechanism_piece_force(
        piece_local=(-3.0, 0.0, 12.0),
        piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4,
        piece_mass=0.1,
        mechanism=_mech(actuators, piecePath=_path(flywheelPose={"x": -3.0, "y": 0.0, "z": 12.0})),
    )
    template = mechanism_piece_force(
        piece_local=(-3.0, 0.0, 12.0),
        piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4,
        piece_mass=0.1,
        mechanism=_mech(actuators),
    )
    assert assembled.flywheel_load_inch < 0.0
    assert template.flywheel_load_inch == 0.0


def test_open_gate_does_not_feed_unselected_magazine_piece():
    loc = _flywheel_local()
    mechanism = _mech({
        "flywheel": {"rpm": 4500.0, "targetRpm": 4500.0},
        "gate": {"position": 75.0, "travelLimit": [0.0, 75.0]},
        "hood": {"position": 52.0},
    })
    selected = mechanism_piece_force(
        piece_local=loc, piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4, piece_mass=0.1, mechanism=mechanism,
        owned=True, feed_enabled=True,
    )
    waiting = mechanism_piece_force(
        piece_local=loc, piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4, piece_mass=0.1, mechanism=mechanism,
        owned=True, feed_enabled=False,
    )
    assert selected.flywheel_load_inch < 0.0
    assert waiting.flywheel_load_inch == 0.0
    assert waiting.fx < selected.fx


def test_intake_pulls_inward_only_when_spinning():
    spinning = mechanism_piece_force(
        piece_local=(9.5, 0.0, 2.0),
        piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4,
        piece_mass=0.1,
        mechanism=_mech({"intake": {"rpm": 300.0, "targetRpm": 300.0}}),
    )
    idle = mechanism_piece_force(
        piece_local=(9.5, 0.0, 2.0),
        piece_vel_local=(0.0, 0.0, 0.0),
        piece_radius=1.4,
        piece_mass=0.1,
        mechanism=_mech({"intake": {"rpm": 0.0, "targetRpm": 300.0}}),
    )
    assert spinning.fx < -1.0
    assert abs(idle.fx) < abs(spinning.fx) * 0.25


def _flow_xml() -> str:
    return """
    <mujoco model="piece_flow">
      <compiler angle="radian" inertiafromgeom="true"/>
      <option gravity="0 -386.0886 0" timestep="0.002" integrator="implicitfast"/>
      <worldbody>
        <geom name="floor" type="plane" size="80 80 1" pos="0 0 0" zaxis="0 1 0" group="0"/>
        <body name="red_0" pos="0 7 0">
          <joint name="red_0_sx" type="slide" axis="1 0 0" damping="2"/>
          <joint name="red_0_sz" type="slide" axis="0 0 1" damping="2"/>
          <joint name="red_0_yaw" type="hinge" axis="0 1 0" damping="0.4"/>
          <geom name="red_0_chassis" type="box" size="9 0.5 9" mass="15" group="1" contype="0" conaffinity="0"/>
          <body name="red_0_part_flywheel" pos="6.5 3.5 0">
            <joint name="red_0_joint_flywheel_joint" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="2 1" mass="0.65" group="1" contype="0" conaffinity="0"/>
          </body>
          <body name="red_0_part_release_gate" pos="9.5 3.25 0">
            <joint name="red_0_joint_gate_joint" type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.25 1.5 2.5" mass="0.12" group="1" contype="0" conaffinity="0"/>
          </body>
          <body name="red_0_part_intake_roller" pos="8 -5 0">
            <joint name="red_0_joint_intake_joint" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="1 6.5" mass="0.35" group="1" contype="0" conaffinity="0"/>
          </body>
        </body>
        <body name="gp_pollen_00" pos="0 80 0">
          <freejoint name="gp_pollen_00_free"/>
          <geom name="gp_pollen_00_geom" type="sphere" size="1.4" mass="0.1" group="2" contype="8" conaffinity="13"/>
        </body>
      </worldbody>
    </mujoco>
    """


def _backend() -> MujocoFieldBackend:
    return MujocoFieldBackend(_flow_xml(), 72.0, 72.0, robot_hz=7.0, slot_plan={"pollen": 1}, floor_y=0.0)


def _step(backend, robot: Body, piece: Body, mechanism: dict, dt: float = 0.02, **kwargs):
    return backend.step_world(
        WorldStep(
            [robot],
            [piece],
            [],
            perimeter_walls(72, 72),
            max_accel=kwargs.get("max_accel", 30.0),
            max_ang_accel=kwargs.get("max_ang_accel", 1.0),
            robot_mechanism_states={"red_0": mechanism},
            piece_owners=kwargs.get("piece_owners", {}),
            piece_feed_targets={"red_0": piece.id},
        ),
        dt,
    )


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mujoco_ignores_kick_teleport_after_spawn():
    backend = _backend()
    robot = Body("red_0", 0.0, 0.0, 0.0, mass=15.0, hx=9.0, hy=9.0)
    piece = Body("p0", 0.0, 0.0, 0.0, kind="circle", radius=1.4, mass=0.1, z=10.0, type_id="pollen")
    _step(backend, robot, piece, _mech({}))
    spawned_x = piece.x
    piece.x = 40.0
    piece.kick = True
    piece.vx = 400.0
    _step(backend, robot, piece, _mech({}))
    assert abs(piece.x - 40.0) > 10.0
    assert abs(piece.x - spawned_x) < 8.0
    assert abs(piece.vx) < 80.0


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mujoco_intake_moves_piece_inward():
    backend = _backend()
    robot = Body("red_0", 0.0, 0.0, 0.0, mass=15.0, hx=9.0, hy=9.0)
    piece = Body("p0", 11.0, 0.0, 0.0, kind="circle", radius=1.4, mass=0.1, z=2.0, type_id="pollen")
    mech = _mech({"intake": {"rpm": 300.0, "targetRpm": 300.0, "jointId": "intake_joint"}})
    start = piece.x
    for _ in range(25):
        _step(backend, robot, piece, mech)
    assert piece.x < start - 0.4


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mujoco_airborne_drag_slows_piece():
    backend = _backend()
    robot = Body("red_0", 40.0, 0.0, 0.0, mass=15.0, hx=9.0, hy=9.0)
    piece = Body(
        "p0",
        0.0,
        0.0,
        0.0,
        vx=220.0,
        kind="circle",
        radius=1.4,
        mass=0.1,
        z=24.0,
        type_id="pollen",
        wx=0.0,
        wy=-50.0,
        wz=0.0,
    )
    mech = _mech({})
    _step(backend, robot, piece, mech, dt=0.02)
    first = piece.vx
    for _ in range(20):
        _step(backend, robot, piece, mech)
    assert piece.vx < first
    assert piece.vx < 210.0


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mujoco_open_gate_launches_and_closed_gate_does_not():
    loc_x, _ly, loc_z = _flywheel_local()
    open_backend = _backend()
    closed_backend = _backend()
    robot = Body("red_0", 0.0, 0.0, 0.0, mass=15.0, hx=9.0, hy=9.0)
    launched = Body("p0", loc_x, 0.0, 0.0, kind="circle", radius=1.4, mass=0.1, z=loc_z, type_id="pollen")
    blocked = Body("p0", loc_x, 0.0, 0.0, kind="circle", radius=1.4, mass=0.1, z=loc_z, type_id="pollen")
    open_mech = _mech(
        {
            "flywheel": {"rpm": 4500.0, "targetRpm": 4500.0, "jointId": "flywheel_joint"},
            "gate": {"position": 75.0, "travelLimit": [0.0, 75.0], "jointId": "gate_joint"},
            "hood": {"position": 52.0},
        }
    )
    closed_mech = _mech(
        {
            "flywheel": {"rpm": 4500.0, "targetRpm": 4500.0, "jointId": "flywheel_joint"},
            "gate": {"position": 0.0, "travelLimit": [0.0, 75.0], "jointId": "gate_joint"},
            "hood": {"position": 52.0},
        }
    )
    for _ in range(6):
        _step(open_backend, robot, launched, open_mech, max_accel=4.0)
        _step(closed_backend, robot, blocked, closed_mech, max_accel=4.0)
    launch_speed = math.sqrt(launched.vx ** 2 + launched.vz ** 2)
    blocked_speed = math.sqrt(blocked.vx ** 2 + blocked.vz ** 2)
    assert launched.vx > blocked.vx + 8.0
    assert launched.vz > blocked.vz
    assert launch_speed > blocked_speed
    assert launched.vx > 8.0


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mujoco_launch_applies_recoil_to_chassis():
    loc_x, _ly, loc_z = _flywheel_local()
    backend = _backend()
    robot = Body("red_0", 0.0, 0.0, 0.0, mass=15.0, hx=9.0, hy=9.0, vx=0.0)
    piece = Body("p0", loc_x, 0.0, 0.0, kind="circle", radius=1.4, mass=0.1, z=loc_z, type_id="pollen")
    mech = _mech(
        {
            "flywheel": {"rpm": 4500.0, "targetRpm": 4500.0, "jointId": "flywheel_joint"},
            "gate": {"position": 75.0, "travelLimit": [0.0, 75.0], "jointId": "gate_joint"},
            "hood": {"position": 52.0},
        }
    )
    _step(backend, robot, piece, mech, max_accel=1.0, dt=0.02)
    assert piece.vx > 1.0
    assert robot.vx < 0.0


@pytest.mark.skipif(not available(), reason="mujoco extra not installed")
def test_mujoco_exposes_mechanism_transforms_and_joint_state():
    backend = _backend()
    robot = Body("red_0", 0.0, 0.0, 0.0, mass=15.0, hx=9.0, hy=9.0)
    piece = Body("p0", 4.0, 0.0, 0.0, kind="circle", radius=1.4, mass=0.1, z=4.0, type_id="pollen")
    mech = _mech(
        {
            "flywheel": {
                "rpm": 1200.0,
                "targetRpm": 4500.0,
                "jointId": "flywheel_joint",
                "kind": "velocity_motor",
                "torqueNm": 0.2,
            },
            "gate": {"position": 10.0, "kind": "servo", "jointId": "gate_joint", "travelLimit": [0.0, 75.0]},
        }
    )
    _step(backend, robot, piece, mech)
    joints = backend.robot_mechanism_joint_states()
    assert "red_0" in joints
    assert "flywheel_joint" in joints["red_0"]
    assert "positionRad" in joints["red_0"]["flywheel_joint"]
    assert "velocityRadS" in joints["red_0"]["flywheel_joint"]
    parts = {row["id"]: row for row in backend.robot_mechanism_transforms()["red_0"]}
    assert "flywheel" in parts
    assert "vx" in parts["flywheel"]
    assert parts["flywheel"]["z"] > 0.0
