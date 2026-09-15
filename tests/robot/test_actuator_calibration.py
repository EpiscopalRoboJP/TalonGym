"""Actuator calibration: spin-up, current limit, sag, rate, latency, seeds."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from talongym.presets.loader import load_bundle, load_preset
from talongym.robot.dynamics import RPM_TO_RAD_S, ActuatorModel, MechanismDynamics
from talongym.sim.world import World

DT = 0.002


def _biobuzz_robot() -> dict:
    return copy.deepcopy(load_preset("robot", "mecanum_biobuzz_4cap"))


def _stalled_motor(**overrides) -> dict:
    row = {
        "id": "stall",
        "kind": "velocity_motor",
        "motor": {
            "nominalVoltageV": 12,
            "freeSpeedRpm": 6000,
            "stallTorqueNm": 0.5,
            "stallCurrentA": 20,
            "freeCurrentA": 0.6,
        },
        "gearRatio": 1,
        "efficiency": 1.0,
        "loadInertiaKgM2": 8.0,
        "rotorInertiaKgM2": 0.0,
        "currentLimitA": 5,
        "controllerLatencyMs": 0,
        "commandRatePerS": 1e6,
        "targetRpm": 4500,
        "viscousFrictionNmPerRadS": 0.0,
        "coulombFrictionNm": 0.0,
    }
    row.update(overrides)
    return row


def test_flywheel_spinup_is_not_instant_and_approaches_target():
    dynamics = MechanismDynamics(_biobuzz_robot())
    flywheel = dynamics.actuators["flywheel"]
    target = float(flywheel.config["targetRpm"])
    early_rpm = 0.0
    for step in range(25):
        dynamics.step({"flywheel": 1.0}, DT)
        if step == 9:
            early_rpm = flywheel.state.velocity_rad_s / RPM_TO_RAD_S
    assert early_rpm < 0.35 * target
    for _ in range(700):
        dynamics.step({"flywheel": 1.0}, DT)
    late_rpm = abs(flywheel.state.velocity_rad_s) / RPM_TO_RAD_S
    assert late_rpm > early_rpm + 200
    assert late_rpm > 0.4 * target
    assert late_rpm < 1.25 * float(flywheel.config["motor"]["freeSpeedRpm"])


def test_current_limit_clips_stalled_motor():
    limited = ActuatorModel(_stalled_motor(currentLimitA=5))
    unlimited = ActuatorModel(_stalled_motor(currentLimitA=40))
    limited.queue_command(1.0, 0.0)
    unlimited.queue_command(1.0, 0.0)
    limited.advance_command(0.05, DT)
    unlimited.advance_command(0.05, DT)
    limited.integrate(12.0, DT)
    unlimited.integrate(12.0, DT)
    assert limited.state.current_a <= 5.0 + 1e-9
    assert unlimited.state.current_a > 5.0
    dynamics = MechanismDynamics(_biobuzz_robot())
    for _ in range(40):
        dynamics.step({"flywheel": 1.0, "intake": 1.0, "conveyor": 1.0}, DT)
        for ident, actuator in dynamics.actuators.items():
            limit = float(actuator.config["currentLimitA"])
            assert abs(actuator.state.current_a) <= limit + 1e-6, ident


def test_shared_drivetrain_current_sags_battery_and_slows_flywheel():
    robot = _biobuzz_robot()
    idle = MechanismDynamics(copy.deepcopy(robot))
    loaded = MechanismDynamics(copy.deepcopy(robot))
    commands = {"flywheel": 1.0, "intake": 1.0, "conveyor": 1.0}
    for _ in range(250):
        idle.step(commands, DT, drivetrain_current_a=0.0)
        loaded.step(commands, DT, drivetrain_current_a=80.0)
    assert loaded.power.voltage_v < idle.power.voltage_v - 0.4
    assert loaded.power.current_a > idle.power.current_a
    idle_rpm = abs(idle.actuators["flywheel"].state.velocity_rad_s)
    loaded_rpm = abs(loaded.actuators["flywheel"].state.velocity_rad_s)
    assert loaded_rpm < idle_rpm
    assert loaded.power.state_of_charge < idle.power.state_of_charge


def test_command_latency_delays_and_rate_limit_ramps():
    robot = _biobuzz_robot()
    dynamics = MechanismDynamics(robot)
    gate = dynamics.actuators["gate"]
    latency_s = 0.001 * float(gate.config["controllerLatencyMs"])
    rate = float(gate.config["commandRatePerS"])
    assert latency_s >= 0.05
    assert rate < 10.0
    steps_before = max(1, int((0.5 * latency_s) / DT))
    for _ in range(steps_before):
        dynamics.step({"gate": 1.0}, DT)
    assert gate.state.delayed_command == pytest.approx(0.0, abs=1e-9)
    assert gate.state.command == pytest.approx(0.0, abs=1e-6)
    for _ in range(int(latency_s / DT) + 8):
        dynamics.step({"gate": 1.0}, DT)
    assert gate.state.delayed_command == pytest.approx(1.0, abs=1e-9)
    # Rate limit: 3 cmd/s cannot reach a 0→1 step in a few physics ticks.
    assert 0.0 < gate.state.command < 0.5
    for _ in range(int(math.ceil(1.1 / rate / DT))):
        dynamics.step({"gate": 1.0}, DT)
    assert gate.state.command == pytest.approx(1.0, abs=0.05)


def test_hood_rate_limit_is_slower_than_unfiltered_step():
    dynamics = MechanismDynamics(_biobuzz_robot())
    hood = dynamics.actuators["hood"]
    rate = float(hood.config["commandRatePerS"])
    dynamics.step({"hood": 1.0}, DT)
    assert abs(hood.state.command) <= rate * DT + 1e-9
    for _ in range(20):
        dynamics.step({"hood": 1.0}, DT)
    assert abs(hood.state.command) < 0.2


def test_mechanism_step_is_deterministic_for_identical_commands():
    robot = _biobuzz_robot()
    left = MechanismDynamics(copy.deepcopy(robot))
    right = MechanismDynamics(copy.deepcopy(robot))
    rng = np.random.default_rng(11)
    for _ in range(80):
        commands = {
            "intake": float(rng.choice([-1.0, 0.0, 1.0])),
            "conveyor": float(rng.choice([0.0, 1.0])),
            "flywheel": float(rng.uniform(0.0, 1.0)),
            "hood": float(rng.uniform(-1.0, 1.0)),
            "gate": float(rng.choice([-1.0, 1.0])),
        }
        left.step(commands, DT, drivetrain_current_a=12.0)
        right.step(commands, DT, drivetrain_current_a=12.0)
    assert left.state_dict() == right.state_dict()


def test_world_seed_reproduces_actuator_and_pose_state():
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")

    def _run(seed: int) -> dict:
        world = World(bundle, seed=seed, allow_missing_mesh=True)
        world.reset(seed=seed, static_teammate=False, full_noise=True)
        pose = np.array(
            [world.actor().body.x, world.actor().body.y, world.actor().body.heading],
            dtype=np.float64,
        )
        for _ in range(12):
            world.step(pose, 0.4, 2)
        rs = world.actor()
        assert rs.mechanism is not None
        return {
            "pose": (rs.body.x, rs.body.y, rs.body.heading),
            "voltage": rs.mechanism.power.voltage_v,
            "flywheel": rs.mechanism.actuators["flywheel"].state.velocity_rad_s,
            "hood": rs.mechanism.actuators["hood"].state.position,
            "held": tuple(rs.held),
        }

    first = _run(4)
    second = _run(4)
    other = _run(9)
    assert first == second
    assert first["flywheel"] != other["flywheel"] or first["pose"] != other["pose"]
