"""Focused MechanismDynamics coverage for physical piece-path actuators."""

from __future__ import annotations

from talongym.robot.dynamics import RPM_TO_RAD_S, MechanismDynamics


def _motor(**overrides):
    row = {
        "nominalVoltageV": 12.0,
        "freeSpeedRpm": 312.0,
        "stallTorqueNm": 2.1,
        "stallCurrentA": 9.2,
        "freeCurrentA": 0.25,
    }
    row.update(overrides)
    return row


def _velocity(ident: str, **overrides):
    row = {
        "id": ident,
        "kind": "velocity_motor",
        "jointId": f"{ident}_joint",
        "motor": _motor(),
        "gearRatio": 1.0,
        "efficiency": 0.8,
        "rotorInertiaKgM2": 8e-5,
        "loadInertiaKgM2": 3.5e-4,
        "viscousFrictionNmPerRadS": 0.002,
        "coulombFrictionNm": 0.01,
        "currentLimitA": 8.0,
        "controllerLatencyMs": 0.0,
        "commandRatePerS": 1e9,
        "targetRpm": 300.0,
    }
    row.update(overrides)
    return row


def _servo(ident: str, travel: list[float], **overrides):
    row = {
        "id": ident,
        "kind": "servo",
        "jointId": f"{ident}_joint",
        "motor": _motor(nominalVoltageV=6.0, freeSpeedRpm=60.0, stallTorqueNm=1.2, stallCurrentA=2.5),
        "gearRatio": 1.0,
        "efficiency": 0.7,
        "rotorInertiaKgM2": 2e-5,
        "loadInertiaKgM2": 4e-4,
        "viscousFrictionNmPerRadS": 0.02,
        "coulombFrictionNm": 0.03,
        "currentLimitA": 2.0,
        "controllerLatencyMs": 0.0,
        "commandRatePerS": 1e9,
        "travelLimit": travel,
        "kp": 0.22,
        "kd": 0.02,
    }
    row.update(overrides)
    return row


def _robot():
    return {
        "actuators": [
            _velocity("intake"),
            _velocity("flywheel", targetRpm=4500.0, motor=_motor(freeSpeedRpm=6000.0, stallTorqueNm=0.5, stallCurrentA=20.0), currentLimitA=18.0),
            _servo("gate", [0.0, 75.0]),
            _servo("hood", [25.0, 70.0], kind="position_motor", kp=0.18),
        ],
        "powerSystem": {
            "openCircuitVoltageV": 13.0,
            "internalResistanceOhm": 0.018,
            "capacityAh": 3.0,
            "initialStateOfCharge": 1.0,
            "brownoutVoltageV": 9.0,
            "maxCurrentA": 120.0,
        },
        "piecePath": {
            "intakeActuatorId": "intake",
            "conveyorActuatorId": "intake",
            "flywheelActuatorId": "flywheel",
            "gateActuatorId": "gate",
            "hoodActuatorId": "hood",
            "launchEfficiency": 0.235,
        },
        "chassis": {"lengthIn": 18, "widthIn": 18, "heightIn": 14},
    }


def test_state_dict_exposes_piece_path_and_travel():
    dynamics = MechanismDynamics(_robot())
    state = dynamics.state_dict()
    assert state["piecePath"]["flywheelActuatorId"] == "flywheel"
    assert state["chassis"]["lengthIn"] == 18
    assert state["actuators"]["gate"]["travelLimit"] == [0.0, 75.0]
    assert state["actuators"]["flywheel"]["targetRpm"] == 4500.0
    assert "rpm" in state["actuators"]["flywheel"]
    assert "torqueNm" in state["actuators"]["flywheel"]


def test_flywheel_rpm_tracks_command_not_instant_target():
    dynamics = MechanismDynamics(_robot())
    dynamics.step({"flywheel": 1.0}, 0.002)
    first = abs(dynamics.actuators["flywheel"].state.velocity_rad_s) / RPM_TO_RAD_S
    assert first < 800.0
    for _ in range(250):
        dynamics.step({"flywheel": 1.0}, 0.002)
    spun = abs(dynamics.actuators["flywheel"].state.velocity_rad_s) / RPM_TO_RAD_S
    assert spun > 1200.0
    assert spun > first * 4.0


def test_gate_and_hood_move_from_closed_toward_commanded_travel():
    dynamics = MechanismDynamics(_robot())
    assert dynamics.actuators["gate"].state.position == 0.0
    assert dynamics.actuators["hood"].state.position == 25.0
    for _ in range(80):
        dynamics.step({"gate": 1.0, "hood": 1.0}, 0.02)
    assert dynamics.actuators["gate"].state.position > 20.0
    assert dynamics.actuators["hood"].state.position > 40.0


def test_zero_command_does_not_open_gate():
    dynamics = MechanismDynamics(_robot())
    for _ in range(40):
        dynamics.step({"gate": -1.0}, 0.02)
    assert dynamics.actuators["gate"].state.position < 5.0


def test_drivetrain_current_sags_shared_bus():
    idle = MechanismDynamics(_robot())
    loaded = MechanismDynamics(_robot())
    for _ in range(40):
        idle.step({"flywheel": 1.0}, 0.01, drivetrain_current_a=0.0)
        loaded.step({"flywheel": 1.0}, 0.01, drivetrain_current_a=80.0)
    assert loaded.power.voltage_v < idle.power.voltage_v
    assert loaded.actuators["flywheel"].state.velocity_rad_s < idle.actuators["flywheel"].state.velocity_rad_s
