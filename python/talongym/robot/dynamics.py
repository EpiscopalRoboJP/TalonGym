"""Shared-battery actuator dynamics for articulated robot mechanisms."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


RPM_TO_RAD_S = 2.0 * math.pi / 60.0


@dataclass
class ActuatorState:
    requested_command: float = 0.0
    command: float = 0.0
    delayed_command: float = 0.0
    position: float = 0.0
    velocity_rad_s: float = 0.0
    current_a: float = 0.0
    torque_nm: float = 0.0
    temperature_c: float = 25.0
    enabled: bool = True
    command_queue: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PowerState:
    voltage_v: float
    current_a: float = 0.0
    state_of_charge: float = 1.0
    brownout: bool = False


class ActuatorModel:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        thermal = config.get("thermal") or {}
        ambient = float(thermal.get("ambientC", 25.0))
        initial_position = 0.0
        travel = config.get("travelLimit") or []
        if len(travel) == 2:
            initial_position = float(travel[0])
        self.state = ActuatorState(
            position=initial_position,
            temperature_c=ambient,
        )
        motor = config["motor"]
        self.nominal_voltage = float(motor["nominalVoltageV"])
        self.free_omega_motor = float(motor["freeSpeedRpm"]) * RPM_TO_RAD_S
        self.stall_torque = float(motor["stallTorqueNm"])
        self.stall_current = float(motor["stallCurrentA"])
        self.free_current = float(motor["freeCurrentA"])
        self.resistance = self.nominal_voltage / max(self.stall_current, 1e-6)
        self.kt = self.stall_torque / max(self.stall_current, 1e-6)
        self.ke = self.nominal_voltage / max(self.free_omega_motor, 1e-6)
        self.gear_ratio = float(config.get("gearRatio") or 1.0)
        self.efficiency = float(config.get("efficiency") or 1.0)
        self.inertia = max(
            1e-8,
            float(config.get("loadInertiaKgM2") or 0)
            + float(config.get("rotorInertiaKgM2") or 0) * self.gear_ratio**2,
        )

    def queue_command(self, command: float, now_s: float) -> None:
        latency_s = 0.001 * float(self.config.get("controllerLatencyMs") or 0)
        value = float(np.clip(command, -1.0, 1.0))
        if abs(value - self.state.requested_command) > 1e-12:
            self.state.requested_command = value
            self.state.command_queue.append((now_s + latency_s, value))

    def advance_command(self, now_s: float, dt: float) -> None:
        queue = self.state.command_queue
        while queue and queue[0][0] <= now_s + 1e-12:
            _, self.state.delayed_command = queue.pop(0)
        rate = float(self.config.get("commandRatePerS") or 1e9)
        max_delta = rate * dt
        delta = float(
            np.clip(
                self.state.delayed_command - self.state.command,
                -max_delta,
                max_delta,
            )
        )
        # command is also the realizable, post-rate-limit command after this point.
        self.state.command += delta

    def _duty(self) -> float:
        command = self.state.command
        kind = str(self.config["kind"])
        if kind == "velocity_motor":
            target_rpm = float(self.config.get("targetRpm") or 0) * command
            target = target_rpm * RPM_TO_RAD_S
            control_band = max(5.0, 0.15 * abs(target))
            return float(np.clip((target - self.state.velocity_rad_s) / control_band, -1.0, 1.0))
        travel = self.config.get("travelLimit") or [0.0, 1.0]
        lo, hi = float(travel[0]), float(travel[1])
        target = lo + 0.5 * (command + 1.0) * (hi - lo)
        error = target - self.state.position
        kp = float(self.config.get("kp") or 0.1)
        kd = float(self.config.get("kd") or 0.0)
        return float(np.clip(kp * error - kd * self.state.velocity_rad_s, -1.0, 1.0))

    def effort(self, voltage_v: float) -> tuple[float, float]:
        if not self.state.enabled:
            return 0.0, 0.0
        duty = self._duty()
        motor_omega = self.state.velocity_rad_s * self.gear_ratio
        applied_voltage = duty * max(0.0, voltage_v)
        current = (applied_voltage - self.ke * motor_omega) / self.resistance
        current_limit = float(self.config["currentLimitA"])
        current = float(np.clip(current, -current_limit, current_limit))
        motoring_current = max(0.0, abs(current))
        torque_motor = self.kt * current
        torque_output = torque_motor * self.gear_ratio * self.efficiency
        return torque_output, motoring_current

    def integrate(self, voltage_v: float, dt: float) -> None:
        torque, current = self.effort(voltage_v)
        velocity = self.state.velocity_rad_s
        viscous = float(self.config.get("viscousFrictionNmPerRadS") or 0) * velocity
        coulomb = float(self.config.get("coulombFrictionNm") or 0)
        friction = math.copysign(coulomb, velocity) if abs(velocity) > 1e-5 else 0.0
        alpha = (torque - viscous - friction) / self.inertia
        velocity += alpha * dt
        free_output = self.free_omega_motor / self.gear_ratio
        velocity = float(np.clip(velocity, -1.25 * free_output, 1.25 * free_output))
        self.state.velocity_rad_s = velocity

        kind = str(self.config["kind"])
        if kind != "velocity_motor":
            self.state.position += math.degrees(velocity) * dt
            travel = self.config.get("travelLimit") or []
            if len(travel) == 2:
                lo, hi = float(travel[0]), float(travel[1])
                if self.state.position < lo or self.state.position > hi:
                    self.state.position = float(np.clip(self.state.position, lo, hi))
                    self.state.velocity_rad_s = 0.0
        else:
            self.state.position += velocity * dt

        self.state.current_a = current
        self.state.torque_nm = torque
        thermal = self.config.get("thermal") or {}
        if thermal:
            ambient = float(thermal["ambientC"])
            thermal_mass = float(thermal["thermalMassJPerC"])
            thermal_resistance = float(thermal["thermalResistanceCPerW"])
            copper_loss = current * current * self.resistance
            cooling = (self.state.temperature_c - ambient) / thermal_resistance
            self.state.temperature_c += (copper_loss - cooling) * dt / thermal_mass
            if self.state.temperature_c >= float(thermal["shutdownC"]):
                self.state.enabled = False


class MechanismDynamics:
    """All robot actuators coupled through one finite battery."""

    def __init__(self, robot: dict[str, Any]) -> None:
        self.robot = robot
        self.actuators = {
            str(row["id"]): ActuatorModel(row)
            for row in robot.get("actuators") or []
        }
        power = robot.get("powerSystem") or {}
        self.open_circuit_voltage = float(power.get("openCircuitVoltageV") or 12.0)
        self.internal_resistance = float(power.get("internalResistanceOhm") or 0.0)
        self.capacity_ah = float(power.get("capacityAh") or 3.0)
        self.brownout_voltage = float(power.get("brownoutVoltageV") or 0.0)
        self.max_current = float(power.get("maxCurrentA") or 1e9)
        self.power = PowerState(
            voltage_v=self.open_circuit_voltage,
            state_of_charge=float(power.get("initialStateOfCharge") or 1.0),
        )
        self.time_s = 0.0

    def step(
        self,
        commands: dict[str, float],
        dt: float,
        *,
        drivetrain_current_a: float = 0.0,
    ) -> None:
        for ident, actuator in self.actuators.items():
            actuator.queue_command(float(commands.get(ident, actuator.state.command)), self.time_s)
            actuator.advance_command(self.time_s, dt)

        guessed_voltage = self.power.voltage_v
        mechanism_current = sum(
            actuator.effort(guessed_voltage)[1]
            for actuator in self.actuators.values()
        )
        total_current = min(
            self.max_current,
            max(0.0, float(drivetrain_current_a)) + mechanism_current,
        )
        soc_ocv = self.open_circuit_voltage * (0.9 + 0.1 * self.power.state_of_charge)
        voltage = max(0.0, soc_ocv - total_current * self.internal_resistance)
        brownout = voltage < self.brownout_voltage
        if brownout:
            voltage = max(0.0, voltage)

        for actuator in self.actuators.values():
            if brownout:
                original = actuator.state.command
                actuator.state.command *= max(
                    0.0,
                    min(1.0, voltage / max(self.brownout_voltage, 1e-6)),
                )
                actuator.integrate(voltage, dt)
                actuator.state.command = original
            else:
                actuator.integrate(voltage, dt)

        actual_mechanism_current = sum(
            actuator.state.current_a for actuator in self.actuators.values()
        )
        actual_total = min(
            self.max_current,
            max(0.0, float(drivetrain_current_a)) + actual_mechanism_current,
        )
        self.power.current_a = actual_total
        self.power.voltage_v = max(
            0.0,
            soc_ocv - actual_total * self.internal_resistance,
        )
        self.power.brownout = self.power.voltage_v < self.brownout_voltage
        self.power.state_of_charge = max(
            0.0,
            self.power.state_of_charge - actual_total * dt / (3600.0 * self.capacity_ah),
        )
        self.time_s += dt

    def state_dict(self) -> dict[str, Any]:
        return {
            "batteryVoltageV": self.power.voltage_v,
            "batteryCurrentA": self.power.current_a,
            "stateOfCharge": self.power.state_of_charge,
            "brownout": self.power.brownout,
            "piecePath": dict(self.robot.get("piecePath") or {}),
            "chassis": dict(self.robot.get("chassis") or {}),
            "actuators": {
                ident: {
                    "jointId": actuator.config.get("jointId"),
                    "kind": actuator.config.get("kind"),
                    "command": actuator.state.command,
                    "position": actuator.state.position,
                    "velocityRadS": actuator.state.velocity_rad_s,
                    "rpm": actuator.state.velocity_rad_s / RPM_TO_RAD_S,
                    "currentA": actuator.state.current_a,
                    "torqueNm": actuator.state.torque_nm,
                    "temperatureC": actuator.state.temperature_c,
                    "enabled": actuator.state.enabled,
                    "targetRpm": actuator.config.get("targetRpm"),
                    "travelLimit": list(actuator.config.get("travelLimit") or []),
                }
                for ident, actuator in self.actuators.items()
            },
        }
