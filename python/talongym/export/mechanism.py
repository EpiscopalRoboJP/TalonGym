"""Mechanism command timeline export.

TalonGym is GPL-3.0-or-later. Strings returned by ``export_mechanism_timeline``
are generated OpMode comments and are CC0-1.0, so a team pasting them into
robot code does not GPL that OpMode.
"""

from __future__ import annotations

from typing import Any


IDLE_VERBS = {"", "idle", "none"}


class DriveOnlyExportError(ValueError):
    """Drive-only Road Runner cannot represent required mechanism actions."""

    code = "MECHANISM_ACTIONS_REQUIRED"


def _actuator_map(robot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = robot.get("actuators")
    if isinstance(raw, dict):
        return {str(key): (value if isinstance(value, dict) else {}) for key, value in raw.items()}
    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for row in raw:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            out[str(row["id"])] = row
        return out
    return {}


def mechanism_actions_required(frames: list[dict[str, Any]]) -> bool:
    """True when a replay used non-drive mechanism commands that RR cannot encode."""
    for frame in frames:
        for robot in frame.get("robots") or []:
            verb = str(robot.get("lastVerb") or "idle").strip().lower()
            if verb not in IDLE_VERBS:
                return True
            for row in _actuator_map(robot).values():
                if abs(float(row.get("command") or 0.0)) > 1e-3:
                    return True
        commands = frame.get("mechanismCommands")
        if isinstance(commands, dict) and any(abs(float(v or 0.0)) > 1e-3 for v in commands.values()):
            return True
        if any(bool(piece.get("inFlight")) for piece in frame.get("pieces") or []):
            return True
    return False


def _sorted_actuator_ids(frames: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for frame in frames:
        for robot in frame.get("robots") or []:
            for ident in _actuator_map(robot):
                if ident not in seen:
                    seen.append(ident)
        commands = frame.get("mechanismCommands")
        if isinstance(commands, dict):
            for ident in commands:
                key = str(ident)
                if key not in seen:
                    seen.append(key)
    return seen


def _row_for_frame(frame: dict[str, Any], actuator_ids: list[str]) -> dict[str, Any] | None:
    robots = frame.get("robots") or []
    robot = robots[0] if robots else {}
    actuators = _actuator_map(robot)
    commands = frame.get("mechanismCommands") if isinstance(frame.get("mechanismCommands"), dict) else {}
    values: dict[str, float] = {}
    active = False
    for ident in actuator_ids:
        raw = actuators.get(ident) or {}
        value = raw.get("command")
        if value is None:
            value = commands.get(ident, 0.0)
        command = float(value or 0.0)
        values[ident] = command
        if abs(command) > 1e-3:
            active = True
    verb = str(robot.get("lastVerb") or "idle")
    if verb.strip().lower() not in IDLE_VERBS:
        active = True
    if not active:
        return None
    return {
        "t": float(frame.get("t") or 0.0),
        "verb": verb,
        "commands": values,
        "batteryVoltageV": robot.get("batteryVoltageV"),
    }


def export_mechanism_timeline(frames: list[dict[str, Any]]) -> str:
    """Sequential actuator command comments for pasting beside a drive path."""
    actuator_ids = _sorted_actuator_ids(frames)
    lines = [
        "// TalonGym export — mechanism command timeline",
        "// Not a Road Runner drive path. Apply these actuator commands in AUTO.",
        "// command is normalized [-1, 1] (voltage/velocity/position target).",
    ]
    if actuator_ids:
        lines.append("// actuators: " + ", ".join(actuator_ids))
    last_key: tuple[str, tuple[float, ...]] | None = None
    emitted = 0
    for frame in frames:
        row = _row_for_frame(frame, actuator_ids)
        if row is None:
            continue
        cmd_vals = tuple(round(row["commands"].get(ident, 0.0), 3) for ident in actuator_ids)
        key = (str(row["verb"]), cmd_vals)
        if key == last_key:
            continue
        last_key = key
        parts = [f"{ident}={row['commands'].get(ident, 0.0):.3f}" for ident in actuator_ids]
        battery = row.get("batteryVoltageV")
        extra = f" verb={row['verb']}"
        if battery is not None:
            extra += f" batteryV={float(battery):.2f}"
        lines.append(f"// t={row['t']:.2f}s" + extra + ((" " + " ".join(parts)) if parts else ""))
        emitted += 1
    if emitted == 0:
        lines.append("// (no non-idle mechanism commands in this replay)")
    return "\n".join(lines) + "\n"


def reject_drive_only_message() -> str:
    return (
        "// ERROR: MECHANISM_ACTIONS_REQUIRED\n"
        "// Drive-only Road Runner export cannot represent intake/conveyor/flywheel/hood/gate commands.\n"
        "// Chassis splines below are incomplete. Use the mechanism timeline as the AUTO companion.\n"
    )
