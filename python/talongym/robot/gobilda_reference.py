"""Physical preset traced to goBILDA's 3200-2627-0004 assembly CAD.

Only geometry and published component specifications are treated as measured.
Contact, payload, motor control, and whole-robot mass still need a scale and
shot calibration before this preset is used as a competition predictor.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from talongym.paths import PRESETS_DIR
from talongym.presets.loader import validate_document
from talongym.robot.contract import compile_robot_preset

REFERENCE_ID = "gobilda_biobuzz_reference"
OUTPUT = PRESETS_DIR / "robots" / f"{REFERENCE_ID}.json"
TEMPLATE = PRESETS_DIR / "robots" / "mecanum_biobuzz_4cap.json"
MM_PER_IN = 25.4
HEIGHT_IN = 309.34185 / MM_PER_IN
WHEEL_RADIUS_IN = 52 / MM_PER_IN
HOGBACK_RADIUS_IN = 48 / MM_PER_IN


def _pose(cad_x_mm: float, cad_y_mm: float, cad_z_mm: float) -> dict[str, float]:
    """CAD XYZ millimetres to TalonGym robot XYZ inches, centred on envelope."""
    return {
        "x": round(-(cad_y_mm + 34) / MM_PER_IN, 5),
        "y": round(cad_x_mm / MM_PER_IN, 5),
        "z": round((cad_z_mm + 52) / MM_PER_IN - HEIGHT_IN / 2, 5),
    }


def _box(size_mm: tuple[float, float, float], *, cad_xyz: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "kind": "box",
        "sizeIn": [round(value / MM_PER_IN, 5) for value in size_mm],
        "pose": _pose(*cad_xyz),
        "friction": 0.8,
        "restitution": 0.05,
    }


def _joint(ident: str, parent: str, child: str, axis: list[int], anchor: dict[str, float]) -> dict[str, Any]:
    return {
        "id": ident,
        "type": "hinge",
        "parentPartId": parent,
        "childPartId": child,
        "anchorIn": anchor,
        "axis": axis,
        "limit": [-1000000, 1000000],
        "damping": 0.001,
        "frictionLoss": 0.001,
        "backlash": 0,
    }


def build_document() -> dict[str, Any]:
    doc = copy.deepcopy(json.loads(TEMPLATE.read_text(encoding="utf-8")))
    doc["id"] = REFERENCE_ID
    doc["displayName"] = "goBILDA BIOBUZZ mecanum StarterBot (CAD reference)"
    doc["visualAsset"] = f"robots/{REFERENCE_ID}/robot.glb"
    doc["drivetrain"] = {
        "type": "mecanum",
        "trackWidthIn": round(414 / MM_PER_IN, 5),
        "wheelDiameterIn": round(104 / MM_PER_IN, 5),
        "wheelbaseIn": round(264 / MM_PER_IN, 5),
        "strafeMultiplier": 1.0,
    }
    doc["chassis"].update({
        "lengthIn": round(452 / MM_PER_IN, 5),
        "widthIn": round(452.0917 / MM_PER_IN, 5),
        "heightIn": round(HEIGHT_IN, 5),
        "massKg": 9.5,
    })
    doc["motors"].update({
        "count": 4,
        "freeSpeedRpm": 312,
        "stallTorqueNm": round(24.3 * 0.0980665, 5),
        "currentLimitA": 9.2,
        "gearRatio": 19.2,
    })
    doc["sensors"] = [{"id": "imu", "kind": "imu"}]
    doc["intakes"] = [{
        "id": "front_intake",
        "poseOnRobot": {"x": round(190 / MM_PER_IN, 5), "y": 0, "z": round(39 / MM_PER_IN, 5), "headingDeg": 0},
        "widthIn": round(384 / MM_PER_IN, 5),
        "reachIn": 2.0,
        "heightIn": round(72 / MM_PER_IN, 5),
        "cycleTimeS": 0.5,
        "maxSpeedInPerS": 20,
        "canRunWhileMoving": True,
    }]
    # The large gridplate roof rises from the flywheel toward CAD +Y.
    # Its slope is ~37 degrees from the assembled CAD, with a virtual exit
    # 16 mm past the upper edge to clear the polycarbonate.
    muzzle = {
        "x": round(-(80 + 34) / MM_PER_IN, 5),
        "y": 0,
        "z": round((266 + 52) / MM_PER_IN, 5),
        "pitchDeg": 37,
        "yawDeg": 180,
    }
    doc["launchers"] = [{
        "id": "fixed_gridplate_guide",
        "poseOnRobot": {
            "x": muzzle["x"], "y": 0, "z": muzzle["z"],
            "headingDeg": 180, "pitchDeg": 37,
        },
        "aimMode": "chassis_fixed",
        "muzzleSpeedInPerS": 150,
        "spinupTimeS": 0.5,
        "cycleTimeS": 0.7,
        "canLaunchWhileMoving": True,
    }]
    chassis_collision = [
        _box((384, 48, 48), cad_xyz=(156, 0, 0)),
        _box((384, 48, 48), cad_xyz=(-156, 0, 0)),
        _box((12, 264, 48), cad_xyz=(0, 186, 0)),
        _box((48, 264, 12), cad_xyz=(0, 88, -14)),
        _box((48, 48, 264), cad_xyz=(68, 88, 124)),
        _box((48, 48, 264), cad_xyz=(-68, 88, 124)),
    ]
    # _box inputs are in robot-axis order (length, width, height), while the
    # CAD coordinates above are native XYZ. Main longitudinal rails are
    # 384 mm along robot X, and the rear crossbar is 264 mm along robot Y.
    wheel_rows: list[dict[str, Any]] = []
    wheel_joints: list[dict[str, Any]] = []
    for ident, cad_x, cad_y in (
        ("wheel_fl", 207, -144),
        ("wheel_fr", -207, -144),
        ("wheel_rl", 207, 120),
        ("wheel_rr", -207, 120),
    ):
        pose = _pose(cad_x, cad_y, 0)
        wheel_rows.append({
            "id": ident,
            "parentId": "chassis",
            "pose": pose,
            "massKg": 0.32,
            "collision": [{
                "kind": "cylinder", "radiusIn": round(WHEEL_RADIUS_IN, 5),
                "lengthIn": round(36 / MM_PER_IN, 5),
                "friction": 1.0, "restitution": 0.05,
            }],
        })
        wheel_joints.append(_joint(f"{ident}_joint", "chassis", ident, [0, 1, 0], pose))
    intake_pose = _pose(0, -224, -13)
    conveyor_pose = _pose(-9, -185, 36)
    flywheel_pose = _pose(-5, 40, 120)
    gate_pose = _pose(-69, -30, 105)
    # Multiple separated roller contacts share a synchronous aggregate
    # actuator; the render keeps the actual independent servo hardware.
    doc["rigidParts"] = [
        {
            "id": "chassis", "parentId": None, "pose": {"x": 0, "y": 0, "z": 0},
            "massKg": 7.4, "collision": chassis_collision,
        },
        *wheel_rows,
        {
            "id": "intake_roller", "parentId": "chassis", "pose": intake_pose,
            "massKg": 0.25,
            "collision": [
                {
                    "kind": "cylinder", "radiusIn": round(36 / MM_PER_IN, 5),
                    "lengthIn": round(24 / MM_PER_IN, 5),
                    "pose": {"x": 0, "y": round(side * 156 / MM_PER_IN, 5), "z": 0},
                    "friction": 1.2, "restitution": 0.05,
                }
                for side in (-1, 1)
            ],
        },
        {
            "id": "conveyor_roller", "parentId": "chassis", "pose": conveyor_pose,
            "massKg": 0.3,
            "collision": [{
                "kind": "cylinder", "radiusIn": round(24 / MM_PER_IN, 5),
                "lengthIn": round(184 / MM_PER_IN, 5),
                "friction": 1.2, "restitution": 0.05,
            }],
        },
        {
            "id": "flywheel", "parentId": "chassis", "pose": flywheel_pose,
            "massKg": 0.105,
            "inertiaKgM2": [0.00006, 0.000108, 0.00006],
            "collision": [{
                "kind": "cylinder", "radiusIn": round(HOGBACK_RADIUS_IN, 5),
                "lengthIn": round(24 / MM_PER_IN, 5),
                "friction": 1.25, "restitution": 0.12,
            }],
        },
        {
            "id": "gridplate_guide", "parentId": "chassis",
            "pose": {"x": 0, "y": 0, "z": 0},
            "massKg": 0.19,
            "collision": [{
                "kind": "box",
                "sizeIn": [round(216 / MM_PER_IN, 5), round(176 / MM_PER_IN, 5), round(1.5 / MM_PER_IN, 5)],
                "pose": {
                    **_pose(0, -27, 189),
                    "pitchDeg": 37,
                },
                "friction": 0.45, "restitution": 0.08,
            }],
        },
        {
            "id": "release_gate", "parentId": "chassis", "pose": gate_pose,
            "massKg": 0.12,
            "collision": [{
                "kind": "box", "sizeIn": [1.0, 2.0, 0.3],
                "friction": 0.5, "restitution": 0.05,
            }],
        },
    ]
    doc["joints"] = [
        *wheel_joints,
        _joint("intake_joint", "chassis", "intake_roller", [0, 1, 0], intake_pose),
        _joint("conveyor_joint", "chassis", "conveyor_roller", [0, 1, 0], conveyor_pose),
        _joint("flywheel_joint", "chassis", "flywheel", [0, 1, 0], flywheel_pose),
        {
            "id": "guide_joint", "type": "fixed",
            "parentPartId": "chassis", "childPartId": "gridplate_guide",
        },
        {
            **_joint("gate_joint", "chassis", "release_gate", [0, 1, 0], gate_pose),
            "limit": [0, 75],
        },
    ]
    doc["actuators"] = [row for row in doc["actuators"] if row["id"] != "hood"]
    flywheel = next(row for row in doc["actuators"] if row["id"] == "flywheel")
    flywheel["motor"] = {
        "nominalVoltageV": 12,
        "freeSpeedRpm": 6000,
        "stallTorqueNm": round(1.47 * 0.0980665, 6),
        "stallCurrentA": 9.2,
        "freeCurrentA": 0.25,
    }
    flywheel["loadInertiaKgM2"] = 0.000108
    # Manufacturer example code commands 1,250 encoder ticks/s (2,678 RPM)
    # and starts the windmill at 1,200 ticks/s (2,571 RPM).
    flywheel["targetRpm"] = 1250 * 60 / 28
    flywheel["readyRpm"] = 1200 * 60 / 28
    flywheel["velocityControlBandRpm"] = 50
    flywheel["currentLimitA"] = 9.2
    # The legacy 0.0004 N m/(rad/s) drag term overwhelms this motor's
    # published 0.144 N m stall torque and caps it near 2,100 RPM unloaded.
    # This lower parasitic estimate remains a calibration parameter.
    flywheel["viscousFrictionNmPerRadS"] = 0.00005
    flywheel["coulombFrictionNm"] = 0.003
    doc["piecePath"].update({
        "hoodActuatorId": None,
        "storageSlots": [
            {"x": 4.5, "y": 0, "z": 3.5},
            {"x": 1.5, "y": 0, "z": 3.5},
            {"x": -1.5, "y": 0, "z": 3.5},
            {"x": -4.5, "y": 0, "z": 3.5},
        ],
        "intakePose": {"x": round(190 / MM_PER_IN, 5), "y": 0, "z": round(39 / MM_PER_IN, 5), "yawDeg": 0},
        "muzzlePose": muzzle,
        "muzzleClearanceIn": 0.25,
        "flywheelPose": {
            "x": flywheel_pose["x"], "y": flywheel_pose["y"],
            "z": round((120 + 52) / MM_PER_IN, 5),
        },
        "wheelRadiusIn": round(HOGBACK_RADIUS_IN, 5),
        "launchEfficiency": 0.235,
    })
    doc["mechanismSensors"] = [row for row in doc["mechanismSensors"] if row["id"] != "hood_position"]
    doc["mechanismSensors"] = [
        row for row in doc["mechanismSensors"]
        if row.get("pose") is None or row["id"] != "magazine_exit"
    ]
    doc["mechanisms"]["capacity"] = 4
    doc["mechanisms"]["scoreCycleTimeS"] = 0.7
    doc["warnings"] = [
        {
            "code": "reference_mass_unverified", "severity": "warning",
            "message": "The 9.5 kg assembled mass and individual non-product masses are estimates; weigh the built robot before using acceleration or stability predictions.",
        },
        {
            "code": "launcher_contact_unverified", "severity": "warning",
            "message": "CAD fixes the wheel, guide, and exit direction; 37-degree pitch is an assembled-plate estimate. Ball compression, loaded RPM, and speed efficiency require shot measurements.",
        },
        {
            "code": "intake_aggregate_model", "severity": "warning",
            "message": "Two independently servo-driven front intake wheels are represented by one synchronous physics actuator.",
        },
        {
            "code": "preload_slot_fit_unverified", "severity": "warning",
            "message": "goBILDA documents four-POLLEN storage, but the simulator's four slot coordinates and ball retention have not been measured in the assembled chute.",
        },
    ]
    errors = validate_document("robot", doc)
    if errors:
        raise ValueError("goBILDA reference schema: " + "; ".join(errors[:10]))
    compile_robot_preset(doc, competitive=True)
    return doc


def main() -> None:
    doc = build_document()
    OUTPUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
