"""Compile schema 1.2 catalog assemblies into the existing 1.1 physical contract."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from talongym.presets.loader import validate_document
from talongym.robot.catalog import cache_entry, cad_extra_available, get_part
from talongym.robot.collision import connected_overlap_warnings, validate_assembly_collisions
from talongym.robot.contract import (
    ASSEMBLY_SCHEMA_VERSION,
    PHYSICAL_SCHEMA_VERSION,
    AssemblyError,
    CompiledRobot,
    compile_robot_preset,
)
from talongym.robot.graph import validate_assembly_graph
from talongym.robot.inference import (
    TOPOLOGY_FALLBACK_FIELDS,
    InferenceReport,
    build_inference_report,
    infer_connection_joint,
)
from talongym.robot.mounts import (
    hole_in_part,
    mount_axis,
    part_mount,
    validate_connection_mounts,
    validate_occupancy,
)
from talongym.robot.transforms import (
    matrix_from_pose,
    pose_from_matrix,
    relative_transform,
    solve_mount_transform,
)
from talongym.robot.validation import validate_inferred_assembly

TOPOLOGY_PRESET_ID = "mecanum_biobuzz_4cap"
_TOPOLOGY_MECHANISM_IDS = frozenset({"intake_roller", "conveyor_roller", "flywheel", "hood", "release_gate"})
_DEFAULT_FRICTION = 0.8
_DEFAULT_RESTITUTION = 0.05
_CONTINUOUS_LIMIT = [-1000000.0, 1000000.0]


@dataclass(frozen=True)
class CompiledAssembly:
    preset: dict[str, Any]
    compiled: CompiledRobot
    instance_poses: dict[str, dict[str, float]]
    report: InferenceReport
    warnings: tuple[dict[str, Any], ...]


def _load_topology() -> dict[str, Any]:
    from talongym.presets.loader import load_preset

    return copy.deepcopy(load_preset("robot", TOPOLOGY_PRESET_ID))


def _strip_scoring_topology(preset: dict[str, Any]) -> dict[str, Any]:
    """Drivebase-only 1.1 chassis: catalog parts supply structure, not grafted mechanisms."""
    preset["rigidParts"] = [
        row for row in (preset.get("rigidParts") or []) if str(row.get("id") or "") not in _TOPOLOGY_MECHANISM_IDS
    ]
    preset["joints"] = [
        row
        for row in (preset.get("joints") or [])
        if str(row.get("childPartId") or "") not in _TOPOLOGY_MECHANISM_IDS
        and str(row.get("parentPartId") or "") not in _TOPOLOGY_MECHANISM_IDS
    ]
    preset["actuators"] = []
    preset["mechanismSensors"] = []
    preset["intakes"] = []
    preset["launchers"] = []
    preset.pop("piecePath", None)
    mechanisms = dict(preset.get("mechanisms") or {})
    mechanisms["launchCapable"] = False
    mechanisms["capacity"] = 0
    preset["mechanisms"] = mechanisms
    preset["defaultActionTier"] = "high_level_waypoint"
    return preset


def _minimal_chassis_parent(preset: dict[str, Any]) -> dict[str, Any]:
    """Keep a chassis id for catalog parenting; drop the 4-cap hull collision boxes."""
    parent: dict[str, Any] = {
        "id": "chassis",
        "parentId": None,
        "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
        "massKg": 1e-6,
        "collision": [],
    }
    for row in preset.get("rigidParts") or []:
        if str(row.get("id") or "") != "chassis":
            continue
        parent["pose"] = dict(row.get("pose") or parent["pose"])
        break
    preset["rigidParts"] = [parent]
    return preset


def _restore_scoring_contract(preset: dict[str, Any], template: dict[str, Any]) -> None:
    """Copy functional 4-cap fields without grafting its rigid-part hull."""
    for key in TOPOLOGY_FALLBACK_FIELDS:
        if key in template:
            preset[key] = copy.deepcopy(template[key])


_ACTUATOR_CATALOG_CHILDREN = {
    "intake": ("intake_wheel",),
    "conveyor": ("conveyor_wheel",),
    "flywheel": ("flywheel_wheel",),
    "hood": ("hood_plate",),
    "gate": ("gate_servo", "servo_front"),
}


def _ghost_topology_part(row: dict[str, Any]) -> dict[str, Any]:
    part = copy.deepcopy(row)
    part["collision"] = []
    part.pop("visualAsset", None)
    part["massKg"] = max(float(part.get("massKg") or 0.0), 0.05)
    if not (isinstance(part.get("inertiaKgM2"), list) and len(part.get("inertiaKgM2") or []) == 3):
        part["inertiaKgM2"] = [1e-4, 1e-4, 1e-4]
    return part


def _rewire_scoring_actuators(preset: dict[str, Any], template: dict[str, Any]) -> None:
    """Bind scoring actuators to catalog hinges; ghost 4-cap hood/gate parts only if needed."""
    joints = list(preset.get("joints") or [])
    parts = list(preset.get("rigidParts") or [])
    by_child = {str(row.get("childPartId") or ""): row for row in joints}
    joint_ids = {str(row.get("id") or "") for row in joints}
    part_ids = {str(row.get("id") or "") for row in parts}
    template_parts = {str(row.get("id") or ""): row for row in template.get("rigidParts") or []}
    template_joints = {str(row.get("id") or ""): row for row in template.get("joints") or []}
    template_by_child = {str(row.get("childPartId") or ""): row for row in template.get("joints") or []}

    def ensure_ghost_child(child_id: str) -> None:
        if child_id in part_ids or child_id == "chassis":
            return
        part = template_parts.get(child_id)
        joint = template_by_child.get(child_id)
        if part is None or joint is None:
            return
        parent_id = str(joint.get("parentPartId") or "chassis")
        if parent_id not in part_ids and parent_id != "chassis":
            ensure_ghost_child(parent_id)
        parts.append(_ghost_topology_part(part))
        part_ids.add(child_id)
        ident = str(joint.get("id") or "")
        if ident and ident not in joint_ids:
            joints.append(copy.deepcopy(joint))
            joint_ids.add(ident)
            by_child[child_id] = joints[-1]

    actuators: list[dict[str, Any]] = []
    for row in preset.get("actuators") or []:
        if not isinstance(row, dict):
            continue
        actuator = copy.deepcopy(row)
        ident = str(actuator.get("id") or "")
        catalog_joint = next(
            (
                by_child[child_id]
                for child_id in _ACTUATOR_CATALOG_CHILDREN.get(ident, ())
                if child_id in by_child and str(by_child[child_id].get("type") or "fixed") != "fixed"
            ),
            None,
        )
        if catalog_joint is not None:
            actuator["jointId"] = catalog_joint["id"]
            actuators.append(actuator)
            continue
        joint_id = str(actuator.get("jointId") or "")
        if joint_id in joint_ids:
            actuators.append(actuator)
            continue
        tmpl = template_joints.get(joint_id)
        if tmpl is None:
            if str(actuator.get("kind") or "") not in {"position_motor", "servo"}:
                actuator.pop("jointId", None)
            actuators.append(actuator)
            continue
        ensure_ghost_child(str(tmpl.get("childPartId") or ""))
        if str(tmpl.get("id") or "") not in joint_ids:
            joints.append(copy.deepcopy(tmpl))
            joint_ids.add(str(tmpl["id"]))
        actuator["jointId"] = tmpl["id"]
        actuators.append(actuator)
    preset["actuators"] = actuators
    preset["joints"] = joints
    preset["rigidParts"] = parts


def _catalog_mechanism_tags(
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
) -> set[str]:
    tags: set[str] = set()
    for ident, part in catalog_parts.items():
        if ident not in instances:
            continue
        tags.update(str(tag).lower() for tag in (part.get("tags") or []))
    return tags


def _require_scoring_catalog_parts(
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
) -> None:
    tags = _catalog_mechanism_tags(instances, catalog_parts)
    missing = [name for name in ("intake_roller", "flywheel") if name not in tags]
    if missing:
        raise AssemblyError(
            "launch-capable scoring robots require catalog parts tagged " + " and ".join(missing)
        )


def _scoring_topology_requested(bindings: dict[str, Any], instances: dict[str, dict[str, Any]], catalog_parts: dict[str, dict[str, Any]]) -> bool:
    if bindings.get("includeScoringTopology"):
        return True
    roles = bindings.get("instanceRoles") if isinstance(bindings.get("instanceRoles"), dict) else {}
    if any(str(role) in {"intake", "launcher", "flywheel", "hood", "gate"} for role in roles.values()):
        return True
    for ident, part in catalog_parts.items():
        if ident not in instances:
            continue
        tags = {str(tag).lower() for tag in (part.get("tags") or [])}
        if tags.intersection({"intake_roller", "flywheel"}):
            return True
    return False


def _pattern_index(ref: dict[str, Any]) -> tuple[int, int] | None:
    index = ref.get("patternIndex")
    if not index:
        return None
    return int(index[0]), int(index[1])


def _catalog_parts(instances: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    parts: dict[str, dict[str, Any]] = {}
    for ident, row in instances.items():
        parts[ident] = get_part(str(row["sku"]))
    return parts


def _secondary_payload(
    connection: dict[str, Any],
    parent_part: dict[str, Any],
    child_part: dict[str, Any],
) -> dict[str, Any] | None:
    secondary = connection.get("secondary")
    if not secondary:
        return None
    return {
        "parent_mount": part_mount(parent_part, str(secondary["parent"]["mountId"])),
        "child_mount": part_mount(child_part, str(secondary["child"]["mountId"])),
        "parent_index": _pattern_index(secondary["parent"]),
        "child_index": _pattern_index(secondary["child"]),
    }


def solve_assembly_poses(
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    root_id: str,
    order: tuple[str, ...],
    by_child: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    root_row = instances[root_id]
    if root_row.get("pose"):
        poses[root_id] = matrix_from_pose(root_row["pose"])
    else:
        poses[root_id] = np.eye(4, dtype=np.float64)
    for ident in order:
        if ident == root_id:
            continue
        connection = by_child[ident]
        parent_id = str(connection["parent"]["instanceId"])
        if parent_id not in poses:
            raise AssemblyError(f"connection {connection.get('id')} parent {parent_id} is not placed")
        parent_part = catalog_parts[parent_id]
        child_part = catalog_parts[ident]
        parent_mount, child_mount = validate_connection_mounts(connection, parent_part, child_part)
        if instances[ident].get("pose"):
            raise AssemblyError(f"instance {ident} pose is solved from connections and must be omitted")
        poses[ident] = solve_mount_transform(
            poses[parent_id],
            parent_mount,
            child_mount,
            parent_index=_pattern_index(connection["parent"]),
            child_index=_pattern_index(connection["child"]),
            spin_deg=float(connection.get("spinDeg") or 0.0),
            secondary=_secondary_payload(connection, parent_part, child_part),
        )
    return poses


def _catalog_collision(part: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proxy in part.get("collision") or []:
        kind = str(proxy.get("kind") or "box")
        row: dict[str, Any] = {
            "kind": "convex_mesh" if kind == "convex_hull" else kind,
            "friction": _DEFAULT_FRICTION,
            "restitution": _DEFAULT_RESTITUTION,
        }
        if proxy.get("sizeIn"):
            row["sizeIn"] = list(proxy["sizeIn"])
        if proxy.get("radiusIn") is not None:
            row["radiusIn"] = proxy["radiusIn"]
        if proxy.get("lengthIn") is not None:
            row["lengthIn"] = proxy["lengthIn"]
        if isinstance(proxy.get("pose"), dict):
            row["pose"] = dict(proxy["pose"])
        if row["kind"] == "convex_mesh":
            continue
        rows.append(row)
    if not rows:
        raise AssemblyError(f"catalog part {part.get('sku')} has no usable collision proxy")
    return rows


def _visual_asset(part: dict[str, Any]) -> str | None:
    status = cache_entry(str(part.get("manufacturer") or ""), str(part.get("sku") or ""))
    visual = status.get("visualAsset")
    return str(visual) if visual else None


def _catalog_joint(
    connection: dict[str, Any],
    joint_type: str,
    parent_id: str,
    child_id: str,
    parent_part: dict[str, Any],
    reserved: set[str],
) -> dict[str, Any]:
    ident = f"{child_id}_joint"
    if ident in reserved:
        ident = f"catalog_{child_id}_joint"
    parent_mount = part_mount(parent_part, str(connection["parent"]["mountId"]))
    anchor_local = hole_in_part(parent_mount, _pattern_index(connection["parent"]))
    joint: dict[str, Any] = {
        "id": ident,
        "type": joint_type,
        "parentPartId": parent_id,
        "childPartId": child_id,
        "anchorIn": {"x": float(anchor_local[0]), "y": float(anchor_local[1]), "z": float(anchor_local[2])},
    }
    if joint_type != "fixed":
        axis = mount_axis(parent_mount)
        joint["axis"] = [float(axis[0]), float(axis[1]), float(axis[2])]
        joint["limit"] = list(_CONTINUOUS_LIMIT)
        joint["damping"] = 0.001
        joint["frictionLoss"] = 0.001
        joint["backlash"] = 0
    return joint


def _template_ids(template: dict[str, Any]) -> set[str]:
    ids = {str(row.get("id") or "") for row in template.get("rigidParts") or []}
    ids.update(str(row.get("id") or "") for row in template.get("joints") or [])
    ids.discard("")
    return ids


def _topology_mechanism_mass(preset: dict[str, Any]) -> float:
    total = 0.0
    for row in preset.get("rigidParts") or []:
        if str(row.get("id") or "") in _TOPOLOGY_MECHANISM_IDS:
            total += float(row.get("massKg") or 0.0)
    return total


def _piece_path_fits_chassis(preset: dict[str, Any], length: float, width: float, height: float) -> bool:
    piece_path = preset.get("piecePath") or {}
    slots = list(piece_path.get("storageSlots") or [])
    capacity = int((preset.get("mechanisms") or {}).get("capacity") or 0)
    half_length = 0.5 * length
    half_width = 0.5 * width
    piece_radius = 1.4
    for slot in slots[:capacity]:
        if abs(float(slot.get("x") or 0)) + piece_radius > half_length:
            return False
        if abs(float(slot.get("y") or 0)) + piece_radius > half_width:
            return False
        if not piece_radius <= float(slot.get("z") or 0) <= height - piece_radius:
            return False
    muzzle = piece_path.get("muzzlePose") or {}
    clearance = float(piece_path.get("muzzleClearanceIn") or 0)
    return (
        abs(float(muzzle.get("x") or 0)) >= half_length + clearance
        or abs(float(muzzle.get("y") or 0)) >= half_width + clearance
        or float(muzzle.get("z") or 0) >= height + clearance
    )


def _apply_inferred_contract(
    preset: dict[str, Any],
    report: InferenceReport,
    bindings: dict[str, Any],
    *,
    include_scoring: bool = False,
) -> None:
    drivetrain = dict(preset.get("drivetrain") or {})
    inferred = {
        key: value
        for key, value in (report.drivetrain or {}).items()
        if key in {"type", "trackWidthIn", "wheelbaseIn", "wheelDiameterIn", "strafeMultiplier"}
        and value is not None
        and not (key in {"trackWidthIn", "wheelbaseIn"} and float(value) <= 0)
    }
    drivetrain.update(inferred)
    bound = bindings.get("drivetrain") if isinstance(bindings.get("drivetrain"), dict) else None
    if bound:
        drivetrain.update(bound)
    preset["drivetrain"] = drivetrain

    motors = dict(preset.get("motors") or {})
    motor_ids = {path.motor_id for path in report.transmissions if path.motor_id}
    if motor_ids:
        motors["count"] = len(motor_ids)
    if report.drivetrain.get("gearRatio"):
        motors["gearRatio"] = float(report.drivetrain["gearRatio"])
    if report.drivetrain.get("freeSpeedRpm"):
        motors["freeSpeedRpm"] = float(report.drivetrain["freeSpeedRpm"])
    preset["motors"] = motors

    catalog_mass = report.mass.mass_kg if report.mass is not None else 0.0
    mechanism_mass = _topology_mechanism_mass(preset)
    if catalog_mass > 0:
        preset.setdefault("chassis", {})
        preset["chassis"]["massKg"] = catalog_mass + mechanism_mass
        for row in preset.get("rigidParts") or []:
            if str(row.get("id") or "") == "chassis":
                row["massKg"] = max(catalog_mass, 1e-6)
                break
    if report.mass is not None:
        length = report.mass.aabb_max_in[0] - report.mass.aabb_min_in[0]
        width = report.mass.aabb_max_in[1] - report.mass.aabb_min_in[1]
        height = report.mass.aabb_max_in[2] - report.mass.aabb_min_in[2]
        chassis = preset.setdefault("chassis", {})
        template_height = float(chassis.get("heightIn") or 14.0)
        if _piece_path_fits_chassis(preset, length, width, height):
            chassis["lengthIn"] = max(length, 1.0)
            chassis["widthIn"] = max(width, 1.0)
            chassis["heightIn"] = max(height, 1.0)
            if include_scoring:
                chassis["heightIn"] = max(float(chassis["heightIn"]), template_height, 14.0)
    if report.confirmed:
        preset["defaultActionTier"] = "physical_actuators"
    else:
        preset["defaultActionTier"] = "high_level_waypoint"


def _catalog_rigid_part(
    ident: str,
    part: dict[str, Any],
    parent_id: str,
    pose: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": ident,
        "parentId": parent_id,
        "pose": pose,
        "massKg": float(part["massKg"]),
        "collision": _catalog_collision(part),
    }
    if part.get("inertiaKgM2"):
        row["inertiaKgM2"] = list(part["inertiaKgM2"])
    if part.get("tags"):
        row["tags"] = list(part["tags"])
    visual = _visual_asset(part)
    if visual:
        row["visualAsset"] = visual
    return row


def materialize_physical_preset(
    document: dict[str, Any],
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    root_id: str,
    poses: dict[str, np.ndarray],
    by_child: dict[str, dict[str, Any]],
    report: InferenceReport,
) -> dict[str, Any]:
    bindings = document.get("functionalBindings") if isinstance(document.get("functionalBindings"), dict) else {}
    include_scoring = _scoring_topology_requested(bindings, instances, catalog_parts)
    if include_scoring:
        _require_scoring_catalog_parts(instances, catalog_parts)
    full_template = _load_topology()
    template = _minimal_chassis_parent(_strip_scoring_topology(copy.deepcopy(full_template)))
    if include_scoring:
        _restore_scoring_contract(template, full_template)
    reserved = _template_ids(template)
    collisions = {ident for ident in instances if ident in reserved}
    if collisions:
        raise AssemblyError(f"instance ids reserved by the physical topology template: {sorted(collisions)}")
    preset = template
    preset["schemaVersion"] = PHYSICAL_SCHEMA_VERSION
    preset["id"] = str(document["id"])
    preset["displayName"] = str(document.get("displayName") or document["id"])
    if include_scoring and not bindings.get("includeScoringTopology") and not bindings.get("confirmed"):
        pass

    chassis_world = np.eye(4, dtype=np.float64)
    catalog_parts_out: list[dict[str, Any]] = []
    catalog_joints: list[dict[str, Any]] = []
    catalog_joints.append(
        {
            "id": f"{root_id}_joint",
            "type": "fixed",
            "parentPartId": "chassis",
            "childPartId": root_id,
        }
    )
    catalog_parts_out.append(
        _catalog_rigid_part(
            root_id,
            catalog_parts[root_id],
            "chassis",
            pose_from_matrix(relative_transform(chassis_world, poses[root_id])),
        )
    )
    for ident, connection in by_child.items():
        parent_id = str(connection["parent"]["instanceId"])
        parent_part = catalog_parts[parent_id]
        child_part = catalog_parts[ident]
        joint_type = infer_connection_joint(connection, parent_part, child_part)
        pose = pose_from_matrix(relative_transform(poses[parent_id], poses[ident]))
        catalog_parts_out.append(_catalog_rigid_part(ident, child_part, parent_id, pose))
        catalog_joints.append(_catalog_joint(connection, joint_type, parent_id, ident, parent_part, reserved))

    preset["rigidParts"] = list(preset.get("rigidParts") or []) + catalog_parts_out
    preset["joints"] = list(preset.get("joints") or []) + catalog_joints
    if include_scoring:
        _rewire_scoring_actuators(preset, full_template)
    _apply_inferred_contract(preset, report, bindings, include_scoring=include_scoring)
    return preset


def _cad_warnings(catalog_parts: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    warnings: list[dict[str, Any]] = []
    if not cad_extra_available():
        warnings.append(
            {
                "code": "cad_extra",
                "severity": "warning",
                "message": "Official catalog visuals require pip install -e '.[cad]'; authored collision proxies are shown instead.",
            }
        )
    disabled: list[str] = []
    missing: list[str] = []
    for part in catalog_parts.values():
        sku = str(part.get("sku") or "")
        cad = part.get("cad") if isinstance(part.get("cad"), dict) else {}
        status = cache_entry(str(part.get("manufacturer") or ""), sku)
        if not cad.get("downloadEnabled"):
            disabled.append(sku)
        elif status.get("state") != "ready":
            missing.append(sku)
    if disabled:
        warnings.append(
            {
                "code": "cad_unavailable",
                "severity": "warning",
                "message": "Official CAD download is disabled for " + ", ".join(sorted(set(disabled))),
            }
        )
    if missing:
        shown = sorted(set(missing))
        extra = "" if len(shown) <= 8 else f" (+{len(shown) - 8} more)"
        warnings.append(
            {
                "code": "cad_missing",
                "severity": "warning",
                "message": "Official visuals are not cached for " + ", ".join(shown[:8]) + extra,
            }
        )
    return tuple(warnings)


# Include source content and catalog inputs so edits cannot reuse an obsolete robot.
_MATERIALIZED_ROBOTS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MATERIALIZED_CACHE_LIMIT = 16


def clear_materialize_cache() -> None:
    _MATERIALIZED_ROBOTS.clear()


def materialize_sim_robot(preset: dict[str, Any], *, competitive: bool = True) -> dict[str, Any]:
    """Return a 1.1 physical preset, compiling schema 1.2 assemblies when needed."""
    if str(preset.get("schemaVersion")) != ASSEMBLY_SCHEMA_VERSION:
        return preset
    instances = ((preset.get("assembly") or {}).get("instances") or [])
    skus = sorted({str(row.get("sku") or "") for row in instances})
    catalog_inputs = []
    for sku in skus:
        part = get_part(sku)
        catalog_inputs.append((part, cache_entry(str(part["manufacturer"]), sku)))
    payload = json.dumps(
        (preset, bool(competitive), catalog_inputs), sort_keys=True, separators=(",", ":"), default=str
    )
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    cached = _MATERIALIZED_ROBOTS.get(key)
    if cached is not None:
        _MATERIALIZED_ROBOTS.move_to_end(key)
        return copy.deepcopy(cached)
    compiled = compile_assembly_to_preset(preset, competitive=competitive).preset
    _MATERIALIZED_ROBOTS[key] = copy.deepcopy(compiled)
    if len(_MATERIALIZED_ROBOTS) > _MATERIALIZED_CACHE_LIMIT:
        _MATERIALIZED_ROBOTS.popitem(last=False)
    return compiled


def compile_assembly_to_preset(
    document: dict[str, Any],
    *,
    competitive: bool = True,
) -> CompiledAssembly:
    """Validate a 1.2 assembly and materialize a 1.1 preset for compile_robot_preset()."""
    if str(document.get("schemaVersion")) != ASSEMBLY_SCHEMA_VERSION:
        raise AssemblyError(f"compile_assembly_to_preset requires schema {ASSEMBLY_SCHEMA_VERSION}")
    errors = validate_document("robot", document)
    if errors:
        raise AssemblyError("assembly failed schema: " + "; ".join(errors[:8]))
    assembly = document.get("assembly") if isinstance(document.get("assembly"), dict) else {}
    root_id, order, by_child = validate_assembly_graph(assembly)
    instances = {row["id"]: row for row in assembly["instances"]}
    catalog_parts = _catalog_parts(instances)
    connections = list(assembly.get("connections") or [])
    for connection in connections:
        parent_id = str(connection["parent"]["instanceId"])
        child_id = str(connection["child"]["instanceId"])
        validate_connection_mounts(connection, catalog_parts[parent_id], catalog_parts[child_id])
    validate_occupancy(connections)
    poses = solve_assembly_poses(instances, catalog_parts, root_id, order, by_child)
    validate_assembly_collisions(poses, instances, catalog_parts, connections)
    bindings = document.get("functionalBindings") if isinstance(document.get("functionalBindings"), dict) else {}
    report = build_inference_report(instances, catalog_parts, by_child, bindings=bindings, poses=poses)
    inferred_warnings = validate_inferred_assembly(poses, instances, catalog_parts, by_child, report)
    preset = materialize_physical_preset(document, instances, catalog_parts, root_id, poses, by_child, report)
    compiled = compile_robot_preset(preset, competitive=competitive)
    document_warnings = tuple(document["warnings"]) if isinstance(document.get("warnings"), list) else ()
    overlap_warnings = connected_overlap_warnings(poses, instances, catalog_parts, connections)
    return CompiledAssembly(
        preset=compiled.preset,
        compiled=compiled,
        instance_poses={ident: pose_from_matrix(matrix) for ident, matrix in poses.items()},
        report=report,
        warnings=document_warnings + inferred_warnings + overlap_warnings + _cad_warnings(catalog_parts),
    )
