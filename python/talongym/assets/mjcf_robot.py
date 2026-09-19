"""MJCF robot bodies: catalog collision, mass/inertia, welded structure, wheel placement."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from talongym.assets.mjcf_field import _robot_axis, _robot_pos, _robot_quat, _xml_name
from talongym.robot.transforms import invert_transform, matrix_from_pose, pose_from_matrix, transform_direction

M_TO_IN = 39.37007874015748
IN2 = M_TO_IN * M_TO_IN
_WHEEL_CLEARANCE_IN = 0.2
_MECHANISM_PART_IDS = frozenset(
    {"chassis", "intake_roller", "conveyor_roller", "flywheel", "hood", "release_gate"}
)
_WHEEL_ID_RE = re.compile(r"(^|_)wheels?($|_)")


def _copy_part(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    if isinstance(row.get("pose"), dict):
        copied["pose"] = dict(row["pose"])
    if isinstance(row.get("centerOfMassIn"), dict):
        copied["centerOfMassIn"] = dict(row["centerOfMassIn"])
    copied["collision"] = [dict(item) if isinstance(item, dict) else item for item in (row.get("collision") or [])]
    return copied


def robot_part_map(robot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): _copy_part(row)
        for row in ((robot or {}).get("rigidParts") or [])
        if row.get("id")
    }


def robot_joints_by_child(robot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(row["childPartId"]): row
        for row in ((robot or {}).get("joints") or [])
        if row.get("childPartId")
    }


def robot_children(parts: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for part_id, row in parts.items():
        parent = row.get("parentId")
        if parent is not None:
            children.setdefault(str(parent), []).append(part_id)
    return children


def robot_root_id(parts: dict[str, dict[str, Any]]) -> str:
    return next((part_id for part_id, row in parts.items() if row.get("parentId") is None), "chassis")


def joint_is_fixed(joint: dict[str, Any] | None) -> bool:
    return joint is None or str(joint.get("type") or "fixed") == "fixed"


def is_drive_wheel(part_id: str, part: dict[str, Any], joint: dict[str, Any] | None, robot: dict[str, Any] | None) -> bool:
    ident = str(part_id).lower()
    if part_id in _MECHANISM_PART_IDS or "flywheel" in ident:
        return False
    if _WHEEL_ID_RE.search(ident) or ident.endswith("wheel") or ident.startswith("wheel"):
        return True
    diameter = float(((robot or {}).get("drivetrain") or {}).get("wheelDiameterIn") or 0.0)
    if diameter <= 0.0 or joint is None or str(joint.get("type") or "") != "hinge":
        return False
    for collision in part.get("collision") or []:
        kind = str(collision.get("kind") or "")
        if kind not in {"cylinder", "sphere", "capsule"}:
            continue
        radius = float(collision.get("radiusIn") or 0.0)
        if abs(2.0 * radius - diameter) < 0.25:
            return True
    return False


def _wheel_radius(part: dict[str, Any]) -> float:
    best = 0.0
    for collision in part.get("collision") or []:
        if collision.get("radiusIn") is not None:
            best = max(best, float(collision["radiusIn"]))
        size = list(collision.get("sizeIn") or [])
        if len(size) >= 3:
            best = max(best, 0.5 * min(float(size[0]), float(size[1]), float(size[2])))
    return best


def compose_part_poses(parts: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    children = robot_children(parts)
    root_id = robot_root_id(parts)
    poses: dict[str, np.ndarray] = {root_id: np.eye(4, dtype=np.float64)}
    stack = list(children.get(root_id, []))
    while stack:
        part_id = stack.pop()
        parent_id = str(parts[part_id].get("parentId") or root_id)
        parent_pose = poses.get(parent_id)
        if parent_pose is None:
            continue
        poses[part_id] = parent_pose @ matrix_from_pose(parts[part_id].get("pose"))
        stack.extend(children.get(part_id, []))
    return poses


def apply_wheel_floor_offset(
    parts: dict[str, dict[str, Any]],
    joints_by_child: dict[str, dict[str, Any]],
    robot: dict[str, Any] | None,
    robot_hz: float,
) -> dict[str, Any]:
    """Translate chassis-child subtrees so drive wheels sit with the chassis floor plane."""
    poses = compose_part_poses(parts)
    wheels = [
        part_id
        for part_id, part in parts.items()
        if is_drive_wheel(part_id, part, joints_by_child.get(part_id), robot)
    ]
    stats: dict[str, Any] = {"wheelPartIds": list(wheels), "wheelOffsetZ": 0.0, "wheelContactZ": None}
    if not wheels:
        return stats
    bottoms: list[float] = []
    for part_id in wheels:
        pose = poses.get(part_id)
        if pose is None:
            continue
        radius = _wheel_radius(parts[part_id])
        bottoms.append(float(pose[2, 3]) - radius)
    if not bottoms:
        return stats
    contact_z = -float(robot_hz)
    offset = contact_z - min(bottoms)
    stats["wheelOffsetZ"] = offset
    stats["wheelContactZ"] = min(bottoms) + offset
    root_id = robot_root_id(parts)
    children = robot_children(parts)

    def subtree_has_wheel(part_id: str) -> bool:
        if part_id in wheels:
            return True
        return any(subtree_has_wheel(child) for child in children.get(part_id, []))

    for child in children.get(root_id, []):
        if not subtree_has_wheel(child):
            continue
        pose = parts[child].setdefault("pose", {"x": 0.0, "y": 0.0, "z": 0.0})
        pose["z"] = float(pose.get("z") or 0.0) + offset
    return stats


def _pose_is_identity(pose: dict[str, Any] | None) -> bool:
    row = pose or {}
    return all(abs(float(row.get(key) or 0.0)) < 1e-9 for key in ("x", "y", "z", "rollDeg", "pitchDeg", "yawDeg"))


def _quat_align_z(axis_ftc: list[float] | tuple[float, ...] | None) -> str | None:
    if not axis_ftc or len(axis_ftc) != 3:
        return None
    target = np.array(_robot_axis(list(axis_ftc)).split(), dtype=np.float64)
    norm = float(np.linalg.norm(target))
    if norm < 1e-9:
        return None
    target = target / norm
    source = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-8:
        return "1.00000000 0.00000000 0.00000000 0.00000000"
    if cosine < -1.0 + 1e-8:
        return "0.00000000 1.00000000 0.00000000 0.00000000"
    axis = np.cross(source, target)
    axis = axis / float(np.linalg.norm(axis))
    half = 0.5 * math.acos(cosine)
    qw = math.cos(half)
    qx, qy, qz = math.sin(half) * axis
    return f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f}"


def _joint_axis_in_part(
    part: dict[str, Any],
    joint: dict[str, Any] | None,
) -> list[float] | None:
    if joint is None or joint.get("type") == "fixed":
        return None
    axis = list(joint.get("axis") or [])
    if len(axis) != 3:
        return None
    axis_child = transform_direction(invert_transform(matrix_from_pose(part.get("pose"))), np.array(axis, dtype=np.float64))
    return [float(axis_child[0]), float(axis_child[1]), float(axis_child[2])]


def _diaginertia_kg_in2(part: dict[str, Any], mass: float) -> tuple[float, float, float]:
    authored = part.get("inertiaKgM2")
    if isinstance(authored, list) and len(authored) == 3:
        ix, iy, iz = (max(1e-8, float(authored[0]) * IN2), max(1e-8, float(authored[1]) * IN2), max(1e-8, float(authored[2]) * IN2))
        return ix, iz, iy
    collision = (part.get("collision") or [{}])[0]
    kind = str(collision.get("kind") or "box")
    if kind == "box":
        size = list(collision.get("sizeIn") or [1.0, 1.0, 1.0])
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        ixx = mass * (sy * sy + sz * sz) / 12.0
        iyy = mass * (sx * sx + sz * sz) / 12.0
        izz = mass * (sx * sx + sy * sy) / 12.0
        return max(1e-8, ixx), max(1e-8, izz), max(1e-8, iyy)
    radius = float(collision.get("radiusIn") or 0.5)
    length = float(collision.get("lengthIn") or 2.0 * radius)
    i_axis = 0.5 * mass * radius * radius
    i_radial = 0.25 * mass * radius * radius + mass * length * length / 12.0
    return max(1e-8, i_radial), max(1e-8, i_axis), max(1e-8, i_radial)


def _inertial_xml(part: dict[str, Any], indent: str) -> str | None:
    mass = float(part.get("massKg") or 0.0)
    if mass <= 0.0:
        return None
    ix, iy, iz = _diaginertia_kg_in2(part, mass)
    pos = _robot_pos(part.get("centerOfMassIn"))
    return f'{indent}<inertial pos="{pos}" mass="{mass:.6f}" diaginertia="{ix:.8f} {iy:.8f} {iz:.8f}"/>'


def _geom_quat(collision: dict[str, Any], axis_ftc: list[float] | None) -> str:
    pose = collision.get("pose") if isinstance(collision.get("pose"), dict) else None
    if pose and not _pose_is_identity(pose):
        return _robot_quat(pose)
    aligned = _quat_align_z(axis_ftc) if axis_ftc is not None else None
    if aligned:
        return aligned
    return _robot_quat(pose)


def _collision_geoms(
    rid: str,
    part_id: str,
    part: dict[str, Any],
    *,
    mesh_names: dict[str, str],
    frame_pose: dict[str, float] | None = None,
    mass_on_geom: bool = True,
    density: float | None = None,
    axis_ftc: list[float] | None = None,
    indent: str = "        ",
) -> list[str]:
    rows: list[str] = []
    collisions = list(part.get("collision") or [])
    mass_each = float(part.get("massKg") or 0.1) / max(1, len(collisions))
    for index, collision in enumerate(collisions):
        kind = str(collision.get("kind") or "box")
        geom_pose = dict(collision.get("pose") or {})
        if frame_pose is not None:
            geom_pose = pose_from_matrix(matrix_from_pose(frame_pose) @ matrix_from_pose(collision.get("pose")))
        attrs = [
            f'name="{rid}_part_{_xml_name(part_id)}_geom_{index}"',
            'class="robot"',
            f'pos="{_robot_pos(geom_pose)}"',
            f'quat="{_geom_quat({"pose": geom_pose} if frame_pose is not None else collision, axis_ftc if frame_pose is None else None)}"',
        ]
        if mass_on_geom:
            attrs.append(f'mass="{mass_each:.6f}"')
        if density is not None:
            attrs.append(f'density="{density:.6f}"')
        friction = float(collision.get("friction") or 0.8)
        attrs.append(f'friction="{friction:.4f} 0.05 0.01"')
        if kind == "box":
            size = list(collision.get("sizeIn") or [1.0, 1.0, 1.0])
            attrs.extend(
                [
                    'type="box"',
                    f'size="{0.5 * float(size[0]):.5f} {0.5 * float(size[2]):.5f} {0.5 * float(size[1]):.5f}"',
                ]
            )
        elif kind in {"sphere", "cylinder", "capsule"}:
            attrs.append(f'type="{kind}"')
            radius = float(collision.get("radiusIn") or 0.5)
            if kind == "sphere":
                attrs.append(f'size="{radius:.5f}"')
            else:
                half_length = 0.5 * float(collision.get("lengthIn") or 0.0)
                attrs.append(f'size="{radius:.5f} {half_length:.5f}"')
        elif kind == "convex_mesh":
            attrs.extend(
                [
                    'type="mesh"',
                    f'mesh="{mesh_names[f"{part_id}:{index}"]}"',
                ]
            )
        rows.append(f"{indent}<geom " + " ".join(attrs) + "/>")
    return rows


def emit_robot_bodies(
    n_robots: int,
    robot_hx: float,
    robot_hy: float,
    robot_hz: float,
    *,
    mesh_file: str | None,
    body_y: float,
    robot: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    from xml.sax.saxutils import escape

    robot_ids = ["red_0", "red_1", "blue_0", "blue_1"][: max(1, n_robots)]
    bodies: list[str] = []
    assets: list[str] = []
    parts = robot_part_map(robot)
    joints_by_child = robot_joints_by_child(robot)
    wheel_stats = apply_wheel_floor_offset(parts, joints_by_child, robot, robot_hz) if parts else {}
    children = robot_children(parts)
    chassis_poses = compose_part_poses(parts) if parts else {}

    mesh_names: dict[str, str] = {}
    if parts:
        from talongym.assets.import_robot_cad import resolve_robot_asset

        for part_id, part in parts.items():
            for collision_index, collision in enumerate(part.get("collision") or []):
                if collision.get("kind") != "convex_mesh":
                    continue
                rel = str(collision["asset"])
                mesh_name = _xml_name(f"robot_part_{part_id}_{collision_index}")
                mesh_names[f"{part_id}:{collision_index}"] = mesh_name
                path = resolve_robot_asset(rel)
                if not path.is_file():
                    raise FileNotFoundError(f"robot collision asset missing: {path}")
                assets.append(f'    <mesh name="{mesh_name}" file="{escape(str(path.resolve()))}"/>')

    welded_ids: list[str] = []
    articulated_ids: list[str] = []

    def welded_descendants(part_id: str) -> list[str]:
        found: list[str] = []
        for child in children.get(part_id, []):
            if not joint_is_fixed(joints_by_child.get(child)):
                continue
            found.append(child)
            found.extend(welded_descendants(child))
        return found

    def part_body(rid: str, part_id: str, indent: str) -> list[str]:
        part = parts[part_id]
        joint = joints_by_child.get(part_id)
        fixed = joint_is_fixed(joint)
        if fixed:
            welded_ids.append(part_id)
        else:
            articulated_ids.append(part_id)
        body_attrs = (
            f'name="{rid}_part_{_xml_name(part_id)}" '
            f'pos="{_robot_pos(part.get("pose"))}" '
            f'quat="{_robot_quat(part.get("pose"))}"'
        )
        rows = [f"{indent}<body {body_attrs}>"]
        inertial = _inertial_xml(part, indent + "  ")
        if inertial and (fixed or part.get("inertiaKgM2")):
            rows.append(inertial)
        if not fixed and joint is not None:
            joint_type = str(joint["type"])
            anchor = joint.get("anchorIn") or {}
            part_pose = part.get("pose") or {}
            local_anchor = {
                key: float(anchor.get(key) or 0.0) - float(part_pose.get(key) or 0.0)
                for key in ("x", "y", "z")
            }
            limit = list(joint.get("limit") or [])
            range_attr = ""
            if len(limit) == 2:
                lo, hi = float(limit[0]), float(limit[1])
                if joint_type == "hinge":
                    lo, hi = math.radians(lo), math.radians(hi)
                range_attr = f' limited="true" range="{lo:.8f} {hi:.8f}"'
            rows.append(
                f'{indent}  <joint name="{rid}_joint_{_xml_name(str(joint["id"]))}" '
                f'type="{joint_type}" pos="{_robot_pos(local_anchor)}" '
                f'axis="{_robot_axis(joint.get("axis"))}" damping="{float(joint.get("damping") or 0):.6f}" '
                f'frictionloss="{float(joint.get("frictionLoss") or 0):.6f}"{range_attr}/>'
            )
            axis_ftc = _joint_axis_in_part(part, joint)
            mass_on_geom = not bool(part.get("inertiaKgM2"))
            rows.extend(
                _collision_geoms(
                    rid,
                    part_id,
                    part,
                    mesh_names=mesh_names,
                    mass_on_geom=mass_on_geom,
                    density=0.0 if part.get("inertiaKgM2") else None,
                    axis_ftc=axis_ftc,
                    indent=indent + "  ",
                )
            )
        for child in children.get(part_id, []):
            rows.extend(part_body(rid, child, indent + "  "))
        rows.append(f"{indent}</body>")
        return rows

    stats: dict[str, Any] = {
        "weldedPartIds": [],
        "articulatedPartIds": [],
        **wheel_stats,
        "wheelClearanceIn": _WHEEL_CLEARANCE_IN,
    }

    for rid in robot_ids:
        if parts:
            root_id = robot_root_id(parts)
            root = parts[root_id]
            fused: list[str] = []
            fused.extend(
                _collision_geoms(
                    rid,
                    root_id,
                    root,
                    mesh_names=mesh_names,
                    mass_on_geom=not bool(root.get("inertiaKgM2")),
                    density=0.0 if root.get("inertiaKgM2") else None,
                    indent="      ",
                )
            )
            root_inertial = _inertial_xml(root, "      ") if root.get("inertiaKgM2") else None
            for welded_id in welded_descendants(root_id):
                pose = pose_from_matrix(chassis_poses[welded_id]) if welded_id in chassis_poses else parts[welded_id].get("pose")
                fused.extend(
                    _collision_geoms(
                        rid,
                        welded_id,
                        parts[welded_id],
                        mesh_names=mesh_names,
                        frame_pose=pose,
                        mass_on_geom=False,
                        density=0.0,
                        indent="      ",
                    )
                )
            child_rows: list[str] = []
            for child in children.get(root_id, []):
                child_rows.extend(part_body(rid, child, "      "))
            mechanism_rows = fused + child_rows
            if root_inertial:
                mechanism_rows.insert(0, root_inertial)
            mechanism_xml = "\n".join(mechanism_rows)
            bodies.append(
                f"""    <body name="{rid}" pos="0 {body_y:.3f} 0">
      <joint name="{rid}_sx" type="slide" axis="1 0 0" damping="2"/>
      <joint name="{rid}_sz" type="slide" axis="0 0 1" damping="2"/>
      <joint name="{rid}_yaw" type="hinge" axis="0 1 0" damping="0.4"/>
{mechanism_xml}
    </body>"""
            )
            continue
        if mesh_file:
            geom = (
                f'<geom name="{rid}_geom" class="robot" type="mesh" mesh="robot_hull" mass="15" '
                f'rgba="0.9 0.8 0.6 1"/>'
            )
        else:
            geom = (
                f'<geom name="{rid}_geom" class="robot" type="box" '
                f'size="{robot_hx:.3f} {robot_hz:.3f} {robot_hy:.3f}" '
                f'mass="15" rgba="0.9 0.8 0.6 1"/>'
            )
        bodies.append(
            f"""    <body name="{rid}" pos="0 {body_y:.3f} 0">
      <joint name="{rid}_sx" type="slide" axis="1 0 0" damping="2"/>
      <joint name="{rid}_sz" type="slide" axis="0 0 1" damping="2"/>
      <joint name="{rid}_yaw" type="hinge" axis="0 1 0" damping="0.4"/>
      {geom}
    </body>"""
        )
    stats["weldedPartIds"] = list(dict.fromkeys(welded_ids))
    stats["articulatedPartIds"] = list(dict.fromkeys(articulated_ids))
    return bodies, assets, stats


def kinematic_part_transforms(
    robot: dict[str, Any] | None,
    *,
    robot_x: float,
    robot_y: float,
    heading: float,
    origin_z: float,
    robot_hz: float,
) -> list[dict[str, Any]]:
    """Chassis-relative rigid-part poses in world FTC, matching MJCF instance IDs."""
    parts = robot_part_map(robot)
    if not parts:
        return []
    joints_by_child = robot_joints_by_child(robot)
    apply_wheel_floor_offset(parts, joints_by_child, robot, robot_hz)
    poses = compose_part_poses(parts)
    root_id = robot_root_id(parts)
    robot_mat = matrix_from_pose({"x": robot_x, "y": robot_y, "z": origin_z, "yawDeg": math.degrees(heading)})
    rows: list[dict[str, Any]] = []
    for part_id, local in poses.items():
        if part_id == root_id:
            continue
        if part_id in _MECHANISM_PART_IDS and not (parts.get(part_id) or {}).get("collision"):
            continue
        world = robot_mat @ local
        pose = pose_from_matrix(world)
        quat = _robot_quat(pose).split()
        rows.append(
            {
                "id": part_id,
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "z": float(pose["z"]),
                "qw": float(quat[0]),
                "qx": float(quat[1]),
                "qy": float(quat[2]),
                "qz": float(quat[3]),
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
            }
        )
    return rows
