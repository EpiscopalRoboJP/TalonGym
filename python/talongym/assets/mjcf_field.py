"""Emit MuJoCo MJCF for a field preset (Y-up inches, gravity on).

CAD seasons assemble the static field from convex collision parts in the
committed cad manifest — never field.glb and never a single concave hull.
Non-CAD seasons keep the schematic AABB generator.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from talongym.paths import ASSETS_DIR

IN_G = 386.0886  # 9.80665 m/s^2 in inches/s^2
CAD_MJCF_MARKER = "talongym_cad_field"
CAD_MJCF_VERSION = "1.2.0"

# Geom groups used for contact classification (not name substrings).
GEOM_GROUP_FIELD = 0
GEOM_GROUP_ROBOT = 1
GEOM_GROUP_PIECE = 2
GEOM_GROUP_TRIGGER = 3

PLAYABLE_HALF_IN = 72.0
PLAYABLE_MARGIN_IN = 2.0  # perimeter wall / glass thickness
MIN_PIECE_SLOTS = 8
PIECE_SLOT_PAD = 8

# Fasteners and floor markings that do not change HIVE/CELL openings.
# Do not use short tokens like "nut"/"pin"/"rivet" — those hit Peanut, hinge rivets, etc.
_FASTENER_RE = re.compile(
    r"(?i)("
    r"cable[_-]?tie|"
    r"socket_head_cap_screw|"
    r"pan_head_machine_screw|"
    r"sheet_metal_screw|"
    r"_fhts_|"
    r"elevator_bolt|"
    r"wing_nut|"
    r"nylon_spacer|"
    r"under_tile_washer|"
    r"flanged_bearing|"
    r"quick_release_pin|"
    r"press_in_plug|"
    r"10-32_x_"
    r")"
)
_MARKING_RE = re.compile(r"(?i)(gaffer_tape|panel_sticker)")
_UNDER_FIELD_RE = re.compile(r"(?i)(under_tile|under_field)")
_SCORING_ELEMENT_RE = re.compile(r"(?i)(pollen|nectar)")
_FLOOR_TILE_RE = re.compile(r"(?i)(soft_tiles|_field_soft_tiles)")
_ALLIANCE_AREA_RE = re.compile(r"(?i)(artifact_tray)")

_PIECE_RGBA = {
    "pollen": "0.93 0.78 0.16 1",
    "nectar_red": "0.82 0.18 0.16 1",
    "nectar_blue": "0.16 0.38 0.82 1",
}


def _yup(x: float, y: float, z: float) -> tuple[float, float, float]:
    return float(x), float(z), float(-y)


def _size(hx: float, hy: float, hz: float) -> str:
    return f"{hx:.4f} {hy:.4f} {hz:.4f}"


def _robot_pos(pose: dict[str, Any] | None) -> str:
    row = pose or {}
    x, y, z = _yup(
        float(row.get("x") or 0.0),
        float(row.get("y") or 0.0),
        float(row.get("z") or 0.0),
    )
    return f"{x:.5f} {y:.5f} {z:.5f}"


def _robot_axis(axis: list[Any] | None) -> str:
    row = list(axis or [0.0, 0.0, 1.0])
    x, y, z = _yup(float(row[0]), float(row[1]), float(row[2]))
    return f"{x:.6f} {y:.6f} {z:.6f}"


def _robot_quat(pose: dict[str, Any] | None) -> str:
    """Map FTC Z-up roll/pitch/yaw Euler angles to a MuJoCo Y-up quaternion."""
    row = pose or {}
    roll = math.radians(float(row.get("rollDeg") or 0.0))
    pitch = math.radians(float(row.get("pitchDeg") or 0.0))
    yaw = math.radians(float(row.get("yawDeg") or 0.0))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # FTC Rz(yaw) * Ry(pitch) * Rx(roll).
    rf = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    p = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    temp = tuple(
        tuple(sum(p[i][k] * rf[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )
    rm = tuple(
        tuple(sum(temp[i][k] * p[j][k] for k in range(3)) for j in range(3))
        for i in range(3)
    )
    trace = rm[0][0] + rm[1][1] + rm[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rm[2][1] - rm[1][2]) / scale
        qy = (rm[0][2] - rm[2][0]) / scale
        qz = (rm[1][0] - rm[0][1]) / scale
    else:
        diagonal = [rm[0][0], rm[1][1], rm[2][2]]
        index = max(range(3), key=diagonal.__getitem__)
        nxt = (index + 1) % 3
        last = (index + 2) % 3
        scale = math.sqrt(1.0 + rm[index][index] - rm[nxt][nxt] - rm[last][last]) * 2.0
        quat = [0.0, 0.0, 0.0]
        quat[index] = 0.25 * scale
        qw = (rm[last][nxt] - rm[nxt][last]) / scale
        quat[nxt] = (rm[nxt][index] + rm[index][nxt]) / scale
        quat[last] = (rm[last][index] + rm[index][last]) / scale
        qx, qy, qz = quat
    return f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f}"


def _xml_name(text: str, prefix: str = "", maxlen: int = 60) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._") or "x"
    out = f"{prefix}{cleaned}" if prefix else cleaned
    return out[:maxlen]


@dataclass
class CollisionFilterStats:
    manifest_parts: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    floor_y: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifestParts": self.manifest_parts,
            "kept": self.kept,
            "dropped": dict(self.dropped),
            "floorYIn": self.floor_y,
            "notes": list(self.notes),
        }


@dataclass
class FieldMjcf:
    xml: str
    stats: dict[str, Any]
    floor_y: float
    slot_plan: dict[str, int]
    cad: bool


def part_aabb_contains_yup(part: dict[str, Any], x: float, y: float, z: float, *, margin: float = 0.0) -> bool:
    """True if a Y-up inch point is inside the part's committed AABB."""
    mn = part.get("minIn") or []
    mx = part.get("maxIn") or []
    if len(mn) < 3 or len(mx) < 3:
        return False
    pad = float(margin)
    return (
        float(mn[0]) - pad <= float(x) <= float(mx[0]) + pad
        and float(mn[1]) - pad <= float(y) <= float(mx[1]) + pad
        and float(mn[2]) - pad <= float(z) <= float(mx[2]) + pad
    )


def ftc_point_hits_parts(
    parts: list[dict[str, Any]],
    x: float,
    y: float,
    z: float,
    *,
    margin: float = 0.0,
) -> list[str]:
    """Return collision-part ids whose AABBs contain the FTC (x, y floor, z height) point."""
    px, py, pz = _yup(x, y, z)
    hits: list[str] = []
    for part in parts:
        if part_aabb_contains_yup(part, px, py, pz, margin=margin):
            hits.append(str(part.get("id") or part.get("asset") or ""))
    return hits


def classify_collision_part(
    part: dict[str, Any],
    *,
    playable_half: float = PLAYABLE_HALF_IN,
    margin: float = PLAYABLE_MARGIN_IN,
) -> str:
    """Return a drop reason or 'playable'. Never merges parts into one hull."""
    name = str(part.get("id") or part.get("asset") or "")
    from talongym.assets.cad_common import moving_hive_alliance

    mn = part.get("minIn") or [0.0, 0.0, 0.0]
    mx = part.get("maxIn") or [0.0, 0.0, 0.0]
    if moving_hive_alliance(name, 0.5 * (float(mn[0]) + float(mx[0]))):
        return "articulated_hive"
    if _SCORING_ELEMENT_RE.search(name):
        return "cad_scoring_element"
    if _ALLIANCE_AREA_RE.search(name):
        return "alliance_area"
    if _MARKING_RE.search(name):
        return "floor_marking"
    if _UNDER_FIELD_RE.search(name):
        return "under_field"
    if _FLOOR_TILE_RE.search(name):
        return "floor_tile"
    if _FASTENER_RE.search(name):
        return "fastener"
    limit = float(playable_half) + float(margin)
    if float(mn[0]) < -limit or float(mx[0]) > limit:
        return "off_field"
    if float(mn[2]) < -limit or float(mx[2]) > limit:
        return "off_field"
    if float(mx[1]) < -0.5:
        return "off_field"
    return "playable"


def select_field_collision_parts(
    parts: list[dict[str, Any]],
    *,
    playable_half: float = PLAYABLE_HALF_IN,
    margin: float = PLAYABLE_MARGIN_IN,
) -> tuple[list[dict[str, Any]], CollisionFilterStats]:
    stats = CollisionFilterStats(manifest_parts=len(parts))
    tile_tops: list[float] = []
    kept: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            stats.dropped["invalid"] = stats.dropped.get("invalid", 0) + 1
            continue
        name = str(part.get("id") or "")
        if _FLOOR_TILE_RE.search(name):
            mx = part.get("maxIn") or [0.0, 0.0, 0.0]
            tile_tops.append(float(mx[1]))
        reason = classify_collision_part(part, playable_half=playable_half, margin=margin)
        if reason != "playable":
            stats.dropped[reason] = stats.dropped.get(reason, 0) + 1
            continue
        kept.append(part)
    stats.kept = len(kept)
    stats.floor_y = max(tile_tops) if tile_tops else 0.0
    stats.notes = [
        "Each kept part stays its own convex hull; concave assemblies are never merged.",
        "Floor tiles omitted: chassis joints are planar (no heave) and tile hulls would pin robots.",
        "A Y-normal plane at the measured tile top stands in for tile-top piece contact.",
        "Fasteners, tape/stickers, under-field/alliance-area hardware, and CAD POLLEN/NECTAR instances are dropped.",
        "Off-field drop uses AABB vs the 144 in playable rectangle plus wall margin — no clustering.",
    ]
    return kept, stats


def piece_slot_plan(field: dict[str, Any], n_pieces: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spawn in field.get("spawns") or []:
        tid = str(spawn.get("pieceTypeId") or "")
        if not tid:
            continue
        counts[tid] = counts.get(tid, 0) + len(spawn.get("poses") or [])
    for spec in field.get("gamePieces") or []:
        tid = str(spec.get("typeId") or "")
        if not tid:
            continue
        counts[tid] = max(int(counts.get(tid, 0)), 0)
    plan = {tid: max(n + PIECE_SLOT_PAD, MIN_PIECE_SLOTS) for tid, n in counts.items()}
    if not plan:
        plan = {"": int(n_pieces if n_pieces is not None else 80)}
    return plan


def piece_body_name(type_id: str, index: int) -> str:
    tid = _xml_name(type_id or "piece")
    return f"gp_{tid}_{index:02d}"


def load_field_manifest(field: dict[str, Any], *, require: bool | None = None) -> dict[str, Any] | None:
    from talongym.assets.cad_common import CadImportError, mesh_required
    from talongym.assets.cad_manifest import load_manifest, resolve_asset, verify_manifest_files

    rel = field.get("cadManifest")
    needed = mesh_required(field) if require is None else bool(require)
    if not rel:
        if needed:
            raise CadImportError("mesh_field_collision season is missing cadManifest")
        return None
    path = resolve_asset(str(rel))
    if not path.is_file():
        raise CadImportError(f"cadManifest not found: {rel}")
    document = load_manifest(path)
    problems = verify_manifest_files(document, require_field=needed)
    if problems and needed:
        raise CadImportError("CAD assets missing or stale: " + "; ".join(problems[:12]))
    return document


def verify_collision_asset(field: dict[str, Any]) -> Path | None:
    """Confirm the committed collisionAsset exists. CAD seasons must not point at field.glb."""
    from talongym.assets.cad_common import CadImportError
    from talongym.assets.cad_manifest import resolve_asset

    rel = field.get("collisionAsset")
    if not rel:
        return None
    text = str(rel)
    if text.lower().endswith(".glb"):
        raise CadImportError("collisionAsset must not be a render GLB; use convex CAD parts / MJCF")
    path = resolve_asset(text)
    if not path.is_file():
        raise CadImportError(f"collisionAsset not found: {rel}")
    return path


def _piece_spec_map(field: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    specs = {str(p["typeId"]): dict(p) for p in (field.get("gamePieces") or []) if p.get("typeId")}
    man_pieces = (manifest or {}).get("pieces") or {}
    if isinstance(man_pieces, dict):
        for tid, rec in man_pieces.items():
            if not isinstance(rec, dict):
                continue
            row = dict(specs.get(str(tid)) or {"typeId": tid})
            if rec.get("collisionAsset"):
                row["collisionAsset"] = rec["collisionAsset"]
            if rec.get("visualAsset") and not row.get("visualAsset"):
                row["visualAsset"] = rec["visualAsset"]
            if rec.get("measuredDiameterIn") and not row.get("shape"):
                row["shape"] = {"kind": "circle", "radius": float(rec["measuredDiameterIn"]) / 2.0}
            specs[str(tid)] = row
    return specs


def _mesh_file_attr(rel: str, *, mesh_root: Path | None = None) -> str:
    from talongym.assets.cad_manifest import resolve_asset

    path = resolve_asset(rel)
    if not path.is_file():
        from talongym.assets.cad_common import CadImportError

        raise CadImportError(f"collision mesh missing: {rel}")
    if mesh_root is not None:
        try:
            return escape(str(path.resolve().relative_to(Path(mesh_root).resolve())))
        except ValueError:
            return escape(str(path.resolve()))
    return escape(str(path.resolve()))


def _perimeter_glass_proxy_geoms(parts: list[dict[str, Any]]) -> list[str]:
    """Collapse coplanar CAD glass panels into four continuous collision boxes."""
    groups: dict[tuple[str, int], list[tuple[list[float], list[float]]]] = {}
    for part in parts:
        name = str(part.get("id") or "").lower()
        if "field_side_glass" not in name:
            continue
        mn = [float(v) for v in (part.get("minIn") or [])]
        mx = [float(v) for v in (part.get("maxIn") or [])]
        if len(mn) < 3 or len(mx) < 3:
            continue
        dx, dz = mx[0] - mn[0], mx[2] - mn[2]
        axis = "x" if dx <= dz else "z"
        center = 0.5 * ((mn[0] + mx[0]) if axis == "x" else (mn[2] + mx[2]))
        groups.setdefault((axis, 1 if center >= 0 else -1), []).append((mn, mx))

    geoms: list[str] = []
    for (axis, sign), bounds in sorted(groups.items()):
        if axis == "x":
            axis_min = min(mn[0] for mn, _ in bounds)
            axis_max = max(mx[0] for _, mx in bounds)
            tangent_min = min(mn[2] for mn, _ in bounds)
            tangent_max = max(mx[2] for _, mx in bounds)
            pos = (
                0.5 * (axis_min + axis_max),
                0.5 * (min(mn[1] for mn, _ in bounds) + max(mx[1] for _, mx in bounds)),
                0.5 * (tangent_min + tangent_max),
            )
            size = (
                max(0.02, 0.5 * (axis_max - axis_min)),
                max(0.02, 0.5 * (max(mx[1] for _, mx in bounds) - min(mn[1] for mn, _ in bounds))),
                max(0.02, 0.5 * (tangent_max - tangent_min)),
            )
        else:
            axis_min = min(mn[2] for mn, _ in bounds)
            axis_max = max(mx[2] for _, mx in bounds)
            tangent_min = min(mn[0] for mn, _ in bounds)
            tangent_max = max(mx[0] for _, mx in bounds)
            pos = (
                0.5 * (tangent_min + tangent_max),
                0.5 * (min(mn[1] for mn, _ in bounds) + max(mx[1] for _, mx in bounds)),
                0.5 * (axis_min + axis_max),
            )
            size = (
                max(0.02, 0.5 * (tangent_max - tangent_min)),
                max(0.02, 0.5 * (max(mx[1] for _, mx in bounds) - min(mn[1] for mn, _ in bounds))),
                max(0.02, 0.5 * (axis_max - axis_min)),
            )
        geoms.append(
            f'    <geom name="perimeter_glass_{axis}_{"pos" if sign > 0 else "neg"}" '
            f'class="field" type="box" size="{_size(*size)}" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"/>'
        )
    return geoms


def _aabb_element_geoms(field: dict[str, Any]) -> list[str]:
    skip_types = {"tape", "zone"}
    skip_tags = {
        "leave",
        "park",
        "loading_zone",
        "garden",
        "restricted_for_red",
        "restricted_for_blue",
        "perimeter",
    }
    geoms: list[str] = []
    for el in field.get("elements") or []:
        tags = set(el.get("tags") or [])
        if el.get("type") in skip_types or tags.intersection(skip_tags):
            continue
        pose = el.get("pose") or {}
        shape = el.get("shape") or {}
        if shape.get("kind") != "aabb":
            continue
        w = float(shape.get("width") or 8)
        d = float(shape.get("depth") or 8)
        z = float(pose.get("z") or 6)
        is_cell = "cell" in tags or el.get("type") in {"goal", "cell"}
        is_frame = "frame" in tags or el.get("type") == "hive_frame"
        height = float(shape.get("height") or (14 if is_cell else 24 if is_frame else 10))
        px, py, pz = _yup(float(pose.get("x") or 0), float(pose.get("y") or 0), z)
        eid = escape(str(el["id"]))
        if is_cell:
            geoms.append(
                f'    <geom name="{eid}" type="box" size="{_size(w / 2, height / 2, d / 2)}" pos="{px:.3f} {py:.3f} {pz:.3f}" '
                f'rgba="0.7 0.3 0.3 0.25" contype="0" conaffinity="0" group="{GEOM_GROUP_TRIGGER}" density="0"/>'
            )
        else:
            geoms.append(
                f'    <geom name="{eid}" type="box" size="{_size(w / 2, height / 2, d / 2)}" pos="{px:.3f} {py:.3f} {pz:.3f}" '
                f'rgba="0.4 0.35 0.3 1" contype="1" conaffinity="1" group="{GEOM_GROUP_FIELD}"/>'
            )
    return geoms


def _trigger_geoms(field: dict[str, Any]) -> list[str]:
    geoms: list[str] = []
    for el in field.get("elements") or []:
        if not el.get("isTrigger"):
            continue
        pose = el.get("pose") or {}
        shape = el.get("shape") or {}
        if shape.get("kind") != "aabb":
            continue
        w = float(shape.get("width") or 8)
        d = float(shape.get("depth") or 8)
        z = float(pose.get("z") or 6)
        height = float(shape.get("height") or 12)
        px, py, pz = _yup(float(pose.get("x") or 0), float(pose.get("y") or 0), z)
        eid = escape(_xml_name(str(el.get("triggerId") or el["id"]), prefix="trig_"))
        geoms.append(
            f'    <geom name="{eid}" type="box" size="{_size(w / 2, height / 2, d / 2)}" pos="{px:.3f} {py:.3f} {pz:.3f}" '
            f'rgba="0.2 0.6 0.8 0.15" contype="0" conaffinity="0" group="{GEOM_GROUP_TRIGGER}" density="0"/>'
        )
    return geoms


def _mechanism_cell_proxy_geoms(field: dict[str, Any], alliance: str) -> list[str]:
    """Open-top CELL collision derived from its preset and staged NECTAR height."""
    cell_id = f"{alliance}_cell_up"
    cell = next((el for el in field.get("elements") or [] if el.get("id") == cell_id), None)
    if cell is None:
        return []
    pose = cell.get("pose") or {}
    shape = cell.get("shape") or {}
    width = float(shape.get("width") or 20.0)
    depth = float(shape.get("depth") or 14.0)
    spec = next(
        (row for row in field.get("gamePieces") or [] if row.get("typeId") == f"nectar_{alliance}"),
        {},
    )
    radius = float((spec.get("shape") or {}).get("radius") or 1.8)
    staged = next(
        (
            row
            for row in field.get("spawns") or []
            if row.get("initialVolumeId") == cell_id
        ),
        {},
    )
    centers = [float(row.get("z") or 0.0) for row in staged.get("poses") or []]
    floor_y = min(centers) - radius if centers else float(pose.get("z") or 0.0) - 5.0
    wall_height = 9.0
    cx, _cy, cz = _yup(
        float(pose.get("x") or 0.0),
        float(pose.get("y") or 0.0),
        float(pose.get("z") or 0.0),
    )
    wall_y = floor_y + wall_height / 2.0
    common = 'class="field" type="box" density="0" contype="4" conaffinity="10"'
    geoms = [
        f'      <geom name="{cell_id}_floor" {common} size="{_size(width / 2, 0.25, depth / 2)}" pos="{cx:.3f} {floor_y - 0.25:.3f} {cz:.3f}"/>',
        f'      <geom name="{cell_id}_left" {common} size="{_size(0.25, wall_height / 2, depth / 2)}" pos="{cx - width / 2:.3f} {wall_y:.3f} {cz:.3f}"/>',
        f'      <geom name="{cell_id}_right" {common} size="{_size(0.25, wall_height / 2, depth / 2)}" pos="{cx + width / 2:.3f} {wall_y:.3f} {cz:.3f}"/>',
    ]
    # Leave the downhill edge open so a ±60° scored tip actually releases
    # the physical contents. The opposite wall retains staged pieces at rest.
    if alliance == "red":
        geoms.append(
            f'      <geom name="{cell_id}_back" {common} size="{_size(width / 2, wall_height / 2, 0.25)}" pos="{cx:.3f} {wall_y:.3f} {cz + depth / 2:.3f}"/>'
        )
    else:
        geoms.append(
            f'      <geom name="{cell_id}_front" {common} size="{_size(width / 2, wall_height / 2, 0.25)}" pos="{cx:.3f} {wall_y:.3f} {cz - depth / 2:.3f}"/>'
        )
    return geoms


def _flower_cup_proxy_geoms(field: dict[str, Any]) -> list[str]:
    """Open-top FLOWER cups: floor + three walls + an intake doorway with a retaining sill."""
    spec = next((row for row in field.get("gamePieces") or [] if row.get("typeId") == "pollen"), {})
    radius = float((spec.get("shape") or {}).get("radius") or 1.4)
    inner_half = 2.45
    wall_t = 0.25
    gap_half = radius + 0.15
    sill_h = max(2.2, radius + 0.8)
    geoms: list[str] = []
    common = 'class="field" type="box" density="0" contype="4" conaffinity="10"'
    for el in field.get("elements") or []:
        if el.get("type") != "flower" and "flower" not in (el.get("tags") or []):
            continue
        fid = str(el.get("id") or "flower")
        pose = el.get("pose") or {}
        fx, fy = float(pose.get("x") or 0.0), float(pose.get("y") or 0.0)
        spawn = next(
            (row for row in field.get("spawns") or [] if str(row.get("id") or "").startswith(fid)),
            {},
        )
        zs = [float(row.get("z") or 0.0) for row in spawn.get("poses") or []]
        floor_z = (min(zs) - radius) if zs else 0.0
        top_z = (max(zs) + radius) if zs else 10.0
        wall_h = max(8.0, top_z - floor_z + 0.5)
        cx, _cy, cz = _yup(fx, fy, 0.0)
        floor_y = floor_z
        wall_y = floor_y + wall_h / 2.0
        geoms.append(
            f'    <geom name="{fid}_cup_floor" {common} size="{_size(inner_half + wall_t, 0.25, inner_half + wall_t)}" pos="{cx:.3f} {floor_y - 0.25:.3f} {cz:.3f}"/>'
        )
        omit = "x-" if abs(fx) >= abs(fy) and fx > 0 else "x+" if abs(fx) >= abs(fy) else "y-" if fy > 0 else "y+"
        walls = {
            "x-": (cx - inner_half - wall_t / 2, cz, wall_t / 2, inner_half + wall_t, True),
            "x+": (cx + inner_half + wall_t / 2, cz, wall_t / 2, inner_half + wall_t, True),
            "y-": (cx, cz + inner_half + wall_t / 2, inner_half + wall_t, wall_t / 2, False),
            "y+": (cx, cz - inner_half - wall_t / 2, inner_half + wall_t, wall_t / 2, False),
        }
        for key, (wx, wz, hx, hz, along_x) in walls.items():
            if key != omit:
                geoms.append(
                    f'    <geom name="{fid}_cup_{key}" {common} size="{_size(hx, wall_h / 2, hz)}" pos="{wx:.3f} {wall_y:.3f} {wz:.3f}"/>'
                )
                continue
            long_half = hx if not along_x else hz
            pillar_span = max(0.2, long_half - gap_half)
            pillar_half = pillar_span / 2.0
            offset = gap_half + pillar_half
            sill_y = floor_y + sill_h / 2.0
            if along_x:
                geoms.append(
                    f'    <geom name="{fid}_cup_{key}_a" {common} size="{_size(hx, wall_h / 2, pillar_half)}" pos="{wx:.3f} {wall_y:.3f} {wz - offset:.3f}"/>'
                )
                geoms.append(
                    f'    <geom name="{fid}_cup_{key}_b" {common} size="{_size(hx, wall_h / 2, pillar_half)}" pos="{wx:.3f} {wall_y:.3f} {wz + offset:.3f}"/>'
                )
                geoms.append(
                    f'    <geom name="{fid}_cup_{key}_sill" {common} size="{_size(hx, sill_h / 2, gap_half)}" pos="{wx:.3f} {sill_y:.3f} {wz:.3f}"/>'
                )
            else:
                geoms.append(
                    f'    <geom name="{fid}_cup_{key}_a" {common} size="{_size(pillar_half, wall_h / 2, hz)}" pos="{wx - offset:.3f} {wall_y:.3f} {wz:.3f}"/>'
                )
                geoms.append(
                    f'    <geom name="{fid}_cup_{key}_b" {common} size="{_size(pillar_half, wall_h / 2, hz)}" pos="{wx + offset:.3f} {wall_y:.3f} {wz:.3f}"/>'
                )
                geoms.append(
                    f'    <geom name="{fid}_cup_{key}_sill" {common} size="{_size(gap_half, sill_h / 2, hz)}" pos="{wx:.3f} {sill_y:.3f} {wz:.3f}"/>'
                )
    return geoms


def apply_flower_cup_proxies(xml: str, field: dict[str, Any]) -> str:
    """Insert cup proxies into assembled MJCF if a rebuild has not already added them."""
    if "_cup_floor" in xml:
        return xml
    geoms = _flower_cup_proxy_geoms(field)
    if not geoms:
        return xml
    block = "\n".join(geoms)
    marker = "</worldbody>"
    if marker not in xml:
        return xml
    return xml.replace(marker, f"{block}\n  {marker}", 1)


def _robot_bodies(
    n_robots: int,
    robot_hx: float,
    robot_hy: float,
    robot_hz: float,
    *,
    mesh_file: str | None,
    body_y: float,
    robot: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    robot_ids = ["red_0", "red_1", "blue_0", "blue_1"][: max(1, n_robots)]
    bodies: list[str] = []
    assets: list[str] = []
    parts = {
        str(row["id"]): row
        for row in ((robot or {}).get("rigidParts") or [])
        if row.get("id")
    }
    joints_by_child = {
        str(row["childPartId"]): row
        for row in ((robot or {}).get("joints") or [])
        if row.get("childPartId")
    }
    children: dict[str, list[str]] = {}
    for part_id, row in parts.items():
        parent = row.get("parentId")
        if parent is not None:
            children.setdefault(str(parent), []).append(part_id)

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
                assets.append(
                    f'    <mesh name="{mesh_name}" file="{escape(str(path.resolve()))}"/>'
                )

    def collision_geoms(rid: str, part_id: str, part: dict[str, Any]) -> list[str]:
        rows: list[str] = []
        collisions = list(part.get("collision") or [])
        mass_each = float(part.get("massKg") or 0.1) / max(1, len(collisions))
        for index, collision in enumerate(collisions):
            kind = str(collision.get("kind") or "box")
            attrs = [
                f'name="{rid}_part_{_xml_name(part_id)}_geom_{index}"',
                'class="robot"',
                f'mass="{mass_each:.6f}"',
                f'pos="{_robot_pos(collision.get("pose"))}"',
                f'quat="{_robot_quat(collision.get("pose"))}"',
            ]
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
            rows.append("        <geom " + " ".join(attrs) + "/>")
        return rows

    def part_body(rid: str, part_id: str, indent: str) -> list[str]:
        part = parts[part_id]
        joint = joints_by_child.get(part_id)
        body_attrs = (
            f'name="{rid}_part_{_xml_name(part_id)}" '
            f'pos="{_robot_pos(part.get("pose"))}" '
            f'quat="{_robot_quat(part.get("pose"))}"'
        )
        rows = [f"{indent}<body {body_attrs}>"]
        if joint is not None and joint.get("type") != "fixed":
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
        rows.extend(
            line.replace("        ", f"{indent}  ", 1)
            for line in collision_geoms(rid, part_id, part)
        )
        for child in children.get(part_id, []):
            rows.extend(part_body(rid, child, indent + "  "))
        rows.append(f"{indent}</body>")
        return rows

    for rid in robot_ids:
        if parts:
            root_id = next(
                (part_id for part_id, row in parts.items() if row.get("parentId") is None),
                "chassis",
            )
            root = parts[root_id]
            root_geoms = collision_geoms(rid, root_id, root)
            child_rows: list[str] = []
            for child in children.get(root_id, []):
                child_rows.extend(part_body(rid, child, "      "))
            mechanism_xml = "\n".join(root_geoms + child_rows)
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
    return bodies, assets


def _typed_piece_bodies(
    field: dict[str, Any],
    manifest: dict[str, Any] | None,
    slot_plan: dict[str, int],
    *,
    mesh_root: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    from talongym.assets.cad_common import CadImportError

    specs = _piece_spec_map(field, manifest)
    mesh_assets: list[str] = []
    mesh_names: set[str] = set()
    defaults: list[str] = []
    bodies: list[str] = []
    for type_id, n_slots in slot_plan.items():
        spec = specs.get(type_id) or {}
        rel = spec.get("collisionAsset")
        if not rel:
            raise CadImportError(f"game piece {type_id!r} is missing collisionAsset")
        mesh_name = _xml_name(f"{type_id}_hull")
        if mesh_name not in mesh_names:
            mesh_assets.append(f'    <mesh name="{mesh_name}" file="{_mesh_file_attr(str(rel), mesh_root=mesh_root)}"/>')
            mesh_names.add(mesh_name)
        mass = float(spec.get("massKg") or 0.05)
        rest = float(spec.get("restitution") or 0.35)
        damping = max(0.2, min(1.0, 1.2 - rest))
        rgba = _PIECE_RGBA.get(type_id, "0.85 0.85 0.85 1")
        cls = _xml_name(f"piece_{type_id}")
        defaults.append(
            f'    <default class="{cls}">\n'
            f'      <geom group="{GEOM_GROUP_PIECE}" contype="3" conaffinity="3" condim="3" '
            f'friction="0.8 0.05 0.01" solref="0.02 {damping:.3f}" mass="{mass:.5f}" rgba="{rgba}"/>\n'
            f"    </default>"
        )
        for i in range(int(n_slots)):
            name = piece_body_name(type_id, i)
            bodies.append(
                f"""    <body name="{name}" pos="0 0 0">
      <freejoint name="{name}_free"/>
      <geom name="{name}_geom" class="{cls}" type="mesh" mesh="{mesh_name}"/>
    </body>"""
            )
    return mesh_assets, defaults, bodies


def _legacy_sphere_bodies(n_pieces: int) -> list[str]:
    bodies: list[str] = []
    for i in range(n_pieces):
        name = f"piece_{i:02d}"
        bodies.append(
            f"""    <body name="{name}" pos="0 0 0">
      <joint name="{name}_sx" type="slide" axis="1 0 0" damping="0.05"/>
      <joint name="{name}_sy" type="slide" axis="0 1 0" damping="0.02"/>
      <joint name="{name}_sz" type="slide" axis="0 0 1" damping="0.05"/>
      <geom name="{name}_geom" class="piece" type="sphere" size="1.4" mass="0.05" rgba="0.9 0.8 0.2 1"/>
    </body>"""
        )
    return bodies


def _wrap_mjcf(
    *,
    geoms: list[str],
    bodies: list[str],
    assets: list[str],
    extra_defaults: list[str],
    marker: str | None,
    meshdir: str | None = None,
) -> str:
    meshdir_attr = f' meshdir="{escape(meshdir)}"' if meshdir else ""
    compiler = f'<compiler angle="radian" inertiafromgeom="true" autolimits="true"{meshdir_attr}/>'
    asset_block = ""
    if assets:
        asset_block = "\n  <asset>\n" + "\n".join(assets) + "\n  </asset>"
    extra = ("\n" + "\n".join(extra_defaults)) if extra_defaults else ""
    head = f"  <!-- {marker} -->\n" if marker else ""
    return f"""<mujoco model="talongym_field">
{head}  {compiler}
  <option gravity="0 {-IN_G:.4f} 0" timestep="0.002" integrator="implicitfast" cone="pyramidal"/>
  <default>
    <geom condim="3" solref="0.02 1" solimp="0.9 0.95 0.001"/>
    <default class="field">
      <geom group="{GEOM_GROUP_FIELD}" contype="1" conaffinity="10" condim="3"/>
    </default>
    <default class="robot">
      <geom group="{GEOM_GROUP_ROBOT}" contype="2" conaffinity="15" condim="3"/>
    </default>
    <default class="piece">
      <geom group="{GEOM_GROUP_PIECE}" contype="8" conaffinity="13" condim="3" friction="0.6 0.05 0.01"/>
    </default>
    <default class="trigger">
      <geom group="{GEOM_GROUP_TRIGGER}" contype="0" conaffinity="0" density="0"/>
    </default>{extra}
  </default>{asset_block}
  <worldbody>
{chr(10).join(geoms)}
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""


def _build_cad_mjcf(
    field: dict[str, Any],
    manifest: dict[str, Any],
    *,
    robot: dict[str, Any] | None,
    n_robots: int,
    n_pieces: int | None,
    robot_hx: float,
    robot_hy: float,
    robot_hz: float,
    robot_mesh: Path | None,
    mesh_root: Path | None = None,
) -> FieldMjcf:
    from talongym.assets.cad_common import CadImportError

    field_block = manifest.get("field") or {}
    parts = list(field_block.get("collisionParts") or [])
    if len(parts) < 2:
        raise CadImportError("cad manifest field.collisionParts is empty; refusing AABB/GLB field collision")
    kept, filter_stats = select_field_collision_parts(parts)
    if len(kept) < 2:
        raise CadImportError("CAD filter removed all playable field collision parts")
    floor_y = float(filter_stats.floor_y)
    fw = float(field["fieldSizeIn"]["width"])
    fd = float(field["fieldSizeIn"]["depth"])
    geoms = [
        f'    <geom name="floor" class="field" type="plane" size="{fw / 2:.3f} {fd / 2:.3f} 1" '
        f'pos="0 {floor_y:.3f} 0" zaxis="0 1 0" rgba="0.1 0.3 0.2 1"/>',
    ]
    assets: list[str] = []
    glass_parts = [part for part in kept if "field_side_glass" in str(part.get("id") or "").lower()]
    for i, part in enumerate(kept):
        if part in glass_parts:
            continue
        rel = part.get("asset")
        if not rel:
            raise CadImportError(f"collision part {part.get('id')!r} missing asset")
        mesh_name = f"cad_{i:04d}"
        geom_name = f"cad_{i:04d}"
        assets.append(f'    <mesh name="{mesh_name}" file="{_mesh_file_attr(str(rel), mesh_root=mesh_root)}"/>')
        geoms.append(f'    <geom name="{geom_name}" class="field" type="mesh" mesh="{mesh_name}"/>')
    perimeter_geoms = _perimeter_glass_proxy_geoms(glass_parts)
    if glass_parts and len(perimeter_geoms) != 4:
        raise CadImportError(
            f"expected four continuous perimeter glass proxies, got {len(perimeter_geoms)}"
        )
    geoms.extend(perimeter_geoms)
    geoms.extend(_trigger_geoms(field))
    geoms.extend(_flower_cup_proxy_geoms(field))
    mechanism_ids: list[str] = []
    mechanism_bodies: list[str] = []
    for mechanism in field_block.get("mechanisms") or []:
        mechanism_id = _xml_name(str(mechanism.get("id") or "field_mechanism"))
        alliance = str(mechanism.get("alliance") or mechanism_id.split("_", 1)[0])
        pivot = [float(v) for v in (mechanism.get("pivotIn") or [0.0, 0.0, 0.0])]
        axis = [float(v) for v in (mechanism.get("axis") or [1.0, 0.0, 0.0])]
        tip_angle = float(mechanism.get("tipAngleDeg") or 0.0) * 3.141592653589793 / 180.0
        low, high = (min(-0.05, tip_angle), max(0.05, tip_angle))
        body_geoms: list[str] = []
        for part_index, part in enumerate(mechanism.get("collisionParts") or []):
            part_id = str(part.get("id") or "").lower()
            if any(
                token in part_id
                for token in (
                    "goal_rib",
                    "hive_goal_top_skin",
                    "hive_goal_back_skin",
                    "hive_goal_bottom_skin",
                    "goal_april_tag",
                )
            ):
                continue
            rel = part.get("asset")
            if not rel:
                continue
            mesh_name = _xml_name(f"mechanism_{mechanism_id}_{part_index:02d}")
            assets.append(f'    <mesh name="{mesh_name}" file="{_mesh_file_attr(str(rel), mesh_root=mesh_root)}"/>')
            body_geoms.append(
                f'      <geom name="{mesh_name}_geom" class="field" type="mesh" mesh="{mesh_name}" density="0" contype="4" conaffinity="10"/>'
            )
        body_geoms.extend(_mechanism_cell_proxy_geoms(field, alliance))
        if not body_geoms:
            continue
        pivot_text = " ".join(f"{value:.4f}" for value in pivot)
        axis_text = " ".join(f"{value:.4f}" for value in axis)
        mechanism_bodies.append(
            f"""    <body name="field_mech_{mechanism_id}">
      <inertial pos="{pivot_text}" mass="5" diaginertia="120 120 120"/>
      <joint name="field_mech_{mechanism_id}" type="hinge" pos="{pivot_text}" axis="{axis_text}" range="{low:.5f} {high:.5f}" damping="18" armature="2"/>
{chr(10).join(body_geoms)}
    </body>"""
        )
        mechanism_ids.append(mechanism_id)

    mesh_file = None
    if robot_mesh is not None:
        mesh_path = Path(robot_mesh)
        if mesh_path.is_file():
            mesh_file = mesh_path.name
            assets.append(f'    <mesh name="robot_hull" file="{escape(str(mesh_path.resolve()))}"/>')

    slot_plan = piece_slot_plan(field, n_pieces)
    piece_assets, piece_defaults, piece_bodies = _typed_piece_bodies(
        field, manifest, slot_plan, mesh_root=mesh_root
    )
    assets.extend(piece_assets)
    body_y = float(robot_hz) + 0.2 + floor_y
    bodies, robot_assets = _robot_bodies(
        n_robots,
        robot_hx,
        robot_hy,
        robot_hz,
        mesh_file=mesh_file,
        body_y=body_y,
        robot=robot,
    )
    assets.extend(robot_assets)
    bodies.extend(mechanism_bodies)
    bodies.extend(piece_bodies)
    marker = (
        f"{CAD_MJCF_MARKER} version={CAD_MJCF_VERSION} parts={len(kept)}/{filter_stats.manifest_parts} "
        f"floor_y={floor_y:.3f}"
    )
    xml = _wrap_mjcf(
        geoms=geoms,
        bodies=bodies,
        assets=assets,
        extra_defaults=piece_defaults,
        marker=marker,
        meshdir="." if mesh_root is not None else None,
    )
    stats = {
        "cad": True,
        "filter": filter_stats.as_dict(),
        "slotPlan": dict(slot_plan),
        "nFieldGeoms": len(kept) - len(glass_parts) + len(perimeter_geoms) + 1,
        "nPieceBodies": sum(slot_plan.values()),
        "fieldMechanisms": mechanism_ids,
        "fieldMechanismTargets": {
            _xml_name(str(mechanism.get("id") or "field_mechanism")): (
                float(mechanism.get("tipAngleDeg") or 0.0) * 3.141592653589793 / 180.0
            )
            for mechanism in (field_block.get("mechanisms") or [])
        },
    }
    return FieldMjcf(xml=xml, stats=stats, floor_y=floor_y, slot_plan=slot_plan, cad=True)


def _build_aabb_mjcf(
    field: dict[str, Any],
    *,
    robot: dict[str, Any] | None,
    n_robots: int,
    n_pieces: int,
    robot_hx: float,
    robot_hy: float,
    robot_hz: float,
    robot_mesh: Path | None,
) -> FieldMjcf:
    fw = float(field["fieldSizeIn"]["width"])
    fd = float(field["fieldSizeIn"]["depth"])
    wall_h = float(field["fieldSizeIn"].get("wallHeight") or 12)
    geoms: list[str] = [
        f'    <geom name="floor" class="field" type="plane" size="{fw / 2:.3f} {fd / 2:.3f} 1" pos="0 0 0" zaxis="0 1 0" rgba="0.1 0.3 0.2 1"/>',
        f'    <geom name="wall_n" class="field" type="box" size="{_size(fw / 2 + 1, wall_h / 2, 1)}" pos="{" ".join(f"{v:.3f}" for v in _yup(0, fd / 2, wall_h / 2))}" rgba="0.5 0.6 0.65 1"/>',
        f'    <geom name="wall_s" class="field" type="box" size="{_size(fw / 2 + 1, wall_h / 2, 1)}" pos="{" ".join(f"{v:.3f}" for v in _yup(0, -fd / 2, wall_h / 2))}" rgba="0.5 0.6 0.65 1"/>',
        f'    <geom name="wall_w" class="field" type="box" size="{_size(1, wall_h / 2, fd / 2)}" pos="{" ".join(f"{v:.3f}" for v in _yup(-fw / 2, 0, wall_h / 2))}" rgba="0.5 0.6 0.65 1"/>',
        f'    <geom name="wall_e" class="field" type="box" size="{_size(1, wall_h / 2, fd / 2)}" pos="{" ".join(f"{v:.3f}" for v in _yup(fw / 2, 0, wall_h / 2))}" rgba="0.5 0.6 0.65 1"/>',
    ]
    geoms.extend(_aabb_element_geoms(field))
    assets: list[str] = []
    mesh_file = None
    if robot_mesh is not None:
        mesh_path = Path(robot_mesh)
        if mesh_path.is_file():
            mesh_file = mesh_path.name
            assets.append(f'    <mesh name="robot_hull" file="{escape(str(mesh_path.resolve()))}"/>')
    bodies, robot_assets = _robot_bodies(
        n_robots,
        robot_hx,
        robot_hy,
        robot_hz,
        mesh_file=mesh_file,
        body_y=robot_hz + 0.2,
        robot=robot,
    )
    assets.extend(robot_assets)
    bodies.extend(_legacy_sphere_bodies(n_pieces))
    xml = _wrap_mjcf(geoms=geoms, bodies=bodies, assets=assets, extra_defaults=[], marker=None)
    plan = {"": int(n_pieces)}
    return FieldMjcf(xml=xml, stats={"cad": False, "slotPlan": plan}, floor_y=0.0, slot_plan=plan, cad=False)


def build_field_mjcf(
    field: dict[str, Any],
    *,
    robot: dict[str, Any] | None = None,
    n_robots: int = 4,
    n_pieces: int = 80,
    robot_hx: float = 9.0,
    robot_hy: float = 9.0,
    robot_hz: float = 5.0,
    robot_mesh: Path | None = None,
    manifest: dict[str, Any] | None = None,
    mesh_root: Path | None = None,
) -> FieldMjcf:
    """Build MJCF. CAD seasons use convex parts; others use schematic AABBs."""
    from talongym.assets.cad_common import mesh_required

    doc = manifest
    if doc is None and field.get("cadManifest"):
        doc = load_field_manifest(field, require=mesh_required(field))
    if doc and (doc.get("field") or {}).get("collisionParts"):
        verify_collision_asset(field)
        return _build_cad_mjcf(
            field,
            doc,
            robot=robot,
            n_robots=n_robots,
            n_pieces=n_pieces,
            robot_hx=robot_hx,
            robot_hy=robot_hy,
            robot_hz=robot_hz,
            robot_mesh=robot_mesh,
            mesh_root=mesh_root,
        )
    if mesh_required(field):
        from talongym.assets.cad_common import CadImportError

        raise CadImportError("mesh_field_collision season refused AABB field collision; cadManifest required")
    return _build_aabb_mjcf(
        field,
        robot=robot,
        n_robots=n_robots,
        n_pieces=n_pieces,
        robot_hx=robot_hx,
        robot_hy=robot_hy,
        robot_hz=robot_hz,
        robot_mesh=robot_mesh,
    )


def build_mjcf(
    field: dict[str, Any],
    *,
    robot: dict[str, Any] | None = None,
    n_robots: int = 4,
    n_pieces: int = 80,
    robot_hx: float = 9.0,
    robot_hy: float = 9.0,
    robot_hz: float = 5.0,
    robot_mesh: Path | None = None,
) -> str:
    return build_field_mjcf(
        field,
        robot=robot,
        n_robots=n_robots,
        n_pieces=n_pieces,
        robot_hx=robot_hx,
        robot_hy=robot_hy,
        robot_hz=robot_hz,
        robot_mesh=robot_mesh,
    ).xml


def write_collision_mjcf(field: dict[str, Any], dest: Path | None = None) -> Path:
    """Write the CAD (or AABB) MJCF to the committed collisionAsset path."""
    from talongym.assets.cad_manifest import resolve_asset

    rel = field.get("collisionAsset")
    path = Path(dest) if dest is not None else (resolve_asset(str(rel)) if rel else ASSETS_DIR / "field_mjcf.xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    built = build_field_mjcf(field, mesh_root=path.parent)
    path.write_text(built.xml, encoding="utf-8")
    return path
