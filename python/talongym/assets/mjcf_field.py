"""Emit MuJoCo MJCF for a field preset (Y-up inches, gravity on).

CAD seasons assemble the static field from convex collision parts in the
committed cad manifest — never field.glb and never a single concave hull.
Non-CAD seasons keep the schematic AABB generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from talongym.paths import ASSETS_DIR


IN_G = 386.0886  # 9.80665 m/s^2 in inches/s^2
CAD_MJCF_MARKER = "talongym_cad_field"
CAD_MJCF_VERSION = "1.1.0"

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


def _robot_bodies(
    n_robots: int,
    robot_hx: float,
    robot_hy: float,
    robot_hz: float,
    *,
    mesh_file: str | None,
    body_y: float,
) -> list[str]:
    robot_ids = ["red_0", "red_1", "blue_0", "blue_1"][: max(1, n_robots)]
    bodies: list[str] = []
    for rid in robot_ids:
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
    return bodies


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
  <option gravity="0 {-IN_G:.4f} 0" timestep="0.01" integrator="Euler" cone="pyramidal"/>
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
    for i, part in enumerate(kept):
        rel = part.get("asset")
        if not rel:
            raise CadImportError(f"collision part {part.get('id')!r} missing asset")
        mesh_name = f"cad_{i:04d}"
        geom_name = f"cad_{i:04d}"
        assets.append(f'    <mesh name="{mesh_name}" file="{_mesh_file_attr(str(rel), mesh_root=mesh_root)}"/>')
        geoms.append(f'    <geom name="{geom_name}" class="field" type="mesh" mesh="{mesh_name}"/>')
    geoms.extend(_trigger_geoms(field))
    mechanism_ids: list[str] = []
    mechanism_bodies: list[str] = []
    for mechanism in field_block.get("mechanisms") or []:
        mechanism_id = _xml_name(str(mechanism.get("id") or "field_mechanism"))
        pivot = [float(v) for v in (mechanism.get("pivotIn") or [0.0, 0.0, 0.0])]
        axis = [float(v) for v in (mechanism.get("axis") or [1.0, 0.0, 0.0])]
        tip_angle = float(mechanism.get("tipAngleDeg") or 0.0) * 3.141592653589793 / 180.0
        low, high = (min(-0.05, tip_angle), max(0.05, tip_angle))
        body_geoms: list[str] = []
        for part_index, part in enumerate(mechanism.get("collisionParts") or []):
            rel = part.get("asset")
            if not rel:
                continue
            mesh_name = _xml_name(f"mechanism_{mechanism_id}_{part_index:02d}")
            assets.append(f'    <mesh name="{mesh_name}" file="{_mesh_file_attr(str(rel), mesh_root=mesh_root)}"/>')
            body_geoms.append(
                f'      <geom name="{mesh_name}_geom" class="field" type="mesh" mesh="{mesh_name}" density="0" contype="4" conaffinity="10"/>'
            )
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
    bodies = _robot_bodies(n_robots, robot_hx, robot_hy, robot_hz, mesh_file=mesh_file, body_y=body_y)
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
        "nFieldGeoms": len(kept) + 1,
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
    bodies = _robot_bodies(n_robots, robot_hx, robot_hy, robot_hz, mesh_file=mesh_file, body_y=robot_hz + 0.2)
    bodies.extend(_legacy_sphere_bodies(n_pieces))
    xml = _wrap_mjcf(geoms=geoms, bodies=bodies, assets=assets, extra_defaults=[], marker=None)
    plan = {"": int(n_pieces)}
    return FieldMjcf(xml=xml, stats={"cad": False, "slotPlan": plan}, floor_y=0.0, slot_plan=plan, cad=False)


def build_field_mjcf(
    field: dict[str, Any],
    *,
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
    n_robots: int = 4,
    n_pieces: int = 80,
    robot_hx: float = 9.0,
    robot_hy: float = 9.0,
    robot_hz: float = 5.0,
    robot_mesh: Path | None = None,
) -> str:
    return build_field_mjcf(
        field,
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
