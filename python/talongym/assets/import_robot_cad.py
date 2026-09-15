"""Convert team robot CAD into a light GLB plus a convex-hull collider.

Accepted inputs: GLB/glTF (Y-up), STL/OBJ/STEP (Z-up). Stored assets live under
var/assets/robots/<id>/ and are not committed.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from talongym.paths import VAR_DIR

ALLOWED_SUFFIXES = {".glb", ".gltf", ".stl", ".obj", ".step", ".stp"}
Z_UP_SUFFIXES = {".stl", ".obj", ".step", ".stp"}
ROBOT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-.]{1,63}$")
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
IN_PER_M = 39.37007874015748
MM_PER_IN = 25.4
HULL_FACE_TARGET = 512


class RobotCadError(RuntimeError):
    pass


def robot_assets_root() -> Path:
    return VAR_DIR / "assets"


def robot_asset_dir(robot_id: str) -> Path:
    if not ROBOT_ID_RE.match(robot_id):
        raise RobotCadError(f"invalid robot id {robot_id}")
    return robot_assets_root() / "robots" / robot_id


def resolve_robot_asset(rel: str) -> Path:
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise RobotCadError(f"bad asset path {rel}")
    root = robot_assets_root().resolve()
    dest = (root / rel_path).resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise RobotCadError(f"bad asset path {rel}") from exc
    return dest


def delete_robot_assets(robot_id: str) -> None:
    dest = robot_asset_dir(robot_id)
    if dest.is_dir():
        shutil.rmtree(dest)


def _require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise RobotCadError("trimesh missing; pip install -e '.[cad]'") from exc
    return trimesh


def _to_trimesh(loaded: Any, trimesh: Any) -> Any:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RobotCadError("CAD scene has no triangle mesh")
        return trimesh.util.concatenate(geoms)
    if isinstance(loaded, (list, tuple)):
        geoms = [g for g in loaded if isinstance(g, trimesh.Trimesh)]
        if geoms:
            return trimesh.util.concatenate(geoms)
    raise RobotCadError(f"unsupported mesh type {type(loaded).__name__}")


def _load_mesh(path: Path) -> Any:
    trimesh = _require_trimesh()
    suffix = path.suffix.lower()
    if suffix in {".step", ".stp"}:
        try:
            import cascadio  # noqa: F401
        except ImportError as exc:
            raise RobotCadError("STEP needs cascadio; pip install -e '.[cad]'") from exc
    try:
        loaded = trimesh.load(str(path), force=None)
    except Exception as exc:
        raise RobotCadError(f"could not load {path.name}: {exc}") from exc
    if loaded is None:
        raise RobotCadError(f"trimesh could not load {path.name}")
    return _to_trimesh(loaded, trimesh)


def guess_inch_scale(span: float) -> tuple[float, str]:
    if span < 3.0:
        return IN_PER_M, "m"
    if span > 200.0:
        return 1.0 / MM_PER_IN, "mm"
    return 1.0, "in"


def convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set((round(float(x), 6), round(float(y), 6)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _convex_hull(mesh: Any) -> Any:
    try:
        hull = mesh.convex_hull
        _ = len(hull.vertices)
        return hull
    except Exception:
        return mesh


def _decimate_hull(hull: Any) -> Any:
    if len(getattr(hull, "faces", [])) <= HULL_FACE_TARGET:
        return hull
    try:
        return hull.simplify_quadric_decimation(HULL_FACE_TARGET)
    except Exception:
        return hull


def import_robot_cad(
    source: Path,
    robot_id: str,
    *,
    part_id: str | None = None,
    preserve_origin: bool = False,
    parent_id: str | None = None,
    joint_transform: dict[str, float] | None = None,
) -> dict[str, Any]:
    source = Path(source)
    if not source.is_file():
        raise RobotCadError(f"file not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise RobotCadError(f"unsupported format {suffix}; use GLB, glTF, STL, OBJ, or STEP")
    if source.stat().st_size > MAX_UPLOAD_BYTES:
        raise RobotCadError(f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    mesh = _load_mesh(source)
    if suffix in Z_UP_SUFFIXES:
        # FTC Z-up (x forward, y left, z height) -> Three/MuJoCo Y-up (x, height, -y).
        mesh.apply_transform(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    extents = mesh.extents
    span = float(max(extents)) if len(extents) else 0.0
    if span < 1e-6:
        raise RobotCadError("mesh has zero size")
    scale, units = guess_inch_scale(span)
    if abs(scale - 1.0) > 1e-9:
        mesh.apply_scale(scale)

    if not preserve_origin:
        min_b, max_b = mesh.bounds
        cx = 0.5 * (float(min_b[0]) + float(max_b[0]))
        cz = 0.5 * (float(min_b[2]) + float(max_b[2]))
        mesh.apply_translation([-cx, -float(min_b[1]), -cz])
        height = float(mesh.extents[1])
        mesh.apply_translation([0.0, -height / 2.0, 0.0])

    length_in = float(mesh.extents[0])
    width_in = float(mesh.extents[2])
    height_in = float(mesh.extents[1])

    hull = _decimate_hull(_convex_hull(mesh))
    xy = [(float(v[0]), -float(v[2])) for v in hull.vertices]
    footprint_pts = convex_hull_2d(xy)
    if len(footprint_pts) < 3:
        hx, hy = length_in / 2.0, width_in / 2.0
        footprint_pts = [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)]

    if part_id is not None and not ROBOT_ID_RE.match(part_id):
        raise RobotCadError(f"invalid robot part id {part_id}")
    dest = robot_asset_dir(robot_id)
    if part_id is not None:
        dest = dest / "parts" / part_id
    dest.mkdir(parents=True, exist_ok=True)
    glb_path = dest / "visual.glb"
    stl_path = dest / "collision.stl"
    mesh.export(str(glb_path), file_type="glb")
    hull.export(str(stl_path), file_type="stl")

    rel_glb = str(glb_path.relative_to(robot_assets_root())).replace("\\", "/")
    rel_stl = str(stl_path.relative_to(robot_assets_root())).replace("\\", "/")
    result = {
        "visualAsset": rel_glb,
        "collisionAsset": rel_stl,
        "bbox": {
            "lengthIn": round(length_in, 4),
            "widthIn": round(width_in, 4),
            "heightIn": round(height_in, 4),
        },
        "footprint": [{"x": round(x, 4), "y": round(y, 4)} for x, y in footprint_pts],
        "unitsGuess": units,
        "faceCount": int(len(getattr(hull, "faces", []))),
    }
    if part_id is not None:
        result.update(
            {
                "partId": part_id,
                "parentId": parent_id,
                "jointTransform": dict(joint_transform or {}),
                "originPreserved": bool(preserve_origin),
            }
        )
    return result


def import_robot_part_cad(
    source: Path,
    robot_id: str,
    part_id: str,
    *,
    parent_id: str | None,
    joint_transform: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Import one articulated part without discarding assembly/joint coordinates."""
    return import_robot_cad(
        source,
        robot_id,
        part_id=part_id,
        preserve_origin=True,
        parent_id=parent_id,
        joint_transform=joint_transform,
    )


def write_ascii_box_stl(path: Path, length: float, width: float, height: float) -> Path:
    """Z-up box from the origin, +x length, +y width, +z height. For tests."""
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = float(length), float(width), float(height)
    faces = [
        ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)),
        ((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)),
        ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)),
        ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)),
    ]
    lines = ["solid box"]
    for a, b, c, d in faces:
        for tri in ((a, b, c), (a, c, d)):
            lines.append("  facet normal 0 0 0")
            lines.append("    outer loop")
            for v in tri:
                lines.append(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            lines.append("    endloop")
            lines.append("  endfacet")
    lines.append("endsolid box")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path
