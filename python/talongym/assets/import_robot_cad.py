"""Convert team robot CAD into a light GLB plus a convex-hull collider.

Accepted inputs: GLB/glTF (Y-up), STL/OBJ/STEP (Z-up), or a ZIP of those.
Stored assets live under var/assets/robots/<id>/ (uploads) or
var/assets/robot_parts/ (catalog cache) and are not committed.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from talongym.assets.cad_common import YUP_FROM_ZUP, CadImportError, as_trimesh

ALLOWED_SUFFIXES = {".glb", ".gltf", ".stl", ".obj", ".step", ".stp", ".zip"}
MESH_SUFFIXES = {".glb", ".gltf", ".stl", ".obj", ".step", ".stp"}
Z_UP_SUFFIXES = {".stl", ".obj", ".step", ".stp"}
ROBOT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-.]{1,63}$")
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ZIP_MEMBERS = 64
IN_PER_M = 39.37007874015748
MM_PER_IN = 25.4
HULL_FACE_TARGET = 512
CAD_LONG_AXIS_RATIO = 2.5
CAD_EXTRA_HINT = "pip install -e '.[cad]'"


class RobotCadError(RuntimeError):
    pass


def robot_assets_root() -> Path:
    from talongym import paths

    return Path(paths.VAR_DIR) / "assets"


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


def is_cad_extra_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "trimesh" in msg or "cascadio" in msg or ".[cad]" in msg or "cad extra" in msg


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_sha256(path: Path, expected: str | None) -> str:
    digest = sha256_file(path)
    if expected and digest != expected.lower():
        raise RobotCadError(f"stale CAD {path.name}: sha256 {digest} != pinned {expected.lower()}")
    return digest


def _require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise RobotCadError(f"trimesh missing; {CAD_EXTRA_HINT}") from exc
    return trimesh


def extract_cad_archive(source: Path, dest_dir: Path, *, preferred_name: str | None = None) -> Path:
    """Unpack a ZIP of manufacturer CAD and return the chosen mesh/STEP file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source) as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if len(infos) > MAX_ZIP_MEMBERS:
                raise RobotCadError("ZIP has too many members")
            total = 0
            chosen: zipfile.ZipInfo | None = None
            preferred = (preferred_name or "").lower()
            meshes: list[zipfile.ZipInfo] = []
            for info in infos:
                name = Path(info.filename).name
                if not name or name.startswith(".") or ".." in Path(info.filename).parts:
                    continue
                total += int(info.file_size)
                if total > MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise RobotCadError("ZIP uncompressed size exceeds limit")
                suffix = Path(name).suffix.lower()
                if suffix in MESH_SUFFIXES:
                    meshes.append(info)
                    if preferred and name.lower() == preferred:
                        chosen = info
            if chosen is None and meshes:
                steps = [info for info in meshes if Path(info.filename).suffix.lower() in {".step", ".stp"}]
                pool = steps or meshes
                chosen = max(pool, key=lambda info: info.file_size)
            if chosen is None:
                raise RobotCadError("ZIP contains no GLB, glTF, STL, OBJ, or STEP file")
            target = dest_dir / Path(chosen.filename).name
            with zf.open(chosen, "r") as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            return target
    except zipfile.BadZipFile as exc:
        raise RobotCadError(f"invalid ZIP: {source.name}") from exc


def _to_trimesh(loaded: Any, trimesh: Any) -> Any:
    if isinstance(loaded, (list, tuple)):
        geoms = [g for g in loaded if isinstance(g, trimesh.Trimesh) and len(getattr(g, "faces", []))]
        if not geoms:
            raise RobotCadError("CAD scene has no triangle mesh")
        loaded = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    try:
        return as_trimesh(loaded, trimesh)
    except CadImportError as exc:
        raise RobotCadError(str(exc)) from exc


def _load_mesh(path: Path) -> Any:
    trimesh = _require_trimesh()
    suffix = path.suffix.lower()
    if suffix in {".step", ".stp"}:
        try:
            import cascadio  # noqa: F401
        except ImportError as exc:
            raise RobotCadError(f"STEP needs cascadio; {CAD_EXTRA_HINT}") from exc
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


def collision_span_in(collision: list[Any] | None) -> float | None:
    """Largest authored catalog proxy extent in inches."""
    best = 0.0
    for item in collision or []:
        if not isinstance(item, dict):
            continue
        for value in item.get("sizeIn") or []:
            best = max(best, abs(float(value)))
        radius = abs(float(item.get("radiusIn") or 0.0))
        length = abs(float(item.get("lengthIn") or 0.0))
        best = max(best, 2.0 * radius, length)
    return best if best > 1e-9 else None


def collision_size_yup(collision: list[Any] | None) -> list[float] | None:
    """Collision AABB size in Three/Y-up inches: FTC (x, y, z) -> (x, z, y)."""
    for item in collision or []:
        if not isinstance(item, dict):
            continue
        size = item.get("sizeIn") or []
        if len(size) >= 3:
            return [abs(float(size[0])), abs(float(size[2])), abs(float(size[1]))]
        radius = abs(float(item.get("radiusIn") or 0.0))
        length = abs(float(item.get("lengthIn") or 0.0))
        if radius > 1e-12 or length > 1e-12:
            span = 2.0 * radius if radius > 1e-12 else length
            along = length if length > 1e-12 else span
            return [span, along, span]
    return None


def _unique_long_axis(size: list[float]) -> int | None:
    abs_size = [abs(float(v)) for v in size]
    long = max(range(3), key=lambda i: abs_size[i])
    rest = max(abs_size[j] for j in range(3) if j != long)
    if abs_size[long] > CAD_LONG_AXIS_RATIO * max(rest, 1e-12):
        return long
    return None


def _rotation_aligning(src: list[float], dst: list[float]) -> list[list[float]]:
    def _norm(vec: list[float]) -> list[float]:
        length = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5
        if length < 1e-12:
            return [0.0, 0.0, 0.0]
        return [vec[0] / length, vec[1] / length, vec[2] / length]

    a = _norm(src)
    b = _norm(dst)
    if max(abs(v) for v in a) < 1e-12 or max(abs(v) for v in b) < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cosine = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if cosine > 0.999999:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if cosine < -0.999999:
        helper = [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]
        axis = _norm(
            [
                a[1] * helper[2] - a[2] * helper[1],
                a[2] * helper[0] - a[0] * helper[2],
                a[0] * helper[1] - a[1] * helper[0],
            ]
        )
        x, y, z = axis
        return [
            [2 * x * x - 1, 2 * x * y, 2 * x * z],
            [2 * y * x, 2 * y * y - 1, 2 * y * z],
            [2 * z * x, 2 * z * y, 2 * z * z - 1],
        ]
    vx = a[1] * b[2] - a[2] * b[1]
    vy = a[2] * b[0] - a[0] * b[2]
    vz = a[0] * b[1] - a[1] * b[0]
    k = [[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]]
    kk = [
        [
            k[i][0] * k[0][j] + k[i][1] * k[1][j] + k[i][2] * k[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]
    s2 = vx * vx + vy * vy + vz * vz
    factor = (1.0 - cosine) / s2
    out = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(3):
        for j in range(3):
            out[i][j] = (1.0 if i == j else 0.0) + k[i][j] + kk[i][j] * factor
    return out


def cad_unit_fit_scale(mesh_span: float, target_span: float) -> float:
    mesh = abs(float(mesh_span) or 0.0)
    target = abs(float(target_span) or 0.0)
    if mesh <= 1e-8 or target <= 1e-8:
        return 1.0
    best_scale = 1.0
    best_rel = abs(mesh - target) / target
    for factor in (1.0, MM_PER_IN, IN_PER_M, 1000.0):
        for scale in (factor, 1.0 / factor):
            if abs(scale - 1.0) < 1e-12:
                continue
            rel = abs(mesh * scale - target) / target
            if rel < best_rel:
                best_rel = rel
                best_scale = scale
    if abs(best_scale - 1.0) > 1e-12 and best_rel <= 0.2:
        return best_scale
    return 1.0


def cad_collision_fit(
    mesh_size: list[float],
    mesh_center: list[float],
    target_size: list[float] | None,
) -> dict[str, Any]:
    """Scale, swing the unique long axis onto the proxy, and center end-origin bars."""
    size = [abs(float(v)) for v in mesh_size[:3]]
    while len(size) < 3:
        size.append(0.0)
    target = [abs(float(v)) for v in (target_size or [])[:3]]
    while len(target) < 3:
        target.append(0.0)
    mesh_span = max(size)
    target_span = max(target)
    scale = cad_unit_fit_scale(mesh_span, target_span) if target_span > 1e-8 else 1.0
    fitted = [v * scale for v in size]
    center = [float(mesh_center[i]) * scale if i < len(mesh_center) else 0.0 for i in range(3)]
    mesh_long = _unique_long_axis(fitted)
    target_long = _unique_long_axis(target)
    rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rotated = list(center)
    if mesh_long is not None and target_long is not None and mesh_long != target_long:
        src = [0.0, 0.0, 0.0]
        dst = [0.0, 0.0, 0.0]
        src[mesh_long] = 1.0
        dst[target_long] = 1.0
        rotation = _rotation_aligning(src, dst)
        rotated = [
            rotation[0][0] * center[0] + rotation[0][1] * center[1] + rotation[0][2] * center[2],
            rotation[1][0] * center[0] + rotation[1][1] * center[1] + rotation[1][2] * center[2],
            rotation[2][0] * center[0] + rotation[2][1] * center[1] + rotation[2][2] * center[2],
        ]
    translation = [0.0, 0.0, 0.0]
    long = target_long if target_long is not None else mesh_long
    if long is not None and abs(rotated[long]) > 0.2 * max(fitted[mesh_long if mesh_long is not None else long], 1e-12):
        translation = [-rotated[0], -rotated[1], -rotated[2]]
    return {"scale": scale, "rotation": rotation, "translation": translation}


def _align_mesh_to_collision(mesh: Any, collision: list[Any] | None) -> None:
    target = collision_size_yup(collision)
    if target is None:
        return
    extents = [float(v) for v in mesh.extents]
    bounds = mesh.bounds
    center = [0.5 * (float(bounds[0][i]) + float(bounds[1][i])) for i in range(3)]
    fit = cad_collision_fit(extents, center, target)
    if abs(float(fit["scale"]) - 1.0) > 1e-9:
        mesh.apply_scale(float(fit["scale"]))
    rotation = fit["rotation"]
    if any(abs(rotation[i][j] - (1.0 if i == j else 0.0)) > 1e-9 for i in range(3) for j in range(3)):
        matrix = [
            [rotation[0][0], rotation[0][1], rotation[0][2], 0.0],
            [rotation[1][0], rotation[1][1], rotation[1][2], 0.0],
            [rotation[2][0], rotation[2][1], rotation[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        mesh.apply_transform(matrix)
    translation = fit["translation"]
    if any(abs(float(v)) > 1e-9 for v in translation):
        mesh.apply_translation([float(v) for v in translation])


def resolve_catalog_inch_scale(
    span: float,
    declared: float = 1.0,
    target_in: float | None = None,
) -> tuple[float, str]:
    """Pick m/mm/in (or the preset scale) so the mesh lands on the collision size.

    Cascadio tessellation is often metres even when the STEP header and catalog
    preset say millimetres. Blindly applying scaleToInches=1/25.4 then yields a
    ~0.002 in mesh that the viewport treats as loaded and hides the proxy.
    """
    declared_scale = float(declared) if declared and abs(float(declared)) > 1e-12 else 1.0
    candidates: list[tuple[float, str]] = [
        (1.0, "in"),
        (IN_PER_M, "m"),
        (1.0 / MM_PER_IN, "mm"),
        (declared_scale, "declared"),
    ]
    if target_in is not None and target_in > 1e-9 and span > 1e-12:
        scale, units = min(candidates, key=lambda row: abs(span * row[0] - float(target_in)))
        return scale, units
    if abs(declared_scale - 1.0) > 1e-12:
        if abs(declared_scale - IN_PER_M) < 1e-6:
            return declared_scale, "m"
        if abs(declared_scale - (1.0 / MM_PER_IN)) < 1e-6:
            return declared_scale, "mm"
        return declared_scale, "declared"
    return guess_inch_scale(span)


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


def _apply_catalog_transform(
    mesh: Any,
    transform: dict[str, Any],
    *,
    target_span_in: float | None = None,
) -> tuple[float, str]:
    declared = float(transform.get("scaleToInches") or 1.0)
    extents = mesh.extents
    span = float(max(extents)) if len(extents) else 0.0
    scale, units = resolve_catalog_inch_scale(span, declared, target_span_in)
    if abs(scale - 1.0) > 1e-9:
        mesh.apply_scale(scale)
    if transform.get("rotatedZupToYup"):
        mesh.apply_transform(YUP_FROM_ZUP)
    translation = transform.get("translationIn") or [0.0, 0.0, 0.0]
    if any(abs(float(v)) > 1e-12 for v in translation):
        mesh.apply_translation([float(v) for v in translation])
    return scale, units


def _prepare_visual_mesh(mesh: Any, trimesh: Any) -> Any:
    try:
        mesh.fix_normals()
    except Exception:
        pass
    try:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=[192, 177, 151, 255])
    except Exception:
        pass
    return mesh


def import_robot_cad(
    source: Path,
    robot_id: str,
    *,
    part_id: str | None = None,
    preserve_origin: bool = False,
    parent_id: str | None = None,
    joint_transform: dict[str, float] | None = None,
    dest_root: Path | None = None,
    visual_transform: dict[str, Any] | None = None,
    collision: list[Any] | None = None,
    target_span_in: float | None = None,
    expected_sha256: str | None = None,
    preferred_name: str | None = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> dict[str, Any]:
    source = Path(source)
    if not source.is_file():
        raise RobotCadError(f"file not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise RobotCadError(f"unsupported format {suffix}; use GLB, glTF, STL, OBJ, STEP, or ZIP")
    if source.stat().st_size > max_bytes:
        raise RobotCadError(f"file larger than {max_bytes // (1024 * 1024)} MB")
    digest = verify_source_sha256(source, expected_sha256)

    scratch: Path | None = None
    mesh_source = source
    if suffix == ".zip":
        scratch = source.parent / f"{source.stem}.unzipped"
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        mesh_source = extract_cad_archive(source, scratch, preferred_name=preferred_name)
        suffix = mesh_source.suffix.lower()

    try:
        mesh = _load_mesh(mesh_source)
        target = target_span_in if target_span_in is not None else collision_span_in(collision)
        if visual_transform:
            _scale, units = _apply_catalog_transform(mesh, visual_transform, target_span_in=target)
        else:
            if suffix in Z_UP_SUFFIXES:
                # FTC Z-up (x forward, y left, z height) -> Three/MuJoCo Y-up (x, height, -y).
                mesh.apply_transform(YUP_FROM_ZUP)
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

        if collision:
            _align_mesh_to_collision(mesh, collision)

        extents = mesh.extents
        span = float(max(extents)) if len(extents) else 0.0
        if span < 1e-6:
            raise RobotCadError("mesh has zero size")

        length_in = float(mesh.extents[0])
        width_in = float(mesh.extents[2])
        height_in = float(mesh.extents[1])

        mesh = _prepare_visual_mesh(mesh, _require_trimesh())
        hull = _decimate_hull(_convex_hull(mesh))
        xy = [(float(v[0]), -float(v[2])) for v in hull.vertices]
        footprint_pts = convex_hull_2d(xy)
        if len(footprint_pts) < 3:
            hx, hy = length_in / 2.0, width_in / 2.0
            footprint_pts = [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)]

        if dest_root is not None:
            dest = Path(dest_root).resolve()
            try:
                dest.relative_to(robot_assets_root().resolve())
            except ValueError as exc:
                raise RobotCadError(f"bad catalog dest {dest_root}") from exc
        else:
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
            "sha256": digest,
            "originPreserved": bool(preserve_origin),
        }
        if part_id is not None:
            result.update(
                {
                    "partId": part_id,
                    "parentId": parent_id,
                    "jointTransform": dict(joint_transform or {}),
                }
            )
        return result
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


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
