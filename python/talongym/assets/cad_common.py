"""Shared STEP download, unit/frame alignment, tessellation, and convex parts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from talongym.paths import VAR_DIR

CAD_GENERATOR_VERSION = "1.2.0"
# Catalog cache only. Bumped independently so field cad_manifest.json stays valid.
CATALOG_CAD_GENERATOR_VERSION = "1.4.0"
IN_PER_M = 39.37007874015748
MM_PER_IN = 25.4
FIELD_SPAN_IN = 144.0
# OpenCASCADE linear deflection in source units (metres for Onshape AP242).
DEFAULT_TOL_BY_UNIT = {"m": 0.005, "mm": 5.0, "in": 0.2, "unknown": 0.005}
PIECE_TOL_BY_UNIT = {"m": 0.0004, "mm": 0.4, "in": 0.015, "unknown": 0.0004}
VISUAL_FACE_TARGET = 800_000
COLLISION_HULL_FACE_TARGET = 96
MIN_PART_EXTENT_IN = 0.5
MIN_PART_FACES = 8
USER_AGENT = "TalonGym/0.1"
STEP_MAGIC = b"ISO-10303-21"
# FTC (x, y floor, z height) -> Three/MuJoCo Y-up (x, height, -y).
YUP_FROM_ZUP = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class CadImportError(RuntimeError):
    pass


def cad_cache_dir() -> Path:
    dest = VAR_DIR / "cad"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_xyz(values: Any, digits: int = 3) -> list[float]:
    return [round(float(v), digits) for v in values]


def default_linear_deflection(units: str, *, piece: bool = False) -> float:
    table = PIECE_TOL_BY_UNIT if piece else DEFAULT_TOL_BY_UNIT
    return float(table.get(units, table["unknown"]))


def sniff_step_units(step_path: Path | None = None, text: str | None = None) -> str:
    """Best-effort length unit from an AP242/Onshape STEP header or unit block."""
    if text is None:
        if step_path is None:
            raise CadImportError("sniff_step_units needs a path or text")
        with Path(step_path).open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(0)
            head = fh.read(min(size, 750_000))
            fh.seek(max(0, size - 2_000_000))
            tail = fh.read()
        text = (head + tail).decode("latin-1", errors="ignore")
    blob = text.upper()
    if "SI_UNIT(.MILLI.,.METRE.)" in blob:
        return "mm"
    if "SI_UNIT($,.METRE.)" in blob or "SI_UNIT(*,.METRE.)" in blob:
        return "m"
    if "CONVERSION_BASED_UNIT" in blob and "INCH" in blob:
        return "in"
    return "unknown"


def guess_field_inch_scale(span: float) -> tuple[float, str]:
    """Pick m/mm/in so the largest extent lands near a 144 in FTC field."""
    candidates = [
        (1.0, "in", abs(float(span) - FIELD_SPAN_IN)),
        (1.0 / MM_PER_IN, "mm", abs(float(span) / MM_PER_IN - FIELD_SPAN_IN)),
        (IN_PER_M, "m", abs(float(span) * IN_PER_M - FIELD_SPAN_IN)),
    ]
    _scale, units, _err = min(candidates, key=lambda row: row[2])
    return _scale, units


def guess_piece_inch_scale(span: float, spec_diameter_in: float) -> tuple[float, str]:
    """Pick m/mm/in so max extent lands near the official scoring-element diameter."""
    spec = float(spec_diameter_in)
    candidates = [
        (1.0, "in", abs(float(span) - spec)),
        (1.0 / MM_PER_IN, "mm", abs(float(span) / MM_PER_IN - spec)),
        (IN_PER_M, "m", abs(float(span) * IN_PER_M - spec)),
    ]
    _scale, units, _err = min(candidates, key=lambda row: row[2])
    return _scale, units


def infer_up_axis(extents: Any) -> int:
    vals = [float(extents[0]), float(extents[1]), float(extents[2])]
    return int(min(range(3), key=lambda i: vals[i]))


def scale_for_units(units: str | None, span: float, *, spec_diameter_in: float | None = None) -> tuple[float, str]:
    if spec_diameter_in is not None:
        return guess_piece_inch_scale(span, spec_diameter_in)
    if units in (None, "", "unknown"):
        return guess_field_inch_scale(span)
    if units == "m":
        return IN_PER_M, "m"
    if units == "mm":
        return 1.0 / MM_PER_IN, "mm"
    return 1.0, "in"


def resolve_inch_scale(
    span: float,
    sniffed: str | None,
    *,
    spec_diameter_in: float | None = None,
    target_in: float | None = None,
) -> tuple[float, str]:
    """Prefer the scale that lands on the known inch size. Cascadio meshes are often metres even when STEP headers say INCH."""
    target = float(spec_diameter_in if spec_diameter_in is not None else (target_in or FIELD_SPAN_IN))
    guessed_scale, guessed_units = (
        guess_piece_inch_scale(span, spec_diameter_in) if spec_diameter_in is not None else guess_field_inch_scale(span)
    )
    if sniffed in (None, "", "unknown"):
        return guessed_scale, guessed_units
    if sniffed == "m":
        sniffed_scale, sniffed_name = IN_PER_M, "m"
    elif sniffed == "mm":
        sniffed_scale, sniffed_name = 1.0 / MM_PER_IN, "mm"
    else:
        sniffed_scale, sniffed_name = 1.0, "in"
    sniffed_err = abs(float(span) * sniffed_scale - target)
    guessed_err = abs(float(span) * guessed_scale - target)
    if sniffed_err > guessed_err + 0.5:
        return guessed_scale, guessed_units
    return sniffed_scale, sniffed_name


def filename_from_content_disposition(header: str | None) -> str | None:
    if not header:
        return None
    starred = re.search(r"filename\*\s*=\s*(?:UTF-8'')?([^;]+)", header, re.I)
    if starred:
        raw = starred.group(1).strip().strip('"')
        name = unquote(raw)
        return Path(name).name or None
    quoted = re.search(r'filename\s*=\s*"([^"]+)"', header, re.I)
    if quoted:
        return Path(quoted.group(1)).name or None
    plain = re.search(r"filename\s*=\s*([^;]+)", header, re.I)
    if plain:
        return Path(plain.group(1).strip().strip('"')).name or None
    return None


def sanitize_filename(name: str, fallback: str = "cad.step") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or fallback


def mesh_required(field: dict[str, Any]) -> bool:
    caps = field.get("requiredCapabilities") or []
    return "mesh_field_collision" in caps


def bounds_dict(obj: Any) -> dict[str, list[float]]:
    min_b, max_b = obj.bounds
    return {
        "minIn": round_xyz(min_b),
        "maxIn": round_xyz(max_b),
        "extentsIn": round_xyz(obj.extents),
    }


def _require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise CadImportError("trimesh missing; pip install -e '.[cad]'") from exc
    return trimesh


def _require_cascadio():
    try:
        import cascadio
    except ImportError as exc:
        raise CadImportError("cascadio missing; pip install -e '.[cad]'") from exc
    return cascadio


def face_count(obj: Any) -> int:
    faces = getattr(obj, "faces", None)
    if faces is not None:
        try:
            return int(len(faces))
        except TypeError:
            pass
    geoms = getattr(obj, "geometry", None)
    if isinstance(geoms, dict):
        total = 0
        for geom in geoms.values():
            geom_faces = getattr(geom, "faces", None)
            if geom_faces is None:
                continue
            total += int(len(geom_faces))
        return total
    return 0


def as_trimesh(obj: Any, trimesh: Any) -> Any:
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        dumped = obj.to_geometry() if hasattr(obj, "to_geometry") else obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
        geoms = dumped if isinstance(dumped, (list, tuple)) else [dumped]
        geoms = [g for g in geoms if isinstance(g, trimesh.Trimesh) and len(getattr(g, "faces", []))]
        if not geoms:
            raise CadImportError("STEP scene has no triangle mesh")
        if len(geoms) == 1:
            return geoms[0]
        return trimesh.util.concatenate(geoms)
    raise CadImportError(f"unsupported mesh type {type(obj).__name__}")


def _paint_lab_mesh(mesh: Any) -> Any:
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def decimate_visual(obj: Any, trimesh: Any, target: int = VISUAL_FACE_TARGET) -> tuple[Any, bool]:
    mesh = as_trimesh(obj, trimesh)
    if len(mesh.faces) <= target:
        return _paint_lab_mesh(mesh), False
    try:
        reduced = mesh.simplify_quadric_decimation(face_count=int(target))
    except Exception as exc:
        raise CadImportError(
            "field mesh is too dense and fast-simplification is missing; pip install -e '.[cad]'"
        ) from exc
    if reduced is None or len(getattr(reduced, "faces", [])) < 32:
        return _paint_lab_mesh(mesh), False
    return _paint_lab_mesh(reduced), True


def split_disconnected(mesh: Any, trimesh: Any) -> list[Any]:
    try:
        chunks = mesh.split(only_watertight=False)
    except Exception:
        return [mesh]
    out = [c for c in chunks if isinstance(c, trimesh.Trimesh) and len(getattr(c, "faces", [])) >= 4]
    return out or [mesh]


def iter_assembly_parts(obj: Any, trimesh: Any) -> list[tuple[str, Any]]:
    """Bake scene-graph parts. Never concatenate the whole field into one mesh."""
    if isinstance(obj, trimesh.Trimesh):
        chunks = split_disconnected(obj, trimesh)
        if len(chunks) == 1:
            return [("part_0000", chunks[0])]
        return [(f"part_{i:04d}", chunk) for i, chunk in enumerate(chunks)]
    if not isinstance(obj, trimesh.Scene):
        raise CadImportError(f"unsupported mesh type {type(obj).__name__}")
    baked: list[tuple[str, Any]] = []
    nodes = list(getattr(obj.graph, "nodes_geometry", None) or [])
    if not nodes:
        nodes = list(getattr(obj, "geometry", {}).keys())
    seen: dict[str, int] = {}
    for node in sorted(str(n) for n in nodes):
        transform = None
        geom_name = node
        graph = getattr(obj, "graph", None)
        if graph is not None and hasattr(graph, "get"):
            try:
                transform, geom_name = graph.get(node)
            except Exception:
                transform, geom_name = None, node
        geom = obj.geometry.get(geom_name) if geom_name in obj.geometry else obj.geometry.get(node)
        if not isinstance(geom, trimesh.Trimesh) or len(getattr(geom, "faces", [])) < 1:
            continue
        mesh = geom.copy()
        if transform is not None:
            mesh.apply_transform(transform)
        key = sanitize_filename(str(node), fallback="part")
        n = seen.get(key, 0)
        seen[key] = n + 1
        label = key if n == 0 else f"{key}_{n:02d}"
        baked.append((label, mesh))
    if not baked:
        return iter_assembly_parts(as_trimesh(obj, trimesh), trimesh)
    if len(baked) == 1:
        chunks = split_disconnected(baked[0][1], trimesh)
        if len(chunks) > 1:
            return [(f"part_{i:04d}", chunk) for i, chunk in enumerate(chunks)]
    return baked


def filter_structural_parts(parts: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    kept: list[tuple[str, Any]] = []
    for name, mesh in parts:
        if len(getattr(mesh, "faces", [])) < MIN_PART_FACES:
            continue
        extent = float(max(mesh.extents)) if len(getattr(mesh, "extents", [])) else 0.0
        if extent < MIN_PART_EXTENT_IN:
            continue
        kept.append((name, mesh))
    return kept or list(parts)


def _farthest_point_sample(points: Any, count: int) -> Any:
    import numpy as np

    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    k = min(int(count), n)
    if k >= n:
        return pts
    order = np.lexsort(pts.T)
    pts = pts[order]
    chosen = [0]
    dist = np.full(n, np.inf)
    for _ in range(k - 1):
        last = pts[chosen[-1]]
        dist = np.minimum(dist, np.linalg.norm(pts - last, axis=1))
        chosen.append(int(dist.argmax()))
    return pts[np.array(chosen, dtype=int)]


def convex_hull_mesh(mesh: Any, face_target: int = COLLISION_HULL_FACE_TARGET) -> Any:
    """Convex hull via scipy (MuJoCo-suitable). Avoid trimesh.convex_hull — it needs networkx."""
    trimesh = _require_trimesh()
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise CadImportError("scipy missing; pip install -e '.[cad]'") from exc
    points = np.asarray(getattr(mesh, "vertices", []), dtype=float)
    if points.ndim != 2 or points.shape[0] < 4:
        raise CadImportError("not enough vertices for a convex hull")
    unique = np.unique(np.round(points, 6), axis=0)
    if unique.shape[0] < 4:
        unique = points
    try:
        qhull = ConvexHull(unique)
    except Exception as exc:
        raise CadImportError(f"convex hull failed: {exc}") from exc
    extrema = unique[np.unique(qhull.vertices)]
    # Dense tessellations put thousands of extrema on a sphere; cap before the second hull.
    sample_target = max(12, min(int(face_target), 48))
    if extrema.shape[0] > sample_target:
        extrema = _farthest_point_sample(extrema, sample_target)
    try:
        qhull = ConvexHull(extrema)
    except Exception as exc:
        raise CadImportError(f"convex hull failed: {exc}") from exc
    return trimesh.Trimesh(vertices=extrema, faces=qhull.simplices, process=True)


def hull_is_much_larger(mesh: Any, hull: Any, ratio: float = 2.5) -> bool:
    try:
        volume = float(mesh.volume)
        hull_volume = float(hull.volume)
    except Exception:
        return False
    if volume <= 1e-6:
        return False
    return hull_volume / volume > ratio


def stage_step(source: Path, dest_name: str | None = None) -> Path:
    source = Path(source)
    if not source.is_file():
        raise CadImportError(f"STEP not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".step", ".stp"}:
        raise CadImportError(f"expected .step/.stp, got {suffix or 'no suffix'}")
    dest = cad_cache_dir() / sanitize_filename(dest_name or source.name)
    if dest.resolve() != source.resolve():
        shutil.copy2(source, dest)
    return dest


def download_step(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    timeout: float = 180.0,
) -> tuple[Path, str, str]:
    """Stream a STEP (or Content-Disposition attachment) to dest. Returns path, sha256, filename."""
    import httpx

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    hasher = hashlib.sha256()
    filename = Path(unquote(urlparse(url).path)).name
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                cd_name = filename_from_content_disposition(resp.headers.get("content-disposition"))
                if cd_name:
                    filename = cd_name
                first = True
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        if first:
                            first = False
                            stripped = chunk.lstrip()
                            if stripped.lower().startswith(b"<!doctype") or stripped.lower().startswith(b"<html"):
                                raise CadImportError(f"URL returned HTML, not STEP: {url}")
                        hasher.update(chunk)
                        fh.write(chunk)
    except CadImportError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise CadImportError(f"STEP download failed ({url}): {exc}") from exc
    if tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        raise CadImportError(f"STEP download too small: {url}")
    head = tmp.read_bytes()[:24]
    if not head.lstrip().startswith(STEP_MAGIC) and b"ISO-10303" not in head:
        # Some exporters prepend a BOM or whitespace; sniff a larger prefix.
        prefix = tmp.read_bytes()[:256].lstrip()
        if not prefix.startswith(STEP_MAGIC):
            tmp.unlink(missing_ok=True)
            raise CadImportError(f"downloaded file is not a STEP (ISO-10303-21): {url}")
    digest = hasher.hexdigest()
    if expected_sha256 and digest != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise CadImportError(
            f"SHA-256 mismatch for {url}: got {digest}, expected {expected_sha256.lower()}"
        )
    final_name = sanitize_filename(filename, fallback=dest.name)
    final = dest if dest.name == final_name else dest.with_name(final_name)
    if dest.name != final_name:
        # Caller passed a preferred dest; keep it if the suffix is already .step/.stp.
        if dest.suffix.lower() in {".step", ".stp"}:
            final = dest
    if final.exists():
        final.unlink()
    tmp.rename(final)
    return final, digest, filename


def tessellate_step_to_glb(
    step_path: Path,
    dest_glb: Path,
    *,
    units: str = "unknown",
    tol_linear: float | None = None,
    merge_primitives: bool = True,
    piece: bool = False,
) -> Path:
    cascadio = _require_cascadio()
    dest_glb.parent.mkdir(parents=True, exist_ok=True)
    tol = float(tol_linear) if tol_linear is not None else default_linear_deflection(units, piece=piece)
    kwargs: dict[str, Any] = {
        "tol_linear": tol,
        "tol_angular": 0.25,
        "merge_primitives": merge_primitives,
        # OpenCASCADE's parallel field tessellator can segfault on this 35 MB
        # assembly at useful visual tolerances; piece files remain safe and small.
        "use_parallel": piece,
    }
    try:
        cascadio.step_to_glb(str(step_path), str(dest_glb), include_materials=True, **kwargs)
    except TypeError:
        cascadio.step_to_glb(str(step_path), str(dest_glb), **kwargs)
    if not dest_glb.is_file() or dest_glb.stat().st_size < 64:
        raise CadImportError(f"cascadio wrote no GLB at {dest_glb}")
    return dest_glb


def load_tessellated_step(
    step_path: Path,
    *,
    units: str | None = None,
    tol_linear: float | None = None,
    raw_glb: Path | None = None,
    merge_primitives: bool = True,
    piece: bool = False,
) -> tuple[Any, dict[str, Any]]:
    trimesh = _require_trimesh()
    _require_cascadio()
    step_path = Path(step_path)
    sniffed = units or sniff_step_units(step_path)
    raw_glb = raw_glb or (cad_cache_dir() / f"{step_path.stem}.raw.glb")
    reuse = (
        raw_glb.is_file()
        and raw_glb.stat().st_size > 64
        and raw_glb.stat().st_mtime >= step_path.stat().st_mtime
        and tol_linear is None
    )
    if not reuse:
        tessellate_step_to_glb(
            step_path,
            raw_glb,
            units=sniffed,
            tol_linear=tol_linear,
            merge_primitives=merge_primitives,
            piece=piece,
        )
    loaded = trimesh.load(str(raw_glb), force=None)
    if loaded is None:
        raise CadImportError(f"trimesh could not load tessellated {raw_glb}")
    meta = {
        "rawGlb": str(raw_glb),
        "sniffedUnits": sniffed,
        "tolLinear": float(tol_linear) if tol_linear is not None else default_linear_deflection(sniffed, piece=piece),
        "mergePrimitives": merge_primitives,
        "faceCount": face_count(loaded),
    }
    return loaded, meta


def align_field_mesh(obj: Any, units: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Scale to inches, Z-up CAD -> Y-up Lab, origin at field center, sit on the floor."""
    extents = obj.extents
    span = float(max(extents)) if len(extents) else 0.0
    if span < 1e-9:
        raise CadImportError("STEP mesh has zero size")
    scale, units_guess = resolve_inch_scale(span, units)
    if abs(scale - 1.0) > 1e-9:
        obj.apply_scale(scale)
    up = infer_up_axis(obj.extents)
    rotated = False
    if up == 2:
        obj.apply_transform(YUP_FROM_ZUP)
        rotated = True
    min_b, max_b = obj.bounds
    cx = 0.5 * (float(min_b[0]) + float(max_b[0]))
    cz = 0.5 * (float(min_b[2]) + float(max_b[2]))
    translation = [0.0, 0.0, 0.0]
    # Official FTC CAD is origin-at-center; only slide if the assembly is clearly offset.
    if abs(cx) > 8.0 or abs(cz) > 8.0:
        obj.apply_translation([-cx, 0.0, -cz])
        translation[0] -= cx
        translation[2] -= cz
        min_b, max_b = obj.bounds
    floor = float(min_b[1])
    if abs(floor) > 0.05:
        obj.apply_translation([0.0, -floor, 0.0])
        translation[1] -= floor
    meta = {
        "unitsGuess": units_guess,
        "rotatedZupToYup": rotated,
        "scaleToInches": scale,
        "translationIn": round_xyz(translation),
        "faceCount": face_count(obj),
        **bounds_dict(obj),
    }
    return obj, meta


def align_piece_mesh(
    obj: Any,
    units: str | None = None,
    *,
    spec_diameter_in: float,
    assume_z_up: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Scale to inches, STEP Z-up -> Y-up, origin at the geometric center (free-joint body)."""
    extents = obj.extents
    span = float(max(extents)) if len(extents) else 0.0
    if span < 1e-9:
        raise CadImportError("piece STEP mesh has zero size")
    scale, units_guess = resolve_inch_scale(span, units, spec_diameter_in=spec_diameter_in)
    if abs(scale - 1.0) > 1e-9:
        obj.apply_scale(scale)
    rotated = False
    if assume_z_up:
        obj.apply_transform(YUP_FROM_ZUP)
        rotated = True
    min_b, max_b = obj.bounds
    cx = 0.5 * (float(min_b[0]) + float(max_b[0]))
    cy = 0.5 * (float(min_b[1]) + float(max_b[1]))
    cz = 0.5 * (float(min_b[2]) + float(max_b[2]))
    obj.apply_translation([-cx, -cy, -cz])
    measured = float(max(obj.extents)) if len(obj.extents) else 0.0
    meta = {
        "unitsGuess": units_guess,
        "rotatedZupToYup": rotated,
        "scaleToInches": scale,
        "translationIn": round_xyz([-cx, -cy, -cz]),
        "faceCount": face_count(obj),
        "specDiameterIn": float(spec_diameter_in),
        "measuredDiameterIn": round(measured, 3),
        **bounds_dict(obj),
    }
    return obj, meta


def check_piece_diameter(measured_in: float, spec_diameter_in: float, *, rel_tol: float = 0.12) -> None:
    spec = float(spec_diameter_in)
    measured = float(measured_in)
    abs_tol = max(0.35, spec * rel_tol)
    if abs(measured - spec) > abs_tol:
        raise CadImportError(
            f"piece diameter {measured:.3f} in is outside {spec:.3f} in ± {abs_tol:.3f} in"
        )


def export_visual_glb(obj: Any, dest: Path, trimesh: Any, *, face_target: int = VISUAL_FACE_TARGET) -> tuple[Path, dict[str, Any]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    visual = obj
    decimated = False
    if face_count(obj) > face_target:
        visual, decimated = decimate_visual(obj, trimesh, target=face_target)
    visual.export(str(dest), file_type="glb")
    if not dest.is_file() or dest.stat().st_size < 64:
        raise CadImportError(f"failed to write visual GLB {dest}")
    return dest, {"decimated": decimated, "faceCount": face_count(visual), "bytes": dest.stat().st_size}


def field_tile_surface_y(parts: list[tuple[str, Any]]) -> float:
    """Return the official soft-tile top in the current Y-up frame."""
    tops = [
        float(mesh.bounds[1][1])
        for name, mesh in parts
        if "soft_tiles" in name.lower() and len(getattr(mesh, "bounds", [])) == 2
    ]
    if not tops:
        raise CadImportError("field STEP has no named soft-tile parts; cannot establish z=0")
    return max(tops)


def classify_field_visual_part(
    name: str,
    mesh: Any,
    *,
    playable_half: float = 72.0,
    margin: float = 2.0,
) -> str:
    """Classify STEP parts for the Lab visual.

    The full-field download also contains staged scoring elements and alliance-area
    hardware. Those are runtime state, not static field geometry.
    """
    lowered = str(name).lower()
    if "pollen" in lowered or "nectar" in lowered:
        return "scoring_element"
    if "under_tile" in lowered or "under_field" in lowered:
        return "under_field"
    if "artifact_tray" in lowered:
        return "alliance_area"
    if any(
        token in lowered
        for token in (
            "socket_head_cap_screw",
            "pan_head_machine_screw",
            "sheet_metal_screw",
            "elevator_bolt",
            "wing_nut",
            "hex_lock_nut",
            "fender_washer",
            "nylon_spacer",
            "flanged_bearing",
            "quick_release_pin",
            "press_in_plug",
            "cable_tie",
        )
    ):
        return "fastener"
    min_b, max_b = mesh.bounds
    limit = float(playable_half) + float(margin)
    if (
        float(min_b[0]) < -limit
        or float(max_b[0]) > limit
        or float(min_b[2]) < -limit
        or float(max_b[2]) > limit
    ):
        return "off_field"
    return "playable"


_MOVING_HIVE_TOKENS = (
    "goal_rib",
    "hive_goal_top_skin",
    "hive_goal_back_skin",
    "hive_goal_bottom_skin",
    "10.5in_churro_lite",
    "basket_base_tube",
    "goal_april_tag",
)


def moving_hive_alliance(name: str, center_x: float) -> str | None:
    """Identify the two rotating HIVE baskets from stable STEP part names."""
    lowered = str(name).lower()
    if not any(token in lowered for token in _MOVING_HIVE_TOKENS):
        return None
    return "red" if float(center_x) < 0.0 else "blue"


def split_moving_hive_parts(
    parts: list[tuple[str, Any]],
) -> tuple[list[tuple[str, Any]], dict[str, list[tuple[str, Any]]]]:
    static: list[tuple[str, Any]] = []
    moving: dict[str, list[tuple[str, Any]]] = {"red": [], "blue": []}
    for name, mesh in parts:
        alliance = moving_hive_alliance(name, float(mesh.centroid[0]))
        if alliance:
            moving[alliance].append((name, mesh))
        else:
            static.append((name, mesh))
    return static, moving


def select_field_visual_parts(
    parts: list[tuple[str, Any]],
    *,
    playable_half: float = 72.0,
    margin: float = 2.0,
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    kept: list[tuple[str, Any]] = []
    dropped: dict[str, int] = {}
    for name, mesh in parts:
        reason = classify_field_visual_part(name, mesh, playable_half=playable_half, margin=margin)
        if reason == "playable":
            kept.append((name, mesh))
        else:
            dropped[reason] = dropped.get(reason, 0) + 1
    if not kept:
        raise CadImportError("field visual filter removed every STEP part")
    return kept, {"sourceParts": len(parts), "kept": len(kept), "dropped": dropped}


def export_visual_parts_glb(
    parts: list[tuple[str, Any]],
    dest: Path,
    trimesh: Any,
    *,
    face_target: int = VISUAL_FACE_TARGET,
) -> tuple[Path, dict[str, Any]]:
    """Export a named, filtered scene while retaining per-part base colors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total_faces = sum(len(getattr(mesh, "faces", [])) for _name, mesh in parts)
    ratio = min(1.0, float(face_target) / max(1, total_faces))
    scene = trimesh.Scene()
    decimated = False
    written_faces = 0
    for name, source in parts:
        mesh = source.copy()
        try:
            mesh.merge_vertices()
            mesh.update_faces(mesh.area_faces > 1e-6)
            if hasattr(mesh, "nondegenerate_faces"):
                mesh.update_faces(mesh.nondegenerate_faces())
            elif hasattr(mesh, "remove_degenerate_faces"):
                mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals(multibody=True)
        except (AttributeError, TypeError, ValueError):
            # Some trimesh releases expose only a subset of the repair API.
            # Export still proceeds; the generated-asset quality tests catch
            # any remaining invalid triangles.
            pass
        source_faces = len(getattr(mesh, "faces", []))
        budget = max(24, int(round(source_faces * ratio)))
        lowered = name.lower()
        preserve_topology = any(
            token in lowered
            for token in (
                "field_side_glass",
                "gaffer_tape",
                "panel_sticker",
            )
        )
        quality_budget = 0
        if any(token in lowered for token in ("goal_rib", "hive_goal_", "a-frame")):
            quality_budget = 4096
        elif "flower" in lowered:
            quality_budget = 2048
        elif "rail_with_rivet" in lowered:
            quality_budget = 1024
        if quality_budget:
            budget = max(budget, min(source_faces, quality_budget))
        if source_faces > budget and not preserve_topology:
            try:
                color = getattr(mesh.visual, "main_color", None)
                mesh = mesh.simplify_quadric_decimation(face_count=budget)
                if color is not None:
                    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=color)
                mesh.update_faces(mesh.area_faces > 1e-6)
                mesh.remove_unreferenced_vertices()
                decimated = True
            except Exception as exc:
                raise CadImportError(
                    "field visual is too dense and could not be simplified; "
                    "pip install -e '.[cad]'"
                ) from exc
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        written_faces += len(getattr(mesh, "faces", []))
    scene.export(str(dest), file_type="glb")
    if not dest.is_file() or dest.stat().st_size < 64:
        raise CadImportError(f"failed to write visual GLB {dest}")
    return dest, {
        "decimated": decimated,
        "sourceFaceCount": total_faces,
        "faceCount": written_faces,
        "bytes": dest.stat().st_size,
    }


def write_convex_collision_parts(
    parts: list[tuple[str, Any]],
    dest_dir: Path,
    *,
    rel_prefix: str,
    allow_single_concave: bool = False,
) -> list[dict[str, Any]]:
    """Write one convex STL per assembly part. Refuse a single concave field hull."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    structural = filter_structural_parts(parts)
    if not structural:
        raise CadImportError("STEP assembly produced no collision parts")
    hulls: list[tuple[str, Any, Any]] = []
    concave = 0
    for name, mesh in structural:
        hull = convex_hull_mesh(mesh)
        if hull_is_much_larger(mesh, hull):
            concave += 1
        hulls.append((name, mesh, hull))
    if len(hulls) == 1 and not allow_single_concave:
        raise CadImportError(
            "field STEP collapsed to one collision mesh; MuJoCo convexifies concave "
            "meshes and would seal HIVE/CELL openings. Preserve assembly parts."
        )
    written: list[dict[str, Any]] = []
    used: set[str] = set()
    for index, (name, _mesh, hull) in enumerate(hulls):
        stem = sanitize_filename(name, fallback=f"part_{index:04d}")
        if stem in used:
            stem = f"{stem}_{index:04d}"
        used.add(stem)
        path = dest_dir / f"{stem}.stl"
        hull.export(str(path), file_type="stl")
        rel = f"{rel_prefix}/{stem}.stl".replace("\\", "/")
        written.append(
            {
                "id": stem,
                "asset": rel,
                "convex": True,
                "faceCount": int(len(getattr(hull, "faces", []))),
                **bounds_dict(hull),
            }
        )
    return written


def dump_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
