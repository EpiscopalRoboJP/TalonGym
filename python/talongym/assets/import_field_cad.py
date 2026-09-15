"""Download official field STEP if possible; always emit glTF + MJCF colliders.

Raw STEP is written under var/cad/ and is not committed. Converted assets live in
assets/seasons/<slug>/. HubSpot often hides the direct CAD URL — we scrape the
archive page, then fall back to tessellating the field preset AABBs.

When a local STEP is supplied, Lab glTF is tessellated from that file. MuJoCo
colliders stay as field-preset AABBs so robots can still drive under the hive.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from talongym.assets.cad_layout import (
    build_layout,
    cad_part_name,
    lab_bounds_to_placement,
    sync_field_to_cad_layout,
    write_preset_json,
)
from talongym.assets.gltf_boxes import solids_from_field, write_glb
from talongym.assets.mjcf_field import build_mjcf
from talongym.paths import ASSETS_DIR, PRESETS_DIR, VAR_DIR

DEFAULT_CAD_PAGE = "https://ftc-resources.firstinspires.org/ftc/archive/2027/field"
STEP_RE = re.compile(r"""href=["']([^"']+\.(?:step|stp|STEP|STP))["']""", re.I)
ONSHAPE_RE = re.compile(r"""href=["'](https://[^"']*onshape[^"']*)["']""", re.I)
IN_PER_M = 39.37007874015748
MM_PER_IN = 25.4
FIELD_SPAN_IN = 144.0
# OpenCASCADE linear deflection in source units (metres for Onshape AP242).
# 20 mm is coarse enough for Lab; we still quadric-decimate after tessellation.
DEFAULT_TOL_BY_UNIT = {"m": 0.02, "mm": 20.0, "in": 0.75, "unknown": 0.02}
VISUAL_FACE_TARGET = 350_000
# Screws, pins, rivets and cable ties are ~30% of official field CAD faces and invisible at field scale.
HARDWARE_MAX_EXTENT_IN = 2.0
PART_MIN_FACES = 500
MAX_DECIMATED_FRACTION = 0.5
# FTC (x, y floor, z height) -> Three/MuJoCo Y-up (x, height, -y).
YUP_FROM_ZUP = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class CadImportError(RuntimeError):
    pass


def _season_slug(field: dict[str, Any]) -> str:
    season = field.get("season") or {}
    return str(season.get("slug") or field.get("id") or "field")


def asset_dir_for(field: dict[str, Any], year_hint: str = "2026") -> Path:
    slug = _season_slug(field)
    return ASSETS_DIR / "seasons" / f"{slug}_{year_hint}"


def load_field_json(path: Path | None = None) -> dict[str, Any]:
    path = path or (PRESETS_DIR / "seasons" / "biobuzz_2026" / "field.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_step_urls(page_url: str = DEFAULT_CAD_PAGE, html: str | None = None) -> list[str]:
    if html is None:
        import urllib.request

        req = urllib.request.Request(page_url, headers={"User-Agent": "TalonGym/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    found = [urljoin(page_url, m.group(1)) for m in STEP_RE.finditer(html)]
    # Prefer filenames that look like the full field, not a single scoring volume.
    found.sort(key=lambda u: (0 if "flower" in u.lower() else 1, len(u)))
    fieldish = [u for u in found if "flower" not in u.lower()]
    return fieldish or found


def download_step(url: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "TalonGym/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    if dest.stat().st_size < 1024:
        raise CadImportError(f"STEP download too small: {dest}")
    return dest


def cad_cache_dir() -> Path:
    dest = VAR_DIR / "cad"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def stage_step(source: Path) -> Path:
    source = Path(source)
    if not source.is_file():
        raise CadImportError(f"STEP not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".step", ".stp"}:
        raise CadImportError(f"expected .step/.stp, got {suffix or 'no suffix'}")
    dest = cad_cache_dir() / re.sub(r"[^A-Za-z0-9._-]+", "_", source.name)
    if dest.resolve() != source.resolve():
        shutil.copy2(source, dest)
    return dest


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


def infer_up_axis(extents: Any) -> int:
    vals = [float(extents[0]), float(extents[1]), float(extents[2])]
    return int(min(range(3), key=lambda i: vals[i]))


def default_linear_deflection(units: str) -> float:
    return float(DEFAULT_TOL_BY_UNIT.get(units, DEFAULT_TOL_BY_UNIT["unknown"]))


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


def _face_count(obj: Any) -> int:
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


def _as_trimesh(obj: Any, trimesh: Any) -> Any:
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
    """Keep STEP tessellation exportable; Lab lights/materials are applied in FieldScene."""
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _visual_parts(
    obj: Any,
    trimesh: Any,
    piece_parts: frozenset[str] = frozenset(),
) -> tuple[list[Any], list[tuple[str, Any]]]:
    """Baked mesh per assembly instance for Lab, plus (instance name, Lab bounds) placements.

    Fasteners too small to see are dropped. In-field instances of ``piece_parts`` are placed
    but not drawn: the sim spawns and renders those game pieces itself.
    """
    if isinstance(obj, trimesh.Trimesh):
        return [obj], []
    if not isinstance(obj, trimesh.Scene):
        raise CadImportError(f"unsupported mesh type {type(obj).__name__}")
    parts = []
    placements = []
    half = 0.5 * FIELD_SPAN_IN
    for node in obj.graph.nodes_geometry:
        transform, geom_name = obj.graph[node]
        geom = obj.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or not len(geom.faces):
            continue
        part = geom.copy()
        part.apply_transform(transform)
        if float(max(part.extents)) < HARDWARE_MAX_EXTENT_IN:
            continue
        lo, hi = part.bounds
        placements.append((geom_name, part.bounds.copy()))
        in_field = abs(0.5 * (lo[0] + hi[0])) < half and abs(0.5 * (lo[2] + hi[2])) < half
        if in_field and cad_part_name(geom_name) in piece_parts:
            continue
        # glTF splits vertices at every CAD face edge; unwelded seams read as open borders
        # to the decimator, which then shrinks each surface patch away from its neighbours.
        part.merge_vertices(merge_tex=True, merge_norm=True)
        parts.append(part)
    if not parts:
        raise CadImportError("STEP scene has no triangle mesh")
    return parts, placements


def _part_face_budgets(face_counts: list[int], target: int) -> list[int]:
    """Scale every part by one ratio, never taking a part below PART_MIN_FACES."""

    def budgets(ratio: float) -> list[int]:
        return [min(n, max(PART_MIN_FACES, int(n * ratio))) for n in face_counts]

    if sum(face_counts) <= target:
        return list(face_counts)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if sum(budgets(mid)) > target:
            hi = mid
        else:
            lo = mid
    return budgets(lo)


def _decimate_visual(parts: list[Any], trimesh: Any, target: int = VISUAL_FACE_TARGET) -> tuple[Any, bool]:
    # Decimating the merged assembly as one mesh lets collapses cross part boundaries,
    # which punches holes in flat panels and leaves floating shards. Budget per part.
    counts = [len(p.faces) for p in parts]
    budgets = _part_face_budgets(counts, target)
    decimated = False
    out = []
    for part, count, budget in zip(parts, counts, budgets, strict=True):
        if budget < count:
            try:
                reduced = part.simplify_quadric_decimation(face_count=int(budget))
            except Exception as exc:
                raise CadImportError(
                    "field mesh is too dense and fast-simplification is missing; pip install -e '.[cad]'"
                ) from exc
            # A decimator that stalls far above budget only made the collapses it was forced
            # into, which on foam tiles open holes where four corners meet. Keep those exact.
            kept = len(getattr(reduced, "faces", [])) if reduced is not None else 0
            if 4 <= kept <= count * MAX_DECIMATED_FRACTION:
                part = reduced
                decimated = True
        out.append(part)
    mesh = out[0] if len(out) == 1 else trimesh.util.concatenate(out)
    return _paint_lab_mesh(mesh), decimated


def tile_surface_height(mesh: Any, bin_in: float = 0.05, search_in: float = 6.0) -> float:
    """Height of the largest upward-facing surface near the floor inside the field footprint.

    Official CAD puts parts (under-tile bars) below the foam tiles, so the lowest vertex is
    not where robots drive. Lab and physics treat y=0 as the tile top.
    """
    normals = mesh.face_normals
    centers = mesh.triangles_center
    half = 0.5 * FIELD_SPAN_IN
    low = float(mesh.bounds[0][1])
    mask = (
        (normals[:, 1] > 0.99)
        & (abs(centers[:, 0]) < half)
        & (abs(centers[:, 2]) < half)
        & (centers[:, 1] < low + search_in)
    )
    if not mask.any():
        return low
    bins = ((centers[mask, 1] - low) / bin_in).astype(int)
    areas: dict[int, float] = {}
    for b, area in zip(bins.tolist(), mesh.area_faces[mask].tolist(), strict=True):
        areas[b] = areas.get(b, 0.0) + area
    best = max(areas, key=areas.__getitem__)
    ys = centers[mask, 1][bins == best]
    return float(ys.max())


def align_field_mesh(obj: Any, units: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Scale to inches, Z-up CAD -> Y-up Lab, origin at field center, sit on the floor."""
    extents = obj.extents
    span = float(max(extents)) if len(extents) else 0.0
    if span < 1e-9:
        raise CadImportError("STEP mesh has zero size")
    if units in (None, "", "unknown"):
        scale, units_guess = guess_field_inch_scale(span)
    elif units == "m":
        scale, units_guess = IN_PER_M, "m"
    elif units == "mm":
        scale, units_guess = 1.0 / MM_PER_IN, "mm"
    else:
        scale, units_guess = 1.0, "in"
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
    # Official FTC CAD is origin-at-center; only slide if the assembly is clearly offset.
    if abs(cx) > 8.0 or abs(cz) > 8.0:
        obj.apply_translation([-cx, 0.0, -cz])
        min_b, max_b = obj.bounds
    floor = float(min_b[1])
    if abs(floor) > 0.05:
        obj.apply_translation([0.0, -floor, 0.0])
    min_b, max_b = obj.bounds
    meta = {
        "unitsGuess": units_guess,
        "rotatedZupToYup": rotated,
        "faceCount": _face_count(obj),
        "extentsIn": [round(float(v), 3) for v in obj.extents],
        "minIn": [round(float(v), 3) for v in min_b],
        "maxIn": [round(float(v), 3) for v in max_b],
    }
    return obj, meta


def tessellate_step_to_glb(
    step_path: Path,
    dest_glb: Path,
    *,
    units: str = "unknown",
    tol_linear: float | None = None,
) -> Path:
    cascadio = _require_cascadio()
    dest_glb.parent.mkdir(parents=True, exist_ok=True)
    tol = float(tol_linear) if tol_linear is not None else default_linear_deflection(units)
    kwargs: dict[str, Any] = {
        "tol_linear": tol,
        "tol_angular": 0.5,
        "merge_primitives": True,
        "use_parallel": True,
    }
    try:
        cascadio.step_to_glb(str(step_path), str(dest_glb), include_materials=True, **kwargs)
    except TypeError:
        cascadio.step_to_glb(str(step_path), str(dest_glb), **kwargs)
    if not dest_glb.is_file() or dest_glb.stat().st_size < 64:
        raise CadImportError(f"cascadio wrote no GLB at {dest_glb}")
    return dest_glb


def convert_step_to_trimesh(
    step_path: Path,
    *,
    units: str | None = None,
    tol_linear: float | None = None,
    raw_glb: Path | None = None,
    field: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Lab mesh, import meta, and the CAD coordinate layout (see cad_layout)."""
    field = field or {}
    piece_parts = frozenset(gp["cadPart"] for gp in field.get("gamePieces") or [] if gp.get("cadPart"))
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
        tessellate_step_to_glb(step_path, raw_glb, units=sniffed, tol_linear=tol_linear)
    loaded = trimesh.load(str(raw_glb), force=None)
    if loaded is None:
        raise CadImportError(f"trimesh could not load tessellated {raw_glb}")
    aligned, meta = align_field_mesh(loaded, units=sniffed)
    parts, placements = _visual_parts(aligned, trimesh, piece_parts)
    visual, decimated = _decimate_visual(parts, trimesh)
    tile_top = tile_surface_height(visual)
    visual.apply_translation([0.0, -tile_top, 0.0])
    layout = build_layout(
        [lab_bounds_to_placement(name, b[0], b[1], tile_top) for name, b in placements], field, tile_top
    )
    meta["tileSurfaceIn"] = round(tile_top, 3)
    meta["minIn"] = [round(float(v), 3) for v in visual.bounds[0]]
    meta["maxIn"] = [round(float(v), 3) for v in visual.bounds[1]]
    meta["rawGlb"] = str(raw_glb)
    meta["sniffedUnits"] = sniffed
    meta["tolLinear"] = float(tol_linear) if tol_linear is not None else default_linear_deflection(sniffed)
    meta["decimated"] = decimated
    meta["faceCount"] = _face_count(visual)
    return visual, meta, layout


def write_assets_from_field(
    field: dict[str, Any],
    dest_dir: Path | None = None,
    year_hint: str = "2026",
) -> dict[str, str]:
    dest_dir = dest_dir or asset_dir_for(field, year_hint)
    dest_dir.mkdir(parents=True, exist_ok=True)
    glb_path = dest_dir / "field.glb"
    xml_path = dest_dir / "field_mjcf.xml"
    write_glb(glb_path, solids_from_field(field))
    xml_path.write_text(build_mjcf(field), encoding="utf-8")
    rel_glb = str(glb_path.relative_to(ASSETS_DIR)).replace("\\", "/")
    rel_xml = str(xml_path.relative_to(ASSETS_DIR)).replace("\\", "/")
    return {"glb": str(glb_path), "mjcf": str(xml_path), "backgroundAsset": rel_glb, "collisionAsset": rel_xml}


def write_step_visual_and_aabb_colliders(
    field: dict[str, Any],
    mesh: Any,
    *,
    year_hint: str = "2026",
) -> dict[str, str]:
    dest_dir = asset_dir_for(field, year_hint)
    dest_dir.mkdir(parents=True, exist_ok=True)
    glb_path = dest_dir / "field.glb"
    xml_path = dest_dir / "field_mjcf.xml"
    mesh.export(str(glb_path), file_type="glb")
    xml_path.write_text(build_mjcf(field), encoding="utf-8")
    rel_glb = str(glb_path.relative_to(ASSETS_DIR)).replace("\\", "/")
    rel_xml = str(xml_path.relative_to(ASSETS_DIR)).replace("\\", "/")
    return {
        "glb": str(glb_path),
        "mjcf": str(xml_path),
        "backgroundAsset": rel_glb,
        "collisionAsset": rel_xml,
        "glbBytes": str(glb_path.stat().st_size),
    }


def try_official_step(page_url: str = DEFAULT_CAD_PAGE) -> Path | None:
    urls = discover_step_urls(page_url)
    if not urls:
        return None
    dest = cad_cache_dir() / "field.step"
    try:
        return download_step(urls[-1], dest)
    except Exception:
        return None


def _import_step(
    field: dict[str, Any],
    field_path: Path,
    step: Path,
    tol_linear: float | None,
    year_hint: str,
) -> dict[str, Any]:
    mesh, meta, layout = convert_step_to_trimesh(step, tol_linear=tol_linear, field=field)
    synced, sync_report = sync_field_to_cad_layout(field, layout)
    if synced != field:
        write_preset_json(field_path, synced)
    paths = write_step_visual_and_aabb_colliders(synced, mesh, year_hint=year_hint)
    layout_path = Path(paths["glb"]).with_name("field_layout.json")
    layout_path.write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "source": f"official_step:{step.name}",
        "step": str(step),
        **paths,
        "layout": str(layout_path),
        "fieldPreset": str(field_path),
        "presetSync": sync_report,
        **meta,
    }


def import_field_cad(
    field_path: Path | None = None,
    page_url: str = DEFAULT_CAD_PAGE,
    year_hint: str = "2026",
    step_path: Path | None = None,
    tol_linear: float | None = None,
) -> dict[str, Any]:
    field_path = Path(field_path) if field_path else PRESETS_DIR / "seasons" / "biobuzz_2026" / "field.json"
    field = load_field_json(field_path)
    if step_path is not None:
        return _import_step(field, field_path, stage_step(Path(step_path)), tol_linear, year_hint)
    step = try_official_step(page_url)
    note = "aabb_tessellation"
    if step is not None:
        try:
            return _import_step(field, field_path, step, tol_linear, year_hint)
        except Exception as exc:
            note = f"step_failed:{exc}"
    paths = write_assets_from_field(field, year_hint=year_hint)
    return {"ok": True, "source": note, **paths, "cadPage": page_url}


def main() -> None:
    result = import_field_cad()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
