"""Deterministic official field STEP import: visual GLB, convex parts, manifest.

Raw STEP stays under var/cad/ (gitignored). Derived GLB/STL/manifest live in
assets/seasons/<slug>/ and are committed for the shipped season; rebuild with
import-field-cad after an official STEP change.
Seasons that require mesh_field_collision fail if the
verified STEP or derived collision parts are missing — they are never replaced
with schematic AABB boxes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from talongym.assets.cad_common import (
    CAD_GENERATOR_VERSION,
    CadImportError,
    align_field_mesh,
    as_trimesh,
    bounds_dict,
    cad_cache_dir,
    decimate_visual,
    download_step,
    export_visual_parts_glb,
    field_tile_surface_y,
    guess_field_inch_scale,
    infer_up_axis,
    iter_assembly_parts,
    load_tessellated_step,
    mesh_required,
    round_xyz,
    select_field_visual_parts,
    split_moving_hive_parts,
    moving_hive_alliance,
    sha256_file,
    sniff_step_units,
    stage_step,
    write_convex_collision_parts,
)
from talongym.assets.cad_manifest import (
    asset_relpath,
    bounds_block,
    build_manifest,
    load_manifest,
    manifest_relpath,
    source_block,
    transform_block,
    validate_manifest,
    verify_manifest_files,
    write_manifest,
)
from talongym.assets.cad_sources import (
    FIELD_SOURCE,
    OFFICIAL_FIELD_PAGE,
    OFFICIAL_FIELD_STEP_URL,
    expected_sha256,
)
from talongym.assets.gltf_boxes import solids_from_field, write_glb
from talongym.assets.mjcf_field import build_mjcf
from talongym.paths import ASSETS_DIR, PRESETS_DIR

DEFAULT_CAD_PAGE = OFFICIAL_FIELD_PAGE
STEP_RE = re.compile(r"""href=["']([^"']+\.(?:step|stp|STEP|STP))["']""", re.I)

# Re-export helpers tests and CLI already import.
_as_trimesh = as_trimesh
__all__ = [
    "CAD_GENERATOR_VERSION",
    "CadImportError",
    "DEFAULT_CAD_PAGE",
    "OFFICIAL_FIELD_STEP_URL",
    "align_field_mesh",
    "asset_dir_for",
    "discover_step_urls",
    "guess_field_inch_scale",
    "import_field_cad",
    "infer_up_axis",
    "sniff_step_units",
    "stage_step",
    "verify_season_cad",
    "write_assets_from_field",
]


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
        import httpx

        resp = httpx.get(page_url, headers={"User-Agent": "TalonGym/0.1"}, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    found = [urljoin(page_url, m.group(1)) for m in STEP_RE.finditer(html)]
    found.sort(key=lambda u: (0 if "flower" in u.lower() else 1, len(u)))
    fieldish = [u for u in found if "flower" not in u.lower()]
    return fieldish or found


def fetch_official_field_step(
    *,
    url: str | None = None,
    page_url: str = DEFAULT_CAD_PAGE,
    dest_name: str | None = None,
) -> tuple[Path, str, str]:
    expected = expected_sha256(FIELD_SOURCE)
    dest = cad_cache_dir() / (dest_name or str(FIELD_SOURCE.get("filename") or "field.step"))
    primary = url or OFFICIAL_FIELD_STEP_URL
    try:
        return download_step(primary, dest, expected_sha256=expected)
    except CadImportError as primary_exc:
        urls = []
        try:
            urls = discover_step_urls(page_url)
        except Exception:
            urls = []
        last = primary_exc
        for candidate in urls:
            try:
                return download_step(candidate, dest, expected_sha256=expected)
            except CadImportError as exc:
                last = exc
        raise CadImportError(
            f"official field STEP missing from {primary} (and archive page fallback). {last}"
        ) from last


def write_assets_from_field(
    field: dict[str, Any],
    dest_dir: Path | None = None,
    year_hint: str = "2026",
) -> dict[str, str]:
    """Schematic AABB tessellation for seasons that do not require CAD collision."""
    dest_dir = dest_dir or asset_dir_for(field, year_hint)
    dest_dir.mkdir(parents=True, exist_ok=True)
    glb_path = dest_dir / "field.glb"
    xml_path = dest_dir / "field_mjcf.xml"
    write_glb(glb_path, solids_from_field(field))
    xml_path.write_text(build_mjcf(field), encoding="utf-8")
    rel_glb = str(glb_path.relative_to(ASSETS_DIR)).replace("\\", "/")
    rel_xml = str(xml_path.relative_to(ASSETS_DIR)).replace("\\", "/")
    return {"glb": str(glb_path), "mjcf": str(xml_path), "backgroundAsset": rel_glb, "collisionAsset": rel_xml}


def convert_step_to_trimesh(
    step_path: Path,
    *,
    units: str | None = None,
    tol_linear: float | None = None,
    raw_glb: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load + align a field STEP, returning the visual mesh (concatenated) and metadata."""
    from talongym.assets.cad_common import _require_trimesh

    trimesh = _require_trimesh()
    loaded, tess_meta = load_tessellated_step(
        step_path,
        units=units,
        tol_linear=tol_linear,
        raw_glb=raw_glb,
        merge_primitives=True,
        piece=False,
    )
    aligned, meta = align_field_mesh(loaded, units=tess_meta.get("sniffedUnits"))
    visual, decimated = decimate_visual(aligned, trimesh)
    meta.update(tess_meta)
    meta["decimated"] = decimated
    meta["faceCount"] = int(meta.get("faceCount") or 0)
    return visual, meta


def _write_field_derived(
    field: dict[str, Any],
    aligned: Any,
    *,
    dest_dir: Path,
    sha256: str,
    source_url: str,
    filename: str,
    tess_meta: dict[str, Any],
    align_meta: dict[str, Any],
) -> dict[str, Any]:
    from talongym.assets.cad_common import _require_trimesh

    trimesh = _require_trimesh()
    dest_dir.mkdir(parents=True, exist_ok=True)
    glb_path = dest_dir / "field.glb"
    collision_dir = dest_dir / "collision"
    initial_parts = iter_assembly_parts(aligned, trimesh)
    tile_surface_y = field_tile_surface_y(initial_parts)
    base_translation = list(align_meta.get("translationIn") or [0.0, 0.0, 0.0])
    if abs(tile_surface_y) > 1e-6:
        aligned.apply_translation([0.0, -tile_surface_y, 0.0])
        align_meta["translationIn"] = round_xyz(
            [base_translation[0], float(base_translation[1]) - tile_surface_y, base_translation[2]]
        )
        align_meta.update(bounds_dict(aligned))
    parts = iter_assembly_parts(aligned, trimesh)
    visual_parts, visual_filter = select_field_visual_parts(parts)
    static_visual_parts, moving_visual_parts = split_moving_hive_parts(visual_parts)
    collision_parts = write_convex_collision_parts(
        parts,
        collision_dir,
        rel_prefix=str((dest_dir.relative_to(ASSETS_DIR) / "collision")).replace("\\", "/"),
        allow_single_concave=False,
    )
    _visual_path, visual_meta = export_visual_parts_glb(static_visual_parts, glb_path, trimesh)
    mechanism_dir = dest_dir / "mechanisms"
    hive_pivots = {
        "red": [-12.75, 42.03, 1.11],
        "blue": [12.75, 42.03, -1.11],
    }
    mechanisms: list[dict[str, Any]] = []
    for alliance, pivot in hive_pivots.items():
        centered_parts: list[tuple[str, Any]] = []
        for name, source_mesh in moving_visual_parts[alliance]:
            mesh = source_mesh.copy()
            mesh.apply_translation([-pivot[0], -pivot[1], -pivot[2]])
            centered_parts.append((name, mesh))
        mechanism_path = mechanism_dir / f"{alliance}_hive.glb"
        _path, mechanism_visual_meta = export_visual_parts_glb(
            centered_parts,
            mechanism_path,
            trimesh,
            face_target=400_000,
        )
        mechanism_collision = [
            part
            for part in collision_parts
            if moving_hive_alliance(
                str(part.get("id") or ""),
                0.5 * (float(part["minIn"][0]) + float(part["maxIn"][0])),
            )
            == alliance
        ]
        mechanisms.append(
            {
                "id": f"{alliance}_hive",
                "alliance": alliance,
                "visualAsset": asset_relpath(mechanism_path),
                "visualFaceCount": int(mechanism_visual_meta.get("faceCount") or 0),
                "pivotIn": pivot,
                "axis": [1.0, 0.0, 0.0],
                "tipAngleDeg": 60.0 if alliance == "red" else -60.0,
                "collisionParts": mechanism_collision,
            }
        )
    field_block = {
        **source_block(
            source_url=source_url,
            filename=filename,
            sha256=sha256,
            units=str(align_meta.get("unitsGuess") or tess_meta.get("sniffedUnits") or "unknown"),
        ),
        "transform": transform_block(align_meta),
        "boundsIn": bounds_block(align_meta),
        "visualAsset": asset_relpath(glb_path),
        "visualFaceCount": int(visual_meta.get("faceCount") or 0),
        "collisionParts": collision_parts,
        "mechanisms": mechanisms,
        "partCount": len(collision_parts),
        "tolLinear": tess_meta.get("tolLinear"),
        "tessellation": {
            "tolLinear": tess_meta.get("tolLinear"),
            "mergePrimitives": tess_meta.get("mergePrimitives", True),
            "visualFaceTarget": 800_000,
            "tileSurfaceSourceYIn": round(float(tile_surface_y), 3),
            "visualFilter": visual_filter,
        },
    }
    field_block["transform"]["translationIn"][1] = round(float(base_translation[1]) - tile_surface_y, 3)
    return {
        "glb": str(glb_path),
        "backgroundAsset": asset_relpath(glb_path),
        "collisionDir": str(collision_dir),
        "fieldBlock": field_block,
        "partCount": len(collision_parts),
    }


def import_field_cad(
    field_path: Path | None = None,
    page_url: str = DEFAULT_CAD_PAGE,
    year_hint: str = "2026",
    step_path: Path | None = None,
    tol_linear: float | None = None,
    step_url: str | None = None,
    include_pieces: bool = True,
    skip_field: bool = False,
) -> dict[str, Any]:
    field = load_field_json(field_path)
    dest_dir = asset_dir_for(field, year_hint)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cad_needed = mesh_required(field)
    existing_manifest: dict[str, Any] = {}
    existing_path = dest_dir / "cad_manifest.json"
    if existing_path.is_file():
        try:
            existing_manifest = json.loads(existing_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}
    field_block: dict[str, Any] | None = existing_manifest.get("field") if skip_field else None
    paths: dict[str, Any] = {}
    source = "official_step"

    if not skip_field:
        staged: Path | None = None
        digest = ""
        filename = ""
        source_url = step_url or OFFICIAL_FIELD_STEP_URL
        if step_path is not None:
            staged = stage_step(Path(step_path))
            digest = sha256_file(staged)
            expected = expected_sha256(FIELD_SOURCE)
            if expected and digest != expected:
                raise CadImportError(
                    f"field STEP hash mismatch: got {digest}, expected {expected}"
                )
            filename = staged.name
            source = f"official_step:{staged.name}"
            source_url = f"file:{staged.name}"
        else:
            try:
                staged, digest, filename = fetch_official_field_step(url=step_url, page_url=page_url)
                source = f"official_step:{staged.name}"
            except CadImportError:
                if cad_needed:
                    raise
                paths = write_assets_from_field(field, dest_dir=dest_dir, year_hint=year_hint)
                source = "aabb_tessellation"
        if staged is not None:
            loaded, tess_meta = load_tessellated_step(
                staged,
                tol_linear=tol_linear,
                raw_glb=cad_cache_dir() / f"{staged.stem}.raw.glb",
                merge_primitives=True,
                piece=False,
            )
            aligned, align_meta = align_field_mesh(loaded, units=tess_meta.get("sniffedUnits"))
            derived = _write_field_derived(
                field,
                aligned,
                dest_dir=dest_dir,
                sha256=digest,
                source_url=source_url if not str(source_url).startswith("file:") else OFFICIAL_FIELD_STEP_URL,
                filename=filename or staged.name,
                tess_meta=tess_meta,
                align_meta=align_meta,
            )
            field_block = derived.pop("fieldBlock")
            paths.update(derived)
            paths["step"] = str(staged)
            paths["sha256"] = digest
            paths.update(align_meta)

    pieces: dict[str, Any] = dict(existing_manifest.get("pieces") or {}) if not include_pieces else {}
    if include_pieces:
        from talongym.assets.import_piece_cad import import_season_pieces

        pieces = import_season_pieces(dest_dir, tol_linear=tol_linear)

    notes = (
        "Physical geometry from official STEP. Scoring volumes/zones/triggers stay in the "
        "field preset. Collision parts are convex hulls of CAD solids (not one concave mesh). "
        "Runtime MuJoCo assembles those convex parts plus typed free-joint piece hulls."
    )
    manifest = build_manifest(
        season_slug=_season_slug(field),
        field=field_block,
        pieces=pieces or None,
        notes=notes,
    )
    manifest_path = write_manifest(dest_dir, manifest)
    result: dict[str, Any] = {
        "ok": True,
        "source": source,
        "cadManifest": manifest_relpath(dest_dir),
        "manifestPath": str(manifest_path),
        "generatorVersion": CAD_GENERATOR_VERSION,
        "pieces": {k: {"visualAsset": v.get("visualAsset"), "collisionAsset": v.get("collisionAsset")} for k, v in pieces.items()},
        **paths,
    }
    if not field_block and cad_needed and not skip_field:
        raise CadImportError(
            "mesh_field_collision season requires official field STEP; refusing AABB fallback"
        )
    return result


def verify_season_cad(
    field_path: Path | None = None,
    year_hint: str = "2026",
) -> dict[str, Any]:
    field = load_field_json(field_path)
    rel = field.get("cadManifest")
    if not rel:
        if mesh_required(field):
            raise CadImportError("mesh_field_collision season is missing cadManifest")
        return {"ok": True, "problems": ["no cadManifest"]}
    path = ASSETS_DIR / Path(str(rel))
    if not path.is_file():
        raise CadImportError(f"cadManifest not found: {rel}")
    document = load_manifest(path)
    problems = verify_manifest_files(document, require_field=mesh_required(field))
    schema_problems = validate_manifest(document)
    problems.extend(schema_problems)
    if problems and mesh_required(field):
        raise CadImportError("CAD assets missing or stale: " + "; ".join(problems[:12]))
    return {"ok": not problems, "problems": problems, "cadManifest": rel}


def main() -> None:
    result = import_field_cad()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
