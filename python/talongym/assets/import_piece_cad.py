"""Import official POLLEN / NECTAR STEP files into reusable GLB + convex hulls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from talongym.assets.cad_common import (
    CadImportError,
    _require_trimesh,
    align_piece_mesh,
    as_trimesh,
    cad_cache_dir,
    check_piece_diameter,
    convex_hull_mesh,
    download_step,
    export_visual_glb,
    load_tessellated_step,
    sha256_file,
    stage_step,
)
from talongym.assets.cad_manifest import asset_relpath, bounds_block, source_block, transform_block
from talongym.assets.cad_sources import PIECE_SOURCES, expected_sha256, piece_source


def piece_asset_dir(dest_dir: Path) -> Path:
    return dest_dir / "pieces"


def _piece_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "sourceUrl",
        "sourceFilename",
        "sha256",
        "units",
        "transform",
        "boundsIn",
        "visualAsset",
        "collisionAsset",
        "convex",
        "specDiameterIn",
        "measuredDiameterIn",
        "faceCount",
    )
    return {k: record[k] for k in keep if k in record}


def fetch_piece_step(
    type_id: str,
    *,
    step_path: Path | None = None,
    url: str | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    spec = piece_source(type_id)
    expected = expected_sha256(spec)
    if step_path is not None:
        staged = stage_step(Path(step_path), dest_name=str(spec["filename"]))
        digest = sha256_file(staged)
        if expected and digest != expected:
            raise CadImportError(
                f"{type_id} STEP hash mismatch: got {digest}, expected {expected}"
            )
        return staged, digest, spec
    dest = cad_cache_dir() / str(spec["filename"])
    path, digest, _filename = download_step(
        url or str(spec["sourceUrl"]),
        dest,
        expected_sha256=expected,
    )
    return path, digest, spec


def import_piece_cad(
    type_id: str,
    dest_dir: Path,
    *,
    step_path: Path | None = None,
    url: str | None = None,
    tol_linear: float | None = None,
) -> dict[str, Any]:
    trimesh = _require_trimesh()
    staged, digest, spec = fetch_piece_step(type_id, step_path=step_path, url=url)
    loaded, tess_meta = load_tessellated_step(
        staged,
        tol_linear=tol_linear,
        raw_glb=cad_cache_dir() / f"{staged.stem}.raw.glb",
        merge_primitives=True,
        piece=True,
    )
    aligned, meta = align_piece_mesh(
        loaded,
        units=tess_meta.get("sniffedUnits"),
        spec_diameter_in=float(spec["specDiameterIn"]),
        assume_z_up=True,
    )
    check_piece_diameter(float(meta["measuredDiameterIn"]), float(spec["specDiameterIn"]))
    visual_mesh = as_trimesh(aligned, trimesh)
    pieces_dir = piece_asset_dir(dest_dir)
    pieces_dir.mkdir(parents=True, exist_ok=True)
    glb_path = pieces_dir / f"{type_id}.glb"
    hull_path = pieces_dir / f"{type_id}_hull.stl"
    _, visual_meta = export_visual_glb(visual_mesh, glb_path, trimesh, face_target=80_000)
    hull_source = visual_mesh
    if visual_meta.get("decimated"):
        loaded_visual = trimesh.load(str(glb_path), force=None)
        hull_source = as_trimesh(loaded_visual, trimesh)
    hull = convex_hull_mesh(hull_source)
    hull.export(str(hull_path), file_type="stl")
    if not hull_path.is_file() or hull_path.stat().st_size < 64:
        raise CadImportError(f"failed to write piece hull {hull_path}")
    record = {
        **source_block(
            source_url=str(spec["sourceUrl"]),
            filename=staged.name,
            sha256=digest,
            units=str(meta.get("unitsGuess") or tess_meta.get("sniffedUnits") or "unknown"),
        ),
        "transform": transform_block(meta),
        "boundsIn": bounds_block(meta),
        "visualAsset": asset_relpath(glb_path),
        "collisionAsset": asset_relpath(hull_path),
        "convex": True,
        "specDiameterIn": float(spec["specDiameterIn"]),
        "measuredDiameterIn": float(meta["measuredDiameterIn"]),
        "faceCount": int(visual_meta.get("faceCount") or meta.get("faceCount") or 0),
        "tolLinear": tess_meta.get("tolLinear"),
        "step": str(staged),
    }
    return {**record, "manifest": _piece_manifest_record(record)}


def import_season_pieces(
    dest_dir: Path,
    *,
    types: list[str] | None = None,
    tol_linear: float | None = None,
) -> dict[str, Any]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for type_id in types or list(PIECE_SOURCES):
        full = import_piece_cad(type_id, dest_dir, tol_linear=tol_linear)
        out[type_id] = full.get("manifest") or _piece_manifest_record(full)
    return out


def apply_piece_refs(field: dict[str, Any], pieces: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of field with per-type visual/collision asset paths filled in."""
    updated = dict(field)
    game = []
    for piece in field.get("gamePieces") or []:
        row = dict(piece)
        rec = pieces.get(str(row.get("typeId")))
        if rec:
            row["visualAsset"] = rec["visualAsset"]
            row["collisionAsset"] = rec["collisionAsset"]
        game.append(row)
    updated["gamePieces"] = game
    return updated
