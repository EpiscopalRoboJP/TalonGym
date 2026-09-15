"""Versioned CAD asset manifest: source URLs, hashes, units, transforms, bounds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from talongym.assets.cad_common import CAD_GENERATOR_VERSION, CadImportError, sha256_file
from talongym.paths import ASSETS_DIR, SCHEMAS_DIR


MANIFEST_SCHEMA_NAME = "cad-manifest.schema.json"
MANIFEST_SCHEMA_VERSION = "1.0.0"
COORDINATE_SYSTEM = "ftc_inches_yup"


def manifest_schema() -> dict[str, Any]:
    return json.loads((SCHEMAS_DIR / MANIFEST_SCHEMA_NAME).read_text(encoding="utf-8"))


def validate_manifest(document: dict[str, Any]) -> list[str]:
    errors = sorted(Draft202012Validator(manifest_schema()).iter_errors(document), key=lambda e: list(e.path))
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


def manifest_relpath(dest_dir: Path) -> str:
    rel = dest_dir.relative_to(ASSETS_DIR) / "cad_manifest.json"
    return str(rel).replace("\\", "/")


def asset_relpath(path: Path) -> str:
    return str(path.relative_to(ASSETS_DIR)).replace("\\", "/")


def transform_block(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "scaleToInches": float(meta.get("scaleToInches") or 1.0),
        "rotatedZupToYup": bool(meta.get("rotatedZupToYup")),
        "translationIn": [float(v) for v in (meta.get("translationIn") or [0.0, 0.0, 0.0])],
    }


def source_block(
    *,
    source_url: str,
    filename: str,
    sha256: str,
    units: str,
) -> dict[str, Any]:
    return {
        "sourceUrl": source_url,
        "sourceFilename": filename,
        "sha256": sha256,
        "units": units,
    }


def bounds_block(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "minIn": [float(v) for v in meta.get("minIn") or []],
        "maxIn": [float(v) for v in meta.get("maxIn") or []],
        "extentsIn": [float(v) for v in meta.get("extentsIn") or []],
    }


def build_manifest(
    *,
    season_slug: str,
    field: dict[str, Any] | None = None,
    pieces: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "generatorVersion": CAD_GENERATOR_VERSION,
        "seasonSlug": season_slug,
        "coordinateSystem": COORDINATE_SYSTEM,
    }
    if notes:
        doc["notes"] = notes
    if field is not None:
        doc["field"] = field
    if pieces:
        doc["pieces"] = pieces
    return doc


def write_manifest(dest_dir: Path, document: dict[str, Any]) -> Path:
    errors = validate_manifest(document)
    if errors:
        raise CadImportError("cad manifest failed schema: " + "; ".join(errors[:8]))
    path = dest_dir / "cad_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_manifest(document)
    if errors:
        raise CadImportError("cad manifest failed schema: " + "; ".join(errors[:8]))
    return document


def resolve_asset(rel: str) -> Path:
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise CadImportError(f"bad CAD asset path {rel}")
    dest = (ASSETS_DIR / rel_path).resolve()
    try:
        dest.relative_to(ASSETS_DIR.resolve())
    except ValueError as exc:
        raise CadImportError(f"bad CAD asset path {rel}") from exc
    return dest


def verify_manifest_files(document: dict[str, Any], *, require_field: bool = False) -> list[str]:
    """Return human-readable problems (missing files, stale generator, non-convex claims)."""
    problems: list[str] = []
    if document.get("generatorVersion") != CAD_GENERATOR_VERSION:
        problems.append(
            f"generatorVersion {document.get('generatorVersion')!r} != {CAD_GENERATOR_VERSION}"
        )
    field = document.get("field")
    if require_field and not field:
        problems.append("cad-required season is missing field assets in the manifest")
    if isinstance(field, dict):
        visual = field.get("visualAsset")
        if visual:
            path = resolve_asset(str(visual))
            if not path.is_file():
                problems.append(f"missing field visual {visual}")
        parts = field.get("collisionParts") or []
        if require_field and not parts:
            problems.append("field collisionParts is empty; MuJoCo cannot use a concave field mesh")
        for part in parts:
            if not isinstance(part, dict):
                continue
            rel = part.get("asset")
            if not rel:
                problems.append("collision part missing asset")
                continue
            path = resolve_asset(str(rel))
            if not path.is_file():
                problems.append(f"missing collision part {rel}")
            if part.get("convex") is not True:
                problems.append(f"collision part {rel} is not marked convex")
        for mechanism in field.get("mechanisms") or []:
            if not isinstance(mechanism, dict):
                continue
            mechanism_id = str(mechanism.get("id") or "<unknown>")
            visual = mechanism.get("visualAsset")
            if not visual or not resolve_asset(str(visual)).is_file():
                problems.append(f"missing field mechanism {mechanism_id} visual {visual}")
            if not mechanism.get("collisionParts"):
                problems.append(f"field mechanism {mechanism_id} has no collision parts")
    pieces = document.get("pieces") or {}
    if isinstance(pieces, dict):
        for type_id, spec in pieces.items():
            if not isinstance(spec, dict):
                continue
            for key in ("visualAsset", "collisionAsset"):
                rel = spec.get(key)
                if not rel:
                    problems.append(f"piece {type_id} missing {key}")
                    continue
                path = resolve_asset(str(rel))
                if not path.is_file():
                    problems.append(f"missing piece {type_id} {key} {rel}")
            if spec.get("convex") is not True:
                problems.append(f"piece {type_id} collision is not marked convex")
    return problems


def verify_source_hash(step_path: Path, expected: str | None) -> str:
    digest = sha256_file(step_path)
    if expected and digest != expected.lower():
        raise CadImportError(
            f"stale STEP {step_path.name}: sha256 {digest} != pinned {expected.lower()}"
        )
    return digest
