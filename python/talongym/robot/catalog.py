"""Load and search curated goBILDA/REV robot-part catalogs."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from talongym.paths import PRESETS_DIR, SCHEMAS_DIR

CATALOG_SCHEMA_NAME = "robot-part-catalog.schema.json"
CATALOG_SCHEMA_VERSION = "1.0.0"
CATALOG_DIR = PRESETS_DIR / "robot_parts"
MANUFACTURERS = ("gobilda", "rev")
SKU_RE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$"

ALLOWED_CAD_HOSTS = frozenset(
    {
        "www.gobilda.com",
        "gobilda.com",
        "www.revrobotics.com",
        "revrobotics.com",
    }
)

MM_TO_IN = 1.0 / 25.4
DEFAULT_VISUAL_TRANSFORM = {
    "scaleToInches": MM_TO_IN,
    "rotatedZupToYup": True,
    "translationIn": [0.0, 0.0, 0.0],
}


class CatalogError(RuntimeError):
    pass


def catalog_schema() -> dict[str, Any]:
    return json.loads((SCHEMAS_DIR / CATALOG_SCHEMA_NAME).read_text(encoding="utf-8"))


def validate_catalog_document(document: dict[str, Any]) -> list[str]:
    errors = sorted(Draft202012Validator(catalog_schema()).iter_errors(document), key=lambda e: list(e.path))
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


def catalog_path(manufacturer: str) -> Path:
    if manufacturer not in MANUFACTURERS:
        raise CatalogError(f"unknown manufacturer {manufacturer}")
    return CATALOG_DIR / f"{manufacturer}.json"


def robot_parts_root() -> Path:
    from talongym import paths

    return Path(paths.VAR_DIR) / "assets" / "robot_parts"


def sanitize_sku(sku: str) -> str:
    text = str(sku).strip()
    if not text:
        raise CatalogError("empty sku")
    return text.replace("\\", "/").split("/")[-1]


def catalog_part_dir(manufacturer: str, sku: str) -> Path:
    if manufacturer not in MANUFACTURERS:
        raise CatalogError(f"unknown manufacturer {manufacturer}")
    safe = sanitize_sku(sku).lower()
    dest = (robot_parts_root() / manufacturer / safe).resolve()
    try:
        dest.relative_to(robot_parts_root().resolve())
    except ValueError as exc:
        raise CatalogError(f"bad catalog sku {sku}") from exc
    return dest


@lru_cache(maxsize=1)
def load_catalogs() -> dict[str, dict[str, Any]]:
    catalogs: dict[str, dict[str, Any]] = {}
    for manufacturer in MANUFACTURERS:
        path = catalog_path(manufacturer)
        document = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_catalog_document(document)
        if errors:
            raise CatalogError(f"{path.name} failed schema: " + "; ".join(errors[:8]))
        catalogs[manufacturer] = document
    return catalogs


def reload_catalogs() -> dict[str, dict[str, Any]]:
    load_catalogs.cache_clear()
    return load_catalogs()


def iter_parts() -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for manufacturer, document in load_catalogs().items():
        for part in document.get("parts") or []:
            if not isinstance(part, dict):
                continue
            row = dict(part)
            row["manufacturer"] = manufacturer
            parts.append(row)
    return parts


def get_part(sku: str) -> dict[str, Any]:
    wanted = sanitize_sku(sku).lower()
    matches = [part for part in iter_parts() if str(part.get("sku") or "").lower() == wanted]
    if not matches:
        raise CatalogError(f"unknown catalog sku {sku}")
    if len(matches) > 1:
        raise CatalogError(f"duplicate catalog sku {sku}")
    return matches[0]


def search_parts(
    *,
    query: str | None = None,
    manufacturer: str | None = None,
    tag: str | None = None,
) -> list[dict[str, Any]]:
    needle = (query or "").strip().lower()
    maker = (manufacturer or "").strip().lower() or None
    wanted_tag = (tag or "").strip().lower() or None
    if maker and maker not in MANUFACTURERS:
        raise CatalogError(f"unknown manufacturer {maker}")
    out: list[dict[str, Any]] = []
    for part in iter_parts():
        if maker and part.get("manufacturer") != maker:
            continue
        tags = [str(t).lower() for t in (part.get("tags") or [])]
        if wanted_tag and wanted_tag not in tags:
            continue
        if needle:
            hay = " ".join(
                [
                    str(part.get("sku") or ""),
                    str(part.get("displayName") or ""),
                    str(part.get("manufacturer") or ""),
                    " ".join(tags),
                ]
            ).lower()
            if needle not in hay:
                continue
        out.append(part)
    return out


@lru_cache(maxsize=1)
def cad_extra_available() -> bool:
    if find_spec("trimesh") is None or find_spec("cascadio") is None:
        return False
    try:
        import cascadio  # noqa: F401
    except ImportError:
        return False
    return True


def part_summary(part: dict[str, Any], *, cache: dict[str, Any] | None = None) -> dict[str, Any]:
    cad = part.get("cad") if isinstance(part.get("cad"), dict) else {}
    sku = str(part.get("sku") or "")
    manufacturer = str(part.get("manufacturer") or "")
    status = dict((cache or {}).get(sku.lower()) or cache_entry(manufacturer, sku))
    download_enabled = bool(cad.get("downloadEnabled"))
    if not download_enabled and status.get("state") in {None, "missing"}:
        status["state"] = "unavailable"
        status["reason"] = "download_disabled"
        status["visualAsset"] = None
        status["collisionAsset"] = None
        status["thumbnailAsset"] = None
    elif not download_enabled:
        status.setdefault("reason", "download_disabled")
    elif not cad_extra_available() and status.get("state") == "missing":
        status["state"] = "unavailable"
        status["reason"] = "cad_extra_required"
        status["visualAsset"] = None
        status["collisionAsset"] = None
    return {
        "sku": sku,
        "manufacturer": manufacturer,
        "displayName": part.get("displayName"),
        "productUrl": part.get("productUrl"),
        "tags": list(part.get("tags") or []),
        "massKg": part.get("massKg"),
        "mounts": list(part.get("mounts") or []),
        "downloadEnabled": download_enabled,
        "cadFormat": cad.get("format"),
        "preview": part_preview(part, status),
        "cache": status,
    }


def part_preview(part: dict[str, Any], status: dict[str, Any] | None = None) -> dict[str, Any]:
    cache = status or {}
    visual = cache.get("visualAsset")
    ready = cache.get("state") == "ready" and bool(visual)
    collision = next((row for row in (part.get("collision") or []) if isinstance(row, dict)), {}) or {}
    kind = str(collision.get("kind") or "box")
    preview: dict[str, Any] = {
        "source": "cad" if ready else "proxy",
        "kind": kind if kind in {"box", "sphere", "cylinder", "capsule", "convex_hull"} else "box",
        "tags": list(part.get("tags") or []),
        "visualAsset": visual if ready else None,
        "thumbnailAsset": cache.get("thumbnailAsset") if ready else None,
    }
    if collision.get("sizeIn"):
        preview["sizeIn"] = [float(v) for v in collision["sizeIn"][:3]]
    if collision.get("radiusIn") is not None:
        preview["radiusIn"] = float(collision["radiusIn"])
    if collision.get("lengthIn") is not None:
        preview["lengthIn"] = float(collision["lengthIn"])
    return preview


def cache_meta_path(manufacturer: str, sku: str) -> Path:
    return catalog_part_dir(manufacturer, sku) / "cache.json"


def cad_conversion_fingerprint(part: dict[str, Any]) -> str:
    inputs = {key: part.get(key) for key in ("cad", "visualTransform", "collision")}
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_entry(manufacturer: str, sku: str) -> dict[str, Any]:
    path = cache_meta_path(manufacturer, sku)
    if not path.is_file():
        return {"state": "missing", "visualAsset": None, "collisionAsset": None}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"state": "invalid", "visualAsset": None, "collisionAsset": None}
    visual = document.get("visualAsset")
    collision = document.get("collisionAsset")
    thumbnail = document.get("thumbnailAsset")
    visual_ok = bool(visual) and (robot_parts_root().parent / str(visual)).is_file()
    collision_ok = bool(collision) and (robot_parts_root().parent / str(collision)).is_file()
    thumbnail_ok = bool(thumbnail) and (robot_parts_root().parent / str(thumbnail)).is_file()
    part = get_part(sku)
    from talongym.assets.cad_common import CATALOG_CAD_GENERATOR_VERSION

    obsolete = (
        document.get("generatorVersion") != CATALOG_CAD_GENERATOR_VERSION
        or document.get("conversionFingerprint") != cad_conversion_fingerprint(part)
    )
    if document.get("stale") or obsolete:
        state = "stale"
    elif visual_ok and collision_ok and thumbnail_ok:
        state = "ready"
    else:
        state = "incomplete"
    usable = state == "ready"
    return {
        "state": state,
        "visualAsset": visual if usable and visual_ok else None,
        "collisionAsset": collision if usable and collision_ok else None,
        "thumbnailAsset": thumbnail if usable and thumbnail_ok else None,
        "sha256": document.get("sha256"),
        "generatorVersion": document.get("generatorVersion"),
    }


def cache_status() -> dict[str, Any]:
    parts = iter_parts()
    entries = [part_summary(part) for part in parts]
    counts = {"ready": 0, "stale": 0, "incomplete": 0, "missing": 0, "invalid": 0, "unavailable": 0}
    for row in entries:
        state = str((row.get("cache") or {}).get("state") or "missing")
        counts[state] = counts.get(state, 0) + 1
    extra = cad_extra_available()
    return {
        "root": str(robot_parts_root()),
        "partCount": len(entries),
        "counts": counts,
        "cadExtra": extra,
        "cadUnavailableReason": None if extra else "CAD extra required; pip install -e '.[cad]'",
        "parts": entries,
    }
