"""Allowlisted manufacturer CAD download, conversion, and cache for catalog parts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from talongym.assets.cad_common import CATALOG_CAD_GENERATOR_VERSION, USER_AGENT
from talongym.assets.import_robot_cad import (
    RobotCadError,
    import_robot_cad,
    is_cad_extra_error,
)
from talongym.robot.catalog import (
    ALLOWED_CAD_HOSTS,
    CatalogError,
    cache_entry,
    cache_meta_path,
    cad_extra_available,
    catalog_part_dir,
    get_part,
    robot_parts_root,
)

MAX_CATALOG_DOWNLOAD_BYTES = 64 * 1024 * 1024
STEP_MAGIC = b"ISO-10303-21"
ZIP_MAGIC = b"PK\x03\x04"

_jobs: dict[str, dict[str, Any]] = {}
_job_lock = threading.Lock()
_sku_jobs: dict[str, str] = {}
_convert_locks: dict[str, threading.Lock] = {}
_convert_locks_guard = threading.Lock()


def assert_allowlisted_url(url: str, expected: str | None = None) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise CatalogError("CAD URL must be https without credentials")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_CAD_HOSTS:
        raise CatalogError(f"CAD host {host or '<none>'} is not allowlisted")
    if ".." in Path(parsed.path).parts:
        raise CatalogError("CAD URL path is invalid")
    if expected is not None and url != expected:
        raise CatalogError("CAD URL is not the catalog allowlist entry")


def _part_lock(sku: str) -> threading.Lock:
    key = sku.lower()
    with _convert_locks_guard:
        lock = _convert_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _convert_locks[key] = lock
        return lock


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_catalog_thumbnail(visual_path: Path, destination: Path, *, size: int = 320) -> Path:
    """Render a deterministic isometric SVG from the cached official mesh."""
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise RobotCadError("trimesh missing; pip install -e '.[cad]'") from exc
    loaded = trimesh.load(visual_path, force="scene")
    geometries = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    meshes = [mesh for mesh in geometries if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces)]
    if not meshes:
        raise RobotCadError(f"cached visual has no renderable triangles: {visual_path}")
    mesh = trimesh.util.concatenate(meshes)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if len(faces) > 3200:
        faces = faces[np.linspace(0, len(faces) - 1, 3200, dtype=int)]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    vertices = vertices - center
    yaw = math.radians(38)
    pitch = math.radians(24)
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    rotated = vertices @ (ry @ rx).T
    span = max(float(np.ptp(rotated[:, 0])), float(np.ptp(rotated[:, 1])), 1e-6)
    scale = size * 0.76 / span
    points = np.column_stack(
        (
            size * 0.5 + rotated[:, 0] * scale,
            size * 0.53 - rotated[:, 1] * scale,
            rotated[:, 2],
        )
    )
    triangles: list[tuple[float, str]] = []
    light = np.array([0.35, 0.75, 0.56])
    light /= np.linalg.norm(light)
    for face in faces:
        tri3 = rotated[face]
        normal = np.cross(tri3[1] - tri3[0], tri3[2] - tri3[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            continue
        brightness = 0.42 + 0.48 * abs(float(np.dot(normal / norm, light)))
        base = (192, 177, 151)
        color = tuple(max(0, min(255, round(channel * brightness))) for channel in base)
        tri = points[face]
        coords = " ".join(f"{point[0]:.1f},{point[1]:.1f}" for point in tri)
        triangles.append((float(tri[:, 2].mean()), f'<polygon points="{coords}" fill="rgb{color}" stroke="#30282b" stroke-width=".35"/>'))
    triangles.sort(key=lambda row: row[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
                '<rect width="100%" height="100%" rx="18" fill="#241f21"/>',
                *(row[1] for row in triangles),
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )
    return destination


def invalidate_catalog_cache(manufacturer: str, sku: str) -> None:
    dest = catalog_part_dir(manufacturer, sku)
    if dest.is_dir():
        shutil.rmtree(dest)


def cache_is_fresh(part: dict[str, Any]) -> bool:
    manufacturer = str(part.get("manufacturer") or "")
    sku = str(part.get("sku") or "")
    meta_path = cache_meta_path(manufacturer, sku)
    if not meta_path.is_file():
        return False
    try:
        document = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if document.get("stale") or document.get("generatorVersion") != CATALOG_CAD_GENERATOR_VERSION:
        return False
    expected = (part.get("cad") or {}).get("sha256")
    if expected and document.get("sha256") != str(expected).lower():
        return False
    status = cache_entry(manufacturer, sku)
    return status.get("state") == "ready" and bool(status.get("visualAsset") and status.get("thumbnailAsset"))


def mark_cache_stale(part: dict[str, Any], reason: str) -> None:
    manufacturer = str(part["manufacturer"])
    sku = str(part["sku"])
    path = cache_meta_path(manufacturer, sku)
    document: dict[str, Any] = {}
    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            document = {}
    document.update({"stale": True, "staleReason": reason})
    _write_json(path, document)


def download_catalog_cad(url: str, dest: Path, *, expected: str, timeout: float = 120.0) -> tuple[Path, str]:
    import httpx

    assert_allowlisted_url(url, expected)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    hasher = hashlib.sha256()
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                final_host = (urlparse(str(resp.url)).hostname or "").lower()
                if final_host not in ALLOWED_CAD_HOSTS:
                    raise CatalogError(f"CAD redirect host {final_host or '<none>'} is not allowlisted")
                written = 0
                first = True
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        if first:
                            first = False
                            stripped = chunk.lstrip()
                            if stripped.lower().startswith(b"<!doctype") or stripped.lower().startswith(b"<html"):
                                raise CatalogError(f"URL returned HTML, not CAD: {url}")
                        written += len(chunk)
                        if written > MAX_CATALOG_DOWNLOAD_BYTES:
                            raise CatalogError(f"CAD download exceeds {MAX_CATALOG_DOWNLOAD_BYTES // (1024 * 1024)} MB")
                        hasher.update(chunk)
                        fh.write(chunk)
    except CatalogError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise CatalogError(f"CAD download failed ({url}): {exc}") from exc
    if tmp.stat().st_size < 64:
        tmp.unlink(missing_ok=True)
        raise CatalogError(f"CAD download too small: {url}")
    head = tmp.read_bytes()[:32]
    if dest.suffix.lower() == ".zip" and not head.startswith(ZIP_MAGIC):
        tmp.unlink(missing_ok=True)
        raise CatalogError(f"downloaded file is not a ZIP: {url}")
    if dest.suffix.lower() in {".step", ".stp"}:
        prefix = tmp.read_bytes()[:256].lstrip()
        if not prefix.startswith(STEP_MAGIC) and b"ISO-10303" not in prefix:
            tmp.unlink(missing_ok=True)
            raise CatalogError(f"downloaded file is not a STEP (ISO-10303-21): {url}")
    if dest.exists():
        dest.unlink()
    tmp.rename(dest)
    return dest, hasher.hexdigest()


def convert_catalog_part(sku: str, *, force: bool = False, source: Path | None = None) -> dict[str, Any]:
    part = get_part(sku)
    cad = part.get("cad") if isinstance(part.get("cad"), dict) else {}
    manufacturer = str(part["manufacturer"])
    part_sku = str(part["sku"])
    dest = catalog_part_dir(manufacturer, part_sku)
    with _part_lock(part_sku):
        if not force and cache_is_fresh(part):
            status = cache_entry(manufacturer, part_sku)
            if not status.get("thumbnailAsset") and status.get("visualAsset"):
                visual_path = robot_parts_root().parent / str(status["visualAsset"])
                thumbnail_path = generate_catalog_thumbnail(visual_path, dest / "thumbnail.svg")
                document = json.loads(cache_meta_path(manufacturer, part_sku).read_text(encoding="utf-8"))
                document["thumbnailAsset"] = str(thumbnail_path.relative_to(robot_parts_root().parent)).replace("\\", "/")
                _write_json(cache_meta_path(manufacturer, part_sku), document)
                status = cache_entry(manufacturer, part_sku)
            return {"sku": part_sku, "manufacturer": manufacturer, "cache": status, "reused": True}
        if not cad.get("downloadEnabled") and source is None:
            raise CatalogError(f"download disabled for {part_sku}; obtain CAD from the manufacturer product page")
        url = str(cad.get("sourceUrl") or "")
        filename = str(cad.get("filename") or f"{part_sku}.step")
        dest.mkdir(parents=True, exist_ok=True)
        source_path = source
        digest: str | None = None
        if source_path is None:
            source_path = dest / filename
            _downloaded, digest = download_catalog_cad(url, source_path, expected=url)
        else:
            source_path = Path(source_path)
            if not source_path.is_file():
                raise CatalogError(f"local CAD not found: {source_path}")
        expected = cad.get("sha256")
        try:
            imported = import_robot_cad(
                source_path,
                "catalog",
                dest_root=dest,
                preserve_origin=True,
                visual_transform=part.get("visualTransform"),
                collision=part.get("collision") if isinstance(part.get("collision"), list) else None,
                expected_sha256=str(expected) if expected else None,
                preferred_name=str(cad.get("memberFilename") or part_sku),
                max_bytes=MAX_CATALOG_DOWNLOAD_BYTES,
            )
        except RobotCadError as exc:
            mark_cache_stale(part, str(exc))
            raise
        digest = imported.get("sha256") or digest
        visual_path = robot_parts_root().parent / str(imported["visualAsset"])
        thumbnail_path = generate_catalog_thumbnail(visual_path, dest / "thumbnail.svg")
        thumbnail_asset = str(thumbnail_path.relative_to(robot_parts_root().parent)).replace("\\", "/")
        meta = {
            "sku": part_sku,
            "manufacturer": manufacturer,
            "generatorVersion": CATALOG_CAD_GENERATOR_VERSION,
            "sha256": digest,
            "sourceFilename": source_path.name,
            "visualAsset": imported["visualAsset"],
            "collisionAsset": imported["collisionAsset"],
            "thumbnailAsset": thumbnail_asset,
            "bbox": imported.get("bbox"),
            "unitsGuess": imported.get("unitsGuess"),
            "originPreserved": True,
            "stale": False,
        }
        _write_json(cache_meta_path(manufacturer, part_sku), meta)
        return {
            "sku": part_sku,
            "manufacturer": manufacturer,
            "cache": cache_entry(manufacturer, part_sku),
            "import": {k: imported[k] for k in ("visualAsset", "collisionAsset", "bbox", "unitsGuess", "faceCount") if k in imported},
            "reused": False,
        }


def get_job(job_id: str) -> dict[str, Any] | None:
    with _job_lock:
        row = _jobs.get(job_id)
        return dict(row) if row else None


def job_for_sku(sku: str) -> dict[str, Any] | None:
    with _job_lock:
        job_id = _sku_jobs.get(str(sku).lower())
        row = _jobs.get(job_id) if job_id else None
        return dict(row) if row else None


def register_sku_job(sku: str, *, state: str = "running", force: bool = False) -> tuple[str, bool]:
    job_id = uuid.uuid4().hex[:16]
    key = str(sku).lower()
    with _job_lock:
        existing_id = _sku_jobs.get(key)
        existing = _jobs.get(existing_id) if existing_id else None
        if existing and existing.get("state") in {"queued", "running"}:
            return str(existing_id), False
        _jobs[job_id] = {"id": job_id, "sku": sku, "state": state, "force": force}
        _sku_jobs[key] = job_id
    return job_id, True


def finish_sku_job(job_id: str, sku: str, **fields: Any) -> None:
    with _job_lock:
        row = {"id": job_id, "sku": sku, **fields}
        _jobs[job_id] = row
        if _sku_jobs.get(str(sku).lower()) == job_id:
            _sku_jobs[str(sku).lower()] = job_id


def start_catalog_convert(sku: str, *, force: bool = False) -> dict[str, Any]:
    part = get_part(sku)
    cad = part.get("cad") if isinstance(part.get("cad"), dict) else {}
    if not cad.get("downloadEnabled"):
        raise CatalogError(f"download disabled for {part['sku']}; obtain CAD from the manufacturer product page")
    if not cad_extra_available():
        raise CatalogError("CAD extra required; pip install -e '.[cad]' to convert manufacturer STEP files")
    if cache_is_fresh(part) and not force:
        return {"jobId": None, "state": "ready", "sku": part["sku"], "cache": cache_entry(str(part["manufacturer"]), str(part["sku"]))}
    existing = job_for_sku(str(part["sku"]))
    if existing and existing.get("state") in {"queued", "running"}:
        return {"jobId": existing["id"], "state": existing["state"], "sku": part["sku"]}
    job_id, created = register_sku_job(str(part["sku"]), state="queued", force=force)
    if not created:
        row = job_for_sku(str(part["sku"])) or existing or {"id": job_id, "state": "queued"}
        return {"jobId": row.get("id") or job_id, "state": row.get("state") or "queued", "sku": part["sku"]}
    threading.Thread(target=_convert_worker, args=(job_id, str(part["sku"]), force), daemon=True).start()
    return {"jobId": job_id, "state": "queued", "sku": part["sku"]}


def _convert_worker(job_id: str, sku: str, force: bool) -> None:
    finish_sku_job(job_id, sku, state="running")
    try:
        result = convert_catalog_part(sku, force=force)
        finish_sku_job(job_id, sku, state="done", result=result)
    except (CatalogError, RobotCadError) as exc:
        code = "CAD_EXTRA" if is_cad_extra_error(exc) else "CAD_IMPORT"
        finish_sku_job(job_id, sku, state="failed", error={"code": code, "message": str(exc)})


def reset_jobs_for_tests() -> None:
    with _job_lock:
        _jobs.clear()
        _sku_jobs.clear()
    from talongym.assets.catalog_cache_batch import reset_batches_for_tests

    reset_batches_for_tests()
    if robot_parts_root().is_dir():
        shutil.rmtree(robot_parts_root(), ignore_errors=True)
