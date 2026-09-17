"""Allowlisted catalog search, cache status, and on-demand CAD conversion."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse

from talongym.assets.catalog_cache_batch import (
    cancel_batch,
    empty_batch_snapshot,
    get_batch,
    latest_batch,
    retry_batch,
    start_cache_all,
)
from talongym.assets.import_catalog_cad import (
    get_job,
    invalidate_catalog_cache,
    start_catalog_convert,
)
from talongym.assets.import_robot_cad import RobotCadError, is_cad_extra_error
from talongym.robot.assembly import compile_assembly_to_preset
from talongym.robot.catalog import CatalogError, cache_status, get_part, part_summary, search_parts
from talongym.robot.contract import AssemblyError
from talongym.robot.recipes import RecipeError, instantiate_recipe, list_public_recipes

router = APIRouter(prefix="/api/v1", tags=["catalog"])


def _http_catalog_error(exc: BaseException) -> HTTPException:
    msg = str(exc)
    if is_cad_extra_error(exc):
        return HTTPException(503, {"error": {"code": "CAD_EXTRA", "message": msg}})
    if isinstance(exc, CatalogError) and msg.startswith("unknown catalog sku"):
        return HTTPException(404, {"error": {"code": "NOT_FOUND", "message": msg}})
    if isinstance(exc, CatalogError) and "disabled" in msg:
        return HTTPException(409, {"error": {"code": "CAD_DISABLED", "message": msg}})
    if isinstance(exc, CatalogError) and ("allowlist" in msg or "https" in msg or "redirect host" in msg):
        return HTTPException(400, {"error": {"code": "CAD_URL", "message": msg}})
    if isinstance(exc, RecipeError):
        status = 404 if "unknown drivebase recipe" in msg else 422
        return HTTPException(status, {"error": {"code": "RECIPE", "message": msg}})
    if isinstance(exc, AssemblyError):
        return HTTPException(422, {"error": {"code": "ASSEMBLY", "message": msg}})
    status = 422
    code = "CAD_IMPORT"
    if isinstance(exc, CatalogError):
        code = "CATALOG"
        if "unknown manufacturer" in msg:
            status = 400
    return HTTPException(status, {"error": {"code": code, "message": msg}})


@router.get("/catalog")
def list_catalog(
    q: str | None = Query(default=None),
    manufacturer: str | None = Query(default=None),
    tag: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        parts = search_parts(query=q, manufacturer=manufacturer, tag=tag)
    except CatalogError as exc:
        raise _http_catalog_error(exc) from exc
    return {"parts": [part_summary(part) for part in parts], "count": len(parts)}


@router.get("/catalog/cache")
def catalog_cache() -> dict[str, Any]:
    return cache_status()


@router.get("/catalog/cache/all")
def catalog_cache_all_status() -> dict[str, Any]:
    return latest_batch() or empty_batch_snapshot()


@router.get("/catalog/cache/all/{batch_id}")
def catalog_cache_all_batch(batch_id: str) -> dict[str, Any]:
    row = get_batch(batch_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": batch_id}})
    return row


@router.post("/catalog/cache/all")
def catalog_cache_all(body: dict[str, Any] | None = None) -> Any:
    payload = body or {}
    skus = payload.get("skus")
    if skus is not None and not isinstance(skus, list):
        raise HTTPException(400, {"error": {"code": "CATALOG", "message": "skus must be a list"}})
    try:
        snapshot = start_cache_all(
            force=bool(payload.get("force")),
            manufacturer=payload.get("manufacturer"),
            skus=[str(sku) for sku in skus] if isinstance(skus, list) else None,
        )
    except CatalogError as exc:
        raise _http_catalog_error(exc) from exc
    status = 202 if snapshot.get("state") in {"queued", "running", "cancelling"} else 200
    return JSONResponse(snapshot, status_code=status)


@router.post("/catalog/cache/all/{batch_id}/cancel")
def catalog_cache_all_cancel(batch_id: str) -> dict[str, Any]:
    row = cancel_batch(batch_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": batch_id}})
    return row


@router.post("/catalog/cache/all/{batch_id}/retry")
def catalog_cache_all_retry(batch_id: str) -> Any:
    row = retry_batch(batch_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": batch_id}})
    status = 202 if row.get("state") in {"queued", "running", "cancelling"} else 200
    return JSONResponse(row, status_code=status)


@router.get("/catalog/jobs/{job_id}")
def catalog_job(job_id: str) -> dict[str, Any]:
    row = get_job(job_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": job_id}})
    return row


@router.get("/catalog/parts/{sku}")
def catalog_part(sku: str) -> dict[str, Any]:
    try:
        part = get_part(sku)
    except CatalogError as exc:
        raise _http_catalog_error(exc) from exc
    return {**part, **part_summary(part)}


@router.post("/catalog/parts/{sku}/download")
def download_catalog_part(sku: str, force: bool = Query(default=False)) -> Any:
    try:
        started = start_catalog_convert(sku, force=force)
    except (CatalogError, RobotCadError) as exc:
        raise _http_catalog_error(exc) from exc
    if started.get("jobId") is None:
        return JSONResponse({**started, "accepted": False}, status_code=200)
    return JSONResponse({**started, "accepted": True}, status_code=202)


@router.delete("/catalog/parts/{sku}/cache")
def delete_catalog_part_cache(sku: str) -> dict[str, Any]:
    try:
        part = get_part(sku)
        invalidate_catalog_cache(str(part["manufacturer"]), str(part["sku"]))
    except CatalogError as exc:
        raise _http_catalog_error(exc) from exc
    return {"ok": True, "sku": part["sku"]}


@router.get("/catalog/recipes")
def catalog_recipes() -> dict[str, Any]:
    return {"recipes": list_public_recipes()}


@router.post("/catalog/recipes/{recipe_id}/instantiate")
def catalog_instantiate_recipe(
    recipe_id: str,
    parameters: Annotated[dict[str, Any] | None, Body()] = None,
) -> dict[str, Any]:
    try:
        document = instantiate_recipe(recipe_id, parameters or {})
    except (RecipeError, CatalogError, AssemblyError) as exc:
        raise _http_catalog_error(exc) from exc
    return {
        "id": document["id"],
        "displayName": document.get("displayName"),
        "schemaVersion": document["schemaVersion"],
        "assembly": document["assembly"],
        "functionalBindings": document.get("functionalBindings"),
        "warnings": document.get("warnings") or [],
        "document": document,
    }


@router.post("/catalog/assemblies/compile")
def catalog_compile_assembly(document: dict[str, Any]) -> dict[str, Any]:
    try:
        compiled = compile_assembly_to_preset(document, competitive=True)
    except (AssemblyError, CatalogError, RecipeError) as exc:
        raise _http_catalog_error(exc) from exc
    return {
        "preset": compiled.preset,
        "instancePoses": compiled.instance_poses,
        "report": {
            "confirmed": compiled.report.confirmed,
            "roles": compiled.report.roles,
            "notes": list(compiled.report.notes),
            "confirmationRequired": list(compiled.report.confirmation_required),
            "drivetrain": compiled.report.drivetrain,
            "weldedInstanceIds": list(compiled.report.welded_instance_ids),
            "articulatedInstanceIds": list(compiled.report.articulated_instance_ids),
        },
        "warnings": list(compiled.warnings),
        "physical": compiled.compiled.physical,
    }
