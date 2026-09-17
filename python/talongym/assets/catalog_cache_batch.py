"""Bounded background Cache-all CAD jobs for the robot-part catalog."""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from talongym.assets import import_catalog_cad as cad_mod
from talongym.assets.import_robot_cad import RobotCadError, is_cad_extra_error
from talongym.robot.catalog import MANUFACTURERS, CatalogError, iter_parts

BATCH_ITEM_STATES = ("queued", "converting", "ready", "failed", "skipped", "cancelled")
SKIP_DOWNLOAD_DISABLED = "download_disabled"
SKIP_NO_SOURCE = "no_source_url"
SKIP_CAD_EXTRA = "cad_extra_required"
SKIP_ALREADY_CACHED = "already_cached"

_batches: dict[str, dict[str, Any]] = {}
_batch_lock = threading.Lock()
_running_batches: set[str] = set()
_active_batch_id: str | None = None


def cache_concurrency() -> int:
    raw = os.environ.get("TALONGYM_CAD_CACHE_CONCURRENCY", "2")
    try:
        value = int(raw)
    except ValueError:
        value = 2
    return max(1, min(4, value))


def classify_cache_target(part: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    sku = str(part.get("sku") or "")
    cad = part.get("cad") if isinstance(part.get("cad"), dict) else {}
    item: dict[str, Any] = {
        "sku": sku,
        "manufacturer": part.get("manufacturer"),
        "displayName": part.get("displayName"),
    }
    if not cad.get("downloadEnabled"):
        item.update({"state": "skipped", "reason": SKIP_DOWNLOAD_DISABLED})
        return item
    if not str(cad.get("sourceUrl") or "").strip():
        item.update({"state": "skipped", "reason": SKIP_NO_SOURCE})
        return item
    if cad_mod.cache_is_fresh(part) and not force:
        item.update({"state": "ready", "reason": SKIP_ALREADY_CACHED})
        return item
    if not cad_mod.cad_extra_available():
        item.update({"state": "skipped", "reason": SKIP_CAD_EXTRA})
        return item
    item.update({"state": "queued"})
    return item


def _batch_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {state: 0 for state in BATCH_ITEM_STATES}
    for item in items:
        state = str(item.get("state") or "queued")
        counts[state] = counts.get(state, 0) + 1
    return counts


def empty_batch_snapshot() -> dict[str, Any]:
    return {
        "id": None,
        "state": "idle",
        "force": False,
        "manufacturer": None,
        "counts": {state: 0 for state in BATCH_ITEM_STATES},
        "total": 0,
        "concurrency": cache_concurrency(),
        "items": [],
    }


def _batch_snapshot(batch: dict[str, Any]) -> dict[str, Any]:
    items = [dict(row) for row in batch.get("items") or []]
    return {
        "id": batch["id"],
        "state": batch["state"],
        "force": bool(batch.get("force")),
        "manufacturer": batch.get("manufacturer"),
        "counts": _batch_counts(items),
        "total": len(items),
        "concurrency": int(batch.get("concurrency") or cache_concurrency()),
        "items": items,
    }


def get_batch(batch_id: str) -> dict[str, Any] | None:
    with _batch_lock:
        row = _batches.get(batch_id)
        return _batch_snapshot(row) if row else None


def latest_batch() -> dict[str, Any] | None:
    with _batch_lock:
        if _active_batch_id and _active_batch_id in _batches:
            return _batch_snapshot(_batches[_active_batch_id])
        if not _batches:
            return None
        newest = max(_batches.values(), key=lambda row: str(row.get("id") or ""))
        return _batch_snapshot(newest)


def _active_running_batch() -> dict[str, Any] | None:
    with _batch_lock:
        if _active_batch_id and _active_batch_id in _batches:
            row = _batches[_active_batch_id]
            if row.get("state") in {"queued", "running", "cancelling"}:
                return _batch_snapshot(row)
    return None


def start_cache_all(
    *,
    force: bool = False,
    manufacturer: str | None = None,
    skus: list[str] | None = None,
) -> dict[str, Any]:
    existing = _active_running_batch()
    if existing:
        return existing
    sku_filter: set[str] | None
    if skus is None:
        sku_filter = None
    else:
        sku_filter = {str(sku).strip().lower() for sku in skus if str(sku).strip()}
    maker = (manufacturer or "").strip().lower() or None
    if maker and maker not in MANUFACTURERS:
        raise CatalogError(f"unknown manufacturer {maker}")
    items: list[dict[str, Any]] = []
    for part in iter_parts():
        sku = str(part.get("sku") or "")
        if maker and str(part.get("manufacturer") or "") != maker:
            continue
        if sku_filter is not None and sku.lower() not in sku_filter:
            continue
        items.append(classify_cache_target(part, force=force))
    batch_id = uuid.uuid4().hex[:16]
    queued = any(row.get("state") == "queued" for row in items)
    batch = {
        "id": batch_id,
        "state": "queued" if queued else "done",
        "force": force,
        "manufacturer": maker,
        "concurrency": cache_concurrency(),
        "cancel": False,
        "items": items,
    }
    with _batch_lock:
        global _active_batch_id
        _batches[batch_id] = batch
        _active_batch_id = batch_id
    if queued:
        _ensure_runner(batch_id)
    return _batch_snapshot(batch)


def cancel_batch(batch_id: str) -> dict[str, Any] | None:
    with _batch_lock:
        batch = _batches.get(batch_id)
        if not batch:
            return None
        batch["cancel"] = True
        converting = 0
        for item in batch["items"]:
            if item.get("state") == "queued":
                item["state"] = "cancelled"
                item["reason"] = "cancelled"
            if item.get("state") == "converting":
                converting += 1
        batch["state"] = "cancelling" if converting else "cancelled"
        return _batch_snapshot(batch)


def retry_batch(batch_id: str) -> dict[str, Any] | None:
    with _batch_lock:
        batch = _batches.get(batch_id)
        if not batch:
            return None
        batch["cancel"] = False
        retried = 0
        for item in batch["items"]:
            if item.get("state") == "failed":
                item["state"] = "queued"
                item.pop("error", None)
                item.pop("reason", None)
                retried += 1
        should_start = bool(retried)
        if retried and batch.get("state") not in {"queued", "running"}:
            batch["state"] = "queued"
        snapshot = _batch_snapshot(batch)
    if should_start:
        _ensure_runner(batch_id)
    return snapshot


def _claim_next_sku(batch_id: str) -> str | None:
    with _batch_lock:
        batch = _batches.get(batch_id)
        if not batch:
            return None
        if batch.get("cancel"):
            for item in batch["items"]:
                if item.get("state") == "queued":
                    item["state"] = "cancelled"
                    item["reason"] = "cancelled"
            return None
        for item in batch["items"]:
            sku = str(item.get("sku") or "")
            if item.get("state") != "queued" or not sku:
                continue
            item["state"] = "converting"
            batch["state"] = "running"
            return sku
    return None


def _set_item_state(batch_id: str, sku: str, state: str, **extra: Any) -> None:
    wanted = sku.lower()
    with _batch_lock:
        batch = _batches.get(batch_id)
        if not batch:
            return
        for item in batch["items"]:
            if str(item.get("sku") or "").lower() != wanted:
                continue
            item["state"] = state
            item.update(extra)
            break


def _finalize_batch(batch_id: str) -> None:
    with _batch_lock:
        batch = _batches.get(batch_id)
        if not batch:
            return
        if any(row.get("state") in {"queued", "converting"} for row in batch["items"]):
            return
        batch["state"] = "cancelled" if batch.get("cancel") else "done"


def _ensure_runner(batch_id: str) -> None:
    with _batch_lock:
        if batch_id in _running_batches:
            return
        _running_batches.add(batch_id)
    threading.Thread(target=_batch_runner, args=(batch_id,), daemon=True).start()


def _batch_has(batch_id: str, *states: str) -> bool:
    with _batch_lock:
        batch = _batches.get(batch_id)
        if not batch:
            return False
        wanted = set(states)
        return any(row.get("state") in wanted for row in batch.get("items") or [])


def _wait_existing_job(sku: str, timeout: float = 180.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout
    row = cad_mod.job_for_sku(sku)
    if not row or row.get("state") not in {"queued", "running"}:
        return None
    while time.time() < deadline:
        row = cad_mod.job_for_sku(sku)
        if not row or row.get("state") not in {"queued", "running"}:
            return row
        time.sleep(0.04)
    return cad_mod.job_for_sku(sku)


def _batch_convert_one(batch_id: str, sku: str, force: bool) -> None:
    try:
        waited = _wait_existing_job(sku)
        if waited is not None:
            if waited.get("state") == "done":
                _set_item_state(batch_id, sku, "ready", reused=True)
                return
            if waited.get("state") == "failed":
                _set_item_state(
                    batch_id,
                    sku,
                    "failed",
                    error=waited.get("error") or {"code": "CAD_IMPORT", "message": "CAD conversion failed"},
                )
                return
        job_id, created = cad_mod.register_sku_job(sku, state="running", force=force)
        if not created:
            waited = _wait_existing_job(sku) or cad_mod.job_for_sku(sku)
            if waited and waited.get("state") == "done":
                _set_item_state(batch_id, sku, "ready", reused=True)
                return
            if waited and waited.get("state") == "failed":
                _set_item_state(
                    batch_id,
                    sku,
                    "failed",
                    error=waited.get("error") or {"code": "CAD_IMPORT", "message": "CAD conversion failed"},
                )
                return
        try:
            result = cad_mod.convert_catalog_part(sku, force=force)
            cad_mod.finish_sku_job(job_id, sku, state="done", result=result)
            _set_item_state(batch_id, sku, "ready", reused=bool(result.get("reused")))
        except (CatalogError, RobotCadError, Exception) as exc:
            error = {
                "code": "CAD_EXTRA" if is_cad_extra_error(exc) else "CAD_IMPORT",
                "message": str(exc),
            }
            cad_mod.finish_sku_job(job_id, sku, state="failed", error=error)
            _set_item_state(batch_id, sku, "failed", error=error)
    finally:
        _finalize_batch(batch_id)


def _batch_runner(batch_id: str) -> None:
    try:
        with _batch_lock:
            batch = _batches.get(batch_id)
            concurrency = int((batch or {}).get("concurrency") or cache_concurrency())
            force = bool((batch or {}).get("force"))
        futures: set[Any] = set()
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="cad-cache") as pool:
            while True:
                while len(futures) < concurrency:
                    sku = _claim_next_sku(batch_id)
                    if sku is None:
                        break
                    futures.add(pool.submit(_batch_convert_one, batch_id, sku, force))
                if futures:
                    done, futures = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                    for finished in done:
                        finished.result()
                    continue
                if _batch_has(batch_id, "queued", "converting"):
                    time.sleep(0.04)
                    continue
                break
        _finalize_batch(batch_id)
    finally:
        with _batch_lock:
            _running_batches.discard(batch_id)
        _finalize_batch(batch_id)


def reset_batches_for_tests() -> None:
    with _batch_lock:
        global _active_batch_id
        _batches.clear()
        _running_batches.clear()
        _active_batch_id = None
