from __future__ import annotations

import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from talongym.api.app import app
from talongym.assets.import_robot_cad import write_ascii_box_stl


def test_cache_all_skips_disabled_and_returns_without_blocking(monkeypatch):
    from talongym.assets import catalog_cache_batch, import_catalog_cad
    from talongym.robot.catalog import get_part

    monkeypatch.setattr(import_catalog_cad, "cad_extra_available", lambda: False)
    import_catalog_cad.reset_jobs_for_tests()
    client = TestClient(app)
    idle = client.get("/api/v1/catalog/cache/all")
    assert idle.status_code == 200
    assert idle.json()["state"] == "idle"
    t0 = time.perf_counter()
    res = client.post("/api/v1/catalog/cache/all", json={})
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.75
    assert res.status_code == 200
    body = res.json()
    assert body["state"] == "done"
    assert body["counts"]["queued"] == 0
    assert body["counts"]["skipped"] >= 1
    reasons = {row["reason"] for row in body["items"] if row["state"] == "skipped"}
    assert "download_disabled" in reasons
    assert "cad_extra_required" in reasons
    disabled = next(row for row in body["items"] if row["sku"] == "REV-31-1595")
    assert disabled["state"] == "skipped"
    assert disabled["reason"] == "download_disabled"
    enabled = get_part("5203-2402-0019")
    classified = catalog_cache_batch.classify_cache_target(enabled)
    assert classified["state"] == "skipped"
    assert classified["reason"] == "cad_extra_required"
    blank = catalog_cache_batch.classify_cache_target(
        {"sku": "blank", "manufacturer": "gobilda", "cad": {"downloadEnabled": True, "sourceUrl": ""}}
    )
    assert blank["state"] == "skipped"
    assert blank["reason"] == "no_source_url"
    bad = client.post("/api/v1/catalog/cache/all", json={"manufacturer": "andymark"})
    assert bad.status_code == 400
    missing = client.get("/api/v1/catalog/cache/all/does-not-exist")
    assert missing.status_code == 404


def test_cache_all_queues_bounded_jobs_cancel_and_retry(monkeypatch):
    import threading

    from talongym.assets import import_catalog_cad
    from talongym.robot.catalog import CatalogError

    monkeypatch.setenv("TALONGYM_CAD_CACHE_CONCURRENCY", "2")
    monkeypatch.setattr(import_catalog_cad, "cad_extra_available", lambda: True)
    import_catalog_cad.reset_jobs_for_tests()

    started = threading.Event()
    release = threading.Event()
    inflight = 0
    max_inflight = 0
    calls: list[str] = []
    lock = threading.Lock()
    fail_once = {"5203-2402-0019": 1}

    def fake_convert(sku, *, force=False, source=None):
        nonlocal inflight, max_inflight
        with lock:
            calls.append(sku)
            inflight += 1
            max_inflight = max(max_inflight, inflight)
        started.set()
        if not release.wait(2.0):
            raise CatalogError("test convert timed out")
        with lock:
            inflight -= 1
            if fail_once.get(sku):
                fail_once[sku] -= 1
                raise CatalogError("mock convert failed")
        return {"sku": sku, "reused": False}

    monkeypatch.setattr(import_catalog_cad, "convert_catalog_part", fake_convert)
    client = TestClient(app)
    skus = ["1207-0001-0001", "5203-2402-0019", "3606-0100-0096"]
    t0 = time.perf_counter()
    started_res = client.post("/api/v1/catalog/cache/all", json={"skus": skus, "force": True})
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5
    assert started_res.status_code == 202
    first = started_res.json()
    batch_id = first["id"]
    assert first["counts"]["queued"] + first["counts"]["converting"] == 3
    assert started.wait(1.0)
    time.sleep(0.08)
    assert max_inflight <= 2

    duplicate = client.post("/api/v1/catalog/cache/all", json={"skus": skus, "force": True})
    assert duplicate.status_code == 202
    assert duplicate.json()["id"] == batch_id

    cancelled = client.post(f"/api/v1/catalog/cache/all/{batch_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["counts"]["cancelled"] >= 1
    release.set()

    deadline = time.time() + 3
    row = cancelled.json()
    while time.time() < deadline:
        row = client.get(f"/api/v1/catalog/cache/all/{batch_id}").json()
        if row["state"] in {"done", "cancelled"} and row["counts"]["converting"] == 0:
            break
        time.sleep(0.05)
    assert row["counts"]["converting"] == 0
    assert row["counts"]["cancelled"] + row["counts"]["failed"] + row["counts"]["ready"] == 3

    import_catalog_cad.reset_jobs_for_tests()
    release.clear()
    started.clear()
    fail_once["5203-2402-0019"] = 1
    second = client.post("/api/v1/catalog/cache/all", json={"skus": ["1207-0001-0001", "5203-2402-0019"], "force": True})
    assert second.status_code == 202
    batch_id = second.json()["id"]
    assert started.wait(1.0)
    release.set()
    deadline = time.time() + 3
    row = second.json()
    while time.time() < deadline:
        row = client.get("/api/v1/catalog/cache/all").json()
        if row["state"] == "done":
            break
        time.sleep(0.05)
    assert row["state"] == "done"
    assert row["counts"]["failed"] == 1
    assert row["counts"]["ready"] == 1
    retried = client.post(f"/api/v1/catalog/cache/all/{batch_id}/retry")
    assert retried.status_code in {200, 202}
    deadline = time.time() + 3
    while time.time() < deadline:
        row = client.get(f"/api/v1/catalog/cache/all/{batch_id}").json()
        if row["state"] == "done" and row["counts"]["failed"] == 0:
            break
        time.sleep(0.05)
    assert row["counts"]["failed"] == 0
    assert row["counts"]["ready"] == 2
    assert all(sku in calls for sku in ("1207-0001-0001", "5203-2402-0019"))


def test_cache_all_dedupes_inflight_single_download(monkeypatch):
    import threading

    from talongym.assets import import_catalog_cad
    from talongym.robot.catalog import CatalogError

    monkeypatch.setattr(import_catalog_cad, "cad_extra_available", lambda: True)
    import_catalog_cad.reset_jobs_for_tests()
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def fake_convert(sku, *, force=False, source=None):
        calls.append(sku)
        started.set()
        if not release.wait(2.0):
            raise CatalogError("test convert timed out")
        return {"sku": sku, "reused": False}

    monkeypatch.setattr(import_catalog_cad, "convert_catalog_part", fake_convert)
    client = TestClient(app)
    t0 = time.perf_counter()
    single = client.post("/api/v1/catalog/parts/1207-0001-0001/download")
    assert time.perf_counter() - t0 < 0.5
    assert single.status_code == 202
    assert started.wait(1.0)
    batch = client.post("/api/v1/catalog/cache/all", json={"skus": ["1207-0001-0001"], "force": True})
    assert batch.status_code == 202
    time.sleep(0.12)
    release.set()
    deadline = time.time() + 3
    row = batch.json()
    while time.time() < deadline:
        row = client.get("/api/v1/catalog/cache/all").json()
        if row["state"] in {"done", "cancelled"} and row["counts"]["converting"] == 0:
            break
        time.sleep(0.05)
    assert row["state"] == "done"
    assert row["counts"]["ready"] == 1
    assert calls.count("1207-0001-0001") == 1


def test_cache_all_reuses_fresh_cache_and_stays_in_var(tmp_path, monkeypatch):
    pytest.importorskip("trimesh")
    from talongym.assets import import_catalog_cad
    from talongym.robot.catalog import robot_parts_root

    stl = write_ascii_box_stl(tmp_path / "proxy.stl", 16, 8, 4)
    zipped = tmp_path / "proxy.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.write(stl, "proxy.stl")

    downloads: list[str] = []

    def fake_download(url, dest, *, expected, timeout=120.0):
        downloads.append(url)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zipped.read_bytes())
        return dest, "aa" * 32

    monkeypatch.setattr(import_catalog_cad, "cad_extra_available", lambda: True)
    monkeypatch.setattr(import_catalog_cad, "download_catalog_cad", fake_download)
    import_catalog_cad.reset_jobs_for_tests()
    client = TestClient(app)
    started = client.post("/api/v1/catalog/cache/all", json={"skus": ["1207-0001-0001"], "force": True})
    assert started.status_code == 202
    deadline = time.time() + 8
    row = started.json()
    while time.time() < deadline:
        row = client.get("/api/v1/catalog/cache/all").json()
        if row["state"] == "done":
            break
        time.sleep(0.05)
    assert row["state"] == "done", row
    assert row["counts"]["ready"] == 1
    assert downloads
    cached = robot_parts_root() / "gobilda" / "1207-0001-0001"
    assert cached.is_dir()
    assert "var" in str(cached).replace("\\", "/")
    again = client.post("/api/v1/catalog/cache/all", json={"skus": ["1207-0001-0001"]})
    assert again.status_code == 200
    assert again.json()["counts"]["ready"] == 1
    assert again.json()["items"][0]["reason"] == "already_cached"
    assert len(downloads) == 1


def test_cache_all_scopes_manufacturer_and_empty_sku_filter(monkeypatch):
    from talongym.assets import import_catalog_cad

    monkeypatch.setattr(import_catalog_cad, "cad_extra_available", lambda: False)
    import_catalog_cad.reset_jobs_for_tests()
    client = TestClient(app)
    gobilda = client.post("/api/v1/catalog/cache/all", json={"manufacturer": "gobilda"})
    assert gobilda.status_code == 200
    body = gobilda.json()
    assert body["manufacturer"] == "gobilda"
    assert body["total"] >= 1
    assert all(row.get("manufacturer") == "gobilda" for row in body["items"])
    empty = client.post("/api/v1/catalog/cache/all", json={"skus": []})
    assert empty.status_code == 200
    assert empty.json()["total"] == 0
    assert empty.json()["items"] == []
