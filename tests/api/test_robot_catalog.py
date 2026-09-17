from __future__ import annotations

import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from talongym.api.app import app
from talongym.assets.import_robot_cad import write_ascii_box_stl


def _error(res):
    body = res.json()
    err = body.get("error") or (body.get("detail") or {}).get("error") or body.get("detail") or body
    if isinstance(err, dict):
        return err
    return {"code": str(res.status_code), "message": str(err)}


def test_catalog_list_and_get_part():
    client = TestClient(app)
    listing = client.get("/api/v1/catalog", params={"manufacturer": "gobilda", "tag": "motor"})
    assert listing.status_code == 200
    body = listing.json()
    skus = {row["sku"] for row in body["parts"]}
    assert "5203-2402-0019" in skus
    assert all(row["manufacturer"] == "gobilda" for row in body["parts"])
    motor_row = next(row for row in body["parts"] if row["sku"] == "5203-2402-0019")
    assert motor_row["preview"]["source"] in {"cad", "proxy"}
    assert motor_row["preview"]["kind"] in {"box", "sphere", "cylinder", "capsule", "convex_hull"}
    listing_all = client.get("/api/v1/catalog", params={"manufacturer": "gobilda", "tag": "channel"})
    assert listing_all.status_code == 200
    channel = next(row for row in listing_all.json()["parts"] if str(row["sku"]).startswith("1120"))
    assert channel["preview"]["kind"] in {"box", "convex_hull"}
    assert channel["preview"].get("sizeIn") or channel["preview"].get("radiusIn")
    part = client.get("/api/v1/catalog/parts/5203-2402-0019")
    assert part.status_code == 200
    detail = part.json()
    assert detail["cad"]["downloadEnabled"] is True
    assert detail["mounts"]
    missing = client.get("/api/v1/catalog/parts/nope-sku")
    assert missing.status_code == 404
    bad = client.get("/api/v1/catalog", params={"manufacturer": "andymark"})
    assert bad.status_code == 400


def test_catalog_download_disabled_and_cache_status():
    client = TestClient(app)
    blocked = client.post("/api/v1/catalog/parts/REV-31-1595/download")
    assert blocked.status_code == 409
    assert _error(blocked)["code"] == "CAD_DISABLED"
    cache = client.get("/api/v1/catalog/cache")
    assert cache.status_code == 200
    assert cache.json()["partCount"] >= 40


def test_catalog_download_fails_fast_without_cad_extra(monkeypatch):
    from talongym.assets import import_catalog_cad

    monkeypatch.setattr(import_catalog_cad, "cad_extra_available", lambda: False)
    client = TestClient(app)
    blocked = client.post("/api/v1/catalog/parts/5203-2402-0019/download")
    assert blocked.status_code == 503
    err = _error(blocked)
    assert err["code"] == "CAD_EXTRA"
    assert "pip install" in err["message"]


def test_robot_preset_list_marks_shipped_disk_presets():
    client = TestClient(app)
    rows = {row["id"]: row for row in client.get("/api/v1/presets/robot").json()}
    assert rows["mecanum_biobuzz_4cap"]["shipped"] is True
    assert rows["mecanum_meepmeep_defaults"]["shipped"] is True
    assert rows["gobilda_mecanum_starter"]["shipped"] is True
    assert rows["rev_mecanum_starter"]["shipped"] is True


def test_catalog_local_convert_and_asset_and_job(tmp_path, monkeypatch):
    pytest.importorskip("trimesh")
    from talongym.assets import import_catalog_cad

    stl = write_ascii_box_stl(tmp_path / "proxy.stl", 16, 8, 4)
    zipped = tmp_path / "proxy.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.write(stl, "proxy.stl")

    def fake_download(url, dest, *, expected, timeout=120.0):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zipped.read_bytes())
        return dest, "aa" * 32

    monkeypatch.setattr(import_catalog_cad, "download_catalog_cad", fake_download)
    import_catalog_cad.reset_jobs_for_tests()
    client = TestClient(app)
    started = client.post("/api/v1/catalog/parts/1207-0001-0001/download")
    assert started.status_code == 202
    job_id = started.json()["jobId"]
    deadline = time.time() + 8
    row = None
    while time.time() < deadline:
        row = client.get(f"/api/v1/catalog/jobs/{job_id}").json()
        if row["state"] in {"done", "failed"}:
            break
        time.sleep(0.05)
    assert row is not None
    assert row["state"] == "done", row
    ready = client.post("/api/v1/catalog/parts/1207-0001-0001/download")
    assert ready.status_code == 200
    visual = row["result"]["import"]["visualAsset"]
    glb = client.get(f"/api/v1/robot-assets/{visual}")
    assert glb.status_code == 200
    assert glb.content[:4] == b"glTF"
    detail = client.get("/api/v1/catalog/parts/1207-0001-0001").json()
    assert detail["cache"]["state"] == "ready"
    assert detail["cache"]["visualAsset"]
    assert detail["cache"]["thumbnailAsset"]
    assert detail["preview"]["source"] == "cad"
    thumbnail = detail["preview"]["thumbnailAsset"]
    assert thumbnail.endswith("/thumbnail.svg")
    rendered = client.get(f"/api/v1/robot-assets/{thumbnail}")
    assert rendered.status_code == 200
    assert b"<svg" in rendered.content[:200]
    assert b"<polygon" in rendered.content
    gone = client.delete("/api/v1/catalog/parts/1207-0001-0001/cache")
    assert gone.status_code == 200


def test_robot_part_model_upload(tmp_path):
    pytest.importorskip("trimesh")
    from talongym.assets.import_robot_cad import delete_robot_assets

    stl = write_ascii_box_stl(tmp_path / "part.stl", 6, 4, 2)
    client = TestClient(app)
    robot_id = "test_api_part"
    try:
        with stl.open("rb") as fh:
            res = client.post(
                f"/api/v1/presets/robot/{robot_id}/parts/intake/model",
                files={"file": ("part.stl", fh, "model/stl")},
                data={"parent_id": "chassis", "joint_transform_json": '{"x": 1.5}'},
            )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["partId"] == "intake"
        assert body["originPreserved"] is True
        assert body["parentId"] == "chassis"
        assert body["jointTransform"]["x"] == pytest.approx(1.5)
        glb = client.get(f"/api/v1/robot-assets/{body['visualAsset']}")
        assert glb.status_code == 200
    finally:
        delete_robot_assets(robot_id)


def test_part_upload_reports_cad_extra(monkeypatch, tmp_path):
    from talongym.assets import import_robot_cad

    def boom(*_args, **_kwargs):
        raise import_robot_cad.RobotCadError("trimesh missing; pip install -e '.[cad]'")

    monkeypatch.setattr(import_robot_cad, "import_robot_part_cad", boom)
    stl = tmp_path / "part.stl"
    stl.write_bytes(b"solid empty\nendsolid empty\n")
    client = TestClient(app)
    with stl.open("rb") as fh:
        res = client.post(
            "/api/v1/presets/robot/test_api_part/parts/intake/model",
            files={"file": ("part.stl", fh, "model/stl")},
        )
    assert res.status_code == 503
    assert _error(res)["code"] == "CAD_EXTRA"
    assert "pip install" in _error(res)["message"]


def test_recipe_and_compile_routes_cover_all_four_drivebases():
    client = TestClient(app)
    listing = client.get("/api/v1/catalog/recipes")
    assert listing.status_code == 200
    recipes = {row["id"]: row for row in listing.json()["recipes"]}
    ids = list(recipes)
    assert ids == ["gobilda_mecanum", "gobilda_tank", "rev_mecanum", "rev_tank"]
    assert recipes["rev_mecanum"]["cartridgeSkus"] == ["REV-41-1601", "REV-41-1602", "REV-41-1603"]
    assert recipes["rev_tank"]["defaultParameters"]["cartridgeSku"] == "REV-41-1603"
    assert recipes["gobilda_mecanum"]["cartridgeSkus"] == []
    missing = client.post("/api/v1/catalog/recipes/andymark_swerve/instantiate", json={})
    assert missing.status_code == 404
    expected_track = {
        "gobilda_mecanum": 7.244,
        "gobilda_tank": 4.409,
        "rev_mecanum": 8.189,
        "rev_tank": 8.189,
    }
    expected_gear = {
        "gobilda_mecanum": 19.2,
        "gobilda_tank": 19.2,
        "rev_mecanum": 5.0,
        "rev_tank": 5.0,
    }
    for recipe_id in ids:
        built = client.post(f"/api/v1/catalog/recipes/{recipe_id}/instantiate", json={})
        assert built.status_code == 200, built.text
        body = built.json()
        instance_ids = {row["id"] for row in body["assembly"]["instances"]}
        assert {"left_rail", "motor_fl", "wheel_fl"} <= instance_ids
        if recipe_id.startswith("rev"):
            assert "cartridge_fl" in instance_ids
        compiled = client.post("/api/v1/catalog/assemblies/compile", json=body["document"])
        assert compiled.status_code == 200, compiled.text
        preset = compiled.json()["preset"]
        compiled_ids = {row["id"] for row in preset["rigidParts"]}
        assert instance_ids <= compiled_ids
        assert compiled.json()["physical"] is True
        assert compiled.json()["report"]["confirmed"] is False
        assert preset["drivetrain"]["trackWidthIn"] == pytest.approx(expected_track[recipe_id], abs=0.35)
        assert preset["motors"]["gearRatio"] == pytest.approx(expected_gear[recipe_id])
        assert preset["drivetrain"]["trackWidthIn"] != pytest.approx(3.622, abs=0.05)


def test_recipe_draft_roundtrip_reloads_all_four_drivebases():
    from talongym.robot.recipes import RECIPE_IDS

    client = TestClient(app)
    for recipe_id in RECIPE_IDS:
        built = client.post(f"/api/v1/catalog/recipes/{recipe_id}/instantiate", json={})
        assert built.status_code == 200, built.text
        draft_id = f"draft_{recipe_id}"
        document = dict(built.json()["document"])
        document["id"] = draft_id
        saved = client.put(f"/api/v1/presets/robot/{draft_id}/draft", json=document)
        assert saved.status_code == 200, saved.text
        loaded = client.get(f"/api/v1/presets/robot/{draft_id}/draft")
        assert loaded.status_code == 200
        body = loaded.json()
        assert body["id"] == draft_id
        assert body["assembly"]["recipe"]["id"] == recipe_id
        instance_ids = {row["id"] for row in body["assembly"]["instances"]}
        assert {"left_rail", "motor_fl", "wheel_fl"} <= instance_ids
        compiled = client.post("/api/v1/catalog/assemblies/compile", json=body)
        assert compiled.status_code == 200, compiled.text
        assert compiled.json()["physical"] is True
        client.delete(f"/api/v1/presets/robot/{draft_id}/draft")


def test_robot_draft_roundtrip_does_not_replace_shipped_preset():
    client = TestClient(app)
    preset_id = "test_catalog_draft"
    missing = client.get(f"/api/v1/presets/robot/{preset_id}/draft")
    assert missing.status_code == 404
    saved = client.put(
        f"/api/v1/presets/robot/{preset_id}/draft",
        json={"schemaVersion": "1.2.0", "id": preset_id, "displayName": "Draft", "assembly": {"instances": [], "connections": []}},
    )
    assert saved.status_code == 200
    loaded = client.get(f"/api/v1/presets/robot/{preset_id}/draft")
    assert loaded.status_code == 200
    assert loaded.json()["displayName"] == "Draft"
    shipped = client.get("/api/v1/presets/robot/mecanum_biobuzz_4cap")
    assert shipped.status_code == 200
    assert shipped.json()["id"] == "mecanum_biobuzz_4cap"
    blocked = client.put(
        "/api/v1/presets/robot/mecanum_biobuzz_4cap/draft",
        json={"schemaVersion": "1.2.0", "id": "mecanum_biobuzz_4cap", "displayName": "Shadow", "assembly": {"instances": [], "connections": []}},
    )
    assert blocked.status_code == 409
    assert _error(blocked)["code"] == "SHIPPED"
    assert client.get("/api/v1/presets/robot/mecanum_biobuzz_4cap/draft").status_code == 404
    gone = client.delete(f"/api/v1/presets/robot/{preset_id}/draft")
    assert gone.status_code == 204
    assert client.get(f"/api/v1/presets/robot/{preset_id}/draft").status_code == 404
