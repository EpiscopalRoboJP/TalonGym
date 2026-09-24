from __future__ import annotations

import sys
from types import ModuleType

import pytest

from talongym.assets.import_catalog_cad import assert_allowlisted_url
from talongym.robot import catalog as catalog_mod
from talongym.robot.catalog import (
    CatalogError,
    cache_status,
    get_part,
    load_catalogs,
    search_parts,
    validate_catalog_document,
)
from talongym.robot.mounts import hole_in_part, part_mount


def test_catalog_manifests_match_schema():
    catalogs = load_catalogs()
    assert set(catalogs) == {"gobilda", "rev"}
    for name, document in catalogs.items():
        assert validate_catalog_document(document) == []
        assert document["manufacturer"] == name
        assert document["parts"]


def test_cad_availability_requires_step_converter(monkeypatch):
    catalog_mod.cad_extra_available.cache_clear()
    monkeypatch.setattr(catalog_mod, "find_spec", lambda name: object() if name == "trimesh" else None)
    assert catalog_mod.cad_extra_available() is False
    monkeypatch.setattr(catalog_mod, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "cascadio", ModuleType("cascadio"))
    catalog_mod.cad_extra_available.cache_clear()
    assert catalog_mod.cad_extra_available() is True
    catalog_mod.cad_extra_available.cache_clear()


def test_unverified_wheel_cad_is_not_downloadable():
    for sku in (
        "3213-3606-0001",
        "3213-3606-0002",
        "3213-3606-0003",
    ):
        assert get_part(sku)["cad"]["downloadEnabled"] is False


def test_scoring_wheels_have_manufacturer_cad_and_matching_bores():
    for sku, manufacturer, bore_mm, diameter_mm in (
        ("3615-4008-0072", "gobilda", 8.0, 72.0),
        ("3615-4008-0096", "gobilda", 8.0, 96.0),
        ("REV-41-2034", "rev", 5.0, 50.8),
        ("REV-41-1267", "rev", 5.0, 90.0),
    ):
        part = get_part(sku)
        assert part["manufacturer"] == manufacturer
        assert part["cad"]["downloadEnabled"] is True
        assert_allowlisted_url(part["cad"]["sourceUrl"])
        assert part_mount(part, "bore")["diameterMm"] == bore_mm
        assert 2 * part["collision"][0]["radiusIn"] * 25.4 == pytest.approx(diameter_mm, abs=0.01)


def test_mecanum_wheels_select_distinct_models_from_official_set():
    left = get_part("3606-0000-0096")
    right = get_part("3606-0100-0096")
    assert left["cad"]["sourceUrl"] == right["cad"]["sourceUrl"]
    assert left["cad"]["memberFilename"] == "3606-0000-0096 assembly.STEP"
    assert right["cad"]["memberFilename"] == "3606-0100-0096 assembly.STEP"
    assert left["massKg"] == right["massKg"] == 0.207


def test_catalog_seeds_drivebase_and_optional_electronics():
    tags = {part["sku"]: set(part["tags"]) for part in search_parts()}
    assert any("channel" in tags[sku] for sku in tags if sku.startswith("1120-"))
    assert any("extrusion" in row for row in tags.values())
    assert any("wheel_mecanum" in row for row in tags.values())
    assert any("wheel_traction" in row for row in tags.values())
    assert any("motor" in row and "gearbox" in row for row in tags.values())
    assert any("servo" in row for row in tags.values())
    assert any("intake_roller" in row for row in tags.values())
    assert any("flywheel" in row for row in tags.values())
    assert any("control_hub" in row for row in tags.values())
    assert any("battery" in row for row in tags.values())
    assert any("sensor" in row for row in tags.values())
    channel = get_part("1120-0007-0192")
    web = part_mount(channel, "web")
    assert list(web["uAxis"]) == [1.0, 0.0, 0.0]
    first = hole_in_part(web, (0, 1))
    last = hole_in_part(web, (23, 1))
    assert last[0] - first[0] == pytest.approx(7.244094488188976, abs=1e-6)
    assert last[1] == pytest.approx(first[1], abs=1e-6)
    cartridge = get_part("REV-41-1603")
    assert cartridge["gearRatio"] == 5.0
    gobilda_motor = get_part("5203-2402-0019")
    assert gobilda_motor["cad"]["downloadEnabled"] is True
    assert gobilda_motor["cad"]["sourceUrl"].startswith("https://www.gobilda.com/content/step_files/")
    face = part_mount(gobilda_motor, "face")
    output = part_mount(gobilda_motor, "output")
    assert list(face["axis"]) == list(output["axis"])
    wheel = get_part("3213-3606-0001")
    assert list(part_mount(wheel, "bore")["axis"]) == [0.0, 0.0, 1.0]
    assert part_mount(get_part("REV-41-1621"), "motor_face")["axis"][1] == pytest.approx(1.0)
    hub = get_part("REV-31-1595")
    assert hub["cad"]["downloadEnabled"] is False
    assert "sourceUrl" not in hub["cad"]


def test_search_filters_manufacturer_and_tag():
    motors = search_parts(manufacturer="rev", tag="motor")
    assert {part["sku"] for part in motors} >= {"REV-41-1600", "REV-41-1300"}
    assert all(part["manufacturer"] == "rev" for part in motors)
    named = search_parts(query="yellow jacket")
    assert "5203-2402-0019" in {part["sku"] for part in named}
    with pytest.raises(CatalogError, match="unknown manufacturer"):
        search_parts(manufacturer="andymark")


def test_allowlisted_urls_reject_unknown_hosts():
    ok = "https://www.gobilda.com/content/step_files/1207-0001-0001.zip"
    assert_allowlisted_url(ok, ok)
    with pytest.raises(CatalogError, match="allowlisted"):
        assert_allowlisted_url("https://evil.example/cad.zip", "https://evil.example/cad.zip")
    with pytest.raises(CatalogError, match="allowlist entry"):
        assert_allowlisted_url("https://www.gobilda.com/other.zip", ok)
    with pytest.raises(CatalogError, match="https"):
        assert_allowlisted_url("http://www.gobilda.com/content/step_files/1207-0001-0001.zip", ok)


def test_cache_status_starts_missing_or_unavailable():
    status = cache_status()
    assert status["partCount"] >= 40
    counted = sum(status["counts"].values())
    assert counted == status["partCount"]
    assert status["counts"]["unavailable"] >= 1
    hub = get_part("REV-31-1595")
    from talongym.robot.catalog import part_summary

    summary = part_summary(hub)
    assert summary["cache"]["state"] == "unavailable"
    assert summary["cache"]["reason"] == "download_disabled"
    assert "cadExtra" in status


def test_part_catalogs_are_not_indexed_as_robot_presets():
    from talongym.presets.loader import preset_index

    idx = preset_index(refresh=True)
    assert "gobilda_mecanum" not in idx["robot"]
    assert "gobilda" not in idx["robot"]
    assert "rev" not in idx["robot"]
    assert "mecanum_biobuzz_4cap" in idx["robot"]
    assert "gobilda_mecanum_starter" in idx["robot"]
    assert "rev_tank_starter" in idx["robot"]
    from talongym.presets.loader import is_shipped_preset

    assert is_shipped_preset("robot", "mecanum_biobuzz_4cap") is True
    assert is_shipped_preset("robot", "e2e_catalog_gobilda_mecanum") is False


def test_cache_status_degrades_explicitly_without_cad_extra(monkeypatch):
    from talongym.robot import catalog as catalog_mod
    from talongym.robot.catalog import part_summary

    monkeypatch.setattr(catalog_mod, "cad_extra_available", lambda: False)
    monkeypatch.setattr(
        catalog_mod,
        "cache_entry",
        lambda *_args, **_kwargs: {"state": "missing", "visualAsset": None, "collisionAsset": None},
    )
    status = cache_status()
    assert status["cadExtra"] is False
    assert "pip install" in str(status["cadUnavailableReason"])
    motor = part_summary(get_part("5203-2402-0019"))
    assert motor["downloadEnabled"] is True
    assert motor["cache"]["state"] == "unavailable"
    assert motor["cache"]["reason"] == "cad_extra_required"
    hub = part_summary(get_part("REV-31-1595"))
    assert hub["cache"]["state"] == "unavailable"
    assert hub["cache"]["reason"] == "download_disabled"


def test_cache_is_fresh_requires_visual_and_thumbnail(tmp_path, monkeypatch):
    import json

    from talongym.assets.cad_common import CATALOG_CAD_GENERATOR_VERSION
    from talongym.assets.import_catalog_cad import cache_is_fresh
    from talongym.robot import catalog as catalog_mod

    var = tmp_path / "isolated_var"
    monkeypatch.setattr("talongym.paths.VAR_DIR", var)
    part = get_part("1207-0001-0001")
    assert cache_is_fresh(part) is False
    dest = catalog_mod.catalog_part_dir(part["manufacturer"], part["sku"])
    dest.mkdir(parents=True, exist_ok=True)
    visual = dest / "visual.glb"
    collision = dest / "collision.stl"
    thumb = dest / "thumbnail.svg"
    visual.write_bytes(b"glTF" + b"\0" * 12)
    collision.write_bytes(b"solid empty\nendsolid empty\n")
    meta = {
        "sku": part["sku"],
        "manufacturer": part["manufacturer"],
        "generatorVersion": CATALOG_CAD_GENERATOR_VERSION,
        "conversionFingerprint": catalog_mod.cad_conversion_fingerprint(part),
        "sha256": str((part.get("cad") or {}).get("sha256") or "aa" * 32).lower(),
        "visualAsset": str(visual.relative_to(var / "assets")).replace("\\", "/"),
        "collisionAsset": str(collision.relative_to(var / "assets")).replace("\\", "/"),
        "stale": False,
    }
    catalog_mod.cache_meta_path(part["manufacturer"], part["sku"]).write_text(json.dumps(meta), encoding="utf-8")
    assert cache_is_fresh(part) is False
    thumb.write_text("<svg xmlns='http://www.w3.org/2000/svg'><polygon points='0,0 1,0 0,1'/></svg>", encoding="utf-8")
    meta["thumbnailAsset"] = str(thumb.relative_to(var / "assets")).replace("\\", "/")
    catalog_mod.cache_meta_path(part["manufacturer"], part["sku"]).write_text(json.dumps(meta), encoding="utf-8")
    assert cache_is_fresh(part) is True
    changed = dict(part)
    changed["collision"] = [{"kind": "box", "sizeIn": [100, 100, 100]}]
    assert cache_is_fresh(changed) is False
    assert catalog_mod.cache_entry(part["manufacturer"], part["sku"])["state"] == "ready"
    meta["generatorVersion"] = "1.2.0"
    catalog_mod.cache_meta_path(part["manufacturer"], part["sku"]).write_text(json.dumps(meta), encoding="utf-8")
    assert cache_is_fresh(part) is False
    stale = catalog_mod.cache_entry(part["manufacturer"], part["sku"])
    assert stale["state"] == "stale"
    assert stale["visualAsset"] is None
    assert stale["collisionAsset"] is None
    assert stale["thumbnailAsset"] is None
    assert catalog_mod.part_preview(part, stale)["source"] == "proxy"
