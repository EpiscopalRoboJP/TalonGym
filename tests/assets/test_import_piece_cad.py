import pytest

from talongym.assets.cad_common import CadImportError, align_piece_mesh, check_piece_diameter
from talongym.assets.cad_manifest import (
    build_manifest,
    validate_manifest,
    verify_manifest_files,
)
from talongym.assets.cad_sources import PIECE_SOURCES


def test_piece_diameter_matches_manual_spec():
    check_piece_diameter(2.81, 2.8)
    check_piece_diameter(3.55, 3.6)
    with pytest.raises(CadImportError, match="diameter"):
        check_piece_diameter(10.0, 2.8)


def test_align_piece_mm_sphere_to_inches_centered():
    trimesh = pytest.importorskip("trimesh")
    # 2.8 in diameter = 71.12 mm; icosphere radius is 35.56 mm, Z-up.
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=35.56)
    sphere.apply_translation([10.0, 4.0, 20.0])
    aligned, meta = align_piece_mesh(sphere, units="mm", spec_diameter_in=2.8, assume_z_up=True)
    assert meta["unitsGuess"] == "mm"
    assert meta["rotatedZupToYup"] is True
    assert meta["measuredDiameterIn"] == pytest.approx(2.8, abs=0.08)
    check_piece_diameter(meta["measuredDiameterIn"], 2.8)
    center = aligned.bounds.mean(axis=0)
    assert center[0] == pytest.approx(0.0, abs=0.05)
    assert center[1] == pytest.approx(0.0, abs=0.05)
    assert center[2] == pytest.approx(0.0, abs=0.05)


def test_piece_catalog_covers_three_official_types():
    assert PIECE_SOURCES["pollen"]["sourceUrl"].endswith("Pollen.STEP")
    assert "Nectar.STEP" in PIECE_SOURCES["nectar_red"]["sourceUrl"]
    assert "Nectar.STEP" in PIECE_SOURCES["nectar_blue"]["sourceUrl"]
    assert len(PIECE_SOURCES["pollen"]["sha256"]) == 64


def test_manifest_schema_accepts_convex_parts_and_pieces():
    digest = "ab" * 32
    doc = build_manifest(
        season_slug="biobuzz",
        field={
            "sourceUrl": "https://ftc-resources.firstinspires.org/ftc/archive/2027/field/field-cad-step",
            "sourceFilename": "BIOBUZZ_Full_Field.20260912.step",
            "sha256": digest,
            "units": "m",
            "transform": {"scaleToInches": 39.37, "rotatedZupToYup": True, "translationIn": [0.0, 0.0, 0.0]},
            "boundsIn": {"minIn": [-72.0, 0.0, -72.0], "maxIn": [72.0, 24.0, 72.0], "extentsIn": [144.0, 24.0, 144.0]},
            "visualAsset": "seasons/biobuzz_2026/field.glb",
            "collisionParts": [
                {
                    "id": "wall_west",
                    "asset": "seasons/biobuzz_2026/collision/wall_west.stl",
                    "convex": True,
                    "faceCount": 12,
                    "minIn": [-72.0, 0.0, -72.0],
                    "maxIn": [-70.0, 12.0, 72.0],
                    "extentsIn": [2.0, 12.0, 144.0],
                },
                {
                    "id": "wall_east",
                    "asset": "seasons/biobuzz_2026/collision/wall_east.stl",
                    "convex": True,
                    "faceCount": 12,
                    "minIn": [70.0, 0.0, -72.0],
                    "maxIn": [72.0, 12.0, 72.0],
                    "extentsIn": [2.0, 12.0, 144.0],
                },
            ],
            "partCount": 2,
        },
        pieces={
            "pollen": {
                "sourceUrl": PIECE_SOURCES["pollen"]["sourceUrl"],
                "sourceFilename": "am-5851_yellow_Pollen.STEP",
                "sha256": PIECE_SOURCES["pollen"]["sha256"],
                "units": "mm",
                "transform": {"scaleToInches": 0.03937, "rotatedZupToYup": True, "translationIn": [0.0, 0.0, 0.0]},
                "boundsIn": {"minIn": [-1.4, -1.4, -1.4], "maxIn": [1.4, 1.4, 1.4], "extentsIn": [2.8, 2.8, 2.8]},
                "visualAsset": "seasons/biobuzz_2026/pieces/pollen.glb",
                "collisionAsset": "seasons/biobuzz_2026/pieces/pollen_hull.stl",
                "convex": True,
                "specDiameterIn": 2.8,
                "measuredDiameterIn": 2.8,
            }
        },
        notes="test",
    )
    assert validate_manifest(doc) == []
    problems = verify_manifest_files(doc, require_field=True)
    assert any("missing field visual" in p or "missing collision part" in p for p in problems)


def test_manifest_rejects_single_collision_part():
    digest = "cd" * 32
    doc = build_manifest(
        season_slug="biobuzz",
        field={
            "sourceUrl": "https://example.test/field.step",
            "sourceFilename": "field.step",
            "sha256": digest,
            "units": "in",
            "transform": {"scaleToInches": 1.0, "rotatedZupToYup": True, "translationIn": [0.0, 0.0, 0.0]},
            "boundsIn": {"minIn": [0.0, 0.0, 0.0], "maxIn": [1.0, 1.0, 1.0], "extentsIn": [1.0, 1.0, 1.0]},
            "visualAsset": "seasons/biobuzz_2026/field.glb",
            "collisionParts": [
                {"id": "all", "asset": "seasons/biobuzz_2026/collision/all.stl", "convex": True}
            ],
        },
    )
    errors = validate_manifest(doc)
    assert errors, errors
    assert any("collisionParts" in e for e in errors)


def test_committed_piece_transform_bounds_and_assets():
    from talongym.paths import ASSETS_DIR
    from talongym.presets.loader import load_preset

    field = load_preset("field", "biobuzz_2026_field_v1")
    manifest_path = ASSETS_DIR / field["cadManifest"]
    if not manifest_path.is_file():
        pytest.skip("derived CAD assets not generated")
    import json

    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = {p["typeId"]: p for p in field["gamePieces"]}
    for type_id, spec_d in (("pollen", 2.8), ("nectar_red", 3.6), ("nectar_blue", 3.6)):
        rec = doc["pieces"][type_id]
        assert rec["transform"]["rotatedZupToYup"] is True
        assert rec["units"] in {"m", "mm", "in"}
        assert rec["measuredDiameterIn"] == pytest.approx(spec_d, abs=0.15)
        assert rec["convex"] is True
        visual = ASSETS_DIR / rec["visualAsset"]
        hull = ASSETS_DIR / rec["collisionAsset"]
        assert visual.is_file() and visual.stat().st_size > 1000
        assert hull.is_file() and 200 < hull.stat().st_size < 50_000
        assert specs[type_id]["visualAsset"] == rec["visualAsset"]
        assert specs[type_id]["collisionAsset"] == rec["collisionAsset"]
        assert specs[type_id]["shape"]["radius"] == pytest.approx(spec_d / 2.0, abs=0.05)
        center = [
            0.5 * (a + b)
            for a, b in zip(rec["boundsIn"]["minIn"], rec["boundsIn"]["maxIn"])
        ]
        assert all(abs(c) < 0.6 for c in center)
