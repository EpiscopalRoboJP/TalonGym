from pathlib import Path

import pytest

from talongym.assets.cad_common import (
    CadImportError,
    field_tile_surface_y,
    filename_from_content_disposition,
    iter_assembly_parts,
    select_field_visual_parts,
    write_convex_collision_parts,
)
from talongym.assets.cad_sources import FIELD_SOURCE, OFFICIAL_FIELD_STEP_URL, PIECE_SOURCES, expected_sha256
from talongym.assets.import_field_cad import (
    align_field_mesh,
    guess_field_inch_scale,
    infer_up_axis,
    sniff_step_units,
    stage_step,
)


def test_sniff_onshape_metre_units():
    header = (
        "ISO-10303-21; DATA; "
        "#525217=( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );"
    )
    assert sniff_step_units(text=header) == "m"
    assert sniff_step_units(text="SI_UNIT(.MILLI.,.METRE.)") == "mm"
    assert sniff_step_units(text="CONVERSION_BASED_UNIT('INCH', #1)") == "in"


def test_guess_field_inch_scale_picks_metres():
    scale, units = guess_field_inch_scale(3.6576)
    assert units == "m"
    assert scale == pytest.approx(39.37007874015748)
    scale_in, units_in = guess_field_inch_scale(144.0)
    assert units_in == "in"
    assert scale_in == pytest.approx(1.0)


def test_infer_up_axis_is_smallest_extent():
    assert infer_up_axis([144.0, 144.0, 12.0]) == 2
    assert infer_up_axis([144.0, 12.0, 144.0]) == 1


def test_stage_step_rejects_missing(tmp_path: Path):
    with pytest.raises(CadImportError, match="not found"):
        stage_step(tmp_path / "missing.step")
    with pytest.raises(CadImportError, match="expected"):
        stage_step(__file__)


def test_official_field_binary_endpoint_is_pinned():
    assert OFFICIAL_FIELD_STEP_URL.endswith("/field/field-cad-step")
    assert expected_sha256(FIELD_SOURCE) == "05b35961c7df847741031f00fda73ddd068537f809e92a11b1cd59a94bcc8331"
    assert set(PIECE_SOURCES) == {"pollen", "nectar_red", "nectar_blue"}
    assert PIECE_SOURCES["pollen"]["specDiameterIn"] == 2.8
    assert PIECE_SOURCES["nectar_red"]["specDiameterIn"] == 3.6


def test_content_disposition_prefers_rfc5987_filename():
    header = (
        'inline; filename="BIOBUZZ_Full Field.20260912.step"; '
        "filename*=UTF-8''BIOBUZZ_Full%20Field.20260912.step"
    )
    assert filename_from_content_disposition(header) == "BIOBUZZ_Full Field.20260912.step"


def test_resolve_inch_scale_overrides_wrong_inch_header():
    from talongym.assets.cad_common import resolve_inch_scale

    # Cascadio metres, STEP header said INCH.
    scale, units = resolve_inch_scale(0.07112, "in", spec_diameter_in=2.8)
    assert units == "m"
    assert scale == pytest.approx(39.37007874015748)


def test_convex_hull_mesh_is_small():
    trimesh = pytest.importorskip("trimesh")
    from talongym.assets.cad_common import convex_hull_mesh

    dense = trimesh.creation.icosphere(subdivisions=4, radius=1.4)
    hull = convex_hull_mesh(dense, face_target=96)
    assert len(hull.faces) <= 128
    assert len(hull.faces) >= 8


def test_cli_import_field_cad_accepts_step():
    from typer.testing import CliRunner

    from talongym.cli import app

    env = {"COLUMNS": "120", "TERM": "dumb"}
    result = CliRunner().invoke(app, ["import-field-cad", "--help"], env=env)
    assert result.exit_code == 0
    help_text = (result.stdout or "") + (result.output or "")
    help_text = help_text.replace("\n", " ")
    assert "--step" in help_text
    assert "--url" in help_text
    assert "--verify" in help_text
    assert "--skip-pieces" in help_text
    piece_help = CliRunner().invoke(app, ["import-piece-cad", "--help"], env=env)
    assert piece_help.exit_code == 0
    piece_text = ((piece_help.stdout or "") + (piece_help.output or "")).replace("\n", " ")
    assert "--type" in piece_text


def test_as_trimesh_bakes_scene_transform():
    trimesh = pytest.importorskip("trimesh")
    from talongym.assets.import_field_cad import _as_trimesh

    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    scene = trimesh.Scene(box)
    scene.apply_translation([10.0, 0.0, 0.0])
    mesh = _as_trimesh(scene, trimesh)
    assert float(mesh.centroid[0]) == pytest.approx(10.0, abs=0.15)


def test_align_metre_zup_field_to_lab_inches():
    trimesh = pytest.importorskip("trimesh")

    # 144 x 144 x 12 in expressed in metres, Z-up, origin at center, sitting on z=0.
    box = trimesh.creation.box(extents=[3.6576, 3.6576, 0.3048])
    box.apply_translation([0.0, 0.0, 0.1524])
    aligned, meta = align_field_mesh(box, units="m")
    assert meta["unitsGuess"] == "m"
    assert meta["rotatedZupToYup"] is True
    assert aligned.extents[0] == pytest.approx(144.0, abs=0.2)
    assert aligned.extents[2] == pytest.approx(144.0, abs=0.2)
    assert aligned.extents[1] == pytest.approx(12.0, abs=0.2)
    assert aligned.bounds[0][1] == pytest.approx(0.0, abs=0.05)


def test_iter_assembly_parts_keeps_scene_nodes():
    trimesh = pytest.importorskip("trimesh")
    wall = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    hive = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    hive.apply_translation([12.0, 0.0, 0.0])
    scene = trimesh.Scene()
    scene.add_geometry(wall, node_name="wall_west")
    scene.add_geometry(hive, node_name="hive_frame")
    parts = iter_assembly_parts(scene, trimesh)
    names = {name for name, _mesh in parts}
    assert len(parts) >= 2
    assert any("wall" in n or "hive" in n or n.startswith("part_") for n in names)
    xs = [float(mesh.centroid[0]) for _name, mesh in parts]
    assert max(xs) - min(xs) > 8.0


def test_field_collision_refuses_single_part(tmp_path: Path):
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=[10.0, 2.0, 10.0])
    with pytest.raises(CadImportError, match="one collision mesh"):
        write_convex_collision_parts([("field", box)], tmp_path, rel_prefix="collision")


def test_field_collision_writes_convex_parts(tmp_path: Path):
    trimesh = pytest.importorskip("trimesh")
    a = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    b = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    b.apply_translation([20.0, 0.0, 0.0])
    parts = write_convex_collision_parts(
        [("hive_leg", a), ("wall_east", b)],
        tmp_path,
        rel_prefix="seasons/biobuzz_2026/collision",
    )
    assert len(parts) == 2
    assert all(row["convex"] is True for row in parts)
    assert all((tmp_path / f"{row['id']}.stl").is_file() for row in parts)
    assert parts[0]["asset"].startswith("seasons/biobuzz_2026/collision/")


def test_field_visual_filter_removes_embedded_pieces_and_off_field_hardware():
    trimesh = pytest.importorskip("trimesh")
    tile = trimesh.creation.box(extents=[24.0, 0.6, 24.0])
    tile.apply_translation([0.0, 1.4, 0.0])
    pollen = trimesh.creation.icosphere(radius=1.4)
    tray = trimesh.creation.box(extents=[10.0, 3.0, 15.0])
    tray.apply_translation([77.0, 2.0, 0.0])
    kept, stats = select_field_visual_parts(
        [
            ("am-2499_Field_Soft_Tiles", tile),
            ("am-5851_Pollen", pollen),
            ("am-5706_Artifact_Tray", tray),
        ]
    )
    assert [name for name, _mesh in kept] == ["am-2499_Field_Soft_Tiles"]
    assert stats["dropped"] == {"scoring_element": 1, "alliance_area": 1}
    assert field_tile_surface_y([("am-2499_Field_Soft_Tiles", tile)]) == pytest.approx(1.7)


def test_mesh_season_refuses_aabb_fallback(monkeypatch):
    from talongym.assets.import_field_cad import import_field_cad

    def boom(**_kwargs):
        raise CadImportError("official field STEP missing from endpoint")

    monkeypatch.setattr("talongym.assets.import_field_cad.fetch_official_field_step", boom)
    with pytest.raises(CadImportError, match="official field STEP missing|refusing AABB"):
        import_field_cad(include_pieces=False)


def test_hash_mismatch_on_local_field_step(tmp_path: Path, monkeypatch):
    from talongym.assets.import_field_cad import import_field_cad

    step = tmp_path / "fake.step"
    step.write_bytes(b"ISO-10303-21;HEADER;ENDSEC;DATA;ENDSEC;END-ISO-10303-21;\n" + b"x" * 2048)
    monkeypatch.setattr(
        "talongym.assets.import_field_cad.expected_sha256",
        lambda _spec: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    with pytest.raises(CadImportError, match="hash mismatch"):
        import_field_cad(step_path=step, include_pieces=False)


def test_shipped_field_preset_points_at_cad_manifest():
    from talongym.presets.loader import load_preset, validate_document

    field = load_preset("field", "biobuzz_2026_field_v1")
    assert field["cadManifest"] == "seasons/biobuzz_2026/cad_manifest.json"
    assert field["provenance"]["geometryProvenance"] == "official_cad"
    assert field["provenance"]["cadUrl"].endswith("/field-cad-step")
    pollen = next(p for p in field["gamePieces"] if p["typeId"] == "pollen")
    assert pollen["visualAsset"].endswith("pollen.glb")
    assert pollen["collisionAsset"].endswith("pollen_hull.stl")
    assert validate_document("field", field) == []


def test_committed_cad_assets_verify():
    from talongym.assets.import_field_cad import verify_season_cad
    from talongym.paths import ASSETS_DIR
    from talongym.presets.loader import load_preset

    field = load_preset("field", "biobuzz_2026_field_v1")
    pollen = ASSETS_DIR / field["gamePieces"][0]["visualAsset"]
    if not pollen.is_file():
        pytest.skip("derived CAD assets not generated")
    result = verify_season_cad()
    assert result["ok"] is True
    assert result["problems"] == []
    manifest = ASSETS_DIR / field["cadManifest"]
    assert manifest.is_file()
    import json

    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert doc["field"]["partCount"] >= 2
    assert set(doc["pieces"]) == {"pollen", "nectar_red", "nectar_blue"}
    field_block = doc["field"]
    assert field_block["transform"]["rotatedZupToYup"] in {True, False}
    assert field_block["transform"]["scaleToInches"] == pytest.approx(39.37007874015748, rel=1e-6)
    assert field_block["units"] == "m"
    assert doc["coordinateSystem"] == "ftc_inches_yup"
    bounds = field_block["boundsIn"]
    assert bounds["minIn"][1] < 0.0  # under-field source hardware remains recorded
    assert bounds["extentsIn"][0] >= 140
    assert bounds["extentsIn"][2] >= 140
    assert abs(bounds["extentsIn"][0] - 144) < 120  # off-field hardware extends past the 144 in tiles
    assert field_block["sha256"] == expected_sha256(FIELD_SOURCE)
    assert field_block["partCount"] == len(field_block["collisionParts"]) >= 2
    tess = field_block["tessellation"]
    assert tess["tileSurfaceSourceYIn"] > 1.0
    assert tess["visualFilter"]["dropped"]["scoring_element"] == 56
    trimesh = pytest.importorskip("trimesh")
    visual = trimesh.load(str(ASSETS_DIR / field_block["visualAsset"]))
    names = [str(name).lower() for name in visual.graph.nodes_geometry]
    assert len(names) > 100
    assert not any("pollen" in name or "nectar" in name for name in names)
    assert abs(float(visual.bounds[0][0])) <= 74.1
    assert abs(float(visual.bounds[1][0])) <= 74.1
    assert float(visual.bounds[0][1]) > -1.0
    assert not any("hex_lock_nut" in name or "fender_washer" in name for name in names)
    for mechanism in field_block["mechanisms"]:
        scene = trimesh.load(str(ASSETS_DIR / mechanism["visualAsset"]), force="scene")
        ribs = [
            mesh
            for name, mesh in scene.geometry.items()
            if "goal_rib" in str(name).lower()
        ]
        assert ribs
        assert all(float(mesh.area_faces.min()) > 1e-10 for mesh in ribs)
