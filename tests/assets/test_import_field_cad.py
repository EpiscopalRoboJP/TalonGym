from pathlib import Path

import pytest

from talongym.assets.import_field_cad import (
    CadImportError,
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


def test_cli_import_field_cad_accepts_step():
    from typer.testing import CliRunner

    from talongym.cli import app

    result = CliRunner().invoke(app, ["import-field-cad", "--help"])
    assert result.exit_code == 0
    assert "--step" in result.stdout


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


def test_part_face_budgets_fit_target_without_starving_small_parts():
    from talongym.assets.import_field_cad import PART_MIN_FACES, _part_face_budgets

    counts = [12, 300, 8_000, 150_000, 150_000]
    budgets = _part_face_budgets(counts, 40_000)
    assert sum(budgets) <= 40_000
    assert budgets[:2] == [12, 300]
    assert all(b >= min(n, PART_MIN_FACES) for n, b in zip(counts, budgets, strict=True))
    assert _part_face_budgets(counts, 10**7) == counts


def test_visual_parts_drop_hardware_and_in_field_game_pieces():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("fast_simplification")
    from talongym.assets.import_field_cad import _decimate_visual, _visual_parts

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=[24.0, 0.6, 24.0]), node_name="tile", geom_name="tile")
    scene.add_geometry(trimesh.creation.icosphere(subdivisions=5, radius=2.5), node_name="rock", geom_name="rock")
    scene.add_geometry(trimesh.creation.cylinder(radius=0.1, height=1.0), node_name="screw", geom_name="screw")
    ball = trimesh.creation.icosphere(subdivisions=2, radius=1.4)
    scene.add_geometry(ball, node_name="ball", geom_name="ball")
    stock = trimesh.creation.icosphere(subdivisions=2, radius=1.4)
    stock.apply_translation([-75.0, 0.0, 0.0])
    scene.add_geometry(stock, node_name="ball_1", geom_name="ball_1")

    parts, placements = _visual_parts(scene, trimesh, frozenset({"ball"}))
    names = [name for name, _bounds in placements]
    # The screw is too small to place; both balls are placed, but only the out-of-field one is drawn.
    assert sorted(names) == ["ball", "ball_1", "rock", "tile"]
    assert len(parts) == 3

    mesh, decimated = _decimate_visual(parts, trimesh, target=2_000)
    assert decimated is True
    # The tile is untouched and the rock is reduced but still spans its size.
    assert float(mesh.bounds[1][1]) == pytest.approx(2.5, abs=0.2)
    assert float(mesh.bounds[0][0]) == pytest.approx(-76.4, abs=0.2)
    assert len(mesh.faces) <= 2_500


def test_tile_surface_height_ignores_parts_under_the_tiles():
    trimesh = pytest.importorskip("trimesh")
    from talongym.assets.import_field_cad import tile_surface_height

    bar = trimesh.creation.box(extents=[48.0, 1.0, 2.0])
    bar.apply_translation([0.0, 0.5, 0.0])
    tiles = trimesh.creation.box(extents=[144.0, 0.6, 144.0])
    tiles.apply_translation([0.0, 1.3, 0.0])
    mesh = trimesh.util.concatenate([bar, tiles])
    assert tile_surface_height(mesh) == pytest.approx(1.6, abs=0.01)
