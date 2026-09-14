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
    from talongym.cli import app
    from typer.testing import CliRunner

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
