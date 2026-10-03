"""Source-backed dimensions and deliberately uncalibrated shot parameters."""

from __future__ import annotations

import json

import pytest

from talongym.paths import ASSETS_DIR
from talongym.presets.loader import validate_document
from talongym.robot.contract import compile_robot_preset
from talongym.robot.gobilda_reference import OUTPUT, build_document
from talongym.sim.world import _ready_rpm


def test_reference_preset_matches_generated_source_and_cad_envelope():
    document = build_document()
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == document
    assert validate_document("robot", document) == []
    assert compile_robot_preset(document, competitive=True).physical
    manifest = json.loads((ASSETS_DIR / "robots/gobilda_biobuzz_reference/cad_manifest.json").read_text())
    bounds = manifest["visualBoundsIn"]
    assert bounds[0][1] == pytest.approx(0.0, abs=0.01)
    assert bounds[1][1] == pytest.approx(document["chassis"]["heightIn"], abs=0.02)
    assert bounds[1][0] - bounds[0][0] == pytest.approx(document["chassis"]["lengthIn"], abs=0.02)
    assert bounds[1][2] - bounds[0][2] == pytest.approx(document["chassis"]["widthIn"], abs=0.02)


def test_reference_drivetrain_and_launcher_orientation():
    document = build_document()
    parts = {part["id"]: part for part in document["rigidParts"]}
    assert parts["wheel_fl"]["pose"]["x"] > parts["wheel_rl"]["pose"]["x"]
    assert parts["wheel_fl"]["pose"]["y"] > parts["wheel_fr"]["pose"]["y"]
    assert parts["wheel_fl"]["pose"]["z"] == pytest.approx(parts["wheel_rr"]["pose"]["z"])
    assert document["drivetrain"]["trackWidthIn"] == pytest.approx(414 / 25.4, abs=1e-4)
    assert document["drivetrain"]["wheelbaseIn"] == pytest.approx(264 / 25.4, abs=1e-4)
    assert document["piecePath"]["muzzlePose"]["yawDeg"] == 180
    assert document["piecePath"]["muzzlePose"]["x"] < parts["flywheel"]["pose"]["x"]
    flywheel = next(row for row in document["actuators"] if row["id"] == "flywheel")
    assert flywheel["targetRpm"] == pytest.approx(1250 * 60 / 28)
    assert flywheel["readyRpm"] == pytest.approx(1200 * 60 / 28)
    assert _ready_rpm(flywheel) == pytest.approx(flywheel["readyRpm"])
    assert _ready_rpm({"targetRpm": 1000}) == pytest.approx(800)
    assert any(row["code"] == "launcher_contact_unverified" for row in document["warnings"])
