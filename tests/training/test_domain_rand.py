import pytest

from talongym.presets.loader import LoadedPresets, load_bundle
from talongym.sim.world import World


def _bundle_with_dr(**kwargs):
    bundle = load_bundle(
        "biobuzz_2026_field_v1",
        "mecanum_biobuzz_4cap",
        "biobuzz_2026_scoring_v1",
        "biobuzz_auto_lightweight",
    )
    dr = dict(bundle.training.get("domainRandomization") or {})
    dr.update(kwargs)
    training = dict(bundle.training)
    training["domainRandomization"] = dr
    return LoadedPresets(field=bundle.field, robot=bundle.robot, scoring=bundle.scoring, training=training)


def test_zero_jitter_keeps_start_slot():
    bundle = _bundle_with_dr(poseJitterIn=0.0, headingJitterDeg=0.0, fieldBuildToleranceIn=0.0)
    world = World(bundle, seed=0)
    world.reset(seed=0)
    slot = world._start_slot("red", 0)
    assert slot is not None
    body = world.actor().body
    assert body.y == float(slot["pose"]["y"])
    assert world._start_pose_error(body.x, body.y, body.heading, "red") is None
    assert abs(abs(body.x) - world.playable_half_w + world.robot_hx) < 0.2


def test_pose_jitter_moves_spawn():
    bundle = _bundle_with_dr(poseJitterIn=8.0, headingJitterDeg=0.0, fieldBuildToleranceIn=0.0)
    world = World(bundle, seed=0)
    world.reset(seed=0)
    slot = world._start_slot("red", 0)
    dx = abs(world.actor().body.x - float(slot["pose"]["x"]))
    dy = abs(world.actor().body.y - float(slot["pose"]["y"]))
    assert dx + dy > 0.05
    flower_spawn = next(spawn for spawn in bundle.field["spawns"] if spawn["id"] == "flower_1_pollen")
    staged = next(piece for piece in world.pieces.values() if piece.type_id == "pollen" and not piece.held_by)
    assert staged.x == pytest.approx(float(flower_spawn["poses"][0]["x"]), abs=0.02)
    assert staged.y == pytest.approx(float(flower_spawn["poses"][0]["y"]), abs=0.02)


def test_motor_strength_and_noise_scale():
    bundle = _bundle_with_dr(motorStrengthRange=[0.5, 0.5], sensorNoiseScale=2.0)
    world = World(bundle, seed=1)
    world.reset(seed=1, full_noise=True)
    assert abs(world.motor_strength - 0.5) < 1e-9
    assert world.pos_noise == world._base_pos_noise * 2.0
    world.reset(seed=1, full_noise=False)
    assert world.pos_noise == world._base_pos_noise * 2.0 * 0.15
