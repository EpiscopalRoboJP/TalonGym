from talongym.presets.loader import load_preset
from talongym.training.curriculum import (
    curriculum_spawn,
    curriculum_unlocks,
    full_noise,
    mechanism_ready,
    opponent_for,
    resolved_action_tier,
    stage_info,
    teammate_for,
)


def test_biobuzz_curriculum_is_legal_spawn_with_full_noise():
    training = load_preset("training", "biobuzz_auto_lightweight")
    assert curriculum_unlocks(training, 0.0) == []
    assert curriculum_unlocks(training, 0.99) == []
    assert curriculum_spawn(training, 0.0) == "legal"
    assert curriculum_spawn(training, 0.99) == "legal"
    assert mechanism_ready(training, 0.0) is False
    assert mechanism_ready(training, 0.99) is False
    assert full_noise(training, 0.0) is True
    assert full_noise(training, 0.99) is True
    assert teammate_for(training, 0.0) == "none"
    assert opponent_for(training, 0.0) == "static"
    assert stage_info(training, 0.0)["unlock"] == []
    assert stage_info(training, 0.99)["index"] == 0


def test_empty_curriculum_defaults_full_noise():
    assert full_noise({"domainRandomization": {"curriculum": []}}, 0.0) is True
    assert full_noise({}, 0.5) is True
    staged = {"domainRandomization": {"curriculum": [{"untilFrac": 1.0, "unlock": []}]}}
    assert full_noise(staged, 0.5) is False


def test_resolved_action_tier_honors_robot_default():
    assert resolved_action_tier({"actionTier": "physical_actuators"}, {"defaultActionTier": "high_level_waypoint"}) == "physical_actuators"
    assert resolved_action_tier({}, {"defaultActionTier": "physical_actuators"}) == "physical_actuators"
    assert resolved_action_tier(None, None) == "high_level_waypoint"
