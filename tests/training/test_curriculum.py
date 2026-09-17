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


def test_biobuzz_curriculum_switches_with_progress():
    training = load_preset("training", "biobuzz_auto_lightweight")
    early = curriculum_unlocks(training, 0.0)
    mid = curriculum_unlocks(training, 0.5)
    late = curriculum_unlocks(training, 0.99)
    assert "scripted_launch" not in early
    assert "ballistic_launch" not in early
    assert "scripted_launch" not in mid
    assert "ballistic_launch" not in mid
    assert curriculum_spawn(training, 0.0) == "launch"
    assert curriculum_spawn(training, 0.3) == "approach"
    assert curriculum_spawn(training, 0.5) == "legal"
    assert curriculum_spawn(training, 0.99) == "legal"
    assert early == ["spawn_at_launch", "mechanism_ready"]
    assert "spawn_approach" in curriculum_unlocks(training, 0.3)
    assert stage_info(training, 0.5)["unlock"] == []
    assert "full_noise" in late
    assert mechanism_ready(training, 0.0) is True
    assert mechanism_ready(training, 0.99) is False
    assert full_noise(training, 0.0) is False
    assert full_noise(training, 0.99) is True
    assert teammate_for(training, 0.0) == "none"
    assert opponent_for(training, 0.0) == "static"
    info = stage_info(training, 0.1)
    assert info["index"] == 0
    assert stage_info(training, 0.3)["index"] == 1
    assert stage_info(training, 0.5)["index"] == 2
    assert stage_info(training, 0.99)["index"] == 3


def test_resolved_action_tier_honors_robot_default():
    assert resolved_action_tier({"actionTier": "physical_actuators"}, {"defaultActionTier": "high_level_waypoint"}) == "physical_actuators"
    assert resolved_action_tier({}, {"defaultActionTier": "physical_actuators"}) == "physical_actuators"
    assert resolved_action_tier(None, None) == "high_level_waypoint"
