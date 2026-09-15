from talongym.presets.loader import load_preset
from talongym.training.curriculum import (
    curriculum_unlocks,
    full_noise,
    opponent_for,
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
    assert "full_noise" in late
    assert full_noise(training, 0.0) is False
    assert full_noise(training, 0.99) is True
    assert teammate_for(training, 0.0) == "none"
    assert opponent_for(training, 0.0) == "static"
    info = stage_info(training, 0.1)
    assert info["index"] == 0
    assert stage_info(training, 0.5)["index"] == 1
    assert stage_info(training, 0.99)["index"] == 2
