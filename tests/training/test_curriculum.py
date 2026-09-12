from talongym.presets.loader import load_preset
from talongym.training.curriculum import (
    curriculum_unlocks,
    full_noise,
    motif_known_at_t0,
    opponent_for,
    stage_info,
    teammate_for,
)


def test_decode_curriculum_switches_with_progress():
    training = load_preset("training", "decode_auto_lightweight")
    early = curriculum_unlocks(training, 0.0)
    late = curriculum_unlocks(training, 0.5)
    assert "motif_known_at_t0" in early
    assert "motif_must_sense" not in early
    assert motif_known_at_t0(training, 0.0) is True
    assert "motif_must_sense" in late
    assert motif_known_at_t0(training, 0.5) is False
    assert full_noise(training, 0.0) is False
    assert full_noise(training, 0.99) is True
    assert teammate_for(training, 0.0) != "scripted"
    assert teammate_for(training, 1.0) == "scripted"
    assert opponent_for(training, 0.0) == "static"
    info = stage_info(training, 0.1)
    assert info["index"] == 0
    assert stage_info(training, 0.9)["index"] == 1
