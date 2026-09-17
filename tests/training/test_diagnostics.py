"""Training health gates: distinct curriculum stages, baseline launch, unhealthy checkpoints."""

from __future__ import annotations

import pytest

from talongym.presets.loader import load_preset
from talongym.training.curriculum import curriculum_spawn, stage_info
from talongym.training.diagnostics import EpisodeHealth, summarize_episode


def test_curriculum_stages_are_distinct_physical_scaffolds():
    training = load_preset("training", "biobuzz_auto_lightweight")
    stages = [stage_info(training, frac) for frac in (0.0, 0.3, 0.6, 0.99)]
    unlocks = [tuple(stage["unlock"]) for stage in stages]
    assert len(set(unlocks)) == 4
    assert curriculum_spawn(training, 0.0) == "launch"
    assert curriculum_spawn(training, 0.3) == "approach"
    assert curriculum_spawn(training, 0.6) == "legal"
    assert "mechanism_ready" in stages[0]["unlock"]
    assert "full_noise" in stages[-1]["unlock"]


def test_summarize_episode_marks_zero_launch_policies_unhealthy():
    frames = [
        {
            "trueScore": 0,
            "launchAttempts": 0,
            "robots": [{"id": "red_0", "x": 1, "y": 1, "headingDeg": 90, "lastVerb": "idle", "held": ["a", "b"], "collisionTimeS": 12.0}],
            "pieces": [{"id": "a", "heldBy": "red_0"}, {"id": "b", "heldBy": "red_0"}],
            "collision": {"wall": True, "collisionTimeS": 12.0},
        }
    ]
    health = summarize_episode(frames)
    assert health.launches == 0
    assert health.healthy is False
    assert any("zero physical launches" in warning for warning in health.warnings)
    assert any("wall contact" in warning for warning in health.warnings)
    metrics = health.as_metrics()
    assert metrics["evalLaunchCount"] == 0
    assert metrics["evalCheckpointHealthy"] is False


def test_summarize_episode_accepts_physical_launch_and_score():
    frames = [
        {
            "trueScore": 8,
            "launchAttempts": 2,
            "t": 0.0,
            "robots": [{"id": "red_0", "lastVerb": "score", "held": [], "collisionTimeS": 0.2}],
            "pieces": [
                {"id": "p1", "launchedBy": "red_0", "scored": True},
                {"id": "p2", "inFlight": True},
            ],
        }
    ]
    health = summarize_episode(frames)
    assert health.launches >= 1
    assert health.true_score == 8
    assert health.healthy is True


def test_summarize_episode_counts_wall_contact_duration_from_frames():
    frames = [
        {"t": 0.0, "trueScore": 3, "launchAttempts": 1, "robots": [{"id": "red_0", "lastVerb": "score", "collisionTimeS": 0.0}], "pieces": [{"id": "p1", "launchedBy": "red_0"}], "collision": {"wall": False}},
        {"t": 1.0, "trueScore": 3, "launchAttempts": 1, "robots": [{"id": "red_0", "lastVerb": "score", "collisionTimeS": 0.0}], "pieces": [{"id": "p1", "launchedBy": "red_0"}], "collision": {"wall": True}},
        {"t": 10.0, "trueScore": 3, "launchAttempts": 1, "robots": [{"id": "red_0", "lastVerb": "score", "collisionTimeS": 0.0}], "pieces": [{"id": "p1", "launchedBy": "red_0"}], "collision": {"wall": True}},
    ]
    health = summarize_episode(frames)
    assert health.wall_contact_s >= 9.0
    assert health.healthy is False
    assert any("wall contact" in warning for warning in health.warnings)


@pytest.mark.require_mesh
def test_scripted_baseline_launches_and_scores_from_launch_pose():
    from talongym.training.diagnostics import assert_scripted_baseline_scores

    health = assert_scripted_baseline_scores(seed=1)
    assert health.launches >= 3
    assert health.true_score >= 3 or health.scored_pieces >= 1


def test_scripted_baseline_stops_once_thresholds_met(monkeypatch):
    steps = {"n": 0}

    class FakeEnv:
        def __init__(self, **kwargs):
            self.frames = []

        def reset(self, seed=None, options=None):
            return {}, {}

        def step(self, action):
            steps["n"] += 1
            n = steps["n"]
            launches = min(n, 4)
            scored = 1 if n >= 4 else 0
            self.frames.append(
                {
                    "trueScore": 8 if n >= 4 else 0,
                    "launchAttempts": launches,
                    "t": n * 0.04,
                    "robots": [{"id": "red_0", "lastVerb": "score", "held": [], "collisionTimeS": 0.0}],
                    "pieces": [
                        {"id": f"p{i}", "launchedBy": "red_0", "scored": i < scored}
                        for i in range(launches)
                    ],
                }
            )
            return {}, 0.0, False, False, {}

        def close(self):
            pass

    monkeypatch.setattr("talongym.env.ftc_auto.FTCAutoEnv", FakeEnv)
    monkeypatch.setattr("talongym.training.diagnostics.scripted_auto", lambda obs, info: {})
    from talongym.training.diagnostics import assert_scripted_baseline_scores

    health = assert_scripted_baseline_scores()
    assert health.launches >= 3
    assert health.true_score >= 3 or health.scored_pieces >= 1
    assert steps["n"] == 4
    assert steps["n"] < 50
