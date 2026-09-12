from talongym.export.roadrunner import to_roadrunner_java


def test_rr1_contains_action_builder():
    java = to_roadrunner_java([[0.0, 0.0, 0.0], [24.0, 12.0, 1.57]])
    assert "actionBuilder" in java
    assert "splineTo" in java
    assert "TrajectorySequenceBuilder" not in java


def test_legacy_dialect():
    java = to_roadrunner_java([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dialect="rr05_trajectory_sequence")
    assert "trajectorySequenceBuilder" in java
