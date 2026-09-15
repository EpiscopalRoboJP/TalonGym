import math

from talongym.export.roadrunner import to_roadrunner_java


def test_rr1_contains_action_builder():
    java = to_roadrunner_java([[0.0, 0.0, 0.0], [24.0, 12.0, 1.57]])
    assert "actionBuilder" in java
    assert "splineTo" in java
    assert "TrajectorySequenceBuilder" not in java


def test_legacy_dialect():
    java = to_roadrunner_java([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dialect="rr05_trajectory_sequence")
    assert "trajectorySequenceBuilder" in java


def test_rr1_ends_facing_the_policy_heading_not_the_path_tangent():
    """splineTo's 2nd argument is the path tangent, which only equals the robot's
    final heading on a straight run. Here the robot drives straight in +x (tangent 0)
    but must arrive facing +y (e.g. to score sideways); the export must keep that
    final heading distinct from the tangent via splineToLinearHeading, not collapse
    the two the way a bare splineTo(Vector2d, tangent) call would."""
    waypoints = [[0.0, 0.0, 0.0], [24.0, 0.0, math.pi / 2]]
    java = to_roadrunner_java(waypoints)
    assert "splineToLinearHeading" in java
    assert "new Pose2d(24.00, 0.00, 1.5708)" in java  # final heading preserved
    assert ", 0.0000)" in java  # tangent (direction of travel) stays separate, near 0
