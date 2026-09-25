import math

import numpy as np

from talongym.training.policies import LAUNCH_POSES, PARK_POSES, scripted_biobuzz


def _obs(x: float, y: float, heading: float = 0.0, held: float = 0.0, t_left: float = 20.0) -> dict[str, np.ndarray]:
    return {
        "pose_noisy": np.array([x, y, heading], dtype=np.float32),
        "held_count": np.array([held], dtype=np.float32),
        "nearest_pieces": np.zeros((6, 6), dtype=np.float32),
        "time_remaining_s": np.array([t_left], dtype=np.float32),
        "teammate_pose_noisy": np.zeros(3, dtype=np.float32),
    }


def _target(action: dict) -> tuple[float, float, float]:
    x, y, h = (float(v) for v in action["target_pose"])
    return x, y, h


def test_empty_robot_parks_in_red_loading_zone_by_default():
    x, y, heading = _target(scripted_biobuzz(_obs(-56.0, -24.0), info=None))
    assert (x, y) == PARK_POSES[0][:2]
    # The chassis overlaps red_park while its center stays clear of the wall.
    assert (x, y) == (-52.5, 23.0)
    assert math.isclose(heading, 0.0, abs_tol=1e-6)


def test_scripted_biobuzz_mirrors_targets_for_blue():
    """The opponent alliance drives to its own side of the field, not red's."""
    rx, ry, rh = _target(scripted_biobuzz(_obs(-56.0, -24.0), info={"alliance": "red"}))
    bx, by, bh = _target(scripted_biobuzz(_obs(56.0, 24.0), info={"alliance": "blue"}))
    assert (bx, by) == (-rx, -ry)
    assert math.isclose((bh - rh - math.pi + math.pi) % (2 * math.pi) - math.pi, 0.0, abs_tol=1e-6)
    assert bx > 0 and by < 0


def test_loaded_robot_fires_only_on_launch_spot():
    lx, ly, lh = LAUNCH_POSES[0]
    away = scripted_biobuzz(_obs(-56.0, -24.0, held=4), info={"robot_id": "red_0"})
    assert _target(away)[:2] == (lx, ly) and away["mechanism"] == 0
    there = scripted_biobuzz(_obs(lx, ly, lh, held=4), info={"robot_id": "red_0"})
    assert there["mechanism"] == 2


def test_loaded_robot_leaves_for_park_when_time_runs_out():
    lx, ly, lh = LAUNCH_POSES[0]
    action = scripted_biobuzz(_obs(lx, ly, lh, held=2, t_left=3.0), info={"robot_id": "red_0"})
    assert action["mechanism"] == 0
    assert _target(action)[:2] != (lx, ly)


def test_takes_free_park_when_teammate_holds_preferred_one():
    obs = _obs(-50.0, 60.0)
    obs["teammate_pose_noisy"] = np.array([PARK_POSES[0][0], PARK_POSES[0][1], 0.0], dtype=np.float32)
    px, py, _ = _target(scripted_biobuzz(obs, info={"robot_id": "red_0"}))
    assert (px, py) == PARK_POSES[1][:2]
