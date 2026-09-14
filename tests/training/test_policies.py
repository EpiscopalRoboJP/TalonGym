import math

import numpy as np

from talongym.training.policies import scripted_biobuzz


def _idle_obs(x: float, y: float) -> dict[str, np.ndarray]:
    return {
        "pose_noisy": np.array([x, y, 0.0], dtype=np.float32),
        "held_count": np.array([0.0], dtype=np.float32),
        "nearest_pieces": np.zeros((6, 6), dtype=np.float32),
        "time_remaining_s": np.array([20.0], dtype=np.float32),
    }


def test_scripted_biobuzz_targets_red_cell_by_default():
    action = scripted_biobuzz(_idle_obs(-60.0, -24.0), info=None)
    x, y, heading = (float(v) for v in action["target_pose"])
    assert x < 0 and y < 0
    assert math.isclose(heading, -math.pi / 2, abs_tol=1e-6)


def test_scripted_biobuzz_mirrors_targets_for_blue():
    """The opponent alliance drives to its own side of the field, not red's."""
    red_action = scripted_biobuzz(_idle_obs(-60.0, -24.0), info={"alliance": "red"})
    blue_action = scripted_biobuzz(_idle_obs(60.0, 24.0), info={"alliance": "blue"})
    rx, ry, rh = (float(v) for v in red_action["target_pose"])
    bx, by, bh = (float(v) for v in blue_action["target_pose"])
    assert (bx, by) == (-rx, -ry)
    assert math.isclose((bh - rh - math.pi + math.pi) % (2 * math.pi) - math.pi, 0.0, abs_tol=1e-6)
    assert bx > 0 and by > 0
