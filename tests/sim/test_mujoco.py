import numpy as np
import pytest

from talongym.sim.mujoco_backend import available, pose_rmse


def test_mujoco_optional():
    if not available():
        pytest.skip("mujoco extra not installed")
    from talongym.sim.mujoco_backend import MujocoValidationBackend
    from talongym.sim.physics import Body, WorldStep, perimeter_walls

    mj = MujocoValidationBackend(72, 72)
    bot = Body("red_0", 0, 0, 0, vx=5)
    mj.step_world(WorldStep([bot], [], [], perimeter_walls(72, 72)), 0.02)
    assert np.isfinite(bot.x)


def test_pose_rmse():
    assert pose_rmse([(0, 0), (1, 0)], [(0, 0), (1, 0)]) == 0.0


def test_ftc_yaw_quat_roundtrip():
    from talongym.sim.mujoco_backend import ftc_yaw_to_mj_quat, mj_quat_to_ftc_yaw

    for heading in (0.0, 0.7, -1.2, 3.14):
        q = ftc_yaw_to_mj_quat(heading)
        got = mj_quat_to_ftc_yaw(*q)
        assert abs(((got - heading + 3.14159) % 6.28318) - 3.14159) < 1e-6
