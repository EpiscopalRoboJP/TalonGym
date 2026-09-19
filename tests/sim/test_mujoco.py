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


def test_compile_applies_constraint_budget():
    if not available():
        pytest.skip("mujoco extra not installed")
    from talongym.assets.mjcf_field import MJCF_NCONMAX
    from talongym.sim.mujoco_backend import _compile_mj_model

    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <geom type="plane" size="2 2 0.1"/>
        <body pos="0 1 0"><freejoint/><geom type="sphere" size="0.2" mass="1"/></body>
      </worldbody>
    </mujoco>
    """
    _mj, model = _compile_mj_model(xml, None)
    assert int(model.nconmax) >= MJCF_NCONMAX


def test_nefc_overflow_rewinds_and_flags_wall():
    if not available():
        pytest.skip("mujoco extra not installed")
    from talongym.sim.mujoco_backend import MujocoFieldBackend, _is_constraint_overflow
    from talongym.sim.physics import Body, WorldStep, perimeter_walls

    assert _is_constraint_overflow(
        Exception("mj_makeConstraint: nefc mis-allocation: found nefc=1490 but allocated 1488")
    )
    xml = """
    <mujoco model="piece_flow">
      <compiler angle="radian" inertiafromgeom="true"/>
      <option gravity="0 -386.0886 0" timestep="0.002" integrator="implicitfast"/>
      <worldbody>
        <geom name="floor" type="plane" size="80 80 1" pos="0 0 0" zaxis="0 1 0" group="0"/>
        <body name="red_0" pos="0 7 0">
          <joint name="red_0_sx" type="slide" axis="1 0 0" damping="2"/>
          <joint name="red_0_sz" type="slide" axis="0 0 1" damping="2"/>
          <joint name="red_0_yaw" type="hinge" axis="0 1 0" damping="0.4"/>
          <geom name="red_0_chassis" type="box" size="9 0.5 9" mass="15" group="1"/>
        </body>
        <body name="gp_pollen_00" pos="0 80 0">
          <freejoint name="gp_pollen_00_free"/>
          <geom name="gp_pollen_00_geom" type="sphere" size="1.4" mass="0.1" group="2"/>
        </body>
      </worldbody>
    </mujoco>
    """
    backend = MujocoFieldBackend(xml, 72.0, 72.0, robot_hz=7.0, slot_plan={"pollen": 1}, floor_y=0.0)
    real_step = backend._mujoco.mj_step

    def boom(model, data):
        raise backend._mujoco.FatalError(
            "mj_makeConstraint: nefc mis-allocation: found nefc=1490 but allocated 1488"
        )

    backend._mujoco.mj_step = boom
    try:
        robot = Body("red_0", 0.0, 0.0, 0.0, mass=15.0, hx=9.0, hy=9.0)
        piece = Body("p0", 4.0, 0.0, 0.0, kind="circle", radius=1.4, mass=0.1, z=10.0, type_id="pollen")
        flags = backend.step_world(
            WorldStep([robot], [piece], [], perimeter_walls(72, 72)),
            0.02,
        )
    finally:
        backend._mujoco.mj_step = real_step
    assert flags.wall
    assert np.isfinite(robot.x)
    assert np.isfinite(piece.x)


def test_ftc_yaw_quat_roundtrip():
    from talongym.sim.mujoco_backend import ftc_yaw_to_mj_quat, mj_quat_to_ftc_yaw

    for heading in (0.0, 0.7, -1.2, 3.14):
        q = ftc_yaw_to_mj_quat(heading)
        got = mj_quat_to_ftc_yaw(*q)
        assert abs(((got - heading + 3.14159) % 6.28318) - 3.14159) < 1e-6
