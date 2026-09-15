from talongym.presets.loader import load_preset
from talongym.robot.drivetrain import clip_twist, mecanum_module_speeds


def test_tank_drops_strafe_and_clip():
    robot = load_preset("robot", "mecanum_meepmeep_defaults")
    vx, vy, om = clip_twist(100.0, 50.0, 10.0, robot)
    assert abs(vx) <= 30.1
    assert abs(vy) > 0
    tank = dict(robot)
    tank["drivetrain"] = dict(robot["drivetrain"], type="tank")
    vx2, vy2, _ = clip_twist(10.0, 10.0, 0.0, tank)
    assert vy2 == 0.0
    speeds = mecanum_module_speeds(10.0, 0.0, 0.0, robot)
    assert speeds.shape == (4,)
