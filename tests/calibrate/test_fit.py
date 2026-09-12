from talongym.calibrate import fit_from_log, overlay_robot
from talongym.presets.loader import load_preset


def test_fit_recovers_speed_order():
    samples = []
    x = 0.0
    for i in range(20):
        samples.append({"t": i * 0.1, "x": x, "y": 0.0, "headingDeg": 0.0})
        x += 2.5  # 25 in/s
    fit = fit_from_log(samples)
    assert 10 < fit["maxVelInPerS"] < 40
    robot = load_preset("robot", "mecanum_meepmeep_defaults")
    overlay = overlay_robot(robot, fit)
    assert overlay["constraints"]["maxVelInPerS"] == fit["maxVelInPerS"]
    assert overlay["calibration"]["rmse"] >= 0
