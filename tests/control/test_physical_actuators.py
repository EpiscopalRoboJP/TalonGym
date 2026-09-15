import numpy as np

from talongym.env.ftc_auto import BoxActionDictObsEnv, FTCAutoEnv


def test_high_level_action_space_preserved():
    env = FTCAutoEnv(record=False, static_teammate=False)
    assert "target_pose" in env.action_space.spaces
    assert "mechanism" in env.action_space.spaces
    assert "drive" not in env.action_space.spaces
    obs, info = env.reset(seed=1)
    assert "mechanism_sensors" in obs
    assert obs["mechanism_sensors"].shape[0] >= 1
    assert "mechanism_sensors" in info
    env.close()


def test_physical_actuators_tier_uses_world_action_dict():
    env = FTCAutoEnv(action_tier="physical_actuators", record=True, static_teammate=False)
    obs, info = env.reset(seed=2)
    assert "drive" in env.action_space.spaces
    assert "actuators" in env.action_space.spaces
    n_act = int(env.action_space["actuators"].shape[0])
    assert n_act >= 1
    parsed = env._parse_action(
        {
            "drive": np.array([0.2, 0.0, 0.0], dtype=np.float32),
            "actuators": np.zeros(n_act, dtype=np.float32),
        }
    )
    assert parsed is not None
    assert "velocity" in parsed
    assert "actuators" in parsed
    high_level = env._parse_action(
        {
            "target_pose": obs["pose_noisy"],
            "speed_frac": np.array([0.7], dtype=np.float32),
            "mechanism": 1,
        }
    )
    assert high_level is not None
    assert "target_pose" in high_level
    assert high_level["mechanism"] == 1
    mixed = env._parse_action(
        {
            "target_pose": obs["pose_noisy"],
            "speed_frac": np.array([0.5], dtype=np.float32),
            "mechanism": 2,
            "actuators": {"flywheel": 1.0, "gate": 1.0},
        }
    )
    assert mixed is not None
    assert mixed["actuators"]["flywheel"] == 1.0
    obs, _rew, terminated, _trunc, info = env.step(mixed)
    assert terminated is False
    assert "mechanism_sensors" in obs
    assert "batteryVoltageV" in info["privileged"]
    assert "actuators" in info["privileged"]
    frame = env.frames[-1]
    assert frame["physicalPieces"] is True
    assert frame["robots"][0]["batteryVoltageV"] > 0
    assert "parts" in frame["robots"][0]
    assert "mechanismSensors" in frame
    env.close()


def test_physical_box_wrapper_matches_actuator_count():
    env = FTCAutoEnv(action_tier="physical_actuators", record=False, static_teammate=False)
    boxed = BoxActionDictObsEnv(env)
    n_act = max(1, len(env._actuator_ids))
    assert boxed.action_space.shape == (3 + n_act,)
    obs, _ = boxed.reset(seed=3)
    assert "mechanism_sensors" in obs
    next_obs, _rew, terminated, _trunc, _info = boxed.step(boxed.action_space.sample())
    assert terminated in {True, False}
    assert next_obs["mechanism_sensors"].shape[0] == env._sensor_bank.size
    env.close()


def test_mechanism_sensor_defaults_without_backend_fields():
    env = FTCAutoEnv(record=True, static_teammate=False)
    obs, info = env.reset(seed=4)
    sensors = obs["mechanism_sensors"]
    assert np.all(np.isfinite(sensors))
    names = env._sensor_bank.as_dict(sensors)
    if names:
        assert "battery_voltage" in names or any("battery" in key for key in names)
    privileged = info["privileged"]
    assert float(privileged.get("batteryVoltageV") or 0) > 0
    env.step(
        {
            "target_pose": obs["pose_noisy"],
            "speed_frac": np.array([0.4], dtype=np.float32),
            "mechanism": 0,
        }
    )
    frame = env.frames[-1]
    assert frame["robots"][0].get("lastVerb") in {"idle", "intake", "score", "open_gate", "stow"}
    env.close()
