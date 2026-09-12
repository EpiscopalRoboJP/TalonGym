from talongym.presets.loader import load_preset, preset_index
from talongym.training.compute import (
    detect_compute_profile,
    easy_training_id,
    recommended_n_envs,
    resolve_training,
    training_family,
    training_id_for_profile,
)


def test_profile_from_hardware():
    laptop = {"cpuCount": 8, "ramGb": 8.0, "cuda": False, "platform": "win32"}
    desk = {"cpuCount": 16, "ramGb": 32.0, "cuda": False, "platform": "linux"}
    box = {"cpuCount": 48, "ramGb": 128.0, "cuda": True, "platform": "linux"}
    assert detect_compute_profile(laptop) == "lightweight_cpu"
    assert detect_compute_profile(desk) == "workstation"
    assert detect_compute_profile(box) == "cloud"
    assert recommended_n_envs("lightweight_cpu", laptop) == 8
    assert recommended_n_envs("workstation", desk) == 64
    assert recommended_n_envs("cloud", box) == 384


def test_env_override(monkeypatch):
    monkeypatch.setenv("TALONGYM_COMPUTE_PROFILE", "cloud")
    assert detect_compute_profile({"cpuCount": 2, "ramGb": 4.0, "cuda": False}) == "cloud"


def test_training_family_and_easy_ids():
    assert training_family("decode_auto_lightweight") == "decode_auto"
    assert training_id_for_profile("biobuzz_auto_workstation", "auto") == "biobuzz_auto_easy"
    assert easy_training_id("centerstage_auto_cloud") == "centerstage_auto_easy"
    idx = preset_index(refresh=True)["training"]
    for season in ("decode_auto", "into_the_deep_auto", "centerstage_auto", "biobuzz_auto"):
        for suffix in ("lightweight", "workstation", "cloud", "easy"):
            assert f"{season}_{suffix}" in idx


def test_easy_preset_autodetects_n_envs(monkeypatch):
    monkeypatch.delenv("TALONGYM_COMPUTE_PROFILE", raising=False)
    training = load_preset("training", "decode_auto_easy")
    assert training["computeProfile"] == "auto"
    resolved = resolve_training(training, hardware={"cpuCount": 16, "ramGb": 32.0, "cuda": False})
    assert resolved["computeProfile"] == "workstation"
    assert resolved["nEnvs"] == 64
    assert resolved["requestedComputeProfile"] == "auto"


def test_explicit_workstation_keeps_json_n_envs():
    training = load_preset("training", "decode_auto_workstation")
    resolved = resolve_training(training, hardware={"cpuCount": 4, "ramGb": 8.0, "cuda": False})
    assert resolved["computeProfile"] == "workstation"
    assert resolved["nEnvs"] == 256
