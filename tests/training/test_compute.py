from talongym.presets.loader import load_preset, preset_index
from talongym.training.compute import (
    ENV_DEVICE,
    ENV_EVAL_WORKERS,
    ENV_SIM_WORKERS,
    _arch_supports_capability,
    detect_compute_profile,
    easy_training_id,
    recommended_eval_workers,
    recommended_n_envs,
    recommended_sim_workers,
    resolve_torch_device,
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
    assert training_family("biobuzz_auto_lightweight") == "biobuzz_auto"
    assert training_id_for_profile("biobuzz_auto_workstation", "auto") == "biobuzz_auto_easy"
    assert easy_training_id("biobuzz_auto_cloud") == "biobuzz_auto_easy"
    idx = preset_index(refresh=True)["training"]
    for suffix in ("lightweight", "workstation", "cloud", "easy"):
        assert f"biobuzz_auto_{suffix}" in idx


def test_easy_preset_autodetects_n_envs(monkeypatch):
    monkeypatch.delenv("TALONGYM_COMPUTE_PROFILE", raising=False)
    training = load_preset("training", "biobuzz_auto_easy")
    assert training["computeProfile"] == "auto"
    resolved = resolve_training(training, hardware={"cpuCount": 16, "ramGb": 32.0, "cuda": False})
    assert resolved["computeProfile"] == "workstation"
    assert resolved["nEnvs"] == 64
    assert resolved["requestedComputeProfile"] == "auto"


def test_explicit_workstation_keeps_json_n_envs():
    training = load_preset("training", "biobuzz_auto_workstation")
    resolved = resolve_training(training, hardware={"cpuCount": 4, "ramGb": 8.0, "cuda": False})
    assert resolved["computeProfile"] == "workstation"
    assert resolved["nEnvs"] == 256


def test_apple_silicon_mps_counts_as_an_accelerator_like_cuda():
    """A laptop with Metal but no CUDA should scale up the same way a CUDA laptop does."""
    cuda_laptop = {"cpuCount": 8, "ramGb": 8.0, "cuda": True, "mps": False}
    mps_laptop = {"cpuCount": 8, "ramGb": 8.0, "cuda": False, "mps": True}
    neither = {"cpuCount": 8, "ramGb": 8.0, "cuda": False, "mps": False}
    assert detect_compute_profile(mps_laptop) == detect_compute_profile(cuda_laptop) == "workstation"
    assert detect_compute_profile(neither) == "lightweight_cpu"


def test_resolve_torch_device_prefers_explicit_and_env_override(monkeypatch):
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    assert resolve_torch_device("cpu") == "cpu"
    monkeypatch.setenv(ENV_DEVICE, "mps")
    assert resolve_torch_device() == "mps"
    assert resolve_torch_device("cuda:1") == "cuda:1"


def test_resolve_torch_device_falls_back_to_cpu_without_an_accelerator(monkeypatch):
    monkeypatch.delenv(ENV_DEVICE, raising=False)
    monkeypatch.setattr("talongym.training.compute.cuda_available", lambda: False)
    monkeypatch.setattr("talongym.training.compute.mps_available", lambda: False)
    assert resolve_torch_device() == "cpu"


def test_cuda_arch_check_rejects_gpus_the_torch_build_has_no_kernels_for():
    cu130 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
    assert _arch_supports_capability(cu130, (12, 0))
    assert _arch_supports_capability(cu130, (8, 9))
    assert not _arch_supports_capability(cu130, (6, 1))
    assert not _arch_supports_capability(["sm_50", "sm_60", "sm_86"], (12, 0))
    assert _arch_supports_capability([], (6, 1))


def test_sim_workers_leave_a_core_and_never_exceed_n_envs(monkeypatch):
    monkeypatch.delenv(ENV_SIM_WORKERS, raising=False)
    assert recommended_sim_workers(1, {"cpuCount": 16}) == 1
    assert recommended_sim_workers(8, {"cpuCount": 8}) == 7
    assert recommended_sim_workers(256, {"cpuCount": 16}) == 8
    assert recommended_sim_workers(4, {"cpuCount": 1}) == 1
    assert recommended_sim_workers(256, {"cpuCount": 48, "ramGb": 12.0}) == 3


def test_sim_and_eval_worker_env_overrides(monkeypatch):
    monkeypatch.setenv(ENV_SIM_WORKERS, "3")
    monkeypatch.delenv(ENV_EVAL_WORKERS, raising=False)
    assert recommended_sim_workers(256, {"cpuCount": 48}) == 3
    assert recommended_eval_workers(500, {"cpuCount": 48}) == 8
    monkeypatch.setenv(ENV_EVAL_WORKERS, "2")
    assert recommended_eval_workers(500, {"cpuCount": 48}) == 2
    assert recommended_sim_workers(256, {"cpuCount": 48}) == 3


def test_describe_compute_includes_sim_workers(monkeypatch):
    monkeypatch.delenv(ENV_SIM_WORKERS, raising=False)
    monkeypatch.delenv(ENV_EVAL_WORKERS, raising=False)
    from talongym.training.compute import describe_compute

    info = describe_compute("biobuzz_auto_lightweight")
    assert info["simWorkers"] >= 1
    assert info["simWorkers"] <= info["nEnvs"]
    assert info["evalWorkers"] >= 1
    assert info["simWorkersEnvVar"] == ENV_SIM_WORKERS
    assert info["evalWorkersEnvVar"] == ENV_EVAL_WORKERS
