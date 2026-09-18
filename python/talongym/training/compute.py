"""Detect laptop / workstation / cloud and resolve training-run nEnvs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROFILES = ("lightweight_cpu", "workstation", "cloud")
ENV_PROFILE = "TALONGYM_COMPUTE_PROFILE"
ENV_DEVICE = "TALONGYM_TORCH_DEVICE"
ENV_SIM_WORKERS = "TALONGYM_SIM_WORKERS"
ENV_EVAL_WORKERS = "TALONGYM_EVAL_WORKERS"
# Keep in sync with requirements.txt.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"

PROFILE_N_ENVS = {
    "lightweight_cpu": 8,
    "workstation": 256,
    "cloud": 1024,
}

PROFILE_SUFFIX = {
    "lightweight_cpu": "lightweight",
    "workstation": "workstation",
    "cloud": "cloud",
    "auto": "easy",
}

_FAMILY_SUFFIXES = ("_lightweight", "_workstation", "_cloud", "_easy")


def cpu_count() -> int:
    return max(1, int(os.cpu_count() or 4))


def ram_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
        except (OSError, ValueError, IndexError):
            pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return float(stat.ullTotalPhys) / (1024.0**3)
        except Exception:
            pass
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024.0**3)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def _arch_supports_capability(arch_list: list[str], capability: tuple[int, int]) -> bool:
    """Whether a CUDA torch build compiled for `arch_list` (e.g. ["sm_75", "sm_120"])
    has kernels for a GPU of this compute capability. Kernels built for sm_XY run on
    any sm_XZ with Z >= Y, so only the major version must match. An empty list means
    torch didn't say, so assume it works."""
    sm = [a for a in arch_list if a.startswith("sm_") and a[3:].isdigit() and len(a) > 4]
    if not sm:
        return True
    major, minor = capability
    return any(int(a[3:-1]) == major and int(a[-1]) <= minor for a in sm)


def cuda_available() -> bool:
    """True for NVIDIA CUDA, and also for AMD ROCm builds of torch (ROCm exposes
    itself through the same torch.cuda API). False when torch sees a GPU but was
    built without kernels for it (e.g. a CUDA 13 wheel on a Pascal card), since
    training would crash with "no kernel image is available" instead of using CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        if getattr(torch.version, "hip", None):
            return True
        return _arch_supports_capability(torch.cuda.get_arch_list(), torch.cuda.get_device_capability(0))
    except Exception:
        return False


def nvidia_gpu() -> str | None:
    """Name of the first NVIDIA GPU per nvidia-smi, independent of torch."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return names[0] if names else None


def torch_status() -> dict[str, Any]:
    """Which torch build is installed and, when an NVIDIA GPU goes unused, why."""
    gpu = nvidia_gpu()
    status: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cudaBuild": None,
        "rocmBuild": None,
        "error": None,
        "nvidiaGpu": gpu,
        "hint": None,
    }
    fix = f"Reinstall with: python -m pip install --force-reinstall --no-deps torch --extra-index-url {TORCH_CUDA_INDEX}"
    try:
        import torch
    except ImportError as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        missing = getattr(exc, "name", None) == "torch"
        status["hint"] = (
            "torch is not installed; training needs it. Install with: python -m pip install -r requirements.txt"
            if missing
            else f"torch is installed but broken (likely an interrupted install). {fix}"
        )
        return status
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["hint"] = f"torch failed to import. {fix}"
        return status
    status["installed"] = True
    status["version"] = str(getattr(torch, "__version__", None))
    status["cudaBuild"] = getattr(torch.version, "cuda", None)
    status["rocmBuild"] = getattr(torch.version, "hip", None)
    if gpu and not cuda_available():
        if not status["cudaBuild"]:
            status["hint"] = f"{gpu} found but this torch build is CPU-only. {fix}"
        elif not torch.cuda.is_available():
            status["hint"] = (
                f"{gpu} found but torch (CUDA {status['cudaBuild']}) can't use it; "
                "update the NVIDIA driver to one that supports this CUDA version."
            )
        else:
            status["hint"] = (
                f"{gpu} is too old for this torch build (CUDA {status['cudaBuild']}); training uses CPU. "
                "Install an older CUDA build, e.g. --index-url https://download.pytorch.org/whl/cu126"
            )
    return status


def mps_available() -> bool:
    """True on Apple Silicon with a torch build that has Metal (MPS) support."""
    try:
        import torch

        backend = getattr(torch.backends, "mps", None)
        return bool(backend and backend.is_available())
    except Exception:
        return False


def resolve_torch_device(preferred: str | None = None) -> str:
    """Pick the best torch device for this machine: CUDA/ROCm, then Apple Silicon
    Metal, else CPU. Override with TALONGYM_TORCH_DEVICE=cuda|mps|cpu (or a specific
    device string such as cuda:1); an explicit `preferred` wins over the env var."""
    forced = (preferred or os.environ.get(ENV_DEVICE) or "").strip()
    if forced:
        return forced
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def detect_hardware() -> dict[str, Any]:
    return {
        "cpuCount": cpu_count(),
        "ramGb": ram_gb(),
        "cuda": cuda_available(),
        "mps": mps_available(),
        "torchDevice": resolve_torch_device(),
        "platform": sys.platform,
    }


def detect_compute_profile(hardware: dict[str, Any] | None = None) -> str:
    forced = (os.environ.get(ENV_PROFILE) or "").strip().lower()
    if forced in PROFILES:
        return forced
    hw = hardware or detect_hardware()
    cpu = int(hw.get("cpuCount") or 4)
    ram = hw.get("ramGb")
    ram_f = float(ram) if ram is not None else None
    accelerator = bool(hw.get("cuda") or hw.get("mps"))
    if cpu >= 32 and (accelerator or ram_f is None or ram_f >= 64):
        return "cloud"
    if accelerator or cpu >= 12 or (ram_f is not None and ram_f >= 24):
        return "workstation"
    return "lightweight_cpu"


def recommended_n_envs(profile: str | None = None, hardware: dict[str, Any] | None = None) -> int:
    hw = hardware or detect_hardware()
    profile = profile if profile in PROFILES else detect_compute_profile(hw)
    cpu = int(hw.get("cpuCount") or 4)
    ram = hw.get("ramGb")
    if profile == "lightweight_cpu":
        n = max(4, min(PROFILE_N_ENVS[profile], cpu))
    elif profile == "workstation":
        n = max(16, min(PROFILE_N_ENVS[profile], cpu * 4))
    else:
        n = max(32, min(PROFILE_N_ENVS[profile], cpu * 8))
    if ram is not None:
        n = min(n, max(2, int(float(ram) * 4)))
    return max(1, int(n))


MAX_DEFAULT_WORKERS = 8


def _parse_worker_override(env_name: str) -> int | None:
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _default_workers(n: int, hardware: dict[str, Any] | None = None) -> int:
    """Leave one core for PPO / the Lab. Cap copies of the mesh field (override via env)."""
    n = max(1, int(n))
    hw = hardware or detect_hardware()
    cpu = max(1, int(hw.get("cpuCount") or cpu_count()))
    spare = max(1, cpu - 1) if cpu > 1 else 1
    cap = min(spare, MAX_DEFAULT_WORKERS)
    ram = hw.get("ramGb")
    if ram is not None:
        cap = min(cap, max(1, int(float(ram) // 4)))
    return min(n, cap)


def recommended_sim_workers(n_envs: int, hardware: dict[str, Any] | None = None) -> int:
    """Process count for CPU physics. Default min(nEnvs, cores-1, 8); override with TALONGYM_SIM_WORKERS."""
    n_envs = max(1, int(n_envs))
    forced = _parse_worker_override(ENV_SIM_WORKERS)
    if forced is not None:
        return min(n_envs, forced)
    return _default_workers(n_envs, hardware)


def recommended_eval_workers(n_trials: int, hardware: dict[str, Any] | None = None) -> int:
    """Process count for held-out trials. Override with TALONGYM_EVAL_WORKERS, not SIM_WORKERS."""
    n_trials = max(1, int(n_trials))
    forced = _parse_worker_override(ENV_EVAL_WORKERS)
    if forced is not None:
        return min(n_trials, forced)
    return _default_workers(n_trials, hardware)


def training_family(training_id: str) -> str:
    for suffix in _FAMILY_SUFFIXES:
        if training_id.endswith(suffix):
            return training_id[: -len(suffix)]
    return training_id


def training_id_for_profile(training_id: str, profile: str) -> str:
    suffix = PROFILE_SUFFIX.get(profile, "lightweight")
    return f"{training_family(training_id)}_{suffix}"


def easy_training_id(current_id: str | None) -> str:
    from talongym.presets.loader import preset_index

    base = current_id or "biobuzz_auto_lightweight"
    candidate = training_id_for_profile(base, "auto")
    if candidate in preset_index()["training"]:
        return candidate
    return base


def resolve_training(
    training: dict[str, Any] | None,
    *,
    easy: bool = False,
    profile: str | None = None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a training-run document with computeProfile and nEnvs resolved."""
    doc = dict(training or {})
    hw = hardware or detect_hardware()
    declared = str(doc.get("computeProfile") or "lightweight_cpu")
    requested = (profile or "").strip().lower() or None
    use_auto = bool(easy) or declared == "auto" or requested == "auto"
    if requested in PROFILES and not use_auto:
        resolved = requested
    elif use_auto:
        resolved = detect_compute_profile(hw)
    elif declared in PROFILES:
        resolved = declared
    else:
        resolved = detect_compute_profile(hw)
    out = dict(doc)
    out["requestedComputeProfile"] = declared
    out["computeProfile"] = resolved
    if use_auto or "nEnvs" not in doc:
        out["nEnvs"] = recommended_n_envs(resolved, hw)
    else:
        out["nEnvs"] = int(doc["nEnvs"])
    return out


def describe_compute(training_id: str | None = None) -> dict[str, Any]:
    hw = detect_hardware()
    profile = detect_compute_profile(hw)
    current = training_id or "biobuzz_auto_lightweight"
    n_envs = recommended_n_envs(profile, hw)
    return {
        "hardware": hw,
        "profile": profile,
        "nEnvs": n_envs,
        "simWorkers": recommended_sim_workers(n_envs, hw),
        "evalWorkers": recommended_eval_workers(500, hw),
        "override": (os.environ.get(ENV_PROFILE) or "").strip() or None,
        "trainingId": current,
        "easyTrainingId": easy_training_id(current),
        "profileTrainingId": training_id_for_profile(current, profile),
        "envVar": ENV_PROFILE,
        "simWorkersEnvVar": ENV_SIM_WORKERS,
        "evalWorkersEnvVar": ENV_EVAL_WORKERS,
        "torchDevice": hw.get("torchDevice"),
        "deviceOverride": (os.environ.get(ENV_DEVICE) or "").strip() or None,
        "deviceEnvVar": ENV_DEVICE,
        "torch": torch_status(),
    }
