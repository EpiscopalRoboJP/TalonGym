"""Detect laptop / workstation / cloud and resolve training-run nEnvs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROFILES = ("lightweight_cpu", "workstation", "cloud")
ENV_PROFILE = "TALONGYM_COMPUTE_PROFILE"
ENV_DEVICE = "TALONGYM_TORCH_DEVICE"

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


def cuda_available() -> bool:
    """True for NVIDIA CUDA, and also for AMD ROCm builds of torch (ROCm exposes
    itself through the same torch.cuda API)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
    return {
        "hardware": hw,
        "profile": profile,
        "nEnvs": recommended_n_envs(profile, hw),
        "override": (os.environ.get(ENV_PROFILE) or "").strip() or None,
        "trainingId": current,
        "easyTrainingId": easy_training_id(current),
        "profileTrainingId": training_id_for_profile(current, profile),
        "envVar": ENV_PROFILE,
        "torchDevice": hw.get("torchDevice"),
        "deviceOverride": (os.environ.get(ENV_DEVICE) or "").strip() or None,
        "deviceEnvVar": ENV_DEVICE,
    }
