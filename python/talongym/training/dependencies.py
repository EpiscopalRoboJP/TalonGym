"""Helpers for actionable optional training dependency errors."""

from __future__ import annotations

import shlex
import subprocess
import sys
import threading

from talongym.paths import REPO_ROOT


_install_lock = threading.Lock()
_install_state_lock = threading.Lock()
_install_state: dict[str, str] = {
    "status": "idle",
    "message": "Training dependencies have not been installed from the Lab.",
}


def install_command(extra: str) -> str:
    """Return a shell-safe editable-install command for the active interpreter."""
    if sys.platform == "win32":
        executable = subprocess.list2cmdline([sys.executable])
    else:
        executable = shlex.quote(sys.executable)
    return f'{executable} -m pip install -e ".[{extra}]"'


def rl_install_argv() -> list[str]:
    """Return the fixed pip command for the runtime RecurrentPPO training stack."""
    return [sys.executable, "-m", "pip", "install", "-e", ".[rl,mujoco]"]


def _dependency_state() -> dict[str, str]:
    with _install_state_lock:
        return dict(_install_state)


def _install_training_dependencies() -> None:
    try:
        result = subprocess.run(rl_install_argv(), cwd=REPO_ROOT, check=False)
        with _install_state_lock:
            if result.returncode == 0:
                _install_state.update(status="succeeded", message="Training dependencies installed. You can start a run now.")
            else:
                _install_state.update(
                    status="failed",
                    message=f"pip exited with status {result.returncode}. Check the Lab server console for details, then retry.",
                )
    except Exception as exc:
        with _install_state_lock:
            _install_state.update(status="failed", message=f"Could not start pip: {exc}")
    finally:
        _install_lock.release()


def training_dependency_status() -> dict[str, object]:
    """Report whether RecurrentPPO and BIOBUZZ mesh training imports are available."""
    import importlib

    missing: list[str] = []
    for module, package in (
        ("torch", "torch"),
        ("stable_baselines3", "stable-baselines3"),
        ("sb3_contrib", "sb3-contrib"),
        ("mujoco", "mujoco"),
    ):
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    return {"installed": not missing, "missing": missing, "install": _dependency_state()}


def start_training_dependency_install() -> dict[str, str]:
    """Start one background install of the fixed local training extras."""
    if not _install_lock.acquire(blocking=False):
        return _dependency_state()
    with _install_state_lock:
        _install_state.update(status="running", message="Installing training dependencies. Torch may take a few minutes to download.")
    try:
        thread = threading.Thread(target=_install_training_dependencies, name="talongym-dependency-install", daemon=True)
        thread.start()
    except Exception:
        _install_lock.release()
        raise
    return _dependency_state()
