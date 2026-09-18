import time

import pytest


def cad_assets_present() -> bool:
    from talongym.paths import ASSETS_DIR

    return (ASSETS_DIR / "seasons/biobuzz_2026/cad_manifest.json").is_file()


def mesh_runtime_available() -> bool:
    from talongym.sim.mujoco_backend import available

    return available() and cad_assets_present()


def pytest_runtest_setup(item):
    if item.get_closest_marker("require_cad") and not cad_assets_present():
        pytest.skip("derived CAD assets not generated")
    if item.get_closest_marker("require_mesh") and not mesh_runtime_available():
        pytest.skip("BIOBUZZ mesh runtime needs mujoco and derived CAD")


@pytest.fixture(autouse=True)
def _isolate_var_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("talongym.paths.VAR_DIR", tmp_path / "var")
    from talongym.api import db

    db.disconnect()
    yield
    time.sleep(0.2)
    db.disconnect()


@pytest.fixture(autouse=True)
def _allow_missing_mesh_unless_required(monkeypatch, request):
    """BIOBUZZ needs MuJoCo. Without it, scoring/API tests use the planar fallback."""
    if request.node.get_closest_marker("require_mesh"):
        return
    from talongym.sim.mujoco_backend import available

    if available():
        return
    monkeypatch.setenv("TALONGYM_ALLOW_MISSING_MESH", "1")
    from talongym.sim.world import World

    orig = World.__init__

    def wrapped(self, bundle, seed=0, control_hz=25, substeps=2, allow_missing_mesh=False):
        orig(
            self,
            bundle,
            seed=seed,
            control_hz=control_hz,
            substeps=substeps,
            allow_missing_mesh=True,
        )

    monkeypatch.setattr(World, "__init__", wrapped)
