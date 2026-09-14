import time

import pytest


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
