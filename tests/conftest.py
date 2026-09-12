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
