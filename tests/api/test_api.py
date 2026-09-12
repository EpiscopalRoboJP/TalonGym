from __future__ import annotations

import time

from fastapi.testclient import TestClient

from talongym.api.app import app
from talongym.presets.loader import load_bundle


def test_health_and_presets():
    client = TestClient(app)
    h = client.get("/api/v1/health")
    assert h.status_code == 200
    body = h.json()
    assert body["ok"] is True
    assert body["computeProfile"] in {"lightweight_cpu", "workstation", "cloud"}
    assert body["nEnvs"] >= 1
    compute = client.get("/api/v1/compute")
    assert compute.status_code == 200
    info = compute.json()
    assert info["profile"] == body["computeProfile"]
    assert info["easyTrainingId"].endswith("_easy")
    fields = client.get("/api/v1/presets/field")
    assert fields.status_code == 200
    ids = {row["id"] for row in fields.json()}
    assert "decode_2025_field_tu32" in ids
    assert "into_the_deep_2024_field" in ids
    assert "biobuzz_2026_field_v1" in ids


def test_defaults_get_and_put():
    client = TestClient(app)
    got = client.get("/api/v1/defaults")
    assert got.status_code == 200
    body = got.json()
    assert body["fieldId"] == "decode_2025_field_tu32"
    assert body["scoringId"] == "decode_2025_scoring_tu32"
    assert body["season"] == "decode"
    put = client.put("/api/v1/defaults", json={"trainingId": "biobuzz_auto_lightweight"})
    assert put.status_code == 200
    switched = put.json()
    assert switched["fieldId"] == "biobuzz_2026_field_v1"
    assert switched["robotId"] == "mecanum_biobuzz_4cap"
    assert switched["scoringId"] == "biobuzz_2026_scoring_v1"
    assert switched["season"] == "biobuzz"
    bundle = load_bundle()
    assert bundle.field["id"] == "biobuzz_2026_field_v1"
    assert bundle.scoring["id"] == "biobuzz_2026_scoring_v1"
    restore = client.put("/api/v1/defaults", json={"trainingId": "decode_auto_lightweight"})
    assert restore.status_code == 200
    assert restore.json()["season"] == "decode"


def test_demo_replay_and_export():
    client = TestClient(app)
    created = client.post("/api/v1/replays/demo")
    assert created.status_code == 201
    rid = created.json()["replayId"]
    meta = client.get(f"/api/v1/replays/{rid}")
    assert meta.status_code == 200
    assert meta.json()["trueScore"] >= 3
    chunks = client.get(f"/api/v1/replays/{rid}/chunks", params={"fromStep": 0, "limit": 50})
    assert chunks.status_code == 200
    frames = chunks.json()["frames"]
    assert frames
    assert frames[0]["robots"]
    assert any(el["type"] == "zone" for el in frames[0]["elements"])
    java = client.post(f"/api/v1/replays/{rid}/export/roadrunner")
    assert java.status_code == 200
    assert "actionBuilder" in java.text


def test_preset_validate_is_not_swallowed_by_create_route():
    client = TestClient(app)
    scoring = client.get("/api/v1/presets/scoring/decode_2025_scoring_tu32")
    assert scoring.status_code == 200
    doc = scoring.json()
    doc.pop("_kind", None)
    res = client.post("/api/v1/presets/validate", json={"kind": "scoring", "document": doc})
    assert res.status_code == 200
    assert res.json()["ok"] is True


def _frame() -> dict:
    return {
        "t": 0.4,
        "trueScore": 3,
        "robots": [{"id": "red_0", "x": 0.0, "y": -48.0, "headingDeg": 90.0, "held": [], "dynamic": True, "alliance": "red"}],
        "pieces": [],
        "elements": [{"id": "z", "type": "zone", "pose": {"x": 0, "y": 0}, "shape": {"kind": "rect", "width": 1, "depth": 1}}],
        "explains": [],
        "queues": {},
        "gate": {},
        "matchVarsPrivileged": {},
        "observedMatchVars": {},
        "fieldSizeIn": {"width": 144, "depth": 144},
        "vision": [],
    }


def _wait_run(client: TestClient, run_id: str, states: set[str], timeout: float = 8.0) -> dict:
    deadline = time.time() + timeout
    last: dict | None = None
    while time.time() < deadline:
        res = client.get(f"/api/v1/runs/{run_id}")
        assert res.status_code == 200
        last = res.json()
        if last["state"] in states:
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} stuck in {last}")


def test_start_run_passes_training_id(monkeypatch):
    captured: dict = {}

    def fake_train(**kwargs):
        captured["bundle"] = kwargs.get("bundle")
        on_metrics = kwargs.get("on_metrics")
        if on_metrics:
            on_metrics({"envSteps": 64, "trueScoreMean": 3.0, "algo": "recurrent_ppo", "nEnvs": 2})
        on_rollout = kwargs.get("on_rollout")
        if on_rollout:
            on_rollout([_frame()])
        return {
            "algo": "recurrent_ppo",
            "frames": [_frame()],
            "steps": 64,
            "checkpoint": "mem://ckpt",
            "metrics": {"envSteps": 64},
        }

    monkeypatch.setattr("talongym.api.jobs.train_ppo", fake_train)
    client = TestClient(app)
    res = client.post(
        "/api/v1/runs",
        json={
            "demo": True,
            "nEnvs": 2,
            "budget": {"totalEnvSteps": 128},
            "presets": {
                "fieldId": "biobuzz_2026_field_v1",
                "robotId": "mecanum_biobuzz_4cap",
                "scoringId": "biobuzz_2026_scoring_v1",
                "trainingId": "biobuzz_auto_lightweight",
            },
        },
    )
    assert res.status_code == 202
    run_id = res.json()["runId"]
    row = _wait_run(client, run_id, {"succeeded", "failed"})
    assert row["state"] == "succeeded"
    assert row["config"]["presets"]["trainingId"] == "biobuzz_auto_lightweight"
    assert captured["bundle"].training["id"] == "biobuzz_auto_lightweight"
    assert captured["bundle"].field["id"] == "biobuzz_2026_field_v1"
    assert row["metrics"].get("replayId")


def test_cancel_run(monkeypatch):
    def fake_train(**kwargs):
        should_stop = kwargs.get("should_stop")
        while should_stop and not should_stop():
            time.sleep(0.02)
        return {"algo": "recurrent_ppo", "frames": [], "steps": 8, "cancelled": True}

    monkeypatch.setattr("talongym.api.jobs.train_ppo", fake_train)
    client = TestClient(app)
    res = client.post("/api/v1/runs", json={"demo": True, "presets": {"trainingId": "decode_auto_lightweight"}})
    assert res.status_code == 202
    run_id = res.json()["runId"]
    _wait_run(client, run_id, {"running", "queued", "cancelled"})
    cancel = client.post(f"/api/v1/runs/{run_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["state"] == "cancelled"
    row = _wait_run(client, run_id, {"cancelled", "failed", "succeeded"})
    assert row["state"] == "cancelled"
    time.sleep(0.3)


def test_easy_run_uses_autodetect_training_id(monkeypatch):
    captured: dict = {}

    def fake_train(**kwargs):
        captured["bundle"] = kwargs.get("bundle")
        captured["n_envs"] = kwargs.get("n_envs")
        on_metrics = kwargs.get("on_metrics")
        if on_metrics:
            on_metrics({"envSteps": 32, "trueScoreMean": 3.0, "algo": "recurrent_ppo", "nEnvs": kwargs.get("n_envs")})
        return {"algo": "recurrent_ppo", "frames": [_frame()], "steps": 32, "metrics": {"envSteps": 32}}

    monkeypatch.setattr("talongym.api.jobs.train_ppo", fake_train)
    monkeypatch.setenv("TALONGYM_COMPUTE_PROFILE", "lightweight_cpu")
    client = TestClient(app)
    res = client.post(
        "/api/v1/runs",
        json={
            "demo": False,
            "easy": True,
            "computeProfile": "auto",
            "presets": {"trainingId": "decode_auto_lightweight"},
        },
    )
    assert res.status_code == 202
    run_id = res.json()["runId"]
    row = _wait_run(client, run_id, {"succeeded", "failed"})
    assert row["state"] == "succeeded"
    assert captured["bundle"].training["id"] == "decode_auto_easy"
    assert captured["n_envs"] >= 1


def test_field_asset_glb_served():
    client = TestClient(app)
    res = client.get("/api/v1/field-assets/seasons/biobuzz_2026/field.glb")
    assert res.status_code == 200
    assert res.content[:4] == b"glTF"
    missing = client.get("/api/v1/field-assets/seasons/nope/missing.glb")
    assert missing.status_code == 404
    traversal = client.get("/api/v1/field-assets/../pyproject.toml")
    assert traversal.status_code in {400, 404}
