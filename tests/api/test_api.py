from __future__ import annotations

import json
import threading
import time

import pytest
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
    assert "biobuzz_2026_field_v1" in ids
    assert ids == {"biobuzz_2026_field_v1"}


def test_defaults_get_and_put():
    client = TestClient(app)
    got = client.get("/api/v1/defaults")
    assert got.status_code == 200
    body = got.json()
    assert body["fieldId"] == "biobuzz_2026_field_v1"
    assert body["scoringId"] == "biobuzz_2026_scoring_v1"
    assert body["season"] == "biobuzz"
    put = client.put("/api/v1/defaults", json={"trainingId": "biobuzz_auto_easy"})
    assert put.status_code == 200
    switched = put.json()
    assert switched["fieldId"] == "biobuzz_2026_field_v1"
    assert switched["robotId"] == "mecanum_biobuzz_4cap"
    assert switched["scoringId"] == "biobuzz_2026_scoring_v1"
    assert switched["season"] == "biobuzz"
    assert switched["trainingId"] == "biobuzz_auto_easy"
    bundle = load_bundle()
    assert bundle.field["id"] == "biobuzz_2026_field_v1"
    assert bundle.scoring["id"] == "biobuzz_2026_scoring_v1"
    restore = client.put("/api/v1/defaults", json={"trainingId": "biobuzz_auto_lightweight"})
    assert restore.status_code == 200
    assert restore.json()["trainingId"] == "biobuzz_auto_lightweight"
    assert restore.json()["season"] == "biobuzz"


@pytest.mark.require_mesh
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
    scoring = client.get("/api/v1/presets/scoring/biobuzz_2026_scoring_v1")
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
    assert row.get("artifactIds")
    arts = client.get(f"/api/v1/runs/{run_id}/artifacts")
    assert arts.status_code == 200
    kinds = {a["kind"] for a in arts.json()}
    assert "checkpoint" in kinds
    assert "roadrunner" in kinds


def test_cancel_run(monkeypatch):
    def fake_train(**kwargs):
        should_stop = kwargs.get("should_stop")
        while should_stop and not should_stop():
            time.sleep(0.02)
        return {"algo": "recurrent_ppo", "frames": [], "steps": 8, "cancelled": True}

    monkeypatch.setattr("talongym.api.jobs.train_ppo", fake_train)
    client = TestClient(app)
    res = client.post("/api/v1/runs", json={"demo": True, "presets": {"trainingId": "biobuzz_auto_lightweight"}})
    assert res.status_code == 202
    run_id = res.json()["runId"]
    _wait_run(client, run_id, {"running", "queued", "cancelling"})
    cancel = client.post(f"/api/v1/runs/{run_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["state"] == "cancelling"
    mid = client.get(f"/api/v1/runs/{run_id}")
    assert mid.status_code == 200
    assert mid.json()["state"] in {"cancelling", "cancelled"}
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
            "presets": {"trainingId": "biobuzz_auto_lightweight"},
        },
    )
    assert res.status_code == 202
    run_id = res.json()["runId"]
    row = _wait_run(client, run_id, {"succeeded", "failed"})
    assert row["state"] == "succeeded"
    assert captured["bundle"].training["id"] == "biobuzz_auto_easy"
    assert captured["n_envs"] >= 1


@pytest.mark.require_cad
def test_field_asset_glb_served():
    client = TestClient(app)
    res = client.get("/api/v1/field-assets/seasons/biobuzz_2026/field.glb")
    assert res.status_code == 200
    assert res.content[:4] == b"glTF"
    missing = client.get("/api/v1/field-assets/seasons/nope/missing.glb")
    assert missing.status_code == 404
    traversal = client.get("/api/v1/field-assets/../pyproject.toml")
    assert traversal.status_code in {400, 404}


def test_robot_asset_traversal_rejected():
    client = TestClient(app)
    traversal = client.get("/api/v1/robot-assets/../pyproject.toml")
    assert traversal.status_code in {400, 404}
    missing = client.get("/api/v1/robot-assets/robots/nope/visual.glb")
    assert missing.status_code == 404


def test_robot_model_upload_and_delete(tmp_path):
    pytest.importorskip("trimesh")
    from talongym.assets.import_robot_cad import delete_robot_assets, write_ascii_box_stl

    stl = write_ascii_box_stl(tmp_path / "box.stl", 18, 14, 10)
    client = TestClient(app)
    robot_id = "test_api_cube"
    try:
        with stl.open("rb") as fh:
            res = client.post(
                f"/api/v1/presets/robot/{robot_id}/model",
                files={"file": ("box.stl", fh, "model/stl")},
            )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["visualAsset"].endswith("visual.glb")
        glb = client.get(f"/api/v1/robot-assets/{body['visualAsset']}")
        assert glb.status_code == 200
        assert glb.content[:4] == b"glTF"
        gone = client.delete(f"/api/v1/presets/robot/{robot_id}/model")
        assert gone.status_code == 200
        missing = client.get(f"/api/v1/robot-assets/{body['visualAsset']}")
        assert missing.status_code == 404
    finally:
        delete_robot_assets(robot_id)


def test_biobuzz_frames_get_cad_background():
    from talongym.assets.cad_common import CAD_GENERATOR_VERSION
    from talongym.api.cad_frames import ensure_background_asset

    frame = {
        "elements": [{"id": "red_cell_up", "type": "goal", "tags": ["hive", "up_cell"]}],
        "fieldSizeIn": {"width": 144, "depth": 144},
    }
    stamped = ensure_background_asset(frame)
    assert stamped["backgroundAsset"] == "seasons/biobuzz_2026/field.glb"
    assert stamped["cadManifest"] == "seasons/biobuzz_2026/cad_manifest.json"
    assert stamped["cadSourceSha256"] == "05b35961c7df847741031f00fda73ddd068537f809e92a11b1cd59a94bcc8331"
    assert stamped["cadAssetVersion"].endswith(f":{CAD_GENERATOR_VERSION}")
    kept = ensure_background_asset({**frame, "backgroundAsset": "custom.glb"})
    assert kept["backgroundAsset"] == "custom.glb"
    assert "cadManifest" not in kept


def test_ws_run_streams_live_metrics(monkeypatch):
    gate = threading.Event()

    def fake_train(**kwargs):
        on_metrics = kwargs.get("on_metrics")
        assert gate.wait(timeout=4)
        for step in (32, 64, 96):
            if on_metrics:
                on_metrics(
                    {
                        "envSteps": step,
                        "progressFrac": step / 96,
                        "trueScoreMean": step / 10,
                        "algo": "recurrent_ppo",
                    }
                )
            time.sleep(0.02)
        return {
            "algo": "recurrent_ppo",
            "frames": [_frame()],
            "steps": 96,
            "checkpoint": "mem://ckpt",
            "metrics": {"envSteps": 96, "progressFrac": 1.0},
        }

    monkeypatch.setattr("talongym.api.jobs.train_ppo", fake_train)
    client = TestClient(app)
    res = client.post("/api/v1/runs", json={"demo": True, "presets": {"trainingId": "biobuzz_auto_lightweight"}})
    assert res.status_code == 202
    run_id = res.json()["runId"]
    steps: list[int] = []
    with client.websocket_connect(f"/api/v1/ws/runs/{run_id}") as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "status"
        gate.set()
        for _ in range(20):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "metrics":
                steps.append(int((msg.get("payload") or {}).get("envSteps") or 0))
            if steps and max(steps) >= 96:
                break
    assert any(s >= 32 for s in steps), steps
    assert max(steps) >= 96


def test_evaluations_return_200_and_store_objective(monkeypatch):
    def fake_trials(n, policy, **kwargs):
        scores = [3.0] * n
        return {
            "mean": 3.0,
            "lo": 2.0,
            "hi": 4.0,
            "median": 3.0,
            "p10": 2.0,
            "p90": 4.0,
            "min": 1.0,
            "max": 5.0,
            "nTrials": n,
            "scores": scores,
            "seeds": list(range(n)),
            "collisionRate": 0.0,
            "collisionTimeMean": 0.0,
            "firstContactS": None,
            "restrictedEntryRate": 0.0,
            "bestScore": 3.0,
            "bestFrames": [],
            "bestLabelEligible": False,
            "objective": kwargs.get("objective") or "mean_true_score",
            "objectiveValue": 2.0 if kwargs.get("objective") == "p10_true_score" else 3.0,
        }

    import importlib

    api_mod = importlib.import_module("talongym.api.app")
    monkeypatch.setattr(api_mod, "run_trials", fake_trials)
    client = TestClient(app)
    res = client.post("/api/v1/evaluations", json={"nTrials": 8, "objective": "p10_true_score"})
    assert res.status_code == 200
    body = res.json()
    assert body["evaluationId"]
    report = body["report"]
    assert report["objective"] == "p10_true_score"
    assert report["objectiveValue"] == 2.0


def test_frozen_policy_missing_checkpoint_skips(monkeypatch):
    captured: dict = {}

    def fake_train(**kwargs):
        captured["frozen"] = kwargs.get("frozen_policy")
        return {"algo": "recurrent_ppo", "frames": [_frame()], "steps": 8, "metrics": {"envSteps": 8}}

    monkeypatch.setattr("talongym.api.jobs.train_ppo", fake_train)
    client = TestClient(app)
    res = client.post(
        "/api/v1/runs",
        json={
            "demo": True,
            "presets": {"trainingId": "biobuzz_auto_lightweight", "opponentPolicy": "frozen_policy"},
        },
    )
    assert res.status_code == 202
    run_id = res.json()["runId"]
    row = _wait_run(client, run_id, {"succeeded", "failed"})
    assert row["state"] == "succeeded"
    assert captured.get("frozen") is None
    logs = client.get(f"/api/v1/runs/{run_id}").json().get("log") or ""
    assert "frozen_policy" in logs
    assert "skipping" in logs.lower()


def test_onnx_export_is_501_without_distill(monkeypatch):
    def fake_train(**kwargs):
        return {"algo": "recurrent_ppo", "frames": [_frame()], "steps": 8, "checkpoint": "mem://ckpt"}

    monkeypatch.setattr("talongym.api.jobs.train_ppo", fake_train)
    client = TestClient(app)
    res = client.post("/api/v1/runs", json={"demo": True, "presets": {"trainingId": "biobuzz_auto_lightweight"}})
    run_id = res.json()["runId"]
    _wait_run(client, run_id, {"succeeded", "failed"})
    onnx = client.post(f"/api/v1/runs/{run_id}/export/onnx")
    assert onnx.status_code == 501
    assert onnx.json()["error"]["code"] == "ONNX_LSTM"


def test_old_replay_does_not_invent_piece_cad():
    from talongym.api.cad_frames import ensure_background_asset

    old = {
        "elements": [{"id": "red_cell_up", "type": "goal", "tags": ["hive", "up_cell"]}],
        "pieces": [{"id": "p1", "x": 4.0, "y": -8.0, "color": "Y"}],
        "fieldSizeIn": {"width": 144, "depth": 144},
    }
    stamped = ensure_background_asset(old)
    assert stamped["cadManifest"].endswith("cad_manifest.json")
    assert "typeId" not in stamped["pieces"][0]
    assert "visualAsset" not in stamped["pieces"][0]
    assert "gamePieces" not in stamped


def test_demo_frames_match_frontend_cad_contract():
    from talongym.sim.mujoco_backend import available

    if not available():
        pytest.skip("mujoco extra not installed")
    client = TestClient(app)
    created = client.post("/api/v1/replays/demo")
    if created.status_code == 503:
        pytest.skip("mesh field unavailable")
    assert created.status_code == 201
    rid = created.json()["replayId"]
    chunks = client.get(f"/api/v1/replays/{rid}/chunks", params={"fromStep": 0, "limit": 5})
    assert chunks.status_code == 200
    frame = chunks.json()["frames"][0]
    assert frame["cadManifest"] == "seasons/biobuzz_2026/cad_manifest.json"
    assert frame["cadSourceSha256"] == "05b35961c7df847741031f00fda73ddd068537f809e92a11b1cd59a94bcc8331"
    catalog = {row["typeId"]: row for row in frame["gamePieces"]}
    pollen = next(p for p in frame["pieces"] if p["typeId"] == "pollen")
    assert pollen["visualAsset"] == catalog["pollen"]["visualAsset"]
    assert pollen["radius"] == pytest.approx(catalog["pollen"]["shape"]["radius"])
    qw, qx, qy, qz = pollen["qw"], pollen["qx"], pollen["qy"], pollen["qz"]
    assert abs(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2 - 1.0) < 0.05
    glb = client.get(f"/api/v1/field-assets/{pollen['visualAsset']}")
    assert glb.status_code == 200
    assert glb.content[:4] == b"glTF"
    manifest = client.get("/api/v1/field-assets/seasons/biobuzz_2026/cad_manifest.json")
    assert manifest.status_code == 200
    assert manifest.json()["field"]["sha256"] == frame["cadSourceSha256"]
