from fastapi.testclient import TestClient

from talongym.api.app import app
from talongym.presets.loader import load_bundle


def test_health_and_presets():
    client = TestClient(app)
    h = client.get("/api/v1/health")
    assert h.status_code == 200
    assert h.json()["ok"] is True
    fields = client.get("/api/v1/presets/field")
    assert fields.status_code == 200
    ids = {row["id"] for row in fields.json()}
    assert "decode_2025_field_tu32" in ids
    assert "into_the_deep_2024_field" in ids
    assert "biobuzz_2026_field_v1" in ids


def test_defaults_get_and_put(tmp_path, monkeypatch):
    monkeypatch.setattr("talongym.paths.VAR_DIR", tmp_path)
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
