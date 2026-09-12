from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from talongym import __version__
from talongym.api import db, jobs
from talongym.eval.harness import run_trials
from talongym.export.roadrunner import export_from_replay
from talongym.paths import WEB_DIST
from talongym.presets.loader import PresetError, load_bundle, validate_document
from talongym.sim.physics import default_backend
from talongym.training.policies import scripted_auto
from talongym.training.ppo import record_policy_episode, load_trained_policy

app = FastAPI(title="TalonGym", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API = "/api/v1"


class ValidateBody(BaseModel):
    kind: str
    document: dict[str, Any]


class RunBody(BaseModel):
    presets: dict[str, str] = Field(default_factory=dict)
    budget: dict[str, int] | None = None
    nEnvs: int | None = None
    demo: bool = True
    resume: bool = False
    easy: bool = False
    computeProfile: str | None = None
    algorithm: dict[str, Any] | None = None


class EvalBody(BaseModel):
    nTrials: int = 32
    objective: str = "mean_true_score"
    policy: str = "scripted"
    runId: str | None = None
    checkpoint: str | None = None


class ExportBody(BaseModel):
    dialect: str = "rr1_actions"


@app.on_event("startup")
def _startup() -> None:
    db.connect()
    db.fail_orphan_runs()


class DefaultsBody(BaseModel):
    fieldId: str | None = None
    robotId: str | None = None
    scoringId: str | None = None
    trainingId: str | None = None


@app.get(f"{API}/health")
def health() -> dict[str, Any]:
    from talongym.training.compute import detect_compute_profile, recommended_n_envs

    engine = default_backend(72.0, 72.0)
    return {
        "ok": True,
        "engine": engine.name,
        "db": db.backend_name(),
        "capabilityVersion": 1,
        "version": __version__,
        "computeProfile": detect_compute_profile(),
        "nEnvs": recommended_n_envs(),
        "note": "planar2d is the Phase 0 Rapier fallback when the Rust crate is a stub.",
    }


@app.get(f"{API}/compute")
def compute_info() -> dict[str, Any]:
    from talongym.presets.defaults import get_defaults
    from talongym.training.compute import describe_compute

    return describe_compute(get_defaults().get("trainingId"))


@app.get(f"{API}/defaults")
def get_defaults_bundle() -> dict[str, Any]:
    from talongym.presets.defaults import describe_defaults

    return describe_defaults()


@app.put(f"{API}/defaults")
def put_defaults_bundle(body: DefaultsBody) -> dict[str, Any]:
    from talongym.presets.defaults import describe_defaults, set_defaults

    if not any([body.fieldId, body.robotId, body.scoringId, body.trainingId]):
        raise HTTPException(422, {"error": {"code": "BAD_DEFAULTS", "message": "Provide trainingId or field/robot/scoring ids"}})
    try:
        set_defaults(
            field_id=body.fieldId,
            robot_id=body.robotId,
            scoring_id=body.scoringId,
            training_id=body.trainingId,
        )
    except PresetError as exc:
        raise HTTPException(422, {"error": {"code": "BAD_DEFAULTS", "message": str(exc)}}) from exc
    return describe_defaults()


@app.get(f"{API}/presets/{{kind}}")
def list_presets(kind: str) -> list[dict[str, Any]]:
    if kind not in {"field", "robot", "scoring", "training"}:
        raise HTTPException(400, {"error": {"code": "BAD_KIND", "message": kind}})
    return db.list_kind(kind)


@app.get(f"{API}/presets/{{kind}}/{{preset_id}}")
def get_preset(kind: str, preset_id: str) -> dict[str, Any]:
    doc = db.get_preset(preset_id)
    if not doc:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": preset_id}})
    return doc


@app.post(f"{API}/presets/validate")
def validate(body: ValidateBody) -> dict[str, Any]:
    try:
        errs = validate_document(body.kind, body.document)
    except PresetError as exc:
        return {"ok": False, "errors": [str(exc)], "unsupportedCapabilities": []}
    unsupported = [e for e in errs if e.startswith("unsupportedCapabilities")]
    return {"ok": not errs, "errors": errs, "unsupportedCapabilities": unsupported}


@app.post(f"{API}/presets/{{kind}}", status_code=201)
def create_preset(kind: str, document: dict[str, Any]) -> dict[str, str]:
    if kind not in {"field", "robot", "scoring", "training"}:
        raise HTTPException(400, {"error": {"code": "BAD_KIND", "message": kind}})
    errs = validate_document(kind, document)
    if errs:
        raise HTTPException(422, {"error": {"code": "SCHEMA", "message": "; ".join(errs[:12])}})
    db.upsert_preset(kind, document)
    return {"id": document["id"]}


@app.put(f"{API}/presets/{{kind}}/{{preset_id}}")
def replace_preset(kind: str, preset_id: str, document: dict[str, Any]) -> dict[str, str]:
    document["id"] = preset_id
    errs = validate_document(kind, document)
    if errs:
        raise HTTPException(422, {"error": {"code": "SCHEMA", "message": "; ".join(errs[:12])}})
    db.upsert_preset(kind, document)
    return {"id": preset_id}


@app.delete(f"{API}/presets/{{kind}}/{{preset_id}}", status_code=204)
def delete_preset(kind: str, preset_id: str) -> None:
    ok = db.delete_preset(preset_id)
    if not ok:
        raise HTTPException(409, {"error": {"code": "IN_USE", "message": preset_id}})


@app.post(f"{API}/runs", status_code=202)
def start_run(body: RunBody) -> dict[str, str]:
    run_id = jobs.start_training(body.model_dump())
    return {"runId": run_id}


@app.get(f"{API}/runs")
def runs() -> list[dict[str, Any]]:
    return db.list_runs()


@app.get(f"{API}/runs/{{run_id}}")
def get_run(run_id: str) -> dict[str, Any]:
    row = db.get_run(run_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": run_id}})
    return row


@app.post(f"{API}/runs/{{run_id}}/cancel")
def cancel_run(run_id: str) -> dict[str, str]:
    row = db.get_run(run_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": run_id}})
    jobs.request_cancel(run_id)
    if row["state"] in {"queued", "running", "cancelling"}:
        db.save_run(run_id, row["config"], "cancelled", row.get("metrics") or {}, row.get("log"))
        jobs.emit(run_id, {"type": "status", "payload": {"state": "cancelled"}})
    return {"state": "cancelled"}


@app.post(f"{API}/runs/{{run_id}}/export/onnx")
def export_onnx(run_id: str, distill: bool = False) -> Any:
    if not distill:
        return JSONResponse(
            {"error": {"code": "ONNX_LSTM", "message": "LSTM ONNX is best-effort; use Road Runner waypoint export for deployment. Pass distill=true for a feed-forward demo."}},
            status_code=501,
        )
    from talongym.training.distill import distill_from_scripted

    result = distill_from_scripted()
    db.save_artifact(run_id, "onnx_ff", result["path"])
    return JSONResponse(result, status_code=201)


@app.get(f"{API}/replays")
def replays() -> list[dict[str, Any]]:
    return db.list_replays()


@app.get(f"{API}/replays/{{replay_id}}")
def get_replay(replay_id: str) -> dict[str, Any]:
    row = db.get_replay(replay_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": replay_id}})
    meta = row["meta"]
    frames = row["frames"]
    return {
        "id": replay_id,
        "duration": frames[-1]["t"] if frames else 0,
        "hz": 25,
        "nFrames": len(frames),
        "trueScore": frames[-1].get("trueScore") if frames else 0,
        **meta,
    }


@app.get(f"{API}/replays/{{replay_id}}/chunks")
def replay_chunks(replay_id: str, fromStep: int = 0, limit: int = 500) -> dict[str, Any]:
    row = db.get_replay(replay_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": replay_id}})
    frames = row["frames"][fromStep : fromStep + min(limit, 500)]
    return {"fromStep": fromStep, "frames": frames, "done": fromStep + len(frames) >= len(row["frames"])}


@app.post(f"{API}/replays/{{replay_id}}/export/roadrunner")
def export_rr(replay_id: str, body: ExportBody | None = None) -> PlainTextResponse:
    row = db.get_replay(replay_id)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": replay_id}})
    dialect = (body.dialect if body else "rr1_actions")
    return PlainTextResponse(export_from_replay(row["frames"], dialect=dialect), media_type="text/plain")


@app.post(f"{API}/replays/demo", status_code=201)
def demo_replay() -> dict[str, str]:
    frames = record_policy_episode(scripted_auto, seed=4)
    rid = db.save_replay(frames, {"source": "demo", "trueScore": frames[-1].get("trueScore") if frames else 0})
    return {"replayId": rid}


@app.post(f"{API}/evaluations", status_code=202)
def start_eval(body: EvalBody) -> dict[str, Any]:
    n = max(8, min(int(body.nTrials), 500))
    policy = scripted_auto
    bundle = None
    run_presets: dict[str, str] = {}
    if body.runId:
        run = db.get_run(body.runId)
        if run:
            run_presets = (run.get("config") or {}).get("presets") or {}
            bundle = load_bundle(
                run_presets.get("fieldId"),
                run_presets.get("robotId"),
                run_presets.get("scoringId"),
                run_presets.get("trainingId"),
            )
    if body.policy in {"checkpoint", "trained"}:
        path = body.checkpoint
        if not path and body.runId:
            arts = [a for a in db.list_artifacts(body.runId) if a.get("kind") == "checkpoint"]
            if arts:
                path = arts[-1].get("payload")
        if not path:
            raise HTTPException(400, {"error": {"code": "NO_CHECKPOINT", "message": "No trained policy artifact"}})
        policy = load_trained_policy(path)
    if bundle is None:
        bundle = load_bundle()
    report = run_trials(n, policy, bundle=bundle, seed0=10_000_000, record_best=True)
    frames = report.pop("bestFrames", [])
    replay_id = db.save_replay(frames, {"source": "eval", "trueScore": report.get("bestScore")}) if frames else None
    report["replayId"] = replay_id
    report["policy"] = body.policy
    report["runId"] = body.runId
    report["bestLabelEligible"] = bool(n >= 500)
    if n < 500:
        report["bestLabelEligible"] = False
    eid = db.save_evaluation(report, run_id=body.runId)
    row = db.get_evaluation(eid) or {"id": eid, "report": report}
    return {"evaluationId": eid, "replayId": replay_id, **row}


@app.get(f"{API}/evaluations")
def evaluations() -> list[dict[str, Any]]:
    return db.list_evaluations()


@app.get(f"{API}/evaluations/{{eid}}")
def get_eval(eid: str) -> dict[str, Any]:
    row = db.get_evaluation(eid)
    if not row:
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": eid}})
    return row


@app.get(f"{API}/comparisons/latest")
def comparison() -> dict[str, Any]:
    evs = db.list_evaluations()
    ranked = sorted(evs, key=lambda e: (e["report"] or {}).get("mean") or 0, reverse=True)
    eligible = all((e["report"] or {}).get("bestLabelEligible") for e in ranked) and len(ranked) >= 1
    if any((e["report"] or {}).get("nTrials", 0) < 500 for e in ranked):
        eligible = False
    if len(ranked) >= 2:
        a = ranked[0]["report"]
        b = ranked[1]["report"]
        overlap = not (a.get("hi", 0) < b.get("lo", 0) or b.get("hi", 0) < a.get("lo", 0))
        if overlap:
            eligible = False
    return {"evaluations": ranked, "bestLabelEligible": eligible}


@app.websocket(f"{API}/ws/runs/{{run_id}}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    outgoing: asyncio.Queue[str] = asyncio.Queue()
    seq = 0

    def push(msg: dict) -> None:
        nonlocal seq
        seq += 1
        payload = json.dumps({"v": 1, "seq": seq, **msg}, default=str)
        loop.call_soon_threadsafe(outgoing.put_nowait, payload)

    jobs.subscribe(run_id, push)
    try:
        run = db.get_run(run_id)
        await websocket.send_text(
            json.dumps({"v": 1, "seq": 0, "type": "status", "payload": run or {"state": "unknown"}})
        )
        if run and run.get("metrics"):
            seq = 1
            await websocket.send_text(json.dumps({"v": 1, "seq": 1, "type": "metrics", "payload": run["metrics"]}))
        for msg in jobs.latest_messages(run_id):
            if msg.get("type") not in {"rollout", "log"}:
                continue
            seq += 1
            await websocket.send_text(json.dumps({"v": 1, "seq": seq, **msg}))

        async def pump_out() -> None:
            while True:
                await websocket.send_text(await outgoing.get())

        async def pump_in() -> None:
            while True:
                await websocket.receive_text()

        done, pending = await asyncio.wait(
            [asyncio.create_task(pump_out()), asyncio.create_task(pump_in())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                raise exc
    except WebSocketDisconnect:
        pass
    finally:
        jobs.unsubscribe(run_id, push)


@app.get(f"{API}/field-assets/{{asset_path:path}}")
def field_asset(asset_path: str) -> FileResponse:
    from talongym.paths import ASSETS_DIR

    rel = Path(asset_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(400, {"error": {"code": "BAD_ASSET", "message": asset_path}})
    dest = (ASSETS_DIR / rel).resolve()
    try:
        dest.relative_to(ASSETS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(400, {"error": {"code": "BAD_ASSET", "message": asset_path}}) from exc
    if not dest.is_file():
        raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": asset_path}})
    media = "model/gltf-binary" if dest.suffix.lower() == ".glb" else "application/octet-stream"
    if dest.suffix.lower() in {".gltf", ".json"}:
        media = "model/gltf+json"
    if dest.suffix.lower() in {".xml", ".mjcf"}:
        media = "application/xml"
    return FileResponse(dest, media_type=media)


def mount_frontend(application: FastAPI) -> None:
    dist = WEB_DIST
    if dist.exists():
        application.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @application.get("/{full_path:path}")
        def spa(full_path: str):
            if full_path.startswith("api"):
                raise HTTPException(404, {"error": {"code": "NOT_FOUND", "message": full_path}})
            candidate = dist / full_path
            if full_path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")


mount_frontend(app)
