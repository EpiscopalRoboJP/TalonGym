from __future__ import annotations

import threading
from typing import Any, Callable

from talongym.api import db
from talongym.eval.harness import run_trials
from talongym.export.roadrunner import export_from_replay
from talongym.presets.loader import load_bundle
from talongym.training.policies import scripted_auto
from talongym.training.ppo import record_policy_episode, train_ppo

_listeners: dict[str, list[Callable[[dict], None]]] = {}
_cancel: set[str] = set()


def request_cancel(run_id: str) -> None:
    _cancel.add(run_id)


def is_cancelled(run_id: str) -> bool:
    return run_id in _cancel


def subscribe(run_id: str, fn: Callable[[dict], None]) -> None:
    _listeners.setdefault(run_id, []).append(fn)


def unsubscribe(run_id: str, fn: Callable[[dict], None]) -> None:
    lst = _listeners.get(run_id) or []
    if fn in lst:
        lst.remove(fn)


def emit(run_id: str, msg: dict[str, Any]) -> None:
    for fn in list(_listeners.get(run_id) or []):
        try:
            fn(msg)
        except Exception:
            pass


def start_training(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    run_id = db.new_id()
    db.save_run(run_id, config, "queued")
    db.enqueue_job(run_id, run_id, config, "queued")
    threading.Thread(target=_train_worker, args=(run_id, config), daemon=True).start()
    return run_id


def _train_worker(run_id: str, config: dict[str, Any]) -> None:
    db.save_run(run_id, config, "running", {"envSteps": 0})
    emit(run_id, {"type": "status", "payload": {"state": "running", "step": 0}})
    presets = config.get("presets") or {}
    bundle = load_bundle(
        presets.get("fieldId"),
        presets.get("robotId"),
        presets.get("scoringId"),
    )
    total = int((config.get("budget") or {}).get("totalEnvSteps") or 8_192)
    n_envs = int(config.get("nEnvs") or 4)
    demo = bool(config.get("demo", True))
    if demo:
        total = min(total, 4_096)

    logs: list[str] = []
    last_metrics: dict[str, Any] = {}

    def log(msg: str) -> None:
        logs.append(msg)
        emit(run_id, {"type": "log", "payload": {"level": "info", "message": msg}})

    def metrics(m: dict) -> None:
        last_metrics.update(m)
        db.save_run(run_id, config, "running", m, "\n".join(logs))
        emit(run_id, {"type": "metrics", "payload": m})

    try:
        frozen = None
        if (config.get("presets") or {}).get("opponentPolicy") == "frozen_policy":
            from talongym.paths import VAR_DIR
            from talongym.training.ppo import load_trained_policy

            ckpt = VAR_DIR / "ckpts" / "recurrent_ppo.zip"
            if ckpt.exists():
                frozen = load_trained_policy(ckpt)
        result = train_ppo(
            bundle=bundle,
            total_steps=total,
            n_envs=n_envs,
            log=log,
            on_metrics=metrics,
            allow_scripted=demo,
            frozen_policy=frozen,
            should_stop=lambda: is_cancelled(run_id),
        )
        if result.get("cancelled") or is_cancelled(run_id):
            db.save_run(run_id, config, "cancelled", last_metrics, "\n".join(logs))
            emit(run_id, {"type": "status", "payload": {"state": "cancelled", "step": result.get("steps")}})
            return
        if result.get("checkpoint"):
            db.save_artifact(run_id, "checkpoint", result["checkpoint"])
        frames = result.get("frames") or record_policy_episode(scripted_auto, bundle=bundle, seed=3)
        rid = db.save_replay(
            frames,
            {
                "source": "train",
                "runId": run_id,
                "algo": result.get("algo"),
                "trueScore": frames[-1].get("trueScore") if frames else 0,
                "durationS": frames[-1].get("t") if frames else 0,
                "hz": 25,
            },
        )
        java = export_from_replay(frames)
        db.save_artifact(run_id, "roadrunner", java)
        db.save_run(
            run_id,
            config,
            "succeeded",
            {
                "envSteps": result.get("steps"),
                "replayId": rid,
                "algo": result.get("algo"),
                "trueScoreMean": last_metrics.get("trueScoreMean"),
                "shapingMean": last_metrics.get("shapingMean"),
                "objectiveMean": last_metrics.get("objectiveMean"),
            },
            "\n".join(logs),
        )
        emit(
            run_id,
            {
                "type": "status",
                "payload": {"state": "succeeded", "replayId": rid, "step": result.get("steps")},
            },
        )
        emit(run_id, {"type": "rollout", "payload": {"replayId": rid, "frames": frames[::5]}})
    except Exception as exc:
        db.save_run(run_id, config, "failed", {"error": str(exc)}, str(exc))
        emit(run_id, {"type": "error", "payload": {"code": "TRAIN_FAILED", "message": str(exc)}})


def start_evaluation(n_trials: int = 32, policy_name: str = "scripted") -> str:
    eid_holder = {"id": db.new_id()}
    threading.Thread(target=_eval_worker, args=(eid_holder, n_trials, policy_name), daemon=True).start()
    return eid_holder["id"]


def _eval_worker(holder: dict, n_trials: int, policy_name: str) -> None:
    policy = scripted_auto
    report = run_trials(n_trials, policy, seed0=10_000_000, record_best=True)
    frames = report.pop("bestFrames", [])
    replay_id = db.save_replay(frames, {"source": "eval", "trueScore": report.get("bestScore")}) if frames else None
    report["replayId"] = replay_id
    eid = db.save_evaluation(report)
    holder["id"] = eid
    db.save_artifact(eid, "eval_report", str(report.get("mean")))
