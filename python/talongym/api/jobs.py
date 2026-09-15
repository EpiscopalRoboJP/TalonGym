"""In-process training workers. SQLite `jobs` is a status log, not a work queue."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from talongym import paths
from talongym.api import db
from talongym.api.cad_frames import ensure_background_asset
from talongym.export.roadrunner import export_from_replay
from talongym.presets.loader import load_bundle
from talongym.training.policies import scripted_auto
from talongym.training.ppo import record_policy_episode, train_ppo

_listeners: dict[str, list[Callable[[dict], None]]] = {}
_cancel: set[str] = set()
_latest: dict[str, dict[str, dict[str, Any]]] = {}


def request_cancel(run_id: str) -> None:
    _cancel.add(run_id)


def is_cancelled(run_id: str) -> bool:
    return run_id in _cancel


def clear_cancel(run_id: str) -> None:
    _cancel.discard(run_id)


def subscribe(run_id: str, fn: Callable[[dict], None]) -> None:
    _listeners.setdefault(run_id, []).append(fn)


def unsubscribe(run_id: str, fn: Callable[[dict], None]) -> None:
    lst = _listeners.get(run_id) or []
    if fn in lst:
        lst.remove(fn)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def ws_rollout_frames(frames: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Downsample and JSON-sanitize frames so Lab WebSocket payloads stay small."""
    if not frames:
        return []
    keep = (
        "t",
        "trueScore",
        "robots",
        "pieces",
        "elements",
        "fieldSizeIn",
        "vision",
        "aprilTags",
        "backgroundAsset",
        "robotDesign",
        "cadManifest",
        "cadSourceSha256",
        "cadAssetVersion",
        "gamePieces",
    )
    step = max(1, len(frames) // 48)
    out: list[dict[str, Any]] = []
    for fr in frames[::step][:48]:
        if not isinstance(fr, dict):
            continue
        slim = ensure_background_asset({k: _jsonable(fr.get(k)) for k in keep if k in fr}) or {}
        out.append(slim)
    return out


def emit(run_id: str, msg: dict[str, Any]) -> None:
    kind = str(msg.get("type") or "")
    if kind == "rollout":
        payload = dict(msg.get("payload") or {})
        payload["frames"] = ws_rollout_frames(payload.get("frames") or [])
        msg = {**msg, "payload": payload}
    msg = _jsonable(msg)
    if kind:
        _latest.setdefault(run_id, {})[kind] = msg
    for fn in list(_listeners.get(run_id) or []):
        try:
            fn(msg)
        except Exception:
            pass


def latest_messages(run_id: str) -> list[dict[str, Any]]:
    return list((_latest.get(run_id) or {}).values())


def _checkpoint_exists(path: str | None) -> bool:
    if not path:
        return False
    try:
        return Path(str(path)).exists()
    except OSError:
        return False


def resolve_frozen_opponent(
    config: dict[str, Any],
    bundle: Any,
    log: Callable[[str], None],
    run_id: str | None = None,
) -> Any:
    """Load a frozen opponent from a configured checkpoint. Never a hardcoded zip."""
    presets = config.get("presets") if isinstance(config.get("presets"), dict) else {}
    training_presets = ((bundle.training or {}).get("presets") if bundle is not None else None) or {}
    wanted = presets.get("opponentPolicy") == "frozen_policy" or training_presets.get("opponentPolicy") == "frozen_policy"
    if not wanted:
        return None
    path = config.get("frozenPolicyPath") or presets.get("frozenPolicyPath")
    if not path and bundle is not None:
        path = (bundle.training or {}).get("frozenPolicyPath")
    source_run = config.get("frozenPolicyRunId") or presets.get("frozenPolicyRunId")
    if not path and source_run:
        arts = [a for a in db.list_artifacts(str(source_run)) if a.get("kind") == "checkpoint"]
        if arts:
            path = arts[-1].get("payload")
    if not path and run_id:
        arts = [a for a in db.list_artifacts(run_id) if a.get("kind") == "checkpoint"]
        if arts:
            path = arts[-1].get("payload")
    if not path:
        log("frozen_policy requested but no checkpoint path or run artifact was configured; skipping opponent")
        return None
    if not _checkpoint_exists(str(path)):
        log(f"frozen_policy checkpoint missing at {path}; skipping opponent")
        return None
    from talongym.training.ppo import load_trained_policy

    try:
        return load_trained_policy(str(path))
    except Exception as exc:
        log(f"frozen_policy failed to load {path}: {exc}; skipping opponent")
        return None


def start_training(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    run_id = db.new_id()
    db.save_run(run_id, config, "queued")
    db.enqueue_job(run_id, run_id, config, "queued")
    threading.Thread(target=_train_worker, args=(run_id, config), daemon=True).start()
    return run_id


def _resolved_budget(config: dict[str, Any], training: dict[str, Any] | None) -> tuple[int, int, bool, dict[str, Any]]:
    from talongym.training.compute import resolve_training

    resolved = resolve_training(
        training,
        easy=bool(config.get("easy")),
        profile=config.get("computeProfile"),
    )
    demo = bool(config.get("demo", False))
    body_budget = config.get("budget") if isinstance(config.get("budget"), dict) else None
    if body_budget and body_budget.get("totalEnvSteps"):
        total = int(body_budget["totalEnvSteps"])
    else:
        total = int((resolved.get("budget") or {}).get("totalEnvSteps") or 8192)
    if config.get("nEnvs") is not None:
        n_envs = int(config["nEnvs"])
    else:
        n_envs = int(resolved.get("nEnvs") or 4)
    if demo:
        total = min(total, 4096)
        n_envs = min(max(1, n_envs), 4)
    return max(1, total), max(1, n_envs), demo, resolved


def _train_worker(run_id: str, config: dict[str, Any]) -> None:
    if is_cancelled(run_id):
        db.save_run(run_id, config, "cancelled", {"envSteps": 0})
        emit(run_id, {"type": "status", "payload": {"state": "cancelled", "step": 0}})
        clear_cancel(run_id)
        return
    db.save_run(run_id, config, "running", {"envSteps": 0})
    emit(run_id, {"type": "status", "payload": {"state": "running", "step": 0}})
    presets = dict(config.get("presets") or {})
    training_id = presets.get("trainingId") or None
    if config.get("easy"):
        from talongym.training.compute import easy_training_id

        training_id = easy_training_id(training_id)
        presets["trainingId"] = training_id
        config = {**config, "presets": presets}
    bundle = load_bundle(
        presets.get("fieldId") or None,
        presets.get("robotId") or None,
        presets.get("scoringId") or None,
        training_id,
    )
    total, n_envs, demo, resolved = _resolved_budget(config, bundle.training)
    if bundle.training is not None:
        bundle.training = {**bundle.training, **resolved}
    logs: list[str] = []
    last_metrics: dict[str, Any] = {
        "envSteps": 0,
        "nEnvs": n_envs,
        "algo": "recurrent_ppo",
        "computeProfile": resolved.get("computeProfile"),
    }

    def log(msg: str) -> None:
        logs.append(msg)
        emit(run_id, {"type": "log", "payload": {"level": "info", "message": msg}})

    def metrics(m: dict) -> None:
        last_metrics.update(_jsonable(m) if isinstance(m, dict) else {})
        state = "cancelling" if is_cancelled(run_id) else "running"
        db.save_run(run_id, config, state, last_metrics, "\n".join(logs))
        emit(run_id, {"type": "metrics", "payload": dict(last_metrics)})

    def rollout(frames: list[dict[str, Any]]) -> None:
        emit(run_id, {"type": "rollout", "payload": {"frames": frames[::5] if frames else []}})

    try:
        log(
            f"computeProfile={resolved.get('computeProfile')} nEnvs={n_envs} "
            f"training={(bundle.training or {}).get('id')} "
            f"requested={resolved.get('requestedComputeProfile')}"
        )
        frozen = resolve_frozen_opponent(config, bundle, log, run_id=run_id)
        algo_name = str(
            ((config.get("algorithm") or {}).get("name") if config.get("algorithm") else None)
            or ((bundle.training or {}).get("algorithm") or {}).get("name")
            or "recurrent_ppo"
        )
        train_kwargs = dict(
            bundle=bundle,
            total_steps=total,
            n_envs=n_envs,
            log=log,
            on_metrics=metrics,
            on_rollout=rollout,
            allow_scripted=demo,
            frozen_policy=frozen,
            should_stop=lambda: is_cancelled(run_id),
            save_dir=paths.VAR_DIR / "ckpts" / run_id,
            resume=bool(config.get("resume")),
            demo=demo,
            match_setup=config.get("matchSetup"),
        )
        if algo_name == "rllib_ppo" and not demo:
            try:
                from talongym.training.rllib import train_rllib

                log("rllib_ppo is a one-shot toy trainer; not a production scale path")
                result = train_rllib(total_steps=total, log=log)
            except RuntimeError as exc:
                log(f"{exc}; falling back to RecurrentPPO")
                train_kwargs["n_envs"] = min(n_envs, 64)
                result = train_ppo(**train_kwargs)
        else:
            result = train_ppo(**train_kwargs)
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
        merged = {
            **last_metrics,
            **(result.get("metrics") or {}),
            "envSteps": result.get("steps"),
            "replayId": rid,
            "algo": result.get("algo"),
            "checkpoint": result.get("checkpoint"),
        }
        last_metrics.update(merged)
        db.save_run(run_id, config, "succeeded", last_metrics, "\n".join(logs))
        emit(
            run_id,
            {
                "type": "status",
                "payload": {"state": "succeeded", "replayId": rid, "step": result.get("steps")},
            },
        )
        emit(run_id, {"type": "metrics", "payload": dict(last_metrics)})
        emit(run_id, {"type": "rollout", "payload": {"replayId": rid, "frames": frames[::5]}})
    except Exception as exc:
        db.save_run(run_id, config, "failed", {**last_metrics, "error": str(exc)}, str(exc))
        emit(run_id, {"type": "error", "payload": {"code": "TRAIN_FAILED", "message": str(exc)}})
        emit(run_id, {"type": "status", "payload": {"state": "failed"}})
    finally:
        clear_cancel(run_id)
        row = db.get_run(run_id)
        if row:
            db.enqueue_job(run_id, run_id, config, row.get("state") or "done")
