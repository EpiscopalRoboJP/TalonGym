"""Fit robot constraints from Dashboard / WPILOG-shaped pose samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _rows(samples: list[dict]) -> np.ndarray:
    rows = []
    for s in samples:
        t = float(s.get("t", s.get("time", 0.0)))
        x = float(s.get("x", s.get("x_in", 0.0)))
        y = float(s.get("y", s.get("y_in", 0.0)))
        h = float(s.get("headingDeg", s.get("heading_deg", 0.0)))
        rows.append((t, x, y, np.deg2rad(h)))
    return np.asarray(rows, dtype=np.float64)


def fit_from_log(samples: list[dict]) -> dict[str, float]:
    if not samples:
        return {"maxVelInPerS": 30.0, "maxAccelInPerS2": 30.0, "positionNoiseStdIn": 0.3, "intakeCycleTimeS": 0.4, "rmse": 0.0}
    arr = _rows(samples)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    if arr.shape[0] < 3:
        return {"maxVelInPerS": 30.0, "maxAccelInPerS2": 30.0, "positionNoiseStdIn": 0.3, "intakeCycleTimeS": 0.4, "rmse": 0.0}
    dt = np.diff(arr[:, 0])
    dt = np.clip(dt, 1e-3, None)
    dxy = np.diff(arr[:, 1:3], axis=0)
    speed = np.hypot(dxy[:, 0], dxy[:, 1]) / dt
    accel = np.diff(speed) / np.clip(dt[1:], 1e-3, None)
    max_vel = float(np.percentile(speed, 95))
    max_acc = float(np.percentile(np.abs(accel), 95)) if accel.size else 30.0
    # Residual vs mean velocity as a slip/noise proxy.
    rmse = float(np.std(speed - np.mean(speed))) if speed.size else 0.0
    noise = float(np.std(dxy))
    intakes = [s for s in samples if s.get("event") == "intake"]
    cycle = 0.4
    if len(intakes) >= 2:
        ts = np.diff(sorted(float(s.get("t", 0.0)) for s in intakes))
        ts = ts[ts > 0.05]
        if ts.size:
            cycle = float(np.median(ts))
    return {
        "maxVelInPerS": float(np.clip(max_vel, 4.0, 80.0)),
        "maxAccelInPerS2": float(np.clip(max_acc, 4.0, 120.0)),
        "positionNoiseStdIn": float(np.clip(noise, 0.01, 4.0)),
        "intakeCycleTimeS": float(np.clip(cycle, 0.1, 2.0)),
        "rmse": rmse,
    }


def overlay_robot(robot: dict[str, Any], fit: dict[str, float]) -> dict[str, Any]:
    out = json.loads(json.dumps(robot))
    cons = dict(out.get("constraints") or {})
    cons["maxVelInPerS"] = fit["maxVelInPerS"]
    cons["maxAccelInPerS2"] = fit["maxAccelInPerS2"]
    out["constraints"] = cons
    odo = dict(out.get("odometry") or {})
    odo["positionNoiseStdIn"] = fit["positionNoiseStdIn"]
    out["odometry"] = odo
    mech = dict(out.get("mechanisms") or {})
    mech["intakeCycleTimeS"] = fit["intakeCycleTimeS"]
    out["mechanisms"] = mech
    out["calibration"] = {"rmse": fit["rmse"], "source": "pose_log"}
    return out


def fit_file(log_path: Path, robot: dict[str, Any]) -> dict[str, Any]:
    raw = json.loads(Path(log_path).read_text(encoding="utf-8"))
    samples = raw if isinstance(raw, list) else raw.get("samples") or raw.get("poses") or []
    fit = fit_from_log(list(samples))
    return overlay_robot(robot, fit)
