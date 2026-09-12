"""Curriculum unlocks from a training-run preset (no torch dependency)."""

from __future__ import annotations

from typing import Any


def curriculum_unlocks(training: dict[str, Any] | None, frac: float) -> list[str]:
    stages = ((training or {}).get("domainRandomization") or {}).get("curriculum") or []
    frac = min(1.0, max(0.0, float(frac)))
    chosen: list[str] = []
    for stage in stages:
        chosen = list(stage.get("unlock") or [])
        if frac <= float(stage.get("untilFrac", 1.0)):
            break
    return chosen


def motif_known_at_t0(training: dict[str, Any] | None, frac: float) -> bool:
    unlocks = curriculum_unlocks(training, frac)
    if "motif_must_sense" in unlocks:
        return False
    return "motif_known_at_t0" in unlocks


def stage_info(training: dict[str, Any] | None, frac: float) -> dict[str, Any]:
    stages = ((training or {}).get("domainRandomization") or {}).get("curriculum") or []
    frac = min(1.0, max(0.0, float(frac)))
    if not stages:
        return {"index": 0, "untilFrac": 1.0, "unlock": []}
    for i, stage in enumerate(stages):
        until = float(stage.get("untilFrac", 1.0))
        if frac <= until:
            return {"index": i, "untilFrac": until, "unlock": list(stage.get("unlock") or [])}
    last = stages[-1]
    return {"index": len(stages) - 1, "untilFrac": float(last.get("untilFrac", 1.0)), "unlock": list(last.get("unlock") or [])}


def teammate_for(training: dict[str, Any] | None, frac: float, default: str = "none") -> str:
    unlocks = curriculum_unlocks(training, frac)
    if "scripted_teammate" in unlocks:
        return "scripted"
    return str(((training or {}).get("presets") or {}).get("teammatePolicy") or default)


def opponent_for(training: dict[str, Any] | None, frac: float, default: str = "none") -> str:
    unlocks = curriculum_unlocks(training, frac)
    if "scripted_opponent" in unlocks:
        return "scripted"
    return str(((training or {}).get("presets") or {}).get("opponentPolicy") or default)


def full_noise(training: dict[str, Any] | None, frac: float) -> bool:
    return "full_noise" in curriculum_unlocks(training, frac)


def scripted_launch(training: dict[str, Any] | None, frac: float) -> bool:
    return "scripted_launch" in curriculum_unlocks(training, frac)


def ballistic_launch(training: dict[str, Any] | None, frac: float) -> bool:
    return "ballistic_launch" in curriculum_unlocks(training, frac)


def objective_value(report: dict[str, Any], objective: str) -> float:
    if objective == "p10_true_score":
        return float(report.get("p10") or 0.0)
    if objective == "lcb_true_score":
        return float(report.get("lo") or 0.0)
    return float(report.get("mean") or 0.0)
