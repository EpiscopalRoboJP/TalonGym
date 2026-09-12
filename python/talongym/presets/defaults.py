from __future__ import annotations

import json
from typing import Any

from talongym import paths
from talongym.presets.loader import PresetError, load_preset, preset_index

REPO_DEFAULTS = paths.PRESETS_DIR / "defaults.json"
_KEYS = ("fieldId", "robotId", "scoringId", "trainingId")


def _user_path():
    return paths.VAR_DIR / "defaults.json"


def get_defaults() -> dict[str, str]:
    data = json.loads(REPO_DEFAULTS.read_text(encoding="utf-8"))
    user = _user_path()
    if user.exists():
        try:
            overlay = json.loads(user.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            overlay = {}
        if isinstance(overlay, dict):
            for key in _KEYS:
                if overlay.get(key):
                    data[key] = overlay[key]
    return {key: str(data[key]) for key in _KEYS if data.get(key)}


def describe_defaults() -> dict[str, Any]:
    ids = get_defaults()
    field = load_preset("field", ids["fieldId"])
    return {
        **ids,
        "season": (field.get("season") or {}).get("slug"),
    }


def _validate_bundle(field_id: str, robot_id: str, scoring_id: str, training_id: str | None) -> None:
    idx = preset_index()
    if field_id not in idx["field"]:
        raise PresetError(f"Unknown field preset '{field_id}'")
    if robot_id not in idx["robot"]:
        raise PresetError(f"Unknown robot preset '{robot_id}'")
    if scoring_id not in idx["scoring"]:
        raise PresetError(f"Unknown scoring preset '{scoring_id}'")
    if training_id and training_id not in idx["training"]:
        raise PresetError(f"Unknown training preset '{training_id}'")
    scoring = load_preset("scoring", scoring_id)
    expected = scoring.get("fieldPresetId")
    if expected and expected != field_id:
        raise PresetError(f"scoring preset {scoring_id} fieldPresetId '{expected}' != field '{field_id}'")


def set_defaults(
    field_id: str | None = None,
    robot_id: str | None = None,
    scoring_id: str | None = None,
    training_id: str | None = None,
) -> dict[str, str]:
    current = get_defaults()
    if training_id:
        training = load_preset("training", training_id)
        block = training.get("presets") or {}
        current["fieldId"] = str(block["fieldId"])
        current["robotId"] = str(block["robotId"])
        current["scoringId"] = str(block["scoringId"])
        current["trainingId"] = training_id
    if field_id:
        current["fieldId"] = field_id
    if robot_id:
        current["robotId"] = robot_id
    if scoring_id:
        current["scoringId"] = scoring_id
    if not current.get("fieldId") or not current.get("robotId") or not current.get("scoringId"):
        raise PresetError("defaults require fieldId, robotId, and scoringId")
    _validate_bundle(
        current["fieldId"],
        current["robotId"],
        current["scoringId"],
        current.get("trainingId"),
    )
    dest = _user_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return current
