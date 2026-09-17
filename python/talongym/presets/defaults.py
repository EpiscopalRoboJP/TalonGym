from __future__ import annotations

import json
from typing import Any

from talongym import paths
from talongym.presets.loader import PresetError, load_json, load_preset, preset_index

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
    season = None
    field_id = ids.get("fieldId")
    if field_id:
        field = None
        path = preset_index()["field"].get(field_id)
        if path is not None:
            field = load_json(path)
        else:
            try:
                from talongym.api import db

                field = db.load_runtime_document("field", field_id)
            except Exception as exc:
                raise PresetError(f"Unknown field preset '{field_id}'") from exc
        season = (field.get("season") or {}).get("slug")
    return {
        **ids,
        "season": season,
    }


def _known_preset(kind: str, preset_id: str) -> bool:
    if preset_id in preset_index().get(kind, {}):
        return True
    try:
        from talongym.api import db

        doc = db.get_preset(preset_id)
    except Exception:
        return False
    return bool(doc and doc.get("_kind") == kind)


def _load_known(kind: str, preset_id: str) -> dict[str, Any]:
    if preset_id in preset_index().get(kind, {}):
        return load_preset(kind, preset_id)
    from talongym.api import db

    return db.load_runtime_document(kind, preset_id)


def _validate_bundle(field_id: str, robot_id: str, scoring_id: str, training_id: str | None) -> None:
    if not _known_preset("field", field_id):
        raise PresetError(f"Unknown field preset '{field_id}'")
    if not _known_preset("robot", robot_id):
        raise PresetError(f"Unknown robot preset '{robot_id}'")
    if not _known_preset("scoring", scoring_id):
        raise PresetError(f"Unknown scoring preset '{scoring_id}'")
    if training_id and not _known_preset("training", training_id):
        raise PresetError(f"Unknown training preset '{training_id}'")
    scoring = _load_known("scoring", scoring_id)
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
