from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from talongym.paths import PRESETS_DIR, SCHEMAS_DIR

ENGINE_CAPABILITIES = frozenset(
    {
        "planar_drive",
        "static_colliders",
        "dynamic_game_pieces",
        "trigger_volumes",
        "occluders",
        "apriltag_vision",
        "randomized_match_variable",
        "partial_observability_sentinel",
        "n_stage_scoring",
        "end_of_phase_scoring",
        "alliance_aggregate_scoring",
        "ranking_point_thresholds",
        "mid_episode_geometry_change",
        "retained_queue",
        "scripted_mechanism_fsm",
        "multi_robot_collision",
        "phase_clock",
        "mesh_field_collision",
    }
)

LATEST_KNOWN_MANUAL = {
    "decode": "TU32",
    "into_the_deep": "archive-2025",
    "centerstage": "archive-2024",
    "biobuzz": "V1",
}


def _load_schema(name: str) -> dict[str, Any]:
    path = SCHEMAS_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


_VALIDATORS = {
    "field": Draft202012Validator(_load_schema("field-preset.schema.json")),
    "robot": Draft202012Validator(_load_schema("robot-preset.schema.json")),
    "scoring": Draft202012Validator(_load_schema("scoring-rules-preset.schema.json")),
    "training": Draft202012Validator(_load_schema("training-run-config.schema.json")),
}


class PresetError(ValueError):
    pass


def validate_document(kind: str, document: dict[str, Any]) -> list[str]:
    if kind not in _VALIDATORS:
        raise PresetError(f"Unknown preset kind {kind}")
    errors = sorted(_VALIDATORS[kind].iter_errors(document), key=lambda e: list(e.path))
    messages = [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]
    caps = document.get("requiredCapabilities") or []
    unknown = [c for c in caps if c not in ENGINE_CAPABILITIES]
    if unknown:
        messages.append(f"unsupportedCapabilities: {unknown}")
    return messages


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class LoadedPresets:
    field: dict[str, Any]
    robot: dict[str, Any]
    scoring: dict[str, Any]
    training: dict[str, Any] | None = None
    field_path: Path | None = None
    robot_path: Path | None = None
    scoring_path: Path | None = None


def _index_presets() -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {"field": {}, "robot": {}, "scoring": {}, "training": {}}
    for path in PRESETS_DIR.rglob("*.json"):
        try:
            data = load_json(path)
        except json.JSONDecodeError:
            continue
        pid = data.get("id")
        if not pid:
            continue
        if "fieldSizeIn" in data:
            index["field"][pid] = path
        elif "drivetrain" in data:
            index["robot"][pid] = path
        elif "nodes" in data and "scoreChannels" in data:
            index["scoring"][pid] = path
        elif "algorithm" in data:
            index["training"][pid] = path
    return index


_INDEX: dict[str, dict[str, Path]] | None = None


def preset_index(refresh: bool = False) -> dict[str, dict[str, Path]]:
    global _INDEX
    if _INDEX is None or refresh:
        _INDEX = _index_presets()
    return _INDEX


def list_presets(kind: str) -> list[dict[str, Any]]:
    items = []
    for pid, path in preset_index()[kind].items():
        data = load_json(path)
        season = (data.get("season") or {}).get("slug")
        revision = (data.get("provenance") or {}).get("manualRevision")
        stale = bool(season and revision and LATEST_KNOWN_MANUAL.get(season) and revision != LATEST_KNOWN_MANUAL[season])
        items.append(
            {
                "id": pid,
                "displayName": data.get("displayName"),
                "season": season,
                "manualRevision": revision,
                "stale": stale,
                "verifyAgainstManual": (data.get("provenance") or {}).get("verifyAgainstManual", False),
                "computeProfile": data.get("computeProfile"),
                "path": str(path),
            }
        )
    return sorted(items, key=lambda x: x["id"])


def load_preset(kind: str, preset_id: str) -> dict[str, Any]:
    path = preset_index()[kind].get(preset_id)
    if path is None:
        raise PresetError(f"Unknown {kind} preset '{preset_id}'")
    data = load_json(path)
    errs = validate_document(kind, data)
    if errs:
        raise PresetError(f"{kind} preset {preset_id} invalid: " + "; ".join(errs[:8]))
    return data


def load_bundle(
    field_id: str | None = None,
    robot_id: str | None = None,
    scoring_id: str | None = None,
    training_id: str | None = None,
) -> LoadedPresets:
    from talongym.presets.defaults import get_defaults

    defaults = get_defaults()
    field_id = field_id or defaults["fieldId"]
    robot_id = robot_id or defaults["robotId"]
    scoring_id = scoring_id or defaults["scoringId"]
    training_id = training_id or defaults.get("trainingId")
    field = load_preset("field", field_id)
    robot = load_preset("robot", robot_id)
    scoring = load_preset("scoring", scoring_id)
    training = load_preset("training", training_id) if training_id else None
    idx = preset_index()
    return LoadedPresets(
        field=field,
        robot=robot,
        scoring=scoring,
        training=training,
        field_path=idx["field"][field_id],
        robot_path=idx["robot"][robot_id],
        scoring_path=idx["scoring"][scoring_id],
    )


def save_preset(kind: str, document: dict[str, Any], dest: Path) -> Path:
    errs = validate_document(kind, document)
    if errs:
        raise PresetError("; ".join(errs))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    preset_index(refresh=True)
    return dest
