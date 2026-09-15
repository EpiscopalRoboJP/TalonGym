"""Field CAD coordinate layout and preset sync.

import-field-cad records where every visible CAD part instance sits, in the field preset's
FTC frame (inches, origin at field center on the tile surface, +z up). Presets opt in per
element / game piece type with ``cadPart``; ``sync_field_to_cad_layout`` then snaps those
element poses and regenerates game piece spawns so every ball drawn in Lab is a sim piece.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from pathlib import Path
from typing import Any

FIELD_HALF_IN = 72.0
# Pieces closer than this (in x/y) are one spawn group: a stack in a flower, a row in a garden.
SPAWN_CLUSTER_IN = 4.0
# A spawn group is named after the nearest preset element within this x/y distance.
SPAWN_NAME_RADIUS_IN = 12.0
# Tape strips outlining one zone touch or nearly touch; separate zones are much farther apart.
TOUCH_GAP_IN = 1.5
# Start slots are cleared for the largest legal FTC robot footprint (18 in square).
START_ROBOT_HALF_IN = 9.0
START_CLEARANCE_IN = 0.5
_INSTANCE_SUFFIX = re.compile(r"_\d+$")


def cad_part_name(instance_name: str) -> str:
    return _INSTANCE_SUFFIX.sub("", instance_name)


def lab_bounds_to_placement(instance: str, lo: Any, hi: Any, tile_top: float) -> dict[str, Any]:
    """Lab Y-up bounds (x, height, -y) -> FTC-frame center and size."""
    cx, cy, cz = (0.5 * (float(lo[i]) + float(hi[i])) for i in range(3))
    x, y = round(cx, 2), round(-cz, 2)
    return {
        "part": cad_part_name(instance),
        "instance": instance,
        "x": x,
        "y": y,
        "z": round(cy - tile_top, 2),
        "sizeIn": [round(float(hi[0] - lo[0]), 2), round(float(hi[2] - lo[2]), 2), round(float(hi[1] - lo[1]), 2)],
        "inField": abs(x) < FIELD_HALF_IN and abs(y) < FIELD_HALF_IN,
    }


def build_layout(placements: list[dict[str, Any]], field: dict[str, Any], tile_top: float) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "fieldId": field.get("id"),
        "coordinateSystem": field.get("coordinateSystem") or {},
        "units": "inch",
        "tileSurfaceCadHeightIn": round(tile_top, 3),
        "parts": sorted(placements, key=lambda p: (p["part"], p["x"], p["y"], p["z"])),
    }


def _dist_xy(a: dict[str, Any], b: dict[str, Any]) -> float:
    return ((float(a["x"]) - float(b["x"])) ** 2 + (float(a["y"]) - float(b["y"])) ** 2) ** 0.5


def _assign(elements: list[dict[str, Any]], instances: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Instance per element minimizing total x/y travel (exact up to 8 elements, greedy beyond)."""
    poses = [el.get("pose") or {"x": 0.0, "y": 0.0} for el in elements]
    if not instances:
        return [None] * len(elements)
    if len(elements) <= 8 and len(instances) <= 8:
        best: tuple[float, tuple[int, ...]] | None = None
        k = min(len(elements), len(instances))
        for combo in itertools.permutations(range(len(instances)), k):
            cost = sum(_dist_xy(poses[i], instances[j]) for i, j in enumerate(combo))
            if best is None or cost < best[0]:
                best = (cost, combo)
        assert best is not None
        out: list[dict[str, Any] | None] = [instances[j] for j in best[1]]
        return out + [None] * (len(elements) - k)
    free = list(instances)
    picked: list[dict[str, Any] | None] = []
    for pose in poses:
        if not free:
            picked.append(None)
            continue
        inst = min(free, key=lambda p: _dist_xy(pose, p))
        free.remove(inst)
        picked.append(inst)
    return picked


def _bbox(p: dict[str, Any]) -> tuple[float, float, float, float]:
    sx, sy = float(p["sizeIn"][0]) / 2.0, float(p["sizeIn"][1]) / 2.0
    return p["x"] - sx, p["y"] - sy, p["x"] + sx, p["y"] + sy


def _touching_groups(instances: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Single-link groups of instances whose x/y boxes are within TOUCH_GAP_IN of each other."""
    groups: list[list[dict[str, Any]]] = []
    for inst in sorted(instances, key=lambda p: (p["x"], p["y"])):
        ax0, ay0, ax1, ay1 = _bbox(inst)
        near = []
        for g in groups:
            for q in g:
                bx0, by0, bx1, by1 = _bbox(q)
                if ax0 <= bx1 + TOUCH_GAP_IN and bx0 <= ax1 + TOUCH_GAP_IN and ay0 <= by1 + TOUCH_GAP_IN and by0 <= ay1 + TOUCH_GAP_IN:
                    near.append(g)
                    break
        merged = [inst]
        for g in near:
            merged.extend(g)
            groups.remove(g)
        groups.append(merged)
    return groups


def _group_center(group: list[dict[str, Any]]) -> dict[str, Any]:
    boxes = [_bbox(p) for p in group]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return {
        "x": round(0.5 * (x0 + x1), 2),
        "y": round(0.5 * (y0 + y1), 2),
        "instances": sorted(p["instance"] for p in group),
    }


def _collider_boxes(field: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    boxes = []
    for el in field.get("elements") or []:
        if not el.get("isCollider") or "perimeter" in (el.get("tags") or []):
            continue
        pose, shape = el.get("pose") or {}, el.get("shape") or {}
        if shape.get("kind") == "circle":
            hx = hy = float(shape.get("radius") or 0.0)
        else:
            hx, hy = float(shape.get("width") or 0.0) / 2.0, float(shape.get("depth") or 0.0) / 2.0
        x, y = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
        boxes.append((x - hx, y - hy, x + hx, y + hy))
    return boxes


def _clear_start_slots(field: dict[str, Any]) -> dict[str, Any]:
    """Slide start slots along y (then x) just far enough that an 18 in robot misses every collider."""
    boxes = _collider_boxes(field)
    limit = FIELD_HALF_IN - START_ROBOT_HALF_IN

    def blocked(x: float, y: float) -> bool:
        r = START_ROBOT_HALF_IN
        return any(x - r < b[2] and x + r > b[0] and y - r < b[3] and y + r > b[1] for b in boxes)

    moved: dict[str, Any] = {}
    for slot in field.get("startSlots") or []:
        pose = slot.get("pose") or {}
        x, y = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
        if not blocked(x, y):
            continue
        clear = START_ROBOT_HALF_IN + START_CLEARANCE_IN
        candidates = [(x, b[3] + clear) for b in boxes] + [(x, b[1] - clear) for b in boxes]
        candidates += [(b[2] + clear, y) for b in boxes] + [(b[0] - clear, y) for b in boxes]
        ok = [
            (cx, cy)
            for cx, cy in candidates
            if abs(cx) <= limit and abs(cy) <= limit and not blocked(cx, cy)
        ]
        if not ok:
            moved[slot["id"]] = {"from": (x, y), "to": None}
            continue
        nx, ny = min(ok, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        pose["x"], pose["y"] = round(nx, 2), round(ny, 2)
        moved[slot["id"]] = {"from": (x, y), "to": (pose["x"], pose["y"])}
    return moved


def _clusters(pieces: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for piece in sorted(pieces, key=lambda p: (p["x"], p["y"], p["z"])):
        near = [g for g in groups if any(_dist_xy(piece, q) <= SPAWN_CLUSTER_IN for q in g)]
        merged = [piece]
        for g in near:
            merged.extend(g)
            groups.remove(g)
        groups.append(merged)
    return groups


def sync_field_to_cad_layout(field: dict[str, Any], layout: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy of ``field`` with cadPart elements snapped and cadPart piece spawns rebuilt."""
    out = copy.deepcopy(field)
    parts = [p for p in layout.get("parts") or [] if p.get("inField")]
    report: dict[str, Any] = {"elements": {}, "spawns": {}, "missingCadParts": []}

    by_part: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for el in out.get("elements") or []:
        cad = el.get("cadPart")
        if cad:
            key = (cad,) if isinstance(cad, str) else tuple(sorted(cad))
            by_part.setdefault(key, []).append(el)
    for key, elements in by_part.items():
        instances = [p for p in parts if p["part"] in key]
        missing = [k for k in key if not any(p["part"] == k for p in instances)]
        report["missingCadParts"].extend(missing)
        if not instances:
            continue
        groups = [_group_center(g) for g in _touching_groups(instances)]
        if len(elements) <= len(groups):
            picks = _assign(elements, groups)
        else:
            # More elements than CAD groups (a park zone drawn by the same tape as its
            # loading zone): each takes its nearest group.
            picks = [min(groups, key=lambda g, el=el: _dist_xy(el["pose"], g)) for el in elements]
        for el, group in zip(elements, picks, strict=True):
            if group is None:
                continue
            pose = el.setdefault("pose", {"x": 0.0, "y": 0.0, "headingDeg": 0})
            before = (pose.get("x"), pose.get("y"))
            pose["x"], pose["y"] = group["x"], group["y"]
            report["elements"][el["id"]] = {"from": before, "to": (group["x"], group["y"]), "instances": group["instances"]}
    report["startSlots"] = _clear_start_slots(out)

    named = [el for el in out.get("elements") or [] if el.get("pose")]
    cad_types = {gp["typeId"]: gp["cadPart"] for gp in out.get("gamePieces") or [] if gp.get("cadPart")}
    new_spawns: list[dict[str, Any]] = []
    for type_id, part in cad_types.items():
        pieces = [p for p in parts if p["part"] == part]
        if not pieces:
            report["missingCadParts"].append(part)
            continue
        used: dict[str, int] = {}
        for group in _clusters(pieces):
            cx = sum(p["x"] for p in group) / len(group)
            cy = sum(p["y"] for p in group) / len(group)
            center = {"x": cx, "y": cy}
            anchor = min(named, key=lambda el: _dist_xy(center, el["pose"]), default=None)
            if anchor is not None and _dist_xy(center, anchor["pose"]) <= SPAWN_NAME_RADIUS_IN:
                base = f"{anchor['id']}_{type_id}"
            else:
                base = f"{type_id}_cad"
            used[base] = used.get(base, 0) + 1
            spawn_id = base if used[base] == 1 else f"{base}_{used[base]}"
            poses = [
                {"x": p["x"], "y": p["y"], "headingDeg": 0, "z": p["z"]}
                for p in sorted(group, key=lambda p: (p["z"], p["x"], p["y"]))
            ]
            new_spawns.append({"id": spawn_id, "pieceTypeId": type_id, "poses": poses})
            report["spawns"][spawn_id] = len(poses)
    if cad_types:
        kept = [
            sp
            for sp in out.get("spawns") or []
            if sp.get("pieceTypeId") not in cad_types or sp.get("preloadEligible")
        ]
        out["spawns"] = sorted(new_spawns, key=lambda sp: sp["id"]) + kept
    return out, report


def _inline_json(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        body = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {_inline_json(v)}" for k, v in value.items())
        return "{ " + body + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_inline_json(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def format_preset_json(value: Any, depth: int = 0) -> str:
    """House style for presets: nest two levels, then inline (lists of objects stay expanded)."""
    if not isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    holds_objects = isinstance(value, list) and any(isinstance(v, dict) for v in value)
    if depth >= 3 and not holds_objects:
        return _inline_json(value)
    pad = "  " * depth
    if isinstance(value, dict):
        rows = [f"{pad}  {json.dumps(k, ensure_ascii=False)}: {format_preset_json(v, depth + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(rows) + f"\n{pad}}}"
    rows = [f"{pad}  {format_preset_json(v, depth + 1)}" for v in value]
    return "[\n" + ",\n".join(rows) + f"\n{pad}]"


def write_preset_json(path: Path, value: Any) -> None:
    Path(path).write_text(format_preset_json(value) + "\n", encoding="utf-8")
