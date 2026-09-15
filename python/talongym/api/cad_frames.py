"""Stamp CAD background paths and piece sizes onto snapshots that predate them."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from talongym.presets.loader import load_json, preset_index

BIOBUZZ_GLB = "seasons/biobuzz_2026/field.glb"


def ensure_background_asset(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(frame, dict):
        return frame
    if frame.get("backgroundAsset"):
        return frame
    els = frame.get("elements") or []
    for el in els:
        if not isinstance(el, dict):
            continue
        tags = el.get("tags") or []
        if el.get("id") in {"red_cell_up", "hive_frame_west", "red_garden"} or el.get("type") == "hive_frame" or "hive" in tags:
            out = dict(frame)
            out["backgroundAsset"] = BIOBUZZ_GLB
            return out
    return frame


@lru_cache(maxsize=1)
def _piece_radii() -> dict[str, float]:
    radii: dict[str, float] = {}
    for path in preset_index()["field"].values():
        for gp in load_json(path).get("gamePieces") or []:
            shape = gp.get("shape") or {}
            if shape.get("kind") == "circle" and shape.get("radius"):
                radii.setdefault(gp["typeId"], float(shape["radius"]))
            elif shape.get("width") or shape.get("depth"):
                radii.setdefault(gp["typeId"], max(float(shape.get("width") or 0), float(shape.get("depth") or 0)) / 2.0)
    return radii


def ensure_piece_radius(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    """Replays recorded before frames carried piece radius get it from the field preset by typeId."""
    if not isinstance(frame, dict):
        return frame
    pieces = frame.get("pieces")
    if not isinstance(pieces, list) or all(not isinstance(p, dict) or "radius" in p for p in pieces):
        return frame
    radii = _piece_radii()
    filled = []
    for p in pieces:
        if isinstance(p, dict) and "radius" not in p and p.get("typeId") in radii:
            p = {**p, "radius": radii[p["typeId"]]}
        filled.append(p)
    return {**frame, "pieces": filled}


def normalize_frame(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    return ensure_piece_radius(ensure_background_asset(frame))
