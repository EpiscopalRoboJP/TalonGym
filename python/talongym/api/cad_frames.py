"""Stamp CAD background paths onto snapshots that predate backgroundAsset."""

from __future__ import annotations

from typing import Any

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
