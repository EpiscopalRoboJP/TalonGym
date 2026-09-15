"""Stamp CAD background paths onto snapshots that predate backgroundAsset."""

from __future__ import annotations

from typing import Any

from talongym.assets.cad_common import CAD_GENERATOR_VERSION

BIOBUZZ_GLB = "seasons/biobuzz_2026/field.glb"
BIOBUZZ_MANIFEST = "seasons/biobuzz_2026/cad_manifest.json"
BIOBUZZ_FIELD_SHA256 = "05b35961c7df847741031f00fda73ddd068537f809e92a11b1cd59a94bcc8331"


def _hive_heuristic(frame: dict[str, Any]) -> bool:
    els = frame.get("elements") or []
    for el in els:
        if not isinstance(el, dict):
            continue
        tags = el.get("tags") or []
        if el.get("id") in {"red_cell_up", "hive_frame_west", "red_garden"} or el.get("type") == "hive_frame" or "hive" in tags:
            return True
    return False


def stamp_cad_contract(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fill cadManifest / source hash for official BIOBUZZ field GLB frames.

    Typeless old-replay pieces are left alone so the viewer keeps the 2.5 in primitive.
    """
    if not isinstance(frame, dict) or frame.get("backgroundAsset") != BIOBUZZ_GLB:
        return frame
    need_manifest = not frame.get("cadManifest")
    need_hash = not frame.get("cadSourceSha256")
    need_version = not frame.get("cadAssetVersion")
    if not (need_manifest or need_hash or need_version):
        return frame
    out = dict(frame)
    if need_manifest:
        out["cadManifest"] = BIOBUZZ_MANIFEST
    if need_hash:
        out["cadSourceSha256"] = BIOBUZZ_FIELD_SHA256
    if need_version:
        out["cadAssetVersion"] = f"{BIOBUZZ_FIELD_SHA256}:{CAD_GENERATOR_VERSION}"
    return out


def ensure_background_asset(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(frame, dict):
        return frame
    stamped = frame
    if not stamped.get("backgroundAsset") and _hive_heuristic(stamped):
        stamped = dict(stamped)
        stamped["backgroundAsset"] = BIOBUZZ_GLB
    return stamp_cad_contract(stamped)
