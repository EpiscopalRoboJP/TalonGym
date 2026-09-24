"""World-space collision checks with exceptions for connected mating pairs."""

from __future__ import annotations

from typing import Any

import numpy as np

from talongym.robot.contract import AssemblyError
from talongym.robot.transforms import matrix_from_pose, transform_point

_OVERLAP_EPS = 1e-4


def _proxy_size(proxy: dict[str, Any]) -> tuple[float, float, float] | None:
    kind = str(proxy.get("kind") or "box")
    if kind == "box":
        size = list(proxy.get("sizeIn") or [])
        if len(size) != 3:
            return None
        return float(size[0]), float(size[1]), float(size[2])
    radius = float(proxy.get("radiusIn") or 0.0)
    if kind == "sphere":
        span = 2.0 * radius
        return span, span, span
    length = float(proxy.get("lengthIn") or 0.0)
    if kind in {"cylinder", "capsule"}:
        span = 2.0 * radius
        along = length + (2.0 * radius if kind == "capsule" else 0.0)
        return span, span, along if along > 0 else span
    size = list(proxy.get("sizeIn") or [])
    if len(size) == 3:
        return float(size[0]), float(size[1]), float(size[2])
    return None


def proxy_aabb(part_world: np.ndarray, proxy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    size = _proxy_size(proxy)
    if size is None:
        return None
    local = matrix_from_pose(proxy.get("pose") if isinstance(proxy.get("pose"), dict) else None)
    world = part_world @ local
    hx, hy, hz = 0.5 * size[0], 0.5 * size[1], 0.5 * size[2]
    corners = np.array(
        [[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)],
        dtype=np.float64,
    )
    points = np.vstack([transform_point(world, corner) for corner in corners])
    return points.min(axis=0), points.max(axis=0)


def part_aabbs(part_world: np.ndarray, part: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for proxy in part.get("collision") or []:
        box = proxy_aabb(part_world, proxy)
        if box is not None:
            boxes.append(box)
    return boxes


def aabbs_overlap(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.all(left[0] < right[1] - _OVERLAP_EPS) and np.all(right[0] < left[1] - _OVERLAP_EPS))


def aabbs_coincident(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.allclose(left[0], right[0], atol=1e-3, rtol=0) and np.allclose(left[1], right[1], atol=1e-3, rtol=0))


def union_aabb(boxes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not boxes:
        return None
    lows = np.vstack([box[0] for box in boxes])
    highs = np.vstack([box[1] for box in boxes])
    return lows.min(axis=0), highs.max(axis=0)


def assembly_aabb(
    poses: dict[str, np.ndarray],
    catalog_parts: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray] | None:
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for ident, part in catalog_parts.items():
        if ident not in poses:
            continue
        boxes.extend(part_aabbs(poses[ident], part))
    return union_aabb(boxes)


def validate_assembly_collisions(
    poses: dict[str, np.ndarray],
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    connections: list[dict[str, Any]],
) -> None:
    mated = {
        frozenset({str(row["parent"]["instanceId"]), str(row["child"]["instanceId"])})
        for row in connections
    }
    parent: dict[str, str] = {ident: ident for ident in instances}

    def find(ident: str) -> str:
        while parent[ident] != ident:
            parent[ident] = parent[parent[ident]]
            ident = parent[ident]
        return ident

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pair in mated:
        if len(pair) == 2:
            left, right = tuple(pair)
            union(left, right)
    boxes = {
        ident: part_aabbs(poses[ident], catalog_parts[ident])
        for ident in instances
    }
    ordered = list(instances)
    for index, left_id in enumerate(ordered):
        for right_id in ordered[index + 1 :]:
            pair = frozenset({left_id, right_id})
            if pair in mated:
                continue
            overlapping = any(
                aabbs_overlap(left_box, right_box) for left_box in boxes[left_id] for right_box in boxes[right_id]
            )
            if not overlapping:
                continue
            if find(left_id) == find(right_id):
                # Legacy proxy origins produce many ambiguous intersections, but two
                # identical volumes at one pose cannot represent distinct parts.
                if any(
                    aabbs_coincident(left_box, right_box)
                    for left_box in boxes[left_id]
                    for right_box in boxes[right_id]
                ):
                    raise AssemblyError(f"coincident collision volumes between {left_id} and {right_id}")
                continue
            raise AssemblyError(
                f"interpenetration between {left_id} and {right_id} outside mating clearance"
            )


def connected_overlap_warnings(
    poses: dict[str, np.ndarray],
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    connections: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Report overlaps that legacy origin-centred proxies cannot safely reject yet."""
    direct = {
        frozenset({str(row["parent"]["instanceId"]), str(row["child"]["instanceId"])})
        for row in connections
    }
    boxes = {ident: part_aabbs(poses[ident], catalog_parts[ident]) for ident in instances}
    overlaps: list[tuple[str, str]] = []
    ordered = list(instances)
    for index, left_id in enumerate(ordered):
        for right_id in ordered[index + 1 :]:
            if frozenset({left_id, right_id}) in direct:
                continue
            if any(
                aabbs_overlap(left_box, right_box)
                for left_box in boxes[left_id]
                for right_box in boxes[right_id]
            ):
                overlaps.append((left_id, right_id))
    if not overlaps:
        return ()
    examples = ", ".join(f"{left}/{right}" for left, right in overlaps[:3])
    suffix = "" if len(overlaps) <= 3 else f", plus {len(overlaps) - 3} more"
    return (
        {
            "code": "connected_proxy_overlap",
            "severity": "warning",
            "message": (
                f"{len(overlaps)} connected proxy pair(s) overlap ({examples}{suffix}); "
                "verify authored collision origins and clearance before relying on contact physics"
            ),
            "instanceId": overlaps[0][0],
        },
    )
