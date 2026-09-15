"""Minimal Y-up glTF 2.0 writer for AABB field solids (FTC inches)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


def ftc_to_yup(x: float, y: float, z: float) -> tuple[float, float, float]:
    """FTC (x, y floor, z height) -> Three/MuJoCo Y-up (x, height, -y)."""
    return float(x), float(z), float(-y)


def _box_mesh(hx: float, hy: float, hz: float) -> tuple[list[float], list[float], list[int]]:
    """Axis-aligned box centered at origin; half-extents hx, hy (height), hz."""
    # Face-specific verts for correct normals
    faces_v = [
        # -Z
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        # +Z
        (hx, -hy, hz), (-hx, -hy, hz), (-hx, hy, hz), (hx, hy, hz),
        # +X
        (hx, -hy, -hz), (hx, -hy, hz), (hx, hy, hz), (hx, hy, -hz),
        # -X
        (-hx, -hy, hz), (-hx, -hy, -hz), (-hx, hy, -hz), (-hx, hy, hz),
        # +Y
        (-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz),
        # -Y
        (-hx, -hy, hz), (hx, -hy, hz), (hx, -hy, -hz), (-hx, -hy, -hz),
    ]
    face_n = [
        (0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
    ]
    positions: list[float] = []
    norms: list[float] = []
    indices: list[int] = []
    for fi in range(6):
        base = fi * 4
        n = face_n[fi]
        for k in range(4):
            p = faces_v[base + k]
            positions.extend(p)
            norms.extend(n)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return positions, norms, indices


def _rgba(color: str) -> list[float]:
    table = {
        "floor": [0.11, 0.29, 0.22, 1.0],
        "wall": [0.54, 0.63, 0.68, 1.0],
        "hive": [0.72, 0.58, 0.28, 1.0],
        "cell_red": [0.72, 0.22, 0.22, 0.85],
        "cell_blue": [0.22, 0.38, 0.72, 0.85],
        "flower": [0.85, 0.72, 0.22, 1.0],
        "frame": [0.35, 0.32, 0.28, 1.0],
    }
    return table.get(color, [0.5, 0.5, 0.5, 1.0])


def solids_from_field(field: dict[str, Any]) -> list[dict[str, Any]]:
    """Build render solids. Tape / zones are omitted (viewer overlays)."""
    fw = float(field["fieldSizeIn"]["width"])
    fd = float(field["fieldSizeIn"]["depth"])
    wall_h = float(field["fieldSizeIn"].get("wallHeight") or 12)
    solids: list[dict[str, Any]] = [
        {"name": "floor", "x": 0.0, "y": 0.0, "z": -0.25, "hx": fw / 2, "hy": 0.25, "hz": fd / 2, "color": "floor"},
        {"name": "wall_n", "x": 0.0, "y": fd / 2, "z": wall_h / 2, "hx": fw / 2 + 1, "hy": wall_h / 2, "hz": 1.0, "color": "wall"},
        {"name": "wall_s", "x": 0.0, "y": -fd / 2, "z": wall_h / 2, "hx": fw / 2 + 1, "hy": wall_h / 2, "hz": 1.0, "color": "wall"},
        {"name": "wall_w", "x": -fw / 2, "y": 0.0, "z": wall_h / 2, "hx": 1.0, "hy": wall_h / 2, "hz": fd / 2, "color": "wall"},
        {"name": "wall_e", "x": fw / 2, "y": 0.0, "z": wall_h / 2, "hx": 1.0, "hy": wall_h / 2, "hz": fd / 2, "color": "wall"},
    ]
    skip_types = {"tape", "zone"}
    skip_tags = {"leave", "park", "loading_zone", "garden", "restricted_for_red", "restricted_for_blue", "perimeter"}
    for el in field.get("elements") or []:
        tags = set(el.get("tags") or [])
        if el.get("type") in skip_types or tags.intersection(skip_tags):
            continue
        pose = el.get("pose") or {}
        shape = el.get("shape") or {}
        if shape.get("kind") != "aabb":
            continue
        w = float(shape.get("width") or 8)
        d = float(shape.get("depth") or 8)
        z = float(pose.get("z") or 6)
        height = float(shape.get("height") or (16 if "cell" in tags or el.get("type") in {"goal", "cell"} else 20 if "hive" in tags else 10))
        color = "frame"
        if el.get("alliance") == "red":
            color = "cell_red"
        elif el.get("alliance") == "blue":
            color = "cell_blue"
        if "flower" in tags or el.get("type") == "flower":
            color = "flower"
        if "frame" in tags:
            color = "frame"
        solids.append(
            {
                "name": str(el["id"]),
                "x": float(pose.get("x") or 0),
                "y": float(pose.get("y") or 0),
                "z": z,
                "hx": w / 2.0,
                "hy": height / 2.0,
                "hz": d / 2.0,
                "color": color,
            }
        )
    return solids


def write_glb(path: Path, solids: list[dict[str, Any]]) -> Path:
    """Write a binary glTF with one mesh per solid. Positions are Y-up inches."""
    bin_parts: list[bytes] = []
    accessors: list[dict[str, Any]] = []
    buffer_views: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    mat_index: dict[str, int] = {}

    def add_material(color: str) -> int:
        if color in mat_index:
            return mat_index[color]
        rgba = _rgba(color)
        idx = len(materials)
        materials.append(
            {
                "name": color,
                "pbrMetallicRoughness": {
                    "baseColorFactor": rgba,
                    "metallicFactor": 0.05,
                    "roughnessFactor": 0.7,
                },
                "doubleSided": True,
                "alphaMode": "BLEND" if rgba[3] < 0.99 else "OPAQUE",
            }
        )
        mat_index[color] = idx
        return idx

    for solid in solids:
        positions, normals, indices = _box_mesh(solid["hx"], solid["hy"], solid["hz"])
        mx, my, mz = ftc_to_yup(solid["x"], solid["y"], solid["z"])
        pos_b = struct.pack("<" + "f" * len(positions), *positions)
        nrm_b = struct.pack("<" + "f" * len(normals), *normals)
        idx_b = struct.pack("<" + "H" * len(indices), *indices)
        pad = (4 - (len(idx_b) % 4)) % 4
        idx_b_padded = idx_b + b"\x00" * pad

        off = sum(len(p) for p in bin_parts)
        bin_parts.append(pos_b)
        buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(pos_b), "target": 34962})
        pos_view = len(buffer_views) - 1
        n_verts = len(positions) // 3
        accessors.append(
            {
                "bufferView": pos_view,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC3",
                "min": [-solid["hx"], -solid["hy"], -solid["hz"]],
                "max": [solid["hx"], solid["hy"], solid["hz"]],
            }
        )
        pos_acc = len(accessors) - 1

        off = sum(len(p) for p in bin_parts)
        bin_parts.append(nrm_b)
        buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(nrm_b), "target": 34962})
        accessors.append({"bufferView": len(buffer_views) - 1, "componentType": 5126, "count": n_verts, "type": "VEC3"})
        nrm_acc = len(accessors) - 1

        off = sum(len(p) for p in bin_parts)
        bin_parts.append(idx_b_padded)
        buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(idx_b), "target": 34963})
        accessors.append({"bufferView": len(buffer_views) - 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"})
        idx_acc = len(accessors) - 1

        mi = add_material(str(solid.get("color") or "frame"))
        meshes.append(
            {
                "name": solid["name"],
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                        "indices": idx_acc,
                        "material": mi,
                    }
                ],
            }
        )
        nodes.append({"name": solid["name"], "mesh": len(meshes) - 1, "translation": [mx, my, mz]})

    blob = b"".join(bin_parts)
    gltf = {
        "asset": {"version": "2.0", "generator": "talongym-field-cad"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad
    bin_pad = (4 - (len(blob) % 4)) % 4
    blob += b"\x00" * bin_pad
    total = 12 + 8 + len(json_bytes) + 8 + len(blob)
    header = struct.pack("<4sII", b"glTF", 2, total)
    json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    bin_chunk = struct.pack("<I4s", len(blob), b"BIN\x00") + blob
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + json_chunk + bin_chunk)
    return path
