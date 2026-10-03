"""Build the goBILDA 2026-27 mecanum StarterBot visual from official STEP CAD.

The source assembly is Z-up in metres after cascadio conversion. TalonGym's
robot visuals are Y-up in inches, with +X toward the intake. The resulting GLB
retains every assembly component and its manufacturer placement.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from talongym.paths import ASSETS_DIR

REFERENCE_ID = "gobilda_biobuzz_reference"
SOURCE_URL = "https://www.gobilda.com/content/step_files/3200-2627-0004.zip"
STEP_MEMBER = "3200-2627-0004.step"
MM_PER_IN = 25.4
SIMPLIFICATION_FACE_RATIO = 0.28

# The CAD's drive axle centre is at Z=0. The lowest wheel point is Z=-52 mm.
# The envelope centre in CAD Y is -34 mm. CAD -Y points toward the intake.
CAD_TO_ROBOT = np.array(
    [
        [0, -1000 / MM_PER_IN, 0, -34 / MM_PER_IN],
        [0, 0, 1000 / MM_PER_IN, 52 / MM_PER_IN],
        [-1000 / MM_PER_IN, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)


def build_visual(*, source_glb: Path | None = None, source_zip: Path | None = None) -> Path:
    """Convert a trusted source locally; never route the 419 MB STEP through uploads."""
    import trimesh

    official_zip_sha256 = None

    if source_glb is not None:
        raw = source_glb.read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        if source_zip is not None:
            official_zip_sha256 = hashlib.sha256(source_zip.read_bytes()).hexdigest()
        scene = trimesh.load(io.BytesIO(raw), file_type="glb", force="scene")
        source_kind = "cascadio-derived GLB"
    else:
        if source_zip is None:
            with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
                zip_bytes = response.read()
        else:
            zip_bytes = source_zip.read_bytes()
        source_sha256 = hashlib.sha256(zip_bytes).hexdigest()
        official_zip_sha256 = source_sha256
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            step_bytes = archive.read(STEP_MEMBER)
        import cascadio

        raw = cascadio.load(step_bytes, tol_linear=1.0, use_parallel=True)
        scene = trimesh.load(io.BytesIO(raw), file_type="glb", force="scene")
        source_kind = "official STEP ZIP"

    source_bounds = np.asarray(scene.bounds, dtype=float)
    if not np.allclose(
        source_bounds,
        [[-0.2259, -0.26, -0.052], [0.2262, 0.192, 0.2574]],
        atol=0.006,
    ):
        raise ValueError(f"unexpected goBILDA assembly bounds (metres): {source_bounds}")
    scene.apply_transform(CAD_TO_ROBOT)
    source_faces = sum(len(mesh.faces) for mesh in scene.geometry.values())
    for name, mesh in list(scene.geometry.items()):
        if len(mesh.faces) < 800:
            continue
        # Keep the wheel tread and ball guide at greater detail. Every
        # manufacturer's component stays in the scene at its exact transform.
        important = name.startswith(("3626-", "3625-", "1117-", "RollerTire", "Core:"))
        ratio = 0.5 if important else SIMPLIFICATION_FACE_RATIO
        reduced = mesh.simplify_quadric_decimation(
            face_count=max(300, round(len(mesh.faces) * ratio))
        )
        if len(reduced.faces) >= len(mesh.faces):
            continue
        material = getattr(mesh.visual, "material", None)
        if material is not None:
            reduced.visual = trimesh.visual.TextureVisuals(material=material)
        scene.geometry[name] = reduced
    visual_faces = sum(len(mesh.faces) for mesh in scene.geometry.values())
    dest = ASSETS_DIR / "robots" / REFERENCE_ID
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / "robot.glb"
    output.write_bytes(scene.export(file_type="glb"))
    manifest = {
        "referenceUrl": SOURCE_URL,
        "sourceKind": source_kind,
        "sourceSha256": source_sha256,
        "officialZipSha256": official_zip_sha256,
        "visualSha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "cadBoundsM": source_bounds.tolist(),
        "visualBoundsIn": np.asarray(scene.bounds).tolist(),
        "frame": "robot +X toward intake; Y-up GLB in inches; ground Y=0",
        "cascadioToleranceMm": 1.0,
        "sourceFaces": source_faces,
        "visualFaces": visual_faces,
        "simplificationFaceRatio": SIMPLIFICATION_FACE_RATIO,
    }
    (dest / "cad_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glb", type=Path, help="Previously converted raw cascadio GLB")
    parser.add_argument("--source-zip", type=Path, help="Downloaded official goBILDA STEP ZIP")
    args = parser.parse_args()
    print(build_visual(source_glb=args.source_glb, source_zip=args.source_zip))


if __name__ == "__main__":
    main()
