"""Download official field STEP if possible; always emit glTF + MJCF colliders.

Raw STEP is written under var/cad/ and is not committed. Converted assets live in
assets/seasons/<slug>/. HubSpot often hides the direct CAD URL — we scrape the
archive page, then fall back to tessellating the field preset AABBs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from talongym.assets.gltf_boxes import solids_from_field, write_glb
from talongym.assets.mjcf_field import build_mjcf
from talongym.paths import ASSETS_DIR, PRESETS_DIR, VAR_DIR

DEFAULT_CAD_PAGE = "https://ftc-resources.firstinspires.org/ftc/archive/2027/field"
STEP_RE = re.compile(r"""href=["']([^"']+\.(?:step|stp|STEP|STP))["']""", re.I)
ONSHAPE_RE = re.compile(r"""href=["'](https://[^"']*onshape[^"']*)["']""", re.I)


class CadImportError(RuntimeError):
    pass


def _season_slug(field: dict[str, Any]) -> str:
    season = field.get("season") or {}
    return str(season.get("slug") or field.get("id") or "field")


def asset_dir_for(field: dict[str, Any], year_hint: str = "2026") -> Path:
    slug = _season_slug(field)
    return ASSETS_DIR / "seasons" / f"{slug}_{year_hint}"


def load_field_json(path: Path | None = None) -> dict[str, Any]:
    path = path or (PRESETS_DIR / "seasons" / "biobuzz_2026" / "field.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_step_urls(page_url: str = DEFAULT_CAD_PAGE, html: str | None = None) -> list[str]:
    if html is None:
        import urllib.request

        req = urllib.request.Request(page_url, headers={"User-Agent": "TalonGym/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    found = [urljoin(page_url, m.group(1)) for m in STEP_RE.finditer(html)]
    # Prefer filenames that look like the full field, not a single scoring volume.
    found.sort(key=lambda u: (0 if "flower" in u.lower() else 1, len(u)))
    fieldish = [u for u in found if "flower" not in u.lower()]
    return fieldish or found


def download_step(url: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "TalonGym/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    if dest.stat().st_size < 1024:
        raise CadImportError(f"STEP download too small: {dest}")
    return dest


def convert_step_to_trimesh(step_path: Path):
    try:
        import trimesh
    except ImportError as exc:
        raise CadImportError("trimesh missing; pip install -e '.[cad]'") from exc
    try:
        import cascadio  # noqa: F401
    except ImportError:
        pass
    mesh = trimesh.load(str(step_path), force="mesh")
    if mesh is None:
        raise CadImportError(f"trimesh could not load {step_path}")
    return mesh


def write_assets_from_field(
    field: dict[str, Any],
    dest_dir: Path | None = None,
    year_hint: str = "2026",
) -> dict[str, str]:
    dest_dir = dest_dir or asset_dir_for(field, year_hint)
    dest_dir.mkdir(parents=True, exist_ok=True)
    glb_path = dest_dir / "field.glb"
    xml_path = dest_dir / "field_mjcf.xml"
    write_glb(glb_path, solids_from_field(field))
    xml_path.write_text(build_mjcf(field), encoding="utf-8")
    rel_glb = str(glb_path.relative_to(ASSETS_DIR)).replace("\\", "/")
    rel_xml = str(xml_path.relative_to(ASSETS_DIR)).replace("\\", "/")
    return {"glb": str(glb_path), "mjcf": str(xml_path), "backgroundAsset": rel_glb, "collisionAsset": rel_xml}


def try_official_step(page_url: str = DEFAULT_CAD_PAGE) -> Path | None:
    urls = discover_step_urls(page_url)
    if not urls:
        return None
    dest = VAR_DIR / "cad" / "field.step"
    try:
        return download_step(urls[-1], dest)
    except Exception:
        return None


def import_field_cad(
    field_path: Path | None = None,
    page_url: str = DEFAULT_CAD_PAGE,
    year_hint: str = "2026",
) -> dict[str, Any]:
    field = load_field_json(field_path)
    step = try_official_step(page_url)
    note = "aabb_tessellation"
    if step is not None:
        try:
            mesh = convert_step_to_trimesh(step)
            dest_dir = asset_dir_for(field, year_hint)
            dest_dir.mkdir(parents=True, exist_ok=True)
            glb = dest_dir / "field.glb"
            mesh.export(str(glb))
            xml = dest_dir / "field_mjcf.xml"
            xml.write_text(build_mjcf(field), encoding="utf-8")
            note = f"official_step:{step.name}"
            return {
                "ok": True,
                "source": note,
                "glb": str(glb),
                "mjcf": str(xml),
                "step": str(step),
            }
        except Exception as exc:
            note = f"step_failed:{exc}"
    paths = write_assets_from_field(field, year_hint=year_hint)
    return {"ok": True, "source": note, **paths, "cadPage": page_url}


def main() -> None:
    result = import_field_cad()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
