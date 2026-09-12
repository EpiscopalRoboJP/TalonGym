from __future__ import annotations

from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
REPO_ROOT = PYTHON_DIR.parent.parent
PRESETS_DIR = REPO_ROOT / "presets"
SCHEMAS_DIR = REPO_ROOT / "schemas"
ASSETS_DIR = REPO_ROOT / "assets"
VAR_DIR = REPO_ROOT / "var"
WEB_DIST = REPO_ROOT / "web" / "dist"
WEB_SRC = REPO_ROOT / "web"
