"""Official BIOBUZZ STEP sources. Raw files stay in var/cad/; hashes pin rebuilds."""

from __future__ import annotations

from typing import Any

# FIRST Playing Field Resources binary endpoint (Content-Disposition STEP).
OFFICIAL_FIELD_PAGE = "https://ftc-resources.firstinspires.org/ftc/archive/2027/field"
OFFICIAL_FIELD_STEP_URL = "https://ftc-resources.firstinspires.org/ftc/archive/2027/field/field-cad-step"

# AndyMark scoring-element STEP exports (POLLEN / NECTAR).
PIECE_SOURCES: dict[str, dict[str, Any]] = {
    "pollen": {
        "typeId": "pollen",
        "displayName": "POLLEN",
        "sourceUrl": (
            "https://s3.amazonaws.com/docusync-files/"
            "3876dea7b683c2085286da2899504555aab1a13720ab8c02c1d0107e1095f445/"
            "am-5851_yellow%20Pollen.STEP"
        ),
        "filename": "am-5851_yellow_Pollen.STEP",
        "specDiameterIn": 2.8,
        "sha256": "3876dea7b683c2085286da2899504555aab1a13720ab8c02c1d0107e1095f445",
    },
    "nectar_red": {
        "typeId": "nectar_red",
        "displayName": "Red NECTAR",
        "sourceUrl": (
            "https://s3.amazonaws.com/docusync-files/"
            "d25b13b0bcf422196f68afb374a823209ace6f3609b9b4c99b55b8617cb04610/"
            "am-5852_red%20Red%20Alliance%20Nectar.STEP"
        ),
        "filename": "am-5852_red_Red_Alliance_Nectar.STEP",
        "specDiameterIn": 3.6,
        "sha256": "d25b13b0bcf422196f68afb374a823209ace6f3609b9b4c99b55b8617cb04610",
    },
    "nectar_blue": {
        "typeId": "nectar_blue",
        "displayName": "Blue NECTAR",
        "sourceUrl": (
            "https://s3.amazonaws.com/docusync-files/"
            "d085dfaa85e816f8f80b1e667b3d94f4c10d7fff3e2194cf48bec6ab628e9c15/"
            "am-5852_blue%20Blue%20Alliance%20Nectar.STEP"
        ),
        "filename": "am-5852_blue_Blue_Alliance_Nectar.STEP",
        "specDiameterIn": 3.6,
        "sha256": "d085dfaa85e816f8f80b1e667b3d94f4c10d7fff3e2194cf48bec6ab628e9c15",
    },
}

# None sha256 means "accept and record"; a 64-char hex pin must match on rebuild.

FIELD_SOURCE: dict[str, Any] = {
    "sourceUrl": OFFICIAL_FIELD_STEP_URL,
    "pageUrl": OFFICIAL_FIELD_PAGE,
    "filename": "BIOBUZZ_Full_Field.20260912.step",
    "sha256": "05b35961c7df847741031f00fda73ddd068537f809e92a11b1cd59a94bcc8331",
}


def piece_source(type_id: str) -> dict[str, Any]:
    spec = PIECE_SOURCES.get(type_id)
    if spec is None:
        known = ", ".join(sorted(PIECE_SOURCES))
        raise KeyError(f"unknown piece type {type_id!r}; expected one of {known}")
    return spec


def expected_sha256(spec: dict[str, Any]) -> str | None:
    digest = spec.get("sha256")
    if not digest:
        return None
    text = str(digest).strip().lower()
    if len(text) != 64:
        return None
    return text
