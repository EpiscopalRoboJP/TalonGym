from pathlib import Path

ENGINE_ROOTS = [
    Path("python/talongym/sim"),
    Path("python/talongym/rules"),
    Path("engine"),
]
BANNED = ("artifact", "obelisk", "motif", "pollen", "specimen", "pixel")


def test_no_season_nouns_in_engine():
    hits: list[str] = []
    for root in ENGINE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".rs"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for word in BANNED:
                if word in text:
                    hits.append(f"{path}:{word}")
    assert hits == [], hits
