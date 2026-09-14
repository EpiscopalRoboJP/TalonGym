# BIOBUZZ 2026–2027 — Competition Manual V1

Game: **BIOBUZZ™ presented by RTX**, FIRST CANOPY™. Preset ids `biobuzz_2026_field_v1` / `biobuzz_2026_scoring_v1`. Provenance: Competition Manual **V1** (2026-09-12). Collision volumes are `hand_authored` schematic AABBs (`field_mjcf.xml`). Lab `field.glb` is tessellated from official **am-5850** Field CAD via `python -m talongym import-field-cad --step`. Raw STEP is not committed.

GARDEN tape sits in the **same alliance corner as the LOADING ZONE** (audience-left for red).

## Encoded facts (§8–11, 16)

- Field 144 × 144 in. HIVE structure at center (frame 49.46 × 38.95 in, pivots 43.95 in high); 4 FLOWERS on the perimeter wall; LOADING ZONE 23 × 11 in; GARDEN 23 × 2 in tape.
- Scoring elements: **POLLEN** 2.8 in yellow (40), **NECTAR** 3.6 in red/blue (8 each) — §9.8, not the pre-Kickoff “~3 in” figure.
- Staging: 4 POLLEN in each FLOWER, 4 in each GARDEN, 4 pre-loaded per ROBOT. 3 NECTAR in each upward CELL (modeled as `red_up_cell_load` init, not floor pieces). 5 NECTAR stay in each ALLIANCE AREA (off-field).
- AUTO (Table 10-2): LEAVE 3, PARK 5, HIVE TIP 20. CELL / FLOWER / GARDEN points are end-of-match only.
- RP thresholds, all other events (Table 10-3; champs TBA): SWARM = LEAVE+PARK ≥ 16; POLLINATOR 1 = tips ≥ 4; POLLINATOR 2 = tips ≥ 7.
- AprilTags: 36h11, 3.25 in, 4-tag clusters on CELL bottoms.

## Verified against Competition Manual V1 PDF (2026-09-14)

The full manual (`ftc-resources.firstinspires.org/ftc/game/manual`) was fetched and checked line-by-line. Confirmed correct: field/HIVE/CELL/FLOWER/SCORING ELEMENT dimensions and counts, staging counts, AUTO point values (LEAVE 3, PARK 5, HIVE TIP 20 — Table 10-2), RP thresholds (SWARM ≥16, POLLINATOR 1 ≥4 tips, POLLINATOR 2 ≥7 tips — Table 10-3, "All Other Events"), and the AprilTag cluster ID pattern (0–3, 4–7, 38–41, 42–45 — manual explicitly confirms 0, 7, 38–41 and the four-per-cluster structure fixes the rest).

One bug found and fixed: `restricted_entry` (the G402 AUTO-interference proxy in `scoring.json`) was docking −15 points; Table 10-4 sets a Major Foul at 20 points, so it's now −20. See `docs/MENTOR_SIGNOFF.md` for the full writeup, including the remaining caveat that this proxy fouls mere zone entry while the real G402 only fouls actual interference.

## Placeholders (`verifyAgainstManual: true`)

- Exact FLOWER / hive x-offsets (figures/CAD STEP not committed; HubSpot hides the direct file).
- HIVE TIP threshold: scoring uses `red_up_cell_load ≥ 7` (3 staged NECTAR + 4 launched). The manual describes the HIVE as bi-stable and tipping once "enough" POLLEN/NECTAR are LAUNCHED into the upward CELL, but never states a piece count — this is a physical/weight threshold presumably in the (not-yet-published) Field Setup Guide, not something a text read of the manual can resolve.
- The same `red_cell_up` volume stays the “up” cell after a tip; spilled pieces are consumed rather than dumped to the floor.

Do not invent further scoring. Bump `manualRevision` on Team Updates rather than editing engine code.
