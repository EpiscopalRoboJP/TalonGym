# BIOBUZZ 2026–2027 — Competition Manual V1

Game: **BIOBUZZ™ presented by RTX**, FIRST CANOPY™. Preset ids `biobuzz_2026_field_v1` / `biobuzz_2026_scoring_v1`. Provenance: Competition Manual **V1** (2026-09-12). Physical field and scoring-element geometry is generated from official STEP (`cadManifest`, `python -m talongym import-field-cad`). Scoring volumes remain explicit preset data. Runtime MuJoCo loads the CAD-assembled `field_mjcf.xml` (convex assembly parts + typed piece hulls), never `field.glb` or a single concave field mesh. Raw STEP and derived tessellation are not committed.

Official CAD registration puts the red GARDEN on the audience-side corner and
the red LOADING ZONE on the opposite side of the red wall (blue is rotationally
symmetric). The Lab uses these CAD-registered centers for its overlays.

## Encoded facts (§8–11, 16)

- Field 144 × 144 in. HIVE structure at center (frame 49.46 × 38.95 in, pivots 43.95 in high); 4 FLOWERS on the perimeter wall; LOADING ZONE 23 × 11 in; GARDEN 23 × 2 in tape.
- Scoring elements: **POLLEN** 2.8 in yellow (40), **NECTAR** 3.6 in red/blue (8 each) — §9.8, not the pre-Kickoff “~3 in” figure.
- Staging: 4 POLLEN in each FLOWER, 4 in each GARDEN, 4 pre-loaded per ROBOT. 3 NECTAR in each upward CELL (modeled as `red_up_cell_load` init, not floor pieces). 5 NECTAR stay in each ALLIANCE AREA (off-field).
- AUTO (Table 10-2): LEAVE 3, PARK 5, HIVE TIP 20. CELL / FLOWER / GARDEN points are end-of-match only.
- G304 starts: robots touch the alliance perimeter, stay on columns A–C (red) / D–F (blue), and must not occupy a LOADING ZONE or FLOWER. Default slots are `red_0` / `blue_0` on the audience-side wall and `red_1` / `blue_1` at mid-wall, clear of tape.
- RP thresholds, all other events (Table 10-3; champs TBA): SWARM = LEAVE+PARK ≥ 16; POLLINATOR 1 = tips ≥ 4; POLLINATOR 2 = tips ≥ 7.
- AprilTags: 36h11, 3.25 in, 4-tag clusters on CELL bottoms.

## Placeholders (`verifyAgainstManual: true`)

- Exact FLOWER / hive x-offsets (figures/CAD STEP not committed; HubSpot hides the direct file).
- HIVE TIP threshold: scoring uses `red_up_cell_load ≥ 7` (3 staged NECTAR + 4 launched). Confirm against CAD / Field Setup Guide.
- Three alliance NECTAR are physical bodies in each initial upward CELL. Launched pieces remain physical, and the articulated HIVE collision releases them during a tip.
- AprilTag IDs 1–3, 4–6, and 42–45 are inferred. Manual extract only confirmed 0, 7, and 38–41.

Do not invent further scoring. Bump `manualRevision` on Team Updates rather than editing engine code.
