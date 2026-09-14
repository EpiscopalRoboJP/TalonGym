# Mentor sign-off — BIOBUZZ V1 preset freeze

Do not treat these as engine work. Until they are answered, every BIOBUZZ preset keeps `"verifyAgainstManual": true` and the UI shows an unverified banner.

Official documents:

- Competition Manual / Game Details: https://ftc-resources.firstinspires.org/ftc/game/manual-10
- Field CAD/STEP: https://ftc-resources.firstinspires.org/ftc/archive/2027/field

Encoded facts and remaining placeholders: [presets/seasons/biobuzz_2026/README.md](../presets/seasons/biobuzz_2026/README.md).

## Point values and RP (Tables 10-2 and 10-3)

Confirm the PDF you will ship still lists AUTO as:

| Achievement | Value in our preset | Mentor OK? |
|-------------|---------------------|------------|
| LEAVE | 3 | |
| PARK | 5 | |
| HIVE TIP | 20 | |

RP thresholds we encoded as **match-wide proxies** (not AUTO-complete), other events:

| RP | Other events |
|----|--------------|
| SWARM (LEAVE+PARK) | ≥ 16 |
| POLLINATOR 1 | tips ≥ 4 |
| POLLINATOR 2 | tips ≥ 7 |

Championship / regional championship columns are TBA. AUTO-only training must not be labeled “Ranking Points earned.” Which default **optimization objective** should the team leaderboard use: mean match points, 10th percentile, LCB, or RP-proxy probability?

## HIVE TIP and geometry

Preset threshold is `red_up_cell_load ≥ 7` (3 staged NECTAR + 4 launched). Confirm against CAD / Field Setup Guide. The same `red_cell_up` volume stays the “up” cell after a tip; spilled pieces are consumed rather than dumped to the floor.

Exact FLOWER / hive x-offsets are schematic until official STEP is reachable. CAD general tolerance ±1 in becomes domain-randomization default, not a collision shrink.

## AprilTags

Manual extract confirmed ids 0, 7, and 38–41. Clusters 1–3, 4–6, and 42–45 are inferred. Confirm against official artwork.

## Rules we can encode as data

- G402 no AUTO opponent-side interference: **foul during training.** Restricted-volume entry deducts −15 from `trueScore` (Major Foul proxy, VERIFY AGAINST MANUAL) and is highlighted in replay. Point value still needs mentor confirmation against the current manual.
- Championship TRANSITION is 15 s vs 8 s typical: ignore for AUTO policy training?
- Which start slots does *this* team actually use?

## Geometry freeze

Replace schematic poses in `presets/seasons/biobuzz_2026/field.json` with CAD-extracted inches when the official STEP URL is reachable. Record `contentSha256` of the STEP/Onshape export. Re-run `python -m talongym import-field-cad`.
