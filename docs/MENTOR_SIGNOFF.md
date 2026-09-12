# Mentor sign-off — DECODE TU32 preset freeze

Do not treat these as engine work. Until they are answered, every DECODE preset keeps `"verifyAgainstManual": true` and the UI shows an unverified banner.

Official documents:

- Competition Manual TU32: https://ftc-resources.firstinspires.org/ftc/archive/2026/game/manual
- Game Details HTML: https://ftc-resources.firstinspires.org/ftc/game/manual-10
- Field CAD/STEP: https://ftc-resources.firstinspires.org/ftc/archive/2026/field
- Combined Team Updates (TU32 final): https://ftc-resources.firstinspires.org/ftc/game/tu-combined

## Point values and RP (Tables 10-2 and 10-3)

Confirm the PDF you will ship still lists AUTO as:

| Achievement | Value in our preset | Mentor OK? |
|-------------|---------------------|------------|
| LEAVE | 3 | |
| CLASSIFIED | 3 | |
| OVERFLOW | 1 | |
| PATTERN per matching index | 2 | |

RP thresholds we encoded as **match-wide proxies** (not AUTO-complete):

| RP | Championship | Regional championship | Other events |
|----|--------------|----------------------|--------------|
| GOAL (artifacts through square) | 67 | 42 | 36 |
| PATTERN points | 22 | 22 | 18 |

AUTO-only training must not be labeled “Ranking Points earned.” Which default **optimization objective** should the team leaderboard use: mean match points, 10th percentile, LCB, or RP-proxy probability?

## Motif / AprilTags

Manual: Obelisk tags 21, 22, 23; goal tags red 24, blue 20. Community code often maps GPP→21, PGP→22, PPG→23. Confirm against official artwork. Side faces of the Obelisk are partially obstructed (TU32); front face is the intended read.

## GATE and PATTERN Q&A

Official Q&A: if the GATE is open at the end of AUTO, PATTERN is not assessed because artifacts are not retained. Should the sim use a binary `gateRetaining` flag or a continuous opening angle?

Overflow when a high-speed launch skips the 9th slot: stochastic (fit from logs) or deterministic occupancy only?

## Rules we can encode as data

- G402 no AUTO opponent-side interference: foul during training, eval-only metric, or ignore in MVP?
- Championship TRANSITION is 15 s vs 8 s typical: ignore for AUTO policy training?
- Which start slots does *this* team actually use?

## Geometry freeze

Replace schematic poses in `presets/seasons/decode_2025/field.json` with CAD-extracted inches. Record `contentSha256` of the STEP/Onshape export. CAD general tolerance ±1 in becomes domain-randomization default, not a collision shrink.
