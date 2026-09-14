# Mentor sign-off — BIOBUZZ V1 preset freeze

Do not treat these as engine work. Until they are answered, every BIOBUZZ preset keeps `"verifyAgainstManual": true` and the UI shows an unverified banner — that flag requires a second human to sign off, not just a text-match against the PDF.

Official documents:

- Competition Manual / Game Details: https://ftc-resources.firstinspires.org/ftc/game/manual-10
- Field CAD/STEP: https://ftc-resources.firstinspires.org/ftc/archive/2027/field

Encoded facts and remaining placeholders: [presets/seasons/biobuzz_2026/README.md](../presets/seasons/biobuzz_2026/README.md).

**2026-09-14: the Competition Manual V1 PDF was fetched and checked line-by-line against every table below.** Results are marked ✅ confirmed / ⚠️ mismatch found and fixed / ❓ still genuinely unresolved by the manual. This does not replace mentor sign-off — a second person should still spot-check the PDF themselves before flipping `verifyAgainstManual` to `false`.

## Point values and RP (Tables 10-2 and 10-3)

Table 10-2, confirmed against the V1 PDF:

| Achievement | Value in our preset | Manual (Table 10-2) | Status |
|-------------|---------------------|----------------------|--------|
| LEAVE | 3 (AUTO only) | 3, AUTO only | ✅ |
| PARK | 5 (AUTO) | 5 in AUTO **and** a separate 5 in TELEOP (two distinct achievements — TELEOP PARK is out of scope since this trainer only simulates AUTO) | ✅ for the AUTO half |
| HIVE TIP | 20 (either phase) | 20, same value whether achieved in AUTO or TELEOP | ✅ |

Not modeled (correctly, since only AUTO is simulated — TELEOP-only per the manual): POLLEN/NECTAR remaining in an up CELL at match end (2 pts each), Bottom NECTAR Bonus (5 pts), POLLEN/NECTAR in an *owned* FLOWER (2 pts each, can't start until G410's last-60-seconds gate), GARDEN pieces (1 pt each, GARDEN scoring is assessed at end of TELEOP).

RP thresholds ("All Other Events" column, the only one published — Regional/Championship are literally "TBA" in the manual itself, not a gap in our extraction):

| RP | Our preset | Manual (Table 10-3) | Status |
|----|------------|----------------------|--------|
| SWARM (LEAVE+PARK) | ≥ 16 | ≥ 16 points | ✅ |
| POLLINATOR 1 | tips ≥ 4 | ≥ 4 TIPS | ✅ |
| POLLINATOR 2 | tips ≥ 7 | ≥ 7 TIPS | ✅ |

AUTO-only training must not be labeled "Ranking Points earned." Which default **optimization objective** should the team leaderboard use: mean match points, 10th percentile, LCB, or RP-proxy probability? (Still a mentor/product call, not a manual question.)

## HIVE TIP and geometry

Preset threshold is `red_up_cell_load ≥ 7` (3 staged NECTAR + 4 launched). **❓ Still unresolved** — the manual only says a HIVE is bi-stable and tips "when enough POLLEN or NECTAR are LAUNCHED into the upwards-facing CELL"; it never states a piece count (this is a physical/weight threshold, presumably in the Field Setup Guide, which the manual marks "coming soon"). Keep VERIFY on this one specifically.

Exact FLOWER / hive x-offsets are schematic until official STEP is reachable. CAD general tolerance ±1 in becomes domain-randomization default, not a collision shrink. (Unchanged — still needs the STEP file.)

## AprilTags

✅ **Upgraded from "inferred" to manual-confirmed pattern.** The V1 PDF explicitly lists: cluster 1 (red CELL, side opposite audience) starts at ID 0; cluster 2 (red CELL, audience side) ends at ID 7; cluster 3 (blue CELL, audience side) is IDs 38, 39, 40, 41 (fully legible); cluster 4 (blue CELL, opposite audience) follows immediately after. A PDF text-extraction quirk dropped the single-digit numerals in clusters 1 and 2 (both `pdftotext -layout` and `-raw` lose them; likely a font-subsetting issue, not missing content), but the four-consecutive-IDs-per-cluster structure is explicit in the manual's prose, which fixes clusters 1, 2, and 4 as 0–3, 4–7, and 42–45 respectively — exactly what `field.json` already encoded. Worth a quick visual confirmation against the official AprilTag artwork on the Playing Field Resources page since the exact glyphs couldn't be extracted, but this is no longer a blind guess.

## Rules we can encode as data

- **⚠️ Fixed 2026-09-14:** G402 "No AUTO opponent interference" was encoded as a flat −15 for merely entering the opponent's half of the field during AUTO. Table 10-4 confirms a Major Foul is a 20-point credit to the opponent (Minor Foul = 5, Major Foul = 20) — the point value is now −20. Note this proxy is still a simplification of the actual rule: per the manual, crossing into the opponent's side (FIELD columns D-E-F for red, A-B-C for blue, per Figure 9-5) is explicitly *not itself* a foul — it's called "a risky gameplay strategy that may be seen as STRATEGIC," and G402 only fouls when a team "disrupts AUTO for the opposing ALLIANCE." Zone-entry-as-proxy will flag robots that cross the centerline without interfering with anything, which the real rule would not. Decide whether that's an acceptable training-signal simplification (documented) or worth a tighter interference heuristic (e.g., contact with the opponent robot).
- **✅ Resolved 2026-09-14:** the old "Championship TRANSITION is 15 s vs 8 s typical" note is stale/incorrect and has been dropped. The V1 manual states one TRANSITION duration throughout (8 s, Section 10.4); Section 15.2 Game Modification (FIRST Championship overrides) only calls out SCORING ELEMENT counts/distribution, RP thresholds, and field risers/decals — nothing about MATCH period timing. Mentor confirmed 8 s is correct everywhere; `scoring.json`'s `TRANSITION` phase (`durationS: 8`) already matches and needed no change.
- Which start slots does *this* team actually use?

## Geometry freeze

Replace schematic poses in `presets/seasons/biobuzz_2026/field.json` with CAD-extracted inches when the official STEP URL is reachable. Record `contentSha256` of the STEP/Onshape export. Re-run `python -m talongym import-field-cad`.
