# 2026–27 starter launcher CAD reference

These are manufacturer starter assemblies, not a CAD model of Team 17986 or Team 27268's actual robot. The four scoring starters currently checked into TalonGym are synthetic overlays. Their geometry and shot trajectory must not be presented as either manufacturer's assembled launcher or as calibrated scoring physics.

## goBILDA BIOBUZZ StarterBot (mecanum)

- [Complete manufacturer STEP assembly](https://www.gobilda.com/content/step_files/3200-2627-0004.zip) and [35-page assembly instructions](https://www.gobilda.com/content/user_manuals/3200-2627-0004_assembly-instructions.min.pdf).
- Step 51 mounts **one 96 mm Hogback Traction Wheel, SKU 3626-0014-0096**, to one Hyper Hub with four M4 × 12 mm screws and washers. The [wheel specification](https://www.gobilda.com/hogback-traction-wheel-96mm-diameter-50a-durometer/) gives 82 g and 50A tread. The individual manufacturer STEP has a **24 mm axial width**; its tessellated outer diameter is approximately 96 mm.
- Step 38 converts a 19.2:1 Yellow Jacket to **1:1**. The [1:1 motor specification](https://www.gobilda.com/5203-series-yellow-jacket-motor-1-1-ratio-24mm-length-8mm-rex-shaft-6000-rpm-3-3-5v-encoder/) gives **6,000 RPM no-load at 12 V**. This is not loaded flywheel speed or projectile speed.
- Step 55 puts the hub on the motor shaft with approximately **1 mm clearance** between the Hyper Hub and adjacent screw heads.
- The cutting guide on page 7 makes **Large Gridplate H, 216 × 176 mm**, from the [1.5 mm polycarbonate gridplate](https://www.gobilda.com/ftc-starter-kit-2026-2027-season/). Step 57 fixes that plate above the wheel with six zip ties. It is not SKU 1202-0001-0001, the angle mount used as `hood_plate` in the TalonGym overlay.
- A 1 mm-deflection tessellation of the complete STEP gives an approximate **452 × 452 × 309 mm** overall assembly envelope. This is a whole-robot measurement, not a launcher envelope.

## REV DUO BIOBUZZ Starter Bot

- [Complete manufacturer Onshape assembly](https://cad.onshape.com/documents/9cb9eb90e973fd37d10ee178/w/b9c02b0f4e09998f08becbd2/e/a55c018c62564f3ea8812a47) and [126-page assembly instructions](https://www.revrobotics.com/content/technical-resources/DUO/FTC-Kickoff-Concepts/2026-27/2026-27_REV_DUO_FTC_Starter_Bot-Build_Guide_1567.pdf).
- Pages 105–107 build the flywheel from **two REV-41-1267 Grip Wheels**, four 90-tooth gears, two shaft collars, and one **252 mm long, 5 mm hex shaft**. The wheel order is collar → two gears → two wheels → two gears → collar, leaving approximately **30 mm of shaft exposed** at one end. [Each wheel](https://www.revrobotics.com/rev-41-1267/) is **90 mm diameter × 25 mm wide**, 88 g, with a 5 mm hex bore and 65A tread.
- The launcher uses a **1:1 UltraPlanetary output and HD Hex motor**. The [bare motor's 6,000 RPM figure](https://www.revrobotics.com/REV-41-1291/) is a 12 V free speed, not a measured firing speed.
- Page 4 cuts the launcher floor from **329 × 84 × 2 mm polycarbonate**. Page 89 bends it during assembly. Its final angle cannot be read from the flat cut size. REV-41-1305 is a structural 90° bracket, not this floor.

## Modeling boundary

The current `mecanum_biobuzz_4cap` template supplies a **52° muzzle pitch**, muzzle coordinates, **0.235 launch efficiency**, and other trajectory inputs to all four catalog scoring starters. Neither manufacturer guide verifies those values. The compiler now derives flywheel radius from the selected catalog wheel, but launch pitch, contact geometry, loaded RPM, ball compression, and exit speed still need an assembled CAD measurement and physical shot calibration. The scripted scoring check currently launches four pieces but scores zero for all four catalog starters; training should continue to fail closed on that result.

Do not merely substitute the goBILDA Hogback SKU for the Boot Wheel: the Hogback requires its Hyper Hub and the correct bolt pattern. Do not duplicate the REV wheel without adding the common shaft, gears, bearings, collars, and axial positions. Those changes need a whole-assembly collision and actuator review before the mismatch warnings can be removed.
