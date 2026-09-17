import { buildSetupPreview, runMatchesSceneSelection, trainSceneFrame, type SetupFieldDoc } from "./setupPreview";
import type { Frame, RobotPreset } from "../api";

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) throw new Error(message);
}

function fieldDoc(partial?: Partial<SetupFieldDoc>): SetupFieldDoc {
  return {
    id: "test_field",
    displayName: "Test field",
    fieldSizeIn: { width: 144, depth: 144 },
    backgroundAsset: "seasons/biobuzz_2026/field.glb",
    cadManifest: "seasons/biobuzz_2026/cad_manifest.json",
    startSlots: [
      { id: "red_0", alliance: "red", slot: 0, pose: { x: -61.6, y: -48, headingDeg: 0 } },
      { id: "blue_0", alliance: "blue", slot: 0, pose: { x: 61.6, y: 48, headingDeg: 180 } },
    ],
    elements: [{ id: "perimeter", type: "wall", pose: { x: 0, y: 0 }, shape: { kind: "aabb", width: 144, depth: 144 } }],
    gamePieces: [{ typeId: "pollen", shape: { kind: "circle", radius: 1.4 }, color: "yellow", visualAsset: "pollen.glb" }],
    spawns: [{ id: "garden", pieceTypeId: "pollen", poses: [{ x: 10, y: 12, z: 1.4 }] }],
    ...partial,
  };
}

function robotDoc(id: string, displayName: string): RobotPreset {
  return {
    schemaVersion: "1.1.0",
    id,
    displayName,
    drivetrain: { type: "mecanum", trackWidthIn: 14 },
    chassis: { lengthIn: 16, widthIn: 14, heightIn: 10, massKg: 8 },
    motors: { count: 4 },
    constraints: { maxVelInPerS: 40, maxAccelInPerS2: 40, maxAngVelDegPerS: 180 },
    mechanisms: { capacity: 3 },
    sensors: [],
    visualAsset: `robots/${id}/visual.glb`,
    rigidParts: [
      { id: "chassis", massKg: 4, collision: [{ kind: "box", sizeIn: [16, 14, 2] }], tags: ["chassis"] },
      { id: "wheel_fl", massKg: 0.2, collision: [{ kind: "cylinder", radiusIn: 2, lengthIn: 1 }], tags: ["wheel"] },
    ],
  };
}

function liveFrame(robotId: string): Frame {
  return {
    t: 1,
    trueScore: 4,
    robots: [{ id: "red_0", x: 0, y: 0, headingDeg: 0, held: [], dynamic: true }],
    robotDesign: { visualAsset: `robots/${robotId}/visual.glb` },
    pieces: [],
    elements: [],
    matchVarsPrivileged: {},
    observedMatchVars: {},
    fieldSizeIn: { width: 144, depth: 144 },
    explains: [],
    queues: {},
    gate: {},
    vision: [],
  };
}

const tests: Array<[string, () => void]> = [
  ["setup preview places the selected robot on the official red start", () => {
    const frame = buildSetupPreview(fieldDoc(), robotDoc("catalog_bot", "Catalog bot"));
    assert(frame, "built");
    assert(frame.robots[0].id === "red_0", "actor");
    assert(frame.robots[0].x === -61.6 && frame.robots[0].y === -48, "start slot");
    assert(frame.robotDesign?.visualAsset === "robots/catalog_bot/visual.glb", "robot cad");
    assert(frame.robotDesign?.rigidParts?.length === 2, "catalog parts");
    assert(frame.backgroundAsset === "seasons/biobuzz_2026/field.glb", "field cad");
    assert(frame.pieces[0].typeId === "pollen" && frame.pieces[0].x === 10, "spawn");
  }],
  ["changing the robot preset changes the design on the preview frame", () => {
    const field = fieldDoc();
    const first = buildSetupPreview(field, robotDoc("mecanum_biobuzz_4cap", "BIOBUZZ"));
    const second = buildSetupPreview(field, robotDoc("gobilda_mecanum", "goBILDA"));
    assert(first?.robotDesign?.visualAsset !== second?.robotDesign?.visualAsset, "visual swapped");
    assert(second?.robotDesign?.visualAsset === "robots/gobilda_mecanum/visual.glb", "new bot");
  }],
  ["live rollout yields to the selected robot/field when they no longer match the run", () => {
    const preview = buildSetupPreview(fieldDoc(), robotDoc("new_bot", "New"));
    const live = liveFrame("old_bot");
    const kept = trainSceneFrame({
      liveFrame: live,
      previewFrame: preview,
      runPresets: { fieldId: "test_field", robotId: "old_bot" },
      fieldId: "test_field",
      robotId: "old_bot",
    });
    assert(kept === live, "matching run stays live");
    const swapped = trainSceneFrame({
      liveFrame: live,
      previewFrame: preview,
      runPresets: { fieldId: "test_field", robotId: "old_bot" },
      fieldId: "test_field",
      robotId: "new_bot",
    });
    assert(swapped === preview, "mismatch shows the selected bot");
    assert(runMatchesSceneSelection({ robotId: "old_bot", fieldId: "test_field" }, "other_field", "old_bot") === false, "field mismatch");
  }],
];

let failed = 0;
for (const [name, run] of tests) {
  try {
    run();
    console.log(`ok ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`not ok ${name}`);
    console.error(err);
  }
}
if (failed) {
  throw new Error(`${failed} setupPreview test(s) failed`);
}
console.log(`${tests.length} passed`);
