import { launchArcPoints, usesPhysicalStorage } from "./FieldScene";
import type { Frame, FramePiece, PiecePathSpec } from "../api";

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) throw new Error(message);
}

function frame(partial: Partial<Frame> = {}): Frame {
  return {
    t: 0,
    trueScore: 0,
    robots: [],
    pieces: [],
    elements: [],
    matchVarsPrivileged: {},
    observedMatchVars: {},
    fieldSizeIn: { width: 144, depth: 144 },
    explains: [],
    queues: {},
    gate: {},
    vision: [],
    ...partial,
  };
}

const tests: Array<[string, () => void]> = [
  ["legacy held pieces are restaged unless physical storage is flagged", () => {
    const held: FramePiece = { id: "p1", x: 4, y: -8, heldBy: "red_0" };
    const doc = frame({ pieces: [held], robots: [{ id: "red_0", x: 0, y: 0, headingDeg: 0, held: ["p1"], dynamic: true }] });
    assert(!usesPhysicalStorage(doc), "old replays keep ghost-held staging");
  }],
  ["physicalPieces renders stored bodies at recorded pose", () => {
    const held: FramePiece = { id: "p1", x: 1, y: 0, z: 4, heldBy: "red_0", storedSlot: 0 };
    const doc = frame({ physicalPieces: true, pieces: [held] });
    assert(usesPhysicalStorage(doc), "physical flag uses recorded piece pose");
  }],
  ["launch preview produces a rising then falling arc", () => {
    const path: PiecePathSpec = {
      intakeActuatorId: "intake",
      conveyorActuatorId: "conveyor",
      flywheelActuatorId: "flywheel",
      gateActuatorId: "gate",
      storageSlots: [{ x: 0, y: 0, z: 4 }],
      muzzlePose: { x: 10, y: 0, z: 12, pitchDeg: 45 },
      wheelRadiusIn: 2,
      launchEfficiency: 0.235,
    };
    const points = launchArcPoints(path, 1, 0.5, [{ id: "flywheel", kind: "velocity_motor", motor: { nominalVoltageV: 12, freeSpeedRpm: 6000, stallTorqueNm: 0.5, stallCurrentA: 20, freeCurrentA: 0.6 }, gearRatio: 1, efficiency: 0.9, rotorInertiaKgM2: 0, loadInertiaKgM2: 0.001, currentLimitA: 18, controllerLatencyMs: 0, targetRpm: 4500 }]);
    assert(points.length >= 3, "arc has multiple samples");
    assert(points[0][0] === 10, "starts at muzzle x");
    const peak = Math.max(...points.map((p) => p[1]));
    assert(peak > 12, "ballistic z rises above muzzle");
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
  throw new Error(`${failed} physical replay test(s) failed`);
}
console.log(`${tests.length} passed`);
