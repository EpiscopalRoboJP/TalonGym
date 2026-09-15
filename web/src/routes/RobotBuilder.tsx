import { useEffect, useMemo, useRef, useState } from "react";
import { getJson, postJson, putJson, robotPresetLabel, uploadRobotModel, deleteRobotModel, type ActuatorSpec, type ActionTier, type DefaultsBundle, type IntakeSpec, type JointSpec, type LauncherSpec, type MechanismSensorKind, type MechanismSensorSpec, type PiecePathSpec, type PoseOnRobot, type PowerSystemSpec, type PresetMeta, type RigidPartSpec, type Transform3, type VisualOffset } from "../api";
import { upsertCameraSensor } from "../labHonesty";
import { RobotPreview, launchArcPoints } from "../scene/FieldScene";
import { theme } from "../theme";

type RobotDoc = {
  schemaVersion: string;
  id: string;
  displayName: string;
  drivetrain: {
    type: string;
    trackWidthIn: number;
    wheelDiameterIn?: number;
    wheelbaseIn?: number;
    strafeMultiplier?: number;
  };
  chassis: {
    lengthIn: number;
    widthIn: number;
    heightIn?: number;
    massKg: number;
    collisionShape?: string;
    footprint?: { x: number; y: number }[];
  };
  motors: Record<string, number>;
  constraints: {
    maxVelInPerS: number;
    maxAccelInPerS2: number;
    maxAngVelDegPerS: number;
    maxAngAccelDegPerS2?: number;
  };
  mechanisms: {
    capacity: number;
    intakeCycleTimeS?: number;
    scoreCycleTimeS?: number;
    canIntakeWhileMoving?: boolean;
    canScoreWhileMoving?: boolean;
    launchCapable?: boolean;
    climbCapable?: boolean;
    fsmId?: string;
  };
  intakes?: IntakeSpec[];
  launchers?: LauncherSpec[];
  sensors: { id: string; kind: string; fovDeg?: number; rangeIn?: number; poseOnRobot?: PoseOnRobot }[];
  defaultActionTier?: ActionTier;
  policyInterfaceVersion?: string;
  visualAsset?: string | null;
  collisionAsset?: string | null;
  visualOffset?: VisualOffset;
  rigidParts?: RigidPartSpec[];
  joints?: JointSpec[];
  actuators?: ActuatorSpec[];
  powerSystem?: PowerSystemSpec;
  piecePath?: PiecePathSpec;
  mechanismSensors?: MechanismSensorSpec[];
};

type Sel =
  | { kind: "chassis" }
  | { kind: "intake"; id: string }
  | { kind: "launcher"; id: string }
  | { kind: "part"; id: string }
  | { kind: "joint"; id: string }
  | { kind: "actuator"; id: string }
  | { kind: "mechSensor"; id: string };

const DEFAULT_MOTOR = {
  nominalVoltageV: 12,
  freeSpeedRpm: 312,
  stallTorqueNm: 2.1,
  stallCurrentA: 9.2,
  freeCurrentA: 0.25,
};

function defaultPower(): PowerSystemSpec {
  return {
    openCircuitVoltageV: 13,
    internalResistanceOhm: 0.018,
    capacityAh: 3,
    initialStateOfCharge: 1,
    brownoutVoltageV: 9,
    maxCurrentA: 120,
  };
}

function defaultPiecePath(capacity: number): PiecePathSpec {
  const slots: Transform3[] = [];
  for (let i = 0; i < Math.max(1, capacity); i += 1) {
    slots.push({ x: -4.5 + i * 3, y: 0, z: 4 });
  }
  return {
    intakeActuatorId: "intake",
    conveyorActuatorId: "conveyor",
    flywheelActuatorId: "flywheel",
    hoodActuatorId: "hood",
    turretActuatorId: null,
    gateActuatorId: "gate",
    storageSlots: slots,
    intakePose: { x: 8.5, y: 0, z: 2 },
    muzzlePose: { x: 10.25, y: 0, z: 12, pitchDeg: 52 },
    muzzleClearanceIn: 0.25,
    wheelRadiusIn: 2,
    launchEfficiency: 0.235,
  };
}

function defaultActuator(id: string, kind: ActuatorSpec["kind"], jointId: string | null): ActuatorSpec {
  return {
    id,
    kind,
    jointId,
    motor: { ...DEFAULT_MOTOR },
    gearRatio: 1,
    efficiency: 0.8,
    rotorInertiaKgM2: 0.00008,
    loadInertiaKgM2: 0.00035,
    currentLimitA: 8,
    controllerLatencyMs: 35,
    commandRatePerS: 8,
    ...(kind === "velocity_motor" ? { targetRpm: 300 } : { travelLimit: [0, 90], kp: 0.18, kd: 0.02 }),
  };
}

function muzzleWarning(doc: RobotDoc): string | null {
  const path = doc.piecePath;
  if (!path) return null;
  const chassis = doc.chassis;
  const halfL = 0.5 * (chassis.lengthIn || 0);
  const halfW = 0.5 * (chassis.widthIn || 0);
  const height = chassis.heightIn || 0;
  const muzzle = path.muzzlePose || {};
  const clearance = path.muzzleClearanceIn || 0;
  const outside =
    Math.abs(muzzle.x || 0) >= halfL + clearance ||
    Math.abs(muzzle.y || 0) >= halfW + clearance ||
    (muzzle.z || 0) >= height + clearance;
  if (!outside) return "Muzzle plus clearance still intersects the chassis envelope.";
  if ((path.storageSlots || []).length < doc.mechanisms.capacity) {
    return `Piece path has ${(path.storageSlots || []).length} slots for capacity ${doc.mechanisms.capacity}.`;
  }
  return null;
}

function cam(doc: RobotDoc) {
  return doc.sensors?.find((s) => s.kind === "apriltag_camera");
}

function nextId(prefix: string, used: string[]) {
  let n = 1;
  while (used.includes(`${prefix}_${n}`)) n += 1;
  return `${prefix}_${n}`;
}

function slugify(raw: string) {
  const s = raw
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
  return s.length >= 2 ? s : `robot_${Date.now().toString(36)}`;
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (n: number) => void;
}) {
  return (
    <div className="slider-row">
      <label htmlFor={id}>{label}</label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="readout">
        {Number.isInteger(step) ? value : value.toFixed(2)} {unit}
      </span>
    </div>
  );
}

function intakePoly(intake: IntakeSpec) {
  const pose = intake.poseOnRobot || {};
  const x = pose.x || 0;
  const y = pose.y || 0;
  const a = ((pose.headingDeg || 0) * Math.PI) / 180;
  const reach = intake.reachIn || 5;
  const hw = (intake.widthIn || 12) / 2;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const corners = [
    [x + -s * hw, y + c * hw],
    [x + s * hw, y + -c * hw],
    [x + c * reach + s * hw, y + s * reach - c * hw],
    [x + c * reach - s * hw, y + s * reach + c * hw],
  ];
  return corners.map(([px, py]) => `${px},${-py}`).join(" ");
}

function aimLine(launcher: LauncherSpec) {
  const pose = launcher.poseOnRobot || {};
  const x = pose.x || 0;
  const y = pose.y || 0;
  const a = ((pose.headingDeg || 0) * Math.PI) / 180;
  const pitch = ((pose.pitchDeg || 0) * Math.PI) / 180;
  const len = 8 + Math.min(16, (launcher.muzzleSpeedInPerS || 180) / 20) * Math.cos(pitch);
  return { x1: x, y1: -y, x2: x + Math.cos(a) * len, y2: -(y + Math.sin(a) * len) };
}

export function RobotBuilderPage() {
  const [list, setList] = useState<PresetMeta[]>([]);
  const [id, setId] = useState("mecanum_biobuzz_4cap");
  const [doc, setDoc] = useState<RobotDoc | null>(null);
  const [sel, setSel] = useState<Sel>({ kind: "chassis" });
  const [msg, setMsg] = useState("");
  const [errs, setErrs] = useState<string[]>([]);
  const [saveAsId, setSaveAsId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [lastBbox, setLastBbox] = useState<{ lengthIn: number; widthIn: number; heightIn: number } | null>(null);
  const [previewFlywheel, setPreviewFlywheel] = useState(1);
  const [previewHood, setPreviewHood] = useState(0.6);
  const drag = useRef<{ kind: "intake" | "launcher"; id: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getJson<PresetMeta[]>("/presets/robot").then(setList);
    getJson<DefaultsBundle>("/defaults")
      .then((d) => {
        if (d.robotId) setId(d.robotId);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!id) return;
    getJson<RobotDoc>(`/presets/robot/${id}`).then((d) => {
      delete (d as { _kind?: string })._kind;
      setDoc(d);
      setErrs([]);
      setSel({ kind: "chassis" });
      setSaveAsId("");
      setLastBbox(null);
    });
  }, [id]);

  const intakes = doc?.intakes || [];
  const launchers = doc?.launchers || [];
  const rigidParts = doc?.rigidParts || [];
  const joints = doc?.joints || [];
  const actuators = doc?.actuators || [];
  const mechanismSensors = doc?.mechanismSensors || [];
  const length = doc?.chassis.lengthIn || 18;
  const width = doc?.chassis.widthIn || 18;
  const span = Math.max(length, width) / 2 + 14;

  const selectedIntake = sel.kind === "intake" ? intakes.find((i) => i.id === sel.id) : undefined;
  const selectedLauncher = sel.kind === "launcher" ? launchers.find((l) => l.id === sel.id) : undefined;
  const selectedPart = sel.kind === "part" ? rigidParts.find((p) => p.id === sel.id) : undefined;
  const selectedJoint = sel.kind === "joint" ? joints.find((j) => j.id === sel.id) : undefined;
  const selectedActuator = sel.kind === "actuator" ? actuators.find((a) => a.id === sel.id) : undefined;
  const selectedMechSensor = sel.kind === "mechSensor" ? mechanismSensors.find((s) => s.id === sel.id) : undefined;
  const launchSvg = useMemo(() => {
    if (!doc?.piecePath) return [];
    return launchArcPoints(doc.piecePath, previewFlywheel, previewHood, doc.actuators);
  }, [doc?.piecePath, doc?.actuators, previewFlywheel, previewHood]);

  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let v = Math.ceil(-span / 6) * 6; v <= span; v += 6) out.push(v);
    return out;
  }, [span]);

  function svgToRobot(clientX: number, clientY: number, svg: SVGSVGElement) {
    const rect = svg.getBoundingClientRect();
    const nx = (clientX - rect.left) / rect.width;
    const ny = (clientY - rect.top) / rect.height;
    return { x: nx * span * 2 - span, y: span - ny * span * 2 };
  }

  function patchPose(kind: "intake" | "launcher", mid: string, partial: PoseOnRobot) {
    if (!doc) return;
    if (kind === "intake") {
      setDoc({
        ...doc,
        intakes: intakes.map((i) => (i.id === mid ? { ...i, poseOnRobot: { ...(i.poseOnRobot || {}), ...partial } } : i)),
      });
    } else {
      setDoc({
        ...doc,
        launchers: launchers.map((l) => (l.id === mid ? { ...l, poseOnRobot: { ...(l.poseOnRobot || {}), ...partial } } : l)),
      });
    }
  }

  function pickAt(clientX: number, clientY: number, svg: SVGSVGElement) {
    const { x, y } = svgToRobot(clientX, clientY, svg);
    let best: Sel = { kind: "chassis" };
    let bestD = Infinity;
    for (const intake of intakes) {
      const px = intake.poseOnRobot?.x || 0;
      const py = intake.poseOnRobot?.y || 0;
      const d = Math.hypot(px - x, py - y);
      if (d < 6 && d < bestD) {
        bestD = d;
        best = { kind: "intake", id: intake.id };
      }
    }
    for (const launcher of launchers) {
      const px = launcher.poseOnRobot?.x || 0;
      const py = launcher.poseOnRobot?.y || 0;
      const d = Math.hypot(px - x, py - y);
      if (d < 6 && d < bestD) {
        bestD = d;
        best = { kind: "launcher", id: launcher.id };
      }
    }
    setSel(best);
    if (best.kind !== "chassis") drag.current = best;
  }

  async function validateAndSave(target: RobotDoc, method: "put" | "post") {
    const res = await postJson<{ ok: boolean; errors: string[] }>("/presets/validate", { kind: "robot", document: target });
    if (!res.ok) {
      setErrs(res.errors || ["Invalid robot preset."]);
      setMsg("");
      return false;
    }
    if (method === "post") await postJson("/presets/robot", target);
    else await putJson(`/presets/robot/${target.id}`, target);
    return true;
  }

  async function save() {
    if (!doc) return;
    try {
      const ok = await validateAndSave(doc, "put");
      if (!ok) return;
      setMsg("Saved robot preset via API.");
      setErrs([]);
    } catch (e) {
      setMsg("");
      setErrs([e instanceof Error ? e.message : String(e)]);
    }
  }

  async function saveAs() {
    if (!doc) return;
    const nid = slugify(saveAsId || `${doc.id}_copy`);
    const next = { ...doc, id: nid, displayName: saveAsId ? doc.displayName : `${doc.displayName} copy` };
    try {
      const ok = await validateAndSave(next, "post");
      if (!ok) return;
      setMsg(`Created ${nid} via API.`);
      setErrs([]);
      const rows = await getJson<PresetMeta[]>("/presets/robot");
      setList(rows);
      setId(nid);
    } catch (e) {
      setMsg("");
      setErrs([e instanceof Error ? e.message : String(e)]);
    }
  }

  function addIntake() {
    if (!doc) return;
    const nid = nextId("intake", intakes.map((i) => i.id));
    const item: IntakeSpec = {
      id: nid,
      poseOnRobot: { x: length / 2, y: 0, z: 2, headingDeg: 0 },
      widthIn: 12,
      reachIn: 5,
      heightIn: 4,
      cycleTimeS: doc.mechanisms.intakeCycleTimeS ?? 0.4,
      canRunWhileMoving: doc.mechanisms.canIntakeWhileMoving ?? true,
    };
    setDoc({ ...doc, intakes: [...intakes, item] });
    setSel({ kind: "intake", id: nid });
  }

  function addLauncher() {
    if (!doc) return;
    const nid = nextId("hood", launchers.map((l) => l.id));
    const item: LauncherSpec = {
      id: nid,
      poseOnRobot: { x: 3, y: 0, z: 12, headingDeg: 0, pitchDeg: 50 },
      aimMode: "chassis_fixed",
      muzzleSpeedInPerS: 200,
      spinupTimeS: 0.3,
      cycleTimeS: doc.mechanisms.scoreCycleTimeS ?? 0.6,
      canLaunchWhileMoving: doc.mechanisms.canScoreWhileMoving ?? true,
      yawRangeDeg: [-90, 90],
      pitchRangeDeg: [20, 70],
    };
    setDoc({ ...doc, launchers: [...launchers, item] });
    setSel({ kind: "launcher", id: nid });
  }

  function removeSelected() {
    if (!doc) return;
    if (sel.kind === "intake") {
      setDoc({ ...doc, intakes: intakes.filter((i) => i.id !== sel.id) });
      setSel({ kind: "chassis" });
    }
    if (sel.kind === "launcher") {
      setDoc({ ...doc, launchers: launchers.filter((l) => l.id !== sel.id) });
      setSel({ kind: "chassis" });
    }
    if (sel.kind === "part") {
      setDoc({ ...doc, rigidParts: rigidParts.filter((p) => p.id !== sel.id) });
      setSel({ kind: "chassis" });
    }
    if (sel.kind === "joint") {
      setDoc({ ...doc, joints: joints.filter((j) => j.id !== sel.id) });
      setSel({ kind: "chassis" });
    }
    if (sel.kind === "actuator") {
      setDoc({ ...doc, actuators: actuators.filter((a) => a.id !== sel.id) });
      setSel({ kind: "chassis" });
    }
    if (sel.kind === "mechSensor") {
      setDoc({ ...doc, mechanismSensors: mechanismSensors.filter((s) => s.id !== sel.id) });
      setSel({ kind: "chassis" });
    }
  }

  function enablePhysical() {
    if (!doc) return;
    const chassisPart: RigidPartSpec = doc.rigidParts?.find((p) => p.id === "chassis") || {
      id: "chassis",
      parentId: null,
      pose: { x: 0, y: 0, z: 0 },
      massKg: doc.chassis.massKg,
      collision: [{ kind: "box", sizeIn: [doc.chassis.lengthIn, doc.chassis.widthIn, doc.chassis.heightIn ?? 10] }],
    };
    const parts = doc.rigidParts?.length ? doc.rigidParts : [chassisPart];
    setDoc({
      ...doc,
      schemaVersion: "1.1.0",
      policyInterfaceVersion: doc.policyInterfaceVersion || "1.0.0",
      rigidParts: parts,
      joints: doc.joints || [],
      actuators: doc.actuators || [],
      powerSystem: doc.powerSystem || defaultPower(),
      piecePath: doc.piecePath || defaultPiecePath(doc.mechanisms.capacity),
      mechanismSensors: doc.mechanismSensors || [
        { id: "battery_voltage", kind: "battery_voltage", sampleRateHz: 50, quantization: 0.01, noiseStd: 0.025, latencyMs: 20 },
      ],
    });
    setSel({ kind: "part", id: "chassis" });
  }

  function addPart() {
    if (!doc) return;
    const nid = nextId("part", rigidParts.map((p) => p.id));
    const item: RigidPartSpec = {
      id: nid,
      parentId: "chassis",
      pose: { x: 4, y: 0, z: 2 },
      massKg: 0.4,
      collision: [{ kind: "box", sizeIn: [4, 3, 2] }],
    };
    setDoc({ ...doc, rigidParts: [...rigidParts, item] });
    setSel({ kind: "part", id: nid });
  }

  function addJoint() {
    if (!doc) return;
    const nid = nextId("joint", joints.map((j) => j.id));
    const parent = rigidParts[0]?.id || "chassis";
    const child = rigidParts.find((p) => p.id !== parent)?.id || parent;
    const item: JointSpec = {
      id: nid,
      type: "hinge",
      parentPartId: parent,
      childPartId: child,
      anchorIn: { x: 0, y: 0, z: 0 },
      axis: [0, 1, 0],
      limit: [0, 90],
    };
    setDoc({ ...doc, joints: [...joints, item] });
    setSel({ kind: "joint", id: nid });
  }

  function addActuator() {
    if (!doc) return;
    const nid = nextId("actuator", actuators.map((a) => a.id));
    const item = defaultActuator(nid, "velocity_motor", joints[0]?.id || null);
    setDoc({ ...doc, actuators: [...actuators, item] });
    setSel({ kind: "actuator", id: nid });
  }

  function addMechSensor() {
    if (!doc) return;
    const nid = nextId("sensor", mechanismSensors.map((s) => s.id));
    const item: MechanismSensorSpec = {
      id: nid,
      kind: "rpm",
      actuatorId: actuators[0]?.id,
      sampleRateHz: 50,
      quantization: 1,
      noiseStd: 1,
      latencyMs: 20,
    };
    setDoc({ ...doc, mechanismSensors: [...mechanismSensors, item] });
    setSel({ kind: "mechSensor", id: nid });
  }

  function patchPart(id: string, partial: Partial<RigidPartSpec>) {
    if (!doc) return;
    setDoc({ ...doc, rigidParts: rigidParts.map((p) => (p.id === id ? { ...p, ...partial } : p)) });
  }

  function patchJoint(id: string, partial: Partial<JointSpec>) {
    if (!doc) return;
    setDoc({ ...doc, joints: joints.map((j) => (j.id === id ? { ...j, ...partial } : j)) });
  }

  function patchActuator(id: string, partial: Partial<ActuatorSpec>) {
    if (!doc) return;
    setDoc({ ...doc, actuators: actuators.map((a) => (a.id === id ? { ...a, ...partial } : a)) });
  }

  function patchSensor(id: string, partial: Partial<MechanismSensorSpec>) {
    if (!doc) return;
    setDoc({ ...doc, mechanismSensors: mechanismSensors.map((s) => (s.id === id ? { ...s, ...partial } : s)) });
  }

  function patchPath(partial: Partial<PiecePathSpec>) {
    if (!doc) return;
    setDoc({ ...doc, piecePath: { ...(doc.piecePath || defaultPiecePath(doc.mechanisms.capacity)), ...partial } });
  }

  function patchPower(partial: Partial<PowerSystemSpec>) {
    if (!doc) return;
    setDoc({ ...doc, powerSystem: { ...(doc.powerSystem || defaultPower()), ...partial } });
  }

  async function onUploadModel(file: File | undefined) {
    if (!doc || !file) return;
    setUploading(true);
    setMsg("Converting CAD…");
    try {
      const imported = await uploadRobotModel(doc.id, file);
      const offset = doc.visualOffset || {};
      setLastBbox(imported.bbox);
      setDoc({
        ...doc,
        visualAsset: imported.visualAsset,
        collisionAsset: imported.collisionAsset,
        visualOffset: { x: offset.x || 0, y: offset.y || 0, z: offset.z || 0, yawDeg: offset.yawDeg || 0, scale: offset.scale || 1 },
        chassis: { ...doc.chassis, footprint: imported.footprint },
      });
      setMsg(`Imported ${file.name} (${imported.unitsGuess}, ${imported.bbox.lengthIn.toFixed(1)}×${imported.bbox.widthIn.toFixed(1)}×${imported.bbox.heightIn.toFixed(1)} in). Fit chassis if you want the box to match.`);
      setErrs([]);
    } catch (e) {
      setMsg("");
      setErrs([e instanceof Error ? e.message : String(e)]);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  function fitChassis() {
    if (!doc) return;
    if (lastBbox) {
      setDoc({
        ...doc,
        chassis: {
          ...doc.chassis,
          lengthIn: Math.round(lastBbox.lengthIn * 100) / 100,
          widthIn: Math.round(lastBbox.widthIn * 100) / 100,
          heightIn: Math.round(lastBbox.heightIn * 100) / 100,
        },
      });
      setMsg("Chassis length/width/height set from the model bounds.");
      return;
    }
    if (!doc.chassis.footprint?.length) {
      setErrs(["Upload a model first, then fit chassis to its bounds."]);
      return;
    }
    const xs = doc.chassis.footprint.map((p) => p.x);
    const ys = doc.chassis.footprint.map((p) => p.y);
    const lengthIn = Math.max(...xs) - Math.min(...xs);
    const widthIn = Math.max(...ys) - Math.min(...ys);
    setDoc({
      ...doc,
      chassis: {
        ...doc.chassis,
        lengthIn: Math.round(lengthIn * 100) / 100,
        widthIn: Math.round(widthIn * 100) / 100,
      },
    });
    setMsg("Chassis length/width set from the model footprint.");
  }

  async function clearModel() {
    if (!doc) return;
    try {
      await deleteRobotModel(doc.id);
    } catch {
      /* files may already be gone */
    }
    const { visualAsset: _v, collisionAsset: _c, visualOffset: _o, ...rest } = doc;
    const chassis = { ...doc.chassis };
    delete chassis.footprint;
    if (chassis.collisionShape === "mesh") chassis.collisionShape = "aabb";
    setDoc({ ...rest, chassis });
    setLastBbox(null);
    setMsg("Cleared custom 3D model.");
  }

  function patchOffset(partial: VisualOffset) {
    if (!doc) return;
    setDoc({ ...doc, visualOffset: { ...(doc.visualOffset || {}), ...partial } });
  }

  if (!doc) return <div className="page single">Loading…</div>;
  const robot = doc;
  const camera = cam(robot);

  function setCamera(partial: { fovDeg?: number; rangeIn?: number }) {
    setDoc({
      ...robot,
      sensors: upsertCameraSensor(robot.sensors || [], partial),
    });
  }

  return (
    <div className="page single">
      <div>
        <div className="page-head">
          <h2>Robot design</h2>
          <p className="note">
            Robot frame, inches: origin at chassis center, +x forward, +y left. Place intakes and launchers on the grid.
            Saves go through the API, not browser storage.
          </p>
        </div>
        <label htmlFor="robot-preset">Robot preset</label>
        <select id="robot-preset" value={id} onChange={(e) => setId(e.target.value)} style={{ maxWidth: 420 }}>
          {list.map((p) => (
            <option key={p.id} value={p.id}>
              {robotPresetLabel(p)}
            </option>
          ))}
        </select>
        <div className="form-grid" style={{ marginTop: "0.6rem" }}>
          <label htmlFor="display-name">Display name</label>
          <input id="display-name" value={doc.displayName} onChange={(e) => setDoc({ ...doc, displayName: e.target.value })} />
        </div>

        <div className="robot-work">
          <div>
            <svg
              className="field-grid"
              viewBox={`${-span} ${-span} ${span * 2} ${span * 2}`}
              role="img"
              aria-label="Robot inch grid, +x forward"
              onPointerDown={(e) => pickAt(e.clientX, e.clientY, e.currentTarget)}
              onPointerMove={(e) => {
                if (!drag.current || e.buttons === 0) return;
                const p = svgToRobot(e.clientX, e.clientY, e.currentTarget);
                patchPose(drag.current.kind, drag.current.id, { x: Math.round(p.x * 2) / 2, y: Math.round(p.y * 2) / 2 });
              }}
              onPointerUp={() => {
                drag.current = null;
              }}
              onPointerLeave={() => {
                drag.current = null;
              }}
            >
              <rect x={-span} y={-span} width={span * 2} height={span * 2} fill={theme.scene} />
              {ticks.map((v) => (
                <g key={v}>
                  <line x1={v} y1={-span} x2={v} y2={span} stroke={theme.grid} strokeWidth="0.35" />
                  <line x1={-span} y1={v} x2={span} y2={v} stroke={theme.grid} strokeWidth="0.35" />
                </g>
              ))}
              <rect
                x={-length / 2}
                y={-width / 2}
                width={length}
                height={width}
                fill={theme.chassis}
                fillOpacity={0.85}
                stroke={sel.kind === "chassis" ? theme.cream : theme.muted}
                strokeWidth={sel.kind === "chassis" ? 0.7 : 0.35}
              />
              <polygon points={`${length / 2 - 1.5},0 ${length / 2},${-2} ${length / 2},${2}`} fill={theme.maroonDark} />
              {launchSvg.length > 1 && (
                <polyline
                  points={launchSvg.map(([px, , pz]) => `${px},${pz}`).join(" ")}
                  fill="none"
                  stroke={theme.goldBright}
                  strokeWidth={0.45}
                />
              )}
              {intakes.map((intake) => (
                <polygon
                  key={intake.id}
                  points={intakePoly(intake)}
                  fill={theme.intake}
                  fillOpacity={sel.kind === "intake" && sel.id === intake.id ? 0.85 : 0.45}
                  stroke={sel.kind === "intake" && sel.id === intake.id ? theme.cream : theme.intake}
                  strokeWidth={0.45}
                />
              ))}
              {launchers.map((launcher) => {
                const pose = launcher.poseOnRobot || {};
                const line = aimLine(launcher);
                const on = sel.kind === "launcher" && sel.id === launcher.id;
                return (
                  <g key={launcher.id}>
                    <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} stroke={theme.goldBright} strokeWidth={0.5} />
                    <circle cx={pose.x || 0} cy={-(pose.y || 0)} r={1.4} fill={theme.gold} stroke={on ? theme.cream : "transparent"} strokeWidth={0.4} />
                  </g>
                );
              })}
              <line x1={0} y1={0} x2={6} y2={0} stroke={theme.gold} strokeWidth={0.35} />
              <text x={span - 8} y={span - 2} fill={theme.muted} fontSize="2.2">
                +x fwd
              </text>
            </svg>
            <div className="legend">
              <span>
                <i style={{ background: theme.chassis }} /> chassis
              </span>
              <span>
                <i style={{ background: theme.intake }} /> intake
              </span>
              <span>
                <i style={{ background: theme.gold }} /> launcher
              </span>
            </div>
            <div className="robot-preview" aria-label="Robot 3D preview">
              <RobotPreview
                design={{
                  chassis: doc.chassis,
                  intakes,
                  launchers,
                  visualAsset: typeof doc.visualAsset === "string" ? doc.visualAsset : null,
                  visualOffset: doc.visualOffset,
                  collisionShape: doc.chassis.collisionShape,
                  sensors: doc.sensors,
                  rigidParts: doc.rigidParts,
                  joints: doc.joints,
                  actuators: doc.actuators,
                  piecePath: doc.piecePath,
                }}
                showFov={Boolean(camera)}
                showHull={doc.chassis.collisionShape === "mesh"}
                launchPreview={{ flywheelFrac: previewFlywheel, hoodFrac: previewHood }}
              />
            </div>
          </div>

          <div>
            <div className="row">
              <button type="button" onClick={addIntake}>
                Add intake
              </button>
              <button type="button" onClick={addLauncher}>
                Add launcher
              </button>
              {(sel.kind === "intake" ||
                sel.kind === "launcher" ||
                sel.kind === "part" ||
                sel.kind === "joint" ||
                sel.kind === "actuator" ||
                sel.kind === "mechSensor") && (
                <button type="button" onClick={removeSelected}>
                  Remove selected
                </button>
              )}
            </div>

            <div className="card">
              <h3>Drivetrain</h3>
              <div className="form-grid">
                <label htmlFor="dt">Type</label>
                <select
                  id="dt"
                  value={doc.drivetrain.type}
                  onChange={(e) => setDoc({ ...doc, drivetrain: { ...doc.drivetrain, type: e.target.value } })}
                >
                  <option value="mecanum">mecanum</option>
                  <option value="tank">tank</option>
                  <option value="swerve">swerve</option>
                </select>
                <label htmlFor="track">Track width (in)</label>
                <input
                  id="track"
                  className="narrow"
                  type="number"
                  value={doc.drivetrain.trackWidthIn}
                  onChange={(e) => setDoc({ ...doc, drivetrain: { ...doc.drivetrain, trackWidthIn: Number(e.target.value) } })}
                />
                <label htmlFor="action-tier">Action tier</label>
                <select
                  id="action-tier"
                  value={doc.defaultActionTier || "high_level_waypoint"}
                  onChange={(e) => setDoc({ ...doc, defaultActionTier: e.target.value as ActionTier })}
                >
                  <option value="high_level_waypoint">high_level_waypoint</option>
                  <option value="low_level_velocity">low_level_velocity</option>
                  <option value="physical_actuators">physical_actuators</option>
                </select>
              </div>
              <p className="note">
                Swerve is treated as holonomic (same velocity clip as mecanum). There is no per-module inverse kinematics.
              </p>
            </div>

            <div className="card">
              <h3>Motion limits (MeepMeep)</h3>
              <Slider
                id="max-vel"
                label="Max vel"
                value={doc.constraints.maxVelInPerS}
                min={0}
                max={80}
                step={1}
                unit="in/s"
                onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxVelInPerS: n } })}
              />
              <Slider
                id="max-acc"
                label="Max accel"
                value={doc.constraints.maxAccelInPerS2}
                min={0}
                max={80}
                step={1}
                unit="in/s²"
                onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxAccelInPerS2: n } })}
              />
              <Slider
                id="max-ang"
                label="Max ang vel"
                value={doc.constraints.maxAngVelDegPerS}
                min={0}
                max={360}
                step={1}
                unit="deg/s"
                onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxAngVelDegPerS: n } })}
              />
            </div>

            <div className="card">
              <h3>Chassis</h3>
              <div className="form-grid">
                <label htmlFor="chassis-l">Length (in)</label>
                <input
                  id="chassis-l"
                  className="narrow"
                  type="number"
                  value={doc.chassis.lengthIn}
                  onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, lengthIn: Number(e.target.value) } })}
                />
                <label htmlFor="chassis-w">Width (in)</label>
                <input
                  id="chassis-w"
                  className="narrow"
                  type="number"
                  value={doc.chassis.widthIn}
                  onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, widthIn: Number(e.target.value) } })}
                />
                <label htmlFor="chassis-h">Height (in)</label>
                <input
                  id="chassis-h"
                  className="narrow"
                  type="number"
                  value={doc.chassis.heightIn ?? 10}
                  onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, heightIn: Number(e.target.value) } })}
                />
                <label htmlFor="chassis-m">Mass (kg)</label>
                <input
                  id="chassis-m"
                  className="narrow"
                  type="number"
                  value={doc.chassis.massKg}
                  onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, massKg: Number(e.target.value) } })}
                />
              </div>
              <Slider
                id="cap"
                label="Capacity"
                value={doc.mechanisms.capacity}
                min={0}
                max={10}
                step={1}
                unit=""
                onChange={(n) => setDoc({ ...doc, mechanisms: { ...doc.mechanisms, capacity: n } })}
              />
            </div>

            <div className="card">
              <h3>3D model</h3>
              <p className="note">
                Upload STL, OBJ, GLB, glTF, or STEP. Files convert to a light GLB for the viewer. Physics stays the chassis box unless you enable mesh collision (slower; for fidelity, not overnight PPO).
              </p>
              <input
                ref={fileRef}
                id="robot-cad"
                type="file"
                accept=".glb,.gltf,.stl,.obj,.step,.stp"
                hidden
                onChange={(e) => onUploadModel(e.target.files?.[0])}
              />
              <div className="row">
                <button type="button" disabled={uploading} onClick={() => fileRef.current?.click()}>
                  {uploading ? "Converting…" : "Upload model"}
                </button>
                <button type="button" disabled={!doc.visualAsset} onClick={fitChassis}>
                  Fit chassis to model
                </button>
                <button type="button" disabled={!doc.visualAsset} onClick={() => void clearModel()}>
                  Clear model
                </button>
              </div>
              {typeof doc.visualAsset === "string" && doc.visualAsset && (
                <p className="note">Loaded {doc.visualAsset}</p>
              )}
              <label className="check">
                <input
                  type="checkbox"
                  checked={doc.chassis.collisionShape === "mesh"}
                  disabled={!doc.visualAsset}
                  onChange={(e) =>
                    setDoc({
                      ...doc,
                      chassis: { ...doc.chassis, collisionShape: e.target.checked ? "mesh" : "aabb" },
                    })
                  }
                />
                Use mesh for collision
              </label>
              <div className="form-grid">
                <label htmlFor="off-x">Offset x (in)</label>
                <input
                  id="off-x"
                  className="narrow"
                  type="number"
                  value={doc.visualOffset?.x ?? 0}
                  onChange={(e) => patchOffset({ x: Number(e.target.value) })}
                />
                <label htmlFor="off-y">Offset y (in)</label>
                <input
                  id="off-y"
                  className="narrow"
                  type="number"
                  value={doc.visualOffset?.y ?? 0}
                  onChange={(e) => patchOffset({ y: Number(e.target.value) })}
                />
                <label htmlFor="off-z">Offset z (in)</label>
                <input
                  id="off-z"
                  className="narrow"
                  type="number"
                  value={doc.visualOffset?.z ?? 0}
                  onChange={(e) => patchOffset({ z: Number(e.target.value) })}
                />
                <label htmlFor="off-yaw">Yaw (deg)</label>
                <input
                  id="off-yaw"
                  className="narrow"
                  type="number"
                  value={doc.visualOffset?.yawDeg ?? 0}
                  onChange={(e) => patchOffset({ yawDeg: Number(e.target.value) })}
                />
              </div>
              <Slider
                id="off-scale"
                label="Scale"
                value={doc.visualOffset?.scale ?? 1}
                min={0.1}
                max={4}
                step={0.05}
                unit="×"
                onChange={(n) => patchOffset({ scale: n })}
              />
            </div>

            {selectedIntake && (
              <div className="card">
                <h3>Intake {selectedIntake.id}</h3>
                <div className="form-grid">
                  <label htmlFor="in-x">x (in)</label>
                  <input
                    id="in-x"
                    type="number"
                    className="narrow"
                    value={selectedIntake.poseOnRobot?.x ?? 0}
                    onChange={(e) => patchPose("intake", selectedIntake.id, { x: Number(e.target.value) })}
                  />
                  <label htmlFor="in-y">y (in)</label>
                  <input
                    id="in-y"
                    type="number"
                    className="narrow"
                    value={selectedIntake.poseOnRobot?.y ?? 0}
                    onChange={(e) => patchPose("intake", selectedIntake.id, { y: Number(e.target.value) })}
                  />
                  <label htmlFor="in-h">heading (deg)</label>
                  <input
                    id="in-h"
                    type="number"
                    className="narrow"
                    value={selectedIntake.poseOnRobot?.headingDeg ?? 0}
                    onChange={(e) => patchPose("intake", selectedIntake.id, { headingDeg: Number(e.target.value) })}
                  />
                </div>
                <Slider
                  id="in-w"
                  label="Width"
                  value={selectedIntake.widthIn ?? 12}
                  min={2}
                  max={18}
                  step={0.5}
                  unit="in"
                  onChange={(n) =>
                    setDoc({ ...doc, intakes: intakes.map((i) => (i.id === selectedIntake.id ? { ...i, widthIn: n } : i)) })
                  }
                />
                <Slider
                  id="in-r"
                  label="Reach"
                  value={selectedIntake.reachIn ?? 5}
                  min={1}
                  max={12}
                  step={0.5}
                  unit="in"
                  onChange={(n) =>
                    setDoc({ ...doc, intakes: intakes.map((i) => (i.id === selectedIntake.id ? { ...i, reachIn: n } : i)) })
                  }
                />
                <Slider
                  id="in-c"
                  label="Cycle"
                  value={selectedIntake.cycleTimeS ?? 0.4}
                  min={0.05}
                  max={3}
                  step={0.05}
                  unit="s"
                  onChange={(n) =>
                    setDoc({ ...doc, intakes: intakes.map((i) => (i.id === selectedIntake.id ? { ...i, cycleTimeS: n } : i)) })
                  }
                />
                <label className="check">
                  <input
                    type="checkbox"
                    checked={selectedIntake.canRunWhileMoving ?? true}
                    onChange={(e) =>
                      setDoc({
                        ...doc,
                        intakes: intakes.map((i) => (i.id === selectedIntake.id ? { ...i, canRunWhileMoving: e.target.checked } : i)),
                      })
                    }
                  />
                  Can run while moving
                </label>
              </div>
            )}

            {selectedLauncher && (
              <div className="card">
                <h3>Launcher {selectedLauncher.id}</h3>
                <div className="form-grid">
                  <label htmlFor="ln-mode">Aim mode</label>
                  <select
                    id="ln-mode"
                    value={selectedLauncher.aimMode || "chassis_fixed"}
                    onChange={(e) =>
                      setDoc({
                        ...doc,
                        launchers: launchers.map((l) =>
                          l.id === selectedLauncher.id ? { ...l, aimMode: e.target.value as LauncherSpec["aimMode"] } : l,
                        ),
                      })
                    }
                  >
                    <option value="chassis_fixed">chassis_fixed</option>
                    <option value="turret">turret</option>
                  </select>
                  <label htmlFor="ln-x">x (in)</label>
                  <input
                    id="ln-x"
                    type="number"
                    className="narrow"
                    value={selectedLauncher.poseOnRobot?.x ?? 0}
                    onChange={(e) => patchPose("launcher", selectedLauncher.id, { x: Number(e.target.value) })}
                  />
                  <label htmlFor="ln-y">y (in)</label>
                  <input
                    id="ln-y"
                    type="number"
                    className="narrow"
                    value={selectedLauncher.poseOnRobot?.y ?? 0}
                    onChange={(e) => patchPose("launcher", selectedLauncher.id, { y: Number(e.target.value) })}
                  />
                  <label htmlFor="ln-z">z (in)</label>
                  <input
                    id="ln-z"
                    type="number"
                    className="narrow"
                    value={selectedLauncher.poseOnRobot?.z ?? 12}
                    onChange={(e) => patchPose("launcher", selectedLauncher.id, { z: Number(e.target.value) })}
                  />
                </div>
                <Slider
                  id="ln-yaw"
                  label="Yaw"
                  value={selectedLauncher.poseOnRobot?.headingDeg ?? 0}
                  min={-180}
                  max={180}
                  step={1}
                  unit="deg"
                  onChange={(n) => patchPose("launcher", selectedLauncher.id, { headingDeg: n })}
                />
                <Slider
                  id="ln-pitch"
                  label="Pitch"
                  value={selectedLauncher.poseOnRobot?.pitchDeg ?? 0}
                  min={0}
                  max={85}
                  step={1}
                  unit="deg"
                  onChange={(n) => patchPose("launcher", selectedLauncher.id, { pitchDeg: n })}
                />
                <Slider
                  id="ln-spd"
                  label="Muzzle"
                  value={selectedLauncher.muzzleSpeedInPerS ?? 200}
                  min={40}
                  max={400}
                  step={5}
                  unit="in/s"
                  onChange={(n) =>
                    setDoc({
                      ...doc,
                      launchers: launchers.map((l) => (l.id === selectedLauncher.id ? { ...l, muzzleSpeedInPerS: n } : l)),
                    })
                  }
                />
                <Slider
                  id="ln-spin"
                  label="Spin-up"
                  value={selectedLauncher.spinupTimeS ?? 0}
                  min={0}
                  max={2}
                  step={0.05}
                  unit="s"
                  onChange={(n) =>
                    setDoc({
                      ...doc,
                      launchers: launchers.map((l) => (l.id === selectedLauncher.id ? { ...l, spinupTimeS: n } : l)),
                    })
                  }
                />
                <Slider
                  id="ln-cyc"
                  label="Cycle"
                  value={selectedLauncher.cycleTimeS ?? 0.6}
                  min={0.05}
                  max={3}
                  step={0.05}
                  unit="s"
                  onChange={(n) =>
                    setDoc({
                      ...doc,
                      launchers: launchers.map((l) => (l.id === selectedLauncher.id ? { ...l, cycleTimeS: n } : l)),
                    })
                  }
                />
                <label className="check">
                  <input
                    type="checkbox"
                    checked={selectedLauncher.canLaunchWhileMoving ?? true}
                    onChange={(e) =>
                      setDoc({
                        ...doc,
                        launchers: launchers.map((l) =>
                          l.id === selectedLauncher.id ? { ...l, canLaunchWhileMoving: e.target.checked } : l,
                        ),
                      })
                    }
                  />
                  Can launch while moving
                </label>
              </div>
            )}

            <div className="card">
              <h3>Camera</h3>
              <p className="note">
                {camera
                  ? "FOV and range write the AprilTag camera on this preset. Replay can draw the cone from robot pose."
                  : "This preset has no camera yet. Moving FOV or range creates an AprilTag camera sensor (not a silent no-op)."}
              </p>
              <Slider
                id="fov"
                label="FOV"
                value={camera?.fovDeg ?? 70}
                min={30}
                max={120}
                step={1}
                unit="deg"
                onChange={(n) => setCamera({ fovDeg: n })}
              />
              <Slider
                id="range"
                label="Range"
                value={camera?.rangeIn ?? 96}
                min={12}
                max={200}
                step={1}
                unit="in"
                onChange={(n) => setCamera({ rangeIn: n })}
              />
            </div>
          </div>
        </div>

        <div className="card">
          <h3>Physical mechanism</h3>
          <p className="note">
            Hierarchy, joints, actuators, sensors, and piece path. Schema 1.1 is required for competitive physical simulation.
            Gold arc is a ballistic launch preview from muzzle pose, flywheel RPM, and hood travel.
          </p>
          <div className="row">
            <button type="button" onClick={enablePhysical}>
              Enable physical fields
            </button>
            <button type="button" onClick={addPart}>
              Add part
            </button>
            <button type="button" onClick={addJoint}>
              Add joint
            </button>
            <button type="button" onClick={addActuator}>
              Add actuator
            </button>
            <button type="button" onClick={addMechSensor}>
              Add sensor
            </button>
          </div>
          {muzzleWarning(doc) && <p className="note">{muzzleWarning(doc)}</p>}
          <div className="legend">
            {rigidParts.map((part) => (
              <button key={part.id} type="button" className={sel.kind === "part" && sel.id === part.id ? "on" : ""} onClick={() => setSel({ kind: "part", id: part.id })}>
                {part.id}
                {part.parentId ? ` ← ${part.parentId}` : " (root)"}
              </button>
            ))}
          </div>
          {selectedPart && (
            <div className="form-grid">
              <label htmlFor="part-id">Part id</label>
              <input id="part-id" value={selectedPart.id} onChange={(e) => patchPart(selectedPart.id, { id: e.target.value })} />
              <label htmlFor="part-parent">Parent</label>
              <select
                id="part-parent"
                value={selectedPart.parentId || ""}
                onChange={(e) => patchPart(selectedPart.id, { parentId: e.target.value || null })}
              >
                <option value="">none (root)</option>
                {rigidParts.filter((p) => p.id !== selectedPart.id).map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.id}
                  </option>
                ))}
              </select>
              <label htmlFor="part-mass">Mass (kg)</label>
              <input id="part-mass" className="narrow" type="number" value={selectedPart.massKg} onChange={(e) => patchPart(selectedPart.id, { massKg: Number(e.target.value) })} />
              <label htmlFor="part-x">x (in)</label>
              <input id="part-x" className="narrow" type="number" value={selectedPart.pose?.x ?? 0} onChange={(e) => patchPart(selectedPart.id, { pose: { ...(selectedPart.pose || {}), x: Number(e.target.value) } })} />
              <label htmlFor="part-y">y (in)</label>
              <input id="part-y" className="narrow" type="number" value={selectedPart.pose?.y ?? 0} onChange={(e) => patchPart(selectedPart.id, { pose: { ...(selectedPart.pose || {}), y: Number(e.target.value) } })} />
              <label htmlFor="part-z">z (in)</label>
              <input id="part-z" className="narrow" type="number" value={selectedPart.pose?.z ?? 0} onChange={(e) => patchPart(selectedPart.id, { pose: { ...(selectedPart.pose || {}), z: Number(e.target.value) } })} />
              <label htmlFor="part-visual">Visual asset</label>
              <input id="part-visual" value={selectedPart.visualAsset || ""} onChange={(e) => patchPart(selectedPart.id, { visualAsset: e.target.value || undefined })} />
            </div>
          )}
          <p className="stat">Joints</p>
          <div className="legend">
            {joints.map((joint) => (
              <button key={joint.id} type="button" className={sel.kind === "joint" && sel.id === joint.id ? "on" : ""} onClick={() => setSel({ kind: "joint", id: joint.id })}>
                {joint.id}
              </button>
            ))}
          </div>
          {selectedJoint && (
            <div className="form-grid">
              <label htmlFor="joint-type">Type</label>
              <select id="joint-type" value={selectedJoint.type} onChange={(e) => patchJoint(selectedJoint.id, { type: e.target.value as JointSpec["type"] })}>
                <option value="fixed">fixed</option>
                <option value="hinge">hinge</option>
                <option value="slide">slide</option>
              </select>
              <label htmlFor="joint-parent">Parent part</label>
              <select id="joint-parent" value={selectedJoint.parentPartId} onChange={(e) => patchJoint(selectedJoint.id, { parentPartId: e.target.value })}>
                {rigidParts.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.id}
                  </option>
                ))}
              </select>
              <label htmlFor="joint-child">Child part</label>
              <select id="joint-child" value={selectedJoint.childPartId} onChange={(e) => patchJoint(selectedJoint.id, { childPartId: e.target.value })}>
                {rigidParts.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.id}
                  </option>
                ))}
              </select>
              <label htmlFor="joint-lo">Limit min</label>
              <input id="joint-lo" className="narrow" type="number" value={selectedJoint.limit?.[0] ?? 0} onChange={(e) => patchJoint(selectedJoint.id, { limit: [Number(e.target.value), selectedJoint.limit?.[1] ?? 90] })} />
              <label htmlFor="joint-hi">Limit max</label>
              <input id="joint-hi" className="narrow" type="number" value={selectedJoint.limit?.[1] ?? 90} onChange={(e) => patchJoint(selectedJoint.id, { limit: [selectedJoint.limit?.[0] ?? 0, Number(e.target.value)] })} />
            </div>
          )}
          <p className="stat">Actuators</p>
          <div className="legend">
            {actuators.map((actuator) => (
              <button key={actuator.id} type="button" className={sel.kind === "actuator" && sel.id === actuator.id ? "on" : ""} onClick={() => setSel({ kind: "actuator", id: actuator.id })}>
                {actuator.id}
              </button>
            ))}
          </div>
          {selectedActuator && (
            <div className="form-grid">
              <label htmlFor="act-kind">Kind</label>
              <select id="act-kind" value={selectedActuator.kind} onChange={(e) => patchActuator(selectedActuator.id, { kind: e.target.value as ActuatorSpec["kind"] })}>
                <option value="velocity_motor">velocity_motor</option>
                <option value="position_motor">position_motor</option>
                <option value="servo">servo</option>
              </select>
              <label htmlFor="act-joint">Joint</label>
              <select
                id="act-joint"
                value={selectedActuator.jointId || ""}
                onChange={(e) => patchActuator(selectedActuator.id, { jointId: e.target.value || null })}
              >
                <option value="">none</option>
                {joints.map((j) => (
                  <option key={j.id} value={j.id}>
                    {j.id}
                  </option>
                ))}
              </select>
              <label htmlFor="act-current">Current limit (A)</label>
              <input id="act-current" className="narrow" type="number" value={selectedActuator.currentLimitA} onChange={(e) => patchActuator(selectedActuator.id, { currentLimitA: Number(e.target.value) })} />
              <label htmlFor="act-rpm">Target RPM</label>
              <input id="act-rpm" className="narrow" type="number" value={selectedActuator.targetRpm ?? 0} onChange={(e) => patchActuator(selectedActuator.id, { targetRpm: Number(e.target.value) })} />
              <label htmlFor="act-lat">Latency (ms)</label>
              <input id="act-lat" className="narrow" type="number" value={selectedActuator.controllerLatencyMs} onChange={(e) => patchActuator(selectedActuator.id, { controllerLatencyMs: Number(e.target.value) })} />
            </div>
          )}
          <p className="stat">Mechanism sensors</p>
          <div className="legend">
            {mechanismSensors.map((sensor) => (
              <button key={sensor.id} type="button" className={sel.kind === "mechSensor" && sel.id === sensor.id ? "on" : ""} onClick={() => setSel({ kind: "mechSensor", id: sensor.id })}>
                {sensor.id}
              </button>
            ))}
          </div>
          {selectedMechSensor && (
            <div className="form-grid">
              <label htmlFor="ms-kind">Kind</label>
              <select id="ms-kind" value={selectedMechSensor.kind} onChange={(e) => patchSensor(selectedMechSensor.id, { kind: e.target.value as MechanismSensorKind })}>
                <option value="rpm">rpm</option>
                <option value="encoder">encoder</option>
                <option value="joint_position">joint_position</option>
                <option value="motor_current">motor_current</option>
                <option value="battery_voltage">battery_voltage</option>
                <option value="beam_break">beam_break</option>
              </select>
              <label htmlFor="ms-act">Actuator</label>
              <select
                id="ms-act"
                value={selectedMechSensor.actuatorId || ""}
                onChange={(e) => patchSensor(selectedMechSensor.id, { actuatorId: e.target.value || undefined })}
              >
                <option value="">none</option>
                {actuators.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.id}
                  </option>
                ))}
              </select>
              <label htmlFor="ms-rate">Sample rate (Hz)</label>
              <input id="ms-rate" className="narrow" type="number" value={selectedMechSensor.sampleRateHz} onChange={(e) => patchSensor(selectedMechSensor.id, { sampleRateHz: Number(e.target.value) })} />
              <label htmlFor="ms-noise">Noise std</label>
              <input id="ms-noise" className="narrow" type="number" value={selectedMechSensor.noiseStd ?? 0} onChange={(e) => patchSensor(selectedMechSensor.id, { noiseStd: Number(e.target.value) })} />
              <label htmlFor="ms-lat">Latency (ms)</label>
              <input id="ms-lat" className="narrow" type="number" value={selectedMechSensor.latencyMs ?? 0} onChange={(e) => patchSensor(selectedMechSensor.id, { latencyMs: Number(e.target.value) })} />
            </div>
          )}
          <p className="stat">Electrical</p>
          <div className="form-grid">
            <label htmlFor="ps-v">Open-circuit V</label>
            <input id="ps-v" className="narrow" type="number" value={doc.powerSystem?.openCircuitVoltageV ?? 13} onChange={(e) => patchPower({ openCircuitVoltageV: Number(e.target.value) })} />
            <label htmlFor="ps-r">Internal R (ohm)</label>
            <input id="ps-r" className="narrow" type="number" value={doc.powerSystem?.internalResistanceOhm ?? 0.018} onChange={(e) => patchPower({ internalResistanceOhm: Number(e.target.value) })} />
            <label htmlFor="ps-ah">Capacity (Ah)</label>
            <input id="ps-ah" className="narrow" type="number" value={doc.powerSystem?.capacityAh ?? 3} onChange={(e) => patchPower({ capacityAh: Number(e.target.value) })} />
            <label htmlFor="ps-soc">Initial SoC</label>
            <input id="ps-soc" className="narrow" type="number" step={0.01} value={doc.powerSystem?.initialStateOfCharge ?? 1} onChange={(e) => patchPower({ initialStateOfCharge: Number(e.target.value) })} />
            <label htmlFor="ps-bo">Brownout V</label>
            <input id="ps-bo" className="narrow" type="number" value={doc.powerSystem?.brownoutVoltageV ?? 9} onChange={(e) => patchPower({ brownoutVoltageV: Number(e.target.value) })} />
          </div>
          <p className="stat">Piece path / muzzle</p>
          <div className="form-grid">
            <label htmlFor="pp-intake">Intake actuator</label>
            <input id="pp-intake" value={doc.piecePath?.intakeActuatorId || ""} onChange={(e) => patchPath({ intakeActuatorId: e.target.value })} />
            <label htmlFor="pp-conv">Conveyor actuator</label>
            <input id="pp-conv" value={doc.piecePath?.conveyorActuatorId || ""} onChange={(e) => patchPath({ conveyorActuatorId: e.target.value })} />
            <label htmlFor="pp-fly">Flywheel actuator</label>
            <input id="pp-fly" value={doc.piecePath?.flywheelActuatorId || ""} onChange={(e) => patchPath({ flywheelActuatorId: e.target.value })} />
            <label htmlFor="pp-gate">Gate actuator</label>
            <input id="pp-gate" value={doc.piecePath?.gateActuatorId || ""} onChange={(e) => patchPath({ gateActuatorId: e.target.value })} />
            <label htmlFor="pp-hood">Hood actuator</label>
            <input id="pp-hood" value={doc.piecePath?.hoodActuatorId || ""} onChange={(e) => patchPath({ hoodActuatorId: e.target.value || null })} />
            <label htmlFor="pp-mx">Muzzle x</label>
            <input id="pp-mx" className="narrow" type="number" value={doc.piecePath?.muzzlePose?.x ?? 0} onChange={(e) => patchPath({ muzzlePose: { ...(doc.piecePath?.muzzlePose || {}), x: Number(e.target.value) } })} />
            <label htmlFor="pp-my">Muzzle y</label>
            <input id="pp-my" className="narrow" type="number" value={doc.piecePath?.muzzlePose?.y ?? 0} onChange={(e) => patchPath({ muzzlePose: { ...(doc.piecePath?.muzzlePose || {}), y: Number(e.target.value) } })} />
            <label htmlFor="pp-mz">Muzzle z</label>
            <input id="pp-mz" className="narrow" type="number" value={doc.piecePath?.muzzlePose?.z ?? 12} onChange={(e) => patchPath({ muzzlePose: { ...(doc.piecePath?.muzzlePose || {}), z: Number(e.target.value) } })} />
            <label htmlFor="pp-eff">Launch efficiency</label>
            <input id="pp-eff" className="narrow" type="number" step={0.01} value={doc.piecePath?.launchEfficiency ?? 0.235} onChange={(e) => patchPath({ launchEfficiency: Number(e.target.value) })} />
            <label htmlFor="pp-slots">Storage slots</label>
            <input
              id="pp-slots"
              type="number"
              className="narrow"
              value={(doc.piecePath?.storageSlots || []).length}
              onChange={(e) => {
                const n = Math.max(1, Number(e.target.value));
                const slots = [...(doc.piecePath?.storageSlots || [])];
                while (slots.length < n) slots.push({ x: -4.5 + slots.length * 3, y: 0, z: 4 });
                patchPath({ storageSlots: slots.slice(0, n) });
              }}
            />
          </div>
          <Slider
            id="prev-fly"
            label="Preview flywheel"
            value={previewFlywheel}
            min={0}
            max={1}
            step={0.05}
            unit=""
            onChange={setPreviewFlywheel}
          />
          <Slider
            id="prev-hood"
            label="Preview hood"
            value={previewHood}
            min={0}
            max={1}
            step={0.05}
            unit=""
            onChange={setPreviewHood}
          />
        </div>

        <div className="card">
          <h3>Save</h3>
          <div className="row">
            <button className="primary" type="button" onClick={save}>
              Save robot preset
            </button>
          </div>
          <label htmlFor="save-as">Save as new id</label>
          <div className="row">
            <input
              id="save-as"
              placeholder="team_hood_v1"
              value={saveAsId}
              onChange={(e) => setSaveAsId(e.target.value)}
              style={{ maxWidth: 280 }}
            />
            <button type="button" onClick={saveAs}>
              Save as new
            </button>
          </div>
          <p className="note">Use Save as so shipped presets stay intact. Ids are lowercase with underscores.</p>
        </div>
        {errs.map((e) => (
          <div key={e} className="banner">
            {e}
          </div>
        ))}
        <p className="note">{msg}</p>
      </div>
    </div>
  );
}
