export const API = "/api/v1";

export class ApiError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = "ApiError";
  }
}

function emitError(code: string, message: string) {
  window.dispatchEvent(new CustomEvent("talongym-error", { detail: { code, message } }));
}

async function readError(res: Response, path: string): Promise<ApiError> {
  try {
    const body = await res.json();
    const err = body?.error || body?.detail?.error || body?.detail || body;
    const code = String(err.code || res.status);
    const message = String(err.message || (typeof err === "string" ? err : `${res.status} ${path}`));
    return new ApiError(code, message);
  } catch {
    return new ApiError(String(res.status), `${res.status} ${path}`);
  }
}

async function fail(res: Response, path: string): Promise<never> {
  const err = await readError(res, path);
  emitError(err.code, err.message);
  throw err;
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) await fail(res, path);
  return res.json() as Promise<T>;
}

export async function postJson<T>(path: string, body: unknown = {}): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) await fail(res, path);
  const text = await res.text();
  return text ? (JSON.parse(text) as T) : ({} as T);
}

export async function putJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) await fail(res, path);
  return res.json() as Promise<T>;
}

export async function postText(path: string, body: unknown = {}): Promise<string> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) await fail(res, path);
  return res.text();
}

export type Health = {
  ok: boolean;
  engine: string;
  fieldEngine?: string;
  db?: string;
  version?: string;
  computeProfile?: string;
  nEnvs?: number;
};

export type ComputeInfo = {
  hardware: { cpuCount: number; ramGb: number | null; cuda: boolean; platform: string };
  profile: string;
  nEnvs: number;
  override: string | null;
  trainingId: string;
  easyTrainingId: string;
  profileTrainingId: string;
  envVar: string;
};

export type DefaultsBundle = {
  fieldId: string;
  robotId: string;
  scoringId: string;
  trainingId?: string;
  season?: string;
};

export type StartSlot = {
  id: string;
  alliance: "red" | "blue";
  slot: number;
  pose: { x: number; y: number; headingDeg: number };
  legalRegion?: { kind: string; width?: number; depth?: number; radius?: number };
};

export type MatchRobotSetup = {
  id: string;
  enabled: boolean;
  dynamic?: boolean;
  startSlotId: string;
  offset: { x: number; y: number; headingDeg: number };
};

export type MatchSetup = {
  robots: MatchRobotSetup[];
};

export type PresetMeta = {
  id: string;
  displayName: string;
  season?: string;
  manualRevision?: string;
  verifyAgainstManual?: boolean;
  stale?: boolean;
  computeProfile?: string;
};

export function robotPresetLabel(p: Pick<PresetMeta, "id" | "displayName">): string {
  return p.displayName || p.id;
}

export type VisualOffset = {
  x?: number;
  y?: number;
  z?: number;
  /** Preferred robot-CAD yaw. Robot presets store this, not headingDeg. */
  yawDeg?: number;
  /** Legacy alias accepted on old snapshots; ignored when yawDeg is present. */
  headingDeg?: number;
  scale?: number;
};

export type RobotModelImport = {
  visualAsset: string;
  collisionAsset: string;
  bbox: { lengthIn: number; widthIn: number; heightIn: number };
  footprint: { x: number; y: number }[];
  unitsGuess: string;
  faceCount: number;
};

export async function uploadRobotModel(presetId: string, file: File): Promise<RobotModelImport> {
  const path = `/presets/robot/${presetId}/model`;
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API}${path}`, { method: "POST", body });
  if (!res.ok) await fail(res, path);
  return res.json() as Promise<RobotModelImport>;
}

export async function deleteRobotModel(presetId: string): Promise<void> {
  const path = `/presets/robot/${presetId}/model`;
  const res = await fetch(`${API}${path}`, { method: "DELETE" });
  if (!res.ok) await fail(res, path);
}

export type PoseOnRobot = {
  x?: number;
  y?: number;
  z?: number;
  headingDeg?: number;
  pitchDeg?: number;
};

export type ActionTier = "high_level_waypoint" | "low_level_velocity" | "physical_actuators";

export type Transform3 = {
  x?: number;
  y?: number;
  z?: number;
  rollDeg?: number;
  pitchDeg?: number;
  yawDeg?: number;
};

export type CollisionShapeSpec =
  | {
      kind: "box";
      sizeIn: [number, number, number];
      pose?: Transform3;
      friction?: number;
      restitution?: number;
    }
  | {
      kind: "sphere" | "cylinder" | "capsule";
      radiusIn: number;
      lengthIn?: number;
      pose?: Transform3;
      friction?: number;
      restitution?: number;
    }
  | {
      kind: "convex_mesh";
      asset: string;
      pose?: Transform3;
      friction?: number;
      restitution?: number;
    };

export type RigidPartSpec = {
  id: string;
  parentId?: string | null;
  pose?: Transform3;
  massKg: number;
  centerOfMassIn?: Transform3;
  inertiaKgM2?: [number, number, number];
  collision: CollisionShapeSpec[];
  visualAsset?: string;
  visualPose?: Transform3;
};

export type JointSpec = {
  id: string;
  type: "fixed" | "hinge" | "slide";
  parentPartId: string;
  childPartId: string;
  anchorIn?: Transform3;
  axis?: [number, number, number];
  limit?: [number, number];
  damping?: number;
  frictionLoss?: number;
  backlash?: number;
};

export type MotorCurveSpec = {
  nominalVoltageV: number;
  freeSpeedRpm: number;
  stallTorqueNm: number;
  stallCurrentA: number;
  freeCurrentA: number;
};

export type ActuatorSpec = {
  id: string;
  kind: "velocity_motor" | "position_motor" | "servo";
  jointId?: string | null;
  motor: MotorCurveSpec;
  gearRatio: number;
  efficiency: number;
  rotorInertiaKgM2: number;
  loadInertiaKgM2: number;
  viscousFrictionNmPerRadS?: number;
  coulombFrictionNm?: number;
  currentLimitA: number;
  controllerLatencyMs: number;
  commandRatePerS?: number;
  travelLimit?: [number, number];
  targetRpm?: number;
  kp?: number;
  kd?: number;
};

export type PowerSystemSpec = {
  openCircuitVoltageV: number;
  internalResistanceOhm: number;
  capacityAh: number;
  initialStateOfCharge: number;
  brownoutVoltageV: number;
  maxCurrentA?: number;
  voltageNoiseStdV?: number;
  currentNoiseStdA?: number;
};

export type PiecePathSpec = {
  intakeActuatorId: string;
  conveyorActuatorId: string;
  flywheelActuatorId: string;
  hoodActuatorId?: string | null;
  turretActuatorId?: string | null;
  gateActuatorId: string;
  storageSlots: Transform3[];
  intakePose?: Transform3;
  muzzlePose: Transform3;
  muzzleClearanceIn?: number;
  wheelRadiusIn?: number;
  launchEfficiency?: number;
  compressionIn?: number;
  dragCoefficient?: number;
  magnusCoefficient?: number;
  coefficientVariation?: number;
};

export type MechanismSensorKind = "encoder" | "rpm" | "beam_break" | "joint_position" | "motor_current" | "battery_voltage";

export type MechanismSensorSpec = {
  id: string;
  kind: MechanismSensorKind;
  actuatorId?: string;
  jointId?: string;
  pose?: Transform3;
  sampleRateHz: number;
  quantization?: number;
  noiseStd?: number;
  dropoutRate?: number;
  latencyMs?: number;
};

export type RobotPartTransform = {
  id: string;
  x: number;
  y: number;
  z: number;
  qw?: number;
  qx?: number;
  qy?: number;
  qz?: number;
  visualAsset?: string | null;
};

export type ActuatorTelemetry = {
  rpm?: number;
  currentA?: number;
  command?: number;
  position?: number;
};

export type LaunchPreview = {
  flywheelFrac: number;
  hoodFrac: number;
};

export type IntakeSpec = {
  id: string;
  poseOnRobot?: PoseOnRobot;
  widthIn?: number;
  reachIn?: number;
  heightIn?: number;
  cycleTimeS?: number;
  maxSpeedInPerS?: number;
  canRunWhileMoving?: boolean;
};

export type LauncherSpec = {
  id: string;
  poseOnRobot?: PoseOnRobot;
  aimMode?: "chassis_fixed" | "turret";
  yawRangeDeg?: number[];
  pitchRangeDeg?: number[];
  muzzleSpeedInPerS?: number;
  spinupTimeS?: number;
  cycleTimeS?: number;
  canLaunchWhileMoving?: boolean;
};

export type RobotDesign = {
  chassis?: { lengthIn?: number; widthIn?: number; heightIn?: number; massKg?: number; collisionShape?: string };
  intakes?: IntakeSpec[];
  launchers?: LauncherSpec[];
  visualAsset?: string | null;
  collisionAsset?: string | null;
  visualOffset?: VisualOffset;
  collisionShape?: string;
  rigidParts?: RigidPartSpec[];
  joints?: JointSpec[];
  actuators?: ActuatorSpec[];
  powerSystem?: PowerSystemSpec;
  piecePath?: PiecePathSpec;
  mechanismSensors?: MechanismSensorSpec[];
};

export type GamePieceType = {
  typeId: string;
  displayName?: string;
  shape?: { kind?: string; radius?: number; width?: number; depth?: number };
  color?: string;
  visualAsset?: string | null;
  collisionAsset?: string | null;
};

export type FramePiece = {
  id: string;
  x: number;
  y: number;
  z?: number;
  color?: string;
  heldBy?: string | null;
  storedSlot?: number;
  inFlight?: boolean;
  scored?: boolean;
  typeId?: string;
  radius?: number;
  visualAsset?: string | null;
  headingDeg?: number;
  pitchDeg?: number;
  rollDeg?: number;
  /** MuJoCo/Three Y-up quaternion (w, x, y, z). Preferred when pieces rotate. */
  qw?: number;
  qx?: number;
  qy?: number;
  qz?: number;
};

export type FrameExplain = { id: string; explain: string; points: number };

export type FrameCollision = {
  wall: boolean;
  robot: boolean;
  piece: boolean;
  collisionTimeS?: number;
  firstContactS?: number | null;
  enteredRestricted?: boolean;
};

export type FrameFieldMechanism = {
  id: string;
  angleRad: number;
};

export type FrameRobot = {
  id: string;
  x: number;
  y: number;
  headingDeg: number;
  held: string[];
  dynamic: boolean;
  alliance?: string;
  enteredRestricted?: boolean;
  parts?: RobotPartTransform[];
  actuators?: Record<string, ActuatorTelemetry>;
  batteryVoltageV?: number;
  batteryCurrentA?: number;
  lastVerb?: string;
  mechanismSensors?: Record<string, number>;
};

export type Frame = {
  t: number;
  trueScore: number;
  robots: FrameRobot[];
  robotDesign?: RobotDesign;
  physicalPieces?: boolean;
  mechanismSensors?: Record<string, number>;
  mechanismCommands?: Record<string, number>;
  pieces: FramePiece[];
  gamePieces?: GamePieceType[];
  fieldMechanisms?: FrameFieldMechanism[];
  elements: {
    id: string;
    type?: string;
    alliance?: string;
    pose: { x: number; y: number; headingDeg?: number };
    shape: { kind: string; width?: number; depth?: number; radius?: number };
    tags?: string[];
    isOccluder?: boolean;
  }[];
  aprilTags?: { tagId: number; role?: string; pose: { x: number; y: number; z?: number; headingDeg?: number }; sizeIn?: number }[];
  matchVarsPrivileged: Record<string, string>;
  observedMatchVars: Record<string, string | null>;
  fieldSizeIn: { width: number; depth: number };
  backgroundAsset?: string | null;
  cadManifest?: string | null;
  /** Runtime cache-buster; prefer the CAD source SHA-256 from cad_manifest.field.sha256. */
  cadAssetVersion?: string | null;
  cadSourceSha256?: string | null;
  explains: FrameExplain[];
  stepExplains?: FrameExplain[];
  penalties?: FrameExplain[];
  collision?: FrameCollision;
  queues: Record<string, string[]>;
  gate: Record<string, string>;
  vision: { tagId: number; visible: boolean; occluded: boolean; bearing: number; range: number }[];
};

export async function loadReplayFrames(id: string): Promise<Frame[]> {
  const frames: Frame[] = [];
  let from = 0;
  for (;;) {
    const chunk = await getJson<{ frames: Frame[]; done: boolean }>(`/replays/${id}/chunks?fromStep=${from}&limit=400`);
    frames.push(...chunk.frames);
    from += chunk.frames.length;
    if (chunk.done || chunk.frames.length === 0) break;
  }
  return frames;
}

export type RunRow = {
  id: string;
  state: string;
  metrics: Record<string, unknown>;
  config: Record<string, unknown>;
};
