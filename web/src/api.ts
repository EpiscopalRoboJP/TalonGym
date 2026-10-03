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

export type ToastKind = "success" | "error" | "info";

export function notify(message: string, kind: ToastKind = "success", title?: string) {
  window.dispatchEvent(new CustomEvent("talongym-toast", { detail: { kind, message, title } }));
}

export function statePill(state: string) {
  if (state === "succeeded") return "pill state ok";
  if (state === "failed" || state === "cancelled") return "pill state bad";
  if (state === "running" || state === "queued" || state === "cancelling") return "pill state warn live";
  return "pill state";
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

/** GET that returns null on 404 without emitting a toast. Used for optional draft/recipe routes. */
export async function getJsonOptional<T>(path: string): Promise<T | null> {
  const res = await fetch(`${API}${path}`);
  if (res.status === 404 || res.status === 501) return null;
  if (!res.ok) await fail(res, path);
  return res.json() as Promise<T>;
}

export async function putJsonOptional<T>(path: string, body: unknown): Promise<T | null> {
  const res = await fetch(`${API}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.status === 404 || res.status === 501) return null;
  if (!res.ok) await fail(res, path);
  return res.json() as Promise<T>;
}

export async function postJsonOptional<T>(path: string, body: unknown = {}): Promise<T | null> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (res.status === 404 || res.status === 501) return null;
  if (!res.ok) await fail(res, path);
  const text = await res.text();
  return text ? (JSON.parse(text) as T) : ({} as T);
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

export async function patchJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "PATCH",
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

export async function postForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API}${path}`, { method: "POST", body: form });
  if (!res.ok) await fail(res, path);
  return res.json() as Promise<T>;
}

export async function deleteJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, { method: "DELETE" });
  if (!res.ok) await fail(res, path);
  const text = await res.text();
  return text ? (JSON.parse(text) as T) : ({} as T);
}

export type Health = {
  ok: boolean;
  engine: string;
  fieldEngine?: string;
  db?: string;
  version?: string;
  computeProfile?: string;
  nEnvs?: number;
  credits?: string;
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
  shipped?: boolean;
};

export function presetIsShipped(preset: Pick<PresetMeta, "id" | "shipped">): boolean {
  if (typeof preset.shipped === "boolean") return preset.shipped;
  return (
    preset.id === "mecanum_biobuzz_4cap" ||
    preset.id === "mecanum_meepmeep_defaults" ||
    preset.id.endsWith("_starter")
  );
}

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
  originPreserved?: boolean;
  sha256?: string;
  partId?: string;
  parentId?: string | null;
  jointTransform?: Record<string, number>;
};

export async function uploadRobotModel(presetId: string, file: File): Promise<RobotModelImport> {
  const path = `/presets/robot/${presetId}/model`;
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API}${path}`, { method: "POST", body });
  if (!res.ok) await fail(res, path);
  return res.json() as Promise<RobotModelImport>;
}

export function robotPartModelPath(presetId: string, partId: string): string {
  return `/presets/robot/${presetId}/parts/${partId}/model`;
}

export async function uploadRobotPartModel(
  presetId: string,
  partId: string,
  file: File,
  opts?: { parentId?: string | null; jointTransform?: Record<string, number> },
): Promise<RobotModelImport> {
  const path = robotPartModelPath(presetId, partId);
  const body = new FormData();
  body.append("file", file);
  if (opts?.parentId) body.append("parent_id", opts.parentId);
  body.append("joint_transform_json", JSON.stringify(opts?.jointTransform || {}));
  const res = await fetch(`${API}${path}`, { method: "POST", body });
  if (!res.ok) await fail(res, path);
  return res.json() as Promise<RobotModelImport>;
}

export function catalogSearchPath(opts?: { q?: string; manufacturer?: string; tag?: string }): string {
  const params = new URLSearchParams();
  if (opts?.q) params.set("q", opts.q);
  if (opts?.manufacturer) params.set("manufacturer", opts.manufacturer);
  if (opts?.tag) params.set("tag", opts.tag);
  const query = params.toString();
  return query ? `/catalog?${query}` : "/catalog";
}

export type CatalogCacheState = "ready" | "stale" | "incomplete" | "missing" | "invalid" | "unavailable";

export type CatalogPreviewSource = "cad" | "proxy";

export type CatalogPartPreview = {
  source: CatalogPreviewSource;
  kind: "box" | "sphere" | "cylinder" | "capsule" | "convex_hull";
  sizeIn?: [number, number, number];
  radiusIn?: number;
  lengthIn?: number;
  visualAsset?: string | null;
  thumbnailAsset?: string | null;
  tags?: string[];
};

export type CatalogPartSummary = {
  sku: string;
  manufacturer: string;
  displayName: string;
  productUrl?: string;
  tags: string[];
  massKg: number;
  mounts?: CatalogMount[];
  downloadEnabled: boolean;
  cadFormat?: string;
  preview?: CatalogPartPreview;
  cache: {
    state: CatalogCacheState;
    visualAsset?: string | null;
    collisionAsset?: string | null;
    thumbnailAsset?: string | null;
    sha256?: string;
    generatorVersion?: string;
    reason?: string;
  };
};

export async function listCatalogParts(opts?: { q?: string; manufacturer?: string; tag?: string }): Promise<{ parts: CatalogPartSummary[]; count: number }> {
  return getJson(catalogSearchPath(opts));
}

export type CatalogManufacturer = "gobilda" | "rev";
export type CatalogMountKind =
  | "threaded_hole"
  | "clearance_hole"
  | "shaft"
  | "hub"
  | "bearing"
  | "clamp"
  | "mating_face";
export type CatalogMountStandard =
  | "gobilda_pattern"
  | "gobilda_8mm_rex"
  | "rev_15mm"
  | "rev_m3"
  | "rev_5mm_hex"
  | "rev_maxspline"
  | "adapter_required";

export type CatalogHolePattern = {
  type: "grid" | "linear" | "circle" | "single";
  pitchMm: number;
  countU?: number;
  countV?: number;
  count?: number;
};

export type CatalogMount = {
  id: string;
  kind: CatalogMountKind;
  standard: CatalogMountStandard;
  diameterMm?: number;
  spacingMm?: number;
  axis: [number, number, number];
  uAxis?: [number, number, number];
  depthMm?: number;
  allowedHardware?: string[];
  transform: Transform3;
  pattern?: CatalogHolePattern;
};

export type CatalogCollisionProxy = {
  kind: "box" | "sphere" | "cylinder" | "capsule" | "convex_hull";
  sizeIn?: [number, number, number];
  radiusIn?: number;
  lengthIn?: number;
  pose?: Transform3;
};

export type CatalogPart = CatalogPartSummary & {
  cad?: {
    sourceUrl?: string;
    filename?: string;
    format?: string;
    units?: string;
    origin?: string;
    downloadEnabled?: boolean;
    terms?: string;
  };
  inertiaKgM2?: [number, number, number];
  collision?: CatalogCollisionProxy[];
  mounts?: CatalogMount[];
  visualTransform?: {
    scaleToInches: number;
    rotatedZupToYup: boolean;
    translationIn: [number, number, number];
  };
};

export async function getCatalogPart(sku: string): Promise<CatalogPart> {
  return getJson(`/catalog/parts/${encodeURIComponent(sku)}`);
}

export async function getCatalogCache(): Promise<{
  partCount: number;
  counts: Record<string, number>;
  cadExtra?: boolean;
  cadUnavailableReason?: string | null;
  parts: CatalogPartSummary[];
}> {
  return getJson("/catalog/cache");
}

export async function requestCatalogDownload(sku: string, force = false): Promise<{ jobId: string | null; state: string; accepted: boolean }> {
  const suffix = force ? "?force=true" : "";
  const path = `/catalog/parts/${encodeURIComponent(sku)}/download${suffix}`;
  const res = await fetch(`${API}${path}`, { method: "POST" });
  if (!res.ok) await fail(res, path);
  return res.json();
}

export async function getCatalogJob(jobId: string): Promise<{ id: string; state: string; sku?: string; error?: { code: string; message: string } }> {
  return getJson(`/catalog/jobs/${encodeURIComponent(jobId)}`);
}

export type CacheBatchItemState = "queued" | "converting" | "ready" | "failed" | "skipped" | "cancelled";

export type CacheBatchCounts = Record<CacheBatchItemState, number>;

export type CacheBatchItem = {
  sku: string;
  manufacturer?: string;
  displayName?: string;
  state: CacheBatchItemState | string;
  reason?: string;
  error?: { code?: string; message?: string };
  reused?: boolean;
};

export type CatalogCacheBatch = {
  id: string | null;
  state: "idle" | "queued" | "running" | "cancelling" | "cancelled" | "done" | string;
  force?: boolean;
  manufacturer?: string | null;
  counts: CacheBatchCounts;
  total: number;
  concurrency?: number;
  items: CacheBatchItem[];
};

export function catalogCacheAllPath(batchId?: string): string {
  return batchId ? `/catalog/cache/all/${encodeURIComponent(batchId)}` : "/catalog/cache/all";
}

export function catalogCacheAllCancelPath(batchId: string): string {
  return `${catalogCacheAllPath(batchId)}/cancel`;
}

export function catalogCacheAllRetryPath(batchId: string): string {
  return `${catalogCacheAllPath(batchId)}/retry`;
}

export async function requestCatalogCacheAll(opts?: {
  force?: boolean;
  manufacturer?: string;
  skus?: string[];
}): Promise<CatalogCacheBatch> {
  return postJson(catalogCacheAllPath(), {
    force: Boolean(opts?.force),
    manufacturer: opts?.manufacturer || undefined,
    skus: opts?.skus,
  });
}

export function emptyCatalogCacheBatch(): CatalogCacheBatch {
  return {
    id: null,
    state: "idle",
    counts: { queued: 0, converting: 0, ready: 0, failed: 0, skipped: 0, cancelled: 0 },
    total: 0,
    items: [],
  };
}

export async function getCatalogCacheAll(batchId?: string): Promise<CatalogCacheBatch> {
  if (!batchId) {
    return (await getJsonOptional<CatalogCacheBatch>(catalogCacheAllPath())) || emptyCatalogCacheBatch();
  }
  return getJson(catalogCacheAllPath(batchId));
}

export async function cancelCatalogCacheAll(batchId: string): Promise<CatalogCacheBatch> {
  return postJson(catalogCacheAllCancelPath(batchId), {});
}

export async function retryCatalogCacheAll(batchId: string): Promise<CatalogCacheBatch> {
  return postJson(catalogCacheAllRetryPath(batchId), {});
}

export const ASSEMBLY_SCHEMA_VERSION = "1.2.0";
export const PHYSICAL_SCHEMA_VERSION = "1.1.0";

export type MountRef = {
  instanceId: string;
  mountId: string;
  patternIndex?: [number, number];
};

export type AssemblyInstance = {
  id: string;
  sku: string;
  pose?: Transform3;
};

export type AssemblyConnection = {
  id: string;
  parent: MountRef;
  child: MountRef;
  spinDeg?: number;
  jointType?: "fixed" | "hinge" | "slide";
  secondary?: { parent: MountRef; child: MountRef };
};

export type DrivebaseRecipeParameters = {
  lengthSku?: string;
  widthSku?: string;
  motorSku?: string;
  wheelSku?: string;
  wheelSkuOpposite?: string;
  cartridgeSku?: string;
  includeElectronics?: boolean;
};

export type AssemblyRecipeOrigin = {
  id: string;
  parameters?: DrivebaseRecipeParameters;
};

export type RobotAssembly = {
  rootInstanceId?: string;
  recipe?: AssemblyRecipeOrigin;
  instances: AssemblyInstance[];
  connections: AssemblyConnection[];
};

export type FunctionalBindings = {
  confirmed?: boolean;
  drivetrain?: { type: string; trackWidthIn: number; wheelDiameterIn?: number; wheelbaseIn?: number };
  instanceRoles?: Record<string, string>;
};

export type AssemblyWarning = {
  code: string;
  severity?: "info" | "warning";
  message: string;
  instanceId?: string;
};

export type DrivebaseRecipe = {
  id: string;
  displayName: string;
  manufacturer: CatalogManufacturer;
  drivetrain: "mecanum" | "tank";
  description?: string;
  defaultParameters: DrivebaseRecipeParameters;
  lengthSkus: string[];
  widthSkus: string[];
  motorSkus: string[];
  wheelSkus: string[];
  cartridgeSkus?: string[];
};

export type RecipeInstantiateResult = {
  assembly: RobotAssembly;
  functionalBindings?: FunctionalBindings;
  warnings?: AssemblyWarning[];
  document?: RobotPreset;
};

export function catalogRecipesPath(): string {
  return "/catalog/recipes";
}

export function catalogRecipeInstantiatePath(recipeId: string): string {
  return `/catalog/recipes/${encodeURIComponent(recipeId)}/instantiate`;
}

export function catalogAssemblyCompilePath(): string {
  return "/catalog/assemblies/compile";
}

export function robotDraftPath(presetId: string): string {
  return `/presets/robot/${encodeURIComponent(presetId)}/draft`;
}

export async function listDrivebaseRecipes(): Promise<{ recipes: DrivebaseRecipe[] }> {
  return getJson(catalogRecipesPath());
}

export async function instantiateDrivebaseRecipe(
  recipeId: string,
  parameters: DrivebaseRecipeParameters,
): Promise<RecipeInstantiateResult> {
  return postJson(catalogRecipeInstantiatePath(recipeId), parameters);
}

export async function loadRobotDraft<T>(presetId: string): Promise<T | null> {
  return getJsonOptional<T>(robotDraftPath(presetId));
}

export async function saveRobotDraft<T>(presetId: string, document: unknown): Promise<T | null> {
  return putJsonOptional<T>(robotDraftPath(presetId), document);
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
  cacheState?: CatalogCacheState;
  cacheReason?: string | null;
  tags?: string[];
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

export type CameraSensorSpec = {
  id: string;
  kind: string;
  fovDeg?: number;
  rangeIn?: number;
  poseOnRobot?: PoseOnRobot;
};

export type RobotDesign = {
  chassis?: { lengthIn?: number; widthIn?: number; heightIn?: number; massKg?: number; collisionShape?: string };
  intakes?: IntakeSpec[];
  launchers?: LauncherSpec[];
  visualAsset?: string | null;
  collisionAsset?: string | null;
  visualOffset?: VisualOffset;
  collisionShape?: string;
  sensors?: CameraSensorSpec[];
  rigidParts?: RigidPartSpec[];
  joints?: JointSpec[];
  actuators?: ActuatorSpec[];
  powerSystem?: PowerSystemSpec;
  piecePath?: PiecePathSpec;
  mechanismSensors?: MechanismSensorSpec[];
};

export type RobotPreset = RobotDesign & {
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
  sensors: CameraSensorSpec[];
  defaultActionTier?: ActionTier | string;
  policyInterfaceVersion?: string;
  assembly?: RobotAssembly;
  functionalBindings?: FunctionalBindings;
  warnings?: AssemblyWarning[];
  [k: string]: unknown;
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
  phase?: string;
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
  name?: string | null;
  hasCheckpoint?: boolean;
};

export function runLabel(row: Pick<RunRow, "id" | "name" | "config">): string {
  const named = typeof row.name === "string" && row.name.trim() ? row.name.trim() : "";
  if (named) return named;
  const fromConfig = row.config && typeof row.config.name === "string" ? row.config.name.trim() : "";
  return fromConfig || row.id;
}

const REPLAY_SOURCE_LABEL: Record<string, string> = {
  demo: "Scripted demo",
  train: "Training run",
  eval: "Evaluation",
};

export type ReplayListRow = {
  id: string;
  name?: string | null;
  runId?: string | null;
  source?: string | null;
  algo?: string | null;
  trueScore?: number;
};

/** Prefer the training run's name, then its run id, then the replay source. */
export function replayLabel(row: Pick<ReplayListRow, "id" | "name" | "runId" | "source">): string {
  const named = typeof row.name === "string" && row.name.trim() ? row.name.trim() : "";
  if (named) return named;
  if (row.runId) return row.runId;
  return REPLAY_SOURCE_LABEL[row.source || ""] || row.source || row.id;
}

export function replayMetaLine(row: ReplayListRow): string {
  const title = replayLabel(row);
  const bits: string[] = [];
  if (row.runId && title !== row.runId) bits.push(row.runId);
  else if (title !== row.id) bits.push(row.id);
  if (row.source === "eval") bits.push("evaluation");
  if (row.algo) bits.push(row.algo);
  return bits.join(" · ");
}
