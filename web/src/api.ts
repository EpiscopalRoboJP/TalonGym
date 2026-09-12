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

export type Health = { ok: boolean; engine: string; db?: string; version?: string; computeProfile?: string; nEnvs?: number };

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

export type PresetMeta = {
  id: string;
  displayName: string;
  season?: string;
  manualRevision?: string;
  verifyAgainstManual?: boolean;
  stale?: boolean;
  computeProfile?: string;
};

export type PoseOnRobot = {
  x?: number;
  y?: number;
  z?: number;
  headingDeg?: number;
  pitchDeg?: number;
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
  chassis?: { lengthIn?: number; widthIn?: number; heightIn?: number; massKg?: number };
  intakes?: IntakeSpec[];
  launchers?: LauncherSpec[];
};

export type Frame = {
  t: number;
  trueScore: number;
  robots: { id: string; x: number; y: number; headingDeg: number; held: string[]; dynamic: boolean; alliance?: string }[];
  robotDesign?: RobotDesign;
  pieces: { id: string; x: number; y: number; z?: number; color?: string; heldBy?: string | null; inFlight?: boolean; scored?: boolean }[];
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
  explains: { id: string; explain: string; points: number }[];
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
