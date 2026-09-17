import type {
  Frame,
  GamePieceType,
  MatchRobotSetup,
  RobotDesign,
  RobotPreset,
  StartSlot,
} from "../api";

export type SetupFieldDoc = {
  id?: string;
  displayName?: string;
  fieldSizeIn?: { width: number; depth: number };
  backgroundAsset?: string | null;
  cadManifest?: string | null;
  startSlots?: StartSlot[];
  elements?: {
    id: string;
    type?: string;
    alliance?: string;
    pose?: { x?: number; y?: number; headingDeg?: number };
    shape?: { kind?: string; width?: number; depth?: number; radius?: number };
    tags?: string[];
    isOccluder?: boolean;
  }[];
  gamePieces?: (GamePieceType & { attributes?: { color?: string } })[];
  spawns?: { id: string; pieceTypeId: string; poses: { x: number; y: number; z?: number }[] }[];
  aprilTags?: Frame["aprilTags"];
};

export function robotDesignFromPreset(robot: RobotPreset): RobotDesign {
  return {
    chassis: robot.chassis,
    intakes: robot.intakes,
    launchers: robot.launchers,
    visualAsset: robot.visualAsset,
    collisionAsset: robot.collisionAsset,
    visualOffset: robot.visualOffset,
    collisionShape: robot.collisionShape || robot.chassis?.collisionShape,
    sensors: robot.sensors,
    rigidParts: robot.rigidParts,
    joints: robot.joints,
    actuators: robot.actuators,
    powerSystem: robot.powerSystem,
    piecePath: robot.piecePath,
    mechanismSensors: robot.mechanismSensors,
  };
}

function spawnPieces(field: SetupFieldDoc): Frame["pieces"] {
  const types = new Map((field.gamePieces || []).map((gp) => [gp.typeId, gp]));
  return (field.spawns || []).flatMap((spawn) => {
    const gp = types.get(spawn.pieceTypeId);
    const sh = gp?.shape || {};
    const radius = sh.kind === "circle" ? sh.radius || 2.5 : Math.max(sh.width || 3, sh.depth || 3) / 2;
    return spawn.poses.map((pose, i) => ({
      id: `${spawn.id}-${i}`,
      typeId: spawn.pieceTypeId,
      x: pose.x,
      y: pose.y,
      z: pose.z ?? radius,
      radius,
      color: gp?.attributes?.color || gp?.color,
    }));
  });
}

function placedRobots(
  slots: StartSlot[],
  setup: MatchRobotSetup[] | undefined,
  customStarts: boolean,
): Frame["robots"] {
  const byId = new Map(slots.map((slot) => [slot.id, slot]));
  const rows = customStarts && setup?.length ? setup.filter((row) => row.enabled) : [{ id: "red_0", enabled: true, dynamic: true, startSlotId: "red_0", offset: { x: 0, y: 0, headingDeg: 0 } }];
  return rows.map((row) => {
    const slot = byId.get(row.startSlotId) || byId.get(row.id) || slots.find((candidate) => candidate.alliance === (row.id.startsWith("blue") ? "blue" : "red"));
    const pose = slot?.pose || { x: 0, y: 0, headingDeg: 0 };
    return {
      id: row.id,
      alliance: row.id.startsWith("blue") ? "blue" : "red",
      x: pose.x + (row.offset?.x || 0),
      y: pose.y + (row.offset?.y || 0),
      headingDeg: (pose.headingDeg || 0) + (row.offset?.headingDeg || 0),
      held: [],
      dynamic: Boolean(row.dynamic ?? row.id === "red_0"),
    };
  });
}

export function buildSetupPreview(
  field: SetupFieldDoc | null | undefined,
  robot: RobotPreset | null | undefined,
  options?: { robotSetup?: MatchRobotSetup[]; customStarts?: boolean },
): Frame | null {
  if (!field) return null;
  const size = field.fieldSizeIn || { width: 144, depth: 144 };
  return {
    t: 0,
    trueScore: 0,
    robots: placedRobots(field.startSlots || [], options?.robotSetup, Boolean(options?.customStarts)),
    robotDesign: robot ? robotDesignFromPreset(robot) : undefined,
    pieces: spawnPieces(field),
    gamePieces: (field.gamePieces || []).map((gp) => ({
      typeId: gp.typeId,
      displayName: gp.displayName,
      shape: gp.shape,
      color: gp.color || gp.attributes?.color,
      visualAsset: gp.visualAsset,
      collisionAsset: gp.collisionAsset,
    })),
    elements: (field.elements || []).map((el) => ({
      id: el.id,
      type: el.type,
      alliance: el.alliance,
      pose: { x: el.pose?.x ?? 0, y: el.pose?.y ?? 0, headingDeg: el.pose?.headingDeg },
      shape: { kind: el.shape?.kind || "aabb", width: el.shape?.width, depth: el.shape?.depth, radius: el.shape?.radius },
      tags: el.tags,
      isOccluder: el.isOccluder,
    })),
    aprilTags: field.aprilTags || [],
    matchVarsPrivileged: {},
    observedMatchVars: {},
    fieldSizeIn: { width: size.width, depth: size.depth },
    backgroundAsset: typeof field.backgroundAsset === "string" ? field.backgroundAsset : null,
    cadManifest: typeof field.cadManifest === "string" ? field.cadManifest : null,
    explains: [],
    queues: {},
    gate: {},
    vision: [],
  };
}

export function runMatchesSceneSelection(
  run: { fieldId?: string; robotId?: string } | null | undefined,
  fieldId: string,
  robotId: string,
): boolean {
  if (!run?.fieldId && !run?.robotId) return true;
  if (run.fieldId && run.fieldId !== fieldId) return false;
  if (run.robotId && run.robotId !== robotId) return false;
  return true;
}

export function trainSceneFrame(options: {
  liveFrame: Frame | null;
  previewFrame: Frame | null;
  runPresets?: { fieldId?: string; robotId?: string } | null;
  fieldId: string;
  robotId: string;
}): Frame | null {
  if (options.liveFrame && runMatchesSceneSelection(options.runPresets, options.fieldId, options.robotId)) {
    return options.liveFrame;
  }
  return options.previewFrame;
}
