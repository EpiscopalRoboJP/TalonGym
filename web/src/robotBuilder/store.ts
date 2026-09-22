import type {
  FunctionalBindings,
  RobotAssembly,
  RobotPreset,
  Transform3,
} from "../api";
import { childSubtree, emptyAssembly } from "./assemblyMath";
import { uniqueConnectionId, uniqueInstanceId } from "./recipes";
import { snapConnection, type SnapCandidate } from "./snap";

export type BuilderSel =
  | { kind: "chassis" }
  | { kind: "camera" }
  | { kind: "model" }
  | { kind: "intake"; id: string }
  | { kind: "launcher"; id: string }
  | { kind: "part"; id: string }
  | { kind: "joint"; id: string }
  | { kind: "actuator"; id: string }
  | { kind: "mechSensor"; id: string }
  | { kind: "instance"; id: string };

export type DragState = {
  sku: string;
  sourceInstanceId?: string;
  spinIndex: number;
  pointer: [number, number, number];
  candidate: SnapCandidate | null;
};

export type BuilderSnapshot = {
  doc: RobotPreset;
};

export type BuilderState = {
  doc: RobotPreset | null;
  sel: BuilderSel;
  dirty: boolean;
  canUndo: boolean;
  canRedo: boolean;
  drag: DragState | null;
  inferenceOpen: boolean;
};

type History = {
  past: BuilderSnapshot[];
  future: BuilderSnapshot[];
};

const MAX_HISTORY = 80;

export function sameSel(a: BuilderSel, b: BuilderSel): boolean {
  if (a.kind !== b.kind) return false;
  if ("id" in a && "id" in b) return a.id === b.id;
  return true;
}

export function cloneDoc(doc: RobotPreset): RobotPreset {
  return structuredClone(doc);
}

export function ensureAssembly(doc: RobotPreset): RobotAssembly {
  return doc.assembly || emptyAssembly();
}

function withAssembly(doc: RobotPreset, assembly: RobotAssembly): RobotPreset {
  return { ...doc, schemaVersion: "1.2.0", assembly };
}

export type BuilderAction =
  | { type: "hydrate"; doc: RobotPreset }
  | { type: "select"; sel: BuilderSel }
  | { type: "patchDoc"; doc: RobotPreset; record?: boolean }
  | { type: "setAssembly"; assembly: RobotAssembly }
  | { type: "snapPlace"; instanceId: string; sku: string; candidate: SnapCandidate }
  | { type: "placeRoot"; instanceId: string; sku: string; pose: Transform3 }
  | { type: "detach"; instanceId: string; pose: Transform3 }
  | { type: "replace"; instanceId: string; sku: string }
  | { type: "duplicate"; instanceId: string }
  | { type: "mirror"; instanceId: string }
  | { type: "pattern"; instanceId: string; count: number }
  | { type: "startDrag"; drag: DragState }
  | { type: "updateDrag"; pointer: [number, number, number]; candidate: SnapCandidate | null; spinIndex?: number }
  | { type: "cancelDrag" }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "markSaved"; doc: RobotPreset }
  | { type: "setInferenceOpen"; open: boolean }
  | { type: "confirmInference"; bindings: FunctionalBindings };

function snapshotOf(doc: RobotPreset | null): BuilderSnapshot | null {
  return doc ? { doc: cloneDoc(doc) } : null;
}

export function createBuilderState(): BuilderState {
  return {
    doc: null,
    sel: { kind: "chassis" },
    dirty: false,
    canUndo: false,
    canRedo: false,
    drag: null,
    inferenceOpen: false,
  };
}

export function reduceBuilder(state: BuilderState, history: History, action: BuilderAction): { state: BuilderState; history: History } {
  const push = (_doc: RobotPreset) => {
    const snap = snapshotOf(state.doc);
    const past = snap ? [...history.past, snap].slice(-MAX_HISTORY) : history.past;
    return { past, future: [] as BuilderSnapshot[] };
  };

  if (action.type === "hydrate") {
    return {
      state: { ...createBuilderState(), doc: action.doc },
      history: { past: [], future: [] },
    };
  }
  if (action.type === "select") {
    return { state: { ...state, sel: action.sel }, history };
  }
  if (action.type === "startDrag") {
    return { state: { ...state, drag: action.drag }, history };
  }
  if (action.type === "updateDrag") {
    if (!state.drag) return { state, history };
    return {
      state: {
        ...state,
        drag: {
          ...state.drag,
          pointer: action.pointer,
          candidate: action.candidate,
          spinIndex: action.spinIndex ?? state.drag.spinIndex,
        },
      },
      history,
    };
  }
  if (action.type === "cancelDrag") {
    return { state: { ...state, drag: null }, history };
  }
  if (action.type === "setInferenceOpen") {
    return { state: { ...state, inferenceOpen: action.open }, history };
  }
  if (action.type === "undo") {
    const prev = history.past[history.past.length - 1];
    if (!prev || !state.doc) return { state, history };
    return {
      state: { ...state, doc: prev.doc, dirty: true, canUndo: history.past.length > 1, canRedo: true, drag: null },
      history: { past: history.past.slice(0, -1), future: [{ doc: cloneDoc(state.doc) }, ...history.future] },
    };
  }
  if (action.type === "redo") {
    const next = history.future[0];
    if (!next || !state.doc) return { state, history };
    return {
      state: { ...state, doc: next.doc, dirty: true, canUndo: true, canRedo: history.future.length > 1, drag: null },
      history: { past: [...history.past, { doc: cloneDoc(state.doc) }], future: history.future.slice(1) },
    };
  }
  if (action.type === "markSaved") {
    if (!state.doc || JSON.stringify(state.doc) !== JSON.stringify(action.doc)) return { state, history };
    return { state: { ...state, dirty: false }, history };
  }
  if (!state.doc) return { state, history };

  if (action.type === "patchDoc") {
    const nextHistory = action.record === false ? history : push(state.doc);
    return {
      state: {
        ...state,
        doc: action.doc,
        dirty: true,
        canUndo: nextHistory.past.length > 0,
        canRedo: false,
      },
      history: nextHistory,
    };
  }
  if (action.type === "setAssembly") {
    const nextHistory = push(state.doc);
    return {
      state: {
        ...state,
        doc: { ...withAssembly(state.doc, action.assembly), functionalBindings: state.doc.functionalBindings ? { ...state.doc.functionalBindings, confirmed: false } : undefined },
        dirty: true,
        canUndo: true,
        canRedo: false,
        drag: null,
      },
      history: nextHistory,
    };
  }
  if (action.type === "snapPlace") {
    const assembly = ensureAssembly(state.doc);
    const existing = assembly.instances.find((row) => row.id === action.instanceId);
    const instances = existing
      ? assembly.instances.map((row) => row.id === action.instanceId ? { id: row.id, sku: action.sku } : row)
      : [...assembly.instances, { id: action.instanceId, sku: action.sku }];
    const connections = [
      ...assembly.connections.filter((row) => row.child.instanceId !== action.instanceId),
      snapConnection(action.candidate, action.instanceId, uniqueConnectionId("snap", assembly)),
    ];
    const nextHistory = push(state.doc);
    return {
      state: {
        ...state,
        doc: { ...withAssembly(state.doc, {
          ...assembly,
          rootInstanceId: assembly.rootInstanceId || assembly.instances[0]?.id || action.instanceId,
          instances,
          connections,
        }), functionalBindings: state.doc.functionalBindings ? { ...state.doc.functionalBindings, confirmed: false } : undefined },
        sel: { kind: "instance", id: action.instanceId },
        dirty: true,
        canUndo: true,
        canRedo: false,
        drag: null,
      },
      history: nextHistory,
    };
  }
  if (action.type === "placeRoot") {
    const assembly = ensureAssembly(state.doc);
    const rootId = assembly.rootInstanceId || action.instanceId;
    const has = assembly.instances.some((row) => row.id === action.instanceId);
    const instances = has
      ? assembly.instances.map((row) => (row.id === action.instanceId ? { ...row, sku: action.sku, pose: action.pose } : row))
      : [...assembly.instances, { id: action.instanceId, sku: action.sku, pose: action.pose }];
    const nextHistory = push(state.doc);
    return {
      state: {
        ...state,
        doc: { ...withAssembly(state.doc, {
          ...assembly,
          rootInstanceId: rootId,
          instances: instances.map((row) => (row.id === rootId ? { ...row, pose: action.pose } : row)),
          connections: assembly.connections,
        }), functionalBindings: state.doc.functionalBindings ? { ...state.doc.functionalBindings, confirmed: false } : undefined },
        sel: { kind: "instance", id: action.instanceId },
        dirty: true,
        canUndo: true,
        canRedo: false,
        drag: null,
      },
      history: nextHistory,
    };
  }
  if (action.type === "detach") {
    const assembly = ensureAssembly(state.doc);
    if (action.instanceId === assembly.rootInstanceId) return { state, history };
    const nextHistory = push(state.doc);
    return {
      state: {
        ...state,
        doc: { ...withAssembly(state.doc, {
          ...assembly,
          instances: assembly.instances.map((row) => row.id === action.instanceId ? { ...row, pose: action.pose } : row),
          connections: assembly.connections.filter((row) => row.child.instanceId !== action.instanceId),
        }), functionalBindings: state.doc.functionalBindings ? { ...state.doc.functionalBindings, confirmed: false } : undefined },
        dirty: true,
        canUndo: true,
        canRedo: false,
      },
      history: nextHistory,
    };
  }
  if (action.type === "replace") {
    const assembly = ensureAssembly(state.doc);
    const nextHistory = push(state.doc);
    return {
      state: {
        ...state,
        doc: { ...withAssembly(state.doc, {
          ...assembly,
          instances: assembly.instances.map((row) => (row.id === action.instanceId ? { ...row, sku: action.sku } : row)),
        }), functionalBindings: state.doc.functionalBindings ? { ...state.doc.functionalBindings, confirmed: false } : undefined },
        dirty: true,
        canUndo: true,
        canRedo: false,
      },
      history: nextHistory,
    };
  }
  if (action.type === "duplicate" || action.type === "mirror" || action.type === "pattern") {
    const assembly = ensureAssembly(state.doc);
    const ids = childSubtree(assembly, action.instanceId);
    const copies = action.type === "pattern" ? Math.max(1, action.count) : 1;
    let instances = [...assembly.instances];
    let connections = [...assembly.connections];
    let lastId = action.instanceId;
    for (let n = 0; n < copies; n += 1) {
      const remap = new Map<string, string>();
      for (const ident of ids) {
        const src = assembly.instances.find((row) => row.id === ident);
        if (!src) continue;
        const nid = uniqueInstanceId(ident.replace(/_\d+$/, "") || "part", { ...assembly, instances, connections });
        remap.set(ident, nid);
        instances = [...instances, { id: nid, sku: src.sku }];
      }
      for (const connection of assembly.connections) {
        const parent = remap.get(connection.parent.instanceId);
        const child = remap.get(connection.child.instanceId);
        if (!parent && !child) continue;
        if (!parent || !child) continue;
        const mirrored = action.type === "mirror";
        connections = [
          ...connections,
          {
            ...connection,
            id: uniqueConnectionId(connection.id.replace(/_\d+$/, "") || "conn", { ...assembly, instances, connections }),
            parent: {
              ...connection.parent,
              instanceId: parent,
              patternIndex: connection.parent.patternIndex
                ? ([mirrored ? connection.parent.patternIndex[0] : connection.parent.patternIndex[0], connection.parent.patternIndex[1]] as [number, number])
                : connection.parent.patternIndex,
            },
            child: { ...connection.child, instanceId: child },
            spinDeg: (connection.spinDeg || 0) + (mirrored ? 180 : 0),
          },
        ];
      }
      const rootCopy = remap.get(action.instanceId);
      const parentConn = assembly.connections.find((row) => row.child.instanceId === action.instanceId);
      if (rootCopy && parentConn) {
        const pattern = parentConn.parent.patternIndex;
        connections = [
          ...connections,
          {
            ...parentConn,
            id: uniqueConnectionId("copy", { ...assembly, instances, connections }),
            parent: {
              ...parentConn.parent,
              patternIndex: pattern ? [pattern[0] + (n + 1) * (action.type === "mirror" ? 0 : 2), pattern[1]] : pattern,
            },
            child: { ...parentConn.child, instanceId: rootCopy },
            spinDeg: (parentConn.spinDeg || 0) + (action.type === "mirror" ? 180 : 0),
          },
        ];
      }
      lastId = remap.get(action.instanceId) || lastId;
    }
    const nextHistory = push(state.doc);
    return {
      state: {
        ...state,
        doc: { ...withAssembly(state.doc, { ...assembly, instances, connections }), functionalBindings: state.doc.functionalBindings ? { ...state.doc.functionalBindings, confirmed: false } : undefined },
        sel: { kind: "instance", id: lastId },
        dirty: true,
        canUndo: true,
        canRedo: false,
      },
      history: nextHistory,
    };
  }
  if (action.type === "confirmInference") {
    const nextHistory = push(state.doc);
    return {
      state: {
        ...state,
        doc: { ...state.doc, functionalBindings: action.bindings },
        inferenceOpen: false,
        dirty: true,
        canUndo: true,
        canRedo: false,
      },
      history: nextHistory,
    };
  }
  return { state, history };
}

export function applyInstancePose(doc: RobotPreset, instanceId: string, pose: Transform3): RobotPreset {
  const assembly = ensureAssembly(doc);
  const freeMounted =
    instanceId === assembly.rootInstanceId ||
    !assembly.connections.some((connection) => connection.child.instanceId === instanceId);
  return {
    ...withAssembly(doc, {
      ...assembly,
      instances: assembly.instances.map((row) => (row.id === instanceId && freeMounted ? { ...row, pose } : row)),
    }),
    functionalBindings: doc.functionalBindings ? { ...doc.functionalBindings, confirmed: false } : undefined,
  };
}

export function patchList<T extends { id: string }>(rows: T[] | undefined, id: string, partial: Partial<T>): T[] {
  return (rows || []).map((row) => (row.id === id ? { ...row, ...partial } : row));
}
