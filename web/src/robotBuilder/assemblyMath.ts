import type { AssemblyConnection, CatalogMount, CatalogPart, MountRef, RobotAssembly, Transform3 } from "../api";
import { holeInPart, mountAxis, mountsCompatible, occupancyError, partMount } from "./mounts";
import {
  add,
  axisAngle,
  identity4,
  matrixFromPose,
  normalize,
  norm,
  PATTERN_MATCH_TOL_IN,
  poseFromMatrix,
  rotationAligning,
  scale,
  signedAngleAround,
  sub,
  transformDirection,
  transformPoint,
  type Vec3,
} from "./transforms";

export class AssemblyGraphError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AssemblyGraphError";
  }
}

export function emptyAssembly(): RobotAssembly {
  return { instances: [], connections: [] };
}

export function indexInstances(assembly: RobotAssembly): Map<string, RobotAssembly["instances"][number]> {
  const rows = new Map<string, RobotAssembly["instances"][number]>();
  for (const row of assembly.instances) {
    if (!row.id) throw new AssemblyGraphError("assembly instance is missing id");
    if (rows.has(row.id)) throw new AssemblyGraphError(`duplicate assembly instance id ${row.id}`);
    if (row.id === "chassis") throw new AssemblyGraphError("instance id chassis is reserved for the compiled physical root");
    rows.set(row.id, row);
  }
  if (!rows.size) throw new AssemblyGraphError("assembly must contain at least one instance");
  return rows;
}

export function rootInstanceId(assembly: RobotAssembly, instances: Map<string, RobotAssembly["instances"][number]>): string {
  const wanted = assembly.rootInstanceId || "";
  if (wanted) {
    if (!instances.has(wanted)) throw new AssemblyGraphError(`rootInstanceId ${wanted} is not an assembly instance`);
    return wanted;
  }
  if (instances.size === 1) return [...instances.keys()][0];
  throw new AssemblyGraphError("assembly with multiple instances requires rootInstanceId");
}

export function parentMap(assembly: RobotAssembly, instances: Map<string, RobotAssembly["instances"][number]>): Map<string, string> {
  const parents = new Map<string, string>();
  for (const connection of assembly.connections) {
    const parentId = connection.parent.instanceId;
    const childId = connection.child.instanceId;
    if (!instances.has(parentId)) throw new AssemblyGraphError(`connection ${connection.id} parent instance ${parentId} is missing`);
    if (!instances.has(childId)) throw new AssemblyGraphError(`connection ${connection.id} child instance ${childId} is missing`);
    if (parentId === childId) throw new AssemblyGraphError(`connection ${connection.id} cannot parent an instance to itself`);
    if (parents.has(childId)) throw new AssemblyGraphError(`instance ${childId} has multiple parents`);
    parents.set(childId, parentId);
  }
  return parents;
}

export type ValidatedGraph = {
  rootId: string;
  order: string[];
  byChild: Map<string, AssemblyConnection>;
};

export function validateAssemblyGraph(assembly: RobotAssembly): ValidatedGraph {
  const instances = indexInstances(assembly);
  const rootId = rootInstanceId(assembly, instances);
  const parents = parentMap(assembly, instances);
  if (parents.has(rootId)) throw new AssemblyGraphError(`root instance ${rootId} cannot be a connection child`);
  const byChild = new Map<string, AssemblyConnection>();
  for (const connection of assembly.connections) byChild.set(connection.child.instanceId, connection);
  const visiting = new Set<string>();
  const ordered: string[] = [];
  const visit = (ident: string) => {
    if (ordered.includes(ident)) return;
    if (visiting.has(ident)) throw new AssemblyGraphError(`assembly connection cycle at ${ident}`);
    visiting.add(ident);
    const parent = parents.get(ident);
    if (parent) visit(parent);
    visiting.delete(ident);
    ordered.push(ident);
  };
  for (const ident of instances.keys()) visit(ident);
  const disconnected = [...instances.keys()].filter((ident) => ident !== rootId && !parents.has(ident)).sort();
  if (disconnected.length) throw new AssemblyGraphError(`disconnected assembly instances: ${disconnected.join(", ")}`);
  return { rootId, order: ordered, byChild };
}

function patternIndex(ref: MountRef): [number, number] | null {
  return ref.patternIndex ? [ref.patternIndex[0], ref.patternIndex[1]] : null;
}

function projectOntoPlane(vector: Vec3, origin: Vec3, axis: Vec3): Vec3 {
  const offset = sub(vector, origin);
  const normal = normalize(axis);
  return sub(offset, scale(normal, offset[0] * normal[0] + offset[1] * normal[1] + offset[2] * normal[2]));
}

export type SecondaryConstraint = {
  parentMount: CatalogMount;
  childMount: CatalogMount;
  parentIndex?: [number, number] | null;
  childIndex?: [number, number] | null;
};

export function solveMountTransform(
  parentWorld: number[][],
  parentMount: CatalogMount,
  childMount: CatalogMount,
  opts?: {
    parentIndex?: [number, number] | null;
    childIndex?: [number, number] | null;
    spinDeg?: number;
    secondary?: SecondaryConstraint;
  },
): number[][] {
  const parentHole = transformPoint(parentWorld, holeInPart(parentMount, opts?.parentIndex));
  const parentAxis = transformDirection(parentWorld, mountAxis(parentMount));
  const childHole = holeInPart(childMount, opts?.childIndex);
  const childAxis = mountAxis(childMount);
  const desiredAxis = scale(parentAxis, -1);
  let rotation = rotationAligning(childAxis, desiredAxis);
  let spinDeg = opts?.spinDeg || 0;
  if (opts?.secondary) {
    spinDeg = spinFromSecondary(parentWorld, parentHole, desiredAxis, rotation, childHole, childMount, opts.secondary);
  }
  if (Math.abs(spinDeg) > 1e-12) {
    rotation = (() => {
      const extra = axisAngle(desiredAxis, (spinDeg * Math.PI) / 180);
      return [
        [
          extra[0][0] * rotation[0][0] + extra[0][1] * rotation[1][0] + extra[0][2] * rotation[2][0],
          extra[0][0] * rotation[0][1] + extra[0][1] * rotation[1][1] + extra[0][2] * rotation[2][1],
          extra[0][0] * rotation[0][2] + extra[0][1] * rotation[1][2] + extra[0][2] * rotation[2][2],
        ],
        [
          extra[1][0] * rotation[0][0] + extra[1][1] * rotation[1][0] + extra[1][2] * rotation[2][0],
          extra[1][0] * rotation[0][1] + extra[1][1] * rotation[1][1] + extra[1][2] * rotation[2][1],
          extra[1][0] * rotation[0][2] + extra[1][1] * rotation[1][2] + extra[1][2] * rotation[2][2],
        ],
        [
          extra[2][0] * rotation[0][0] + extra[2][1] * rotation[1][0] + extra[2][2] * rotation[2][0],
          extra[2][0] * rotation[0][1] + extra[2][1] * rotation[1][1] + extra[2][2] * rotation[2][1],
          extra[2][0] * rotation[0][2] + extra[2][1] * rotation[1][2] + extra[2][2] * rotation[2][2],
        ],
      ];
    })();
  }
  const translation = sub(parentHole, [
    rotation[0][0] * childHole[0] + rotation[0][1] * childHole[1] + rotation[0][2] * childHole[2],
    rotation[1][0] * childHole[0] + rotation[1][1] * childHole[1] + rotation[1][2] * childHole[2],
    rotation[2][0] * childHole[0] + rotation[2][1] * childHole[1] + rotation[2][2] * childHole[2],
  ]);
  return [
    [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
    [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
    [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
    [0, 0, 0, 1],
  ];
}

function spinFromSecondary(
  parentWorld: number[][],
  parentHole: Vec3,
  desiredAxis: Vec3,
  rotation: number[][],
  childHole: Vec3,
  _childMount: CatalogMount,
  secondary: SecondaryConstraint,
): number {
  const parentSecond = transformPoint(parentWorld, holeInPart(secondary.parentMount, secondary.parentIndex));
  const childSecond = holeInPart(secondary.childMount, secondary.childIndex);
  const translation = sub(parentHole, [
    rotation[0][0] * childHole[0] + rotation[0][1] * childHole[1] + rotation[0][2] * childHole[2],
    rotation[1][0] * childHole[0] + rotation[1][1] * childHole[1] + rotation[1][2] * childHole[2],
    rotation[2][0] * childHole[0] + rotation[2][1] * childHole[1] + rotation[2][2] * childHole[2],
  ]);
  const childSecondWorld: Vec3 = add(
    [
      rotation[0][0] * childSecond[0] + rotation[0][1] * childSecond[1] + rotation[0][2] * childSecond[2],
      rotation[1][0] * childSecond[0] + rotation[1][1] * childSecond[1] + rotation[1][2] * childSecond[2],
      rotation[2][0] * childSecond[0] + rotation[2][1] * childSecond[1] + rotation[2][2] * childSecond[2],
    ],
    translation,
  );
  const parentRadial = projectOntoPlane(parentSecond, parentHole, desiredAxis);
  const childRadial = projectOntoPlane(childSecondWorld, parentHole, desiredAxis);
  const parentRadius = norm(parentRadial);
  const childRadius = norm(childRadial);
  if (Math.abs(parentRadius - childRadius) > PATTERN_MATCH_TOL_IN) {
    throw new Error(`secondary hole/pattern constraint does not match (${parentRadius.toFixed(4)} in vs ${childRadius.toFixed(4)} in)`);
  }
  if (parentRadius < PATTERN_MATCH_TOL_IN) return 0;
  return (signedAngleAround(childRadial, parentRadial, desiredAxis) * 180) / Math.PI;
}

export function solveAssemblyPoses(
  assembly: RobotAssembly,
  catalogParts: Record<string, CatalogPart>,
): Record<string, Transform3> {
  const { rootId, order, byChild } = validateAssemblyGraph(assembly);
  const instances = indexInstances(assembly);
  const matrices: Record<string, number[][]> = {};
  const root = instances.get(rootId);
  matrices[rootId] = root?.pose ? matrixFromPose(root.pose) : identity4();
  for (const ident of order) {
    if (ident === rootId) continue;
    const connection = byChild.get(ident);
    if (!connection) throw new AssemblyGraphError(`connection for ${ident} is missing`);
    const parentId = connection.parent.instanceId;
    const parentPart = catalogParts[parentId];
    const childPart = catalogParts[ident];
    if (!parentPart || !childPart) throw new AssemblyGraphError(`catalog part missing for ${parentId} or ${ident}`);
    const parentMount = partMount(parentPart, connection.parent.mountId);
    const childMount = partMount(childPart, connection.child.mountId);
    mountsCompatible(parentPart, parentMount, childPart, childMount);
    if (instances.get(ident)?.pose) {
      throw new AssemblyGraphError(`instance ${ident} pose is solved from connections and must be omitted`);
    }
    const secondary = connection.secondary
      ? {
          parentMount: partMount(parentPart, connection.secondary.parent.mountId),
          childMount: partMount(childPart, connection.secondary.child.mountId),
          parentIndex: patternIndex(connection.secondary.parent),
          childIndex: patternIndex(connection.secondary.child),
        }
      : undefined;
    matrices[ident] = solveMountTransform(matrices[parentId], parentMount, childMount, {
      parentIndex: patternIndex(connection.parent),
      childIndex: patternIndex(connection.child),
      spinDeg: connection.spinDeg || 0,
      secondary,
    });
  }
  const occupied = occupancyError(assembly.connections);
  if (occupied) throw new AssemblyGraphError(occupied);
  const poses: Record<string, Transform3> = {};
  for (const [ident, matrix] of Object.entries(matrices)) poses[ident] = poseFromMatrix(matrix);
  return poses;
}

export function childSubtree(assembly: RobotAssembly, instanceId: string): string[] {
  const children = new Map<string, string[]>();
  for (const connection of assembly.connections) {
    const list = children.get(connection.parent.instanceId) || [];
    list.push(connection.child.instanceId);
    children.set(connection.parent.instanceId, list);
  }
  const out: string[] = [];
  const visit = (ident: string) => {
    out.push(ident);
    for (const child of children.get(ident) || []) visit(child);
  };
  visit(instanceId);
  return out;
}
