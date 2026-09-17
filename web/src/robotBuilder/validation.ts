import type { CatalogPart, RobotAssembly, Transform3 } from "../api";
import { AssemblyGraphError, solveAssemblyPoses, validateAssemblyGraph } from "./assemblyMath";
import { occupancyError } from "./mounts";
import { matrixFromPose, transformPoint } from "./transforms";

export type ValidationIssue = {
  code: string;
  severity: "error" | "warning" | "info";
  message: string;
  instanceId?: string;
};

export type ValidationResult = {
  blocking: ValidationIssue[];
  warnings: ValidationIssue[];
  poses: Record<string, Transform3>;
};

const FTC_ENVELOPE_IN = 18;
const FTC_MASS_KG = 19.278;

function proxySize(proxy: { kind?: string; sizeIn?: number[]; radiusIn?: number; lengthIn?: number }): [number, number, number] | null {
  const kind = proxy.kind || "box";
  if (kind === "box" && proxy.sizeIn?.length === 3) return [proxy.sizeIn[0], proxy.sizeIn[1], proxy.sizeIn[2]];
  const radius = proxy.radiusIn || 0;
  if (kind === "sphere") return [radius * 2, radius * 2, radius * 2];
  if (kind === "cylinder" || kind === "capsule" || kind === "convex_hull") {
    const along = (proxy.lengthIn || 0) + (kind === "capsule" ? 2 * radius : 0);
    const span = Math.max(radius * 2, 0.1);
    return [span, span, along || span];
  }
  return null;
}

function partAabb(world: number[][], part: CatalogPart): { min: [number, number, number]; max: [number, number, number] } | null {
  let min: [number, number, number] | null = null;
  let max: [number, number, number] | null = null;
  for (const proxy of part.collision || []) {
    const size = proxySize(proxy);
    if (!size) continue;
    const local = matrixFromPose(proxy.pose);
    const worldProxy = [
      [
        world[0][0] * local[0][0] + world[0][1] * local[1][0] + world[0][2] * local[2][0],
        world[0][0] * local[0][1] + world[0][1] * local[1][1] + world[0][2] * local[2][1],
        world[0][0] * local[0][2] + world[0][1] * local[1][2] + world[0][2] * local[2][2],
        world[0][0] * local[0][3] + world[0][1] * local[1][3] + world[0][2] * local[2][3] + world[0][3],
      ],
      [
        world[1][0] * local[0][0] + world[1][1] * local[1][0] + world[1][2] * local[2][0],
        world[1][0] * local[0][1] + world[1][1] * local[1][1] + world[1][2] * local[2][1],
        world[1][0] * local[0][2] + world[1][1] * local[1][2] + world[1][2] * local[2][2],
        world[1][0] * local[0][3] + world[1][1] * local[1][3] + world[1][2] * local[2][3] + world[1][3],
      ],
      [
        world[2][0] * local[0][0] + world[2][1] * local[1][0] + world[2][2] * local[2][0],
        world[2][0] * local[0][1] + world[2][1] * local[1][1] + world[2][2] * local[2][1],
        world[2][0] * local[0][2] + world[2][1] * local[1][2] + world[2][2] * local[2][2],
        world[2][0] * local[0][3] + world[2][1] * local[1][3] + world[2][2] * local[2][3] + world[2][3],
      ],
      [0, 0, 0, 1],
    ];
    const hx = size[0] / 2;
    const hy = size[1] / 2;
    const hz = size[2] / 2;
    for (const x of [-hx, hx]) {
      for (const y of [-hy, hy]) {
        for (const z of [-hz, hz]) {
          const p = transformPoint(worldProxy, [x, y, z]);
          if (!min || !max) {
            min = [...p];
            max = [...p];
          } else {
            min = [Math.min(min[0], p[0]), Math.min(min[1], p[1]), Math.min(min[2], p[2])];
            max = [Math.max(max[0], p[0]), Math.max(max[1], p[1]), Math.max(max[2], p[2])];
          }
        }
      }
    }
  }
  return min && max ? { min, max } : null;
}

function overlap(a: { min: [number, number, number]; max: [number, number, number] }, b: { min: [number, number, number]; max: [number, number, number] }) {
  return a.min[0] < b.max[0] - 1e-4 && a.min[1] < b.max[1] - 1e-4 && a.min[2] < b.max[2] - 1e-4 && b.min[0] < a.max[0] - 1e-4 && b.min[1] < a.max[1] - 1e-4 && b.min[2] < a.max[2] - 1e-4;
}

export function validateAssembly(assembly: RobotAssembly, catalog: Record<string, CatalogPart>): ValidationResult {
  const blocking: ValidationIssue[] = [];
  const warnings: ValidationIssue[] = [];
  let poses: Record<string, Transform3> = {};
  if (!assembly.instances.length) {
    return { blocking: [{ code: "empty", severity: "error", message: "Assembly has no parts yet." }], warnings, poses };
  }
  try {
    validateAssemblyGraph(assembly);
  } catch (err) {
    blocking.push({ code: "graph", severity: "error", message: err instanceof Error ? err.message : String(err) });
    return { blocking, warnings, poses };
  }
  const occupied = occupancyError(assembly.connections);
  if (occupied) blocking.push({ code: "occupancy", severity: "error", message: occupied });
  const missingCatalog = assembly.instances.filter((row) => !catalog[row.id]);
  if (missingCatalog.length) {
    warnings.push({
      code: "catalog_loading",
      severity: "info",
      message: `Loading ${missingCatalog.length} catalog part(s)…`,
    });
    return { blocking, warnings, poses };
  }
  try {
    poses = solveAssemblyPoses(assembly, catalog);
  } catch (err) {
    if (!(err instanceof AssemblyGraphError) || !blocking.length) {
      blocking.push({ code: "snap", severity: "error", message: err instanceof Error ? err.message : String(err) });
    }
    return { blocking, warnings, poses };
  }
  const mated = new Set(assembly.connections.map((row) => [row.parent.instanceId, row.child.instanceId].sort().join("|")));
  const ids = assembly.instances.map((row) => row.id);
  const parent: Record<string, string> = {};
  for (const id of ids) parent[id] = id;
  const find = (ident: string): string => {
    if (parent[ident] !== ident) parent[ident] = find(parent[ident]);
    return parent[ident];
  };
  const union = (left: string, right: string) => {
    const a = find(left);
    const b = find(right);
    if (a !== b) parent[b] = a;
  };
  for (const connection of assembly.connections) union(connection.parent.instanceId, connection.child.instanceId);
  for (let i = 0; i < ids.length; i += 1) {
    for (let j = i + 1; j < ids.length; j += 1) {
      const pair = [ids[i], ids[j]].sort().join("|");
      if (mated.has(pair) || find(ids[i]) === find(ids[j])) continue;
      const left = catalog[ids[i]] && poses[ids[i]] ? partAabb(matrixFromPose(poses[ids[i]]), catalog[ids[i]]) : null;
      const right = catalog[ids[j]] && poses[ids[j]] ? partAabb(matrixFromPose(poses[ids[j]]), catalog[ids[j]]) : null;
      if (left && right && overlap(left, right)) {
        blocking.push({
          code: "collision",
          severity: "error",
          message: `interpenetration between ${ids[i]} and ${ids[j]} outside mating clearance`,
          instanceId: ids[i],
        });
      }
    }
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let mass = 0;
  for (const instance of assembly.instances) {
    const part = catalog[instance.id];
    const pose = poses[instance.id];
    if (!part || !pose) continue;
    mass += part.massKg || 0;
    const box = partAabb(matrixFromPose(pose), part);
    if (!box) continue;
    minX = Math.min(minX, box.min[0]);
    maxX = Math.max(maxX, box.max[0]);
    minY = Math.min(minY, box.min[1]);
    maxY = Math.max(maxY, box.max[1]);
  }
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  if (Number.isFinite(spanX) && (spanX > FTC_ENVELOPE_IN || maxY - minY > FTC_ENVELOPE_IN)) {
    warnings.push({
      code: "ftc_envelope",
      severity: "warning",
      message: `Starting envelope ${spanX.toFixed(1)} × ${spanY.toFixed(1)} in exceeds the 18 in FTC cube.`,
    });
  }
  if (mass > FTC_MASS_KG) {
    warnings.push({
      code: "ftc_mass",
      severity: "warning",
      message: `Assembled mass ${mass.toFixed(2)} kg exceeds the 42.5 lb FTC limit.`,
    });
  }
  return { blocking, warnings, poses };
}

export function canSaveAssembly(result: ValidationResult): boolean {
  return result.blocking.length === 0;
}
