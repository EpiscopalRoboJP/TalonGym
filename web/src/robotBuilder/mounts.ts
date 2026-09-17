import type { AssemblyConnection, CatalogMount, CatalogPart, MountRef } from "../api";
import { add, cross, dot, normalize, planeBasis, scale, sub, type Vec3 } from "./transforms";

export const MM_TO_IN = 1 / 25.4;
export const DIAMETER_TOL_MM = 0.2;
export const MAX_CLEARANCE_OVER_THREAD_MM = 1.5;
export const ADAPTER_STANDARD = "adapter_required";

const HARDWARE_FAMILIES: Record<string, string> = {
  m3_screw: "m3",
  m3_t_nut: "m3",
  m4_screw: "m4",
  m4_t_nut: "m4",
};
const STANDARD_FAMILIES: Record<string, string> = {
  rev_m3: "m3",
  rev_15mm: "m3",
  gobilda_pattern: "m4",
};
const HOLE_KINDS = new Set(["threaded_hole", "clearance_hole"]);

const COMPATIBLE_KIND_PAIRS: ReadonlySet<string> = new Set([
  "clearance_hole|threaded_hole",
  "threaded_hole|clearance_hole",
  "clearance_hole|clearance_hole",
  "clearance_hole|mating_face",
  "mating_face|clearance_hole",
  "hub|shaft",
  "shaft|hub",
  "bearing|shaft",
  "shaft|bearing",
  "clamp|shaft",
  "shaft|clamp",
  "mating_face|mating_face",
]);

const ROTATING_KIND_PAIRS: ReadonlySet<string> = new Set([
  "hub|shaft",
  "shaft|hub",
  "bearing|shaft",
  "shaft|bearing",
  "clamp|shaft",
  "shaft|clamp",
]);

export function partMount(part: CatalogPart, mountId: string): CatalogMount {
  const mount = (part.mounts || []).find((row) => row.id === mountId);
  if (!mount) throw new Error(`part ${part.sku} has no mount ${mountId}`);
  return mount;
}

export function mountAxis(mount: CatalogMount): Vec3 {
  return [mount.axis[0], mount.axis[1], mount.axis[2]];
}

export function patternCounts(pattern?: CatalogMount["pattern"]): [number, number] {
  const row = pattern;
  const countU = row?.countU || row?.count || 1;
  const countV = row?.countV || 1;
  const kind = row?.type || "single";
  if (kind === "single") return [1, 1];
  if (kind === "linear") return [Math.max(countU, 1), 1];
  if (kind === "circle") return [Math.max(row?.count || countU, 1), 1];
  return [Math.max(countU, 1), Math.max(countV, 1)];
}

export function patternHoles(mount: CatalogMount): [number, number][] {
  const [countU, countV] = patternCounts(mount.pattern);
  const holes: [number, number][] = [];
  for (let u = 0; u < countU; u += 1) {
    for (let v = 0; v < countV; v += 1) holes.push([u, v]);
  }
  return holes.length ? holes : [[0, 0]];
}

function patternOffsetUv(pattern: CatalogMount["pattern"] | undefined, index: [number, number] | null | undefined): [number, number] {
  if (!index || !pattern) return [0, 0];
  const kind = pattern.type || "single";
  const [countU, countV] = patternCounts(pattern);
  const [uI, vI] = index;
  if (uI < 0 || vI < 0 || uI >= countU || vI >= countV) {
    throw new Error(`pattern index ${index.join(",")} is outside ${kind} ${countU}x${countV}`);
  }
  const pitch = (pattern.pitchMm || 0) * MM_TO_IN;
  if (kind === "single") return [0, 0];
  if (kind === "circle") {
    const angle = (2 * Math.PI * uI) / countU;
    return [pitch * Math.cos(angle), pitch * Math.sin(angle)];
  }
  const du = (uI - (countU - 1) / 2) * pitch;
  const dv = kind === "linear" ? 0 : (vI - (countV - 1) / 2) * pitch;
  return [du, dv];
}

export function patternBasis(mount: CatalogMount): [Vec3, Vec3] {
  const axis = mountAxis(mount);
  if (!mount.uAxis) return planeBasis(axis);
  const raw = normalize(mount.uAxis);
  const projected = sub(raw, scale(axis, dot(raw, axis)));
  const uAxis = normalize(projected);
  return [uAxis, cross(axis, uAxis)];
}

export function holeInPart(mount: CatalogMount, index?: [number, number] | null): Vec3 {
  const pose = mount.transform || {};
  const origin: Vec3 = [pose.x || 0, pose.y || 0, pose.z || 0];
  const [uAxis, vAxis] = patternBasis(mount);
  const offset = patternOffsetUv(mount.pattern, index);
  return add(add(origin, scale(uAxis, offset[0])), scale(vAxis, offset[1]));
}

export function inferredJointType(parentMount: CatalogMount, childMount: CatalogMount): "fixed" | "hinge" {
  const pair = `${parentMount.kind}|${childMount.kind}`;
  return ROTATING_KIND_PAIRS.has(pair) ? "hinge" : "fixed";
}

export function kindsCompatible(parentKind: string, childKind: string): boolean {
  return COMPATIBLE_KIND_PAIRS.has(`${parentKind}|${childKind}`);
}

export function hardwareFamilies(mount: CatalogMount): Set<string> {
  const families = new Set<string>();
  for (const item of mount.allowedHardware || []) {
    const family = HARDWARE_FAMILIES[item.toLowerCase()];
    if (family) families.add(family);
  }
  const standardFamily = STANDARD_FAMILIES[mount.standard];
  if (standardFamily) families.add(standardFamily);
  return families;
}

export function standardsCompatible(parentMount: CatalogMount, childMount: CatalogMount): boolean {
  if (parentMount.standard === childMount.standard) return true;
  if (parentMount.standard === ADAPTER_STANDARD || childMount.standard === ADAPTER_STANDARD) return true;
  const kinds = new Set([parentMount.kind, childMount.kind]);
  const shared = [...hardwareFamilies(parentMount)].some((family) => hardwareFamilies(childMount).has(family));
  if (kinds.has("clearance_hole") && kinds.has("mating_face") && shared) return true;
  if (kinds.has("threaded_hole") && kinds.has("clearance_hole") && shared) return true;
  return false;
}

export function diametersCompatible(parentMount: CatalogMount, childMount: CatalogMount): boolean {
  if (parentMount.diameterMm == null || childMount.diameterMm == null) return true;
  const kinds = new Set([parentMount.kind, childMount.kind]);
  if (kinds.has("threaded_hole") && kinds.has("clearance_hole")) {
    const thread = parentMount.kind === "threaded_hole" ? parentMount.diameterMm : childMount.diameterMm;
    const clear = parentMount.kind === "clearance_hole" ? parentMount.diameterMm : childMount.diameterMm;
    if (Math.abs(thread - clear) <= DIAMETER_TOL_MM) return true;
    const over = clear - thread;
    return over > 0 && over <= MAX_CLEARANCE_OVER_THREAD_MM;
  }
  return Math.abs(parentMount.diameterMm - childMount.diameterMm) <= DIAMETER_TOL_MM;
}

export function mountCompatibilityError(
  parentPart: CatalogPart,
  parentMount: CatalogMount,
  childPart: CatalogPart,
  childMount: CatalogMount,
): string | null {
  if (!kindsCompatible(parentMount.kind, childMount.kind)) {
    return `incompatible mount kinds ${parentMount.kind} and ${childMount.kind}`;
  }
  if (!standardsCompatible(parentMount, childMount)) {
    return `mismatched mount standards ${parentMount.standard} and ${childMount.standard}`;
  }
  if (parentPart.manufacturer && childPart.manufacturer && parentPart.manufacturer !== childPart.manufacturer) {
    if (parentMount.standard !== ADAPTER_STANDARD && childMount.standard !== ADAPTER_STANDARD) {
      return "cross-brand connection requires a catalog adapter";
    }
  }
  const holePair = HOLE_KINDS.has(parentMount.kind) && HOLE_KINDS.has(childMount.kind);
  const nutPair = new Set([parentMount.kind, childMount.kind]);
  if (holePair || (nutPair.has("clearance_hole") && nutPair.has("mating_face"))) {
    const parentFamilies = hardwareFamilies(parentMount);
    const childFamilies = hardwareFamilies(childMount);
    if (parentFamilies.size && childFamilies.size && ![...parentFamilies].some((family) => childFamilies.has(family))) {
      return `incompatible hardware ${parentMount.allowedHardware} and ${childMount.allowedHardware}`;
    }
  }
  if (!diametersCompatible(parentMount, childMount)) {
    return `incompatible mount diameters ${parentMount.diameterMm} mm and ${childMount.diameterMm} mm`;
  }
  return null;
}

export function mountsCompatible(
  parentPart: CatalogPart,
  parentMount: CatalogMount,
  childPart: CatalogPart,
  childMount: CatalogMount,
): void {
  const error = mountCompatibilityError(parentPart, parentMount, childPart, childMount);
  if (error) throw new Error(error);
}

function occupancyKey(ref: MountRef): string {
  const index = ref.patternIndex;
  return index ? `${ref.instanceId}|${ref.mountId}|${index[0]}|${index[1]}` : `${ref.instanceId}|${ref.mountId}|*`;
}

function occupancyConflicts(occupied: Set<string>, key: string): boolean {
  const [instanceId, mountId] = key.split("|");
  if (occupied.has(`${instanceId}|${mountId}|*`)) return true;
  if (key.endsWith("|*")) {
    for (const item of occupied) {
      if (item.startsWith(`${instanceId}|${mountId}|`)) return true;
    }
    return false;
  }
  return occupied.has(key);
}

export function occupancyError(connections: AssemblyConnection[]): string | null {
  const occupied = new Set<string>();
  const claim = (ref: MountRef, connectionId: string): string | null => {
    const key = occupancyKey(ref);
    if (occupancyConflicts(occupied, key)) {
      const hole = ref.patternIndex ? ` hole ${ref.patternIndex.join(",")}` : "";
      return `duplicate occupancy of ${ref.instanceId}.${ref.mountId}${hole} by connection ${connectionId}`;
    }
    occupied.add(key);
    return null;
  };
  for (const connection of connections) {
    const ident = connection.id;
    const err =
      claim(connection.parent, ident) ||
      claim(connection.child, ident) ||
      (connection.secondary ? claim(connection.secondary.parent, ident) || claim(connection.secondary.child, ident) : null);
    if (err) return err;
  }
  return null;
}
