import type { AssemblyConnection, CatalogMount, CatalogPart, MountRef, RobotAssembly, Transform3 } from "../api";
import { solveMountTransform } from "./assemblyMath";
import { holeInPart, mountCompatibilityError, patternHoles } from "./mounts";
import { identity4, matrixFromPose, poseFromMatrix, transformPoint } from "./transforms";

export const ORIENTATION_SPINS = [0, 90, 180, 270];
export const SNAP_RANGE_IN = 8;

export type SnapCandidate = {
  parentInstanceId: string;
  parentMountId: string;
  parentIndex?: [number, number];
  childMountId: string;
  childIndex?: [number, number];
  spinDeg: number;
  pose: Transform3;
  score: number;
};

export type MountTarget = {
  instanceId: string;
  mountId: string;
  patternIndex: [number, number];
};

export function compatibleMountPairs(parentPart: CatalogPart, childPart: CatalogPart): { parent: CatalogMount; child: CatalogMount }[] {
  const pairs: { parent: CatalogMount; child: CatalogMount }[] = [];
  for (const parent of parentPart.mounts || []) {
    for (const child of childPart.mounts || []) {
      if (!mountCompatibilityError(parentPart, parent, childPart, child)) pairs.push({ parent, child });
    }
  }
  return pairs;
}

export function worldMountHoles(
  part: CatalogPart,
  world: number[][],
): { mount: CatalogMount; index: [number, number]; world: [number, number, number] }[] {
  const holes: { mount: CatalogMount; index: [number, number]; world: [number, number, number] }[] = [];
  for (const mount of part.mounts || []) {
    for (const index of patternHoles(mount)) {
      holes.push({ mount, index, world: transformPoint(world, holeInPart(mount, index)) });
    }
  }
  return holes;
}

export function snapCandidatesFor(
  parentPart: CatalogPart,
  parentWorld: number[][],
  childPart: CatalogPart,
  pointer: [number, number, number],
  spinDeg: number,
): SnapCandidate[] {
  const out: SnapCandidate[] = [];
  for (const pair of compatibleMountPairs(parentPart, childPart)) {
    for (const parentIndex of patternHoles(pair.parent)) {
      for (const childIndex of patternHoles(pair.child)) {
        try {
          const matrix = solveMountTransform(parentWorld, pair.parent, pair.child, {
            parentIndex,
            childIndex,
            spinDeg,
          });
          const parentHole = transformPoint(parentWorld, holeInPart(pair.parent, parentIndex));
          const dist = Math.hypot(parentHole[0] - pointer[0], parentHole[1] - pointer[1], parentHole[2] - pointer[2]);
          out.push({
            parentInstanceId: "",
            parentMountId: pair.parent.id,
            parentIndex,
            childMountId: pair.child.id,
            childIndex,
            spinDeg,
            pose: poseFromMatrix(matrix),
            score: dist,
          });
        } catch {
          /* skip unsolvable orientations */
        }
      }
    }
  }
  return out.sort((a, b) => a.score - b.score);
}

export function candidatesForMount(
  target: MountTarget,
  parentPart: CatalogPart,
  parentPose: Transform3,
  childPart: CatalogPart,
  spinDeg: number,
): SnapCandidate[] {
  return snapCandidatesFor(parentPart, matrixFromPose(parentPose), childPart, [0, 0, 0], spinDeg)
    .filter(
      (candidate) =>
        candidate.parentMountId === target.mountId &&
        candidate.parentIndex?.[0] === target.patternIndex[0] &&
        candidate.parentIndex?.[1] === target.patternIndex[1],
    )
    .map((candidate) => ({ ...candidate, parentInstanceId: target.instanceId, score: 0 }));
}

export function partFitsMount(parentPart: CatalogPart, target: MountTarget, childPart: CatalogPart): boolean {
  const parentMount = (parentPart.mounts || []).find((mount) => mount.id === target.mountId);
  if (!parentMount) return false;
  return (childPart.mounts || []).some(
    (childMount) => !mountCompatibilityError(parentPart, parentMount, childPart, childMount),
  );
}

export function bestSnap(
  assembly: RobotAssembly,
  poses: Record<string, Transform3>,
  catalog: Record<string, CatalogPart>,
  childPart: CatalogPart,
  pointer: [number, number, number],
  spinIndex: number,
  excludeIds: Set<string> = new Set(),
): SnapCandidate | null {
  const spinDeg = ORIENTATION_SPINS[((spinIndex % ORIENTATION_SPINS.length) + ORIENTATION_SPINS.length) % ORIENTATION_SPINS.length];
  let best: SnapCandidate | null = null;
  for (const instance of assembly.instances) {
    if (excludeIds.has(instance.id)) continue;
    const parentPart = catalog[instance.id];
    const pose = poses[instance.id];
    if (!parentPart || !pose) continue;
    const hit = snapCandidatesFor(parentPart, matrixFromPose(pose), childPart, pointer, spinDeg)[0];
    if (!hit) continue;
    const ranked = { ...hit, parentInstanceId: instance.id };
    if (!best || ranked.score < best.score) best = ranked;
  }
  if (!best || best.score > SNAP_RANGE_IN) return null;
  return best;
}

export function snapConnection(candidate: SnapCandidate, childInstanceId: string, connectionId: string): AssemblyConnection {
  const parent: MountRef = { instanceId: candidate.parentInstanceId, mountId: candidate.parentMountId };
  const child: MountRef = { instanceId: childInstanceId, mountId: candidate.childMountId };
  if (candidate.parentIndex) parent.patternIndex = candidate.parentIndex;
  if (candidate.childIndex) child.patternIndex = candidate.childIndex;
  return {
    id: connectionId,
    parent,
    child,
    spinDeg: candidate.spinDeg,
  };
}

export function cycleSpin(index: number, delta = 1): number {
  return ((index + delta) % ORIENTATION_SPINS.length + ORIENTATION_SPINS.length) % ORIENTATION_SPINS.length;
}

export function highlightMounts(
  part: CatalogPart,
  world: number[][],
  childPart: CatalogPart,
): { mountId: string; index: [number, number]; world: [number, number, number]; compatible: boolean }[] {
  const compatible = new Set(compatibleMountPairs(part, childPart).map((pair) => pair.parent.id));
  return worldMountHoles(part, world).map((hole) => ({
    mountId: hole.mount.id,
    index: hole.index,
    world: hole.world,
    compatible: compatible.has(hole.mount.id),
  }));
}

export function identityWorld(): number[][] {
  return identity4();
}
