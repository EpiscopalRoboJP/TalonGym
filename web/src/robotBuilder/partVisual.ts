import type { CatalogCollisionProxy, CatalogPart, CollisionShapeSpec, RigidPartSpec, Transform3 } from "../api";
import { theme } from "../theme";
import { identity4, matrixFromPose, mul4, poseFromMatrix } from "./transforms";

export function ftcSizeToThree(sizeIn?: number[] | null): [number, number, number] {
  const sx = Number(sizeIn?.[0]) || 1;
  const sy = Number(sizeIn?.[1]) || 1;
  const sz = Number(sizeIn?.[2]) || 1;
  return [sx, sz, sy];
}

export function collisionToSpec(part: CatalogPart): CollisionShapeSpec[] {
  const rows: CollisionShapeSpec[] = [];
  for (const proxy of part.collision || []) {
    const converted = proxyToSpec(proxy, part.cache?.collisionAsset);
    if (converted) rows.push(converted);
  }
  if (!rows.length && part.cache?.collisionAsset) {
    rows.push({ kind: "convex_mesh", asset: part.cache.collisionAsset });
  }
  if (!rows.length) {
    rows.push({ kind: "box", sizeIn: [2, 2, 1] });
  }
  return rows;
}

function proxyToSpec(proxy: CatalogCollisionProxy, collisionAsset?: string | null): CollisionShapeSpec | null {
  if (proxy.kind === "box" && proxy.sizeIn) {
    return { kind: "box", sizeIn: proxy.sizeIn, pose: proxy.pose };
  }
  if (proxy.kind === "sphere" && proxy.radiusIn) {
    return { kind: "sphere", radiusIn: proxy.radiusIn, pose: proxy.pose };
  }
  if ((proxy.kind === "cylinder" || proxy.kind === "capsule") && proxy.radiusIn) {
    return { kind: proxy.kind, radiusIn: proxy.radiusIn, lengthIn: proxy.lengthIn, pose: proxy.pose };
  }
  if (proxy.kind === "convex_hull") {
    if (collisionAsset) return { kind: "convex_mesh", asset: collisionAsset, pose: proxy.pose };
    if (proxy.sizeIn) return { kind: "box", sizeIn: proxy.sizeIn, pose: proxy.pose };
    if (proxy.radiusIn) return { kind: "sphere", radiusIn: proxy.radiusIn, pose: proxy.pose };
  }
  return null;
}

export function catalogPartToRigid(id: string, part: CatalogPart, pose?: Transform3): RigidPartSpec {
  return {
    id,
    pose,
    massKg: part.massKg,
    visualAsset: part.cache?.visualAsset || undefined,
    cacheState: part.cache?.state,
    cacheReason: part.cache?.reason,
    collision: collisionToSpec(part),
    tags: part.tags,
  };
}

export function proxyColor(tags?: string[] | null, id?: string): string {
  const ident = `${id || ""} ${(tags || []).join(" ")}`.toLowerCase();
  if (ident.includes("wheel")) return theme.intake;
  if (ident.includes("motor") || ident.includes("gearbox") || ident.includes("cartridge")) return theme.brand;
  if (ident.includes("servo")) return theme.goldDark;
  if (ident.includes("sensor") || ident.includes("hub") || ident.includes("control")) return theme.allianceBlueBright;
  if (ident.includes("bracket")) return "#9a7d5c";
  if (ident.includes("plate")) return "#c4b090";
  if (ident.includes("channel") || ident.includes("extrusion") || ident.includes("rail")) return "#d9cbb3";
  if (id === "chassis") return theme.chassis;
  if (ident.includes("fly")) return theme.gold;
  return theme.goldLight;
}

export function composeRigidPartPoses(parts: RigidPartSpec[]): Record<string, Transform3> {
  const byId = new Map(parts.map((part) => [part.id, part]));
  const cache = new Map<string, number[][]>();
  const visiting = new Set<string>();
  const matrixOf = (id: string): number[][] => {
    const hit = cache.get(id);
    if (hit) return hit;
    if (visiting.has(id)) return identity4();
    const part = byId.get(id);
    if (!part) return identity4();
    visiting.add(id);
    const local = matrixFromPose(part.pose);
    const parentId = part.parentId;
    const world = parentId && parentId !== id && byId.has(parentId) ? mul4(matrixOf(parentId), local) : local;
    visiting.delete(id);
    cache.set(id, world);
    return world;
  };
  const out: Record<string, Transform3> = {};
  for (const part of parts) out[part.id] = poseFromMatrix(matrixOf(part.id));
  return out;
}

export function collisionSpan(part: Pick<RigidPartSpec, "collision">): [number, number, number] {
  const collision = part.collision?.[0];
  if (collision && "sizeIn" in collision && collision.sizeIn) return ftcSizeToThree(collision.sizeIn);
  const radius = collision && "radiusIn" in collision ? collision.radiusIn || 1 : 1;
  const length = collision && "lengthIn" in collision ? collision.lengthIn || radius * 2 : radius * 2;
  return [Math.max(radius * 2, 1), Math.max(length, 1), Math.max(radius * 2, 1)];
}
