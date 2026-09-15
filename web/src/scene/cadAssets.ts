import { API, type Frame, type FramePiece, type GamePieceType, type VisualOffset } from "../api";

export const BIOBUZZ_GLB = "seasons/biobuzz_2026/field.glb";
export const LEGACY_PIECE_RADIUS_IN = 2.5;
export const CAD_MANIFEST_NAME = "cad_manifest.json";

export type CadPieceRecord = {
  visualAsset?: string;
  sha256?: string;
  specDiameterIn?: number;
  measuredDiameterIn?: number;
};

export type CadManifest = {
  generatorVersion?: string;
  field?: {
    sha256?: string;
    visualAsset?: string;
    mechanisms?: {
      id: string;
      alliance?: string;
      visualAsset: string;
      pivotIn: [number, number, number];
      axis: [number, number, number];
      tipAngleDeg: number;
    }[];
  };
  pieces?: Record<string, CadPieceRecord>;
};

export type PieceCatalogEntry = {
  typeId: string;
  visualAsset?: string | null;
  radius?: number;
  sha256?: string;
};

export type PieceCatalog = Map<string, PieceCatalogEntry>;

export function visualOffsetYawDeg(offset?: VisualOffset | null): number {
  if (!offset) return 0;
  if (typeof offset.yawDeg === "number" && Number.isFinite(offset.yawDeg)) return offset.yawDeg;
  if (typeof offset.headingDeg === "number" && Number.isFinite(offset.headingDeg)) return offset.headingDeg;
  return 0;
}

export function resolveBackgroundAsset(frame: Frame | null | undefined, fallback?: string | null) {
  if (frame?.backgroundAsset) return frame.backgroundAsset;
  if (fallback) return fallback;
  const els = frame?.elements || [];
  if (els.some((el) => el.id === "red_cell_up" || el.type === "hive_frame" || (el.tags || []).includes("hive"))) {
    return BIOBUZZ_GLB;
  }
  return null;
}

export function siblingCadManifestPath(assetPath: string | null | undefined): string | null {
  if (!assetPath) return null;
  const trimmed = assetPath.split("?")[0].replace(/\\/g, "/");
  if (!trimmed.toLowerCase().endsWith(".glb")) return null;
  const slash = trimmed.lastIndexOf("/");
  if (slash < 0) return CAD_MANIFEST_NAME;
  return `${trimmed.slice(0, slash)}/${CAD_MANIFEST_NAME}`;
}

export function resolveCadManifestPath(frame: Frame | null | undefined, fallback?: string | null): string | null {
  if (frame?.cadManifest) return frame.cadManifest;
  if (fallback) return fallback;
  return siblingCadManifestPath(resolveBackgroundAsset(frame));
}

export function explicitCadCacheToken(frame: Frame | null | undefined): string | null {
  const raw = frame?.cadAssetVersion || frame?.cadSourceSha256;
  return normalizeCacheToken(raw);
}

export function cadCacheTokenFromManifest(manifest: CadManifest | null | undefined): string | null {
  const source = normalizeCacheToken(manifest?.field?.sha256);
  const generator = normalizeCacheToken(manifest?.generatorVersion);
  return source && generator ? `${source}:${generator}` : source || generator;
}

export function normalizeCacheToken(raw: string | null | undefined): string | null {
  if (typeof raw !== "string") return null;
  const token = raw.trim();
  return token ? token : null;
}

export function fieldAssetUrl(assetPath: string, cacheToken?: string | null): string {
  const base = `${API}/field-assets/${assetPath}`;
  return cacheToken ? `${base}?v=${encodeURIComponent(cacheToken)}` : base;
}

export async function fetchCadManifest(rel: string): Promise<CadManifest | null> {
  try {
    const res = await fetch(`${API}/field-assets/${rel}`);
    if (!res.ok) return null;
    return (await res.json()) as CadManifest;
  } catch {
    return null;
  }
}

function radiusFromDiameter(value: number | undefined): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return undefined;
  return value / 2;
}

function radiusFromType(spec: GamePieceType | PieceCatalogEntry | CadPieceRecord | undefined): number | undefined {
  if (!spec) return undefined;
  if ("radius" in spec && typeof spec.radius === "number" && spec.radius > 0) return spec.radius;
  if ("shape" in spec && spec.shape && typeof spec.shape.radius === "number" && spec.shape.radius > 0) {
    return spec.shape.radius;
  }
  if ("measuredDiameterIn" in spec) {
    const measured = radiusFromDiameter(spec.measuredDiameterIn);
    if (measured) return measured;
  }
  if ("specDiameterIn" in spec) return radiusFromDiameter(spec.specDiameterIn);
  return undefined;
}

export function buildPieceCatalog(frame: Frame | null | undefined, manifest?: CadManifest | null): PieceCatalog {
  const catalog: PieceCatalog = new Map();
  const manifestPieces = manifest?.pieces || {};
  for (const [typeId, rec] of Object.entries(manifestPieces)) {
    catalog.set(typeId, {
      typeId,
      visualAsset: rec.visualAsset || null,
      radius: radiusFromType(rec),
      sha256: rec.sha256,
    });
  }
  for (const spec of frame?.gamePieces || []) {
    const prev = catalog.get(spec.typeId);
    catalog.set(spec.typeId, {
      typeId: spec.typeId,
      visualAsset: spec.visualAsset || prev?.visualAsset || null,
      radius: radiusFromType(spec) ?? prev?.radius,
      sha256: prev?.sha256,
    });
  }
  return catalog;
}

export function inferPieceVisualAsset(typeId: string | undefined, backgroundAsset: string | null | undefined): string | null {
  if (!typeId || !backgroundAsset) return null;
  const trimmed = backgroundAsset.split("?")[0].replace(/\\/g, "/");
  const slash = trimmed.lastIndexOf("/");
  if (slash < 0) return null;
  return `${trimmed.slice(0, slash)}/pieces/${typeId}.glb`;
}

export function resolvePieceVisualAsset(
  piece: FramePiece,
  catalog: PieceCatalog,
  backgroundAsset?: string | null,
): string | null {
  if (piece.visualAsset) return piece.visualAsset;
  const typed = piece.typeId ? catalog.get(piece.typeId) : undefined;
  if (typed?.visualAsset) return typed.visualAsset;
  return inferPieceVisualAsset(piece.typeId, backgroundAsset);
}

export function resolvePieceRadiusIn(piece: FramePiece, catalog: PieceCatalog): number {
  if (typeof piece.radius === "number" && Number.isFinite(piece.radius) && piece.radius > 0) return piece.radius;
  const typed = piece.typeId ? catalog.get(piece.typeId) : undefined;
  if (typed?.radius && typed.radius > 0) return typed.radius;
  return LEGACY_PIECE_RADIUS_IN;
}

export function usesCadPieceVisual(
  piece: FramePiece,
  catalog: PieceCatalog,
  backgroundAsset?: string | null,
): boolean {
  return Boolean(resolvePieceVisualAsset(piece, catalog, backgroundAsset));
}

export function ftcToThreePosition(x: number, y: number, z: number): [number, number, number] {
  return [x, z, -y];
}

export function hasYupQuaternion(piece: Pick<FramePiece, "qw" | "qx" | "qy" | "qz">): boolean {
  const vals = [piece.qw, piece.qx, piece.qy, piece.qz];
  if (vals.some((n) => typeof n !== "number" || !Number.isFinite(n))) return false;
  const len = Math.hypot(piece.qw as number, piece.qx as number, piece.qy as number, piece.qz as number);
  return len > 1e-6;
}

/** Three.js quaternion xyzw from MuJoCo/Three Y-up wxyz. */
export function pieceThreeQuaternion(piece: FramePiece): [number, number, number, number] | null {
  if (!hasYupQuaternion(piece)) return null;
  return [piece.qx as number, piece.qy as number, piece.qz as number, piece.qw as number];
}

/** Three Euler XYZ: pitch about X, FTC heading about Y, roll about Z. */
export function pieceThreeEulerRad(piece: FramePiece): [number, number, number] {
  const heading = ((piece.headingDeg || 0) * Math.PI) / 180;
  const pitch = ((piece.pitchDeg || 0) * Math.PI) / 180;
  const roll = ((piece.rollDeg || 0) * Math.PI) / 180;
  return [pitch, heading, roll];
}

export function pieceThreePose(piece: FramePiece, catalog: PieceCatalog) {
  const radius = resolvePieceRadiusIn(piece, catalog);
  const z = typeof piece.z === "number" && Number.isFinite(piece.z) ? piece.z : radius;
  return {
    position: ftcToThreePosition(piece.x, piece.y, z),
    quaternion: pieceThreeQuaternion(piece),
    euler: pieceThreeEulerRad(piece),
    radius,
  };
}

export function pieceAssetUrl(
  visualAsset: string,
  catalog: PieceCatalog,
  typeId?: string,
  fieldToken?: string | null,
): string {
  const pieceToken = typeId ? catalog.get(typeId)?.sha256 : undefined;
  return fieldAssetUrl(visualAsset, pieceToken || fieldToken);
}
