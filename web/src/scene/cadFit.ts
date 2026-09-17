/** Classic m/mm/in mistakes are at least 25× off an authored collision proxy. */
export const CAD_UNIT_FIT_RATIO = 25;
export const CAD_LONG_AXIS_RATIO = 2.5;
const CLASSIC_UNIT_SCALES = [1, 25.4, 39.37007874015748, 1000];

export type Vec3 = [number, number, number];
/** Row-major 3×3, applied to column vectors. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];

export type CadCollisionFit = {
  ready: boolean;
  scale: number;
  rotation: Mat3;
  translation: Vec3;
};

const IDENTITY: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function maxAbs(values: Vec3): number {
  return Math.max(Math.abs(values[0]), Math.abs(values[1]), Math.abs(values[2]));
}

export function cadUnitFitScale(meshSpan: number, targetSpan: number): number {
  const mesh = Math.abs(Number(meshSpan) || 0);
  const target = Math.abs(Number(targetSpan) || 0);
  if (!(mesh > 1e-8) || !(target > 1e-8)) return 1;
  let bestScale = 1;
  let bestRel = Math.abs(mesh - target) / target;
  for (const factor of CLASSIC_UNIT_SCALES) {
    for (const scale of [factor, 1 / factor]) {
      if (scale === 1) continue;
      const rel = Math.abs(mesh * scale - target) / target;
      if (rel < bestRel) {
        bestRel = rel;
        bestScale = scale;
      }
    }
  }
  if (bestScale !== 1 && bestRel <= 0.2) return bestScale;
  return 1;
}

export function uniqueLongAxis(size: Vec3): 0 | 1 | 2 | null {
  const abs: Vec3 = [Math.abs(size[0]), Math.abs(size[1]), Math.abs(size[2])];
  let long: 0 | 1 | 2 = 0;
  if (abs[1] > abs[long]) long = 1;
  if (abs[2] > abs[long]) long = 2;
  const rest = long === 0 ? Math.max(abs[1], abs[2]) : long === 1 ? Math.max(abs[0], abs[2]) : Math.max(abs[0], abs[1]);
  if (!(abs[long] > CAD_LONG_AXIS_RATIO * Math.max(rest, 1e-12))) return null;
  return long;
}

function mulMatVec(m: Mat3, v: Vec3): Vec3 {
  return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
}

function rotationAligning(src: Vec3, dst: Vec3): Mat3 {
  const aLen = Math.hypot(src[0], src[1], src[2]);
  const bLen = Math.hypot(dst[0], dst[1], dst[2]);
  if (!(aLen > 1e-12) || !(bLen > 1e-12)) return IDENTITY;
  const a: Vec3 = [src[0] / aLen, src[1] / aLen, src[2] / aLen];
  const b: Vec3 = [dst[0] / bLen, dst[1] / bLen, dst[2] / bLen];
  const c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if (c > 0.999999) return IDENTITY;
  if (c < -0.999999) {
    const helper: Vec3 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const axis: Vec3 = [a[1] * helper[2] - a[2] * helper[1], a[2] * helper[0] - a[0] * helper[2], a[0] * helper[1] - a[1] * helper[0]];
    const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
    const x = axis[0] / n;
    const y = axis[1] / n;
    const z = axis[2] / n;
    return [2 * x * x - 1, 2 * x * y, 2 * x * z, 2 * y * x, 2 * y * y - 1, 2 * y * z, 2 * z * x, 2 * z * y, 2 * z * z - 1];
  }
  const v: Vec3 = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const s2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const k: Mat3 = [0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0];
  const kk: Mat3 = [
    k[0] * k[0] + k[1] * k[3] + k[2] * k[6],
    k[0] * k[1] + k[1] * k[4] + k[2] * k[7],
    k[0] * k[2] + k[1] * k[5] + k[2] * k[8],
    k[3] * k[0] + k[4] * k[3] + k[5] * k[6],
    k[3] * k[1] + k[4] * k[4] + k[5] * k[7],
    k[3] * k[2] + k[4] * k[5] + k[5] * k[8],
    k[6] * k[0] + k[7] * k[3] + k[8] * k[6],
    k[6] * k[1] + k[7] * k[4] + k[8] * k[7],
    k[6] * k[2] + k[7] * k[5] + k[8] * k[8],
  ];
  const f = (1 - c) / s2;
  return [
    1 + k[0] + kk[0] * f,
    k[1] + kk[1] * f,
    k[2] + kk[2] * f,
    k[3] + kk[3] * f,
    1 + k[4] + kk[4] * f,
    k[5] + kk[5] * f,
    k[6] + kk[6] * f,
    k[7] + kk[7] * f,
    1 + k[8] + kk[8] * f,
  ];
}

function unitAxis(axis: 0 | 1 | 2): Vec3 {
  return axis === 0 ? [1, 0, 0] : axis === 1 ? [0, 1, 0] : [0, 0, 1];
}

export function cadCollisionFit(meshSize: Vec3, meshCenter: Vec3, targetSize?: Vec3 | null): CadCollisionFit {
  const size: Vec3 = [Math.abs(meshSize[0] || 0), Math.abs(meshSize[1] || 0), Math.abs(meshSize[2] || 0)];
  const target: Vec3 = [
    Math.abs(Number(targetSize?.[0]) || 0),
    Math.abs(Number(targetSize?.[1]) || 0),
    Math.abs(Number(targetSize?.[2]) || 0),
  ];
  const meshSpan = maxAbs(size);
  const targetSpan = maxAbs(target);
  const scale = targetSpan > 1e-8 ? cadUnitFitScale(meshSpan, targetSpan) : 1;
  const fitted: Vec3 = [size[0] * scale, size[1] * scale, size[2] * scale];
  const center: Vec3 = [(meshCenter[0] || 0) * scale, (meshCenter[1] || 0) * scale, (meshCenter[2] || 0) * scale];
  const meshLong = uniqueLongAxis(fitted);
  const targetLong = uniqueLongAxis(target);
  let rotation = IDENTITY;
  let rotatedCenter = center;
  if (meshLong != null && targetLong != null && meshLong !== targetLong) {
    rotation = rotationAligning(unitAxis(meshLong), unitAxis(targetLong));
    rotatedCenter = mulMatVec(rotation, center);
  }
  const long = targetLong ?? meshLong;
  const translation: Vec3 = [0, 0, 0];
  if (long != null && Math.abs(rotatedCenter[long]) > 0.2 * Math.max(fitted[meshLong ?? long], 1e-12)) {
    translation[0] = -rotatedCenter[0];
    translation[1] = -rotatedCenter[1];
    translation[2] = -rotatedCenter[2];
  }
  const fittedSpan = maxAbs(fitted);
  const lo = Math.min(fittedSpan, targetSpan || fittedSpan);
  const hi = Math.max(fittedSpan, targetSpan || fittedSpan);
  const ready = fittedSpan > 1e-8 && (!(targetSpan > 1e-8) || (lo > 1e-12 && hi / lo <= CAD_UNIT_FIT_RATIO));
  return { ready, scale, rotation, translation };
}

export function cadShouldShowMesh(
  triangleCount: number,
  meshSpan: number,
  targetSpan?: number,
  meshSize?: Vec3,
  meshCenter?: Vec3,
  targetSize?: Vec3,
): { ready: boolean; scale: number; fit: CadCollisionFit } {
  const triangles = Math.max(0, Math.floor(Number(triangleCount) || 0));
  const span = Math.abs(Number(meshSpan) || 0);
  const size: Vec3 = meshSize || [span, span, span];
  const center: Vec3 = meshCenter || [0, 0, 0];
  const target = targetSize || (targetSpan != null ? [targetSpan, targetSpan, targetSpan] : null);
  const fit = cadCollisionFit(size, center, target);
  if (triangles < 1 || !fit.ready) return { ready: false, scale: 1, fit: { ...fit, ready: false, scale: 1, rotation: IDENTITY, translation: [0, 0, 0] } };
  return { ready: true, scale: fit.scale, fit };
}
