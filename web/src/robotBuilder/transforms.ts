import type { Transform3 } from "../api";

export type Vec3 = [number, number, number];
export type Mat3 = [Vec3, Vec3, Vec3];
export type Mat4 = [Vec3, Vec3, Vec3, Vec3] | number[][];

const AXIS_EPS = 1e-9;
const ALIGN_EPS = 1e-8;
export const PATTERN_MATCH_TOL_IN = 0.02;

export function vec(x = 0, y = 0, z = 0): Vec3 {
  return [x, y, z];
}

export function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function norm(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalize(a: Vec3): Vec3 {
  const n = norm(a);
  if (n < AXIS_EPS) throw new Error("mount axis must be a non-zero vector");
  return scale(a, 1 / n);
}

export function identity4(): number[][] {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

function mul3(a: number[][], b: Vec3): Vec3 {
  return [
    a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2],
    a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2],
    a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2],
  ];
}

export function planeBasis(axis: Vec3): [Vec3, Vec3] {
  const normal = normalize(axis);
  const helper: Vec3 = Math.abs(normal[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const u = normalize(cross(helper, normal));
  return [u, cross(normal, u)];
}

export function eulerMatrix(rollRad: number, pitchRad: number, yawRad: number): number[][] {
  const cr = Math.cos(rollRad);
  const sr = Math.sin(rollRad);
  const cp = Math.cos(pitchRad);
  const sp = Math.sin(pitchRad);
  const cy = Math.cos(yawRad);
  const sy = Math.sin(yawRad);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ];
}

export function matrixFromPose(pose?: Transform3 | null): number[][] {
  const row = pose || {};
  const rot = eulerMatrix(
    ((row.rollDeg || 0) * Math.PI) / 180,
    ((row.pitchDeg || 0) * Math.PI) / 180,
    ((row.yawDeg || 0) * Math.PI) / 180,
  );
  return [
    [rot[0][0], rot[0][1], rot[0][2], row.x || 0],
    [rot[1][0], rot[1][1], rot[1][2], row.y || 0],
    [rot[2][0], rot[2][1], rot[2][2], row.z || 0],
    [0, 0, 0, 1],
  ];
}

export function matrixToEuler(rotation: number[][]): [number, number, number] {
  const pitch = Math.asin(Math.min(1, Math.max(-1, -rotation[2][0])));
  const cosinePitch = Math.cos(pitch);
  if (Math.abs(cosinePitch) > 1e-8) {
    return [Math.atan2(rotation[2][1], rotation[2][2]), pitch, Math.atan2(rotation[1][0], rotation[0][0])];
  }
  return [0, pitch, Math.atan2(-rotation[0][1], rotation[1][1])];
}

export function poseFromMatrix(matrix: number[][]): Transform3 {
  const [roll, pitch, yaw] = matrixToEuler(matrix);
  return {
    x: matrix[0][3],
    y: matrix[1][3],
    z: matrix[2][3],
    rollDeg: (roll * 180) / Math.PI,
    pitchDeg: (pitch * 180) / Math.PI,
    yawDeg: (yaw * 180) / Math.PI,
  };
}

export function invertTransform(matrix: number[][]): number[][] {
  const r = [
    [matrix[0][0], matrix[1][0], matrix[2][0]],
    [matrix[0][1], matrix[1][1], matrix[2][1]],
    [matrix[0][2], matrix[1][2], matrix[2][2]],
  ];
  const t: Vec3 = [matrix[0][3], matrix[1][3], matrix[2][3]];
  const invT = scale(mul3(r, t), -1);
  return [
    [r[0][0], r[0][1], r[0][2], invT[0]],
    [r[1][0], r[1][1], r[1][2], invT[1]],
    [r[2][0], r[2][1], r[2][2], invT[2]],
    [0, 0, 0, 1],
  ];
}

export function mul4(a: number[][], b: number[][]): number[][] {
  const out = identity4();
  for (let i = 0; i < 4; i += 1) {
    for (let j = 0; j < 4; j += 1) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
    }
  }
  return out;
}

export function relativeTransform(parent: number[][], child: number[][]): number[][] {
  return mul4(invertTransform(parent), child);
}

export function transformPoint(matrix: number[][], point: Vec3): Vec3 {
  return add(mul3(matrix, point), [matrix[0][3], matrix[1][3], matrix[2][3]]);
}

export function transformDirection(matrix: number[][], direction: Vec3): Vec3 {
  return mul3(matrix, direction);
}

function skew(v: Vec3): number[][] {
  return [
    [0, -v[2], v[1]],
    [v[2], 0, -v[0]],
    [-v[1], v[0], 0],
  ];
}

function mulMat3(a: number[][], b: number[][]): number[][] {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return out;
}

function addMat3(a: number[][], b: number[][]): number[][] {
  return [
    [a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]],
    [a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]],
    [a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]],
  ];
}

function scaleMat3(a: number[][], s: number): number[][] {
  return [
    [a[0][0] * s, a[0][1] * s, a[0][2] * s],
    [a[1][0] * s, a[1][1] * s, a[1][2] * s],
    [a[2][0] * s, a[2][1] * s, a[2][2] * s],
  ];
}

export function axisAngle(axis: Vec3, thetaRad: number): number[][] {
  const n = normalize(axis);
  const k = skew(n);
  const kk = mulMat3(k, k);
  const eye = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  return addMat3(addMat3(eye, scaleMat3(k, Math.sin(thetaRad))), scaleMat3(kk, 1 - Math.cos(thetaRad)));
}

export function rotationAligning(source: Vec3, destination: Vec3): number[][] {
  const src = normalize(source);
  const dest = normalize(destination);
  const cosine = Math.min(1, Math.max(-1, dot(src, dest)));
  if (cosine > 1 - ALIGN_EPS) {
    return [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
  }
  if (cosine < -1 + ALIGN_EPS) {
    const helper: Vec3 = Math.abs(src[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    return axisAngle(normalize(cross(src, helper)), Math.PI);
  }
  const crossed = cross(src, dest);
  const k = skew(crossed);
  const kk = mulMat3(k, k);
  const eye = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  return addMat3(addMat3(eye, k), scaleMat3(kk, (1 - cosine) / dot(crossed, crossed)));
}

export function signedAngleAround(fromVec: Vec3, toVec: Vec3, axis: Vec3): number {
  const normal = normalize(axis);
  let start = sub(fromVec, scale(normal, dot(fromVec, normal)));
  let dest = sub(toVec, scale(normal, dot(toVec, normal)));
  const startN = norm(start);
  const destN = norm(dest);
  if (startN < AXIS_EPS || destN < AXIS_EPS) return 0;
  start = scale(start, 1 / startN);
  dest = scale(dest, 1 / destN);
  return Math.atan2(dot(normal, cross(start, dest)), Math.min(1, Math.max(-1, dot(start, dest))));
}

export function poseEulerRad(pose?: Transform3 | null): [number, number, number] {
  return [((pose?.pitchDeg || 0) * Math.PI) / 180, ((pose?.yawDeg || 0) * Math.PI) / 180, ((pose?.rollDeg || 0) * Math.PI) / 180];
}
