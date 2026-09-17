export type ClientRectLike = {
  left: number;
  top: number;
  width: number;
  height: number;
  right?: number;
  bottom?: number;
};

export function ndcFromClientRect(clientX: number, clientY: number, rect: ClientRectLike): [number, number] | null {
  if (rect.width < 1 || rect.height < 1) return null;
  const x = (clientX - rect.left) / rect.width;
  const y = (clientY - rect.top) / rect.height;
  return [x * 2 - 1, -(y * 2 - 1)];
}

export function threeHitToFtc(x: number, y: number, z: number): [number, number, number] {
  return [x, -z, y];
}

export function clientOverRect(clientX: number, clientY: number, rect: ClientRectLike): boolean {
  const right = rect.right ?? rect.left + rect.width;
  const bottom = rect.bottom ?? rect.top + rect.height;
  return clientX >= rect.left && clientX <= right && clientY >= rect.top && clientY <= bottom;
}

export function pointerPose(pointer: [number, number, number], yawDeg = 0) {
  return { x: pointer[0], y: pointer[1], z: pointer[2], yawDeg };
}
