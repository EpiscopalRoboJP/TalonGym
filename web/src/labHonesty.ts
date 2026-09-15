import type { CameraSensorSpec } from "./api";

export const DRIVE_EXPORT_NOTE =
  "Java is drive-only Road Runner 1.0. Mechanism timeline is comments (t / verb / actuator), not a complete AUTO — intake and launch still need team OpMode code.";

export function namedQueues(queues?: Record<string, string[]>): string {
  const entries = Object.entries(queues || {});
  if (!entries.length) return "empty";
  return entries.map(([name, items]) => `${name}: ${items.length ? items.join(" ") : "empty"}`).join(" · ");
}

/** URL replay id always wins over a later list fetch. */
export function resolveReplayId(urlId: string | undefined, active: string, fallbackId?: string): string {
  if (urlId) return urlId;
  if (active) return active;
  return fallbackId || "";
}

export function parseJsonDocument<T>(text: string): { ok: true; value: T } | { ok: false; error: string } {
  try {
    return { ok: true, value: JSON.parse(text) as T };
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return { ok: false, error: message };
  }
}

export function pickCameraSensor(design?: { sensors?: CameraSensorSpec[] } | null): CameraSensorSpec | undefined {
  const sensors = design?.sensors || [];
  return sensors.find((sensor) => sensor.kind === "apriltag_camera") || sensors.find((sensor) => (sensor.fovDeg ?? 0) > 0);
}

export function upsertCameraSensor(
  sensors: CameraSensorSpec[],
  partial: { fovDeg?: number; rangeIn?: number },
): CameraSensorSpec[] {
  const idx = sensors.findIndex((s) => s.kind === "apriltag_camera");
  if (idx >= 0) {
    const next = sensors.slice();
    next[idx] = { ...next[idx], ...partial };
    return next;
  }
  return [
    ...sensors,
    {
      id: "apriltag_camera",
      kind: "apriltag_camera",
      fovDeg: partial.fovDeg ?? 70,
      rangeIn: partial.rangeIn ?? 96,
      poseOnRobot: { x: 0, y: 0, z: 8, headingDeg: 0 },
    },
  ];
}
