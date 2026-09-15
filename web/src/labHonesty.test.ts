import { namedQueues, parseJsonDocument, pickCameraSensor, resolveReplayId, upsertCameraSensor } from "./labHonesty";

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) throw new Error(message);
}

const tests: Array<[string, () => void]> = [
  ["named queues use keys, not the first Object.values slot", () => {
    assert(namedQueues(undefined) === "empty", "missing queues");
    assert(namedQueues({}) === "empty", "empty map");
    assert(namedQueues({ ramp_blue: ["A", "B"], ramp_red: [] }) === "ramp_blue: A B · ramp_red: empty", "named");
  }],
  ["deep-link replay id wins over empty active and list fallback", () => {
    assert(resolveReplayId("abc", "", "list0") === "abc", "url wins");
    assert(resolveReplayId(undefined, "", "list0") === "list0", "fallback when no url");
    assert(resolveReplayId(undefined, "kept", "list0") === "kept", "active wins over list");
    assert(resolveReplayId("", "kept", "list0") === "kept", "empty url is ignored");
  }],
  ["invalid JSON is an error object, not a throw", () => {
    const bad = parseJsonDocument("{");
    assert(!bad.ok, "should fail");
    if (!bad.ok) assert(bad.error.length > 0, "has message");
    const good = parseJsonDocument<{ id: string }>('{"id":"x"}');
    assert(good.ok && good.value.id === "x", "parses");
  }],
  ["camera sliders create a sensor when none exists", () => {
    const created = upsertCameraSensor([], { fovDeg: 55, rangeIn: 40 });
    assert(created.length === 1, "one sensor");
    assert(created[0].kind === "apriltag_camera", "april tag");
    assert(created[0].fovDeg === 55, "fov written");
    assert(created[0].rangeIn === 40, "range written");
    const updated = upsertCameraSensor(created, { fovDeg: 80 });
    assert(updated[0].fovDeg === 80 && updated[0].rangeIn === 40, "patches existing");
    const withImu = upsertCameraSensor([{ id: "imu", kind: "imu" }], { fovDeg: 45 });
    assert(withImu.length === 2 && withImu[1].kind === "apriltag_camera" && withImu[1].fovDeg === 45, "adds camera");
  }],
  ["FOV helper reads camera sensors from the robot design", () => {
    const cam = pickCameraSensor({
      sensors: [{ id: "cam", kind: "apriltag_camera", fovDeg: 62, rangeIn: 88, poseOnRobot: { x: 4, headingDeg: 10 } }],
    });
    assert(cam?.fovDeg === 62 && cam.rangeIn === 88, "wired camera");
    assert(!pickCameraSensor({ sensors: [] }), "no fake camera");
  }],
];

let failed = 0;
for (const [name, run] of tests) {
  try {
    run();
    console.log(`ok ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`not ok ${name}`);
    console.error(err);
  }
}
if (failed) {
  throw new Error(`${failed} lab honesty test(s) failed`);
}
console.log(`${tests.length} passed`);
