import { replayLabel, replayMetaLine, runLabel } from "./api";

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) throw new Error(message);
}

const tests: Array<[string, () => void]> = [
  ["named runs prefer the name field", () => {
    assert(runLabel({ id: "abc123", name: "flower-fix", config: {} }) === "flower-fix", "name field");
  }],
  ["config.name is a fallback", () => {
    assert(runLabel({ id: "abc123", config: { name: "from config" } }) === "from config", "config");
  }],
  ["blank names fall back to the run id", () => {
    assert(runLabel({ id: "abc123", name: "  ", config: { name: "" } }) === "abc123", "id");
  }],
  ["replay list prefers the run name", () => {
    assert(replayLabel({ id: "rep1", name: "wall-slam 262k", runId: "run1", source: "train" }) === "wall-slam 262k", "name");
  }],
  ["unnamed train replays use the run id", () => {
    assert(replayLabel({ id: "rep1", runId: "run1", source: "train" }) === "run1", "run id");
  }],
  ["demo replays keep the source label", () => {
    assert(replayLabel({ id: "rep1", source: "demo" }) === "Scripted demo", "demo");
  }],
  ["named replay meta keeps the run id", () => {
    assert(
      replayMetaLine({ id: "rep1", name: "wall-slam 262k", runId: "run1", source: "train", algo: "recurrent_ppo" }) ===
        "run1 · recurrent_ppo",
      "meta",
    );
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
  throw new Error(`${failed} runLabel test(s) failed`);
}
