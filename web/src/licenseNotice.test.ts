import { CREDIT_LINE, LICENSE_NOTICE } from "./licenseNotice";

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) throw new Error(message);
}

const tests: Array<[string, () => void]> = [
  ["license notice names both FTC teams", () => {
    assert(LICENSE_NOTICE.includes("17986"), "team 17986");
    assert(LICENSE_NOTICE.includes("904 Robo Eagles"), "Robo Eagles");
    assert(LICENSE_NOTICE.includes("27268"), "team 27268");
    assert(LICENSE_NOTICE.includes("Talon Strike"), "Talon Strike");
    assert(CREDIT_LINE === "Built by FTC Team 17986 904 Robo Eagles and Team 27268 Talon Strike.", "credit line");
  }],
];

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
    console.log(`ok  ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`FAIL ${name}:`, err);
  }
}
if (failed) {
  throw new Error(`${failed} license notice test(s) failed`);
}
console.log(`${tests.length} passed`);
