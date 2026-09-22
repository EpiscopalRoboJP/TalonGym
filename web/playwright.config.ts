import { defineConfig, devices } from "@playwright/test";

const lab = process.env.TALONGYM_LAB_URL || "http://127.0.0.1:8765";
const python = process.env.TALONGYM_PYTHON || (process.platform === "win32" ? "../.venv/Scripts/python.exe" : "../.venv/bin/python");
const quotedPython = `"${python.replaceAll('"', '\\"')}"`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  timeout: 90_000,
  actionTimeout: 10_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: lab,
    trace: "retain-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: {
    command: `${quotedPython} -m talongym lab --host 127.0.0.1 --port 8765`,
    url: `${lab}/api/v1/health`,
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
