import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const PRESET_ID = "e2e_catalog_gobilda_mecanum";
const LEFTOVER_COPY = "mecanum_biobuzz_4cap_copy";
const DEFAULT_ROBOT = "mecanum_biobuzz_4cap";
const PLACE_SKU = "5203-2402-0019";
const INVALID_SKU = "REV-41-1432";
const SHOT_DIR = path.join("test-results", "robot-builder");

const VIEWPORTS = [
  { name: "desktop-1920", width: 1920, height: 1080 },
  { name: "laptop-1366", width: 1366, height: 768 },
  { name: "narrow-1024", width: 1024, height: 768 },
] as const;

async function shot(page: Page, name: string) {
  await mkdir(SHOT_DIR, { recursive: true });
  await page.evaluate(() => document.fonts.ready);
  const dest = path.join(SHOT_DIR, name);
  await page.screenshot({ path: dest, fullPage: false, animations: "disabled" });
  return dest;
}

async function setReactInput(page: Page, testId: string, value: string) {
  const box = page.getByTestId(testId);
  await box.evaluate((el, v) => {
    const input = el as HTMLInputElement;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
    setter?.call(input, v);
    input.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: v, inputType: v ? "insertText" : "deleteContentBackward" }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }, value);
  await expect(box).toHaveValue(value);
}

async function clearCatalogSearch(page: Page) {
  const clearBtn = page.getByTestId("catalog-search-clear");
  if (await clearBtn.count()) await clearBtn.click();
  else await setReactInput(page, "catalog-search", "");
  await expect(page.getByTestId("catalog-search")).toHaveValue("");
}

function catalogItems(page: Page) {
  return page.getByTestId("catalog-list").locator("li");
}

async function openParts(page: Page) {
  if (!(await page.getByTestId("catalog-panel").isVisible().catch(() => false))) {
    await page.getByTestId("open-parts").click();
  }
  await expect(page.getByTestId("catalog-panel")).toBeVisible();
  await expect(page.getByTestId("catalog-list")).toBeVisible();
}

async function closeParts(page: Page) {
  if (await page.getByTestId("catalog-panel").isVisible().catch(() => false)) {
    await page.getByLabel("Close parts").click();
  }
}

async function openSide(page: Page, name: "assembly" | "inspector" | "validation") {
  const expected = name === "assembly" ? "Assembly" : name === "inspector" ? "Inspector" : "Robot check";
  const heading = page.locator(".side-drawer-head strong");
  const current = (await heading.allTextContents())[0]?.trim();
  if (current !== expected) {
    const buttonName = name === "validation" ? /^(Check|Problems)/ : new RegExp(`^${expected}$`);
    await page.getByRole("button", { name: buttonName }).click();
  }
  await expect(page.getByTestId("builder-side-drawer")).toBeVisible();
  await expect(heading).toHaveText(expected);
}

async function deleteTestRobots(request: APIRequestContext) {
  await request.delete(`/api/v1/presets/robot/${PRESET_ID}`);
  await request.delete(`/api/v1/presets/robot/${PRESET_ID}/draft`);
  await request.delete(`/api/v1/presets/robot/${LEFTOVER_COPY}`);
  await request.delete(`/api/v1/presets/robot/${LEFTOVER_COPY}/draft`);
  await request.delete(`/api/v1/presets/robot/${DEFAULT_ROBOT}/draft`);
  await request.delete("/api/v1/catalog/parts/1207-0001-0001/cache");
}

async function assertNotCovered(page: Page, testId: string) {
  const ok = await page.getByTestId(testId).evaluate((el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const x = r.left + r.width / 2;
    const y = r.top + r.height / 2;
    const top = document.elementFromPoint(x, y);
    return Boolean(top && (el === top || el.contains(top)));
  });
  expect(ok, `${testId} is covered by another layer`).toBe(true);
}

async function assertLayout(page: Page) {
  const viewport = page.viewportSize();
  expect(viewport).toBeTruthy();
  const overflowX = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflowX, "horizontal page overflow").toBeLessThanOrEqual(2);
  await expect(page.getByTestId("builder-toolbar")).toBeVisible();
  await expect(page.getByTestId("new-drivebase")).toBeVisible();
  await expect(page.getByTestId("save-robot")).toBeVisible();
  await expect(page.getByTestId("save-as")).toBeVisible();
  await expect(page.getByTestId("open-parts")).toBeVisible();
  await expect(page.getByTestId("builder-viewport")).toBeVisible();
  await expect(page.getByTestId("orientation-cube")).toBeVisible();
  await assertNotCovered(page, "open-parts");
  await assertNotCovered(page, "new-drivebase");
  await assertNotCovered(page, "save-robot");
  await openParts(page);
  await expect(page.getByTestId("catalog-panel")).toBeVisible();
  await expect(page.getByTestId("catalog-cache-all")).toBeVisible();
  await assertNotCovered(page, "catalog-search");
  await assertNotCovered(page, "catalog-cache-all");
  const toolbar = await page.getByTestId("builder-toolbar").boundingBox();
  expect(toolbar).toBeTruthy();
  expect(toolbar!.y).toBeGreaterThanOrEqual(0);
  expect(toolbar!.y + 24).toBeLessThanOrEqual((viewport?.height || 0) + 1);
  expect(toolbar!.x + Math.min(toolbar!.width, 80)).toBeLessThanOrEqual((viewport?.width || 0) + 2);
  const stage = page.getByTestId("builder-viewport");
  const canvas = stage.locator("canvas");
  await expect(canvas).toBeVisible();
  const stageBox = await stage.boundingBox();
  const canvasBox = await canvas.boundingBox();
  expect(stageBox && canvasBox).toBeTruthy();
  expect(canvasBox!.width, "canvas width").toBeGreaterThan(180);
  expect(canvasBox!.height, "canvas height").toBeGreaterThan(140);
  expect(Math.abs(canvasBox!.width - stageBox!.width)).toBeLessThan(8);
  expect(Math.abs(canvasBox!.height - stageBox!.height)).toBeLessThan(8);
  for (const testId of ["catalog-panel", "orientation-cube"]) {
    const box = await page.getByTestId(testId).boundingBox();
    expect(box, `${testId} bounds`).toBeTruthy();
    expect(box!.x).toBeGreaterThanOrEqual(-1);
    expect(box!.y).toBeGreaterThanOrEqual(-1);
    expect(box!.x + box!.width).toBeLessThanOrEqual((viewport?.width || 0) + 1);
    expect(box!.y + box!.height).toBeLessThanOrEqual((viewport?.height || 0) + 1);
  }
  const list = page.getByTestId("catalog-list");
  const catalogScroll = await list.evaluate((el) => ({ height: el.clientHeight, scroll: el.scrollHeight, overflowY: getComputedStyle(el).overflowY }));
  expect(catalogScroll.height).toBeGreaterThan(80);
  expect(["auto", "scroll", "overlay"]).toContain(catalogScroll.overflowY);
  if (catalogScroll.scroll > catalogScroll.height + 8) {
    await catalogItems(page).last().scrollIntoViewIfNeeded();
    await expect(catalogItems(page).last()).toBeVisible();
    await catalogItems(page).first().scrollIntoViewIfNeeded();
  }
}

async function waitForHit(page: Page, id: string) {
  await expect
    .poll(async () =>
      page.evaluate((ident) => {
        const hit = window.__talonBuilder?.project(ident);
        const canvas = document.querySelector("[data-testid='builder-viewport'] canvas");
        if (!hit || !canvas) return false;
        const rect = canvas.getBoundingClientRect();
        return hit.x >= rect.left + 8 && hit.x <= rect.right - 8 && hit.y >= rect.top + 8 && hit.y <= rect.bottom - 8;
      }, id),
    )
    .toBe(true);
  return page.evaluate((ident) => window.__talonBuilder?.project(ident) || null, id);
}

async function instantiateRecipe(page: Page, recipeId = "gobilda_mecanum") {
  await closeParts(page);
  await page.getByTestId("new-drivebase").click();
  await expect(page.getByTestId("drivebase-wizard")).toBeVisible();
  await page.locator("#recipe").selectOption(recipeId);
  await page.getByTestId("instantiate-recipe").click();
  await openSide(page, "assembly");
  await expect(page.getByTestId("assembly-instance-left_rail")).toBeVisible();
}

async function instantiateGobilda(page: Page) {
  await instantiateRecipe(page, "gobilda_mecanum");
  await expect(page.getByTestId("assembly-instance-servo_mount")).toBeVisible();
}

async function openMountPicker(page: Page) {
  const hasAssembly = await page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).length > 0);
  if (hasAssembly) {
    await closeParts(page);
    const parentId = await page.evaluate(() => {
      const ids = Object.keys(window.__talonBuilder?.poses || {});
      return ids.includes("left_rail") ? "left_rail" : ids[0];
    });
    const alreadySelected = await page.evaluate(
      (id) => (window.__talonBuilder?.mounts || []).some((mount) => mount.instanceId === id),
      parentId,
    );
    if (!alreadySelected) {
      await openSide(page, "assembly");
      await page.getByTestId(`assembly-instance-${parentId}`).click();
      await page.getByLabel("Close side panel").click();
    }
    await expect.poll(async () => page.evaluate((id) => (window.__talonBuilder?.mounts || []).filter((mount) => mount.instanceId === id).length, parentId)).toBeGreaterThan(0);
    const mount = await page.evaluate(
      (id) => (window.__talonBuilder?.mounts || []).find((row) => row.instanceId === id && !row.occupied) || null,
      parentId,
    );
    expect(mount).toBeTruthy();
    await page.mouse.click(mount!.x, mount!.y);
    await expect(page.getByTestId("catalog-panel")).toBeVisible();
  } else {
    await openParts(page);
  }
}

async function startCatalogPlace(page: Page, sku: string) {
  await openMountPicker(page);
  await setReactInput(page, "catalog-search", sku);
  const partBtn = page.getByTestId(`catalog-part-${sku}`);
  await expect(partBtn).toBeVisible();
  await partBtn.scrollIntoViewIfNeeded();
  await partBtn.click();
  await page.getByTestId("catalog-place-selected").click();
  await expect(page.getByTestId("builder-viewport")).toHaveAttribute("data-placing", "true");
}

async function waitForVisibleIds(page: Page, ids: string[]) {
  await expect
    .poll(async () => {
      const visible = await page.evaluate(() => window.__talonBuilder?.visibleIds || []);
      return ids.filter((id) => visible.includes(id)).length;
    })
    .toBe(ids.length);
}

const RECIPE_STRUCTURE: Record<string, string[]> = {
  gobilda_mecanum: ["left_rail", "front_rail", "right_rail", "rear_rail", "plate_center", "hub_center", "bracket_fl", "bracket_fr", "motor_fl", "wheel_fl"],
  gobilda_tank: ["left_rail", "front_rail", "right_rail", "rear_rail", "plate_center", "bracket_fl", "motor_fl", "wheel_fl"],
  rev_mecanum: ["left_rail", "front_rail", "right_rail", "rear_rail", "corner_fl", "up_fl", "bracket_fl", "motor_fl", "wheel_fl"],
  rev_tank: ["left_rail", "front_rail", "right_rail", "rear_rail", "corner_fl", "up_fl", "motor_fl", "wheel_fl"],
};

test.describe("Robot Builder catalog assembly", () => {
  test.beforeEach(async ({ request }) => {
    await deleteTestRobots(request);
  });

  test.afterEach(async ({ request }) => {
    await deleteTestRobots(request);
  });

  test("catalog stays complete through search, recipe instantiate, save, and reload", async ({ page, request }) => {
    test.setTimeout(180_000);
    await page.setViewportSize({ width: 1920, height: 1080 });
    const pageErrors: string[] = [];
    const catalogHttpFailures: string[] = [];
    page.on("pageerror", (err) => pageErrors.push(err.message));
    page.on("response", (res) => {
      if (res.url().includes("/api/v1/catalog") && res.status() >= 400) {
        if (res.url().includes("/download") && (res.status() === 409 || res.status() === 503)) return;
        if (/\/catalog\/cache\/all\/?(\?|$)/.test(res.url()) && (res.status() === 404 || res.status() === 501)) return;
        catalogHttpFailures.push(`${res.status()} ${res.url()}`);
      }
    });

    await page.goto("/build/robot");
    await expect(page.getByTestId("new-drivebase")).toBeVisible();
    await openParts(page);
    await expect.poll(async () => catalogItems(page).count()).toBeGreaterThanOrEqual(40);
    await assertLayout(page);

    const allCount = await catalogItems(page).count();
    await expect(page.getByTestId("catalog-list")).not.toContainText(" · missing");
    await expect(page.getByTestId(`catalog-thumb-${PLACE_SKU}`)).toHaveCount(1);
    await expect(page.getByTestId(`catalog-thumb-${PLACE_SKU}`)).toHaveAttribute("data-preview", /cad|proxy/);
    await setReactInput(page, "catalog-search", PLACE_SKU);
    await expect(page.getByTestId(`catalog-thumb-${PLACE_SKU}`)).toBeVisible();
    await expect(page.getByTestId(`catalog-thumb-${PLACE_SKU}`)).toHaveAttribute("data-preview", /cad|proxy/);
    await clearCatalogSearch(page);
    await catalogItems(page).last().scrollIntoViewIfNeeded();
    await expect(catalogItems(page).last()).toBeVisible();
    const catalogShot = await shot(page, "01-catalog-all.png");

    await setReactInput(page, "catalog-search", "mecanum");
    await expect.poll(async () => catalogItems(page).count()).toBeLessThan(allCount);
    await clearCatalogSearch(page);
    await expect.poll(async () => catalogItems(page).count()).toBe(allCount);

    await page.getByRole("button", { name: "goBILDA", exact: true }).click();
    await expect.poll(async () => catalogItems(page).count()).toBeGreaterThan(5);
    const gobildaShot = await shot(page, "02-catalog-gobilda.png");

    await page.getByRole("button", { name: "REV", exact: true }).click();
    await expect.poll(async () => catalogItems(page).count()).toBeGreaterThan(5);
    const revShot = await shot(page, "03-catalog-rev.png");
    await page.getByRole("button", { name: "All brands", exact: true }).click();
    await expect.poll(async () => catalogItems(page).count()).toBe(allCount);

    const cacheSku = PLACE_SKU;
    await setReactInput(page, "catalog-search", cacheSku);
    const cacheBtn = page.getByTestId(`catalog-cache-${cacheSku}`);
    await expect(cacheBtn).toBeVisible();
    await cacheBtn.click();
    await expect(page.getByTestId(`catalog-cache-state-${cacheSku}`)).toContainText(/CAD caching|CAD cached|CAD extra|CAD failed|pip install|obtain CAD|STEP|trimesh/i);
    if (await page.getByTestId("catalog-cache-error").isVisible()) {
      await expect(page.getByTestId("catalog-cache-error")).toContainText(/CAD extra|pip install|failed|download|manufacturer|trimesh|STEP|convert/i);
    }
    await clearCatalogSearch(page);
    await expect.poll(async () => catalogItems(page).count()).toBeGreaterThanOrEqual(40);

    await instantiateGobilda(page);
    await openSide(page, "assembly");
    await expect(page.getByTestId("assembly-instance-motor_fl")).toBeVisible();
    await waitForVisibleIds(page, RECIPE_STRUCTURE.gobilda_mecanum);
    await expect(page.getByTestId("assembly-instance-wheel_fl")).toBeVisible();
    await page.getByTestId("assembly-chassis").click();
    await openSide(page, "inspector");
    await expect(page.locator("#track")).toHaveValue(/^-?\d+(\.\d{1,3})?$/);
    await expect(page.locator("#wheelbase")).toHaveValue(/^-?\d+(\.\d{1,3})?$/);
    await page.waitForTimeout(1200);
    expect((await request.get(`/api/v1/presets/robot/${DEFAULT_ROBOT}/draft`)).status(), "shipped robot must not receive a draft").toBe(404);
    const instanceCount = await page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).length);
    expect(instanceCount).toBeGreaterThanOrEqual(8);

    await openSide(page, "assembly");
    await page.getByTestId("assembly-instance-left_rail").click();
    await expect(page.locator("#inst-sku")).toHaveValue(/.+/);
    const inspectShot = await shot(page, "04-recipe-inspected.png");
    await shot(page, "07-structure-gobilda-mecanum.png");

    await openSide(page, "validation");
    await page.getByTestId("confirm-inference").click();
    await expect(page.getByTestId("inference-dialog")).toBeVisible();
    await page.getByTestId("confirm-inference-submit").click();
    await expect(page.getByTestId("inference-dialog")).toHaveCount(0);

    await expect(page.getByTestId("save-as")).toBeEnabled();
    await page.getByTestId("save-as").click();
    await setReactInput(page, "save-as-id", PRESET_ID);
    await page.getByTestId("save-as-confirm").click();
    await expect(page.getByTestId("app-toast").filter({ hasText: `Created robot preset ${PRESET_ID}` })).toBeVisible();
    await expect(page.getByLabel("Robot preset").locator(`option[value="${PRESET_ID}"]`)).toHaveCount(1);
    await expect(page.getByLabel("Robot preset")).toHaveValue(PRESET_ID);
    await openSide(page, "assembly");
    await expect(page.getByTestId("assembly-instance-left_rail")).toBeVisible();
    const savedShot = await shot(page, "05-saved.png");

    await page.reload();
    await openSide(page, "assembly");
    await expect(page.getByLabel("Robot preset").locator(`option[value="${PRESET_ID}"]`)).toHaveCount(1);
    await page.getByLabel("Robot preset").selectOption(DEFAULT_ROBOT);
    await expect(page.getByTestId("assembly-instance-left_rail")).toHaveCount(0);
    await page.getByLabel("Robot preset").selectOption(PRESET_ID);
    await expect(page.getByLabel("Robot preset")).toHaveValue(PRESET_ID);
    await expect(page.getByTestId("assembly-instance-left_rail")).toBeVisible();
    await expect(page.getByTestId("assembly-instance-motor_fl")).toBeVisible();
    await expect(page.getByTestId("assembly-instance-wheel_fl")).toBeVisible();
    expect(await page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).length)).toBe(instanceCount);
    await openParts(page);
    await expect(page.getByTestId("catalog-list").locator("li")).toHaveCount(allCount);
    const reloadShot = await shot(page, "06-reloaded.png");

    expect(catalogHttpFailures, `catalog HTTP failures: ${catalogHttpFailures.join("; ")}`).toEqual([]);
    expect(pageErrors, `page errors: ${pageErrors.join("; ")}`).toEqual([]);
    expect([catalogShot, gobildaShot, revShot, inspectShot, savedShot, reloadShot].every(Boolean)).toBeTruthy();
    if (cacheSku) {
      const part = await request.get(`/api/v1/catalog/parts/${encodeURIComponent(cacheSku)}`);
      if (part.ok()) {
        const body = (await part.json()) as { cache?: { state?: string } };
        if (body.cache?.state === "ready") {
          await request.delete(`/api/v1/catalog/parts/${encodeURIComponent(cacheSku)}/cache`);
        }
      }
    }
  });

  test("KSP-style click-to-place, rotate, snap, pick up, undo, and reload", async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto("/build/robot");
    await openParts(page);
    await expect.poll(async () => catalogItems(page).count()).toBeGreaterThanOrEqual(40);
    await instantiateGobilda(page);
    await openSide(page, "assembly");
    await waitForVisibleIds(page, ["left_rail", "front_rail", "right_rail", "motor_fl", "wheel_fl"]);
    const beforeCount = await page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).length);
    await page.getByTestId("assembly-instance-left_rail").click();
    await expect(page.locator(".gizmo-mode")).toBeVisible();
    await page.getByLabel("Close side panel").click();
    await expect(page.getByTestId("orientation-cube")).toBeVisible();
    await page.getByRole("button", { name: "Front", exact: true }).click();
    await page.getByTestId("keybind-toggle").click({ force: true });
    await expect(page.getByTestId("keybind-toggle")).toHaveAttribute("class", /keybind-summary/);
    await startCatalogPlace(page, PLACE_SKU);
    await expect(page.getByTestId("placement-hud")).toBeVisible();
    await expect(page.getByTestId("mate-rotation-controls")).toBeVisible();
    const rail = await waitForHit(page, "left_rail");
    expect(rail).toBeTruthy();
    await page.mouse.move(rail!.x, rail!.y, { steps: 16 });
    await page.keyboard.press("r");
    await page.keyboard.press("e");
    await page.keyboard.press("q");
    await shot(page, "08-place-ghost.png");
    await expect.poll(async () => page.getByTestId("drag-status").getAttribute("data-has-snap")).toBe("true");
    await page.keyboard.press("Enter");
    await expect(page.getByTestId("builder-viewport")).toHaveAttribute("data-placing", "false");
    await expect.poll(async () => page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).length)).toBeGreaterThan(beforeCount);
    const placedId = await page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).find((id) => id.startsWith("5203")) || "");
    expect(placedId).toBeTruthy();
    const placedFrom = await waitForHit(page, placedId);
    expect(placedFrom).toBeTruthy();
    await page.mouse.click(placedFrom!.x, placedFrom!.y);
    await expect(page.getByTestId("builder-viewport")).toHaveAttribute("data-placing", "false");
    await expect(page.locator(".gizmo-mode")).toHaveCount(0);

    await page.keyboard.press("Control+z");
    await expect.poll(async () => page.evaluate((id) => Boolean(window.__talonBuilder?.poses?.[id]), placedId)).toBe(false);
    await page.keyboard.press("Control+Shift+z");
    await expect.poll(async () => page.evaluate((id) => Boolean(window.__talonBuilder?.poses?.[id]), placedId)).toBe(true);

    await openMountPicker(page);
    await setReactInput(page, "catalog-search", INVALID_SKU);
    await expect(page.getByTestId(`catalog-part-${INVALID_SKU}`)).toHaveCount(0);
    await shot(page, "10-incompatible-hidden.png");
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("catalog-panel")).toHaveCount(0);
  });

  for (const recipeId of Object.keys(RECIPE_STRUCTURE)) {
    test(`${recipeId} renders structural catalog instances`, async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 });
      await page.goto("/build/robot");
      await openParts(page);
      await expect.poll(async () => catalogItems(page).count()).toBeGreaterThanOrEqual(40);
      await instantiateRecipe(page, recipeId);
      await openSide(page, "assembly");
      await expect.poll(async () => page.locator('[data-testid^="assembly-instance-"]').count()).toBeGreaterThanOrEqual(8);
      const expected = RECIPE_STRUCTURE[recipeId];
      for (const id of expected) {
        await expect(page.getByTestId(`assembly-instance-${id}`)).toBeVisible();
      }
      await waitForVisibleIds(page, expected);
      const visible = await page.evaluate(() => window.__talonBuilder?.visibleIds || []);
      expect(visible.length).toBeGreaterThanOrEqual(expected.length);
      await shot(page, `11-structure-${recipeId}.png`);
    });
  }

  for (const viewport of VIEWPORTS) {
    test(`${viewport.name} keeps controls reachable and click-to-place snaps`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto("/build/robot");
      await openParts(page);
      await expect.poll(async () => catalogItems(page).count()).toBeGreaterThanOrEqual(40);
      await assertLayout(page);
      await shot(page, `${viewport.name}-01-layout.png`);

      await instantiateGobilda(page);
      await openSide(page, "assembly");
      await expect.poll(async () => page.locator('[data-testid^="assembly-instance-"]').count()).toBeGreaterThanOrEqual(8);
      await waitForHit(page, "left_rail");
      await waitForVisibleIds(page, ["left_rail", "front_rail", "right_rail"]);
      await assertLayout(page);
      const beforeCount = await page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).length);

      await openParts(page);
      await setReactInput(page, "catalog-search", PLACE_SKU);
      await startCatalogPlace(page, PLACE_SKU);
      const rail = await waitForHit(page, "left_rail");
      expect(rail).toBeTruthy();
      await page.mouse.move(rail!.x, rail!.y, { steps: 16 });
      await expect(page.getByTestId("builder-viewport")).toHaveAttribute("data-placing", "true");
      await expect(page.getByTestId("placement-hud")).toBeVisible();
      await page.keyboard.press("r");
      await shot(page, `${viewport.name}-02-catalog-drag.png`);
      await expect.poll(async () => page.getByTestId("drag-status").getAttribute("data-has-snap")).toBe("true");
      await expect(page.getByTestId("drag-status")).toContainText(/Snap to/i);
      await page.keyboard.press("Enter");
      await expect(page.getByTestId("builder-viewport")).toHaveAttribute("data-placing", "false");
      await expect.poll(async () => page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).length)).toBeGreaterThan(beforeCount);
      const placedId = await page.evaluate(() => Object.keys(window.__talonBuilder?.poses || {}).find((id) => id.startsWith("5203")) || "");
      expect(placedId).toBeTruthy();

      const placedFrom = await waitForHit(page, placedId);
      expect(placedFrom).toBeTruthy();
      await page.mouse.click(placedFrom!.x, placedFrom!.y);
      await expect(page.getByTestId("builder-viewport")).toHaveAttribute("data-placing", "false");
      await expect(page.locator(".gizmo-mode")).toHaveCount(0);
      await shot(page, `${viewport.name}-03-connected-selection.png`);

      await openMountPicker(page);
      await setReactInput(page, "catalog-search", INVALID_SKU);
      await expect(page.getByTestId(`catalog-part-${INVALID_SKU}`)).toHaveCount(0);
      await shot(page, `${viewport.name}-04-incompatible-hidden.png`);
      await assertLayout(page);
    });
  }

  test("Cache all CAD shows progress, cancel, and retry from mocked jobs", async ({ page }) => {
    let tick = 0;
    let cancelled = false;
    let retried = false;
    let started = false;
    let holdRunning = false;
    const posts: { manufacturer?: string; skus?: string[] }[] = [];
    const snapshot = () => {
      const items = [
        { sku: "1207-0001-0001", state: "converting" as const, displayName: "mecanum" },
        { sku: "5203-2402-0019", state: "queued" as const, displayName: "motor" },
        { sku: "REV-31-1595", state: "skipped" as const, reason: "download_disabled", displayName: "hub" },
        { sku: "no-src", state: "skipped" as const, reason: "no_source_url", displayName: "blank" },
      ];
      if (cancelled) {
        items[1] = { ...items[1], state: "cancelled" as const };
        items[0] = { ...items[0], state: "converting" as const };
      } else if (retried) {
        items[0] = { sku: "1207-0001-0001", state: "ready" as const, displayName: "mecanum" };
        items[1] = { sku: "5203-2402-0019", state: "ready" as const, displayName: "motor" };
      } else if (!holdRunning && tick >= 1) {
        items[0] = { sku: "1207-0001-0001", state: "ready" as const, displayName: "mecanum" };
        items[1] = { sku: "5203-2402-0019", state: "failed" as const, displayName: "motor" };
      }
      const counts = { queued: 0, converting: 0, ready: 0, failed: 0, skipped: 0, cancelled: 0 };
      for (const row of items) counts[row.state] += 1;
      const state = cancelled ? "cancelled" : counts.converting || counts.queued ? "running" : "done";
      return { id: "batch-e2e", state, counts, total: items.length, concurrency: 2, items };
    };
    await page.route(/\/api\/v1\/catalog\/cache\/all(?:\/|$|\?)/, async (route) => {
      const req = route.request();
      const url = req.url();
      if (req.method() === "POST" && url.includes("/cancel")) {
        cancelled = true;
        await route.fulfill({ status: 200, contentType: "application/json", json: snapshot() });
        return;
      }
      if (req.method() === "POST" && url.includes("/retry")) {
        retried = true;
        cancelled = false;
        await route.fulfill({ status: 202, contentType: "application/json", json: snapshot() });
        return;
      }
      if (req.method() === "POST") {
        started = true;
        tick = 0;
        cancelled = false;
        retried = false;
        const body = (req.postDataJSON() as { manufacturer?: string; skus?: string[] } | null) || {};
        posts.push({ manufacturer: body.manufacturer, skus: body.skus });
        await route.fulfill({ status: 202, contentType: "application/json", json: snapshot() });
        return;
      }
      if (!started) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: { id: null, state: "idle", counts: { queued: 0, converting: 0, ready: 0, failed: 0, skipped: 0, cancelled: 0 }, total: 0, items: [] },
        });
        return;
      }
      tick += 1;
      await route.fulfill({ status: 200, contentType: "application/json", json: snapshot() });
    });

    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto("/build/robot");
    await openParts(page);
    await expect.poll(async () => catalogItems(page).count()).toBeGreaterThanOrEqual(40);
    await expect(page.getByTestId("catalog-cache-all")).toBeVisible();
    await expect(page.getByTestId("catalog-search")).toBeEnabled();
    await assertNotCovered(page, "catalog-cache-all");
    await instantiateGobilda(page);
    await openParts(page);

    await page.getByRole("button", { name: "goBILDA", exact: true }).click();
    await expect(page.getByTestId("catalog-cache-all-scope")).toHaveText("Caches goBILDA SKUs");
    holdRunning = true;
    await page.getByTestId("catalog-cache-all").click();
    await expect.poll(() => posts.at(-1)?.manufacturer).toBe("gobilda");
    expect(posts.at(-1)?.skus).toBeUndefined();
    await expect(page.getByTestId("catalog-cache-all-progress")).toBeVisible();
    await page.getByTestId("catalog-cache-all-cancel").click();
    await expect(page.getByTestId("catalog-cache-all-progress")).toHaveAttribute("data-state", /cancelled|done/);
    await page.getByRole("button", { name: "All brands", exact: true }).click();

    await setReactInput(page, "catalog-search", PLACE_SKU);
    await expect(page.getByTestId("catalog-cache-all-scope")).toHaveText("Caches the current catalog filter");
    await page.getByTestId("catalog-cache-all").click();
    await expect.poll(() => (posts.at(-1)?.skus || []).includes(PLACE_SKU)).toBe(true);
    expect(posts.at(-1)?.manufacturer).toBeUndefined();
    await expect(page.getByTestId("catalog-cache-all-progress")).toBeVisible();
    await expect(page.getByTestId("catalog-cache-all-bar")).toBeVisible();
    await expect(page.getByTestId("catalog-cache-all-counts")).toContainText(/Queued \d+ · Converting \d+ · Ready \d+ · Failed \d+ · Skipped \d+/);
    await expect(page.getByTestId("catalog-cache-all-skipped")).toContainText(/skipped \(download disabled \/ no source URL\)/);
    await expect(page.getByTestId("catalog-cache-all-cancel")).toBeVisible();
    await expect(page.getByTestId("catalog-search")).toBeEnabled();
    await startCatalogPlace(page, PLACE_SKU);
    await expect(page.getByTestId("placement-hud")).toBeVisible();
    const rail = await waitForHit(page, "left_rail");
    expect(rail).toBeTruthy();
    await page.mouse.move(rail!.x, rail!.y, { steps: 8 });
    await page.keyboard.press("r");
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("builder-viewport")).toHaveAttribute("data-placing", "false");
    await openParts(page);
    await expect(page.getByTestId("catalog-cache-all-cancel")).toBeVisible();
    await shot(page, "12-cache-all-progress.png");
    await page.getByTestId("catalog-cache-all-cancel").click();
    await expect(page.getByTestId("catalog-cache-all-progress")).toHaveAttribute("data-state", /cancelled|done/);
    await expect(page.getByTestId("catalog-cache-all-counts")).toContainText(/Skipped 2/);
    await expect(page.getByTestId("catalog-cache-all-summary")).toContainText(/Cancelled/);
    await clearCatalogSearch(page);
    holdRunning = false;
    await page.getByTestId("catalog-cache-all").click();
    await expect(page.getByTestId("catalog-cache-all-counts")).toContainText("Failed 1");
    await expect(page.getByTestId("catalog-cache-all-retry")).toBeVisible();
    await expect(page.getByTestId("catalog-cache-all-summary")).toContainText(/Complete: Cached 1 of 4 · Failed 1 · Skipped 2/);
    await page.getByTestId("catalog-cache-all-retry").click();
    await expect(page.getByTestId("catalog-cache-all-counts")).toContainText("Ready 2");
    await expect(page.getByTestId("catalog-cache-all-retry")).toHaveCount(0);
    await expect(page.getByTestId("catalog-cache-all-summary")).toContainText(/Cached 2 of 4/);
    await shot(page, "13-cache-all-retry.png");

    holdRunning = true;
    cancelled = false;
    retried = false;
    tick = 0;
    await expect(page.getByTestId("catalog-cache-all")).toBeEnabled();
    await page.getByTestId("catalog-cache-all").click();
    await expect(page.getByTestId("catalog-cache-all-progress")).toBeVisible();
    for (const viewport of VIEWPORTS) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await assertLayout(page);
      await assertNotCovered(page, "catalog-cache-all-control");
      await shot(page, `${viewport.name}-06-cache-all.png`);
    }
  });

  test("Cache required CAD lists missing SKUs on a scoring starter", async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto("/build/robot");
    const preset = page.getByLabel("Robot preset");
    await expect(preset).toBeVisible();
    const starterOption = preset.locator("option[value='gobilda_mecanum_starter']");
    if (await starterOption.count()) {
      await preset.evaluate((el, value) => {
        const select = el as HTMLSelectElement;
        const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value")?.set;
        setter?.call(select, value);
        select.dispatchEvent(new Event("input", { bubbles: true }));
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }, "gobilda_mecanum_starter");
      await expect(preset).toHaveValue("gobilda_mecanum_starter");
    }
    await openSide(page, "assembly");
    if (!(await page.getByTestId("assembly-instance-left_rail").count())) {
      await instantiateGobilda(page);
    }
    await expect(page.getByTestId("assembly-instance-left_rail")).toBeVisible();
    await expect(page.getByTestId("catalog-cache-required")).toBeVisible();
    await expect(page.getByTestId("catalog-cache-required-btn")).toBeVisible();
    await expect(page.getByTestId("catalog-cache-required-status")).toContainText(/missing official CAD|Required CAD is cached/);
    await expect(page.getByTestId("catalog-cad-counts")).toContainText(/CAD \/ .* proxy/);
    const missing = page.getByTestId("catalog-cache-required-missing");
    if (await missing.count()) {
      await expect(missing.locator("li").first()).toContainText(/CAD not cached|CAD unavailable|CAD incomplete|CAD extra|CAD download/);
    }
  });
});
