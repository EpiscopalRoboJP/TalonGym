import { expect, test } from "@playwright/test";

test("Train display robot and field selects update the setup preview", async ({ page }) => {
  await page.goto("/train");
  const robot = page.getByTestId("train-scene-robot");
  const field = page.getByTestId("train-scene-field");
  await expect(robot).toBeVisible();
  await expect(field).toBeVisible();
  await expect(field).toHaveValue("biobuzz_2026_field_v1");

  const formRobot = page.getByTestId("train-robot");
  await expect(formRobot).toBeVisible();
  const options = await robot.locator("option").allTextContents();
  expect(options.length).toBeGreaterThan(1);
  await expect(robot.locator("option[value='gobilda_mecanum_starter']")).toHaveCount(1);
  const nextRobot = await robot.locator("option").nth(1).getAttribute("value");
  expect(nextRobot).toBeTruthy();
  await robot.selectOption(nextRobot as string);
  await expect(formRobot).toHaveValue(nextRobot as string);
  await expect(page.getByTestId("train-scene")).toContainText(/setup preview/i);
});
