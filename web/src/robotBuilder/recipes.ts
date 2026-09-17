import type { DrivebaseRecipe, DrivebaseRecipeParameters, RecipeInstantiateResult, RobotAssembly } from "../api";
import { instantiateDrivebaseRecipe, listDrivebaseRecipes } from "../api";
import { nextId } from "./ids";

export const RECIPE_IDS = ["gobilda_mecanum", "gobilda_tank", "rev_mecanum", "rev_tank"] as const;

export async function loadRecipes(): Promise<DrivebaseRecipe[]> {
  const remote = await listDrivebaseRecipes();
  const recipes = remote.recipes || [];
  return RECIPE_IDS.map((id) => recipes.find((row) => row.id === id)).filter((row): row is DrivebaseRecipe => Boolean(row));
}

export async function instantiateRecipe(recipe: DrivebaseRecipe, parameters: DrivebaseRecipeParameters): Promise<RecipeInstantiateResult> {
  const result = await instantiateDrivebaseRecipe(recipe.id, { ...recipe.defaultParameters, ...parameters });
  if (!result.assembly?.instances?.length) {
    throw new Error(`Recipe ${recipe.id} did not return a complete assembly`);
  }
  return result;
}

export function uniqueInstanceId(prefix: string, assembly: RobotAssembly): string {
  return nextId(prefix, assembly.instances.map((row) => row.id));
}

export function uniqueConnectionId(prefix: string, assembly: RobotAssembly): string {
  return nextId(prefix, assembly.connections.map((row) => row.id));
}
