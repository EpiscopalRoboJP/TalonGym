import { useEffect, useRef, useState } from "react";
import type { DrivebaseRecipe, DrivebaseRecipeParameters } from "../api";
import { Field, Panel, Switch } from "../ui";

export function DrivebaseWizard({
  recipes,
  onCancel,
  onCreate,
}: {
  recipes: DrivebaseRecipe[];
  onCancel: () => void;
  onCreate: (recipe: DrivebaseRecipe, parameters: DrivebaseRecipeParameters) => void;
}) {
  const [recipeId, setRecipeId] = useState(recipes[0]?.id || "");
  const recipe = recipes.find((row) => row.id === recipeId) || recipes[0];
  const [lengthSku, setLengthSku] = useState(recipe?.defaultParameters.lengthSku || recipe?.lengthSkus[0] || "");
  const [widthSku, setWidthSku] = useState(recipe?.defaultParameters.widthSku || recipe?.widthSkus[0] || "");
  const [motorSku, setMotorSku] = useState(recipe?.defaultParameters.motorSku || "");
  const [wheelSku, setWheelSku] = useState(recipe?.defaultParameters.wheelSku || "");
  const [cartridgeSku, setCartridgeSku] = useState(recipe?.defaultParameters.cartridgeSku || "");
  const [includeElectronics, setIncludeElectronics] = useState(Boolean(recipe?.defaultParameters.includeElectronics));
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const node = dialogRef.current;
    if (!node) return;
    if (!node.open) node.showModal();
    return () => {
      if (node.open) node.close();
    };
  }, []);
  if (!recipe) return null;

  function applyRecipe(next: DrivebaseRecipe) {
    setRecipeId(next.id);
    setLengthSku(next.defaultParameters.lengthSku || next.lengthSkus[0] || "");
    setWidthSku(next.defaultParameters.widthSku || next.widthSkus[0] || "");
    setMotorSku(next.defaultParameters.motorSku || "");
    setWheelSku(next.defaultParameters.wheelSku || "");
    setCartridgeSku(next.defaultParameters.cartridgeSku || "");
    setIncludeElectronics(Boolean(next.defaultParameters.includeElectronics));
  }

  return (
    <dialog ref={dialogRef} className="modal" data-testid="drivebase-wizard" onCancel={onCancel}>
      <Panel title="Guided drivebase" sub="Four complete recipes. The result stays editable in 3D.">
        <div className="stack" style={{ padding: "1rem" }}>
          <Field id="recipe" label="Recipe">
            <select
              id="recipe"
              value={recipe.id}
              onChange={(e) => {
                const next = recipes.find((row) => row.id === e.target.value);
                if (next) applyRecipe(next);
              }}
            >
              {recipes.map((row) => (
                <option key={row.id} value={row.id}>
                  {row.displayName}
                </option>
              ))}
            </select>
          </Field>
          <div className="fields two">
            <Field id="rb-len" label="Side rail / extrusion">
              <select id="rb-len" value={lengthSku} onChange={(e) => setLengthSku(e.target.value)}>
                {recipe.lengthSkus.map((sku) => (
                  <option key={sku} value={sku}>
                    {sku}
                  </option>
                ))}
              </select>
            </Field>
            <Field id="rb-wid" label="Cross rail / extrusion">
              <select id="rb-wid" value={widthSku} onChange={(e) => setWidthSku(e.target.value)}>
                {recipe.widthSkus.map((sku) => (
                  <option key={sku} value={sku}>
                    {sku}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <Field id="rb-motor" label="Motor / gearbox">
            <select id="rb-motor" value={motorSku} onChange={(e) => setMotorSku(e.target.value)}>
              {recipe.motorSkus.map((sku) => (
                <option key={sku} value={sku}>
                  {sku}
                </option>
              ))}
            </select>
          </Field>
          <Field id="rb-wheel" label="Wheel">
            <select id="rb-wheel" value={wheelSku} onChange={(e) => setWheelSku(e.target.value)}>
              {recipe.wheelSkus.map((sku) => (
                <option key={sku} value={sku}>
                  {sku}
                </option>
              ))}
            </select>
          </Field>
          {recipe.cartridgeSkus && recipe.cartridgeSkus.length > 0 && (
            <Field id="rb-cartridge" label="UltraPlanetary cartridge">
              <select id="rb-cartridge" value={cartridgeSku} onChange={(e) => setCartridgeSku(e.target.value)}>
                {recipe.cartridgeSkus.map((sku) => (
                  <option key={sku} value={sku}>
                    {sku}
                  </option>
                ))}
              </select>
            </Field>
          )}
          <Switch checked={includeElectronics} onChange={setIncludeElectronics}>
            Include optional electronics mount
          </Switch>
          <div className="row end">
            <button type="button" className="btn" onClick={onCancel}>
              Cancel
            </button>
            <button
              type="button"
              className="btn primary"
              data-testid="instantiate-recipe"
              onClick={() =>
                onCreate(recipe, {
                  lengthSku,
                  widthSku,
                  motorSku,
                  wheelSku,
                  includeElectronics,
                  ...(cartridgeSku ? { cartridgeSku } : {}),
                })
              }
            >
              Instantiate
            </button>
          </div>
        </div>
      </Panel>
    </dialog>
  );
}
