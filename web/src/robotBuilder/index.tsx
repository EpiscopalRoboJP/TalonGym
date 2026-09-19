import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getCatalogPart,
  getJson,
  listCatalogParts,
  loadRobotDraft,
  notify,
  postJson,
  presetIsShipped,
  putJson,
  robotPresetLabel,
  saveRobotDraft,
  type CatalogManufacturer,
  type CatalogPart,
  type CatalogPartSummary,
  type DefaultsBundle,
  type DrivebaseRecipe,
  type PresetMeta,
  type RobotPreset,
} from "../api";
import { Alert, Empty, Panel, Slider, formatMeasure } from "../ui";
import { AssemblyTree } from "./AssemblyTree";
import { CacheRequiredCadControl } from "./CacheRequiredCadControl";
import { CatalogPanel } from "./CatalogPanel";
import { DrivebaseWizard } from "./DrivebaseWizard";
import { InferenceDialog } from "./InferenceDialog";
import { Inspector } from "./Inspector";
import { buildInferenceReport, competitiveSaveBlocked, confirmBindings } from "./inference";
import { slugify } from "./ids";
import { instantiateRecipe, loadRecipes, uniqueInstanceId } from "./recipes";
import { applyInstancePose, createBuilderState, reduceBuilder, type BuilderAction } from "./store";
import {
  bestSnap,
  candidatesForMount,
  cycleSpin,
  ORIENTATION_SPINS,
  partFitsMount,
  type MountTarget,
} from "./snap";
import { ValidationPanel } from "./ValidationPanel";
import { BuilderViewport } from "./Viewport";
import { childSubtree, emptyAssembly } from "./assemblyMath";
import { pointerPose } from "./pointer";
import { validateAssembly } from "./validation";

function SaveAsDialog({
  value,
  saving,
  onChange,
  onCancel,
  onSave,
}: {
  value: string;
  saving: boolean;
  onChange: (value: string) => void;
  onCancel: () => void;
  onSave: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
    return () => {
      if (dialog?.open) dialog.close();
    };
  }, []);
  return (
    <dialog ref={ref} className="modal save-as-modal" onCancel={onCancel}>
      <Panel title="Save robot as a new design" sub="Shipped robots remain unchanged.">
        <div className="stack save-as-body">
          <label className="field" htmlFor="save-as-dialog-id">
            <span>Design ID</span>
            <input
              id="save-as-dialog-id"
              data-testid="save-as-id"
              autoFocus
              value={value}
              placeholder="my_robot"
              onChange={(event) => onChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") onSave();
              }}
            />
          </label>
          <div className="row end">
            <button type="button" className="btn" onClick={onCancel}>
              Cancel
            </button>
            <button type="button" className="btn primary" data-testid="save-as-confirm" disabled={saving} onClick={onSave}>
              {saving ? "Saving…" : "Save new design"}
            </button>
          </div>
        </div>
      </Panel>
    </dialog>
  );
}

function useHistoryReducer() {
  const history = useRef({ past: [] as { doc: RobotPreset }[], future: [] as { doc: RobotPreset }[] });
  const stateRef = useRef(createBuilderState());
  const [state, setState] = useState(createBuilderState());
  const dispatch = useCallback((action: BuilderAction) => {
    const result = reduceBuilder(stateRef.current, history.current, action);
    history.current = result.history;
    const next = {
      ...result.state,
      canUndo: result.history.past.length > 0,
      canRedo: result.history.future.length > 0,
    };
    stateRef.current = next;
    setState(next);
  }, []);
  return { state, dispatch, getState: () => stateRef.current };
}

export function RobotBuilderPage() {
  const { state, dispatch, getState } = useHistoryReducer();
  const [list, setList] = useState<PresetMeta[]>([]);
  const [id, setId] = useState("gobilda_mecanum_starter");
  const [saveAsId, setSaveAsId] = useState("");
  const [saving, setSaving] = useState(false);
  const [errs, setErrs] = useState<string[]>([]);
  const [wizard, setWizard] = useState(false);
  const [recipes, setRecipes] = useState<DrivebaseRecipe[]>([]);
  const [catalog, setCatalog] = useState<CatalogPartSummary[]>([]);
  const [catalogStatus, setCatalogStatus] = useState<"loading" | "ok" | "error">("loading");
  const [catalogError, setCatalogError] = useState("");
  const [query, setQuery] = useState("");
  const [catalogEpoch, setCatalogEpoch] = useState(0);
  const [manufacturer, setManufacturer] = useState<CatalogManufacturer | "">("");
  const [tag, setTag] = useState("");
  const catalogReq = useRef(0);
  const [partCache, setPartCache] = useState<Record<string, CatalogPart>>({});
  const [previewFlywheel, setPreviewFlywheel] = useState(1);
  const [previewHood, setPreviewHood] = useState(0.6);
  const [replacing, setReplacing] = useState(false);
  const [partsOpen, setPartsOpen] = useState(false);
  const [sidePanel, setSidePanel] = useState<"assembly" | "inspector" | "validation" | null>("assembly");
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [mateMount, setMateMount] = useState<MountTarget | null>(null);
  const [compatibleSkus, setCompatibleSkus] = useState<Set<string> | null | undefined>(undefined);
  const userPickedRobot = useRef(false);

  const doc = state.doc;

  useEffect(() => {
    getJson<PresetMeta[]>("/presets/robot").then(setList);
    getJson<DefaultsBundle>("/defaults")
      .then((d) => {
        if (d.robotId && !userPickedRobot.current) setId(d.robotId);
      })
      .catch(() => undefined);
    void loadRecipes().then(setRecipes);
  }, []);

  useEffect(() => {
    const requestId = ++catalogReq.current;
    const handle = window.setTimeout(() => {
      setCatalogStatus("loading");
      void listCatalogParts({ q: query || undefined, manufacturer: manufacturer || undefined, tag: tag || undefined })
        .then((res) => {
          if (requestId !== catalogReq.current) return;
          setCatalog(res.parts);
          setCatalogStatus("ok");
          setCatalogError("");
        })
        .catch((err) => {
          if (requestId !== catalogReq.current) return;
          setCatalog([]);
          setCatalogStatus("error");
          setCatalogError(err instanceof Error ? err.message : String(err));
        });
    }, 120);
    return () => window.clearTimeout(handle);
  }, [query, manufacturer, tag, catalogEpoch]);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    void (async () => {
      const draft = presetIsShipped({ id }) ? null : await loadRobotDraft<RobotPreset>(id);
      const d = draft || (await getJson<RobotPreset>(`/presets/robot/${id}`));
      if (cancelled) return;
      delete (d as { _kind?: string })._kind;
      dispatch({ type: "hydrate", doc: d });
      setErrs([]);
      setSaveAsId("");
    })();
    return () => {
      cancelled = true;
    };
  }, [id, dispatch]);

  useEffect(() => {
    if (!doc?.assembly) return;
    const skus = [...new Set(doc.assembly.instances.map((row) => row.sku))];
    let cancelled = false;
    void Promise.all(skus.map((sku) => getCatalogPart(sku).catch(() => null))).then((rows) => {
      if (cancelled) return;
      setPartCache((prev) => {
        const next = { ...prev };
        for (const row of rows) {
          if (row) next[row.sku] = row;
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [doc?.assembly, catalogEpoch]);

  const partsByInstance = useMemo(() => {
    const out: Record<string, CatalogPart> = {};
    for (const instance of doc?.assembly?.instances || []) {
      const part = partCache[instance.sku];
      if (part) out[instance.id] = part;
    }
    return out;
  }, [doc?.assembly, partCache]);

  const validation = useMemo(() => {
    if (!doc?.assembly?.instances.length) {
      return { blocking: [], warnings: [], poses: {} };
    }
    return validateAssembly(doc.assembly, partsByInstance);
  }, [doc?.assembly, partsByInstance]);

  useEffect(() => {
    if (!doc || !state.dirty) return;
    if (presetIsShipped(list.find((row) => row.id === doc.id) || { id: doc.id })) return;
    const handle = window.setTimeout(() => {
      void saveRobotDraft(doc.id, doc);
    }, 800);
    return () => window.clearTimeout(handle);
  }, [doc, state.dirty, list]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const typing = (e.target as HTMLElement | null)?.closest("input, textarea, select, [contenteditable='true']");
      if (typing) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        dispatch({ type: e.shiftKey ? "redo" : "undo" });
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();
        dispatch({ type: "redo" });
      }
      if (e.key.toLowerCase() === "r" && state.drag) {
        e.preventDefault();
        const spinIndex = cycleSpin(state.drag.spinIndex);
        const part = partCache[state.drag.sku];
        const assembly = doc?.assembly || emptyAssembly();
        const exclude = new Set(state.drag.sourceInstanceId ? childSubtree(assembly, state.drag.sourceInstanceId) : []);
        const candidate = part
          ? mateMount && partsByInstance[mateMount.instanceId] && validation.poses[mateMount.instanceId]
            ? candidatesForMount(
                mateMount,
                partsByInstance[mateMount.instanceId],
                validation.poses[mateMount.instanceId],
                part,
                ORIENTATION_SPINS[spinIndex],
              )[0] || null
            : bestSnap(assembly, validation.poses, partsByInstance, part, state.drag.pointer, spinIndex, exclude)
          : null;
        dispatch({ type: "updateDrag", pointer: state.drag.pointer, candidate, spinIndex });
      }
      if ((e.key.toLowerCase() === "e" || e.key.toLowerCase() === "q") && state.drag) {
        e.preventDefault();
        const spinIndex = cycleSpin(state.drag.spinIndex, e.key.toLowerCase() === "q" ? -1 : 1);
        const part = partCache[state.drag.sku];
        const assembly = doc?.assembly || emptyAssembly();
        const exclude = new Set(state.drag.sourceInstanceId ? childSubtree(assembly, state.drag.sourceInstanceId) : []);
        const candidate = part
          ? mateMount && partsByInstance[mateMount.instanceId] && validation.poses[mateMount.instanceId]
            ? candidatesForMount(
                mateMount,
                partsByInstance[mateMount.instanceId],
                validation.poses[mateMount.instanceId],
                part,
                ORIENTATION_SPINS[spinIndex],
              )[0] || null
            : bestSnap(assembly, validation.poses, partsByInstance, part, state.drag.pointer, spinIndex, exclude)
          : null;
        dispatch({ type: "updateDrag", pointer: state.drag.pointer, candidate, spinIndex });
      }
      if (e.key.toLowerCase() === "g" && !state.drag && state.sel.kind === "instance") {
        e.preventDefault();
        const selectedId = state.sel.kind === "instance" ? state.sel.id : "";
        const instance = doc?.assembly?.instances.find((row) => row.id === selectedId);
        if (!instance || !doc) return;
        const connected = doc.assembly?.connections.some((row) => row.child.instanceId === instance.id);
        if (instance.id !== doc.assembly?.rootInstanceId && connected) {
          notify("Connected parts move through their mount. Detach it before free positioning.");
          return;
        }
        const pose = validation.poses[instance.id];
        const point: [number, number, number] = [pose?.x || 0, pose?.y || 0, pose?.z || 0];
        const part = partsByInstance[instance.id] || partCache[instance.sku];
        const assembly = doc.assembly || emptyAssembly();
        const exclude = new Set(childSubtree(assembly, instance.id));
        const candidate = part ? bestSnap(assembly, validation.poses, partsByInstance, part, point, 0, exclude) : null;
        dispatch({
          type: "startDrag",
          drag: { sku: instance.sku, sourceInstanceId: instance.id, spinIndex: 0, pointer: point, candidate },
        });
      }
      if (e.key.toLowerCase() === "m" && !state.drag && state.sel.kind === "instance") {
        e.preventDefault();
        setSidePanel(null);
        notify("Choose a mounting point on the selected part.");
      }
      if ((e.key === "Delete" || e.key === "Backspace") && !state.drag && state.sel.kind === "instance" && doc?.assembly) {
        e.preventDefault();
        const drop = new Set(childSubtree(doc.assembly, state.sel.id));
        dispatch({
          type: "setAssembly",
          assembly: {
            ...doc.assembly,
            instances: doc.assembly.instances.filter((row) => !drop.has(row.id)),
            connections: doc.assembly.connections.filter(
              (row) => !drop.has(row.parent.instanceId) && !drop.has(row.child.instanceId),
            ),
          },
        });
      }
      if (e.key === "Escape") {
        if (state.drag) {
          dispatch({ type: "cancelDrag" });
          if (mateMount) setPartsOpen(true);
        } else if (partsOpen) {
          setPartsOpen(false);
        } else {
          setMateMount(null);
          setCompatibleSkus(undefined);
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dispatch, state.drag, state.sel, doc, partCache, partsByInstance, validation.poses, mateMount, partsOpen]);

  async function validateAndSave(target: RobotPreset, method: "put" | "post") {
    if (target.assembly && validateAssembly(target.assembly, partsByInstance).blocking.length) {
      setErrs(["Assembly has blocking errors. Fix snaps or collisions before saving."]);
      return false;
    }
    const gate = competitiveSaveBlocked(target.functionalBindings, target.assembly?.instances.length || 0);
    if (gate) {
      setErrs([gate]);
      return false;
    }
    const res = await postJson<{ ok: boolean; errors: string[] }>("/presets/validate", { kind: "robot", document: target });
    if (!res.ok) {
      setErrs(res.errors || ["Invalid robot preset."]);
      return false;
    }
    if (method === "post") await postJson("/presets/robot", target);
    else await putJson(`/presets/robot/${target.id}`, target);
    return true;
  }

  const shipped = Boolean(
    doc && (list.length ? list.some((row) => row.id === doc.id && presetIsShipped(row)) : presetIsShipped({ id: doc.id })),
  );
  const saveBlocked = Boolean(doc?.assembly && validation.blocking.length);

  async function save() {
    if (!doc) return;
    if (shipped) {
      setErrs(["Shipped presets stay immutable. Use Save as."]);
      return;
    }
    setSaving(true);
    try {
      if (!(await validateAndSave(doc, "put"))) return;
      setErrs([]);
      dispatch({ type: "hydrate", doc });
      notify(`Saved robot preset ${doc.id}.`);
    } catch (e) {
      setErrs([e instanceof Error ? e.message : String(e)]);
    } finally {
      setSaving(false);
    }
  }

  async function saveAs() {
    if (!doc) return;
    const nid = slugify(saveAsId || `${doc.id}_copy`);
    const next = {
      ...doc,
      id: nid,
      displayName: saveAsId ? `${doc.displayName} (${nid})` : `${doc.displayName} copy`,
    };
    setSaving(true);
    try {
      if (!(await validateAndSave(next, "post"))) return;
      setErrs([]);
      notify(`Created robot preset ${nid}.`);
      userPickedRobot.current = true;
      dispatch({ type: "hydrate", doc: next });
      setList(await getJson<PresetMeta[]>("/presets/robot"));
      setId(nid);
      setSaveAsOpen(false);
    } catch (e) {
      setErrs([e instanceof Error ? e.message : String(e)]);
    } finally {
      setSaving(false);
    }
  }

  async function beginPlace(sku: string) {
    const current = state.sel;
    if (replacing && current.kind === "instance") {
      dispatch({ type: "replace", instanceId: current.id, sku });
      setReplacing(false);
      setPartsOpen(false);
      return;
    }
    try {
      const part = partCache[sku] || (await getCatalogPart(sku));
      setPartCache((prev) => ({ ...prev, [sku]: part }));
      const assembly = doc?.assembly || emptyAssembly();
      if (assembly.instances.length && !mateMount) {
        notify("Select a mounting point on the robot before choosing a part.", "error");
        return;
      }
      const candidate =
        mateMount && doc
          ? candidatesForMount(
              mateMount,
              partsByInstance[mateMount.instanceId],
              validation.poses[mateMount.instanceId],
              part,
              ORIENTATION_SPINS[0],
            )[0] || null
          : null;
      dispatch({
        type: "startDrag",
        drag: { sku, spinIndex: 0, pointer: [0, 0, 0], candidate },
      });
      setPartsOpen(false);
    } catch (err) {
      notify(err instanceof Error ? err.message : String(err), "error");
      dispatch({ type: "cancelDrag" });
    }
  }

  function beginReparent(instanceId: string, point: [number, number, number]) {
    const instance = doc?.assembly?.instances.find((row) => row.id === instanceId);
    if (!instance || !doc) return;
    const part = partsByInstance[instanceId] || partCache[instance.sku];
    const assembly = doc.assembly || emptyAssembly();
    const exclude = new Set(childSubtree(assembly, instanceId));
    const candidate = part ? bestSnap(assembly, validation.poses, partsByInstance, part, point, 0, exclude) : null;
    dispatch({
      type: "startDrag",
      drag: { sku: instance.sku, sourceInstanceId: instanceId, spinIndex: 0, pointer: point, candidate },
    });
  }

  function onPointer(point: [number, number, number]) {
    const current = getState();
    if (!current.drag || !current.doc) return;
    const part = partCache[current.drag.sku];
    if (!part) return;
    const assembly = current.doc.assembly || emptyAssembly();
    const exclude = new Set(current.drag.sourceInstanceId ? childSubtree(assembly, current.drag.sourceInstanceId) : []);
    const spin = ORIENTATION_SPINS[((current.drag.spinIndex % ORIENTATION_SPINS.length) + ORIENTATION_SPINS.length) % ORIENTATION_SPINS.length];
    const candidate =
      mateMount && partsByInstance[mateMount.instanceId] && validation.poses[mateMount.instanceId]
        ? candidatesForMount(
            mateMount,
            partsByInstance[mateMount.instanceId],
            validation.poses[mateMount.instanceId],
            part,
            spin,
          )[0] || null
        : bestSnap(assembly, validation.poses, partsByInstance, part, point, current.drag.spinIndex, exclude);
    dispatch({ type: "updateDrag", pointer: point, candidate });
  }

  function rotatePending(delta: number) {
    const current = getState();
    if (!current.drag || !current.doc) return;
    const spinIndex = cycleSpin(current.drag.spinIndex, delta);
    const part = partCache[current.drag.sku];
    if (!part) return;
    const assembly = current.doc.assembly || emptyAssembly();
    const exclude = new Set(current.drag.sourceInstanceId ? childSubtree(assembly, current.drag.sourceInstanceId) : []);
    const spin = ORIENTATION_SPINS[spinIndex];
    const candidate =
      mateMount && partsByInstance[mateMount.instanceId] && validation.poses[mateMount.instanceId]
        ? candidatesForMount(
            mateMount,
            partsByInstance[mateMount.instanceId],
            validation.poses[mateMount.instanceId],
            part,
            spin,
          )[0] || null
        : bestSnap(assembly, validation.poses, partsByInstance, part, current.drag.pointer, spinIndex, exclude);
    dispatch({ type: "updateDrag", pointer: current.drag.pointer, candidate, spinIndex });
  }

  function commitDrag() {
    const current = getState();
    if (!current.drag || !current.doc) return;
    const assembly = current.doc.assembly || emptyAssembly();
    const instanceId = current.drag.sourceInstanceId || uniqueInstanceId(current.drag.sku.replace(/[^a-z0-9]+/gi, "_").toLowerCase() || "part", assembly);
    if (!current.drag.candidate) {
      const isRoot = !assembly.instances.length || current.drag.sourceInstanceId === assembly.rootInstanceId;
      if (isRoot) {
        dispatch({ type: "placeRoot", instanceId, sku: current.drag.sku, pose: pointerPose(current.drag.pointer, ORIENTATION_SPINS[((current.drag.spinIndex % ORIENTATION_SPINS.length) + ORIENTATION_SPINS.length) % ORIENTATION_SPINS.length]) });
        return;
      }
      notify("No compatible mount in range.", "error");
      dispatch({ type: "cancelDrag" });
      return;
    }
    dispatch({ type: "snapPlace", instanceId, sku: current.drag.sku, candidate: current.drag.candidate });
    setMateMount(null);
    setCompatibleSkus(undefined);
  }

  async function chooseMount(target: MountTarget) {
    const parent = partsByInstance[target.instanceId];
    if (!parent) return;
    setMateMount(target);
    setCompatibleSkus(null);
    setSidePanel(null);
    setPartsOpen(true);
    const next = new Set<string>();
    for (const part of catalog) {
      if (partFitsMount(parent, target, part)) next.add(part.sku);
    }
    if (!next.size) {
      const detailed = await Promise.all(
        catalog.map(async (part) => {
          try {
            return await getCatalogPart(part.sku);
          } catch {
            return null;
          }
        }),
      );
      for (const part of detailed) {
        if (part && partFitsMount(parent, target, part)) next.add(part.sku);
      }
    }
    setCompatibleSkus(next);
  }

  const assemblySkus = useMemo(() => [...new Set((doc?.assembly?.instances || []).map((row) => row.sku))], [doc?.assembly]);
  const partsBySku = useMemo(() => {
    const out: Record<string, CatalogPart> = { ...partCache };
    for (const part of Object.values(partsByInstance)) out[part.sku] = part;
    return out;
  }, [partCache, partsByInstance]);
  const pendingPart = state.drag ? partCache[state.drag.sku] : null;
  const ghost =
    state.drag && pendingPart
      ? {
          id: "ghost",
          part: pendingPart,
          pose: state.drag.candidate?.pose || pointerPose(state.drag.pointer, ORIENTATION_SPINS[((state.drag.spinIndex % ORIENTATION_SPINS.length) + ORIENTATION_SPINS.length) % ORIENTATION_SPINS.length]),
        }
      : null;
  const hiddenIds = state.drag?.sourceInstanceId && doc?.assembly ? new Set(childSubtree(doc.assembly, state.drag.sourceInstanceId)) : undefined;
  const inference = (() => {
    if (!doc?.assembly?.instances.length) return null;
    try {
      return buildInferenceReport(doc.assembly, partsByInstance, doc.functionalBindings);
    } catch {
      return null;
    }
  })();
  const sel = state.sel;
  const selectedInstance = sel.kind === "instance" ? doc?.assembly?.instances.find((row) => row.id === sel.id) : undefined;

  if (!doc) {
    return (
      <main className="page">
        <Panel>
          <Empty title="Loading robot…" />
        </Panel>
      </main>
    );
  }

  return (
    <main className="page layout-builder">
      {saveAsOpen && (
        <SaveAsDialog
          value={saveAsId}
          saving={saving}
          onChange={setSaveAsId}
          onCancel={() => setSaveAsOpen(false)}
          onSave={() => void saveAs()}
        />
      )}
      {wizard && (
        <DrivebaseWizard
          recipes={recipes}
          onCancel={() => setWizard(false)}
          onCreate={(recipe, parameters) => {
            void instantiateRecipe(recipe, parameters)
              .then((result) => {
              const inferred = result.functionalBindings?.drivetrain;
              const trackWidthIn = formatMeasure(inferred?.trackWidthIn || doc.drivetrain.trackWidthIn);
              const wheelbaseIn = inferred?.wheelbaseIn != null ? formatMeasure(inferred.wheelbaseIn) : doc.drivetrain.wheelbaseIn;
              dispatch({
                type: "patchDoc",
                doc: {
                  ...doc,
                  schemaVersion: "1.2.0",
                  drivetrain: {
                    ...doc.drivetrain,
                    type: recipe.drivetrain,
                    ...inferred,
                    trackWidthIn,
                    ...(wheelbaseIn != null ? { wheelbaseIn } : {}),
                    ...(inferred?.wheelDiameterIn != null ? { wheelDiameterIn: formatMeasure(inferred.wheelDiameterIn) } : {}),
                  },
                  chassis: { ...doc.chassis },
                  assembly: result.assembly,
                  functionalBindings: result.functionalBindings
                    ? { ...result.functionalBindings, drivetrain: { type: recipe.drivetrain, trackWidthIn, ...(wheelbaseIn != null ? { wheelbaseIn } : {}) } }
                    : result.functionalBindings,
                },
              });
              setWizard(false);
              notify(`Instantiated ${recipe.displayName}.`);
            })
              .catch((err) => notify(err instanceof Error ? err.message : String(err), "error"));
          }}
        />
      )}
      {state.inferenceOpen && inference && (
        <InferenceDialog
          report={inference}
          onCancel={() => dispatch({ type: "setInferenceOpen", open: false })}
          onConfirm={() => dispatch({ type: "confirmInference", bindings: confirmBindings(inference, doc.drivetrain.type, doc.drivetrain.trackWidthIn, doc.drivetrain.wheelbaseIn) })}
        />
      )}
      <div className="toolbar builder-toolbar game-toolbar" data-testid="builder-toolbar">
        <div className="toolbar-cluster">
          <h1>Robot Builder</h1>
          <select
            aria-label="Robot preset"
            value={id}
            onChange={(e) => {
              userPickedRobot.current = true;
              setId(e.target.value);
            }}
          >
            {list.map((p) => (
              <option key={p.id} value={p.id}>
                {robotPresetLabel(p)}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="btn primary add-part-button"
            data-testid="open-parts"
            onClick={() => {
              setMateMount(null);
              setCompatibleSkus(undefined);
              dispatch({ type: "select", sel: { kind: "chassis" } });
              setSidePanel(null);
              setPartsOpen(true);
            }}
          >
            + Add part
          </button>
          <button type="button" className="btn" data-testid="new-drivebase" onClick={() => setWizard(true)} disabled={!recipes.length}>
            New robot
          </button>
          <button type="button" className="btn sm" aria-label="Undo" title="Undo" disabled={!state.canUndo} onClick={() => dispatch({ type: "undo" })}>
            Undo
          </button>
          <button type="button" className="btn sm" aria-label="Redo" title="Redo" disabled={!state.canRedo} onClick={() => dispatch({ type: "redo" })}>
            Redo
          </button>
        </div>
        <span className="spacer" />
        <div className="toolbar-cluster">
          <button type="button" className={`btn ${sidePanel === "assembly" ? "primary" : ""}`} data-testid="toggle-assembly" onClick={() => setSidePanel(sidePanel === "assembly" ? null : "assembly")}>
            Assembly
          </button>
          <button type="button" className={`btn ${sidePanel === "inspector" ? "primary" : ""}`} data-testid="toggle-inspector" onClick={() => setSidePanel(sidePanel === "inspector" ? null : "inspector")}>
            Inspector
          </button>
          <button type="button" className={`btn ${sidePanel === "validation" ? "primary" : ""}`} data-testid="toggle-validation" onClick={() => setSidePanel(sidePanel === "validation" ? null : "validation")}>
            {validation.blocking.length ? `Problems (${validation.blocking.length})` : "Check"}
          </button>
          <button type="button" className="btn" data-testid="save-as" onClick={() => setSaveAsOpen(true)} disabled={saving || saveBlocked}>
            Save as
          </button>
          <button type="button" className="btn primary" data-testid="save-robot" onClick={() => void save()} disabled={saving || shipped || saveBlocked} title={shipped ? "Shipped presets stay immutable. Use Save as." : undefined}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>

      <div className={`builder-workspace${state.drag ? " is-dragging" : ""}${sidePanel ? " has-side-panel" : ""}`}>
        <Panel className="grow builder-stage game-stage" bodyClass="panel-body flush viewport-body">
          <BuilderViewport
            doc={doc}
            parts={partsByInstance}
            poses={validation.poses}
            selectedId={sel.kind === "instance" ? sel.id : null}
            ghost={ghost}
            pendingPart={pendingPart}
            hiddenIds={hiddenIds}
            dragging={Boolean(state.drag)}
            candidate={state.drag?.candidate || null}
            spinIndex={state.drag?.spinIndex || 0}
            selectedMount={mateMount}
            previewFlywheel={previewFlywheel}
            previewHood={previewHood}
            onSelect={(instanceId) => {
              if (state.drag) return;
              dispatch({ type: "select", sel: instanceId ? { kind: "instance", id: instanceId } : { kind: "chassis" } });
              if (instanceId) setSidePanel("inspector");
            }}
            onPointer={onPointer}
            onRelease={commitDrag}
            onCancel={() => {
              dispatch({ type: "cancelDrag" });
              if (mateMount) setPartsOpen(true);
            }}
            onInstanceDown={(instanceId) => {
              if (state.drag) return;
              dispatch({ type: "select", sel: { kind: "instance", id: instanceId } });
              setSidePanel("inspector");
            }}
            onMountSelect={(target) => void chooseMount(target)}
            onRotate={rotatePending}
            onFreePose={(instanceId, pose) => {
              const current = getState().doc;
              if (current) dispatch({ type: "patchDoc", doc: applyInstancePose(current, instanceId, pose) });
            }}
          />
          <CacheRequiredCadControl skus={assemblySkus} parts={partsBySku} onRefresh={() => setCatalogEpoch((n) => n + 1)} />
        </Panel>
        {partsOpen && (
          <div className="parts-drawer-backdrop" data-testid="parts-drawer-backdrop" onMouseDown={(event) => {
            if (event.currentTarget === event.target) setPartsOpen(false);
          }}>
            <CatalogPanel
              parts={catalog}
              query={query}
              manufacturer={manufacturer}
              tag={tag}
              status={catalogStatus}
              error={catalogError}
              onQuery={setQuery}
              onManufacturer={setManufacturer}
              onTag={setTag}
              onPick={(sku) => void beginPlace(sku)}
              onCatalogRefresh={() => setCatalogEpoch((n) => n + 1)}
              replacing={replacing}
              compatibleSkus={compatibleSkus}
              mountContext={
                mateMount
                  ? `${mateMount.instanceId} · ${mateMount.mountId} [${mateMount.patternIndex.join(", ")}]`
                  : undefined
              }
              onClose={() => {
                setPartsOpen(false);
                setReplacing(false);
                setMateMount(null);
                setCompatibleSkus(undefined);
              }}
            />
          </div>
        )}
        {sidePanel && (
          <aside className="builder-side-drawer" data-testid="builder-side-drawer">
            <div className="side-drawer-head">
              <strong>{sidePanel === "assembly" ? "Assembly" : sidePanel === "inspector" ? "Inspector" : "Robot check"}</strong>
              <button type="button" className="btn icon" aria-label="Close side panel" onClick={() => setSidePanel(null)}>×</button>
            </div>
            {sidePanel === "assembly" && (
              <AssemblyTree
                assembly={doc.assembly || emptyAssembly()}
                parts={partsByInstance}
                sel={sel}
                onSelect={(nextSel) => {
                  dispatch({ type: "select", sel: nextSel });
                  if (nextSel.kind === "instance") setSidePanel("inspector");
                }}
              />
            )}
            {sidePanel === "inspector" && (
              <Inspector
            doc={doc}
            sel={sel}
            part={selectedInstance ? partsByInstance[selectedInstance.id] : undefined}
            pose={selectedInstance ? validation.poses[selectedInstance.id] : undefined}
            onPatchDoc={(next) => dispatch({ type: "patchDoc", doc: next })}
            onPatchPose={(pose) => {
              if (!selectedInstance) return;
              dispatch({
                type: "setAssembly",
                assembly: {
                  ...(doc.assembly || emptyAssembly()),
                  instances: (doc.assembly?.instances || []).map((row) => (row.id === selectedInstance.id && row.id === doc.assembly?.rootInstanceId ? { ...row, pose } : row)),
                },
              });
            }}
            onDetach={() => selectedInstance && dispatch({ type: "detach", instanceId: selectedInstance.id })}
            onPickUp={() => {
              if (!selectedInstance) return;
              const connected = doc.assembly?.connections.some((row) => row.child.instanceId === selectedInstance.id);
              if (selectedInstance.id !== doc.assembly?.rootInstanceId && connected) {
                notify("Detach this part before free positioning it.");
                return;
              }
              const pose = validation.poses[selectedInstance.id];
              beginReparent(selectedInstance.id, [pose?.x || 0, pose?.y || 0, pose?.z || 0]);
            }}
            onReplace={() => {
              setReplacing(true);
              setSidePanel(null);
              setPartsOpen(true);
            }}
            onDuplicate={() => selectedInstance && dispatch({ type: "duplicate", instanceId: selectedInstance.id })}
            onMirror={() => selectedInstance && dispatch({ type: "mirror", instanceId: selectedInstance.id })}
            onPattern={() => selectedInstance && dispatch({ type: "pattern", instanceId: selectedInstance.id, count: 2 })}
            onRemove={() => {
              if (!selectedInstance || !doc.assembly) return;
              const drop = new Set(childSubtree(doc.assembly, selectedInstance.id));
              dispatch({
                type: "setAssembly",
                assembly: {
                  ...doc.assembly,
                  instances: doc.assembly.instances.filter((row) => !drop.has(row.id)),
                  connections: doc.assembly.connections.filter((row) => !drop.has(row.parent.instanceId) && !drop.has(row.child.instanceId)),
                },
              });
            }}
              />
            )}
            {sidePanel === "validation" && (
              <div className="side-drawer-scroll">
                <ValidationPanel result={validation} />
                <button type="button" className="btn block" data-testid="confirm-inference" onClick={() => dispatch({ type: "setInferenceOpen", open: true })} disabled={!inference}>
                  Confirm inferred mechanisms
                </button>
                {doc.piecePath && (
                  <Panel title="Launch preview" bodyClass="panel-body">
                    <Slider id="prev-fly" label="Preview flywheel" value={previewFlywheel} min={0} max={1} step={0.05} unit="" onChange={setPreviewFlywheel} />
                    <Slider id="prev-hood" label="Preview hood" value={previewHood} min={0} max={1} step={0.05} unit="" onChange={setPreviewHood} />
                  </Panel>
                )}
                {errs.length > 0 && (
                  <Alert kind="bad">
                    <b>Could not save</b>
                    <ul>{errs.map((error) => <li key={error}>{error}</li>)}</ul>
                  </Alert>
                )}
              </div>
            )}
          </aside>
        )}
      </div>
    </main>
  );
}
