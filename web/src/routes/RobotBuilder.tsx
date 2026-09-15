import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  deleteRobotModel,
  getJson,
  notify,
  postJson,
  putJson,
  robotPresetLabel,
  uploadRobotModel,
  type DefaultsBundle,
  type IntakeSpec,
  type LauncherSpec,
  type PoseOnRobot,
  type PresetMeta,
  type VisualOffset,
} from "../api";
import { RobotPreview } from "../scene/FieldScene";
import { theme } from "../theme";
import { Alert, Empty, Field, Icon, NumberField, Panel, Slider, Switch } from "../ui";

type RobotDoc = {
  schemaVersion: string;
  id: string;
  displayName: string;
  drivetrain: {
    type: string;
    trackWidthIn: number;
    wheelDiameterIn?: number;
    wheelbaseIn?: number;
    strafeMultiplier?: number;
  };
  chassis: {
    lengthIn: number;
    widthIn: number;
    heightIn?: number;
    massKg: number;
    collisionShape?: string;
    footprint?: { x: number; y: number }[];
  };
  motors: Record<string, number>;
  constraints: {
    maxVelInPerS: number;
    maxAccelInPerS2: number;
    maxAngVelDegPerS: number;
    maxAngAccelDegPerS2?: number;
  };
  mechanisms: {
    capacity: number;
    intakeCycleTimeS?: number;
    scoreCycleTimeS?: number;
    canIntakeWhileMoving?: boolean;
    canScoreWhileMoving?: boolean;
    launchCapable?: boolean;
    climbCapable?: boolean;
    fsmId?: string;
  };
  intakes?: IntakeSpec[];
  launchers?: LauncherSpec[];
  sensors: { id: string; kind: string; fovDeg?: number; rangeIn?: number; poseOnRobot?: PoseOnRobot }[];
  defaultActionTier: string;
  visualAsset?: string | null;
  collisionAsset?: string | null;
  visualOffset?: VisualOffset;
  [k: string]: unknown;
};

type Sel =
  | { kind: "chassis" }
  | { kind: "camera" }
  | { kind: "model" }
  | { kind: "intake"; id: string }
  | { kind: "launcher"; id: string };

function cam(doc: RobotDoc) {
  return doc.sensors?.find((s) => s.kind === "apriltag_camera") || doc.sensors?.[0];
}

function nextId(prefix: string, used: string[]) {
  let n = 1;
  while (used.includes(`${prefix}_${n}`)) n += 1;
  return `${prefix}_${n}`;
}

function slugify(raw: string) {
  const s = raw
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
  return s.length >= 2 ? s : `robot_${Date.now().toString(36)}`;
}

function intakePoly(intake: IntakeSpec) {
  const pose = intake.poseOnRobot || {};
  const x = pose.x || 0;
  const y = pose.y || 0;
  const a = ((pose.headingDeg || 0) * Math.PI) / 180;
  const reach = intake.reachIn || 5;
  const hw = (intake.widthIn || 12) / 2;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const corners = [
    [x + -s * hw, y + c * hw],
    [x + s * hw, y + -c * hw],
    [x + c * reach + s * hw, y + s * reach - c * hw],
    [x + c * reach - s * hw, y + s * reach + c * hw],
  ];
  return corners.map(([px, py]) => `${px},${-py}`).join(" ");
}

function aimLine(launcher: LauncherSpec) {
  const pose = launcher.poseOnRobot || {};
  const x = pose.x || 0;
  const y = pose.y || 0;
  const a = ((pose.headingDeg || 0) * Math.PI) / 180;
  const pitch = ((pose.pitchDeg || 0) * Math.PI) / 180;
  const len = 8 + Math.min(16, (launcher.muzzleSpeedInPerS || 180) / 20) * Math.cos(pitch);
  return { x1: x, y1: -y, x2: x + Math.cos(a) * len, y2: -(y + Math.sin(a) * len) };
}

function Section({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <div className="section">
      <h3 className="section-title">
        {title}
        <span className="spacer" />
        {action}
      </h3>
      <div className="stack">{children}</div>
    </div>
  );
}

function sameSel(a: Sel, b: Sel) {
  if (a.kind !== b.kind) return false;
  if ((a.kind === "intake" || a.kind === "launcher") && (b.kind === "intake" || b.kind === "launcher")) return a.id === b.id;
  return true;
}

export function RobotBuilderPage() {
  const [list, setList] = useState<PresetMeta[]>([]);
  const [id, setId] = useState("mecanum_biobuzz_4cap");
  const [doc, setDocState] = useState<RobotDoc | null>(null);
  const [sel, setSel] = useState<Sel>({ kind: "chassis" });
  const [errs, setErrs] = useState<string[]>([]);
  const [saveAsId, setSaveAsId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [lastBbox, setLastBbox] = useState<{ lengthIn: number; widthIn: number; heightIn: number } | null>(null);
  const drag = useRef<{ kind: "intake" | "launcher"; id: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  function setDoc(next: RobotDoc) {
    setDocState(next);
    setDirty(true);
  }

  useEffect(() => {
    getJson<PresetMeta[]>("/presets/robot").then(setList);
    getJson<DefaultsBundle>("/defaults")
      .then((d) => {
        if (d.robotId) setId(d.robotId);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!id) return;
    getJson<RobotDoc>(`/presets/robot/${id}`).then((d) => {
      delete (d as { _kind?: string })._kind;
      setDocState(d);
      setDirty(false);
      setErrs([]);
      setSel({ kind: "chassis" });
      setSaveAsId("");
      setLastBbox(null);
    });
  }, [id]);

  const intakes = doc?.intakes || [];
  const launchers = doc?.launchers || [];
  const length = doc?.chassis.lengthIn || 18;
  const width = doc?.chassis.widthIn || 18;
  const span = Math.max(length, width) / 2 + 14;

  const selectedIntake = sel.kind === "intake" ? intakes.find((i) => i.id === sel.id) : undefined;
  const selectedLauncher = sel.kind === "launcher" ? launchers.find((l) => l.id === sel.id) : undefined;

  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let v = Math.ceil(-span / 6) * 6; v <= span; v += 6) out.push(v);
    return out;
  }, [span]);

  function svgToRobot(clientX: number, clientY: number, svg: SVGSVGElement) {
    const rect = svg.getBoundingClientRect();
    const size = Math.min(rect.width, rect.height);
    const ox = rect.left + (rect.width - size) / 2;
    const oy = rect.top + (rect.height - size) / 2;
    const nx = (clientX - ox) / size;
    const ny = (clientY - oy) / size;
    return { x: nx * span * 2 - span, y: span - ny * span * 2 };
  }

  function patchPose(kind: "intake" | "launcher", mid: string, partial: PoseOnRobot) {
    if (!doc) return;
    if (kind === "intake") {
      setDoc({
        ...doc,
        intakes: intakes.map((i) => (i.id === mid ? { ...i, poseOnRobot: { ...(i.poseOnRobot || {}), ...partial } } : i)),
      });
    } else {
      setDoc({
        ...doc,
        launchers: launchers.map((l) => (l.id === mid ? { ...l, poseOnRobot: { ...(l.poseOnRobot || {}), ...partial } } : l)),
      });
    }
  }

  function patchIntake(mid: string, partial: Partial<IntakeSpec>) {
    if (!doc) return;
    setDoc({ ...doc, intakes: intakes.map((i) => (i.id === mid ? { ...i, ...partial } : i)) });
  }

  function patchLauncher(mid: string, partial: Partial<LauncherSpec>) {
    if (!doc) return;
    setDoc({ ...doc, launchers: launchers.map((l) => (l.id === mid ? { ...l, ...partial } : l)) });
  }

  function pickAt(clientX: number, clientY: number, svg: SVGSVGElement) {
    const { x, y } = svgToRobot(clientX, clientY, svg);
    let best: Sel = { kind: "chassis" };
    let bestD = Infinity;
    for (const intake of intakes) {
      const d = Math.hypot((intake.poseOnRobot?.x || 0) - x, (intake.poseOnRobot?.y || 0) - y);
      if (d < 6 && d < bestD) {
        bestD = d;
        best = { kind: "intake", id: intake.id };
      }
    }
    for (const launcher of launchers) {
      const d = Math.hypot((launcher.poseOnRobot?.x || 0) - x, (launcher.poseOnRobot?.y || 0) - y);
      if (d < 6 && d < bestD) {
        bestD = d;
        best = { kind: "launcher", id: launcher.id };
      }
    }
    setSel(best);
    if (best.kind === "intake" || best.kind === "launcher") drag.current = best;
  }

  async function validateAndSave(target: RobotDoc, method: "put" | "post") {
    const res = await postJson<{ ok: boolean; errors: string[] }>("/presets/validate", { kind: "robot", document: target });
    if (!res.ok) {
      setErrs(res.errors || ["Invalid robot preset."]);
      return false;
    }
    if (method === "post") await postJson("/presets/robot", target);
    else await putJson(`/presets/robot/${target.id}`, target);
    return true;
  }

  async function save() {
    if (!doc) return;
    setSaving(true);
    try {
      if (!(await validateAndSave(doc, "put"))) return;
      setErrs([]);
      setDirty(false);
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
    const next = { ...doc, id: nid, displayName: saveAsId ? doc.displayName : `${doc.displayName} copy` };
    setSaving(true);
    try {
      if (!(await validateAndSave(next, "post"))) return;
      setErrs([]);
      notify(`Created robot preset ${nid}.`);
      setList(await getJson<PresetMeta[]>("/presets/robot"));
      setId(nid);
    } catch (e) {
      setErrs([e instanceof Error ? e.message : String(e)]);
    } finally {
      setSaving(false);
    }
  }

  function addIntake() {
    if (!doc) return;
    const nid = nextId("intake", intakes.map((i) => i.id));
    const item: IntakeSpec = {
      id: nid,
      poseOnRobot: { x: length / 2, y: 0, z: 2, headingDeg: 0 },
      widthIn: 12,
      reachIn: 5,
      heightIn: 4,
      cycleTimeS: doc.mechanisms.intakeCycleTimeS ?? 0.4,
      canRunWhileMoving: doc.mechanisms.canIntakeWhileMoving ?? true,
    };
    setDoc({ ...doc, intakes: [...intakes, item] });
    setSel({ kind: "intake", id: nid });
  }

  function addLauncher() {
    if (!doc) return;
    const nid = nextId("hood", launchers.map((l) => l.id));
    const item: LauncherSpec = {
      id: nid,
      poseOnRobot: { x: 3, y: 0, z: 12, headingDeg: 0, pitchDeg: 50 },
      aimMode: "chassis_fixed",
      muzzleSpeedInPerS: 200,
      spinupTimeS: 0.3,
      cycleTimeS: doc.mechanisms.scoreCycleTimeS ?? 0.6,
      canLaunchWhileMoving: doc.mechanisms.canScoreWhileMoving ?? true,
      yawRangeDeg: [-90, 90],
      pitchRangeDeg: [20, 70],
    };
    setDoc({ ...doc, launchers: [...launchers, item] });
    setSel({ kind: "launcher", id: nid });
  }

  function removeSelected() {
    if (!doc) return;
    if (sel.kind === "intake") setDoc({ ...doc, intakes: intakes.filter((i) => i.id !== sel.id) });
    if (sel.kind === "launcher") setDoc({ ...doc, launchers: launchers.filter((l) => l.id !== sel.id) });
    setSel({ kind: "chassis" });
  }

  async function onUploadModel(file: File | undefined) {
    if (!doc || !file) return;
    setUploading(true);
    try {
      const imported = await uploadRobotModel(doc.id, file);
      const offset = doc.visualOffset || {};
      setLastBbox(imported.bbox);
      setDoc({
        ...doc,
        visualAsset: imported.visualAsset,
        collisionAsset: imported.collisionAsset,
        visualOffset: { x: offset.x || 0, y: offset.y || 0, z: offset.z || 0, yawDeg: offset.yawDeg || 0 },
        chassis: { ...doc.chassis, footprint: imported.footprint },
      });
      notify(
        `Imported ${file.name} (${imported.unitsGuess}, ${imported.bbox.lengthIn.toFixed(1)} × ${imported.bbox.widthIn.toFixed(1)} × ${imported.bbox.heightIn.toFixed(1)} in). Save to keep it.`,
      );
      setErrs([]);
    } catch (e) {
      setErrs([e instanceof Error ? e.message : String(e)]);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  function fitChassis() {
    if (!doc) return;
    if (lastBbox) {
      setDoc({
        ...doc,
        chassis: {
          ...doc.chassis,
          lengthIn: Math.round(lastBbox.lengthIn * 100) / 100,
          widthIn: Math.round(lastBbox.widthIn * 100) / 100,
          heightIn: Math.round(lastBbox.heightIn * 100) / 100,
        },
      });
      notify("Chassis size set from the model bounds.", "info");
      return;
    }
    if (!doc.chassis.footprint?.length) {
      setErrs(["Upload a model first, then fit the chassis to its bounds."]);
      return;
    }
    const xs = doc.chassis.footprint.map((p) => p.x);
    const ys = doc.chassis.footprint.map((p) => p.y);
    setDoc({
      ...doc,
      chassis: {
        ...doc.chassis,
        lengthIn: Math.round((Math.max(...xs) - Math.min(...xs)) * 100) / 100,
        widthIn: Math.round((Math.max(...ys) - Math.min(...ys)) * 100) / 100,
      },
    });
    notify("Chassis length and width set from the model footprint.", "info");
  }

  async function clearModel() {
    if (!doc) return;
    try {
      await deleteRobotModel(doc.id);
    } catch {
      /* files may already be gone */
    }
    const { visualAsset: _v, collisionAsset: _c, visualOffset: _o, ...rest } = doc;
    const chassis = { ...doc.chassis };
    delete chassis.footprint;
    if (chassis.collisionShape === "mesh") chassis.collisionShape = "aabb";
    setDoc({ ...rest, chassis });
    setLastBbox(null);
    notify("Removed the custom 3D model. Save to keep this change.", "info");
  }

  function patchOffset(partial: VisualOffset) {
    if (!doc) return;
    setDoc({ ...doc, visualOffset: { ...(doc.visualOffset || {}), ...partial } });
  }

  if (!doc) {
    return (
      <main className="page">
        <Panel>
          <Empty title="Loading robot…" />
        </Panel>
      </main>
    );
  }
  const camera = cam(doc);
  const hasModel = typeof doc.visualAsset === "string" && Boolean(doc.visualAsset);

  function setCamera(partial: { fovDeg?: number; rangeIn?: number }) {
    if (!doc) return;
    setDoc({
      ...doc,
      sensors: (doc.sensors || []).map((s) => (s === camera || s.kind === "apriltag_camera" ? { ...s, ...partial } : s)),
    });
  }

  const components: { sel: Sel; label: string; meta: string; color: string }[] = [
    { sel: { kind: "chassis" }, label: "Chassis & drivetrain", meta: `${length} × ${width} in · ${doc.drivetrain.type}`, color: theme.chassis },
    ...intakes.map((i) => ({ sel: { kind: "intake", id: i.id } as Sel, label: i.id, meta: `Intake · ${i.widthIn ?? 12} in wide`, color: theme.intake })),
    ...launchers.map((l) => ({ sel: { kind: "launcher", id: l.id } as Sel, label: l.id, meta: `Launcher · ${l.aimMode === "turret" ? "turret" : "fixed"}`, color: theme.gold })),
    { sel: { kind: "camera" }, label: "Camera", meta: `${camera?.fovDeg ?? 70}° FOV · ${camera?.rangeIn ?? 96} in`, color: "#7aa6d8" },
    { sel: { kind: "model" }, label: "3D model", meta: hasModel ? String(doc.visualAsset).split("/").pop() || "Loaded" : "None · using chassis box", color: "#9a9092" },
  ];

  const inspectorTitle =
    sel.kind === "chassis" ? "Chassis & drivetrain" : sel.kind === "camera" ? "Camera" : sel.kind === "model" ? "3D model" : sel.id;

  return (
    <main className="page layout-builder">
      <div className="toolbar">
        <h1>Robot</h1>
        <select aria-label="Robot preset" value={id} onChange={(e) => setId(e.target.value)}>
          {list.map((p) => (
            <option key={p.id} value={p.id}>
              {robotPresetLabel(p)}
            </option>
          ))}
        </select>
        <span className="spacer" />
        {dirty && <span className="pill">Unsaved changes</span>}
        <div className="row" style={{ flexWrap: "nowrap" }}>
          <input aria-label="New preset id" placeholder="new_preset_id" value={saveAsId} onChange={(e) => setSaveAsId(e.target.value)} style={{ width: 180, minWidth: 0 }} />
          <button type="button" className="btn" onClick={saveAs} disabled={saving} title="Create a copy so shipped presets stay intact">
            Save as new
          </button>
        </div>
        <button type="button" className="btn primary" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </button>
      </div>

      <div className="builder-robot">
        <Panel
          className="grow"
          title="Top view"
          sub="inches · +x forward, +y left"
          bodyClass="map-wrap"
          actions={
            <>
              <button type="button" className="btn sm" onClick={addIntake}>
                <Icon name="plus" size={14} /> Intake
              </button>
              <button type="button" className="btn sm" onClick={addLauncher}>
                <Icon name="plus" size={14} /> Launcher
              </button>
            </>
          }
          footer={
            <div className="legend">
              <span>
                <i style={{ background: theme.chassis }} /> Chassis
              </span>
              <span>
                <i style={{ background: theme.intake }} /> Intake
              </span>
              <span>
                <i style={{ background: theme.gold }} /> Launcher & aim
              </span>
              <span className="muted">Drag a mechanism to move it</span>
            </div>
          }
        >
          <svg
            className="map"
            viewBox={`${-span} ${-span} ${span * 2} ${span * 2}`}
            role="img"
            aria-label="Robot top view, +x forward"
            onPointerDown={(e) => {
              pickAt(e.clientX, e.clientY, e.currentTarget);
              if (drag.current) e.currentTarget.setPointerCapture(e.pointerId);
            }}
            onPointerMove={(e) => {
              if (!drag.current || e.buttons === 0) return;
              const p = svgToRobot(e.clientX, e.clientY, e.currentTarget);
              patchPose(drag.current.kind, drag.current.id, { x: Math.round(p.x * 2) / 2, y: Math.round(p.y * 2) / 2 });
            }}
            onPointerUp={() => {
              drag.current = null;
            }}
            onPointerCancel={() => {
              drag.current = null;
            }}
          >
            {ticks.map((v) => (
              <g key={v}>
                <line x1={v} y1={-span} x2={v} y2={span} stroke={theme.grid} strokeWidth={v === 0 ? 0.3 : 0.15} />
                <line x1={-span} y1={v} x2={span} y2={v} stroke={theme.grid} strokeWidth={v === 0 ? 0.3 : 0.15} />
              </g>
            ))}
            <rect
              x={-length / 2}
              y={-width / 2}
              width={length}
              height={width}
              rx={0.8}
              fill={theme.chassis}
              fillOpacity={0.9}
              stroke={sel.kind === "chassis" ? theme.selected : "rgba(0,0,0,0.4)"}
              strokeWidth={sel.kind === "chassis" ? 0.6 : 0.25}
            />
            <polygon points={`${length / 2 - 2.5},-2 ${length / 2 - 0.6},0 ${length / 2 - 2.5},2`} fill={theme.maroonDark} />
            {intakes.map((intake) => {
              const on = sel.kind === "intake" && sel.id === intake.id;
              return (
                <polygon
                  key={intake.id}
                  points={intakePoly(intake)}
                  fill={theme.intake}
                  fillOpacity={on ? 0.9 : 0.55}
                  stroke={on ? theme.selected : theme.intake}
                  strokeWidth={on ? 0.5 : 0.3}
                  style={{ cursor: "grab" }}
                />
              );
            })}
            {launchers.map((launcher) => {
              const pose = launcher.poseOnRobot || {};
              const line = aimLine(launcher);
              const on = sel.kind === "launcher" && sel.id === launcher.id;
              return (
                <g key={launcher.id} style={{ cursor: "grab" }}>
                  <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} stroke={theme.goldBright} strokeWidth={0.45} strokeDasharray="1 0.6" />
                  <circle cx={pose.x || 0} cy={-(pose.y || 0)} r={1.5} fill={theme.gold} stroke={on ? theme.selected : "rgba(0,0,0,0.4)"} strokeWidth={on ? 0.5 : 0.2} />
                </g>
              );
            })}
            <text x={span - 1} y={span - 1.2} fill={theme.mapLabel} fontSize="1.3" textAnchor="end" fontFamily="IBM Plex Mono, monospace">
              6 in grid
            </text>
          </svg>
        </Panel>

        <Panel className="grow" title="3D preview" sub="drag to orbit" bodyClass="viewport">
          <RobotPreview
            design={{
              chassis: doc.chassis,
              intakes,
              launchers,
              visualAsset: hasModel ? (doc.visualAsset as string) : null,
              visualOffset: doc.visualOffset,
              collisionShape: doc.chassis.collisionShape,
            }}
            showHull={doc.chassis.collisionShape === "mesh"}
          />
        </Panel>

        <div className="col">
          <Panel className="fixed" title="Components" bodyClass="panel-body flush">
            <ul className="list">
              {components.map((c) => (
                <li key={`${c.sel.kind}-${c.label}`}>
                  <button type="button" className={`list-item ${sameSel(sel, c.sel) ? "on" : ""}`} onClick={() => setSel(c.sel)} style={{ padding: "0.45rem 1rem" }}>
                    <span className="dot" style={{ background: c.color }} />
                    <span className="grow">
                      <span className="title" style={{ display: "block" }}>
                        {c.label}
                      </span>
                      <span className="meta" style={{ display: "block" }}>
                        {c.meta}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </Panel>

          <Panel
            className="grow"
            title={inspectorTitle}
            bodyClass="panel-body scroll"
            actions={
              (sel.kind === "intake" || sel.kind === "launcher") && (
                <button type="button" className="btn sm danger" onClick={removeSelected}>
                  <Icon name="trash" size={14} /> Remove
                </button>
              )
            }
          >
            {errs.length > 0 && (
              <div style={{ marginBottom: "1rem" }}>
                <Alert kind="bad">
                  <b>Could not save</b>
                  <ul>
                    {errs.map((e) => (
                      <li key={e}>{e}</li>
                    ))}
                  </ul>
                </Alert>
              </div>
            )}

            {sel.kind === "chassis" && (
              <>
                <Section title="General">
                  <Field id="display-name" label="Display name">
                    <input id="display-name" value={doc.displayName} onChange={(e) => setDoc({ ...doc, displayName: e.target.value })} />
                  </Field>
                  <div className="fields two">
                    <NumberField id="chassis-l" label="Length" unit="in" value={doc.chassis.lengthIn} onChange={(n) => setDoc({ ...doc, chassis: { ...doc.chassis, lengthIn: n } })} />
                    <NumberField id="chassis-w" label="Width" unit="in" value={doc.chassis.widthIn} onChange={(n) => setDoc({ ...doc, chassis: { ...doc.chassis, widthIn: n } })} />
                    <NumberField id="chassis-h" label="Height" unit="in" value={doc.chassis.heightIn ?? 10} onChange={(n) => setDoc({ ...doc, chassis: { ...doc.chassis, heightIn: n } })} />
                    <NumberField id="chassis-m" label="Mass" unit="kg" value={doc.chassis.massKg} onChange={(n) => setDoc({ ...doc, chassis: { ...doc.chassis, massKg: n } })} />
                  </div>
                  <Slider id="cap" label="Game piece capacity" value={doc.mechanisms.capacity} min={0} max={10} step={1} unit="" onChange={(n) => setDoc({ ...doc, mechanisms: { ...doc.mechanisms, capacity: n } })} />
                </Section>
                <Section title="Drivetrain">
                  <div className="fields two">
                    <Field id="dt" label="Type">
                      <select id="dt" value={doc.drivetrain.type} onChange={(e) => setDoc({ ...doc, drivetrain: { ...doc.drivetrain, type: e.target.value } })}>
                        <option value="mecanum">Mecanum</option>
                        <option value="tank">Tank</option>
                        <option value="swerve">Swerve</option>
                      </select>
                    </Field>
                    <NumberField id="track" label="Track width" unit="in" value={doc.drivetrain.trackWidthIn} onChange={(n) => setDoc({ ...doc, drivetrain: { ...doc.drivetrain, trackWidthIn: n } })} />
                  </div>
                </Section>
                <Section title="Motion limits">
                  <Slider id="max-vel" label="Max velocity" value={doc.constraints.maxVelInPerS} min={0} max={80} step={1} unit="in/s" onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxVelInPerS: n } })} />
                  <Slider id="max-acc" label="Max acceleration" value={doc.constraints.maxAccelInPerS2} min={0} max={80} step={1} unit="in/s²" onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxAccelInPerS2: n } })} />
                  <Slider id="max-ang" label="Max angular velocity" value={doc.constraints.maxAngVelDegPerS} min={0} max={360} step={1} unit="°/s" onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxAngVelDegPerS: n } })} />
                  <p className="note">Match these to your MeepMeep / Road Runner constraints.</p>
                </Section>
              </>
            )}

            {sel.kind === "camera" && (
              <Section title="AprilTag camera">
                {camera ? (
                  <>
                    <Slider id="fov" label="Field of view" value={camera.fovDeg ?? 70} min={30} max={120} step={1} unit="°" onChange={(n) => setCamera({ fovDeg: n })} />
                    <Slider id="range" label="Detection range" value={camera.rangeIn ?? 96} min={12} max={200} step={1} unit="in" onChange={(n) => setCamera({ rangeIn: n })} />
                  </>
                ) : (
                  <p className="note">This preset has no camera sensor.</p>
                )}
              </Section>
            )}

            {sel.kind === "model" && (
              <>
                <Section title="Mesh">
                  <p className="note">
                    Upload STL, OBJ, GLB, glTF, or STEP. It is converted to a lightweight GLB for the viewer. Physics keeps using the chassis box unless mesh collision is on.
                  </p>
                  <input ref={fileRef} type="file" accept=".glb,.gltf,.stl,.obj,.step,.stp" hidden onChange={(e) => onUploadModel(e.target.files?.[0])} />
                  <div className="row">
                    <button type="button" className="btn" disabled={uploading} onClick={() => fileRef.current?.click()}>
                      <Icon name="upload" size={14} /> {uploading ? "Converting…" : hasModel ? "Replace model" : "Upload model"}
                    </button>
                    <button type="button" className="btn" disabled={!hasModel} onClick={fitChassis}>
                      Fit chassis
                    </button>
                    <button type="button" className="btn danger" disabled={!hasModel} onClick={() => void clearModel()}>
                      <Icon name="trash" size={14} /> Remove
                    </button>
                  </div>
                  {hasModel && <p className="note mono">{doc.visualAsset}</p>}
                  <Switch
                    checked={doc.chassis.collisionShape === "mesh"}
                    disabled={!hasModel}
                    onChange={(v) => setDoc({ ...doc, chassis: { ...doc.chassis, collisionShape: v ? "mesh" : "aabb" } })}
                  >
                    Use mesh for collision
                  </Switch>
                  <p className="note">Slower. Use it for fidelity checks, not overnight training.</p>
                </Section>
                <Section title="Alignment">
                  <div className="fields two">
                    <NumberField id="off-x" label="Offset X" unit="in" value={doc.visualOffset?.x ?? 0} disabled={!hasModel} onChange={(n) => patchOffset({ x: n })} />
                    <NumberField id="off-y" label="Offset Y" unit="in" value={doc.visualOffset?.y ?? 0} disabled={!hasModel} onChange={(n) => patchOffset({ y: n })} />
                    <NumberField id="off-z" label="Offset Z" unit="in" value={doc.visualOffset?.z ?? 0} disabled={!hasModel} onChange={(n) => patchOffset({ z: n })} />
                    <NumberField id="off-yaw" label="Yaw" unit="°" value={doc.visualOffset?.yawDeg ?? 0} disabled={!hasModel} onChange={(n) => patchOffset({ yawDeg: n })} />
                  </div>
                </Section>
              </>
            )}

            {selectedIntake && (
              <>
                <Section title="Placement">
                  <div className="fields three">
                    <NumberField id="in-x" label="X" unit="in" value={selectedIntake.poseOnRobot?.x ?? 0} onChange={(n) => patchPose("intake", selectedIntake.id, { x: n })} />
                    <NumberField id="in-y" label="Y" unit="in" value={selectedIntake.poseOnRobot?.y ?? 0} onChange={(n) => patchPose("intake", selectedIntake.id, { y: n })} />
                    <NumberField id="in-h" label="Heading" unit="°" value={selectedIntake.poseOnRobot?.headingDeg ?? 0} onChange={(n) => patchPose("intake", selectedIntake.id, { headingDeg: n })} />
                  </div>
                </Section>
                <Section title="Mouth">
                  <Slider id="in-w" label="Width" value={selectedIntake.widthIn ?? 12} min={2} max={18} step={0.5} unit="in" onChange={(n) => patchIntake(selectedIntake.id, { widthIn: n })} />
                  <Slider id="in-r" label="Reach" value={selectedIntake.reachIn ?? 5} min={1} max={12} step={0.5} unit="in" onChange={(n) => patchIntake(selectedIntake.id, { reachIn: n })} />
                  <Slider id="in-c" label="Cycle time" value={selectedIntake.cycleTimeS ?? 0.4} min={0.05} max={3} step={0.05} unit="s" onChange={(n) => patchIntake(selectedIntake.id, { cycleTimeS: n })} />
                  <Switch checked={selectedIntake.canRunWhileMoving ?? true} onChange={(v) => patchIntake(selectedIntake.id, { canRunWhileMoving: v })}>
                    Can run while driving
                  </Switch>
                </Section>
              </>
            )}

            {selectedLauncher && (
              <>
                <Section title="Placement">
                  <div className="fields three">
                    <NumberField id="ln-x" label="X" unit="in" value={selectedLauncher.poseOnRobot?.x ?? 0} onChange={(n) => patchPose("launcher", selectedLauncher.id, { x: n })} />
                    <NumberField id="ln-y" label="Y" unit="in" value={selectedLauncher.poseOnRobot?.y ?? 0} onChange={(n) => patchPose("launcher", selectedLauncher.id, { y: n })} />
                    <NumberField id="ln-z" label="Z" unit="in" value={selectedLauncher.poseOnRobot?.z ?? 12} onChange={(n) => patchPose("launcher", selectedLauncher.id, { z: n })} />
                  </div>
                  <Field id="ln-mode" label="Aim mode">
                    <select id="ln-mode" value={selectedLauncher.aimMode || "chassis_fixed"} onChange={(e) => patchLauncher(selectedLauncher.id, { aimMode: e.target.value as LauncherSpec["aimMode"] })}>
                      <option value="chassis_fixed">Fixed to chassis</option>
                      <option value="turret">Turret</option>
                    </select>
                  </Field>
                </Section>
                <Section title="Shot">
                  <Slider id="ln-yaw" label="Yaw" value={selectedLauncher.poseOnRobot?.headingDeg ?? 0} min={-180} max={180} step={1} unit="°" onChange={(n) => patchPose("launcher", selectedLauncher.id, { headingDeg: n })} />
                  <Slider id="ln-pitch" label="Hood pitch" value={selectedLauncher.poseOnRobot?.pitchDeg ?? 0} min={0} max={85} step={1} unit="°" onChange={(n) => patchPose("launcher", selectedLauncher.id, { pitchDeg: n })} />
                  <Slider id="ln-spd" label="Muzzle speed" value={selectedLauncher.muzzleSpeedInPerS ?? 200} min={40} max={400} step={5} unit="in/s" onChange={(n) => patchLauncher(selectedLauncher.id, { muzzleSpeedInPerS: n })} />
                  <Slider id="ln-spin" label="Spin-up time" value={selectedLauncher.spinupTimeS ?? 0} min={0} max={2} step={0.05} unit="s" onChange={(n) => patchLauncher(selectedLauncher.id, { spinupTimeS: n })} />
                  <Slider id="ln-cyc" label="Cycle time" value={selectedLauncher.cycleTimeS ?? 0.6} min={0.05} max={3} step={0.05} unit="s" onChange={(n) => patchLauncher(selectedLauncher.id, { cycleTimeS: n })} />
                  <Switch checked={selectedLauncher.canLaunchWhileMoving ?? true} onChange={(v) => patchLauncher(selectedLauncher.id, { canLaunchWhileMoving: v })}>
                    Can launch while driving
                  </Switch>
                </Section>
              </>
            )}
          </Panel>
        </div>
      </div>
    </main>
  );
}
