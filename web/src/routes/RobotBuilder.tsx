import { useEffect, useMemo, useRef, useState } from "react";
import { getJson, postJson, putJson, robotPresetLabel, uploadRobotModel, deleteRobotModel, type DefaultsBundle, type IntakeSpec, type LauncherSpec, type PoseOnRobot, type PresetMeta, type VisualOffset } from "../api";
import { RobotPreview } from "../scene/FieldScene";
import { theme } from "../theme";

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

type Sel = { kind: "chassis" } | { kind: "intake"; id: string } | { kind: "launcher"; id: string };

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

function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (n: number) => void;
}) {
  return (
    <div className="slider-row">
      <label htmlFor={id}>{label}</label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="readout">
        {Number.isInteger(step) ? value : value.toFixed(2)} {unit}
      </span>
    </div>
  );
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

export function RobotBuilderPage() {
  const [list, setList] = useState<PresetMeta[]>([]);
  const [id, setId] = useState("mecanum_biobuzz_4cap");
  const [doc, setDoc] = useState<RobotDoc | null>(null);
  const [sel, setSel] = useState<Sel>({ kind: "chassis" });
  const [msg, setMsg] = useState("");
  const [errs, setErrs] = useState<string[]>([]);
  const [saveAsId, setSaveAsId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [lastBbox, setLastBbox] = useState<{ lengthIn: number; widthIn: number; heightIn: number } | null>(null);
  const drag = useRef<{ kind: "intake" | "launcher"; id: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

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
      setDoc(d);
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
    const nx = (clientX - rect.left) / rect.width;
    const ny = (clientY - rect.top) / rect.height;
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

  function pickAt(clientX: number, clientY: number, svg: SVGSVGElement) {
    const { x, y } = svgToRobot(clientX, clientY, svg);
    let best: Sel = { kind: "chassis" };
    let bestD = Infinity;
    for (const intake of intakes) {
      const px = intake.poseOnRobot?.x || 0;
      const py = intake.poseOnRobot?.y || 0;
      const d = Math.hypot(px - x, py - y);
      if (d < 6 && d < bestD) {
        bestD = d;
        best = { kind: "intake", id: intake.id };
      }
    }
    for (const launcher of launchers) {
      const px = launcher.poseOnRobot?.x || 0;
      const py = launcher.poseOnRobot?.y || 0;
      const d = Math.hypot(px - x, py - y);
      if (d < 6 && d < bestD) {
        bestD = d;
        best = { kind: "launcher", id: launcher.id };
      }
    }
    setSel(best);
    if (best.kind !== "chassis") drag.current = best;
  }

  async function validateAndSave(target: RobotDoc, method: "put" | "post") {
    const res = await postJson<{ ok: boolean; errors: string[] }>("/presets/validate", { kind: "robot", document: target });
    if (!res.ok) {
      setErrs(res.errors || ["Invalid robot preset."]);
      setMsg("");
      return false;
    }
    if (method === "post") await postJson("/presets/robot", target);
    else await putJson(`/presets/robot/${target.id}`, target);
    return true;
  }

  async function save() {
    if (!doc) return;
    try {
      const ok = await validateAndSave(doc, "put");
      if (!ok) return;
      setMsg("Saved robot preset via API.");
      setErrs([]);
    } catch (e) {
      setMsg("");
      setErrs([e instanceof Error ? e.message : String(e)]);
    }
  }

  async function saveAs() {
    if (!doc) return;
    const nid = slugify(saveAsId || `${doc.id}_copy`);
    const next = { ...doc, id: nid, displayName: saveAsId ? doc.displayName : `${doc.displayName} copy` };
    try {
      const ok = await validateAndSave(next, "post");
      if (!ok) return;
      setMsg(`Created ${nid} via API.`);
      setErrs([]);
      const rows = await getJson<PresetMeta[]>("/presets/robot");
      setList(rows);
      setId(nid);
    } catch (e) {
      setMsg("");
      setErrs([e instanceof Error ? e.message : String(e)]);
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
    if (sel.kind === "intake") {
      setDoc({ ...doc, intakes: intakes.filter((i) => i.id !== sel.id) });
      setSel({ kind: "chassis" });
    }
    if (sel.kind === "launcher") {
      setDoc({ ...doc, launchers: launchers.filter((l) => l.id !== sel.id) });
      setSel({ kind: "chassis" });
    }
  }

  async function onUploadModel(file: File | undefined) {
    if (!doc || !file) return;
    setUploading(true);
    setMsg("Converting CAD…");
    try {
      const imported = await uploadRobotModel(doc.id, file);
      const offset = doc.visualOffset || {};
      setLastBbox(imported.bbox);
      setDoc({
        ...doc,
        visualAsset: imported.visualAsset,
        collisionAsset: imported.collisionAsset,
        visualOffset: { x: offset.x || 0, y: offset.y || 0, z: offset.z || 0, yawDeg: offset.yawDeg || 0, scale: offset.scale || 1 },
        chassis: { ...doc.chassis, footprint: imported.footprint },
      });
      setMsg(`Imported ${file.name} (${imported.unitsGuess}, ${imported.bbox.lengthIn.toFixed(1)}×${imported.bbox.widthIn.toFixed(1)}×${imported.bbox.heightIn.toFixed(1)} in). Fit chassis if you want the box to match.`);
      setErrs([]);
    } catch (e) {
      setMsg("");
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
      setMsg("Chassis length/width/height set from the model bounds.");
      return;
    }
    if (!doc.chassis.footprint?.length) {
      setErrs(["Upload a model first, then fit chassis to its bounds."]);
      return;
    }
    const xs = doc.chassis.footprint.map((p) => p.x);
    const ys = doc.chassis.footprint.map((p) => p.y);
    const lengthIn = Math.max(...xs) - Math.min(...xs);
    const widthIn = Math.max(...ys) - Math.min(...ys);
    setDoc({
      ...doc,
      chassis: {
        ...doc.chassis,
        lengthIn: Math.round(lengthIn * 100) / 100,
        widthIn: Math.round(widthIn * 100) / 100,
      },
    });
    setMsg("Chassis length/width set from the model footprint.");
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
    setMsg("Cleared custom 3D model.");
  }

  function patchOffset(partial: VisualOffset) {
    if (!doc) return;
    setDoc({ ...doc, visualOffset: { ...(doc.visualOffset || {}), ...partial } });
  }

  if (!doc) return <div className="page single">Loading…</div>;
  const robot = doc;
  const camera = cam(robot);

  function setCamera(partial: { fovDeg?: number; rangeIn?: number }) {
    setDoc({
      ...robot,
      sensors: (robot.sensors || []).map((s) => (s === camera || s.kind === "apriltag_camera" ? { ...s, ...partial } : s)),
    });
  }

  return (
    <div className="page single">
      <div>
        <div className="page-head">
          <h2>Robot design</h2>
          <p className="note">
            Robot frame, inches: origin at chassis center, +x forward, +y left. Place intakes and launchers on the grid.
            Saves go through the API, not browser storage.
          </p>
        </div>
        <label htmlFor="robot-preset">Robot preset</label>
        <select id="robot-preset" value={id} onChange={(e) => setId(e.target.value)} style={{ maxWidth: 420 }}>
          {list.map((p) => (
            <option key={p.id} value={p.id}>
              {robotPresetLabel(p)}
            </option>
          ))}
        </select>
        <div className="form-grid" style={{ marginTop: "0.6rem" }}>
          <label htmlFor="display-name">Display name</label>
          <input id="display-name" value={doc.displayName} onChange={(e) => setDoc({ ...doc, displayName: e.target.value })} />
        </div>

        <div className="robot-work">
          <div>
            <svg
              className="field-grid"
              viewBox={`${-span} ${-span} ${span * 2} ${span * 2}`}
              role="img"
              aria-label="Robot inch grid, +x forward"
              onPointerDown={(e) => pickAt(e.clientX, e.clientY, e.currentTarget)}
              onPointerMove={(e) => {
                if (!drag.current || e.buttons === 0) return;
                const p = svgToRobot(e.clientX, e.clientY, e.currentTarget);
                patchPose(drag.current.kind, drag.current.id, { x: Math.round(p.x * 2) / 2, y: Math.round(p.y * 2) / 2 });
              }}
              onPointerUp={() => {
                drag.current = null;
              }}
              onPointerLeave={() => {
                drag.current = null;
              }}
            >
              <rect x={-span} y={-span} width={span * 2} height={span * 2} fill={theme.scene} />
              {ticks.map((v) => (
                <g key={v}>
                  <line x1={v} y1={-span} x2={v} y2={span} stroke={theme.grid} strokeWidth="0.35" />
                  <line x1={-span} y1={v} x2={span} y2={v} stroke={theme.grid} strokeWidth="0.35" />
                </g>
              ))}
              <rect
                x={-length / 2}
                y={-width / 2}
                width={length}
                height={width}
                fill={theme.chassis}
                fillOpacity={0.85}
                stroke={sel.kind === "chassis" ? theme.cream : theme.muted}
                strokeWidth={sel.kind === "chassis" ? 0.7 : 0.35}
              />
              <polygon points={`${length / 2 - 1.5},0 ${length / 2},${-2} ${length / 2},${2}`} fill={theme.maroonDark} />
              {intakes.map((intake) => (
                <polygon
                  key={intake.id}
                  points={intakePoly(intake)}
                  fill={theme.intake}
                  fillOpacity={sel.kind === "intake" && sel.id === intake.id ? 0.85 : 0.45}
                  stroke={sel.kind === "intake" && sel.id === intake.id ? theme.cream : theme.intake}
                  strokeWidth={0.45}
                />
              ))}
              {launchers.map((launcher) => {
                const pose = launcher.poseOnRobot || {};
                const line = aimLine(launcher);
                const on = sel.kind === "launcher" && sel.id === launcher.id;
                return (
                  <g key={launcher.id}>
                    <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} stroke={theme.goldBright} strokeWidth={0.5} />
                    <circle cx={pose.x || 0} cy={-(pose.y || 0)} r={1.4} fill={theme.gold} stroke={on ? theme.cream : "transparent"} strokeWidth={0.4} />
                  </g>
                );
              })}
              <line x1={0} y1={0} x2={6} y2={0} stroke={theme.gold} strokeWidth={0.35} />
              <text x={span - 8} y={span - 2} fill={theme.muted} fontSize="2.2">
                +x fwd
              </text>
            </svg>
            <div className="legend">
              <span>
                <i style={{ background: theme.chassis }} /> chassis
              </span>
              <span>
                <i style={{ background: theme.intake }} /> intake
              </span>
              <span>
                <i style={{ background: theme.gold }} /> launcher
              </span>
            </div>
            <div className="robot-preview" aria-label="Robot 3D preview">
              <RobotPreview
                design={{
                  chassis: doc.chassis,
                  intakes,
                  launchers,
                  visualAsset: typeof doc.visualAsset === "string" ? doc.visualAsset : null,
                  visualOffset: doc.visualOffset,
                  collisionShape: doc.chassis.collisionShape,
                }}
                showHull={doc.chassis.collisionShape === "mesh"}
              />
            </div>
          </div>

          <div>
            <div className="row">
              <button type="button" onClick={addIntake}>
                Add intake
              </button>
              <button type="button" onClick={addLauncher}>
                Add launcher
              </button>
              {(sel.kind === "intake" || sel.kind === "launcher") && (
                <button type="button" onClick={removeSelected}>
                  Remove selected
                </button>
              )}
            </div>

            <div className="card">
              <h3>Drivetrain</h3>
              <div className="form-grid">
                <label htmlFor="dt">Type</label>
                <select
                  id="dt"
                  value={doc.drivetrain.type}
                  onChange={(e) => setDoc({ ...doc, drivetrain: { ...doc.drivetrain, type: e.target.value } })}
                >
                  <option value="mecanum">mecanum</option>
                  <option value="tank">tank</option>
                  <option value="swerve">swerve</option>
                </select>
                <label htmlFor="track">Track width (in)</label>
                <input
                  id="track"
                  className="narrow"
                  type="number"
                  value={doc.drivetrain.trackWidthIn}
                  onChange={(e) => setDoc({ ...doc, drivetrain: { ...doc.drivetrain, trackWidthIn: Number(e.target.value) } })}
                />
              </div>
            </div>

            <div className="card">
              <h3>Motion limits (MeepMeep)</h3>
              <Slider
                id="max-vel"
                label="Max vel"
                value={doc.constraints.maxVelInPerS}
                min={0}
                max={80}
                step={1}
                unit="in/s"
                onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxVelInPerS: n } })}
              />
              <Slider
                id="max-acc"
                label="Max accel"
                value={doc.constraints.maxAccelInPerS2}
                min={0}
                max={80}
                step={1}
                unit="in/s²"
                onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxAccelInPerS2: n } })}
              />
              <Slider
                id="max-ang"
                label="Max ang vel"
                value={doc.constraints.maxAngVelDegPerS}
                min={0}
                max={360}
                step={1}
                unit="deg/s"
                onChange={(n) => setDoc({ ...doc, constraints: { ...doc.constraints, maxAngVelDegPerS: n } })}
              />
            </div>

            <div className="card">
              <h3>Chassis</h3>
              <div className="form-grid">
                <label htmlFor="chassis-l">Length (in)</label>
                <input
                  id="chassis-l"
                  className="narrow"
                  type="number"
                  value={doc.chassis.lengthIn}
                  onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, lengthIn: Number(e.target.value) } })}
                />
                <label htmlFor="chassis-w">Width (in)</label>
                <input
                  id="chassis-w"
                  className="narrow"
                  type="number"
                  value={doc.chassis.widthIn}
                  onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, widthIn: Number(e.target.value) } })}
                />
                <label htmlFor="chassis-h">Height (in)</label>
                <input
                  id="chassis-h"
                  className="narrow"
                  type="number"
                  value={doc.chassis.heightIn ?? 10}
                  onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, heightIn: Number(e.target.value) } })}
                />
                <label htmlFor="chassis-m">Mass (kg)</label>
                <input
                  id="chassis-m"
                  className="narrow"
                  type="number"
                  value={doc.chassis.massKg}
                  onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, massKg: Number(e.target.value) } })}
                />
              </div>
              <Slider
                id="cap"
                label="Capacity"
                value={doc.mechanisms.capacity}
                min={0}
                max={10}
                step={1}
                unit=""
                onChange={(n) => setDoc({ ...doc, mechanisms: { ...doc.mechanisms, capacity: n } })}
              />
            </div>

            <div className="card">
              <h3>3D model</h3>
              <p className="note">
                Upload STL, OBJ, GLB, glTF, or STEP. Files convert to a light GLB for the viewer. Physics stays the chassis box unless you enable mesh collision (slower; for fidelity, not overnight PPO).
              </p>
              <input
                ref={fileRef}
                id="robot-cad"
                type="file"
                accept=".glb,.gltf,.stl,.obj,.step,.stp"
                hidden
                onChange={(e) => onUploadModel(e.target.files?.[0])}
              />
              <div className="row">
                <button type="button" disabled={uploading} onClick={() => fileRef.current?.click()}>
                  {uploading ? "Converting…" : "Upload model"}
                </button>
                <button type="button" disabled={!doc.visualAsset} onClick={fitChassis}>
                  Fit chassis to model
                </button>
                <button type="button" disabled={!doc.visualAsset} onClick={() => void clearModel()}>
                  Clear model
                </button>
              </div>
              {typeof doc.visualAsset === "string" && doc.visualAsset && (
                <p className="note">Loaded {doc.visualAsset}</p>
              )}
              <label className="check">
                <input
                  type="checkbox"
                  checked={doc.chassis.collisionShape === "mesh"}
                  disabled={!doc.visualAsset}
                  onChange={(e) =>
                    setDoc({
                      ...doc,
                      chassis: { ...doc.chassis, collisionShape: e.target.checked ? "mesh" : "aabb" },
                    })
                  }
                />
                Use mesh for collision
              </label>
              <div className="form-grid">
                <label htmlFor="off-x">Offset x (in)</label>
                <input
                  id="off-x"
                  className="narrow"
                  type="number"
                  value={doc.visualOffset?.x ?? 0}
                  onChange={(e) => patchOffset({ x: Number(e.target.value) })}
                />
                <label htmlFor="off-y">Offset y (in)</label>
                <input
                  id="off-y"
                  className="narrow"
                  type="number"
                  value={doc.visualOffset?.y ?? 0}
                  onChange={(e) => patchOffset({ y: Number(e.target.value) })}
                />
                <label htmlFor="off-z">Offset z (in)</label>
                <input
                  id="off-z"
                  className="narrow"
                  type="number"
                  value={doc.visualOffset?.z ?? 0}
                  onChange={(e) => patchOffset({ z: Number(e.target.value) })}
                />
                <label htmlFor="off-yaw">Yaw (deg)</label>
                <input
                  id="off-yaw"
                  className="narrow"
                  type="number"
                  value={doc.visualOffset?.yawDeg ?? 0}
                  onChange={(e) => patchOffset({ yawDeg: Number(e.target.value) })}
                />
              </div>
              <Slider
                id="off-scale"
                label="Scale"
                value={doc.visualOffset?.scale ?? 1}
                min={0.1}
                max={4}
                step={0.05}
                unit="×"
                onChange={(n) => patchOffset({ scale: n })}
              />
            </div>

            {selectedIntake && (
              <div className="card">
                <h3>Intake {selectedIntake.id}</h3>
                <div className="form-grid">
                  <label htmlFor="in-x">x (in)</label>
                  <input
                    id="in-x"
                    type="number"
                    className="narrow"
                    value={selectedIntake.poseOnRobot?.x ?? 0}
                    onChange={(e) => patchPose("intake", selectedIntake.id, { x: Number(e.target.value) })}
                  />
                  <label htmlFor="in-y">y (in)</label>
                  <input
                    id="in-y"
                    type="number"
                    className="narrow"
                    value={selectedIntake.poseOnRobot?.y ?? 0}
                    onChange={(e) => patchPose("intake", selectedIntake.id, { y: Number(e.target.value) })}
                  />
                  <label htmlFor="in-h">heading (deg)</label>
                  <input
                    id="in-h"
                    type="number"
                    className="narrow"
                    value={selectedIntake.poseOnRobot?.headingDeg ?? 0}
                    onChange={(e) => patchPose("intake", selectedIntake.id, { headingDeg: Number(e.target.value) })}
                  />
                </div>
                <Slider
                  id="in-w"
                  label="Width"
                  value={selectedIntake.widthIn ?? 12}
                  min={2}
                  max={18}
                  step={0.5}
                  unit="in"
                  onChange={(n) =>
                    setDoc({ ...doc, intakes: intakes.map((i) => (i.id === selectedIntake.id ? { ...i, widthIn: n } : i)) })
                  }
                />
                <Slider
                  id="in-r"
                  label="Reach"
                  value={selectedIntake.reachIn ?? 5}
                  min={1}
                  max={12}
                  step={0.5}
                  unit="in"
                  onChange={(n) =>
                    setDoc({ ...doc, intakes: intakes.map((i) => (i.id === selectedIntake.id ? { ...i, reachIn: n } : i)) })
                  }
                />
                <Slider
                  id="in-c"
                  label="Cycle"
                  value={selectedIntake.cycleTimeS ?? 0.4}
                  min={0.05}
                  max={3}
                  step={0.05}
                  unit="s"
                  onChange={(n) =>
                    setDoc({ ...doc, intakes: intakes.map((i) => (i.id === selectedIntake.id ? { ...i, cycleTimeS: n } : i)) })
                  }
                />
                <label className="check">
                  <input
                    type="checkbox"
                    checked={selectedIntake.canRunWhileMoving ?? true}
                    onChange={(e) =>
                      setDoc({
                        ...doc,
                        intakes: intakes.map((i) => (i.id === selectedIntake.id ? { ...i, canRunWhileMoving: e.target.checked } : i)),
                      })
                    }
                  />
                  Can run while moving
                </label>
              </div>
            )}

            {selectedLauncher && (
              <div className="card">
                <h3>Launcher {selectedLauncher.id}</h3>
                <div className="form-grid">
                  <label htmlFor="ln-mode">Aim mode</label>
                  <select
                    id="ln-mode"
                    value={selectedLauncher.aimMode || "chassis_fixed"}
                    onChange={(e) =>
                      setDoc({
                        ...doc,
                        launchers: launchers.map((l) =>
                          l.id === selectedLauncher.id ? { ...l, aimMode: e.target.value as LauncherSpec["aimMode"] } : l,
                        ),
                      })
                    }
                  >
                    <option value="chassis_fixed">chassis_fixed</option>
                    <option value="turret">turret</option>
                  </select>
                  <label htmlFor="ln-x">x (in)</label>
                  <input
                    id="ln-x"
                    type="number"
                    className="narrow"
                    value={selectedLauncher.poseOnRobot?.x ?? 0}
                    onChange={(e) => patchPose("launcher", selectedLauncher.id, { x: Number(e.target.value) })}
                  />
                  <label htmlFor="ln-y">y (in)</label>
                  <input
                    id="ln-y"
                    type="number"
                    className="narrow"
                    value={selectedLauncher.poseOnRobot?.y ?? 0}
                    onChange={(e) => patchPose("launcher", selectedLauncher.id, { y: Number(e.target.value) })}
                  />
                  <label htmlFor="ln-z">z (in)</label>
                  <input
                    id="ln-z"
                    type="number"
                    className="narrow"
                    value={selectedLauncher.poseOnRobot?.z ?? 12}
                    onChange={(e) => patchPose("launcher", selectedLauncher.id, { z: Number(e.target.value) })}
                  />
                </div>
                <Slider
                  id="ln-yaw"
                  label="Yaw"
                  value={selectedLauncher.poseOnRobot?.headingDeg ?? 0}
                  min={-180}
                  max={180}
                  step={1}
                  unit="deg"
                  onChange={(n) => patchPose("launcher", selectedLauncher.id, { headingDeg: n })}
                />
                <Slider
                  id="ln-pitch"
                  label="Pitch"
                  value={selectedLauncher.poseOnRobot?.pitchDeg ?? 0}
                  min={0}
                  max={85}
                  step={1}
                  unit="deg"
                  onChange={(n) => patchPose("launcher", selectedLauncher.id, { pitchDeg: n })}
                />
                <Slider
                  id="ln-spd"
                  label="Muzzle"
                  value={selectedLauncher.muzzleSpeedInPerS ?? 200}
                  min={40}
                  max={400}
                  step={5}
                  unit="in/s"
                  onChange={(n) =>
                    setDoc({
                      ...doc,
                      launchers: launchers.map((l) => (l.id === selectedLauncher.id ? { ...l, muzzleSpeedInPerS: n } : l)),
                    })
                  }
                />
                <Slider
                  id="ln-spin"
                  label="Spin-up"
                  value={selectedLauncher.spinupTimeS ?? 0}
                  min={0}
                  max={2}
                  step={0.05}
                  unit="s"
                  onChange={(n) =>
                    setDoc({
                      ...doc,
                      launchers: launchers.map((l) => (l.id === selectedLauncher.id ? { ...l, spinupTimeS: n } : l)),
                    })
                  }
                />
                <Slider
                  id="ln-cyc"
                  label="Cycle"
                  value={selectedLauncher.cycleTimeS ?? 0.6}
                  min={0.05}
                  max={3}
                  step={0.05}
                  unit="s"
                  onChange={(n) =>
                    setDoc({
                      ...doc,
                      launchers: launchers.map((l) => (l.id === selectedLauncher.id ? { ...l, cycleTimeS: n } : l)),
                    })
                  }
                />
                <label className="check">
                  <input
                    type="checkbox"
                    checked={selectedLauncher.canLaunchWhileMoving ?? true}
                    onChange={(e) =>
                      setDoc({
                        ...doc,
                        launchers: launchers.map((l) =>
                          l.id === selectedLauncher.id ? { ...l, canLaunchWhileMoving: e.target.checked } : l,
                        ),
                      })
                    }
                  />
                  Can launch while moving
                </label>
              </div>
            )}

            <div className="card">
              <h3>Camera</h3>
              <Slider
                id="fov"
                label="FOV"
                value={camera?.fovDeg ?? 70}
                min={30}
                max={120}
                step={1}
                unit="deg"
                onChange={(n) => setCamera({ fovDeg: n })}
              />
              <Slider
                id="range"
                label="Range"
                value={camera?.rangeIn ?? 96}
                min={12}
                max={200}
                step={1}
                unit="in"
                onChange={(n) => setCamera({ rangeIn: n })}
              />
            </div>
          </div>
        </div>

        <div className="card">
          <h3>Save</h3>
          <div className="row">
            <button className="primary" type="button" onClick={save}>
              Save robot preset
            </button>
          </div>
          <label htmlFor="save-as">Save as new id</label>
          <div className="row">
            <input
              id="save-as"
              placeholder="team_hood_v1"
              value={saveAsId}
              onChange={(e) => setSaveAsId(e.target.value)}
              style={{ maxWidth: 280 }}
            />
            <button type="button" onClick={saveAs}>
              Save as new
            </button>
          </div>
          <p className="note">Use Save as so shipped presets stay intact. Ids are lowercase with underscores.</p>
        </div>
        {errs.map((e) => (
          <div key={e} className="banner">
            {e}
          </div>
        ))}
        <p className="note">{msg}</p>
      </div>
    </div>
  );
}
