import { useEffect, useState } from "react";
import { getJson, putJson, type DefaultsBundle, type PresetMeta } from "../api";

type RobotDoc = {
  id: string;
  drivetrain: {
    type: string;
    trackWidthIn: number;
    wheelDiameterIn?: number;
    wheelbaseIn?: number;
    strafeMultiplier?: number;
  };
  chassis: { lengthIn: number; widthIn: number; heightIn?: number; massKg?: number };
  constraints: {
    maxVelInPerS: number;
    maxAccelInPerS2: number;
    maxAngVelDegPerS: number;
    maxAngAccelDegPerS2?: number;
  };
  mechanisms: {
    capacity: number;
    intakeCycleTimeS: number;
    scoreCycleTimeS: number;
    canIntakeWhileMoving?: boolean;
    canScoreWhileMoving?: boolean;
  };
  sensors: { id: string; kind: string; fovDeg?: number; rangeIn?: number }[];
  defaultActionTier: string;
  [k: string]: unknown;
};

function cam(doc: RobotDoc) {
  return doc.sensors?.find((s) => s.kind === "apriltag_camera") || doc.sensors?.[0];
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
        {value} {unit}
      </span>
    </div>
  );
}

export function RobotBuilderPage() {
  const [list, setList] = useState<PresetMeta[]>([]);
  const [id, setId] = useState("mecanum_meepmeep_defaults");
  const [doc, setDoc] = useState<RobotDoc | null>(null);
  const [msg, setMsg] = useState("");
  const [errs, setErrs] = useState<string[]>([]);

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
    });
  }, [id]);

  async function save() {
    if (!doc) return;
    try {
      await putJson(`/presets/robot/${doc.id}`, doc);
      setMsg("Saved robot preset via API.");
      setErrs([]);
    } catch (e) {
      setMsg("");
      setErrs([e instanceof Error ? e.message : String(e)]);
    }
  }

  if (!doc) return <div className="page single">Loading…</div>;
  const camera = cam(doc);

  function setCamera(partial: { fovDeg?: number; rangeIn?: number }) {
    setDoc({
      ...doc,
      sensors: (doc.sensors || []).map((s) => (s === camera || s.kind === "apriltag_camera" ? { ...s, ...partial } : s)),
    });
  }

  return (
    <div className="page single">
      <div>
        <div className="page-head">
          <h2>Robot preset</h2>
          <p className="note">MeepMeep default vocabulary: 30 in/s, 30 in/s², 60 deg/s, 15 in track, 18×18 chassis, mecanum.</p>
        </div>
        <label htmlFor="robot-preset">Robot preset</label>
        <select id="robot-preset" value={id} onChange={(e) => setId(e.target.value)} style={{ maxWidth: 420 }}>
          {list.map((p) => (
            <option key={p.id} value={p.id}>
              {p.displayName || p.id}
            </option>
          ))}
        </select>
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
          </div>
        </div>
        <div className="card">
          <h3>Mechanisms and camera</h3>
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
          <Slider
            id="intake"
            label="Intake cycle"
            value={doc.mechanisms.intakeCycleTimeS}
            min={0.1}
            max={3}
            step={0.05}
            unit="s"
            onChange={(n) => setDoc({ ...doc, mechanisms: { ...doc.mechanisms, intakeCycleTimeS: n } })}
          />
          <Slider
            id="score"
            label="Score cycle"
            value={doc.mechanisms.scoreCycleTimeS}
            min={0.1}
            max={3}
            step={0.05}
            unit="s"
            onChange={(n) => setDoc({ ...doc, mechanisms: { ...doc.mechanisms, scoreCycleTimeS: n } })}
          />
          <Slider
            id="fov"
            label="Camera FOV"
            value={camera?.fovDeg ?? 70}
            min={30}
            max={120}
            step={1}
            unit="deg"
            onChange={(n) => setCamera({ fovDeg: n })}
          />
          <Slider
            id="range"
            label="Camera range"
            value={camera?.rangeIn ?? 96}
            min={12}
            max={200}
            step={1}
            unit="in"
            onChange={(n) => setCamera({ rangeIn: n })}
          />
        </div>
        <div className="card">
          <h3>Action tier</h3>
          <label htmlFor="tier">Default</label>
          <select
            id="tier"
            value={doc.defaultActionTier}
            onChange={(e) => setDoc({ ...doc, defaultActionTier: e.target.value })}
            style={{ maxWidth: 280 }}
          >
            <option value="high_level_waypoint">high_level_waypoint</option>
            <option value="low_level_velocity">low_level_velocity</option>
          </select>
        </div>
        <div className="row">
          <button className="primary" type="button" onClick={save}>
            Save robot preset
          </button>
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
