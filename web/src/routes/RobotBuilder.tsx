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
        <h2>Robot preset</h2>
        <p className="note">
          MeepMeep default vocabulary: 30 in/s, 30 in/s², 60 deg/s, 15 in track, 18×18 chassis, mecanum.
        </p>
        <label htmlFor="robot-preset">Robot preset</label>
        <select id="robot-preset" value={id} onChange={(e) => setId(e.target.value)}>
          {list.map((p) => (
            <option key={p.id} value={p.id}>
              {p.displayName || p.id}
            </option>
          ))}
        </select>
        <div className="form-grid" style={{ marginTop: "1rem" }}>
          <label htmlFor="dt">Drivetrain</label>
          <select
            id="dt"
            value={doc.drivetrain.type}
            onChange={(e) => setDoc({ ...doc, drivetrain: { ...doc.drivetrain, type: e.target.value } })}
          >
            <option value="mecanum">mecanum</option>
            <option value="tank">tank</option>
            <option value="swerve">swerve</option>
          </select>
          <label htmlFor="max-vel">Max vel (in/s)</label>
          <input
            id="max-vel"
            type="number"
            value={doc.constraints.maxVelInPerS}
            onChange={(e) => setDoc({ ...doc, constraints: { ...doc.constraints, maxVelInPerS: Number(e.target.value) } })}
          />
          <label htmlFor="max-acc">Max accel (in/s²)</label>
          <input
            id="max-acc"
            type="number"
            value={doc.constraints.maxAccelInPerS2}
            onChange={(e) =>
              setDoc({ ...doc, constraints: { ...doc.constraints, maxAccelInPerS2: Number(e.target.value) } })
            }
          />
          <label htmlFor="max-ang">Max ang vel (deg/s)</label>
          <input
            id="max-ang"
            type="number"
            value={doc.constraints.maxAngVelDegPerS}
            onChange={(e) =>
              setDoc({ ...doc, constraints: { ...doc.constraints, maxAngVelDegPerS: Number(e.target.value) } })
            }
          />
          <label htmlFor="track">Track width (in)</label>
          <input
            id="track"
            type="number"
            value={doc.drivetrain.trackWidthIn}
            onChange={(e) =>
              setDoc({ ...doc, drivetrain: { ...doc.drivetrain, trackWidthIn: Number(e.target.value) } })
            }
          />
          <label htmlFor="chassis-l">Chassis length (in)</label>
          <input
            id="chassis-l"
            type="number"
            value={doc.chassis.lengthIn}
            onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, lengthIn: Number(e.target.value) } })}
          />
          <label htmlFor="chassis-w">Chassis width (in)</label>
          <input
            id="chassis-w"
            type="number"
            value={doc.chassis.widthIn}
            onChange={(e) => setDoc({ ...doc, chassis: { ...doc.chassis, widthIn: Number(e.target.value) } })}
          />
          <label htmlFor="cap">Capacity</label>
          <input
            id="cap"
            type="number"
            value={doc.mechanisms.capacity}
            onChange={(e) => setDoc({ ...doc, mechanisms: { ...doc.mechanisms, capacity: Number(e.target.value) } })}
          />
          <label htmlFor="intake">Intake cycle (s)</label>
          <input
            id="intake"
            type="number"
            step="0.05"
            value={doc.mechanisms.intakeCycleTimeS}
            onChange={(e) =>
              setDoc({ ...doc, mechanisms: { ...doc.mechanisms, intakeCycleTimeS: Number(e.target.value) } })
            }
          />
          <label htmlFor="score">Score cycle (s)</label>
          <input
            id="score"
            type="number"
            step="0.05"
            value={doc.mechanisms.scoreCycleTimeS}
            onChange={(e) =>
              setDoc({ ...doc, mechanisms: { ...doc.mechanisms, scoreCycleTimeS: Number(e.target.value) } })
            }
          />
          <label htmlFor="fov">Camera FOV (deg)</label>
          <input
            id="fov"
            type="number"
            value={camera?.fovDeg ?? 70}
            onChange={(e) => setCamera({ fovDeg: Number(e.target.value) })}
          />
          <label htmlFor="range">Camera range (in)</label>
          <input
            id="range"
            type="number"
            value={camera?.rangeIn ?? 96}
            onChange={(e) => setCamera({ rangeIn: Number(e.target.value) })}
          />
          <label htmlFor="tier">Action tier</label>
          <select
            id="tier"
            value={doc.defaultActionTier}
            onChange={(e) => setDoc({ ...doc, defaultActionTier: e.target.value })}
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
