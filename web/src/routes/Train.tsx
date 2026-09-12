import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { FieldScene } from "../scene/FieldScene";
import { Sparkline } from "../Sparkline";
import { getJson, loadReplayFrames, postJson, putJson, type DefaultsBundle, type Frame, type PresetMeta, type RunRow } from "../api";

type Metrics = {
  envSteps?: number;
  algo?: string;
  replayId?: string;
  trueScoreMean?: number | null;
  objectiveMean?: number | null;
  shapingMean?: number | null;
};

export function TrainPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [current, setCurrent] = useState<RunRow | null>(null);
  const [log, setLog] = useState("Idle.");
  const [fields, setFields] = useState<PresetMeta[]>([]);
  const [robots, setRobots] = useState<PresetMeta[]>([]);
  const [scoring, setScoring] = useState<PresetMeta[]>([]);
  const [training, setTraining] = useState<PresetMeta[]>([]);
  const [fieldId, setFieldId] = useState("decode_2025_field_tu32");
  const [robotId, setRobotId] = useState("mecanum_meepmeep_defaults");
  const [scoringId, setScoringId] = useState("decode_2025_scoring_tu32");
  const [trainingId, setTrainingId] = useState("decode_auto_lightweight");
  const [defaultSeason, setDefaultSeason] = useState("");
  const [demo, setDemo] = useState(true);
  const [trueSeries, setTrueSeries] = useState<number[]>([]);
  const [shapeSeries, setShapeSeries] = useState<number[]>([]);
  const [rollout, setRollout] = useState<Frame[]>([]);
  const [rolloutI, setRolloutI] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  async function refreshList() {
    const list = await getJson<RunRow[]>("/runs");
    setRuns(list);
    return list;
  }

  useEffect(() => {
    getJson<PresetMeta[]>("/presets/field").then(setFields);
    getJson<PresetMeta[]>("/presets/robot").then(setRobots);
    getJson<PresetMeta[]>("/presets/scoring").then(setScoring);
    getJson<PresetMeta[]>("/presets/training").then(setTraining);
    getJson<DefaultsBundle>("/defaults")
      .then((d) => {
        if (d.fieldId) setFieldId(d.fieldId);
        if (d.robotId) setRobotId(d.robotId);
        if (d.scoringId) setScoringId(d.scoringId);
        if (d.trainingId) setTrainingId(d.trainingId);
        if (d.season) setDefaultSeason(d.season);
      })
      .catch(() => undefined);
    refreshList();
  }, []);

  useEffect(() => {
    if (runId) openRun(runId);
  }, [runId]);

  useEffect(() => {
    const id = window.setInterval(async () => {
      const list = await refreshList();
      if (!current) return;
      const row = list.find((r) => r.id === current.id) || (await getJson<RunRow>(`/runs/${current.id}`));
      if (row) {
        setCurrent(row);
        pushSeries(row.metrics as Metrics);
      }
    }, 2000);
    return () => window.clearInterval(id);
  }, [current?.id]);

  useEffect(() => {
    if (rollout.length === 0) return;
    const id = window.setInterval(() => setRolloutI((n) => (n + 1) % rollout.length), 80);
    return () => window.clearInterval(id);
  }, [rollout]);

  function pushSeries(m: Metrics) {
    if (typeof m.trueScoreMean === "number") {
      setTrueSeries((s) => (s.length && s[s.length - 1] === m.trueScoreMean ? s : [...s.slice(-80), m.trueScoreMean as number]));
    }
    if (typeof m.shapingMean === "number") {
      setShapeSeries((s) => (s.length && s[s.length - 1] === m.shapingMean ? s : [...s.slice(-80), m.shapingMean as number]));
    }
  }

  function openRun(id: string) {
    setTrueSeries([]);
    setShapeSeries([]);
    setRollout([]);
    navigate(`/train/${id}`, { replace: true });
    getJson<RunRow>(`/runs/${id}`).then((row) => {
      setCurrent(row);
      const m = row.metrics as Metrics;
      pushSeries(m);
      if (m.replayId) {
        loadReplayFrames(m.replayId).then((frames) => {
          setRollout(frames.filter((_, idx) => idx % 5 === 0));
          setRolloutI(0);
        });
      }
    });
    wsRef.current?.close();
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/api/v1/ws/runs/${id}`);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "metrics" && msg.payload) {
          pushSeries(msg.payload as Metrics);
          setCurrent((prev) =>
            prev && prev.id === id ? { ...prev, metrics: { ...(prev.metrics || {}), ...msg.payload } } : prev,
          );
        }
        if (msg.type === "status" && msg.payload) {
          setCurrent((prev) => (prev && prev.id === id ? { ...prev, state: msg.payload.state || prev.state } : prev));
          if (msg.payload.state) setLog(`Run ${id} · ${msg.payload.state}`);
        }
        if (msg.type === "log" && msg.payload?.message) setLog(String(msg.payload.message));
        if (msg.type === "rollout" && msg.payload?.frames) {
          setRollout(msg.payload.frames as Frame[]);
          setRolloutI(0);
        }
        if (msg.type === "error" && msg.payload) {
          window.dispatchEvent(
            new CustomEvent("talongym-error", {
              detail: { code: msg.payload.code || "TRAIN", message: msg.payload.message || "Training failed" },
            }),
          );
        }
      } catch {
        /* ignore malformed */
      }
    };
  }

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  async function applyDefaults(body: Partial<DefaultsBundle>) {
    const d = await putJson<DefaultsBundle>("/defaults", body);
    if (d.fieldId) setFieldId(d.fieldId);
    if (d.robotId) setRobotId(d.robotId);
    if (d.scoringId) setScoringId(d.scoringId);
    if (d.trainingId) setTrainingId(d.trainingId);
    setDefaultSeason(d.season || "");
    setLog(`Default bundle is ${d.season || d.fieldId} (${d.fieldId}).`);
  }

  async function start() {
    setLog("Queueing training…");
    const res = await postJson<{ runId: string }>("/runs", {
      demo,
      nEnvs: 4,
      budget: { totalEnvSteps: demo ? 4096 : 8192 },
      presets: { fieldId, robotId, scoringId },
    });
    openRun(res.runId);
    setLog(`Run ${res.runId} started. Leaderboard uses true score, not shaping.`);
  }

  async function cancel() {
    if (!current) return;
    await postJson(`/runs/${current.id}/cancel`, {});
    setLog(`Cancel requested for ${current.id}.`);
  }

  const metrics = (current?.metrics || {}) as Metrics;
  const liveFrame = rollout[rolloutI] || null;
  const fieldOptions = useMemo(() => fields, [fields]);

  function fmt(n: unknown) {
    return typeof n === "number" ? n.toFixed(3).replace(/\.?0+$/, "") : "—";
  }

  function pillClass(state: string) {
    if (state === "succeeded") return "pill ok";
    if (state === "failed" || state === "cancelled") return "pill bad";
    if (state === "running" || state === "queued") return "pill warn";
    return "pill";
  }

  return (
    <div className="page">
      <div className="scene">
        <div className="scene-label">{current ? `Rollout — ${current.id}` : "Rollout"}</div>
        {liveFrame ? (
          <FieldScene frame={liveFrame} showFov={false} />
        ) : (
          <div className="empty-scene note">No rollout frames yet.</div>
        )}
      </div>
      <aside className="side">
        <div className="page-head">
          <h2>Training dashboard</h2>
          <p className="note">True score is the leaderboard. Shaping is never ranked.</p>
        </div>
        <div className="card">
          <h3>True score (leaderboard)</h3>
          <Sparkline values={trueSeries} label="True score mean over time" color="#6fbfa3" />
          <div className="stat">mean {fmt(metrics.trueScoreMean)}</div>
        </div>
        <div className="card">
          <h3>Shaping (not leaderboard)</h3>
          <Sparkline values={shapeSeries} label="Shaping mean over time" color="#e0c36a" />
          <div className="stat">mean {fmt(metrics.shapingMean)}</div>
        </div>
        {current && (
          <div className="banner">
            <span className={pillClass(current.state)}>{current.state}</span> {current.id} · algo{" "}
            {String(metrics.algo || "—")} · steps {metrics.envSteps ?? 0}
            {metrics.replayId ? (
              <>
                {" "}
                · <Link to={`/replay/${metrics.replayId}`}>open replay</Link>
              </>
            ) : null}
          </div>
        )}
        <div className="row">
          <button className="primary" type="button" onClick={start}>
            Start run
          </button>
          <button type="button" onClick={cancel} disabled={!current}>
            Cancel
          </button>
        </div>
        <details open={!current}>
          <summary>Run configuration</summary>
          <label htmlFor="train-field">Field preset</label>
          <select id="train-field" value={fieldId} onChange={(e) => setFieldId(e.target.value)}>
            {fieldOptions.map((p) => (
              <option key={p.id} value={p.id}>
                {p.displayName || p.id}
              </option>
            ))}
          </select>
          <label htmlFor="train-robot">Robot preset</label>
          <select id="train-robot" value={robotId} onChange={(e) => setRobotId(e.target.value)}>
            {robots.map((p) => (
              <option key={p.id} value={p.id}>
                {p.displayName || p.id}
              </option>
            ))}
          </select>
          <label htmlFor="train-scoring">Scoring preset</label>
          <select id="train-scoring" value={scoringId} onChange={(e) => setScoringId(e.target.value)}>
            {scoring.map((p) => (
              <option key={p.id} value={p.id}>
                {p.displayName || p.id}
              </option>
            ))}
          </select>
          <p className="note">Default bundle{defaultSeason ? ` (${defaultSeason})` : ""} — used when a run omits preset ids.</p>
          <label htmlFor="train-bundle">Training preset</label>
          <select id="train-bundle" value={trainingId} onChange={(e) => setTrainingId(e.target.value)}>
            {training.map((p) => (
              <option key={p.id} value={p.id}>
                {p.displayName || p.id}
              </option>
            ))}
          </select>
          <div className="row">
            <button type="button" onClick={() => applyDefaults({ trainingId })}>
              Set as default
            </button>
            <button type="button" onClick={() => applyDefaults({ fieldId, robotId, scoringId })}>
              Use current selection
            </button>
          </div>
          <label className="check">
            <input type="checkbox" checked={demo} onChange={(e) => setDemo(e.target.checked)} /> demo budget (scripted
            fallback allowed)
          </label>
        </details>
        <p className="note">{log}</p>
        <h2>Runs</h2>
        {runs.map((r) => {
          const m = r.metrics as Metrics;
          return (
            <div key={r.id} className="card run-card">
              <button type="button" onClick={() => openRun(r.id)}>
                {r.id}
              </button>
              <span className={pillClass(r.state)}>{r.state}</span>
              <span className="stat" style={{ margin: 0 }}>
                true {fmt(m.trueScoreMean)} · shaping {fmt(m.shapingMean)} · {String(m.algo || "—")}
              </span>
              {m.replayId ? <Link to={`/replay/${m.replayId}`}>replay</Link> : null}
            </div>
          );
        })}
      </aside>
    </div>
  );
}
