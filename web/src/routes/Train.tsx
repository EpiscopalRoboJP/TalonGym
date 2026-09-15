import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { FieldScene } from "../scene/FieldScene";
import { Sparkline } from "../Sparkline";
import { theme } from "../theme";
import { getJson, loadReplayFrames, postJson, putJson, robotPresetLabel, type ComputeInfo, type DefaultsBundle, type Frame, type PresetMeta, type RunRow } from "../api";

type Profile = "demo" | "short" | "easy" | "preset";

type Metrics = {
  envSteps?: number;
  nEnvs?: number;
  algo?: string;
  replayId?: string;
  trueScoreMean?: number | null;
  objectiveMean?: number | null;
  shapingMean?: number | null;
  evalTrueScoreMean?: number | null;
  entropy?: number | null;
  approxKl?: number | null;
  progressFrac?: number | null;
  curriculumStage?: number | string | null;
  curriculumUnlock?: string[];
  fps?: number | null;
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
  const [fieldId, setFieldId] = useState("biobuzz_2026_field_v1");
  const [robotId, setRobotId] = useState("mecanum_biobuzz_4cap");
  const [scoringId, setScoringId] = useState("biobuzz_2026_scoring_v1");
  const [trainingId, setTrainingId] = useState("biobuzz_auto_lightweight");
  const [defaultSeason, setDefaultSeason] = useState("");
  const [profile, setProfile] = useState<Profile>("demo");
  const [compute, setCompute] = useState<ComputeInfo | null>(null);
  const [trueSeries, setTrueSeries] = useState<number[]>([]);
  const [evalSeries, setEvalSeries] = useState<number[]>([]);
  const [shapeSeries, setShapeSeries] = useState<number[]>([]);
  const [rollout, setRollout] = useState<Frame[]>([]);
  const [rolloutI, setRolloutI] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const watchingRef = useRef<string | null>(null);
  const replayLoadedRef = useRef<string | null>(null);

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
    getJson<ComputeInfo>("/compute").then(setCompute).catch(() => undefined);
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
        setCurrent((prev) => mergeRun(row, prev));
        const m = row.metrics as Metrics;
        pushSeries(m);
        if (m.replayId && replayLoadedRef.current !== m.replayId) {
          replayLoadedRef.current = m.replayId;
          loadReplayFrames(m.replayId).then((frames) => {
            setRollout((prev) => (prev.length ? prev : frames.filter((_, idx) => idx % 5 === 0)));
            setRolloutI(0);
          });
        }
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
    if (typeof m.evalTrueScoreMean === "number") {
      setEvalSeries((s) => (s.length && s[s.length - 1] === m.evalTrueScoreMean ? s : [...s.slice(-80), m.evalTrueScoreMean as number]));
    }
    if (typeof m.shapingMean === "number") {
      setShapeSeries((s) => (s.length && s[s.length - 1] === m.shapingMean ? s : [...s.slice(-80), m.shapingMean as number]));
    }
  }

  function mergeRun(row: RunRow, live: RunRow | null): RunRow {
    if (!live || live.id !== row.id) return row;
    const liveSteps = Number((live.metrics as Metrics)?.envSteps || 0);
    const rowSteps = Number((row.metrics as Metrics)?.envSteps || 0);
    if (liveSteps <= rowSteps) return row;
    return { ...row, metrics: { ...(row.metrics || {}), ...(live.metrics || {}) } };
  }

  function applyMetrics(id: string, payload: Metrics) {
    pushSeries(payload);
    setCurrent((prev) => {
      if (prev && prev.id !== id) return prev;
      const base = prev ?? { id, state: "running", metrics: {}, config: {} };
      return { ...base, id, metrics: { ...(base.metrics || {}), ...payload } };
    });
  }

  function openRun(id: string) {
    if (watchingRef.current === id && wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
      return;
    }
    watchingRef.current = id;
    replayLoadedRef.current = null;
    setTrueSeries([]);
    setEvalSeries([]);
    setShapeSeries([]);
    setRollout([]);
    navigate(`/train/${id}`, { replace: true });
    getJson<RunRow>(`/runs/${id}`).then((row) => {
      setCurrent((prev) => mergeRun(row, prev && prev.id === id ? prev : null));
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
          applyMetrics(id, msg.payload as Metrics);
        }
        if (msg.type === "status" && msg.payload) {
          setCurrent((prev) => {
            if (prev && prev.id !== id) return prev;
            const base = prev ?? { id, state: "running", metrics: {}, config: {} };
            return { ...base, id, state: msg.payload.state || base.state };
          });
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

  function familyTrainingId(id: string, suffix: "lightweight" | "workstation" | "cloud" | "easy") {
    const base = id.replace(/_(lightweight|workstation|cloud|easy)$/, "");
    return `${base}_${suffix}`;
  }

  function selectBudget(next: Profile) {
    setProfile(next);
    if (next === "easy") {
      const easyId = familyTrainingId(trainingId, "easy");
      if (training.some((p) => p.id === easyId)) setTrainingId(easyId);
    }
  }

  async function start() {
    setLog("Queueing training…");
    const body: Record<string, unknown> = {
      demo: profile === "demo",
      easy: profile === "easy",
      presets: { fieldId, robotId, scoringId, trainingId: profile === "easy" ? familyTrainingId(trainingId, "easy") : trainingId },
    };
    if (profile === "demo") {
      body.nEnvs = 2;
      body.budget = { totalEnvSteps: 4096 };
    } else if (profile === "short") {
      body.nEnvs = 4;
      body.budget = { totalEnvSteps: 16384 };
    } else if (profile === "easy") {
      body.computeProfile = "auto";
    }
    const res = await postJson<{ runId: string }>("/runs", body);
    openRun(res.runId);
    setLog(`Run ${res.runId} started. Leaderboard uses true score, not shaping.`);
  }

  async function cancel() {
    if (!current) return;
    setLog(`Cancel requested for ${current.id}.`);
    await postJson(`/runs/${current.id}/cancel`, {});
  }

  const metrics = (current?.metrics || {}) as Metrics;
  const liveFrame = rollout[rolloutI] || null;
  const fieldOptions = useMemo(() => fields, [fields]);
  const busy = current?.state === "running" || current?.state === "queued";
  const progress = typeof metrics.progressFrac === "number" ? Math.max(0, Math.min(1, metrics.progressFrac)) : 0;

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
          <p className="note">True score is the leaderboard. Shaping is never ranked. RecurrentPPO keeps Dict observations and an LSTM.</p>
        </div>
        <div className="card">
          <h3>True score (leaderboard)</h3>
          <Sparkline values={trueSeries} label="True score mean over time" color={theme.gold} />
          <div className="stat">episode mean {fmt(metrics.trueScoreMean)}</div>
        </div>
        <div className="card">
          <h3>Held-out eval true score</h3>
          <Sparkline values={evalSeries} label="Eval true score mean over time" color={theme.goldLight} />
          <div className="stat">eval mean {fmt(metrics.evalTrueScoreMean)}</div>
        </div>
        <div className="card">
          <h3>Shaping (not leaderboard)</h3>
          <Sparkline values={shapeSeries} label="Shaping mean over time" color={theme.goldBright} />
          <div className="stat">mean {fmt(metrics.shapingMean)}</div>
        </div>
        {current && (
          <div className="banner">
            <span className={pillClass(current.state)}>{current.state}</span> {current.id} · algo{" "}
            {String(metrics.algo || "—")} · {metrics.nEnvs ?? "—"} envs · steps {metrics.envSteps ?? 0}
            {metrics.replayId ? (
              <>
                {" "}
                · <Link to={`/replay/${metrics.replayId}`}>open replay</Link>
                {" · "}
                <Link to="/compare">compare</Link>
              </>
            ) : null}
            <div className="meter" aria-hidden="true">
              <span style={{ width: `${progress * 100}%` }} />
            </div>
            <div className="stat" style={{ marginBottom: 0 }}>
              stage {metrics.curriculumStage ?? "—"}
              {metrics.curriculumUnlock?.length ? ` · ${metrics.curriculumUnlock.join(", ")}` : ""}
              {" · "}entropy {fmt(metrics.entropy)} · KL {fmt(metrics.approxKl)}
              {metrics.fps != null ? ` · ${fmt(metrics.fps)} fps` : ""}
            </div>
          </div>
        )}
        <div className="row">
          <button className="primary" type="button" onClick={start} disabled={busy}>
            Start run
          </button>
          <button type="button" onClick={cancel} disabled={!current || !busy}>
            Cancel
          </button>
        </div>
        <label>Budget</label>
        <div className="seg" role="group" aria-label="Training budget">
          <button type="button" className={profile === "demo" ? "on" : ""} onClick={() => selectBudget("demo")}>
            Demo
          </button>
          <button type="button" className={profile === "short" ? "on" : ""} onClick={() => selectBudget("short")}>
            Short
          </button>
          <button type="button" className={profile === "easy" ? "on" : ""} onClick={() => selectBudget("easy")}>
            Easy
          </button>
          <button type="button" className={profile === "preset" ? "on" : ""} onClick={() => selectBudget("preset")}>
            Preset
          </button>
        </div>
        <p className="note">
          {profile === "demo" && "4,096 steps, 2 envs, scripted fallback if RL extras are missing."}
          {profile === "short" && "16,384-step RecurrentPPO on 4 envs. No scripted fallback."}
          {profile === "easy" &&
            (compute
              ? `Autodetect ${compute.profile} · ${compute.hardware.cpuCount} cores · ${compute.hardware.ramGb != null ? `${compute.hardware.ramGb.toFixed(0)} GB` : "RAM n/a"} · ${compute.nEnvs} envs${compute.hardware.cuda ? " · CUDA" : ""}. Uses the season easy run config.`
              : "Autodetects laptop / workstation / cloud and uses that machine's easy run config.")}
          {profile === "preset" && "Uses the training preset budget and nEnvs (lightweight, workstation, or cloud)."}
        </p>
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
                {robotPresetLabel(p)}
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
                true {fmt(m.trueScoreMean)} · eval {fmt(m.evalTrueScoreMean)} · {String(m.algo || "—")}
              </span>
              {m.replayId ? <Link to={`/replay/${m.replayId}`}>replay</Link> : null}
            </div>
          );
        })}
      </aside>
    </div>
  );
}
