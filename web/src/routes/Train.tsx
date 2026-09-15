import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { FieldScene } from "../scene/FieldScene";
import { Sparkline } from "../Sparkline";
import { theme } from "../theme";
import {
  getJson,
  loadReplayFrames,
  notify,
  postJson,
  putJson,
  robotPresetLabel,
  statePill,
  type ComputeInfo,
  type DefaultsBundle,
  type Frame,
  type PresetMeta,
  type RunRow,
} from "../api";
import { Empty, Field, Icon, Panel, Segmented } from "../ui";

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

const BUDGETS: Record<Profile, { label: string; blurb: string }> = {
  demo: { label: "Demo", blurb: "4,096 steps on 2 envs. Falls back to the scripted policy if RL extras are missing." },
  short: { label: "Short", blurb: "16,384-step RecurrentPPO smoke test on 4 envs." },
  easy: { label: "Easy", blurb: "Autodetects this machine and uses the season's easy run config." },
  preset: { label: "Preset", blurb: "Uses the step budget and env count from the selected training preset." },
};

function fmt(n: unknown, digits = 2) {
  return typeof n === "number" && Number.isFinite(n) ? n.toFixed(digits).replace(/\.?0+$/, "") : "—";
}

function fmtSteps(n: unknown) {
  if (typeof n !== "number") return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function PresetSelect({ id, label, value, onChange, options }: { id: string; label: string; value: string; onChange: (v: string) => void; options: PresetMeta[] }) {
  return (
    <Field id={id} label={label}>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((p) => (
          <option key={p.id} value={p.id}>
            {robotPresetLabel(p)}
          </option>
        ))}
      </select>
    </Field>
  );
}

export function TrainPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [runs, setRuns] = useState<RunRow[] | null>(null);
  const [current, setCurrent] = useState<RunRow | null>(null);
  const [log, setLog] = useState("");
  const [fields, setFields] = useState<PresetMeta[]>([]);
  const [robots, setRobots] = useState<PresetMeta[]>([]);
  const [scoring, setScoring] = useState<PresetMeta[]>([]);
  const [training, setTraining] = useState<PresetMeta[]>([]);
  const [fieldId, setFieldId] = useState("biobuzz_2026_field_v1");
  const [robotId, setRobotId] = useState("mecanum_biobuzz_4cap");
  const [scoringId, setScoringId] = useState("biobuzz_2026_scoring_v1");
  const [trainingId, setTrainingId] = useState("biobuzz_auto_lightweight");
  const [profile, setProfile] = useState<Profile>("demo");
  const [compute, setCompute] = useState<ComputeInfo | null>(null);
  const [trueSeries, setTrueSeries] = useState<number[]>([]);
  const [evalSeries, setEvalSeries] = useState<number[]>([]);
  const [shapeSeries, setShapeSeries] = useState<number[]>([]);
  const [rollout, setRollout] = useState<Frame[]>([]);
  const [rolloutI, setRolloutI] = useState(0);
  const [starting, setStarting] = useState(false);
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
      })
      .catch(() => undefined);
    refreshList()
      .then((list) => {
        if (!runId && list[0]) openRun(list[0].id);
      })
      .catch(() => setRuns([]));
    // Initial load only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (runId) openRun(runId);
    // Only react to URL changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  useEffect(() => {
    const id = window.setInterval(async () => {
      const list = await refreshList();
      if (!current) return;
      const row = list.find((r) => r.id === current.id) || (await getJson<RunRow>(`/runs/${current.id}`));
      if (row) {
        setCurrent(row);
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
    // Poll loop is keyed to the selected run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    setLog("");
    navigate(`/train/${id}`, { replace: true });
    getJson<RunRow>(`/runs/${id}`).then((row) => {
      setCurrent(row);
      const m = row.metrics as Metrics;
      pushSeries(m);
      if (m.replayId) {
        replayLoadedRef.current = m.replayId;
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
          setCurrent((prev) => (prev && prev.id === id ? { ...prev, metrics: { ...(prev.metrics || {}), ...msg.payload } } : prev));
        }
        if (msg.type === "status" && msg.payload) {
          setCurrent((prev) => (prev && prev.id === id ? { ...prev, state: msg.payload.state || prev.state } : prev));
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

  async function saveDefaults() {
    const d = await putJson<DefaultsBundle>("/defaults", { trainingId, fieldId, robotId, scoringId });
    notify(`Saved defaults (${d.season || d.fieldId}). CLI jobs and builders now start from this selection.`);
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
    setStarting(true);
    try {
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
      await refreshList();
      openRun(res.runId);
      notify(`Run ${res.runId} queued.`);
    } finally {
      setStarting(false);
    }
  }

  async function cancel() {
    if (!current) return;
    await postJson(`/runs/${current.id}/cancel`, {});
    notify(`Cancel requested for ${current.id}.`, "info");
  }

  const metrics = (current?.metrics || {}) as Metrics;
  const liveFrame = rollout[rolloutI] || null;
  const busy = current?.state === "running" || current?.state === "queued";
  const progress = typeof metrics.progressFrac === "number" ? Math.max(0, Math.min(1, metrics.progressFrac)) : 0;
  const presets = (current?.config?.presets || {}) as Partial<DefaultsBundle>;

  const easyBlurb = compute
    ? `Detected ${compute.profile}: ${compute.hardware.cpuCount} cores${compute.hardware.ramGb != null ? `, ${compute.hardware.ramGb.toFixed(0)} GB RAM` : ""}${compute.hardware.cuda ? ", CUDA" : ""}. Runs ${compute.nEnvs} envs with the season's easy config.`
    : BUDGETS.easy.blurb;

  return (
    <main className="page layout-train">
      <div className="col">
        <Panel
          title="New training run"
          footer={
            <>
              <button type="button" className="btn primary" style={{ flex: 1 }} onClick={start} disabled={starting || busy}>
                <Icon name="play" size={14} /> {starting ? "Starting…" : "Start run"}
              </button>
              <button type="button" className="btn" onClick={saveDefaults} title="Use these presets by default in the CLI and builders">
                Save as default
              </button>
            </>
          }
        >
          <div className="stack">
            <div className="field">
              <span className="label">Budget</span>
              <Segmented label="Training budget" full value={profile} onChange={selectBudget} options={(Object.keys(BUDGETS) as Profile[]).map((k) => [k, BUDGETS[k].label])} />
              <span className="field-hint">{profile === "easy" ? easyBlurb : BUDGETS[profile].blurb}</span>
            </div>
            <PresetSelect id="train-bundle" label="Training preset" value={trainingId} onChange={setTrainingId} options={training} />
            <PresetSelect id="train-robot" label="Robot" value={robotId} onChange={setRobotId} options={robots} />
            <PresetSelect id="train-field" label="Field" value={fieldId} onChange={setFieldId} options={fields} />
            <PresetSelect id="train-scoring" label="Scoring" value={scoringId} onChange={setScoringId} options={scoring} />
            {busy && <p className="note">The selected run is still in progress. Wait for it to finish or cancel it first.</p>}
          </div>
        </Panel>

        <Panel className="grow" title="Runs" sub={runs ? String(runs.length) : undefined} bodyClass="panel-body flush scroll">
          {runs && runs.length === 0 && <Empty title="No runs yet">Start a Demo run to check your setup.</Empty>}
          <ul className="list">
            {(runs || []).map((r) => {
              const m = r.metrics as Metrics;
              return (
                <li key={r.id}>
                  <button type="button" className={`list-item ${current?.id === r.id ? "on" : ""}`} onClick={() => openRun(r.id)}>
                    <span className="grow">
                      <span className="title mono">{r.id}</span>
                      <span className="meta" style={{ display: "block" }}>
                        {String(m.algo || "—")} · {fmtSteps(m.envSteps)} steps
                      </span>
                    </span>
                    <span className="end">
                      <span className={statePill(r.state)}>{r.state}</span>
                      <span className="meta num" style={{ display: "block", marginTop: 2 }}>
                        score {fmt(m.trueScoreMean)}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </Panel>
      </div>

      {!current ? (
        <Panel className="grow">
          <Empty title="No run selected">Configure a run on the left and press Start, or pick a past run.</Empty>
        </Panel>
      ) : (
        <div className="train-main">
          <section className="panel">
            <header className="panel-head">
              <h2 className="mono">{current.id}</h2>
              <span className={statePill(current.state)}>{current.state}</span>
              <span className="sub">
                {String(metrics.algo || "—")}
                {presets.robotId ? ` · ${presets.robotId}` : ""}
                {presets.trainingId ? ` · ${presets.trainingId}` : ""}
              </span>
              <span className="spacer" />
              {metrics.replayId && (
                <Link className="btn sm" to={`/replay/${metrics.replayId}`}>
                  Open replay
                </Link>
              )}
              <Link className="btn sm" to="/compare">
                <Icon name="chart" size={14} /> Evaluate
              </Link>
              {busy && (
                <button type="button" className="btn sm danger" onClick={cancel}>
                  <Icon name="stop" size={12} /> Cancel
                </button>
              )}
            </header>
            <div style={{ padding: "0.7rem 1rem 0", display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <div className="progress" style={{ flex: 1 }} aria-label="Progress">
                <span style={{ width: `${progress * 100}%` }} />
              </div>
              <span className="num muted" style={{ fontSize: 12 }}>
                {(progress * 100).toFixed(0)}%
              </span>
            </div>
            {log && (
              <p className="note mono" style={{ padding: "0.35rem 1rem 0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {log}
              </p>
            )}
            <div className="stats" style={{ marginTop: "0.4rem", borderTop: "1px solid var(--line)" }}>
              <div className="stat">
                <span className="label">True score</span>
                <span className="value">{fmt(metrics.trueScoreMean)}</span>
                <span className="hint">episode mean · ranked</span>
              </div>
              <div className="stat">
                <span className="label">Held-out eval</span>
                <span className="value">{fmt(metrics.evalTrueScoreMean)}</span>
                <span className="hint">true score on unseen seeds</span>
              </div>
              <div className="stat">
                <span className="label">Env steps</span>
                <span className="value">{fmtSteps(metrics.envSteps)}</span>
                <span className="hint">{metrics.nEnvs ?? "—"} envs{metrics.fps != null ? ` · ${fmt(metrics.fps, 0)} fps` : ""}</span>
              </div>
              <div className="stat">
                <span className="label">Curriculum</span>
                <span className="value">{metrics.curriculumStage ?? "—"}</span>
                <span className="hint" title={metrics.curriculumUnlock?.join(", ")}>
                  {metrics.curriculumUnlock?.length ? metrics.curriculumUnlock.join(", ") : "stage"}
                </span>
              </div>
              <div className="stat">
                <span className="label">Entropy / KL</span>
                <span className="value sm">
                  {fmt(metrics.entropy)} / {fmt(metrics.approxKl, 4)}
                </span>
                <span className="hint">policy health</span>
              </div>
            </div>
          </section>

          <div className="train-split">
            <Panel title="Rollout" sub="latest policy episode, looping" bodyClass="viewport" className="grow">
              {liveFrame ? (
                <FieldScene frame={liveFrame} showFov={false} />
              ) : (
                <Empty dark title="No rollout yet">
                  Frames appear once the run records an episode.
                </Empty>
              )}
            </Panel>
            <Panel title="Learning curves" bodyClass="panel-body scroll" className="grow">
              <div className="chart-block">
                <div className="chart-head">
                  <span className="label">True score</span>
                  <span className="num">{fmt(metrics.trueScoreMean)}</span>
                </div>
                <Sparkline values={trueSeries} label="True score mean over time" color={theme.brand} />
              </div>
              <div className="chart-block">
                <div className="chart-head">
                  <span className="label">Held-out eval</span>
                  <span className="num">{fmt(metrics.evalTrueScoreMean)}</span>
                </div>
                <Sparkline values={evalSeries} label="Eval true score mean over time" color={theme.goldDark} />
              </div>
              <div className="chart-block">
                <div className="chart-head">
                  <span className="label">Shaping reward</span>
                  <span className="num">{fmt(metrics.shapingMean)}</span>
                </div>
                <Sparkline values={shapeSeries} label="Shaping mean over time" color={theme.muted} />
                <p className="note" style={{ marginTop: "0.35rem" }}>
                  Training signal only. Never used for ranking.
                </p>
              </div>
            </Panel>
          </div>
        </div>
      )}
    </main>
  );
}
