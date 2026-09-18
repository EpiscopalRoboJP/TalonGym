import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { FieldScene } from "../scene/FieldScene";
import { buildSetupPreview, trainSceneFrame, type SetupFieldDoc } from "../scene/setupPreview";
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
  type MatchRobotSetup,
  type PresetMeta,
  type RobotPreset,
  type RunRow,
  type StartSlot,
} from "../api";
import { Empty, Field, Icon, NumberField, Panel, Segmented, Switch } from "../ui";

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
  evalLaunchCount?: number | null;
  evalWallContactS?: number | null;
  evalCheckpointHealthy?: boolean | null;
  evalScoredPieces?: number | null;
  healthWarnings?: string[];
  bestSkippedUnhealthy?: boolean | null;
  startupPhase?: string | null;
};

const STARTUP_PHASE_LABEL: Record<string, string> = {
  creating_envs: "Creating envs…",
  bc_warmup: "BC warmup…",
  scripted_baseline: "Checking scripted baseline…",
};

function startupPhaseLabel(phase?: string | null, nEnvs?: number) {
  if (phase === "creating_envs") return `Creating ${nEnvs ?? "—"} envs…`;
  if (phase && STARTUP_PHASE_LABEL[phase] && phase !== "training") return STARTUP_PHASE_LABEL[phase];
  return "";
}

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

function PresetSelect({
  id,
  label,
  value,
  onChange,
  options,
  compact = false,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: PresetMeta[];
  compact?: boolean;
}) {
  const known = options.some((p) => p.id === value);
  const select = (
    <select id={id} value={value} onChange={(e) => onChange(e.target.value)} data-testid={id}>
      {!known && value && <option value={value}>{value}</option>}
      {options.map((p) => (
        <option key={p.id} value={p.id}>
          {robotPresetLabel(p)}
        </option>
      ))}
    </select>
  );
  if (compact) {
    return (
      <label className="train-scene-select" htmlFor={id}>
        <span>{label}</span>
        {select}
      </label>
    );
  }
  return (
    <Field id={id} label={label}>
      {select}
    </Field>
  );
}

const ROBOT_IDS = ["red_0", "red_1", "blue_0", "blue_1"] as const;

function defaultRobotSetup(id: (typeof ROBOT_IDS)[number]): MatchRobotSetup {
  return {
    id,
    enabled: id === "red_0",
    dynamic: id === "red_0",
    startSlotId: id,
    offset: { x: 0, y: 0, headingDeg: 0 },
  };
}

function legalOffsetBounds(slot: StartSlot | undefined) {
  const legal = slot?.legalRegion;
  if (!legal) return { x: 0, y: 0 };
  if (legal.kind === "aabb") return { x: (legal.width || 0) / 2, y: (legal.depth || 0) / 2 };
  if (legal.kind === "circle") {
    const radius = legal.radius || 0;
    return { x: radius, y: radius };
  }
  return { x: 0, y: 0 };
}

function clampStartOffset(slot: StartSlot | undefined, offset: MatchRobotSetup["offset"]): MatchRobotSetup["offset"] {
  const bounds = legalOffsetBounds(slot);
  const heading = Number.isFinite(offset.headingDeg) ? offset.headingDeg : 0;
  return {
    x: Math.max(-bounds.x, Math.min(bounds.x, Number.isFinite(offset.x) ? offset.x : 0)),
    y: Math.max(-bounds.y, Math.min(bounds.y, Number.isFinite(offset.y) ? offset.y : 0)),
    headingDeg: Math.max(-180, Math.min(180, heading)),
  };
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
  const [customStarts, setCustomStarts] = useState(false);
  const [startSlots, setStartSlots] = useState<StartSlot[]>([]);
  const [fieldDoc, setFieldDoc] = useState<SetupFieldDoc | null>(null);
  const [robotDoc, setRobotDoc] = useState<RobotPreset | null>(null);
  const [robotSetup, setRobotSetup] = useState<MatchRobotSetup[]>(ROBOT_IDS.map(defaultRobotSetup));
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
    getJson<SetupFieldDoc>(`/presets/field/${fieldId}`)
      .then((field) => {
        setFieldDoc(field);
        setStartSlots(field.startSlots || []);
      })
      .catch(() => {
        setFieldDoc(null);
        setStartSlots([]);
      });
  }, [fieldId]);

  useEffect(() => {
    getJson<RobotPreset>(`/presets/robot/${robotId}`)
      .then(setRobotDoc)
      .catch(() => setRobotDoc(null));
  }, [robotId]);

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
      if (customStarts) {
        body.matchSetup = {
          robots: robotSetup.map((robot) => {
            const slot = startSlots.find((row) => row.id === robot.startSlotId);
            return { ...robot, offset: clampStartOffset(slot, robot.offset) };
          }),
        };
      }
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
  const startupNote = (metrics.envSteps ?? 0) === 0 ? startupPhaseLabel(metrics.startupPhase, metrics.nEnvs) : "";
  const statusLine = log || startupNote;
  const presets = (current?.config?.presets || {}) as Partial<DefaultsBundle>;
  const previewFrame = useMemo(
    () => buildSetupPreview(fieldDoc, robotDoc, { robotSetup, customStarts }),
    [fieldDoc, robotDoc, robotSetup, customStarts],
  );
  const sceneFrame = trainSceneFrame({
    liveFrame,
    previewFrame,
    runPresets: presets,
    fieldId,
    robotId,
  });
  const showingLive = Boolean(sceneFrame && liveFrame && sceneFrame === liveFrame);
  const robotLabel = robotPresetLabel(robots.find((p) => p.id === robotId) || { id: robotId, displayName: robotDoc?.displayName || robotId });
  const fieldLabel = robotPresetLabel(fields.find((p) => p.id === fieldId) || { id: fieldId, displayName: fieldDoc?.displayName || fieldId });

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
            <Switch checked={customStarts} onChange={setCustomStarts}>
              Configure robot starts
            </Switch>
            {customStarts && (
              <div className="stack">
                <p className="note">
                  G304 starts only. Offsets stay inside each slot&apos;s legal region so AUTO cannot begin already LEAVE- or PARK-qualified.
                </p>
                {robotSetup.map((robot) => {
                  const alliance = robot.id.startsWith("red") ? "red" : "blue";
                  const slot = startSlots.find((row) => row.id === robot.startSlotId);
                  const bounds = legalOffsetBounds(slot);
                  const slotTaken = robotSetup.some(
                    (other) => other.id !== robot.id && other.enabled && robot.enabled && other.startSlotId === robot.startSlotId,
                  );
                  return (
                    <div key={robot.id} className="stack" style={{ padding: "0.6rem 0", borderTop: "1px solid var(--line)" }}>
                      <div className="row">
                        <strong className="mono">{robot.id}</strong>
                        <Switch
                          checked={robot.enabled}
                          disabled={robot.id === "red_0"}
                          onChange={(enabled) =>
                            setRobotSetup((rows) => rows.map((row) => (row.id === robot.id ? { ...row, enabled } : row)))
                          }
                        >
                          On field
                        </Switch>
                        <Switch
                          checked={Boolean(robot.dynamic)}
                          onChange={(dynamic) =>
                            setRobotSetup((rows) => rows.map((row) => (row.id === robot.id ? { ...row, dynamic } : row)))
                          }
                        >
                          Dynamic
                        </Switch>
                      </div>
                      <Field id={`start-slot-${robot.id}`} label="Official slot">
                        <select
                          id={`start-slot-${robot.id}`}
                          value={robot.startSlotId}
                          onChange={(e) =>
                            setRobotSetup((rows) =>
                              rows.map((row) => (row.id === robot.id ? { ...row, startSlotId: e.target.value } : row)),
                            )
                          }
                        >
                          {startSlots
                            .filter((row) => row.alliance === alliance)
                            .map((row) => (
                              <option key={row.id} value={row.id}>
                                {row.id}
                              </option>
                            ))}
                        </select>
                      </Field>
                      {slotTaken && <p className="note">Two robots share this slot; they must not overlap.</p>}
                      <div className="fields two">
                        <NumberField
                          id={`${robot.id}-x`}
                          label="Off-wall"
                          unit="in"
                          value={robot.offset.x}
                          onChange={(n) =>
                            setRobotSetup((rows) =>
                              rows.map((row) =>
                                row.id === robot.id
                                  ? { ...row, offset: clampStartOffset(slot, { ...row.offset, x: n }) }
                                  : row,
                              ),
                            )
                          }
                        />
                        <NumberField
                          id={`${robot.id}-y`}
                          label="Along-wall"
                          unit="in"
                          value={robot.offset.y}
                          onChange={(n) =>
                            setRobotSetup((rows) =>
                              rows.map((row) =>
                                row.id === robot.id
                                  ? { ...row, offset: clampStartOffset(slot, { ...row.offset, y: n }) }
                                  : row,
                              ),
                            )
                          }
                        />
                      </div>
                      <span className="field-hint">
                        Max offset {bounds.x.toFixed(1)} × {bounds.y.toFixed(1)} in
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
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

      <div className="train-main">
        {current ? (
          <section className="panel">
            <header className="panel-head">
              <h2 className="mono">{current.id}</h2>
              <span className={statePill(current.state)}>{current.state}</span>
              <span className="sub">
                {String(metrics.algo || "—")}
                {presets.robotId ? ` · ${presets.robotId}` : ""}
                {presets.fieldId ? ` · ${presets.fieldId}` : ""}
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
            {statusLine && (
              <p
                className="note mono"
                data-testid="train-startup-phase"
                style={{ padding: "0.35rem 1rem 0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
              >
                {statusLine}
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
                <span className="label">Launches / wall</span>
                <span className="value sm" data-testid="train-eval-health">
                  {fmt(metrics.evalLaunchCount, 0)} / {fmt(metrics.evalWallContactS, 1)}s
                </span>
                <span className="hint">
                  {metrics.evalCheckpointHealthy === false
                    ? "unhealthy checkpoint skipped"
                    : metrics.evalCheckpointHealthy
                      ? "checkpoint healthy"
                      : "physical eval"}
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
            {metrics.healthWarnings?.length || metrics.bestSkippedUnhealthy ? (
              <p className="note" data-testid="train-health-warnings" role="status">
                {metrics.bestSkippedUnhealthy ? "Best checkpoint not updated: eval was unhealthy. " : ""}
                {(metrics.healthWarnings || []).join("; ")}
              </p>
            ) : null}
          </section>
        ) : (
          <section className="panel">
            <header className="panel-head">
              <h2>Setup preview</h2>
              <span className="sub">
                {robotLabel} on {fieldLabel}
              </span>
            </header>
            <div className="panel-body">
              <Empty title="No run selected">Change the robot or field below, then press Start, or pick a past run.</Empty>
            </div>
          </section>
        )}

        <div className="train-split">
          <Panel
            title="Display"
            sub={showingLive ? "latest policy episode, looping" : `setup preview · ${robotLabel}`}
            bodyClass="viewport"
            className="grow"
            testId="train-scene"
            actions={
              <div className="train-scene-controls">
                <PresetSelect id="train-scene-robot" label="Robot" value={robotId} onChange={setRobotId} options={robots} compact />
                <PresetSelect id="train-scene-field" label="Field" value={fieldId} onChange={setFieldId} options={fields} compact />
              </div>
            }
          >
            {sceneFrame ? (
              <FieldScene frame={sceneFrame} showFov={false} />
            ) : (
              <Empty dark title="Loading field">
                Choose a robot and field to preview the match setup.
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
                <span className="label">Reward extras</span>
                <span className="num">{fmt(metrics.shapingMean)}</span>
              </div>
              <Sparkline values={shapeSeries} label="Configured reward extras over time" color={theme.muted} />
              <p className="note" style={{ marginTop: "0.35rem" }}>
                Season reward terms other than true score. Ranking uses true score only.
              </p>
            </div>
          </Panel>
        </div>
      </div>
    </main>
  );
}
