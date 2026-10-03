import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Whisker } from "../Sparkline";
import { getJson, notify, postJson, runLabel, type RunRow } from "../api";
import { Alert, Empty, Field, Icon, Panel, RoadRunnerDialog } from "../ui";

type EvalRow = {
  id: string;
  runId?: string | null;
  createdAt?: string;
  report: {
    mean: number;
    lo: number;
    hi: number;
    p10: number;
    nTrials: number;
    bestLabelEligible: boolean;
    replayId?: string;
    collisionTimeMean?: number;
    restrictedEntryRate?: number;
    collisionRate?: number;
    policy?: string;
    runId?: string;
  };
};

const TRIALS = [24, 100, 500];

function n2(v: number | null | undefined) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "—";
}

export function ComparePage() {
  const [rows, setRows] = useState<EvalRow[] | null>(null);
  const [eligible, setEligible] = useState(false);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [runId, setRunId] = useState("");
  const [trials, setTrials] = useState(24);
  const [busy, setBusy] = useState<"scripted" | "checkpoint" | null>(null);
  const [exportId, setExportId] = useState<string | null>(null);

  async function refresh() {
    const cmp = await getJson<{ evaluations: EvalRow[]; bestLabelEligible: boolean }>("/comparisons/latest");
    setRows(cmp.evaluations || []);
    setEligible(cmp.bestLabelEligible);
    const list = await getJson<RunRow[]>("/runs");
    const done = list.filter((r) => r.state === "succeeded");
    setRuns(done);
    setRunId((prev) => prev || done[0]?.id || "");
  }

  useEffect(() => {
    refresh().catch(() => setRows([]));
  }, []);

  async function runEval(policy: "scripted" | "checkpoint") {
    setBusy(policy);
    try {
      const body: Record<string, unknown> = { nTrials: trials, policy };
      if (policy === "checkpoint") body.runId = runId;
      const res = await postJson<{ evaluationId: string; report?: { mean?: number } }>("/evaluations", body);
      await refresh();
      const mean = res.report?.mean;
      notify(`Evaluation ${res.evaluationId} saved${mean != null ? ` · mean ${mean.toFixed(2)}` : ""}.`);
    } finally {
      setBusy(null);
    }
  }

  const bounds = useMemo(() => {
    if (!rows?.length) return { min: 0, max: 1 };
    const lo = Math.min(...rows.map((r) => r.report.lo ?? 0));
    const hi = Math.max(...rows.map((r) => r.report.hi ?? 0));
    const pad = Math.max(0.5, (hi - lo) * 0.05);
    return { min: lo - pad, max: hi + pad };
  }, [rows]);

  return (
    <main className="page scroll layout-eval">
      <Panel
        title="Run an evaluation"
        sub="Scores a policy on held-out seeds with bootstrap 95% confidence intervals."
        actions={
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <label className="label" htmlFor="eval-trials">
              Trials
            </label>
            <select id="eval-trials" style={{ width: "auto" }} value={trials} onChange={(e) => setTrials(Number(e.target.value))} disabled={busy !== null}>
              {TRIALS.map((n) => (
                <option key={n} value={n}>
                  {n} episodes{n >= 500 ? " (can be labeled best)" : ""}
                </option>
              ))}
            </select>
          </div>
        }
      >
        <div className="eval-actions">
          <div className="eval-action">
            <h3>Scripted baseline</h3>
            <p>The hand-written AUTO routine. Use it as the bar a trained policy must clear.</p>
            <div className="row end" style={{ marginTop: "auto" }}>
              <button type="button" className="btn primary" onClick={() => runEval("scripted")} disabled={busy !== null}>
                {busy === "scripted" ? "Evaluating…" : "Evaluate"}
              </button>
            </div>
          </div>
          <div className="eval-action">
            <h3>Trained checkpoint</h3>
            <p>Loads the latest checkpoint saved by a finished training run, using that run's presets.</p>
            <div className="row" style={{ marginTop: "auto", alignItems: "flex-end" }}>
              <div className="grow">
                <Field id="ckpt-run" label="Training run">
                  <select id="ckpt-run" value={runId} onChange={(e) => setRunId(e.target.value)} disabled={busy !== null || runs.length === 0}>
                    {runs.length === 0 && <option value="">No finished runs</option>}
                    {runs.map((r) => (
                      <option key={r.id} value={r.id}>
                        {runLabel(r)} · {String((r.metrics as { algo?: string }).algo || "run")}
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
              <button type="button" className="btn primary" onClick={() => runEval("checkpoint")} disabled={!runId || busy !== null}>
                {busy === "checkpoint" ? "Evaluating…" : "Evaluate"}
              </button>
            </div>
          </div>
        </div>
        {busy && (
          <p className="note" style={{ marginTop: "0.8rem" }}>
            Running {trials} episodes. Larger evaluations can take several minutes; keep this tab open.
          </p>
        )}
      </Panel>

      <Panel title="Leaderboard" sub="Ranked by mean true score" bodyClass="panel-body flush">
        {rows && rows.length > 0 && (
          <div style={{ padding: "0.8rem 1rem", borderBottom: "1px solid var(--line)" }}>
            {eligible ? (
              <Alert kind="ok">
                <b>Rank 1 is statistically best.</b> Every row has at least 500 trials and the top two intervals do not overlap.
              </Alert>
            ) : (
              <Alert>
                No policy is labeled best yet. That needs 500-trial evaluations whose top two confidence intervals do not overlap.
              </Alert>
            )}
          </div>
        )}
        {rows && rows.length === 0 && (
          <Empty title="No evaluations yet">Evaluate the scripted baseline to create the first row.</Empty>
        )}
        {rows && rows.length > 0 && (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th style={{ width: 48 }}>#</th>
                  <th>Policy</th>
                  <th className="r">Mean</th>
                  <th style={{ minWidth: 220 }}>
                    <div className="axis">
                      <span>{bounds.min.toFixed(1)}</span>
                      <span>95% CI</span>
                      <span>{bounds.max.toFixed(1)}</span>
                    </div>
                  </th>
                  <th className="r">P10</th>
                  <th className="r">Trials</th>
                  <th className="r">Restricted</th>
                  <th className="r">Contact time</th>
                  <th>Created</th>
                  <th className="r">Actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => {
                  const policy = r.report.policy || "scripted";
                  const n = r.report.nTrials;
                  const rid = r.report.replayId;
                  return (
                    <tr key={r.id}>
                      <td>
                        <span className={`rank ${idx === 0 && eligible ? "first" : ""}`}>{idx + 1}</span>
                      </td>
                      <td>
                        <div className="row" style={{ gap: "0.4rem" }}>
                          <span className={`pill ${policy === "scripted" ? "" : "brand"}`}>{policy}</span>
                          {idx === 0 && eligible && <span className="pill ok">Best</span>}
                        </div>
                        <div className="mono muted" style={{ fontSize: 12, marginTop: 3 }}>
                          {r.id}
                          {r.report.runId ? ` · run ${r.report.runId}` : ""}
                        </div>
                      </td>
                      <td className="r num" style={{ fontWeight: 600, fontSize: 15 }}>
                        {n2(r.report.mean)}
                      </td>
                      <td>
                        <Whisker lo={r.report.lo} mean={r.report.mean} hi={r.report.hi} min={bounds.min} max={bounds.max} />
                        <div className="num muted" style={{ fontSize: 11, textAlign: "center" }}>
                          {n2(r.report.lo)} – {n2(r.report.hi)}
                        </div>
                      </td>
                      <td className="r num">{n2(r.report.p10)}</td>
                      <td className="r num">
                        {n}
                        {n < 500 && (
                          <div className="muted" style={{ fontSize: 11 }}>
                            below 500
                          </div>
                        )}
                      </td>
                      <td className="r num">{r.report.restrictedEntryRate != null ? `${(r.report.restrictedEntryRate * 100).toFixed(0)}%` : "—"}</td>
                      <td className="r num">{r.report.collisionTimeMean != null ? `${n2(r.report.collisionTimeMean)} s` : "—"}</td>
                      <td className="num muted" style={{ fontSize: 12, whiteSpace: "nowrap" }}>
                        {r.createdAt ? r.createdAt.replace("T", " ").slice(0, 16) : "—"}
                      </td>
                      <td className="r">
                        <div className="row end" style={{ flexWrap: "nowrap" }}>
                          {rid && (
                            <Link className="btn sm" to={`/replay/${rid}`} title="Best episode replay">
                              Replay
                            </Link>
                          )}
                          <button type="button" className="btn sm" disabled={!rid} onClick={() => setExportId(rid || null)} title="Road Runner export">
                            <Icon name="code" size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      <RoadRunnerDialog replayId={exportId} onClose={() => setExportId(null)} />
    </main>
  );
}
