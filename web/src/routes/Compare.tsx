import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Whisker } from "../Sparkline";
import { getJson, postJson, postText, type RunRow } from "../api";

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

export function ComparePage() {
  const [rows, setRows] = useState<EvalRow[]>([]);
  const [eligible, setEligible] = useState(false);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [runId, setRunId] = useState("");
  const [java, setJava] = useState<Record<string, string>>({});
  const [openExport, setOpenExport] = useState<Record<string, boolean>>({});
  const [msg, setMsg] = useState("");

  async function refresh() {
    const cmp = await getJson<{ evaluations: EvalRow[]; bestLabelEligible: boolean }>("/comparisons/latest");
    setRows(cmp.evaluations || []);
    setEligible(cmp.bestLabelEligible);
    const list = await getJson<RunRow[]>("/runs");
    setRuns(list);
    if (!runId && list[0]) setRunId(list[0].id);
  }

  useEffect(() => {
    refresh();
  }, []);

  async function runEval(policy: "scripted" | "checkpoint") {
    setMsg(policy === "checkpoint" ? "Evaluating checkpoint…" : "Evaluating scripted baseline…");
    const body: Record<string, unknown> = { nTrials: 24, policy };
    if (policy === "checkpoint") body.runId = runId;
    const res = await postJson<{ evaluationId: string; report?: { mean?: number } }>("/evaluations", body);
    await refresh();
    const mean = res.report?.mean;
    setMsg(
      mean != null
        ? `Evaluation stored (mean ${mean.toFixed(2)}). 24-trial rows stay unlabeled.`
        : "Evaluation stored. 24-trial rows stay unlabeled.",
    );
  }

  async function exportRow(row: EvalRow) {
    const rid = row.report.replayId;
    if (!rid) return;
    const text = await postText(`/replays/${rid}/export/roadrunner`, { dialect: "rr1_actions" });
    setJava((m) => ({ ...m, [row.id]: text }));
    setOpenExport((m) => ({ ...m, [row.id]: true }));
  }

  const bounds = useMemo(() => {
    if (!rows.length) return { min: 0, max: 1 };
    const lo = Math.min(...rows.map((r) => r.report.lo ?? 0));
    const hi = Math.max(...rows.map((r) => r.report.hi ?? 0));
    const pad = Math.max(0.5, (hi - lo) * 0.05) || 1;
    return { min: lo - pad, max: hi + pad };
  }, [rows]);

  const mid = (bounds.min + bounds.max) / 2;

  return (
    <div className="page single">
      <div>
        <div className="page-head">
          <h2>Comparison / leaderboard</h2>
          <p className="note">
            “Best” is a pre-registered statistical objective with bootstrap CIs. This UI will not label a winner if CIs
            overlap or n&lt;500.
          </p>
        </div>
        {!eligible && (
          <div className="banner">No strategy is labeled statistically best yet (n&lt;500 or overlapping intervals).</div>
        )}
        <div className="row">
          <button className="primary" type="button" onClick={() => runEval("scripted")}>
            Run 24-trial evaluation (scripted)
          </button>
        </div>
        <div className="form-grid">
          <label htmlFor="ckpt-run">Checkpoint run</label>
          <select id="ckpt-run" value={runId} onChange={(e) => setRunId(e.target.value)}>
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id} · {r.state}
              </option>
            ))}
          </select>
        </div>
        <div className="row">
          <button type="button" onClick={() => runEval("checkpoint")} disabled={!runId}>
            Optional checkpoint eval
          </button>
        </div>
        {rows.length > 0 && (
          <div className="axis-ticks" aria-hidden="true">
            <span>{bounds.min.toFixed(1)}</span>
            <span>{mid.toFixed(1)}</span>
            <span>{bounds.max.toFixed(1)}</span>
          </div>
        )}
        {rows.map((r) => {
          const policy = r.report.policy || "scripted";
          const n = r.report.nTrials;
          return (
            <div key={r.id} className="card">
              <div className="run-card">
                <span className="pill">{policy}</span>
                <b className="mono">{r.id}</b>
                {r.createdAt && <span className="stat" style={{ margin: 0 }}>{r.createdAt.replace("T", " ").slice(0, 19)}</span>}
                {n < 500 && <span className="pill warn">n&lt;500</span>}
              </div>
              <div className="stat">
                <b>{r.report.mean?.toFixed(2)}</b> [{r.report.lo?.toFixed(2)}, {r.report.hi?.toFixed(2)}] · p10{" "}
                {r.report.p10?.toFixed(2)} · n={n}
                {r.report.replayId ? (
                  <>
                    {" "}
                    <Link to={`/replay/${r.report.replayId}`}>replay</Link>
                  </>
                ) : null}
              </div>
              <Whisker lo={r.report.lo} mean={r.report.mean} hi={r.report.hi} min={bounds.min} max={bounds.max} />
              <div className="stat">
                collision time {r.report.collisionTimeMean?.toFixed?.(2) ?? "—"}s · restricted rate{" "}
                {r.report.restrictedEntryRate != null ? (r.report.restrictedEntryRate * 100).toFixed(0) + "%" : "—"}
              </div>
              <div className="row">
                <button type="button" onClick={() => exportRow(r)} disabled={!r.report.replayId}>
                  Road Runner 1.0
                </button>
              </div>
              {java[r.id] && (
                <details open={openExport[r.id]}>
                  <summary>Road Runner export</summary>
                  <pre className="note mono rr-export" tabIndex={0}>
                    {java[r.id]}
                  </pre>
                </details>
              )}
            </div>
          );
        })}
        <p className="note">{msg}</p>
      </div>
    </div>
  );
}
