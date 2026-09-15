import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Whisker } from "../Sparkline";
import { getJson, postJson, postText, type RunRow } from "../api";
import { DRIVE_EXPORT_NOTE } from "../labHonesty";

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
  const [busy, setBusy] = useState(false);

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

  async function runEval(policy: "scripted" | "checkpoint", nTrials = 24) {
    if (busy) return;
    const long = nTrials >= 500;
    setBusy(true);
    setMsg(
      long
        ? "Evaluating 500 trials inline (this can take a while)…"
        : policy === "checkpoint"
          ? "Evaluating checkpoint…"
          : "Evaluating scripted baseline…",
    );
    try {
      const body: Record<string, unknown> = { nTrials, policy };
      if (policy === "checkpoint") body.runId = runId;
      const res = await postJson<{ evaluationId: string; report?: { mean?: number; nTrials?: number } }>("/evaluations", body);
      await refresh();
      const mean = res.report?.mean;
      const n = res.report?.nTrials ?? nTrials;
      const unlabeled = n < 500 ? " 24-trial rows cannot earn “best” (n≥500)." : " n=500 can earn “best” if CIs do not overlap.";
      setMsg(mean != null ? `Evaluation stored (mean ${mean.toFixed(2)}).${unlabeled}` : `Evaluation stored.${unlabeled}`);
    } catch {
      setMsg("Evaluation failed.");
    } finally {
      setBusy(false);
    }
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
            24-trial rows cannot earn “best” (n≥500 and non-overlapping bootstrap CIs). Use the 500-trial eval to reach
            eligibility — the API clamps at 500 and runs inline, so the page waits.
          </p>
        </div>
        <div className="banner">
          n=24 rows stay unlabeled
          {eligible ? ". A stored n≥500 row may already be eligible if CIs do not overlap." : ", and CIs may also overlap."}
        </div>
        <div className="row">
          <button className="primary" type="button" onClick={() => runEval("scripted")} disabled={busy}>
            Run 24-trial evaluation (scripted)
          </button>
          <button type="button" onClick={() => runEval("scripted", 500)} disabled={busy}>
            500-trial eval (scripted)
          </button>
        </div>
        <p className="note">24-trial rows cannot earn “best”. n≥500 is required before that label can apply.</p>
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
          <button type="button" onClick={() => runEval("checkpoint")} disabled={!runId || busy}>
            Optional checkpoint eval (24)
          </button>
          <button type="button" onClick={() => runEval("checkpoint", 500)} disabled={!runId || busy}>
            500-trial checkpoint eval
          </button>
        </div>
        {rows.length > 0 && (
          <div className="axis-ticks" aria-hidden="true">
            <span>{bounds.min.toFixed(1)}</span>
            <span>{mid.toFixed(1)}</span>
            <span>{bounds.max.toFixed(1)}</span>
          </div>
        )}
        {rows.length === 0 && (
          <p className="note">
            No evaluations yet. Run a 24-trial scripted eval to add a row, or a 500-trial eval if you want a row that
            can earn “best”. Collision rate is shown when the report includes it.
          </p>
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
                collision time {r.report.collisionTimeMean?.toFixed?.(2) ?? "—"}s · collision rate{" "}
                {r.report.collisionRate != null ? `${(r.report.collisionRate * 100).toFixed(0)}%` : "—"} · restricted
                rate {r.report.restrictedEntryRate != null ? `${(r.report.restrictedEntryRate * 100).toFixed(0)}%` : "—"}
              </div>
              <div className="row">
                <button type="button" onClick={() => exportRow(r)} disabled={!r.report.replayId}>
                  Road Runner 1.0 (drive only)
                </button>
              </div>
              {java[r.id] && (
                <details open={openExport[r.id]}>
                  <summary>Road Runner export (drive only)</summary>
                  <p className="note">{DRIVE_EXPORT_NOTE}</p>
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
