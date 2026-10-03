import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { FieldScene, type SceneView } from "../scene/FieldScene";
import { getJson, loadReplayFrames, notify, postJson, replayLabel, replayMetaLine, type Frame, type FrameExplain, type ReplayListRow } from "../api";
import { Empty, Icon, Panel, RoadRunnerDialog, Segmented, Switch } from "../ui";
import { KeyValues } from "../KeyValues";
import { namedQueues, resolveReplayId } from "../labHonesty";

type ReplayMeta = ReplayListRow;
type Marker = { i: number; kind: "foul" | "contact" };

function signedPoints(n: number) {
  if (n > 0) return `+${n}`;
  if (n < 0) return `−${Math.abs(n)}`;
  return "0";
}

function isFoulExplain(e: FrameExplain) {
  return e.points < 0;
}

function frameHasFoulStep(frame: Frame | null | undefined) {
  return (frame?.stepExplains || []).some(isFoulExplain);
}

function frameHasContact(frame: Frame | null | undefined) {
  return Boolean(frame?.collision?.wall || frame?.collision?.robot);
}

function contactParts(frame: Frame | null | undefined) {
  const parts: string[] = [];
  if (frame?.collision?.wall) parts.push("wall");
  if (frame?.collision?.robot) parts.push("robot");
  if (frame?.collision?.piece) parts.push("piece");
  return parts;
}

export function ReplayPage() {
  const { replayId } = useParams();
  const navigate = useNavigate();
  const [ids, setIds] = useState<ReplayMeta[] | null>(null);
  const [active, setActive] = useState(replayId || "");
  const selectedId = resolveReplayId(replayId, active);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [loading, setLoading] = useState(false);
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showFov, setShowFov] = useState(true);
  const [view, setView] = useState<SceneView>("threeQuarter");
  const [exportId, setExportId] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);

  useEffect(() => {
    if (replayId && replayId !== active) setActive(replayId);
    // Only react to URL changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayId]);

  useEffect(() => {
    getJson<ReplayMeta[]>("/replays")
      .then((list) => {
        setIds(list);
        if (!active && list[0]) selectReplay(list[0].id);
      })
      .catch(() => setIds([]));
    // Initial load only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    setLoading(true);
    setPlaying(false);
    loadReplayFrames(selectedId)
      .then((f) => {
        if (cancelled) return;
        setFrames(f);
        setI(0);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!playing || frames.length === 0) return;
    const id = window.setInterval(() => {
      setI((n) => {
        if (n + 1 >= frames.length) {
          setPlaying(false);
          return n;
        }
        return n + 1;
      });
    }, 40 / speed);
    return () => window.clearInterval(id);
  }, [playing, frames, speed]);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const tag = (ev.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (tag === "BUTTON" && (ev.key === " " || ev.key === "Enter")) return;
      if (document.querySelector("dialog[open]")) return;
      if (ev.key === " " || ev.code === "Space") {
        ev.preventDefault();
        togglePlay();
      } else if (ev.key === "ArrowRight") {
        ev.preventDefault();
        setPlaying(false);
        setI((n) => Math.min(n + 1, Math.max(0, frames.length - 1)));
      } else if (ev.key === "ArrowLeft") {
        ev.preventDefault();
        setPlaying(false);
        setI((n) => Math.max(0, n - 1));
      } else if (ev.key === "Home") {
        ev.preventDefault();
        setPlaying(false);
        setI(0);
      } else if (ev.key === "End") {
        ev.preventDefault();
        setPlaying(false);
        setI(Math.max(0, frames.length - 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // togglePlay only reads frames/i, both covered.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames.length, i]);

  const frame = frames[i] || null;
  const duration = frames.length ? frames[frames.length - 1].t : 0;
  const path = useMemo(
    () =>
      frames
        .filter((_, idx) => idx % 5 === 0)
        .map((f) => f.robots.find((r) => r.dynamic) || f.robots[0])
        .filter(Boolean)
        .map((r) => ({ x: r.x, y: r.y })),
    [frames],
  );
  const markers = useMemo(() => {
    const out: Marker[] = [];
    let prevContact = false;
    frames.forEach((f, idx) => {
      if (frameHasFoulStep(f)) out.push({ i: idx, kind: "foul" });
      const contact = frameHasContact(f);
      if (contact && !prevContact) out.push({ i: idx, kind: "contact" });
      prevContact = contact;
    });
    return out;
  }, [frames]);
  const penalties = frame?.penalties || [];
  const scoreExplains = (frame?.explains || []).filter((e) => e.points >= 0);
  const foulNow = frameHasFoulStep(frame) || (frame?.robots || []).some((r) => r.dynamic && r.enteredRestricted);
  const contact = contactParts(frame);
  const robot = frame?.robots.find((r) => r.dynamic) || frame?.robots[0];
  const meta = ids?.find((r) => r.id === active);

  function selectReplay(id: string) {
    setActive(id);
    navigate(`/replay/${id}`, { replace: true });
  }

  function togglePlay() {
    if (!frames.length) return;
    if (!playing && i >= frames.length - 1) setI(0);
    setPlaying((p) => !p);
  }

  async function recordDemo() {
    setRecording(true);
    try {
      const res = await postJson<{ replayId: string }>("/replays/demo", {});
      const list = await getJson<ReplayMeta[]>("/replays");
      setIds(list);
      selectReplay(res.replayId);
      notify("Recorded a new scripted AUTO replay.");
    } finally {
      setRecording(false);
    }
  }

  return (
    <main className="page layout-replay">
      <Panel
        className="grow"
        bodyClass="viewport"
        title={meta ? replayLabel(meta) : "Replay"}
        sub={
          selectedId ? (
            <span className="mono">
              {meta?.runId && meta.runId !== replayLabel(meta) ? `${meta.runId} · ` : ""}
              {selectedId}
            </span>
          ) : undefined
        }
        actions={
          <>
            <Segmented
              label="Camera"
              value={view}
              onChange={setView}
              options={[
                ["threeQuarter", "Perspective"],
                ["top", "Top"],
              ]}
            />
            <Switch checked={showFov} onChange={setShowFov}>
              Camera FOV
            </Switch>
            <button type="button" className="btn sm" disabled={!selectedId} onClick={() => setExportId(selectedId)}>
              <Icon name="code" size={14} /> Export
            </button>
          </>
        }
        footer={
          <div className="transport" style={{ border: 0, padding: 0, width: "100%" }}>
            <button
              type="button"
              className="btn primary icon"
              onClick={togglePlay}
              disabled={!frames.length}
              aria-label={playing ? "Pause" : "Play"}
              title="Space"
            >
              <Icon name={playing ? "pause" : "play"} />
            </button>
            <span className="time">
              <b>{(frame?.t ?? 0).toFixed(2)}</b> / {duration.toFixed(2)} s
            </span>
            <div className="scrub">
              {markers.length > 0 && (
                <div className="scrub-marks" aria-hidden="true">
                  {markers.map((m) => (
                    <i key={`${m.kind}-${m.i}`} className={m.kind} style={{ left: `${frames.length > 1 ? (m.i / (frames.length - 1)) * 100 : 0}%` }} />
                  ))}
                </div>
              )}
              <input
                type="range"
                aria-label="Timeline"
                min={0}
                max={Math.max(0, frames.length - 1)}
                value={i}
                disabled={!frames.length}
                onChange={(e) => {
                  setPlaying(false);
                  setI(Number(e.target.value));
                }}
              />
            </div>
            <Segmented
              label="Playback speed"
              value={speed}
              onChange={setSpeed}
              options={[
                [0.25, "¼×"],
                [0.5, "½×"],
                [1, "1×"],
                [2, "2×"],
              ]}
            />
          </div>
        }
      >
        {frame ? (
          <>
            <FieldScene frame={frame} showFov={showFov} path={path} view={view} />
            <div className="viewport-badges">
              {frame.phase && <span className="viewport-badge">{frame.phase}</span>}
              {foulNow && <span className="viewport-badge bad">Foul</span>}
              {contact.length > 0 && !foulNow && <span className="viewport-badge warn">Contact · {contact.join(", ")}</span>}
            </div>
            <span className="viewport-hint">Drag to orbit · scroll to zoom · Space play · ←/→ step</span>
          </>
        ) : (
          <Empty dark title={loading ? "Loading replay…" : "No replay selected"}>
            {!loading && <span>Record a scripted demo or train a policy to create one.</span>}
          </Empty>
        )}
      </Panel>

      <div className="col">
        <Panel className="fixed" title="Score" sub={frame ? `at ${frame.t.toFixed(2)} s` : undefined}>
          <div className="score-big">
            <span className="value">{frame?.trueScore ?? 0}</span>
            <span className="label">points (net of fouls)</span>
          </div>
          <div className="flags">
            {penalties.length > 0 && <span className="pill bad">{penalties.length} foul{penalties.length === 1 ? "" : "s"}</span>}
            {contact.length > 0 && <span className="pill warn">Contact: {contact.join(", ")}</span>}
            {robot && <span className="pill">Holding {robot.held.length}</span>}
            {robot && (
              <span className="pill mono" style={{ textTransform: "none" }}>
                x {Math.round(robot.x) || 0} · y {Math.round(robot.y) || 0} · {Math.round(robot.headingDeg) || 0}°
              </span>
            )}
          </div>
          {(penalties.length > 0 || scoreExplains.length > 0) && (
            <ul className="events" style={{ marginTop: "0.8rem" }}>
              {penalties.map((e, n) => (
                <li key={`p-${e.id}-${n}`} className="foul">
                  <span className="pts">{signedPoints(e.points)}</span>
                  <span>{e.explain}</span>
                </li>
              ))}
              {scoreExplains.map((e, n) => (
                <li key={`s-${e.id}-${n}`}>
                  <span className="pts">{signedPoints(e.points)}</span>
                  <span>{e.explain}</span>
                </li>
              ))}
            </ul>
          )}
          {markers.length > 0 && (
            <div className="legend" style={{ marginTop: "0.8rem" }}>
              <span>
                <i style={{ background: "var(--bad)" }} /> Foul on timeline
              </span>
              <span>
                <i style={{ background: "#d49a00" }} /> Contact on timeline
              </span>
            </div>
          )}
        </Panel>

        <Panel className="replay-field" title="Field state" sub={frame ? `t ${frame.t.toFixed(2)} s` : undefined} bodyClass="panel-body scroll">
          <p className="note">Queues {namedQueues(frame?.queues)}</p>
          <p className="label">Gates</p>
          <KeyValues data={frame?.gate as Record<string, unknown>} />
          <p className="label">Match vars (true)</p>
          <KeyValues data={frame?.matchVarsPrivileged} />
          <p className="label">Match vars (observed)</p>
          <KeyValues data={frame?.observedMatchVars as Record<string, unknown>} />
        </Panel>

        <Panel
          className="grow"
          title="Replays"
          sub={ids ? `${ids.length}` : undefined}
          bodyClass="panel-body flush scroll"
          actions={
            <button type="button" className="btn sm" onClick={recordDemo} disabled={recording}>
              <Icon name="record" size={12} /> {recording ? "Recording…" : "Record demo"}
            </button>
          }
        >
          {ids && ids.length === 0 && <Empty title="No replays yet">Training runs and evaluations save their best episode here.</Empty>}
          <ul className="list">
            {(ids || []).map((r) => (
              <li key={r.id}>
                <button type="button" className={`list-item ${r.id === selectedId ? "on" : ""}`} onClick={() => selectReplay(r.id)}>
                  <span className="grow">
                    <span className="title">{replayLabel(r)}</span>
                    <span className="meta mono" style={{ display: "block" }}>
                      {replayMetaLine(r)}
                    </span>
                  </span>
                  <span className="end">
                    <span className="num" style={{ fontWeight: 600 }}>
                      {r.trueScore ?? "—"}
                    </span>
                    <span className="meta" style={{ display: "block" }}>
                      pts
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
      <RoadRunnerDialog replayId={exportId} onClose={() => setExportId(null)} />
    </main>
  );
}
