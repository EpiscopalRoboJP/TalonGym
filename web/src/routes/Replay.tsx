import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { FieldScene, type SceneView } from "../scene/FieldScene";
import { KeyValues } from "../KeyValues";
import { getJson, loadReplayFrames, postJson, postText, type Frame } from "../api";

type ReplayMeta = { id: string; trueScore?: number; source?: string };
type Tab = "score" | "field" | "export";

export function ReplayPage() {
  const { replayId } = useParams();
  const navigate = useNavigate();
  const [ids, setIds] = useState<ReplayMeta[]>([]);
  const [active, setActive] = useState(replayId || "");
  const [frames, setFrames] = useState<Frame[]>([]);
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showFov, setShowFov] = useState(true);
  const [view, setView] = useState<SceneView>("threeQuarter");
  const [java, setJava] = useState("");
  const [tab, setTab] = useState<Tab>("score");

  useEffect(() => {
    if (replayId && replayId !== active) setActive(replayId);
  }, [replayId]);

  useEffect(() => {
    getJson<ReplayMeta[]>("/replays").then((list) => {
      setIds(list);
      if (!active && list[0]) selectReplay(list[0].id);
    });
  }, []);

  useEffect(() => {
    if (!active) return;
    loadReplayFrames(active).then((f) => {
      setFrames(f);
      setI(0);
    });
  }, [active]);

  useEffect(() => {
    if (!playing || frames.length === 0) return;
    const id = window.setInterval(() => {
      setI((n) => (n + 1) % frames.length);
    }, 40 / speed);
    return () => window.clearInterval(id);
  }, [playing, frames, speed]);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const tag = (ev.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (ev.key === " " || ev.code === "Space") {
        ev.preventDefault();
        setPlaying((p) => !p);
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
  }, [frames.length]);

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

  function selectReplay(id: string) {
    setActive(id);
    setJava("");
    navigate(`/replay/${id}`, { replace: true });
  }

  async function demo() {
    const res = await postJson<{ replayId: string }>("/replays/demo", {});
    const list = await getJson<ReplayMeta[]>("/replays");
    setIds(list);
    selectReplay(res.replayId);
  }

  async function exportRr() {
    if (!active) return;
    const text = await postText(`/replays/${active}/export/roadrunner`, { dialect: "rr1_actions" });
    setJava(text);
  }

  return (
    <div className="page">
      <div className="scene">
        <FieldScene frame={frame} showFov={showFov} path={path} view={view} />
        <div className="dock">
          <div className="dock-row">
            <button type="button" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
              {playing ? "Pause" : "Play"}
            </button>
            <span className="dock-time">
              t={frame?.t.toFixed(2) ?? "0.00"} / {duration.toFixed(2)}s
            </span>
            <div className="seg" role="group" aria-label="Playback speed">
              {[0.25, 1, 2].map((s) => (
                <button key={s} type="button" className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>
                  {s}×
                </button>
              ))}
            </div>
            <div className="seg" role="group" aria-label="Camera view">
              <button type="button" className={view === "threeQuarter" ? "on" : ""} onClick={() => setView("threeQuarter")}>
                3/4
              </button>
              <button type="button" className={view === "top" ? "on" : ""} onClick={() => setView("top")}>
                top
              </button>
            </div>
            <label className="check" style={{ margin: 0 }}>
              <input type="checkbox" checked={showFov} onChange={(e) => setShowFov(e.target.checked)} /> FOV
            </label>
          </div>
          <input
            id="replay-scrub"
            className="scrub"
            type="range"
            aria-label="Scrub"
            min={0}
            max={Math.max(0, frames.length - 1)}
            value={i}
            onChange={(e) => setI(Number(e.target.value))}
          />
        </div>
      </div>
      <aside className="side">
        <div className="page-head">
          <h2>Replay</h2>
          <p className="note">Visualizer only. Space play/pause · arrows step · Home/End jump.</p>
        </div>
        <label htmlFor="replay-select">Replay</label>
        <select id="replay-select" value={active} onChange={(e) => selectReplay(e.target.value)}>
          {ids.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} · {r.source || "run"} · {r.trueScore ?? "?"}
            </option>
          ))}
        </select>
        <div className="row">
          <button type="button" onClick={demo}>
            Record scripted AUTO
          </button>
        </div>
        <div className="seg" role="tablist" aria-label="Inspector">
          {(
            [
              ["score", "Score"],
              ["field", "Field state"],
              ["export", "Export"],
            ] as [Tab, string][]
          ).map(([id, label]) => (
            <button key={id} type="button" role="tab" aria-selected={tab === id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </div>
        {tab === "score" && (
          <div className="card">
            <h3>Score</h3>
            <div className="stat">
              true score <b>{frame?.trueScore ?? 0}</b>
            </div>
            <ul className="list">
              {(frame?.explains || []).map((e, n) => (
                <li key={n}>
                  +{e.points} {e.explain}
                </li>
              ))}
            </ul>
          </div>
        )}
        {tab === "field" && (
          <div className="card">
            <h3>Field state</h3>
            <div className="stat">Ramp queue: {(frame?.queues && Object.values(frame.queues)[0]?.join(" ")) || "empty"}</div>
            <p className="stat">Gate</p>
            <KeyValues data={frame?.gate as Record<string, unknown>} />
            <p className="stat">Privileged match vars (inspection-only)</p>
            <KeyValues data={frame?.matchVarsPrivileged} />
            <p className="stat">Observed match vars</p>
            <KeyValues data={frame?.observedMatchVars as Record<string, unknown>} />
          </div>
        )}
        {tab === "export" && (
          <div className="card">
            <h3>Export</h3>
            <div className="row">
              <button type="button" onClick={exportRr}>
                Road Runner 1.0
              </button>
            </div>
            {java && (
              <pre className="note mono rr-export" tabIndex={0}>
                {java}
              </pre>
            )}
            {path.length > 0 && <p className="note">{path.length} planned-path samples (decimated poses).</p>}
          </div>
        )}
      </aside>
    </div>
  );
}
