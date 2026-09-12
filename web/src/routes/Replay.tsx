import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { FieldScene } from "../scene/FieldScene";
import { getJson, loadReplayFrames, postJson, postText, type Frame } from "../api";

type ReplayMeta = { id: string; trueScore?: number; source?: string };

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
  const [java, setJava] = useState("");

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
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const frame = frames[i] || null;
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
        <FieldScene frame={frame} showFov={showFov} path={path} />
      </div>
      <aside className="side">
        <h2>Replay</h2>
        <p className="note">
          Visualizer only — physics stays on the server. Privileged match variables are inspection-only, not what the
          policy saw. Space plays or pauses.
        </p>
        <div className="row">
          <button className="primary" type="button" onClick={demo}>
            Record scripted AUTO
          </button>
          <button type="button" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
            {playing ? "Pause" : "Play"}
          </button>
        </div>
        <label htmlFor="replay-select">Replay</label>
        <select id="replay-select" value={active} onChange={(e) => selectReplay(e.target.value)}>
          {ids.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} · {r.source || "run"} · {r.trueScore ?? "?"}
            </option>
          ))}
        </select>
        <label htmlFor="replay-scrub">Scrub {frame ? `t=${frame.t.toFixed(2)}s` : ""}</label>
        <input
          id="replay-scrub"
          className="scrub"
          type="range"
          min={0}
          max={Math.max(0, frames.length - 1)}
          value={i}
          onChange={(e) => setI(Number(e.target.value))}
        />
        <label htmlFor="replay-speed">Playback speed</label>
        <select id="replay-speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          <option value={0.25}>0.25×</option>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
        </select>
        <div className="stat">
          true score <b>{frame?.trueScore ?? 0}</b>
        </div>
        <label className="stat">
          <input type="checkbox" checked={showFov} onChange={(e) => setShowFov(e.target.checked)} /> camera FOV
        </label>
        <div className="banner">
          Privileged match vars (inspection-only): {frame?.matchVarsPrivileged ? JSON.stringify(frame.matchVarsPrivileged) : "—"}
          <br />
          Observed match vars: {frame?.observedMatchVars ? JSON.stringify(frame.observedMatchVars) : "—"}
        </div>
        <div className="stat">Ramp queue: {(frame?.queues && Object.values(frame.queues)[0]?.join(" ")) || "empty"}</div>
        <div className="stat">Gate: {frame?.gate ? JSON.stringify(frame.gate) : "—"}</div>
        <ul className="list">
          {(frame?.explains || []).map((e, n) => (
            <li key={n}>
              +{e.points} {e.explain}
            </li>
          ))}
        </ul>
        <div className="row">
          <button type="button" onClick={exportRr}>
            Export Road Runner 1.0
          </button>
        </div>
        {java && (
          <pre className="note mono rr-export" tabIndex={0}>
            {java}
          </pre>
        )}
        {path.length > 0 && <p className="note">{path.length} planned-path samples (decimated poses).</p>}
      </aside>
    </div>
  );
}
