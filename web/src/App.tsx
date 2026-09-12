import { NavLink, Route, Routes } from "react-router-dom";
import { useEffect, useState } from "react";
import { getJson, type Health } from "./api";
import { ToastHost } from "./ToastHost";
import { ReplayPage } from "./routes/Replay";
import { TrainPage } from "./routes/Train";
import { FieldBuilderPage } from "./routes/FieldBuilder";
import { RobotBuilderPage } from "./routes/RobotBuilder";
import { ComparePage } from "./routes/Compare";

export function App() {
  const [health, setHealth] = useState("checking…");
  const [ok, setOk] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function ping() {
      try {
        const h = await getJson<Health>("/health");
        if (cancelled) return;
        setOk(Boolean(h.ok));
        const profile = h.computeProfile ? ` · ${h.computeProfile}` : "";
        setHealth(`${h.engine}${h.db ? " · " + h.db : ""}${profile} · ok`);
      } catch {
        if (cancelled) return;
        setOk(false);
        setHealth("API offline");
      }
    }
    ping();
    const id = window.setInterval(ping, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          TALON<span>GYM</span>
        </div>
        <nav aria-label="Lab">
          <NavLink to="/replay" className={({ isActive }) => (isActive ? "active" : "")}>
            Replay
          </NavLink>
          <NavLink to="/train" className={({ isActive }) => (isActive ? "active" : "")}>
            Train
          </NavLink>
          <NavLink to="/build/field" className={({ isActive }) => (isActive ? "active" : "")}>
            Field
          </NavLink>
          <NavLink to="/build/robot" className={({ isActive }) => (isActive ? "active" : "")}>
            Robot
          </NavLink>
          <NavLink to="/compare" className={({ isActive }) => (isActive ? "active" : "")}>
            Compare
          </NavLink>
        </nav>
        <div className={`health ${ok ? "ok" : ""}`}>{health}</div>
      </header>
      <ToastHost />
      <Routes>
        <Route path="/" element={<ReplayPage />} />
        <Route path="/replay" element={<ReplayPage />} />
        <Route path="/replay/:replayId?" element={<ReplayPage />} />
        <Route path="/train/:runId?" element={<TrainPage />} />
        <Route path="/build/field" element={<FieldBuilderPage />} />
        <Route path="/build/robot" element={<RobotBuilderPage />} />
        <Route path="/compare" element={<ComparePage />} />
      </Routes>
    </div>
  );
}
