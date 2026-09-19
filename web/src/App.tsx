import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { useEffect, useState } from "react";
import { getJson, type Health } from "./api";
import { ToastHost } from "./ToastHost";
import { CREDIT_LINE } from "./licenseNotice";
import { BrandMark } from "./ui";
import { ReplayPage } from "./routes/Replay";
import { TrainPage } from "./routes/Train";
import { FieldBuilderPage } from "./routes/FieldBuilder";
import { RobotBuilderPage } from "./routes/RobotBuilder";
import { ComparePage } from "./routes/Compare";

const NAV_MAIN: [string, string][] = [
  ["/train", "Train"],
  ["/compare", "Evaluate"],
  ["/replay", "Replay"],
];
const NAV_BUILD: [string, string][] = [
  ["/build/robot", "Robot"],
  ["/build/field", "Field"],
];

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [state, setState] = useState<"checking" | "ok" | "down">("checking");

  useEffect(() => {
    let cancelled = false;
    async function ping() {
      try {
        const h = await getJson<Health>("/health");
        if (cancelled) return;
        setHealth(h);
        setState(h.ok ? "ok" : "down");
      } catch {
        if (cancelled) return;
        setState("down");
      }
    }
    ping();
    const id = window.setInterval(ping, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const detail = health ? [health.computeProfile, health.engine, health.db, health.version && `v${health.version}`].filter(Boolean).join(" · ") : "";
  const link = ({ isActive }: { isActive: boolean }) => (isActive ? "active" : "");

  return (
    <>
    <div className="app">
      <header className="header">
        <NavLink to="/train" className="brand" aria-label="TalonGym Lab home">
          <span className="brand-mark">
            <BrandMark />
          </span>
          <span className="brand-name">
            <b>TALONGYM</b>
            <small>Autonomous Lab</small>
          </span>
        </NavLink>
        <nav className="nav" aria-label="Lab">
          {NAV_MAIN.map(([to, label]) => (
            <NavLink key={to} to={to} className={link}>
              {label}
            </NavLink>
          ))}
          <span className="nav-sep" aria-hidden="true" />
          {NAV_BUILD.map(([to, label]) => (
            <NavLink key={to} to={to} className={link}>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className={`status ${state === "checking" ? "" : state}`} title={detail || undefined}>
          <i aria-hidden="true" />
          <span>{state === "checking" ? "Connecting…" : state === "ok" ? "API connected" : "API offline"}</span>
          {state === "ok" && health?.computeProfile && <span className="status-detail">· {health.computeProfile}</span>}
        </div>
      </header>
      <Routes>
        <Route path="/" element={<Navigate to="/train" replace />} />
        <Route path="/replay/:replayId?" element={<ReplayPage />} />
        <Route path="/train/:runId?" element={<TrainPage />} />
        <Route path="/build/field" element={<FieldBuilderPage />} />
        <Route path="/build/robot" element={<RobotBuilderPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="*" element={<Navigate to="/train" replace />} />
      </Routes>
      <footer className="app-credit">{CREDIT_LINE}</footer>
    </div>
    <ToastHost />
    </>
  );
}
