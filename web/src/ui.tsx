import { useEffect, useRef, useState, type ReactNode } from "react";
import { notify, postText } from "./api";
import { DRIVE_EXPORT_NOTE } from "./labHonesty";

/* ---------- Icons (inline, stroke = currentColor) ---------- */

const ICONS = {
  play: "M7 5v14l11-7z",
  pause: "M7 5h4v14H7zM13 5h4v14h-4z",
  plus: "M12 5v14M5 12h14",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  code: "M8 7l-5 5 5 5M16 7l5 5-5 5",
  upload: "M12 16V4M7 9l5-5 5 5M4 20h16",
  check: "M5 12l5 5 9-10",
  alert: "M12 3l10 18H2zM12 10v4M12 17.5v.5",
  info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 11v6M12 7.5v.5",
  close: "M6 6l12 12M18 6L6 18",
  record: "M12 19a7 7 0 1 0 0-14 7 7 0 0 0 0 14z",
  stop: "M6 6h12v12H6z",
  external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
  chart: "M4 20V10M10 20V4M16 20v-7M22 20H2",
} as const;

export type IconName = keyof typeof ICONS;

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  const filled = name === "play" || name === "pause" || name === "record" || name === "stop";
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke={filled ? "none" : "currentColor"}
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={ICONS[name]} />
    </svg>
  );
}

export function BrandMark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5h16v3.2h-6.3V20h-3.4V8.2H4z" fill="currentColor" />
      <path d="M13.7 12.6c2.9-.2 5.1.9 6.3 3.4-1.7-.9-3.8-1.2-6.3-.8z" fill="#b4975a" />
    </svg>
  );
}

/* ---------- Layout ---------- */

export function Panel({
  title,
  sub,
  actions,
  children,
  className = "",
  bodyClass = "panel-body",
  footer,
}: {
  title?: ReactNode;
  sub?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
  footer?: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="panel-head">
          {title && <h2>{title}</h2>}
          {sub && <span className="sub">{sub}</span>}
          <span className="spacer" />
          {actions}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
      {footer && <footer className="panel-foot">{footer}</footer>}
    </section>
  );
}

export function Alert({ kind = "info", children }: { kind?: "info" | "warn" | "bad" | "ok"; children: ReactNode }) {
  const icon: IconName = kind === "ok" ? "check" : kind === "info" ? "info" : "alert";
  return (
    <div className={`alert ${kind === "info" ? "" : kind}`} role={kind === "bad" ? "alert" : undefined}>
      <Icon name={icon} />
      <div>{children}</div>
    </div>
  );
}

export function Empty({ title, children, dark = false }: { title: string; children?: ReactNode; dark?: boolean }) {
  return (
    <div className={`empty ${dark ? "dark" : ""}`}>
      <b>{title}</b>
      {children}
    </div>
  );
}

/* ---------- Controls ---------- */

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
  full = false,
}: {
  value: T;
  options: [T, ReactNode][];
  onChange: (v: T) => void;
  label: string;
  full?: boolean;
}) {
  return (
    <div className={`seg ${full ? "full" : ""}`} role="radiogroup" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={String(v)} type="button" role="radio" aria-checked={value === v} className={value === v ? "on" : ""} onClick={() => onChange(v)}>
          {text}
        </button>
      ))}
    </div>
  );
}

export function Field({ id, label, hint, children }: { id?: string; label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

export function NumberField({
  id,
  label,
  value,
  unit,
  step = "any",
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  value: number;
  unit?: string;
  step?: number | "any";
  onChange: (n: number) => void;
  disabled?: boolean;
}) {
  return (
    <Field id={id} label={label}>
      <div className="input-unit">
        <input
          id={id}
          type="number"
          step={step}
          value={Number.isFinite(value) ? value : 0}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))}
        />
        {unit && <span>{unit}</span>}
      </div>
    </Field>
  );
}

export function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (n: number) => void;
}) {
  return (
    <div className="slider">
      <label htmlFor={id}>{label}</label>
      <output htmlFor={id}>
        {Number.isInteger(step) ? value : value.toFixed(2)}
        {unit ? ` ${unit}` : ""}
      </output>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function Switch({
  checked,
  onChange,
  children,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className="switch">
      <input type="checkbox" role="switch" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      {children}
    </label>
  );
}

/* ---------- Road Runner export dialog ---------- */

export function RoadRunnerDialog({ replayId, onClose }: { replayId: string | null; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const dlg = ref.current;
    if (!dlg) return;
    if (!replayId) {
      if (dlg.open) dlg.close();
      return;
    }
    if (!dlg.open) dlg.showModal();
    let cancelled = false;
    setCode("");
    setLoading(true);
    postText(`/replays/${replayId}/export/roadrunner`, { dialect: "rr1_actions" })
      .then((text) => {
        if (!cancelled) setCode(text);
      })
      .catch(() => onClose())
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // onClose identity is not part of the load trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayId]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      notify("Copied Road Runner code to the clipboard.");
    } catch {
      notify("Clipboard is unavailable. Select the code and copy it manually.", "error");
    }
  }

  return (
    <dialog ref={ref} className="modal" onClose={onClose} aria-labelledby="rr-title">
      <header className="panel-head">
        <h2 id="rr-title">Road Runner 1.0 export</h2>
        <span className="sub mono">{replayId}</span>
        <span className="spacer" />
        <button type="button" className="btn sm" onClick={copy} disabled={!code}>
          <Icon name="copy" size={14} /> Copy
        </button>
        <button type="button" className="btn ghost icon" aria-label="Close" onClick={() => ref.current?.close()}>
          <Icon name="close" />
        </button>
      </header>
      <div className="panel-body stack">
        <p className="note">{DRIVE_EXPORT_NOTE}</p>
        <p className="note">Paste these Actions into your AUTO OpMode. They are generated from the replay's decimated robot path.</p>
        {loading ? <p className="note">Generating…</p> : <pre className="code" tabIndex={0}>{code}</pre>}
      </div>
    </dialog>
  );
}
