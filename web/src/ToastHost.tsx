import { useEffect, useState } from "react";
import type { ToastKind } from "./api";

type Toast = { id: number; kind: ToastKind; title?: string; message: string };

export function ToastHost() {
  const [items, setItems] = useState<Toast[]>([]);

  useEffect(() => {
    let n = 0;
    const push = (t: Omit<Toast, "id">) => {
      const id = ++n;
      setItems((prev) => [...prev.slice(-3), { id, ...t }]);
      window.setTimeout(() => setItems((prev) => prev.filter((x) => x.id !== id)), t.kind === "error" ? 9000 : 4000);
    };
    const onErr = (ev: Event) => {
      const d = (ev as CustomEvent<{ code: string; message: string }>).detail;
      push({ kind: "error", title: `Error ${d.code}`, message: d.message });
    };
    const onToast = (ev: Event) => {
      const d = (ev as CustomEvent<{ kind: ToastKind; title?: string; message: string }>).detail;
      push(d);
    };
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setItems([]);
    };
    window.addEventListener("talongym-error", onErr);
    window.addEventListener("talongym-toast", onToast);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("talongym-error", onErr);
      window.removeEventListener("talongym-toast", onToast);
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  if (!items.length) return null;
  return (
    <div className="toasts" role="status" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>
          <div className="grow">
            {t.title && <b>{t.title}</b>}
            {t.message}
          </div>
          <button type="button" aria-label="Dismiss" onClick={() => setItems((prev) => prev.filter((x) => x.id !== t.id))}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
