import { useEffect, useState } from "react";

type Toast = { id: number; code: string; message: string };

export function ToastHost() {
  const [items, setItems] = useState<Toast[]>([]);

  useEffect(() => {
    let n = 0;
    const onErr = (ev: Event) => {
      const detail = (ev as CustomEvent<{ code: string; message: string }>).detail;
      const id = ++n;
      setItems((prev) => [...prev, { id, code: detail.code, message: detail.message }]);
      window.setTimeout(() => setItems((prev) => prev.filter((t) => t.id !== id)), 8000);
    };
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setItems([]);
    };
    window.addEventListener("talongym-error", onErr);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("talongym-error", onErr);
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  if (!items.length) return null;
  return (
    <div className="toasts" role="status" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className="toast">
          <b>{t.code}</b> {t.message}
          <span className="note"> Esc to dismiss</span>
        </div>
      ))}
    </div>
  );
}
