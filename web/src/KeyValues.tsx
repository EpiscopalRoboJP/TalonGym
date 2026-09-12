export function KeyValues({ data }: { data: Record<string, unknown> | null | undefined }) {
  const entries = Object.entries(data || {}).filter(([, v]) => v !== undefined);
  if (!entries.length) return <span className="note">—</span>;
  return (
    <dl className="kv">
      {entries.map(([k, v]) => (
        <span key={k} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{v === null ? "—" : String(v)}</dd>
        </span>
      ))}
    </dl>
  );
}
