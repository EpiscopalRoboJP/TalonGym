import { useEffect, useMemo, useState } from "react";
import { getJson, postJson, putJson, type DefaultsBundle, type PresetMeta } from "../api";

const SEASON_LABEL: Record<string, string> = {
  decode: "DECODE",
  into_the_deep: "INTO THE DEEP",
  centerstage: "CENTERSTAGE",
  biobuzz: "BIOBUZZ",
};

type FieldDoc = {
  id: string;
  displayName?: string;
  season?: { slug?: string };
  provenance?: { verifyAgainstManual?: boolean; manualRevision?: string };
  fieldSizeIn?: { width: number; depth: number };
  elements?: {
    id: string;
    type?: string;
    pose?: { x: number; y: number; headingDeg?: number };
    shape?: { kind?: string; width?: number; depth?: number; radius?: number };
    tags?: string[];
    alliance?: string;
  }[];
  [k: string]: unknown;
};

export function FieldBuilderPage() {
  const [list, setList] = useState<PresetMeta[]>([]);
  const [doc, setDoc] = useState<FieldDoc | null>(null);
  const [id, setId] = useState("decode_2025_field_tu32");
  const [selected, setSelected] = useState<string>("");
  const [msg, setMsg] = useState("");
  const [tab, setTab] = useState<"field" | "scoring">("field");
  const [jsonOpen, setJsonOpen] = useState(false);
  const [jsonText, setJsonText] = useState("");
  const [scoringList, setScoringList] = useState<PresetMeta[]>([]);
  const [scoringId, setScoringId] = useState("");
  const [scoringDoc, setScoringDoc] = useState("");
  const [lint, setLint] = useState<string[]>([]);

  useEffect(() => {
    getJson<PresetMeta[]>("/presets/field").then(setList);
    getJson<PresetMeta[]>("/presets/scoring").then(setScoringList);
    getJson<DefaultsBundle>("/defaults")
      .then((d) => {
        if (d.fieldId) setId(d.fieldId);
        if (d.scoringId) setScoringId(d.scoringId);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!id) return;
    getJson<FieldDoc>(`/presets/field/${id}`).then((d) => {
      delete (d as { _kind?: string })._kind;
      setDoc(d);
      setJsonText(JSON.stringify(d, null, 2));
      setSelected(d.elements?.[0]?.id || "");
    });
  }, [id]);

  useEffect(() => {
    if (!scoringId) return;
    getJson<Record<string, unknown>>(`/presets/scoring/${scoringId}`).then((d) => {
      delete d._kind;
      setScoringDoc(JSON.stringify(d, null, 2));
    });
  }, [scoringId]);

  const meta = list.find((x) => x.id === id);
  const size = doc?.fieldSizeIn || { width: 144, depth: 144 };
  const elements = doc?.elements || [];
  const current = elements.find((e) => e.id === selected);

  function seasonName(row: PresetMeta) {
    return SEASON_LABEL[row.season || ""] || row.displayName || row.id;
  }

  function updatePose(partial: { x?: number; y?: number; headingDeg?: number }) {
    if (!doc || !selected) return;
    const next: FieldDoc = {
      ...doc,
      elements: elements.map((el) =>
        el.id === selected ? { ...el, pose: { x: 0, y: 0, headingDeg: 0, ...(el.pose || {}), ...partial } } : el,
      ),
    };
    setDoc(next);
    setJsonText(JSON.stringify(next, null, 2));
  }

  function pickAt(clientX: number, clientY: number, svg: SVGSVGElement) {
    const rect = svg.getBoundingClientRect();
    const nx = (clientX - rect.left) / rect.width;
    const ny = (clientY - rect.top) / rect.height;
    const x = nx * size.width - size.width / 2;
    const y = size.depth / 2 - ny * size.depth;
    let best = "";
    let bestD = 18;
    for (const el of elements) {
      if (el.type === "wall") continue;
      const px = el.pose?.x ?? 0;
      const py = el.pose?.y ?? 0;
      const d = Math.hypot(px - x, py - y);
      if (d < bestD) {
        bestD = d;
        best = el.id;
      }
    }
    if (best) setSelected(best);
  }

  async function saveField() {
    const parsed = jsonOpen ? (JSON.parse(jsonText) as FieldDoc) : doc;
    if (!parsed) return;
    await putJson(`/presets/field/${parsed.id}`, parsed);
    setMsg("Saved field via API (not browser storage).");
    setDoc(parsed);
  }

  async function saveScoring() {
    const parsed = JSON.parse(scoringDoc);
    await putJson(`/presets/scoring/${parsed.id}`, parsed);
    setMsg("Saved scoring preset via API.");
  }

  async function validateScoring() {
    const parsed = JSON.parse(scoringDoc);
    const res = await postJson<{ ok: boolean; errors: string[] }>("/presets/validate", {
      kind: "scoring",
      document: parsed,
    });
    setLint(res.ok ? ["Valid."] : res.errors || ["Invalid."]);
  }

  const gridTicks = useMemo(() => {
    const step = 24;
    const xs: number[] = [];
    for (let v = -size.width / 2; v <= size.width / 2; v += step) xs.push(v);
    return xs;
  }, [size.width]);

  return (
    <div className="page single">
      <div>
        <h2>Field / scoring builder</h2>
        <div className="row">
          <button type="button" className={tab === "field" ? "primary" : ""} onClick={() => setTab("field")}>
            Field
          </button>
          <button type="button" className={tab === "scoring" ? "primary" : ""} onClick={() => setTab("scoring")}>
            Scoring JSON
          </button>
        </div>
        {tab === "field" && (
          <>
            {meta?.stale && (
              <div className="banner">This season template is stale versus the latest known game manual revision.</div>
            )}
            {meta?.verifyAgainstManual && (
              <div className="banner">
                verifyAgainstManual is true — confirm poses against the official manual and CAD before a tournament.
              </div>
            )}
            <div className="form-grid">
              <label htmlFor="field-season">Season template</label>
              <select id="field-season" value={id} onChange={(e) => setId(e.target.value)}>
                {list.map((p) => (
                  <option key={p.id} value={p.id}>
                    {seasonName(p)} · {p.displayName} ({p.manualRevision || "rev?"})
                  </option>
                ))}
              </select>
            </div>
            <p className="note">Click an element on the inch grid, then edit pose. Origin is field center.</p>
            <div className="field-work">
              <svg
                className="field-grid"
                viewBox={`${-size.width / 2} ${-size.depth / 2} ${size.width} ${size.depth}`}
                role="img"
                aria-label="Field inch grid"
                onClick={(e) => pickAt(e.clientX, e.clientY, e.currentTarget)}
              >
                <rect
                  x={-size.width / 2}
                  y={-size.depth / 2}
                  width={size.width}
                  height={size.depth}
                  fill="#1c4a38"
                />
                {gridTicks.map((v) => (
                  <g key={v}>
                    <line x1={v} y1={-size.depth / 2} x2={v} y2={size.depth / 2} stroke="#2a4a3c" strokeWidth="0.6" />
                    <line x1={-size.width / 2} y1={v} x2={size.width / 2} y2={v} stroke="#2a4a3c" strokeWidth="0.6" />
                  </g>
                ))}
                {elements
                  .filter((el) => el.type !== "wall")
                  .map((el, idx) => {
                    const x = el.pose?.x ?? 0;
                    const y = -(el.pose?.y ?? 0);
                    const w = el.shape?.width || el.shape?.radius || 8;
                    const d = el.shape?.depth || el.shape?.radius || 8;
                    const on = el.id === selected;
                    return (
                      <rect
                        key={`${el.id}-${idx}`}
                        x={x - w / 2}
                        y={y - d / 2}
                        width={w}
                        height={d}
                        fill={on ? "#d4a574" : "#6fbfa3"}
                        opacity={0.85}
                        stroke={on ? "#e6eef3" : "transparent"}
                        strokeWidth={on ? 1.4 : 0}
                      />
                    );
                  })}
              </svg>
              <div>
                <label htmlFor="el-select">Selected element</label>
                <select id="el-select" value={selected} onChange={(e) => setSelected(e.target.value)}>
                  {elements.map((el, idx) => (
                    <option key={`${el.id}-${idx}`} value={el.id}>
                      {el.id} ({el.type})
                    </option>
                  ))}
                </select>
                {current && (
                  <div className="form-grid" style={{ marginTop: "0.75rem" }}>
                    <label htmlFor="el-x">x (in)</label>
                    <input
                      id="el-x"
                      type="number"
                      value={current.pose?.x ?? 0}
                      onChange={(e) => updatePose({ x: Number(e.target.value) })}
                    />
                    <label htmlFor="el-y">y (in)</label>
                    <input
                      id="el-y"
                      type="number"
                      value={current.pose?.y ?? 0}
                      onChange={(e) => updatePose({ y: Number(e.target.value) })}
                    />
                    <label htmlFor="el-h">heading (deg)</label>
                    <input
                      id="el-h"
                      type="number"
                      value={current.pose?.headingDeg ?? 0}
                      onChange={(e) => updatePose({ headingDeg: Number(e.target.value) })}
                    />
                  </div>
                )}
              </div>
            </div>
            <details open={jsonOpen} onToggle={(e) => setJsonOpen((e.target as HTMLDetailsElement).open)}>
              <summary>Advanced JSON</summary>
              <textarea
                aria-label="Field JSON"
                value={jsonText}
                onChange={(e) => setJsonText(e.target.value)}
                className="mono json-area"
              />
            </details>
            <div className="row">
              <button className="primary" type="button" onClick={saveField}>
                Save field preset
              </button>
            </div>
          </>
        )}
        {tab === "scoring" && (
          <>
            <p className="note">
              Scoring rules are data. The engine must not mention season nouns — keep names in this JSON preset.
            </p>
            <label htmlFor="scoring-select">Scoring preset</label>
            <select id="scoring-select" value={scoringId} onChange={(e) => setScoringId(e.target.value)}>
              {scoringList.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.displayName || p.id}
                </option>
              ))}
            </select>
            <textarea
              aria-label="Scoring JSON"
              value={scoringDoc}
              onChange={(e) => setScoringDoc(e.target.value)}
              className="mono json-area"
            />
            <div className="row">
              <button type="button" onClick={validateScoring}>
                Validate
              </button>
              <button className="primary" type="button" onClick={saveScoring}>
                Save scoring preset
              </button>
            </div>
            <ul className="list">
              {lint.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          </>
        )}
        <p className="note">{msg}</p>
      </div>
    </div>
  );
}
