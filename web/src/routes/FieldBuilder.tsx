import { useEffect, useMemo, useState } from "react";
import { getJson, postJson, putJson, type DefaultsBundle, type Frame, type GamePieceType, type PresetMeta } from "../api";
import { parseJsonDocument } from "../labHonesty";
import { FieldScene } from "../scene/FieldScene";
import { theme } from "../theme";

const SEASON_LABEL: Record<string, string> = {
  biobuzz: "BIOBUZZ",
};

type FieldDoc = {
  id: string;
  displayName?: string;
  season?: { slug?: string };
  provenance?: { verifyAgainstManual?: boolean; manualRevision?: string; contentSha256?: string };
  fieldSizeIn?: { width: number; depth: number };
  backgroundAsset?: string | null;
  cadManifest?: string | null;
  gamePieces?: GamePieceType[];
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
  const [id, setId] = useState("biobuzz_2026_field_v1");
  const [selected, setSelected] = useState<string>("");
  const [msg, setMsg] = useState("");
  const [tab, setTab] = useState<"field" | "scoring">("field");
  const [jsonOpen, setJsonOpen] = useState(false);
  const [jsonText, setJsonText] = useState("");
  const [scoringList, setScoringList] = useState<PresetMeta[]>([]);
  const [scoringId, setScoringId] = useState("");
  const [scoringDoc, setScoringDoc] = useState("");
  const [lint, setLint] = useState<string[]>([]);
  const [jsonError, setJsonError] = useState("");

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
  const elements = useMemo(() => {
    const seen = new Set<string>();
    const out: NonNullable<FieldDoc["elements"]> = [];
    for (const el of doc?.elements || []) {
      const k = `${el.id}|${el.pose?.x ?? 0}|${el.pose?.y ?? 0}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(el);
    }
    return out;
  }, [doc]);
  const current = elements.find((e) => e.id === selected);
  const fieldArea = size.width * size.depth;
  const cadPreview = useMemo<Frame | null>(() => {
    if (!doc) return null;
    return {
      t: 0,
      trueScore: 0,
      robots: [],
      pieces: [],
      elements: elements.map((el) => ({
        id: el.id,
        type: el.type,
        alliance: el.alliance,
        pose: { x: el.pose?.x ?? 0, y: el.pose?.y ?? 0, headingDeg: el.pose?.headingDeg },
        shape: { kind: el.shape?.kind || "aabb", width: el.shape?.width, depth: el.shape?.depth, radius: el.shape?.radius },
        tags: el.tags,
      })),
      matchVarsPrivileged: {},
      observedMatchVars: {},
      fieldSizeIn: { width: size.width, depth: size.depth },
      backgroundAsset: typeof doc.backgroundAsset === "string" ? doc.backgroundAsset : null,
      cadManifest: typeof doc.cadManifest === "string" ? doc.cadManifest : null,
      cadSourceSha256: typeof doc.provenance?.contentSha256 === "string" ? doc.provenance.contentSha256 : null,
      gamePieces: Array.isArray(doc.gamePieces) ? doc.gamePieces : undefined,
      explains: [],
      queues: {},
      gate: {},
      vision: [],
    };
  }, [doc, elements, size.depth, size.width]);

  function seasonName(row: PresetMeta) {
    return SEASON_LABEL[row.season || ""] || row.displayName || row.id;
  }

  function elSize(el: NonNullable<FieldDoc["elements"]>[number]) {
    return {
      w: el.shape?.width || el.shape?.radius || 8,
      d: el.shape?.depth || el.shape?.radius || 8,
    };
  }

  function isLarge(el: NonNullable<FieldDoc["elements"]>[number]) {
    const { w, d } = elSize(el);
    return w * d > 0.15 * fieldArea;
  }

  function elColor(el: NonNullable<FieldDoc["elements"]>[number]) {
    const tags = el.tags || [];
    if (tags.some((t) => t.includes("restricted"))) return theme.restricted;
    if (el.type === "tape" || tags.some((t) => t.includes("launch"))) return theme.gold;
    if (el.alliance === "blue") return theme.allianceBlue;
    return theme.maroon;
  }

  function updatePose(partial: { x?: number; y?: number; headingDeg?: number }) {
    if (!doc || !selected) return;
    const next: FieldDoc = {
      ...doc,
      elements: (doc.elements || []).map((el) =>
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
    let bestD = Infinity;
    for (const el of elements) {
      if (el.type === "wall") continue;
      const px = el.pose?.x ?? 0;
      const py = el.pose?.y ?? 0;
      const { w, d: dep } = elSize(el);
      const hit = Math.max(8, Math.min(w, dep) / 2 + 4);
      const dist = Math.hypot(px - x, py - y);
      if (dist < hit && dist < bestD) {
        bestD = dist;
        best = el.id;
      }
    }
    if (best) setSelected(best);
  }

  async function saveField() {
    setJsonError("");
    try {
      let parsed: FieldDoc | null = doc;
      if (jsonOpen) {
        const result = parseJsonDocument<FieldDoc>(jsonText);
        if (!result.ok) {
          setJsonError(`Field JSON is invalid: ${result.error}`);
          return;
        }
        parsed = result.value;
      }
      if (!parsed) return;
      await putJson(`/presets/field/${parsed.id}`, parsed);
      setMsg("Saved field via API (not browser storage).");
      setDoc(parsed);
    } catch (e) {
      setJsonError(e instanceof Error ? e.message : String(e));
    }
  }

  async function saveScoring() {
    setJsonError("");
    const result = parseJsonDocument<{ id: string }>(scoringDoc);
    if (!result.ok) {
      setJsonError(`Scoring JSON is invalid: ${result.error}`);
      setLint([]);
      return;
    }
    try {
      await putJson(`/presets/scoring/${result.value.id}`, result.value);
      setMsg("Saved scoring preset via API.");
    } catch (e) {
      setJsonError(e instanceof Error ? e.message : String(e));
    }
  }

  async function validateScoring() {
    setJsonError("");
    const result = parseJsonDocument<Record<string, unknown>>(scoringDoc);
    if (!result.ok) {
      setJsonError(`Scoring JSON is invalid: ${result.error}`);
      setLint([]);
      return;
    }
    try {
      const res = await postJson<{ ok: boolean; errors: string[] }>("/presets/validate", {
        kind: "scoring",
        document: result.value,
      });
      setLint(res.ok ? ["Valid."] : res.errors || ["Invalid."]);
    } catch (e) {
      setJsonError(e instanceof Error ? e.message : String(e));
    }
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
        <div className="page-head">
          <h2>Field / scoring builder</h2>
          <p className="note">Click an element on the inch grid, then edit pose. Origin is field center.</p>
        </div>
        <div className="seg" role="tablist" aria-label="Builder">
          <button type="button" className={tab === "field" ? "on" : ""} onClick={() => setTab("field")}>
            Field
          </button>
          <button type="button" className={tab === "scoring" ? "on" : ""} onClick={() => setTab("scoring")}>
            Scoring JSON
          </button>
        </div>
        {tab === "field" && (
          <>
            {meta && (meta.stale || meta.verifyAgainstManual) && (
              <div className="banner">
                {seasonName(meta)} · {meta.manualRevision || "rev?"}
                {meta.stale ? " · stale vs latest known manual" : ""}
                {meta.verifyAgainstManual ? " · verifyAgainstManual — confirm CAD poses before a tournament" : ""}
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
            <div className="field-work">
              <div>
                {cadPreview && (
                  <div className="field-cad-preview" aria-label="Field 3D mesh">
                    <FieldScene frame={cadPreview} showFov={false} />
                  </div>
                )}
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
                    fill={theme.field}
                  />
                  {gridTicks.map((v) => (
                    <g key={v}>
                      <line x1={v} y1={-size.depth / 2} x2={v} y2={size.depth / 2} stroke={theme.grid} strokeWidth="0.6" />
                      <line x1={-size.width / 2} y1={v} x2={size.width / 2} y2={v} stroke={theme.grid} strokeWidth="0.6" />
                    </g>
                  ))}
                  {[-size.width / 2, 0, size.width / 2].map((v) => (
                    <text key={`x-${v}`} x={v + 1} y={size.depth / 2 - 2} fill={theme.muted} fontSize="4">
                      {v}
                    </text>
                  ))}
                  {[-size.depth / 2, 0, size.depth / 2].map((v) => (
                    <text key={`y-${v}`} x={-size.width / 2 + 1} y={-v + 4} fill={theme.muted} fontSize="4">
                      {v}
                    </text>
                  ))}
                  {elements
                    .filter((el) => el.type !== "wall" && isLarge(el))
                    .map((el, idx) => {
                      const x = el.pose?.x ?? 0;
                      const y = -(el.pose?.y ?? 0);
                      const { w, d } = elSize(el);
                      const on = el.id === selected;
                      return (
                        <rect
                          key={`lg-${el.id}-${idx}`}
                          x={x - w / 2}
                          y={y - d / 2}
                          width={w}
                          height={d}
                          fill={elColor(el)}
                          fillOpacity={0.06}
                          stroke={on ? theme.cream : elColor(el)}
                          strokeWidth={on ? 1.6 : 0.9}
                        />
                      );
                    })}
                  {elements
                    .filter((el) => el.type === "tape")
                    .map((el, idx) => {
                      const x = el.pose?.x ?? 0;
                      const y = -(el.pose?.y ?? 0);
                      const { w, d } = elSize(el);
                      return (
                        <rect
                          key={`tape-${el.id}-${idx}`}
                          x={x - w / 2}
                          y={y - d / 2}
                          width={w}
                          height={d}
                          fill={theme.gold}
                          opacity={0.9}
                        />
                      );
                    })}
                  {elements
                    .filter((el) => el.type !== "wall" && el.type !== "tape" && !isLarge(el))
                    .map((el, idx) => {
                      const x = el.pose?.x ?? 0;
                      const y = -(el.pose?.y ?? 0);
                      const { w, d } = elSize(el);
                      const on = el.id === selected;
                      return (
                        <rect
                          key={`sm-${el.id}-${idx}`}
                          x={x - w / 2}
                          y={y - d / 2}
                          width={w}
                          height={d}
                          fill={on ? theme.gold : elColor(el)}
                          opacity={0.9}
                          stroke={on ? theme.cream : "transparent"}
                          strokeWidth={on ? 1.6 : 0}
                        />
                      );
                    })}
                </svg>
                <div className="legend">
                  <span>
                    <i style={{ background: theme.restricted }} /> restricted
                  </span>
                  <span>
                    <i style={{ background: theme.gold }} /> tape
                  </span>
                  <span>
                    <i style={{ background: theme.maroon }} /> element
                  </span>
                  <span>
                    <i style={{ background: theme.gold }} /> selected
                  </span>
                </div>
              </div>
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
                onChange={(e) => {
                  setJsonText(e.target.value);
                  setJsonError("");
                }}
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
              onChange={(e) => {
                setScoringDoc(e.target.value);
                setJsonError("");
              }}
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
        {jsonError && <div className="banner">{jsonError}</div>}
        <p className="note">{msg}</p>
      </div>
    </div>
  );
}
