import { useEffect, useMemo, useState } from "react";
import { getJson, notify, postJson, putJson, type DefaultsBundle, type Frame, type PresetMeta } from "../api";
import { FieldScene } from "../scene/FieldScene";
import { theme } from "../theme";
import { Alert, Empty, Icon, NumberField, Panel, Segmented } from "../ui";

type FieldElement = {
  id: string;
  type?: string;
  pose?: { x: number; y: number; headingDeg?: number };
  shape?: { kind?: string; width?: number; depth?: number; radius?: number };
  tags?: string[];
  alliance?: string;
};

type FieldDoc = {
  id: string;
  displayName?: string;
  season?: { slug?: string };
  provenance?: { verifyAgainstManual?: boolean; manualRevision?: string };
  fieldSizeIn?: { width: number; depth: number };
  elements?: FieldElement[];
  [k: string]: unknown;
};

type Tab = "layout" | "json" | "scoring";
type Lint = { ok: boolean; errors: string[] } | null;

const MAP_PAD = 8;

function elSize(el: FieldElement) {
  return {
    w: el.shape?.width || el.shape?.radius || 8,
    d: el.shape?.depth || el.shape?.radius || 8,
  };
}

function elColor(el: FieldElement) {
  const tags = el.tags || [];
  if (tags.some((t) => t.includes("restricted"))) return theme.restricted;
  if (el.type === "tape" || tags.some((t) => t.includes("launch"))) return theme.gold;
  if (el.alliance === "blue") return theme.allianceBlue;
  if (el.alliance === "red") return theme.allianceRed;
  return "#9a9092";
}

function parseJson<T>(text: string): T | null {
  try {
    return JSON.parse(text) as T;
  } catch (e) {
    notify(e instanceof Error ? e.message : String(e), "error", "Invalid JSON");
    return null;
  }
}

function LintResult({ lint }: { lint: Lint }) {
  if (!lint) return <p className="note">Run validation to check this document against its JSON Schema.</p>;
  if (lint.ok) return <Alert kind="ok">Valid. The document matches its schema.</Alert>;
  return (
    <Alert kind="bad">
      <b>{lint.errors.length} problem{lint.errors.length === 1 ? "" : "s"}</b>
      <ul>
        {lint.errors.map((e) => (
          <li key={e}>{e}</li>
        ))}
      </ul>
    </Alert>
  );
}

export function FieldBuilderPage() {
  const [list, setList] = useState<PresetMeta[]>([]);
  const [doc, setDoc] = useState<FieldDoc | null>(null);
  const [id, setId] = useState("biobuzz_2026_field_v1");
  const [selected, setSelected] = useState<string>("");
  const [tab, setTab] = useState<Tab>("layout");
  const [view, setView] = useState<"map" | "3d">("map");
  const [query, setQuery] = useState("");
  const [jsonText, setJsonText] = useState("");
  const [scoringList, setScoringList] = useState<PresetMeta[]>([]);
  const [scoringId, setScoringId] = useState("");
  const [scoringDoc, setScoringDoc] = useState("");
  const [fieldLint, setFieldLint] = useState<Lint>(null);
  const [scoringLint, setScoringLint] = useState<Lint>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getJson<PresetMeta[]>("/presets/field").then(setList);
    getJson<PresetMeta[]>("/presets/scoring").then((rows) => {
      setScoringList(rows);
      setScoringId((prev) => prev || rows[0]?.id || "");
    });
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
      setSelected(d.elements?.find((el) => el.type !== "wall")?.id || "");
      setFieldLint(null);
      setDirty(false);
    });
  }, [id]);

  useEffect(() => {
    if (!scoringId) return;
    getJson<Record<string, unknown>>(`/presets/scoring/${scoringId}`).then((d) => {
      delete d._kind;
      setScoringDoc(JSON.stringify(d, null, 2));
      setScoringLint(null);
    });
  }, [scoringId]);

  const meta = list.find((x) => x.id === id);
  const size = doc?.fieldSizeIn || { width: 144, depth: 144 };
  const elements = useMemo(() => {
    const seen = new Set<string>();
    const out: FieldElement[] = [];
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
  const filtered = elements.filter((el) => {
    const q = query.trim().toLowerCase();
    return !q || el.id.toLowerCase().includes(q) || (el.type || "").toLowerCase().includes(q) || (el.tags || []).some((t) => t.includes(q));
  });
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
      explains: [],
      queues: {},
      gate: {},
      vision: [],
    };
  }, [doc, elements, size.depth, size.width]);

  function isLarge(el: FieldElement) {
    const { w, d } = elSize(el);
    return w * d > 0.15 * fieldArea;
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
    setDirty(true);
  }

  function pickAt(clientX: number, clientY: number, svg: SVGSVGElement) {
    const rect = svg.getBoundingClientRect();
    const vbW = size.width + MAP_PAD * 2;
    const vbH = size.depth + MAP_PAD * 2;
    const scale = Math.min(rect.width / vbW, rect.height / vbH);
    const ox = rect.left + (rect.width - vbW * scale) / 2;
    const oy = rect.top + (rect.height - vbH * scale) / 2;
    const x = (clientX - ox) / scale - MAP_PAD - size.width / 2;
    const y = size.depth / 2 - ((clientY - oy) / scale - MAP_PAD);
    let best = "";
    let bestD = Infinity;
    for (const el of elements) {
      if (el.type === "wall") continue;
      const px = el.pose?.x ?? 0;
      const py = el.pose?.y ?? 0;
      const { w, d: dep } = elSize(el);
      const hit = Math.max(6, Math.min(w, dep) / 2 + 3);
      const dist = Math.hypot(px - x, py - y);
      if (dist < hit && dist < bestD) {
        bestD = dist;
        best = el.id;
      }
    }
    if (best) setSelected(best);
  }

  function switchTab(next: Tab) {
    if (next === tab) return;
    if (tab === "json") {
      const parsed = parseJson<FieldDoc>(jsonText);
      if (!parsed) return;
      setDoc(parsed);
    }
    if (next === "json" && doc) setJsonText(JSON.stringify(doc, null, 2));
    setTab(next);
  }

  async function saveField() {
    const parsed = tab === "json" ? parseJson<FieldDoc>(jsonText) : doc;
    if (!parsed) return;
    setSaving(true);
    try {
      await putJson(`/presets/field/${parsed.id}`, parsed);
      setDoc(parsed);
      setDirty(false);
      notify(`Saved field preset ${parsed.id}.`);
    } finally {
      setSaving(false);
    }
  }

  async function validateField() {
    const parsed = tab === "json" ? parseJson<FieldDoc>(jsonText) : doc;
    if (!parsed) return;
    setFieldLint(await postJson<{ ok: boolean; errors: string[] }>("/presets/validate", { kind: "field", document: parsed }));
  }

  async function saveScoring() {
    const parsed = parseJson<{ id: string }>(scoringDoc);
    if (!parsed) return;
    setSaving(true);
    try {
      await putJson(`/presets/scoring/${parsed.id}`, parsed);
      notify(`Saved scoring preset ${parsed.id}.`);
    } finally {
      setSaving(false);
    }
  }

  async function validateScoring() {
    const parsed = parseJson(scoringDoc);
    if (!parsed) return;
    setScoringLint(await postJson<{ ok: boolean; errors: string[] }>("/presets/validate", { kind: "scoring", document: parsed }));
  }

  const ticks = useMemo(() => {
    const xs: number[] = [];
    for (let v = -size.width / 2; v <= size.width / 2; v += 24) xs.push(v);
    return xs;
  }, [size.width]);

  const W = size.width;
  const D = size.depth;

  return (
    <main className="page layout-builder">
      <div className="toolbar">
        <h1>Field</h1>
        <select aria-label="Field preset" value={id} onChange={(e) => setId(e.target.value)}>
          {list.map((p) => (
            <option key={p.id} value={p.id}>
              {p.displayName || p.id}
              {p.manualRevision ? ` · ${p.manualRevision}` : ""}
            </option>
          ))}
        </select>
        <Segmented
          label="Editor"
          value={tab}
          onChange={switchTab}
          options={[
            ["layout", "Layout"],
            ["json", "Field JSON"],
            ["scoring", "Scoring rules"],
          ]}
        />
        <span className="spacer" />
        {meta?.stale && <span className="pill warn">Older than latest manual</span>}
        {meta?.verifyAgainstManual && (
          <span className="pill warn" title="Confirm CAD poses against the game manual before a tournament">
            Verify against manual
          </span>
        )}
        {dirty && tab !== "scoring" && <span className="pill">Unsaved changes</span>}
        {tab === "scoring" ? (
          <button type="button" className="btn primary" onClick={saveScoring} disabled={saving || !scoringDoc}>
            {saving ? "Saving…" : "Save scoring"}
          </button>
        ) : (
          <button type="button" className="btn primary" onClick={saveField} disabled={saving || !doc}>
            {saving ? "Saving…" : "Save field"}
          </button>
        )}
      </div>

      {tab === "layout" && (
        <div className="builder-field">
          <Panel
            className="grow"
            title={doc?.displayName || "Field"}
            sub={`${W} × ${D} in · origin at center, +y toward far wall`}
            bodyClass={view === "map" ? "map-wrap" : "viewport"}
            actions={
              <Segmented
                label="View"
                value={view}
                onChange={setView}
                options={[
                  ["map", "2D map"],
                  ["3d", "3D"],
                ]}
              />
            }
            footer={
              <div className="legend">
                <span>
                  <i style={{ background: theme.allianceRed }} /> Red alliance
                </span>
                <span>
                  <i style={{ background: theme.allianceBlue }} /> Blue alliance
                </span>
                <span>
                  <i style={{ background: theme.restricted }} /> Restricted
                </span>
                <span>
                  <i style={{ background: theme.gold }} /> Tape / launch
                </span>
                <span>
                  <i style={{ background: "#9a9092" }} /> Neutral
                </span>
                <span>
                  <i style={{ background: "transparent", outline: "2px solid #fff", outlineOffset: -2, boxShadow: "0 0 0 1px #999" }} /> Selected
                </span>
              </div>
            }
          >
            {!doc ? (
              <Empty dark title="Loading field…" />
            ) : view === "3d" ? (
              cadPreview && <FieldScene frame={cadPreview} showFov={false} />
            ) : (
              <svg
                className="map"
                viewBox={`${-W / 2 - MAP_PAD} ${-D / 2 - MAP_PAD} ${W + MAP_PAD * 2} ${D + MAP_PAD * 2}`}
                role="img"
                aria-label="Top-down field map. Click an element to select it."
                onClick={(e) => pickAt(e.clientX, e.clientY, e.currentTarget)}
              >
                <rect x={-W / 2} y={-D / 2} width={W} height={D} fill={theme.field} />
                {ticks.map((v) => (
                  <g key={v}>
                    <line x1={v} y1={-D / 2} x2={v} y2={D / 2} stroke={theme.grid} strokeWidth="0.4" />
                    <line x1={-W / 2} y1={v} x2={W / 2} y2={v} stroke={theme.grid} strokeWidth="0.4" />
                  </g>
                ))}
                {ticks.map((v) => (
                  <text key={`t-${v}`} x={v} y={D / 2 + 5.5} fill={theme.mapLabel} fontSize="3.2" textAnchor="middle" fontFamily="IBM Plex Mono, monospace">
                    {v}
                  </text>
                ))}
                {ticks.map((v) => (
                  <text key={`l-${v}`} x={-W / 2 - 2} y={-v + 1.1} fill={theme.mapLabel} fontSize="3.2" textAnchor="end" fontFamily="IBM Plex Mono, monospace">
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
                        fillOpacity={on ? 0.16 : 0.05}
                        stroke={on ? theme.selected : elColor(el)}
                        strokeWidth={on ? 1.1 : 0.6}
                        strokeDasharray={on ? undefined : "2 1.5"}
                      />
                    );
                  })}
                {elements
                  .filter((el) => el.type === "tape")
                  .map((el, idx) => {
                    const x = el.pose?.x ?? 0;
                    const y = -(el.pose?.y ?? 0);
                    const { w, d } = elSize(el);
                    return <rect key={`tape-${el.id}-${idx}`} x={x - w / 2} y={y - d / 2} width={w} height={d} fill={theme.gold} opacity={0.9} />;
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
                        rx={0.6}
                        fill={elColor(el)}
                        fillOpacity={on ? 1 : 0.8}
                        stroke={on ? theme.selected : "rgba(0,0,0,0.35)"}
                        strokeWidth={on ? 1.2 : 0.3}
                        style={{ cursor: "pointer" }}
                      />
                    );
                  })}
                <rect x={-W / 2} y={-D / 2} width={W} height={D} fill="none" stroke="#bdb4b6" strokeWidth="1.2" />
              </svg>
            )}
          </Panel>

          <Panel
            className="grow"
            title="Elements"
            sub={String(elements.length)}
            bodyClass="panel-body flush scroll"
            footer={
              current ? (
                <div className="stack" style={{ width: "100%", gap: "0.6rem" }}>
                  <div className="row" style={{ justifyContent: "space-between" }}>
                    <b className="mono" style={{ fontSize: 13 }}>
                      {current.id}
                    </b>
                    <span className="note">
                      {current.shape?.kind || "aabb"} · {elSize(current).w} × {elSize(current).d} in
                      {current.alliance ? ` · ${current.alliance}` : ""}
                    </span>
                  </div>
                  <div className="fields three">
                    <NumberField id="el-x" label="X" unit="in" value={current.pose?.x ?? 0} onChange={(n) => updatePose({ x: n })} />
                    <NumberField id="el-y" label="Y" unit="in" value={current.pose?.y ?? 0} onChange={(n) => updatePose({ y: n })} />
                    <NumberField id="el-h" label="Heading" unit="°" value={current.pose?.headingDeg ?? 0} onChange={(n) => updatePose({ headingDeg: n })} />
                  </div>
                  {current.tags && current.tags.length > 0 && (
                    <div className="row" style={{ gap: "0.3rem" }}>
                      {current.tags.map((t) => (
                        <span key={t} className="chip-type">
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <span className="note">Select an element on the map or in the list to edit its pose.</span>
              )
            }
          >
            <div className="list-search">
              <input type="search" placeholder="Filter by id, type, or tag" aria-label="Filter elements" value={query} onChange={(e) => setQuery(e.target.value)} />
            </div>
            <ul className="list">
              {filtered.map((el, idx) => (
                <li key={`${el.id}-${idx}`}>
                  <button type="button" className={`list-item ${el.id === selected ? "on" : ""}`} onClick={() => setSelected(el.id)} style={{ padding: "0.45rem 1rem" }}>
                    <span className="dot" style={{ background: elColor(el) }} />
                    <span className="grow title mono" style={{ fontSize: 12.5 }}>
                      {el.id}
                    </span>
                    <span className="chip-type">{el.type || "element"}</span>
                  </button>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      )}

      {tab === "json" && (
        <div className="builder-json">
          <Panel className="grow" title="Field JSON" sub={doc?.id}>
            <textarea aria-label="Field JSON" className="json-editor" spellCheck={false} value={jsonText}
              onChange={(e) => {
                setJsonText(e.target.value);
                setDirty(true);
              }}
            />
          </Panel>
          <Panel
            title="Validation"
            actions={
              <button type="button" className="btn sm" onClick={validateField}>
                <Icon name="check" size={14} /> Validate
              </button>
            }
          >
            <div className="stack">
              <LintResult lint={fieldLint} />
              <p className="note">Edits here replace the whole document. Switching back to Layout applies them to the map.</p>
            </div>
          </Panel>
        </div>
      )}

      {tab === "scoring" && (
        <div className="builder-json">
          <Panel
            className="grow"
            title="Scoring rules"
            actions={
              <select aria-label="Scoring preset" style={{ width: "auto" }} value={scoringId} onChange={(e) => setScoringId(e.target.value)}>
                {scoringList.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.displayName || p.id}
                  </option>
                ))}
              </select>
            }
          >
            <textarea aria-label="Scoring JSON" className="json-editor" spellCheck={false} value={scoringDoc} onChange={(e) => setScoringDoc(e.target.value)} />
          </Panel>
          <Panel
            title="Validation"
            actions={
              <button type="button" className="btn sm" onClick={validateScoring}>
                <Icon name="check" size={14} /> Validate
              </button>
            }
          >
            <div className="stack">
              <LintResult lint={scoringLint} />
              <p className="note">Point values are data. Copy them from the game manual; do not invent them.</p>
            </div>
          </Panel>
        </div>
      )}
    </main>
  );
}
