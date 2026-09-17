import { useMemo, useState } from "react";
import type { CatalogManufacturer, CatalogPartSummary } from "../api";
import { ApiError, getCatalogJob, getCatalogPart, requestCatalogDownload } from "../api";
import { Empty, Field, Panel } from "../ui";
import { CacheAllCadControl } from "./CacheAllCadControl";
import { catalogCacheLabel } from "./cadVisual";
import { PartThumb } from "./PartThumb";

const CATEGORIES = [
  { id: "all", label: "All", tags: [] },
  { id: "structure", label: "Structure", tags: ["channel", "extrusion", "plate", "bracket"] },
  { id: "drive", label: "Drive", tags: ["wheel_mecanum", "wheel_traction", "motor", "gearbox"] },
  { id: "motion", label: "Motion", tags: ["shaft", "bearing", "hub", "gear", "sprocket", "pulley"] },
  { id: "mechanisms", label: "Mechanisms", tags: ["servo", "intake_roller", "flywheel"] },
  { id: "electronics", label: "Electronics", tags: ["sensor", "control_hub", "battery"] },
] as const;

type CacheJob = { state: string; error?: string };

export function CatalogPanel({
  parts,
  query,
  manufacturer,
  tag,
  status,
  error,
  onQuery,
  onManufacturer,
  onTag,
  onPick,
  onCatalogRefresh,
  replacing,
  onClose,
  compatibleSkus,
  mountContext,
}: {
  parts: CatalogPartSummary[];
  query: string;
  manufacturer: CatalogManufacturer | "";
  tag: string;
  status: "loading" | "ok" | "error";
  error?: string;
  onQuery: (q: string) => void;
  onManufacturer: (m: CatalogManufacturer | "") => void;
  onTag: (tag: string) => void;
  onPick: (sku: string) => void;
  onCatalogRefresh?: () => void;
  replacing?: boolean;
  onClose?: () => void;
  compatibleSkus?: Set<string> | null;
  mountContext?: string;
}) {
  const [busySku, setBusySku] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, CacheJob>>({});
  const [cacheError, setCacheError] = useState("");
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]["id"]>("all");
  const [focusedSku, setFocusedSku] = useState<string | null>(null);
  const visibleParts = useMemo(() => {
    const categoryTags = CATEGORIES.find((row) => row.id === category)?.tags || [];
    const categorized = categoryTags.length
      ? parts.filter((part) => categoryTags.some((categoryTag) => part.tags.includes(categoryTag)))
      : parts;
    return compatibleSkus !== undefined
      ? categorized.filter((part) => compatibleSkus?.has(part.sku))
      : categorized;
  }, [category, compatibleSkus, parts]);
  const focusedPart = visibleParts.find((part) => part.sku === focusedSku) || null;

  async function cachePart(sku: string) {
    setBusySku(sku);
    setCacheError("");
    setJobs((prev) => ({ ...prev, [sku]: { state: "queued" } }));
    try {
      const started = await requestCatalogDownload(sku);
      if (started.jobId) {
        setJobs((prev) => ({ ...prev, [sku]: { state: started.state || "queued" } }));
        const row = await pollCatalogJob(started.jobId, (state) => {
          setJobs((prev) => ({ ...prev, [sku]: { state } }));
        });
        if (row.state === "failed") {
          const message = row.error?.message || "CAD conversion failed";
          setCacheError(message);
          setJobs((prev) => ({ ...prev, [sku]: { state: "failed", error: message } }));
        } else {
          setJobs((prev) => ({ ...prev, [sku]: { state: "done" } }));
          onCatalogRefresh?.();
        }
      } else {
        const part = await getCatalogPart(sku).catch(() => null);
        setJobs((prev) => ({ ...prev, [sku]: { state: part?.cache.state || started.state || "ready" } }));
        onCatalogRefresh?.();
      }
    } catch (err) {
      const message = err instanceof ApiError || err instanceof Error ? err.message : String(err);
      setCacheError(message);
      setJobs((prev) => ({ ...prev, [sku]: { state: "failed", error: message } }));
    } finally {
      setBusySku(null);
    }
  }

  const sub = replacing
    ? "Click a SKU to replace the selected part"
    : status === "loading"
      ? "Loading parts…"
      : status === "error"
        ? "Catalog failed to load"
        : `${visibleParts.length} parts · choose one to place`;
  return (
    <Panel
      className="catalog-panel parts-drawer"
      title={replacing ? "Replace part" : "Add a part"}
      sub={mountContext ? `${mountContext} · ${sub}` : sub}
      bodyClass="panel-body catalog-body"
      testId="catalog-panel"
      actions={
        onClose ? (
          <button type="button" className="btn icon" aria-label="Close parts" onClick={onClose}>
            ×
          </button>
        ) : null
      }
    >
      <div className="catalog-filters" onKeyDown={(e) => e.stopPropagation()}>
        <Field id="cat-q" label="Search">
          <div className="catalog-search-row">
            <input id="cat-q" data-testid="catalog-search" type="text" value={query} onChange={(e) => onQuery(e.target.value)} placeholder="channel, mecanum, motor…" autoComplete="off" />
            {query ? (
              <button type="button" className="btn sm" data-testid="catalog-search-clear" onClick={() => onQuery("")}>
                Clear
              </button>
            ) : null}
          </div>
        </Field>
        <div className="catalog-manufacturers" aria-label="Manufacturer filter">
          {(["", "gobilda", "rev"] as const).map((value) => (
            <button
              key={value || "all"}
              type="button"
              className={`btn sm ${manufacturer === value ? "primary" : ""}`}
              onClick={() => onManufacturer(value)}
            >
              {value === "" ? "All brands" : value === "gobilda" ? "goBILDA" : "REV"}
            </button>
          ))}
        </div>
        <div className="catalog-categories" role="tablist" aria-label="Part category">
          {CATEGORIES.map((row) => (
            <button
              key={row.id}
              type="button"
              role="tab"
              aria-selected={category === row.id}
              className={category === row.id ? "on" : ""}
              onClick={() => {
                setCategory(row.id);
                if (tag) onTag("");
              }}
            >
              {row.label}
            </button>
          ))}
        </div>
        <CacheAllCadControl
          manufacturer={manufacturer}
          query={query}
          tag={category === "all" ? tag : category}
          skus={visibleParts.map((part) => part.sku)}
          filterReady={status === "ok"}
          onCatalogRefresh={onCatalogRefresh}
        />
      </div>
      {status === "error" && <Empty title="Catalog could not load">{error || "Retry search or check that the API is connected."}</Empty>}
      {cacheError && (
        <p className="note" data-testid="catalog-cache-error" role="alert">
          {cacheError}
        </p>
      )}
      {status !== "error" && visibleParts.length === 0 && <Empty title={status === "loading" ? "Loading catalog…" : "No matching parts"}>{status === "loading" ? "Fetching goBILDA and REV SKUs." : "Clear search or filters to see the full catalog."}</Empty>}
      <ul className="list catalog-list" data-testid="catalog-list" aria-live="polite" aria-label="Catalog parts">
        {visibleParts.map((part) => {
          const job = jobs[part.sku];
          const caching = busySku === part.sku || job?.state === "queued" || job?.state === "running";
          const label = caching ? "CAD caching…" : job?.state === "failed" ? job.error || "CAD failed" : catalogCacheLabel(part.cache.state, part.cache.reason);
          return (
            <li key={part.sku} className={`catalog-row catalog-card${focusedSku === part.sku ? " selected" : ""}`}>
              <button
                type="button"
                className="list-item"
                data-testid={`catalog-part-${part.sku}`}
                onClick={() => setFocusedSku(part.sku)}
              >
                <PartThumb part={part} />
                <span className="grow">
                  <span className="title" style={{ display: "block" }}>
                    {part.displayName}
                  </span>
                  <span className="meta" style={{ display: "block" }} data-testid={`catalog-cache-state-${part.sku}`}>
                    {part.sku} · {part.manufacturer} · {label}
                  </span>
                  <span className="meta catalog-part-spec" style={{ display: "block" }}>
                    {part.massKg.toFixed(3)} kg · {part.tags.slice(0, 3).join(" · ")}
                  </span>
                </span>
              </button>
              {part.downloadEnabled && part.cache.state !== "ready" && (
                <button
                  type="button"
                  className="btn sm"
                  data-testid={`catalog-cache-${part.sku}`}
                  aria-label={`Cache CAD for ${part.sku}`}
                  disabled={caching}
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={() => void cachePart(part.sku)}
                >
                  {caching ? "Caching…" : job?.state === "failed" ? "Retry CAD" : "Cache CAD"}
                </button>
              )}
            </li>
          );
        })}
      </ul>
      {focusedPart && (
        <div className="catalog-detail" data-testid="catalog-detail">
          <PartThumb part={focusedPart} />
          <div className="grow">
            <strong>{focusedPart.displayName}</strong>
            <span>{focusedPart.sku} · {focusedPart.manufacturer}</span>
            <span>{focusedPart.massKg.toFixed(3)} kg · {focusedPart.tags.join(" · ")}</span>
            <span>{mountContext ? `Compatible with ${mountContext}` : "Choose as the assembly root"}</span>
          </div>
          <button
            type="button"
            className="btn primary"
            data-testid="catalog-place-selected"
            onClick={() => {
              onPick(focusedPart.sku);
            }}
          >
            {mountContext ? "Preview attachment" : "Place part"}
          </button>
        </div>
      )}
    </Panel>
  );
}

async function pollCatalogJob(jobId: string, onState: (state: string) => void) {
  const deadline = Date.now() + 45_000;
  let row = await getCatalogJob(jobId);
  onState(row.state);
  while (Date.now() < deadline && !["done", "failed"].includes(row.state)) {
    await new Promise((resolve) => window.setTimeout(resolve, 400));
    row = await getCatalogJob(jobId);
    onState(row.state);
  }
  return row;
}
