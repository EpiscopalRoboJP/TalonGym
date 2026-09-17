import { useEffect, useMemo, useRef, useState } from "react";
import type { CatalogCacheBatch, CatalogPart } from "../api";
import { ApiError, cancelCatalogCacheAll, getCatalogCacheAll, requestCatalogCacheAll } from "../api";
import {
  cacheBatchBusy,
  cacheBatchProgressPercent,
  cacheBatchStateLabel,
  catalogCacheLabel,
  formatCacheBatchCounts,
  assemblyCadCounts,
} from "./cadVisual";

function missingSkus(skus: string[], parts: Record<string, CatalogPart | undefined>): string[] {
  const unique = [...new Set(skus.filter(Boolean))];
  return unique.filter((sku) => {
    const part = parts[sku];
    const state = part?.cache?.state;
    const visual = part?.cache?.visualAsset || part?.preview?.visualAsset;
    return state !== "ready" || !visual;
  });
}

export function CacheRequiredCadControl({
  skus,
  parts,
  onRefresh,
}: {
  skus: string[];
  parts: Record<string, CatalogPart | undefined>;
  onRefresh?: () => void;
}) {
  const required = useMemo(() => missingSkus(skus, parts), [skus, parts]);
  const [batch, setBatch] = useState<CatalogCacheBatch | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef<number | null>(null);
  const refreshRef = useRef(onRefresh);
  refreshRef.current = onRefresh;

  function stopPoll() {
    if (pollRef.current != null) {
      window.clearTimeout(pollRef.current);
      pollRef.current = null;
    }
  }

  function applyBatch(next: CatalogCacheBatch | null, previousBusy = false) {
    const row = next && next.id ? next : null;
    setBatch(row);
    const nextBusy = cacheBatchBusy(row?.state);
    if (previousBusy && !nextBusy) refreshRef.current?.();
  }

  function schedulePoll(batchId: string) {
    stopPoll();
    pollRef.current = window.setTimeout(() => {
      void (async () => {
        try {
          const row = await getCatalogCacheAll(batchId);
          applyBatch(row, true);
          if (cacheBatchBusy(row.state)) schedulePoll(batchId);
        } catch (err) {
          const code = err instanceof ApiError ? err.code : "";
          if (code === "404" || code === "NOT_FOUND") {
            applyBatch(null);
            return;
          }
          schedulePoll(batchId);
        }
      })();
    }, 400);
  }

  useEffect(() => () => stopPoll(), []);

  async function cacheRequired() {
    if (!required.length) return;
    setBusy(true);
    setError("");
    try {
      const started = await requestCatalogCacheAll({ skus: required });
      applyBatch(started);
      if (started.id && cacheBatchBusy(started.state)) schedulePoll(started.id);
      if (started.id && !cacheBatchBusy(started.state)) refreshRef.current?.();
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function cancelAll() {
    if (!batch?.id) return;
    setBusy(true);
    try {
      applyBatch(await cancelCatalogCacheAll(batch.id), cacheBatchBusy(batch.state));
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const running = cacheBatchBusy(batch?.state);
  const percent = cacheBatchProgressPercent(batch);
  const current = batch?.items.find((row) => row.state === "converting");
  const failedItems = (batch?.items || []).filter((row) => row.state === "failed");
  const stateLabel = current ? `Caching ${current.displayName || current.sku}…` : cacheBatchStateLabel(batch?.state);
  const counts = assemblyCadCounts(skus, parts);
  const unavailable = required.map((sku) => {
    const part = parts[sku];
    return `${sku}: ${catalogCacheLabel(part?.cache?.state, part?.cache?.reason)}`;
  });

  if (!skus.length) return null;

  return (
    <div className="catalog-cache-required" data-testid="catalog-cache-required" style={{ pointerEvents: "none" }}>
      <div className="catalog-cache-all" style={{ pointerEvents: "auto" }}>
        <button
          type="button"
          className="btn sm"
          data-testid="catalog-cache-required-btn"
          disabled={busy || running || required.length === 0}
          onClick={() => void cacheRequired()}
        >
          {running ? "Caching required CAD…" : "Cache required CAD"}
        </button>
        {running && (
          <button type="button" className="btn sm" data-testid="catalog-cache-required-cancel" disabled={busy} onClick={() => void cancelAll()}>
            Cancel
          </button>
        )}
      </div>
      <span className="meta" data-testid="catalog-cache-required-status">
        {required.length ? `${required.length} SKU${required.length === 1 ? "" : "s"} missing official CAD` : "Required CAD is cached"}
      </span>
      <span className="meta" data-testid="catalog-cad-counts">
        {counts.cad} CAD / {counts.proxy} proxy
      </span>
      {required.length > 0 && (
        <ul className="catalog-cache-required-list" data-testid="catalog-cache-required-missing">
          {unavailable.slice(0, 8).map((row) => (
            <li key={row}>{row}</li>
          ))}
        </ul>
      )}
      {batch?.id && (
        <div className="catalog-cache-progress" data-testid="catalog-cache-required-progress" data-state={batch.state}>
          <span>{stateLabel}</span>
          <div className="progress catalog-cache-all-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
            <span style={{ width: `${percent}%` }} />
          </div>
          <span>{formatCacheBatchCounts(batch.counts)}</span>
        </div>
      )}
      {failedItems.map((row) => (
        <p key={row.sku} className="note" data-testid={`catalog-cache-required-error-${row.sku}`} role="alert">
          {row.sku}: {row.error?.message || row.reason || "CAD import failed"}
        </p>
      ))}
      {error ? (
        <p className="note" data-testid="catalog-cache-required-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
