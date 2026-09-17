import { useEffect, useRef, useState } from "react";
import type { CatalogCacheBatch, CatalogManufacturer } from "../api";
import { ApiError, cancelCatalogCacheAll, getCatalogCacheAll, requestCatalogCacheAll, retryCatalogCacheAll } from "../api";
import {
  cacheBatchBusy,
  cacheBatchProgressPercent,
  cacheBatchStateLabel,
  catalogCacheAllRequest,
  catalogCacheAllScopeLabel,
  formatCacheBatchCounts,
  summarizeCacheBatch,
  summarizeSkipped,
} from "./cadVisual";

export function CacheAllCadControl({
  manufacturer,
  query,
  tag,
  skus,
  filterReady = true,
  onCatalogRefresh,
}: {
  manufacturer: CatalogManufacturer | "";
  query?: string;
  tag?: string;
  skus?: string[];
  filterReady?: boolean;
  onCatalogRefresh?: () => void;
}) {
  const [batch, setBatch] = useState<CatalogCacheBatch | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef<number | null>(null);
  const refreshRef = useRef(onCatalogRefresh);
  refreshRef.current = onCatalogRefresh;

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
    if (previousBusy && !nextBusy) {
      refreshRef.current?.();
    }
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

  useEffect(() => {
    let cancelled = false;
    void getCatalogCacheAll()
      .then((row) => {
        if (cancelled || !row.id) return;
        applyBatch(row);
        if (cacheBatchBusy(row.state)) schedulePoll(row.id);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      stopPoll();
    };
    // Restore an in-flight batch once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function cacheAll() {
    const payload = catalogCacheAllRequest({ manufacturer, query, tag, skus, filterReady });
    if (!payload) return;
    setBusy(true);
    setError("");
    try {
      const started = await requestCatalogCacheAll(payload);
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

  async function retryFailed() {
    if (!batch?.id) return;
    setBusy(true);
    setError("");
    try {
      const next = await retryCatalogCacheAll(batch.id);
      applyBatch(next);
      if (next.id && cacheBatchBusy(next.state)) schedulePoll(next.id);
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const running = cacheBatchBusy(batch?.state);
  const failed = batch?.counts.failed || 0;
  const skippedNote = summarizeSkipped(batch?.items);
  const percent = cacheBatchProgressPercent(batch);
  const current = batch?.items.find((row) => row.state === "converting");
  const stateLabel = current ? `Caching ${current.displayName || current.sku}…` : cacheBatchStateLabel(batch?.state);
  const summary = batch && !running ? summarizeCacheBatch(batch) : "";
  const payload = catalogCacheAllRequest({ manufacturer, query, tag, skus, filterReady });
  const scopeLabel = catalogCacheAllScopeLabel({ manufacturer, query, tag });

  return (
    <div className="catalog-cache-all-control" data-testid="catalog-cache-all-control">
      <div className="catalog-cache-all">
        <button type="button" className="btn sm" data-testid="catalog-cache-all" disabled={busy || running || !payload} onClick={() => void cacheAll()}>
          {running ? "Caching all…" : "Cache all CAD"}
        </button>
        {running && (
          <button type="button" className="btn sm" data-testid="catalog-cache-all-cancel" disabled={busy} onClick={() => void cancelAll()}>
            Cancel
          </button>
        )}
        {!running && failed > 0 && (
          <button type="button" className="btn sm" data-testid="catalog-cache-all-retry" disabled={busy} onClick={() => void retryFailed()}>
            Retry failed
          </button>
        )}
      </div>
      {scopeLabel ? (
        <span className="meta" data-testid="catalog-cache-all-scope">
          {scopeLabel}
        </span>
      ) : null}
      {batch?.id && (
        <div className="catalog-cache-progress" data-testid="catalog-cache-all-progress" data-state={batch.state}>
          <span data-testid="catalog-cache-all-state">{stateLabel}</span>
          <div
            className="progress catalog-cache-all-bar"
            data-testid="catalog-cache-all-bar"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
          >
            <span style={{ width: `${percent}%` }} />
          </div>
          <span data-testid="catalog-cache-all-counts">{formatCacheBatchCounts(batch.counts)}</span>
          {skippedNote ? (
            <span className="meta" data-testid="catalog-cache-all-skipped">
              {skippedNote}
            </span>
          ) : null}
          {summary ? (
            <span className="meta" data-testid="catalog-cache-all-summary">
              {summary}
            </span>
          ) : null}
        </div>
      )}
      {error ? (
        <p className="note" data-testid="catalog-cache-all-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
