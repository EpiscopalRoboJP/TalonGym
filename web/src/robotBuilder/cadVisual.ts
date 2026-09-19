import type { CacheBatchCounts, CacheBatchItem, CatalogCacheBatch, CatalogCacheState, RigidPartSpec } from "../api";

export const EMPTY_CACHE_COUNTS: CacheBatchCounts = {
  queued: 0,
  converting: 0,
  ready: 0,
  failed: 0,
  skipped: 0,
  cancelled: 0,
};

export function hasArticulatedCatalogParts(parts?: { id: string; collision?: unknown[] | null }[]): boolean {
  const rows = parts || [];
  if (rows.some((part) => part.id === "_catalog")) return true;
  const chassis = rows.find((part) => part.id === "chassis");
  const others = rows.some((part) => Boolean(part.id) && part.id !== "chassis");
  if (!others) return false;
  return !((chassis?.collision || []).length > 0);
}

export function hideChassisLump(options: {
  hideBody?: boolean;
  rigidParts?: { id: string; collision?: unknown[] | null }[];
}): boolean {
  return Boolean(options.hideBody) || hasArticulatedCatalogParts(options.rigidParts);
}

export function catalogCacheLabel(state?: CatalogCacheState | string | null, reason?: string | null): string {
  if (state === "ready") return "CAD cached";
  if (state === "stale") return "CAD stale";
  if (state === "incomplete") return "CAD incomplete";
  if (state === "invalid") return "CAD cache invalid";
  if (state === "unavailable") {
    if (reason === "download_disabled") return "CAD download disabled";
    if (reason === "cad_extra_required") return "CAD extra required; pip install -e '.[cad]'";
    return "CAD unavailable";
  }
  if (state === "caching" || reason === "caching") return "CAD caching…";
  return "CAD not cached";
}

export function cadPlaceholderLabel(options: {
  visualAsset?: string | null;
  cacheState?: CatalogCacheState | string;
  cacheReason?: string | null;
  failed?: boolean;
  ready?: boolean;
}): string | null {
  if (options.visualAsset && options.ready && !options.failed) return null;
  if (options.failed) return "CAD failed";
  if (options.visualAsset && !options.ready) return "Loading CAD…";
  return catalogCacheLabel(options.cacheState, options.cacheReason);
}

export function partHasCachedCad(part?: { cache?: { state?: string | null; visualAsset?: string | null } | null; preview?: { visualAsset?: string | null } | null } | null): boolean {
  const visual = part?.cache?.visualAsset || part?.preview?.visualAsset;
  return part?.cache?.state === "ready" && Boolean(visual);
}

export function assemblyCadCounts(
  skus: string[],
  parts: Record<string, { cache?: { state?: string | null; visualAsset?: string | null } | null; preview?: { visualAsset?: string | null } | null } | undefined>,
): { cad: number; proxy: number } {
  let cad = 0;
  let proxy = 0;
  for (const sku of skus) {
    if (partHasCachedCad(parts[sku])) cad += 1;
    else proxy += 1;
  }
  return { cad, proxy };
}

export function formatCacheBatchCounts(counts?: Partial<CacheBatchCounts> | null): string {
  const row = { ...EMPTY_CACHE_COUNTS, ...counts };
  return `Queued ${row.queued} · Converting ${row.converting} · Ready ${row.ready} · Failed ${row.failed} · Skipped ${row.skipped}`;
}

export function cacheBatchBusy(state?: string | null): boolean {
  return state === "queued" || state === "running" || state === "cancelling";
}

export function cacheBatchStateLabel(state?: string | null): string {
  if (state === "queued") return "Queueing catalog CAD…";
  if (state === "running") return "Downloading / converting CAD…";
  if (state === "cancelling") return "Cancelling remaining downloads…";
  if (state === "cancelled") return "Cache cancelled";
  if (state === "done") return "Cache complete";
  return "Cache idle";
}

export function cacheBatchProgressPercent(batch?: Pick<CatalogCacheBatch, "counts" | "total"> | null): number {
  const total = Number(batch?.total || 0);
  if (total <= 0) return 0;
  const counts = { ...EMPTY_CACHE_COUNTS, ...batch?.counts };
  const finished = counts.ready + counts.failed + counts.skipped + counts.cancelled;
  return Math.max(0, Math.min(100, Math.round((finished / total) * 100)));
}

export function skipReasonLabel(reason?: string | null): string {
  if (reason === "download_disabled") return "download disabled";
  if (reason === "no_source_url") return "no source URL";
  if (reason === "cad_extra_required") return "CAD extra required";
  if (reason === "already_cached") return "already cached";
  if (reason === "cancelled") return "cancelled";
  return reason || "skipped";
}

export function summarizeSkipped(items?: CacheBatchItem[] | null): string {
  const skipped = (items || []).filter((row) => row.state === "skipped");
  if (!skipped.length) return "";
  const reasons = [...new Set(skipped.map((row) => skipReasonLabel(row.reason)))];
  return `${skipped.length} skipped (${reasons.join(" / ")})`;
}

export function catalogCacheAllRequest(opts?: {
  manufacturer?: string | null;
  query?: string | null;
  tag?: string | null;
  skus?: string[] | null;
  filterReady?: boolean;
}): { manufacturer?: string; skus?: string[] } | null {
  const manufacturer = (opts?.manufacturer || "").trim() || undefined;
  const scoped = Boolean((opts?.query || "").trim() || (opts?.tag || "").trim());
  if (scoped && opts?.filterReady === false) return null;
  if (scoped) return { manufacturer, skus: [...(opts?.skus || [])] };
  return { manufacturer };
}

export function catalogCacheAllScopeLabel(opts?: { manufacturer?: string | null; query?: string | null; tag?: string | null }): string {
  const manufacturer = (opts?.manufacturer || "").trim();
  const scoped = Boolean((opts?.query || "").trim() || (opts?.tag || "").trim());
  if (scoped) return "Caches the current catalog filter";
  if (manufacturer === "gobilda") return "Caches goBILDA SKUs";
  if (manufacturer === "rev") return "Caches REV SKUs";
  if (manufacturer) return `Caches ${manufacturer} SKUs`;
  return "";
}

export function summarizeCacheBatch(batch?: Pick<CatalogCacheBatch, "counts" | "total" | "state"> | null): string {
  if (!batch) return "";
  const counts = { ...EMPTY_CACHE_COUNTS, ...batch.counts };
  const parts = [
    `Cached ${counts.ready} of ${batch.total}`,
    counts.failed ? `Failed ${counts.failed}` : "",
    counts.skipped ? `Skipped ${counts.skipped}` : "",
    counts.cancelled ? `Cancelled ${counts.cancelled}` : "",
  ].filter(Boolean);
  const prefix = batch.state === "cancelled" ? "Cancelled" : "Complete";
  return `${prefix}: ${parts.join(" · ")}`;
}
