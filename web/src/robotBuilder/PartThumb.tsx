import { useEffect, useRef } from "react";
import type { CatalogPartPreview, CatalogPartSummary } from "../api";
import { API } from "../api";
import { proxyColor } from "./partVisual";

const THUMB = 72;

function isoProject(x: number, y: number, z: number): [number, number] {
  return [(x - y) * 0.86, (x + y) * 0.5 - z];
}

function drawProxyPreview(
  canvas: HTMLCanvasElement,
  preview: Pick<CatalogPartPreview, "kind" | "sizeIn" | "radiusIn" | "lengthIn" | "tags">,
  color: string,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#2a2426";
  ctx.fillRect(0, 0, w, h);
  const size = preview.sizeIn || [2, 2, 1];
  const radius = preview.radiusIn || 0.6;
  const length = preview.lengthIn || radius * 2;
  let sx = size[0];
  let sy = size[1];
  let sz = size[2];
  if (preview.kind === "sphere") {
    sx = sy = sz = radius * 2;
  } else if (preview.kind === "cylinder" || preview.kind === "capsule") {
    sx = radius * 2;
    sy = length;
    sz = radius * 2;
  }
  const corners: [number, number, number][] = [
    [-sx / 2, -sy / 2, 0],
    [sx / 2, -sy / 2, 0],
    [sx / 2, sy / 2, 0],
    [-sx / 2, sy / 2, 0],
    [-sx / 2, -sy / 2, sz],
    [sx / 2, -sy / 2, sz],
    [sx / 2, sy / 2, sz],
    [-sx / 2, sy / 2, sz],
  ];
  const projected = corners.map(([x, y, z]) => isoProject(x, y, z));
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of projected) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const span = Math.max(maxX - minX, maxY - minY, 0.8);
  const scale = (Math.min(w, h) * 0.72) / span;
  const ox = w / 2 - ((minX + maxX) / 2) * scale;
  const oy = h / 2 + ((minY + maxY) / 2) * scale;
  const pt = (i: number): [number, number] => [ox + projected[i][0] * scale, oy - projected[i][1] * scale];
  const face = (idxs: number[], fill: string) => {
    ctx.beginPath();
    const first = pt(idxs[0]);
    ctx.moveTo(first[0], first[1]);
    for (const idx of idxs.slice(1)) {
      const p = pt(idx);
      ctx.lineTo(p[0], p[1]);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = "rgba(20,14,16,0.35)";
    ctx.stroke();
  };
  face([3, 2, 6, 7], color);
  face([1, 2, 6, 5], shade(color, 0.78));
  face([4, 5, 6, 7], shade(color, 1.12));
}

function shade(hex: string, amount: number) {
  const raw = hex.replace("#", "");
  const n = parseInt(raw.length === 3 ? raw.split("").map((c) => c + c).join("") : raw, 16);
  const r = Math.min(255, Math.round(((n >> 16) & 255) * amount));
  const g = Math.min(255, Math.round(((n >> 8) & 255) * amount));
  const b = Math.min(255, Math.round((n & 255) * amount));
  return `rgb(${r},${g},${b})`;
}

export function PartThumb({ part }: { part: CatalogPartSummary }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const preview = part.preview;
  const thumbnail = preview?.thumbnailAsset ? `${API}/robot-assets/${preview.thumbnailAsset}` : null;
  const cad = preview?.source === "cad" && Boolean(thumbnail);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawProxyPreview(canvas, preview || { kind: "box", sizeIn: [2, 2, 1], tags: part.tags }, proxyColor(preview?.tags || part.tags, part.sku));
  }, [part.sku, part.tags, preview]);
  return (
    <span className="catalog-thumb" data-testid={`catalog-thumb-${part.sku}`} data-preview={cad ? "cad" : "proxy"}>
      <canvas ref={canvasRef} width={THUMB} height={THUMB} aria-hidden="true" />
      {thumbnail && <img src={thumbnail} alt={`${part.displayName} rendered preview`} />}
      <span className={`catalog-thumb-badge ${cad ? "cad" : "proxy"}`}>{cad ? "Official CAD" : "Proxy"}</span>
    </span>
  );
}
