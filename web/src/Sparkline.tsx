import { theme } from "./theme";

function fmt(n: number) {
  return Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(2).replace(/\.?0+$/, "");
}

export function Sparkline({
  values,
  label,
  color = theme.brand,
}: {
  values: number[];
  label: string;
  color?: string;
}) {
  const w = 320;
  const h = 96;
  const padL = 34;
  const padR = 6;
  const padY = 10;
  const nums = values.filter((v) => Number.isFinite(v));
  if (nums.length < 1) {
    // HTML rather than SVG so the text never scales with the panel's aspect ratio.
    return (
      <div className="chart chart-empty" role="img" aria-label={label}>
        No samples yet
      </div>
    );
  }
  const series = nums.length === 1 ? [nums[0], nums[0]] : nums;
  const lo = Math.min(...series);
  const hi = Math.max(...series);
  const flat = hi === lo;
  const pad = flat ? Math.max(1, Math.abs(hi) * 0.1) : 0;
  const min = lo - pad;
  const max = hi + pad;
  const span = max - min;
  const x = (i: number) => padL + (i / (series.length - 1)) * (w - padL - padR);
  const y = (v: number) => h - padY - ((v - min) / span) * (h - padY * 2);
  const d = series.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `${d} L${x(series.length - 1).toFixed(1)},${h - padY} L${padL},${h - padY} Z`;
  const last = series[series.length - 1];
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      <line x1={padL} x2={w - padR} y1={padY} y2={padY} stroke={theme.line} />
      <line x1={padL} x2={w - padR} y1={h - padY} y2={h - padY} stroke={theme.line} />
      <text x={padL - 6} y={padY + 4} fill={theme.muted} fontSize="10" textAnchor="end" fontFamily="IBM Plex Mono, monospace">
        {fmt(max)}
      </text>
      <text x={padL - 6} y={h - padY + 3} fill={theme.muted} fontSize="10" textAnchor="end" fontFamily="IBM Plex Mono, monospace">
        {fmt(min)}
      </text>
      <path d={area} fill={color} opacity={0.1} />
      <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
      <circle cx={x(series.length - 1)} cy={y(last)} r="3.5" fill={color} stroke="#fff" strokeWidth="1.5" />
    </svg>
  );
}

export function Whisker({
  lo,
  mean,
  hi,
  min,
  max,
}: {
  lo: number;
  mean: number;
  hi: number;
  min: number;
  max: number;
}) {
  const span = Math.max(1e-6, max - min);
  const left = ((lo - min) / span) * 100;
  const width = Math.max(((hi - lo) / span) * 100, 0.8);
  const mid = ((mean - min) / span) * 100;
  return (
    <div className="whisker" aria-hidden="true">
      <div className="whisker-range" style={{ left: `${left}%`, width: `${width}%` }} />
      <div className="whisker-mean" style={{ left: `${mid}%` }} />
    </div>
  );
}
