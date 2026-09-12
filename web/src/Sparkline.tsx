export function Sparkline({
  values,
  label,
  color = "#6fbfa3",
}: {
  values: number[];
  label: string;
  color?: string;
}) {
  const w = 320;
  const h = 90;
  const nums = values.filter((v) => Number.isFinite(v));
  if (nums.length < 1) {
    return (
      <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
        <text x="12" y="48" fill="#8aa0ae" fontSize="12">
          Waiting for samples…
        </text>
      </svg>
    );
  }
  const series = nums.length === 1 ? [nums[0], nums[0]] : nums;
  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = max - min || 1;
  const d = series
    .map((v, i) => {
      const x = (i / (series.length - 1)) * (w - 8) + 4;
      const y = h - 8 - ((v - min) / span) * (h - 16);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      <path d={d} fill="none" stroke={color} strokeWidth="2" />
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
  const width = ((hi - lo) / span) * 100;
  const mid = ((mean - min) / span) * 100;
  return (
    <div className="whisker" aria-hidden="true">
      <div className="whisker-range" style={{ left: `${left}%`, width: `${Math.max(width, 1)}%` }} />
      <div className="whisker-mean" style={{ left: `${mid}%` }} />
    </div>
  );
}
