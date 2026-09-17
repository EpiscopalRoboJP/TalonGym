export function nextId(prefix: string, used: string[]): string {
  let n = 1;
  while (used.includes(`${prefix}_${n}`)) n += 1;
  return `${prefix}_${n}`;
}

export function slugify(raw: string): string {
  const s = raw
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
  return s.length >= 2 ? s : `robot_${Date.now().toString(36)}`;
}
