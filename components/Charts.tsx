"use client";
export const COLORS: Record<string, string> = { World: "var(--World)", Sports: "var(--Sports)", Business: "var(--Business)", "Sci/Tech": "var(--SciTech)" };
export const fmt = (n: number) => n.toLocaleString("en-US");

export function Bars({ items, color = "var(--acc)", max, unit = "" }: { items: { label: string; value: number; color?: string }[]; color?: string; max?: number; unit?: string }) {
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return (<div>{items.map((i) => (
    <div className="row" key={i.label}>
      <span className="l" title={i.label} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{i.label}</span>
      <span className="bar"><i style={{ width: `${(Math.abs(i.value) / m) * 100}%`, background: i.color ?? color }} /></span>
      <span className="v">{unit === "%" ? i.value.toFixed(1) : fmt(+i.value.toFixed(4))}{unit}</span>
    </div>))}</div>);
}

export function Hist({ bins, color }: { bins: number[]; color: string }) {
  const m = Math.max(...bins);
  return (<svg viewBox="0 0 250 70" width="100%" role="img" aria-label="histogram">{bins.map((b, i) => (
    <rect key={i} x={i * 10 + 1} y={65 - (b / m) * 60} width={8} height={(b / m) * 60} fill={color} rx={1.5} />))}
    <text x="0" y="69" fontSize="6" fill="var(--mute)">0</text><text x="228" y="69" fontSize="6" fill="var(--mute)">100+ words</text></svg>);
}

export function Scatter({ words, x, y, aff, labels }: { words: string[]; x: number[]; y: number[]; aff: number[]; labels: string[] }) {
  const [x0, x1, y0, y1] = [Math.min(...x), Math.max(...x), Math.min(...y), Math.max(...y)];
  const px = (v: number) => 20 + ((v - x0) / (x1 - x0)) * 560, py = (v: number) => 20 + ((y1 - v) / (y1 - y0)) * 360;
  const show = new Set(["war", "game", "stock", "software", "president", "team", "market", "computer", "oil", "internet", "season", "iraq"]);
  return (<svg viewBox="0 0 600 400" width="100%" role="img" aria-label="Word2Vec PCA projection">
    {words.map((w, i) => (<circle key={w} cx={px(x[i])} cy={py(y[i])} r={4} fill={COLORS[labels[aff[i]]]} opacity={0.65}><title>{w}</title></circle>))}
    {words.map((w, i) => show.has(w) && (<text key={w} x={px(x[i]) + 6} y={py(y[i]) + 3} fontSize="11" fill="var(--ink)">{w}</text>))}</svg>);
}

export function Heat({ cm, labels }: { cm: number[][]; labels: string[] }) {
  return (<div className="scroll"><table><thead><tr><th>true ↓ / pred →</th>{labels.map((l) => <th className="n" key={l}>{l}</th>)}</tr></thead><tbody>
    {cm.map((r, i) => { const s = r.reduce((a, b) => a + b, 0); return (<tr key={i}><th>{labels[i]}</th>{r.map((v, j) => (
      <td className="n" key={j} style={{ background: `color-mix(in srgb, ${i === j ? "var(--Sports)" : "#dc2626"} ${Math.round((v / s) * (i === j ? 70 : 400))}%, transparent)` }}>{fmt(v)}</td>))}</tr>); })}
  </tbody></table></div>);
}
