export type Model = {
  labels: string[]; vocabulary: Record<string, number>; idf: number[];
  coef: number[][]; intercept: number[]; stopwords: string[];
  test_accuracy: number; test_macro_f1: number;
};
export type Result = {
  label: number; name: string; scores: number[]; probs: number[];
  terms: { term: string; contribution: number }[]; tokens: number; known: number;
};

/** Mirrors scripts/pipeline.py `clean_web` exactly (lowercase, strip HTML/URLs/non-letters, stopwords, len>2). */
export function clean(text: string, stop: Set<string>): string[] {
  const t = text.toLowerCase().replace(/<[^>]+>/g, " ").replace(/https?:\/\/\S+|www\.\S+/g, " ").replace(/[^a-z\s]/g, " ");
  return t.split(/\s+/).filter((w) => w && w.length > 2 && !stop.has(w));
}

/** Pure-TS TF-IDF (unigram+bigram, sublinear tf, l2) + LinearSVC decision function. */
export function classify(m: Model, text: string, stop?: Set<string>): Result {
  const sw = stop ?? new Set(m.stopwords);
  const toks = clean(text, sw);
  const counts = new Map<number, [string, number]>();
  const bump = (term: string) => {
    const i = m.vocabulary[term];
    if (i === undefined) return;
    const e = counts.get(i);
    e ? (e[1] += 1) : counts.set(i, [term, 1]);
  };
  toks.forEach((t, i) => { bump(t); if (i + 1 < toks.length) bump(t + " " + toks[i + 1]); });
  const vec = new Map<number, [string, number]>();
  let norm = 0;
  counts.forEach(([term, c], i) => { const v = (1 + Math.log(c)) * m.idf[i]; vec.set(i, [term, v]); norm += v * v; });
  norm = Math.sqrt(norm) || 1;
  const scores = m.intercept.slice();
  vec.forEach(([, v], i) => m.coef.forEach((row, k) => { scores[k] += (row[i] * v) / norm; }));
  const label = scores.indexOf(Math.max(...scores));
  const mx = Math.max(...scores);
  const ex = scores.map((s) => Math.exp((s - mx) * 3));
  const z = ex.reduce((a, b) => a + b, 0);
  const terms = [...vec.entries()]
    .map(([i, [term, v]]) => ({ term, contribution: (m.coef[label][i] * v) / norm }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 8);
  return { label, name: m.labels[label], scores, probs: ex.map((e) => e / z), terms, tokens: toks.length, known: vec.size };
}
