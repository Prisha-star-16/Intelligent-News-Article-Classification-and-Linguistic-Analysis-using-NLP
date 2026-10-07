"use client";
import { useEffect, useState } from "react";
import Classifier from "./Classifier";
import { Bars, Hist, Scatter, Heat, COLORS, fmt } from "./Charts";

/* eslint-disable @typescript-eslint/no-explicit-any */
const NAMES = ["World", "Sports", "Business", "Sci/Tech"];
const get = (f: string) => fetch(`/data/${f}.json`).then((r) => r.json());

export default function Dashboard() {
  const [d, setD] = useState<any>(null);
  const [cls, setCls] = useState("World");
  const [sIdx, setSIdx] = useState(0);
  useEffect(() => {
    Promise.all(["eda", "embeddings", "linguistics", "models", "evaluation", "samples"].map(get)).then(([eda, emb, ling, models, ev, samples]) => setD({ eda, emb, ling, models, ev, samples }));
  }, []);
  const ex = d?.samples.examples ?? [];
  const smp = d?.samples.samples[sIdx];

  return (<>
    <nav><b>NewsLens</b>{["classify", "data", "embeddings", "linguistics", "models", "evaluation"].map((s) => <a key={s} href={`#${s}`}>{s}</a>)}</nav>
    <main>
      <h1>Which section does this story belong in?</h1>
      <p className="lead">A news classifier trained on the AG News corpus, running in your browser. Below it: the data, features, linguistics and evaluation behind the model.</p>
      <section id="classify"><Classifier examples={ex} /></section>
      {!d ? <p className="sub" style={{ marginTop: 40 }}>Loading analysis…</p> : (<>
        <section id="data">
          <h2>Dataset</h2><p className="sub">AG News (Zhang, Zhao & LeCun, 2015): four balanced topics. After removing {fmt(d.eda.duplicates_removed)} duplicate articles from the training split.</p>
          <div className="grid">
            <div className="card stat">{fmt(d.eda.train_size)}<small>training articles</small></div>
            <div className="card stat">{fmt(d.eda.test_size)}<small>test articles</small></div>
            <div className="card stat">{fmt(d.eda.vocab_clean)}<small>vocabulary after cleaning (from {fmt(d.eda.vocab_raw)} raw)</small></div>
          </div>
          <h3>Class balance</h3><div className="card"><Bars items={NAMES.map((n) => ({ label: n, value: d.eda.class_distribution[n], color: COLORS[n] }))} /></div>
          <h3>Article length (words)</h3>
          <div className="grid">{NAMES.map((n) => (<div className="card" key={n}><b>{n}</b> <span className="tag">median {d.eda.length_stats[n].median_words}</span><Hist bins={d.eda.length_hist[n]} color={COLORS[n]} /></div>))}</div>
          <h3>Most frequent words</h3>
          <div className="tabs">{NAMES.map((n) => <button key={n} className={cls === n ? "on" : ""} onClick={() => setCls(n)}>{n}</button>)}</div>
          <div className="cols"><div className="card"><Bars items={d.eda.top_words[cls].slice(0, 12).map((w: any) => ({ label: w.word, value: w.count, color: COLORS[cls] }))} /></div>
            <div className="card"><img className="wc" alt={`${cls} word cloud`} src={`/plots/wordcloud_${cls.toLowerCase().replace("/", "_")}.png`} /></div></div>
        </section>

        <section id="embeddings">
          <h2>Word embeddings</h2><p className="sub">Word2Vec (CBOW, 100-d) trained on a 20k-article subset, projected to 2-D with PCA. Colour = nearest topic centroid. Vocabulary: {fmt(d.emb.vocab)} words.</p>
          <div className="card"><Scatter words={d.emb.words} x={d.emb.x} y={d.emb.y} aff={d.emb.affinity} labels={NAMES} />
            <div className="chips">{NAMES.map((n) => <span key={n}><span className="badge" style={{ background: COLORS[n], fontSize: 12 }}>{n}</span> </span>)}</div></div>
          <h3>Nearest neighbours</h3>
          <div className="grid">{Object.entries(d.emb.similar).map(([w, s]: any) => (<div className="card" key={w}><b>{w}</b><div style={{ marginTop: 6 }}>{s.slice(0, 5).map((x: any) => <span className="tag" key={x[0]}>{x[0]}</span>)}</div></div>))}</div>
          <h3>Most distinctive terms (mean TF-IDF)</h3>
          <div className="grid">{NAMES.map((n) => (<div className="card" key={n}><b>{n}</b><Bars items={d.ev.tfidf_terms[n].slice(0, 8).map((t: any) => ({ label: t.term, value: t.score, color: COLORS[n] }))} /></div>))}</div>
        </section>

        <section id="linguistics">
          <h2>Linguistic analysis</h2><p className="sub">spaCy (en_core_web_sm) over {fmt(d.ling.docs_analyzed)} articles: part-of-speech mix, sentence length and named entities by topic.</p>
          <div className="grid">{NAMES.map((n) => (<div className="card" key={n}><b>{n}</b> <span className="tag">{d.ling.avg_sentence_length[n]} tokens / sentence</span>
            <Bars items={Object.entries(d.ling.pos_distribution[n]).slice(0, 6).map(([k, v]: any) => ({ label: k, value: v, color: COLORS[n] }))} unit="%" max={40} /></div>))}</div>
          <h3>Top organisations &amp; places</h3>
          <div className="grid">{NAMES.map((n) => (<div className="card" key={n}><b>{n}</b>
            <p className="sub" style={{ margin: "6px 0 2px" }}>ORG</p>{d.ling.top_entities[n].ORG.slice(0, 4).map((e: any) => <span className="tag" key={e.entity}>{e.entity} · {e.count}</span>)}
            <p className="sub" style={{ margin: "8px 0 2px" }}>GPE</p>{d.ling.top_entities[n].GPE.slice(0, 4).map((e: any) => <span className="tag" key={e.entity}>{e.entity} · {e.count}</span>)}</div>))}</div>
          <h3>Annotated examples</h3>
          <div className="tabs">{d.samples.samples.map((s: any, i: number) => <button key={i} className={sIdx === i ? "on" : ""} onClick={() => setSIdx(i)}>{s.label_name} {(i % 2) + 1}</button>)}</div>
          <div className="card"><p style={{ marginTop: 0 }}>{smp.text}</p>
            <p className="sub" style={{ margin: "0 0 4px" }}>Entities</p>{smp.entities.length ? smp.entities.map((e: any, i: number) => <span className="tag" key={i}>{e.text} <small>{e.label}</small></span>) : <span className="sub">none</span>}
            <p className="sub" style={{ margin: "10px 0 4px" }}>Noun chunks</p>{smp.noun_chunks.map((c: string, i: number) => <span className="tag" key={i}>{c}</span>)}
            <div className="scroll"><table style={{ marginTop: 12 }}><thead><tr><th>token</th><th>POS</th><th>dep</th><th>head</th><th>lemma</th></tr></thead><tbody>
              {smp.tokens.slice(0, 14).map((t: any, i: number) => <tr key={i}><td>{t.text}</td><td>{t.pos}</td><td>{t.dep}</td><td>{t.head}</td><td>{t.lemma}</td></tr>)}</tbody></table></div></div>
        </section>

        <section id="models">
          <h2>Model comparison</h2><p className="sub">Four classifiers × four feature sets, trained on a {fmt(d.models.trained_on)}-article subset and scored on {fmt(d.models.evaluated_on)} validation articles (macro-F1). “Combined” = TF-IDF + Word2Vec + 11 linguistic features.</p>
          <div className="card scroll"><table><thead><tr><th>Features</th><th>Model</th><th className="n">Accuracy</th><th className="n">Macro-F1</th><th className="n">Train (s)</th><th className="n">Infer (ms/doc)</th></tr></thead><tbody>
            {[...d.models.results].sort((a: any, b: any) => (b.val_f1 ?? -1) - (a.val_f1 ?? -1)).map((r: any, i: number) => (
              <tr key={i} style={{ opacity: r.skipped ? 0.45 : 1 }}><td>{r.features}</td><td>{r.model}</td>
                {r.skipped ? <td colSpan={4}>skipped — {r.note}</td> : <><td className="n">{(r.val_acc * 100).toFixed(2)}%</td><td className="n">{r.val_f1.toFixed(4)}</td><td className="n">{r.train_s}</td><td className="n">{r.infer_ms}</td></>}</tr>))}</tbody></table></div>
          <p className="sub" style={{ marginTop: 12 }}><b>Deployed:</b> {d.models.deployed.model} on {d.models.deployed.features}, retrained on the full training set — {d.models.deployed.size_mb} MB of JSON.</p>
        </section>

        <section id="evaluation">
          <h2>Evaluation on the held-out test set</h2>
          <div className="grid"><div className="card stat">{(d.ev.test_accuracy * 100).toFixed(2)}%<small>test accuracy ({fmt(d.eda.test_size)} articles)</small></div><div className="card stat">{d.ev.test_macro_f1.toFixed(4)}<small>macro-F1</small></div></div>
          <h3>Per class</h3><div className="card scroll"><table><thead><tr><th>Class</th><th className="n">Precision</th><th className="n">Recall</th><th className="n">F1</th></tr></thead><tbody>
            {NAMES.map((n) => <tr key={n}><td>{n}</td><td className="n">{d.ev.per_class[n].precision}</td><td className="n">{d.ev.per_class[n].recall}</td><td className="n">{d.ev.per_class[n].f1}</td></tr>)}</tbody></table></div>
          <h3>Confusion matrix</h3><div className="card"><Heat cm={d.ev.confusion_matrix} labels={d.ev.labels} /></div>
          <h3>What each class looks for (top SVM coefficients)</h3>
          <div className="grid">{NAMES.map((n) => (<div className="card" key={n}><b>{n}</b><Bars items={d.ev.top_terms[n].slice(0, 8).map((t: any) => ({ label: t.term, value: t.coef, color: COLORS[n] }))} /></div>))}</div>
          <h3>Sample mistakes</h3>
          <div className="card">{d.ev.misclassified.slice(0, 8).map((m: any, i: number) => (<p key={i} style={{ margin: "0 0 10px" }}><span className="tag">{m.true} → {m.pred}</span>{m.text}</p>))}</div>
        </section>
      </>)}
      <footer>Data: Zhang, Zhao &amp; LeCun (2015), “Character-level convolutional networks for text classification”, NIPS. Model: scikit-learn LinearSVC; inference reimplemented in TypeScript and verified against scikit-learn.</footer>
    </main></>);
}
