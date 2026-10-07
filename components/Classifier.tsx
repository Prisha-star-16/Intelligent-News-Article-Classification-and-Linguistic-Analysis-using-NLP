"use client";
import { useEffect, useRef, useState } from "react";
import { classify, type Model, type Result } from "@/lib/classifier";
import { Bars, COLORS } from "./Charts";

export default function Classifier({ examples }: { examples: string[] }) {
  const [text, setText] = useState("");
  const [res, setRes] = useState<Result | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const model = useRef<Model | null>(null);

  useEffect(() => {
    setState("loading");
    fetch("/model/model.json").then((r) => r.json()).then((m: Model) => { model.current = m; setState("ready"); }).catch(() => setState("error"));
  }, []);
  useEffect(() => {
    if (model.current && text.trim().length > 2) setRes(classify(model.current, text)); else setRes(null);
  }, [text, state]);

  return (<div className="card">
    <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste a headline or short news article…" aria-label="Article text" />
    <div className="chips">{examples.map((e) => <button className="chip" key={e} onClick={() => setText(e)}>{e.slice(0, 44)}…</button>)}</div>
    {state === "loading" && <p className="sub">Loading model (~2 MB, cached after first load)…</p>}
    {state === "error" && <p className="sub neg">Couldn’t load the model file.</p>}
    {res && (<div className="cols">
      <div><h3>Prediction</h3><span className="badge" style={{ background: COLORS[res.name] }}>{res.name}</span>
        <p className="sub" style={{ marginTop: 8 }}>{res.known} known n-grams from {res.tokens} content words.{res.known === 0 && " Try a longer text."}</p>
        <Bars items={res.probs.map((p, i) => ({ label: model.current!.labels[i], value: p * 100, color: COLORS[model.current!.labels[i]] }))} max={100} unit="%" /></div>
      <div><h3>Why — top term contributions</h3>
        <Bars items={res.terms.map((t) => ({ label: t.term, value: t.contribution, color: t.contribution >= 0 ? "var(--Sports)" : "#dc2626" }))} /></div>
    </div>)}
    <p className="sub" style={{ marginTop: 12, marginBottom: 0 }}>Runs 100% in your browser — your text is never sent anywhere. Bars show relative margin (softmax of SVM scores), not calibrated probability.</p>
  </div>);
}
