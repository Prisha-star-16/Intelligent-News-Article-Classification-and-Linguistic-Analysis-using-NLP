import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { classify, clean, type Model } from "../lib/classifier";

const model: Model = JSON.parse(readFileSync("public/model/model.json", "utf8"));
const parity: { text: string; pred: number }[] = JSON.parse(readFileSync("tests/parity.json", "utf8"));
const stop = new Set(model.stopwords);

describe("cleaning", () => {
  it("strips html, urls, digits and stopwords", () => {
    expect(clean("The <b>President</b> signed 3 bills at https://x.gov", stop)).toEqual(["president", "signed", "bills"]);
  });
});
describe("classifier", () => {
  it("matches scikit-learn LinearSVC on 500 held-out test articles", () => {
    const ok = parity.filter((p) => classify(model, p.text, stop).label === p.pred).length;
    expect(ok / parity.length).toBeGreaterThanOrEqual(0.99);
  });
  it("classifies obvious headlines", () => {
    expect(classify(model, "Manchester United won the Champions League final after a penalty shootout", stop).name).toBe("Sports");
    expect(classify(model, "Stocks fell as oil prices and quarterly profits disappointed investors", stop).name).toBe("Business");
  });
});
