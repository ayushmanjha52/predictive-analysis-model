// In-browser delay classifier for the static (GitHub Pages) build.
// Mirrors src/feature_engineering.py + src/predict.py exactly:
// TF-IDF (word 1-2 grams + char_wb 3-5 grams, sublinear tf, l2 per
// block), keyword flags, area / plant-type one-hot, logistic regression,
// home-class restriction and the explicit-device-mention rule.
// tests/test_web_model.js checks it against the Python predictor.

function cleanText(text) {
  if (typeof text !== "string") return "";
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
}

function wordTerms(clean, stopWords) {
  const tokens = clean.split(" ").filter(t => t.length >= 2 && !stopWords.has(t));
  const terms = [...tokens];
  for (let i = 0; i + 1 < tokens.length; i++) terms.push(tokens[i] + " " + tokens[i + 1]);
  return terms;
}

function charWbTerms(clean, minN, maxN) {
  const grams = [];
  for (const word of clean.split(/\s+/).filter(Boolean)) {
    const w = " " + word + " ";
    for (let n = minN; n <= maxN; n++) {
      let offset = 0;
      grams.push(w.slice(offset, offset + n));
      while (offset + n < w.length) {
        offset += 1;
        grams.push(w.slice(offset, offset + n));
      }
      if (offset === 0) break; // short word: counted once
    }
  }
  return grams;
}

class DelayClassifier {
  constructor(m) {
    this.m = m;
    this.stop = new Set(m.stop_words);
    this.featureIndex = new Map(m.feature_names.map((f, i) => [f, i]));
    this.blocks = Object.entries(m.tfidf).map(([name, t]) => ({
      name, ...t, index: new Map(t.vocabulary.map((term, i) => [term, i])),
    }));
    this.explicit = Object.entries(m.explicit_names).map(([dev, p]) => [dev, new RegExp(p, "i")]);
    this.homeClasses = new Set(m.home_classes);
  }

  features(reasonText, area, plantType) {
    const x = new Float64Array(this.m.feature_names.length);
    const set = (name, value) => {
      const i = this.featureIndex.get(name);
      if (i !== undefined) x[i] = value;
    };
    const clean = cleanText(reasonText);

    for (const [name, kws] of Object.entries(this.m.domain_keywords)) {
      set(name, kws.some(k => clean.includes(k)) ? 1 : 0);
    }
    set("text_length", clean.length);
    set("word_count", clean ? clean.split(" ").length : 0);

    const areaClean = area == null ? "UNKNOWN" : String(area).toUpperCase().trim();
    for (const a of this.m.known_areas) {
      set(("area_" + a).replace(/ /g, "_").replace(/-/g, "_"), areaClean === a ? 1 : 0);
    }
    const pt = this.m.plant_types.includes(plantType) ? plantType : "OTHER";
    for (const p of this.m.plant_types) set("pt_" + p, pt === p ? 1 : 0);

    for (const b of this.blocks) {
      const terms = b.analyzer === "word"
        ? wordTerms(clean, this.stop)
        : charWbTerms(clean, b.ngram_range[0], b.ngram_range[1]);
      const counts = new Map();
      for (const t of terms) {
        const j = b.index.get(t);
        if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
      }
      const weights = [];
      let norm = 0;
      for (const [j, c] of counts) {
        const w = (1 + Math.log(c)) * b.idf[j];
        weights.push([j, w]);
        norm += w * w;
      }
      norm = Math.sqrt(norm) || 1;
      for (const [j, w] of weights) set("tfidf_" + b.name + "__" + b.vocabulary[j], w / norm);
    }
    return x;
  }

  predict(reasonText, area = null, plantType = null, topN = 3) {
    const m = this.m;
    plantType = (plantType || m.home_plant_type).toUpperCase();
    const x = this.features(reasonText, area, plantType);

    const logits = m.coef.map((row, k) => row.reduce((s, c, i) => s + c * x[i], m.intercept[k]));
    const mx = Math.max(...logits);
    let proba = logits.map(l => Math.exp(l - mx));
    let total = proba.reduce((a, b) => a + b, 0);
    proba = proba.map(p => p / total);

    if (plantType === m.home_plant_type && this.homeClasses.size) {
      const masked = proba.map((p, k) => this.homeClasses.has(m.classes[k]) ? p : 0);
      total = masked.reduce((a, b) => a + b, 0);
      if (total > 0) proba = masked.map(p => p / total);
    }

    const order = proba.map((p, k) => k).sort((a, b) => proba[b] - proba[a]);
    let candidates = order.slice(0, topN).map(k => ({ device: m.classes[k], confidence: +proba[k].toFixed(4) }));
    let decisionBasis = "model";

    if (m.explicit_rule && plantType === m.home_plant_type) {
      const named = this.explicit.filter(([, rx]) => rx.test(String(reasonText || ""))).map(([d]) => d);
      if (named.length === 1 && this.homeClasses.has(named[0])) {
        const dev = named[0];
        const conf = +Math.max(m.explicit_rule_precision || 0, proba[m.classes.indexOf(dev)]).toFixed(4);
        candidates = [{ device: dev, confidence: conf },
          ...candidates.filter(c => c.device !== dev)].slice(0, topN);
        decisionBasis = "explicit_device_mention";
      }
    }

    const top = candidates[0];
    const d = m.durations[top.device] || null;
    return {
      reason_text: reasonText, area, plant_type: plantType,
      predicted_device: top.device, confidence: top.confidence,
      decision_basis: decisionBasis,
      low_confidence: top.confidence < m.confidence_threshold,
      top_candidates: candidates,
      typical_delay_minutes: d ? d.mean : null,
      based_on_n_historical_events: d ? d.count : 0,
    };
  }
}

if (typeof module !== "undefined") module.exports = { DelayClassifier, cleanText };
