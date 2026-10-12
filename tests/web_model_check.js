// Runs frontend/model.js over cases exported by tests/test_all.py and
// prints the predictions as JSON (used to check parity with Python).
// Usage: node tests/web_model_check.js <model.json> <cases.json>
const fs = require("fs");
const { DelayClassifier } = require("../frontend/model.js");

const [modelPath, casesPath] = process.argv.slice(2);
const clf = new DelayClassifier(JSON.parse(fs.readFileSync(modelPath, "utf8")));
const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));
const out = cases.map(c => {
  const r = clf.predict(c.reason_text, c.area, c.plant_type);
  return { predicted_device: r.predicted_device, confidence: r.confidence, decision_basis: r.decision_basis };
});
process.stdout.write(JSON.stringify(out));
