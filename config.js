// Shared config + helpers used by script.js.
//
// API_BASE_URL is empty (same origin) because app.py serves this
// dashboard. A separate dev server (e.g. Live Server on :5500) falls
// back to a local backend on :8000.
const API_BASE_URL = (location.port === "5500" || location.protocol === "file:")
  ? "http://127.0.0.1:8000"
  : "";

// Categorical palette (8 slots, checked for colour-blind separation on
// the dark surface). Fixed per device, in delay-Pareto order, so a
// device keeps its colour across every chart. Power-plant-only device
// classes share a neutral grey: they are always labelled by name.
const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const DEVICE_COLORS = {
  PHOTOCELL: SERIES[0],
  LVDT: SERIES[1],
  ENCODER: SERIES[2],
  PROXIMITY: SERIES[3],
  HMD: SERIES[4],
  PRESSURE_SWITCH: SERIES[5],
  FLOW_SWITCH: SERIES[6],
  LASER: SERIES[7],
};

function deviceColor(device) {
  return DEVICE_COLORS[device] || "#6f6c66";
}

function formatDevice(device) {
  if (!device) return "UNRESOLVED";
  return device.replace(/_/g, " ");
}

function riskColor(riskLevel) {
  if (riskLevel === "High") return "#d03b3b";
  if (riskLevel === "Medium") return "#fab219";
  if (riskLevel === "Low") return "#0ca30c";
  return "#a8a49c"; // UNKNOWN
}
