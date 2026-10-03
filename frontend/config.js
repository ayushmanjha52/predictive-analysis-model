// Shared config + helpers used by script.js.
//
// API_BASE_URL: empty string = same origin. The FastAPI app serves this
// dashboard itself (see app.py), so on Render -- or any single-server
// deployment -- no URL needs editing. Only when the page is opened from
// a separate dev server (e.g. VS Code Live Server on :5500) does it
// fall back to a local backend on :8000.
const API_BASE_URL = (location.port === "5500" || location.protocol === "file:")
  ? "http://127.0.0.1:8000"
  : "";

const DEVICE_COLORS = {
  PHOTOCELL: "#F4623A",
  LVDT: "#F4B740",
  ENCODER: "#4A90E2",
  HMD: "#3ED6A0",
  PROXIMITY: "#B27FE0",
  PRESSURE_SWITCH: "#8194A6",
  FLOW_SWITCH: "#5BC8E8",
  LASER: "#E87BA8",
  LEVEL_SWITCH: "#9CCC65",
  TEMPERATURE_SWITCH: "#FF8A65",
  LIMIT_SWITCH: "#BA68C8",
  PRESSURE_TRANSMITTER: "#4DB6AC",
  FLOW_TRANSMITTER: "#7986CB",
  LEVEL_TRANSMITTER: "#AED581",
  TEMPERATURE_SENSOR: "#FFB74D",
  SPEED_SENSOR: "#90A4AE",
  VIBRATION_SENSOR: "#F06292",
};

function deviceColor(device) {
  return DEVICE_COLORS[device] || "#5A6C7E";
}

function formatDevice(device) {
  if (!device) return "UNRESOLVED";
  return device.replace(/_/g, " ");
}

function riskColor(riskLevel) {
  if (riskLevel === "High") return "#F4623A";
  if (riskLevel === "Medium") return "#F4B740";
  if (riskLevel === "Low") return "#3ED6A0";
  return "#8194A6"; // UNKNOWN
}
