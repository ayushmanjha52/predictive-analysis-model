// Shared config + helpers used by script.js. Kept separate so the API
// base URL is the one thing you'd change to point at a deployed backend.
//
// PRODUCTION: change this to the real server's address before
// deployment (e.g. "http://10.0.4.12:8000") -- see DEPLOYMENT.md
// Section 5. Never leave this as 127.0.0.1 once the frontend and
// backend run on different machines.
const API_BASE_URL = "http://10.51.202.22:8000";

const DEVICE_COLORS = {
  PHOTOCELL: "#F4623A",
  LVDT: "#F4B740",
  ENCODER: "#4A90E2",
  HMD: "#3ED6A0",
  PROXIMITY: "#B27FE0",
  PRESSURE_SWITCH: "#8194A6",
  FLOW_SWITCH: "#5BC8E8",
  LASER: "#E87BA8",
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