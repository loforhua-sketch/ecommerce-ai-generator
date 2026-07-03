import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "",
  timeout: 180000
});

export function generateDetails(payload) {
  return api.post("/api/generate", payload);
}

export function exportUrl(id) {
  const base = import.meta.env.VITE_API_BASE_URL || "";
  return `${base}/api/generations/${id}/export`;
}
