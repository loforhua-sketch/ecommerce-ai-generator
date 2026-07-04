import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "",
  timeout: 180000
});

export function generateDetails(payload) {
  return api.post("/api/generate", payload);
}

export function exportUrl(id, platform = "taobao") {
  const base = import.meta.env.VITE_API_BASE_URL || "";
  return `${base}/api/generations/${id}/export?platform=${encodeURIComponent(platform)}`;
}

export function exportZipUrl(ids) {
  const base = import.meta.env.VITE_API_BASE_URL || "";
  const query = ids.map((id) => encodeURIComponent(id)).join(",");
  return `${base}/api/generations/export.zip?ids=${query}`;
}
