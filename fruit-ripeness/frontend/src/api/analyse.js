// frontend/src/api/analyze.js
export async function analyzeImage(file) {
  const form = new FormData();
  form.append("file", file); // BE kita minta "file" saja

  const base = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "");
  const res = await fetch(`${base}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json(); // {label, score, probs?, overlay_data_url?}
}