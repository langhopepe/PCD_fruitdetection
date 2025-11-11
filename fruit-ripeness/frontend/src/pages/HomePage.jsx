// src/pages/HomePage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// Pakai .env (VITE_API_BASE_URL) kalau ada, default ke localhost:8000
const API_BASE = '/api';

export default function HomePage() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');

  function onPick(e) {
    const f = e.target.files?.[0];
    setErr('');
    setFile(f || null);
    setPreview(f ? URL.createObjectURL(f) : '');
  }

  async function handleSubmit() {
    setErr('');
    if (!file) return setErr('Pilih gambar dulu.');
    if (file.size > 5 * 1024 * 1024) return setErr('Ukuran file > 5MB.');

    try {
      setLoading(true);
      const form = new FormData();
      form.append('file', file); // backend kita hanya butuh "file"

      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form });

        if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try {
          const data = await res.json();
          // Backend 422 bentuknya: { detail: { error, predicted, confidence, hint } }
          if (data?.detail) {
            if (typeof data.detail === 'string') {
              msg = data.detail;
            } else {
              const { error, predicted, confidence, hint } = data.detail;
              msg = error || msg;
              if (predicted) msg += ` (prediksi: ${String(predicted).toUpperCase()}, conf=${confidence ?? '-'})`;
              if (hint) msg += ` — ${hint}`;
            }
          }
        } catch {
          const text = await res.text().catch(() => '');
          if (text) msg = text;
        }
        throw new Error(msg);
      }

      const result = await res.json(); // {fruit, fruit_conf, label, score, probs, overlay_data_url}
      navigate('/hasil', { state: { result, filename: file.name, preview } });
    } catch (ex) {
      console.error(ex);
      setErr(ex.message || 'Gagal mengirim ke server.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-green-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold text-green-800 p-4 border-b-4 border-green-500 mb-8">
        🍏 Deteksi Kematangan Buah - Upload
      </h1>

      <div className="w-full max-w-md p-6 bg-white rounded-lg shadow-xl border border-green-300">
        <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700 mb-2">
          Pilih Gambar Buah (Max 5MB)
        </label>

        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={onPick}
          className="w-full text-sm text-gray-500 
            file:mr-4 file:py-2 file:px-4 
            file:rounded-full file:border-0 
            file:text-sm file:font-semibold
            file:bg-green-50 file:text-green-700
            hover:file:bg-green-100"
        />

        {preview && (
          <img
            src={preview}
            alt="preview"
            className="mt-4 max-h-64 w-full object-contain rounded-lg border"
          />
        )}

        {err && (
          <div className="mt-3 text-sm rounded-md border border-red-300 bg-red-50 p-3 text-red-700">
            {err}
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={loading || !file}
          className="mt-6 w-full px-8 py-3 bg-green-500 text-white font-semibold rounded-full shadow-lg 
                     transition duration-300 ease-in-out transform hover:scale-105 hover:bg-green-600 
                     focus:outline-none focus:ring-4 focus:ring-green-300 disabled:opacity-50"
        >
          {loading ? 'Menganalisis…' : 'Submit Gambar & Deteksi ➡️'}
        </button>

        <p className="mt-3 text-xs text-gray-400">
          API: {API_BASE}/analyze
        </p>
      </div>
    </div>
  );
}