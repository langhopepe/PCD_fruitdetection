// src/pages/ResultPage.jsx
import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function Bar({ label, value }) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div className="mb-2">
      <div className="flex justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span>{isFinite(pct) ? pct : 0}%</span>
      </div>
      <div className="w-full bg-gray-200 h-2 rounded">
        <div className="h-2 rounded" style={{ width: `${pct}%`, background: '#16a34a' }} />
      </div>
    </div>
  );
}

export default function ResultPage() {
  const { state } = useLocation(); // { result, filename, preview }
  const navigate = useNavigate();

  // Kalau user refresh / masuk langsung tanpa state → balik ke home
  useEffect(() => {
    if (!state?.result) navigate('/');
  }, [state, navigate]);

  if (!state?.result) return null;

  const { result, filename, preview } = state;
  const { label, score, probs, overlay_data_url } = result;

  return (
    <div className="min-h-screen bg-green-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold text-green-800 p-4 border-b-4 border-green-500 mb-8">
        🔍 Hasil Deteksi
      </h1>

      <div className="w-full max-w-4xl grid md:grid-cols-2 gap-6">
        {/* Panel teks */}
        <div className="p-6 bg-white rounded-lg shadow-xl border-l-4 border-green-500">
          <h2 className="text-xl font-semibold text-green-800 mb-2">Status Deteksi</h2>
          <p className="text-sm text-gray-500 mb-4">{filename || 'uploaded image'}</p>

          <div className="text-3xl font-bold mb-2">{String(label).toUpperCase()}</div>
          <div className="text-sm text-gray-600 mb-4">Score (≈ prob. ripe): {(score ?? 0).toFixed(2)}</div>

          <div className="mt-2">
            <Bar label="Unripe"   value={probs?.unripe} />
            <Bar label="Ripe"     value={probs?.ripe} />
            <Bar label="Overripe" value={probs?.overripe} />
          </div>

          <button
            onClick={() => navigate('/')}
            className="mt-6 px-5 py-2 rounded-full border border-green-500 text-green-700 hover:bg-green-200"
          >
            Analisis Gambar Lain
          </button>
        </div>

        {/* Panel gambar */}
        <div className="p-6 bg-white rounded-lg shadow-xl border">
          <img
            src={overlay_data_url || preview}
            alt="hasil"
            className="max-h-[420px] w-full object-contain rounded-md border"
          />
          {!overlay_data_url && (
            <p className="text-xs text-gray-500 mt-2">
              * Overlay tidak tersedia—menampilkan gambar yang diupload.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}