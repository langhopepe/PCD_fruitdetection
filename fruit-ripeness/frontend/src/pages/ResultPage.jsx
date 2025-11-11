// src/pages/ResultPage.jsx
import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function Bar({ label, value }) {
  const v = (typeof value === 'number' && Number.isFinite(value)) ? value : 0;
  const pct = Math.max(0, Math.min(100, Math.round(v * 100)));
  return (
    <div className="mb-2">
      <div className="flex justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span>{pct}%</span>
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

  useEffect(() => {
    if (!state?.result) navigate('/', { replace: true });
  }, [state, navigate]);

  if (!state?.result) return null;

  const { result, filename, preview } = state;
  // ✅ Pastikan kita men-destructure fruit & fruit_conf dari result
  const {
    fruit,
    fruit_conf,
    label,
    score,
    probs,
    overlay_data_url,
  } = result || {};

  // Safeguards
  const allowed = new Set(['unripe', 'ripe', 'overripe']);
  const rawLabel = String(label ?? '').toLowerCase();
  const safeLabel = allowed.has(rawLabel) ? rawLabel : 'unknown';
  const titleLabel = safeLabel === 'unknown' ? 'UNKNOWN' : safeLabel.toUpperCase();

  const pUnripe = probs?.unripe;
  const pRipe = probs?.ripe;
  const pOver = probs?.overripe;
  const maxProb = Math.max(
    (typeof pUnripe === 'number' ? pUnripe : -Infinity),
    (typeof pRipe === 'number' ? pRipe : -Infinity),
    (typeof pOver === 'number' ? pOver : -Infinity)
  );
  const lowConfidence = Number.isFinite(maxProb) && maxProb < 0.6;

  const scoreText = (typeof score === 'number' && Number.isFinite(score)) ? score.toFixed(2) : '—';
  const fruitName = fruit ? String(fruit).toUpperCase() : '—';
  const fruitConf = (typeof fruit_conf === 'number' && Number.isFinite(fruit_conf))
    ? `${Math.round(fruit_conf * 100)}%`
    : '—';

  const imgSrc = overlay_data_url || preview || '';

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

          <div className="text-sm text-gray-700 mb-1">
            Buah terdeteksi: <b>{fruitName}</b>{' '}
            <span className="text-gray-500">(conf {fruitConf})</span>
          </div>

          <div className="text-3xl font-bold mb-2">{titleLabel}</div>
          <div className="text-sm text-gray-600 mb-4">Score (≈ prob. kelas): {scoreText}</div>

          {safeLabel === 'unknown' && (
            <div className="mb-3 p-3 bg-yellow-50 border border-yellow-300 rounded text-yellow-800 text-sm">
              Label tidak dikenal. Pastikan model ripeness hanya punya kelas:
              <b> unripe</b>, <b> ripe</b>, dan <b> overripe</b>.
            </div>
          )}

          {safeLabel !== 'unknown' && lowConfidence && (
            <div className="mb-3 p-3 bg-orange-50 border border-orange-300 rounded text-orange-800 text-sm">
              Model kurang yakin terhadap hasil ini (maks prob &lt; 0.60).
              Coba unggah foto yang lebih terang, dekat, dan latar polos.
            </div>
          )}

          <div className="mt-2">
            <Bar label="Unripe"   value={pUnripe} />
            <Bar label="Ripe"     value={pRipe} />
            <Bar label="Overripe" value={pOver} />
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
          {imgSrc ? (
            <img
              src={imgSrc}
              alt="hasil"
              className="max-h-[420px] w-full object-contain rounded-md border"
            />
          ) : (
            <div className="h-[200px] grid place-items-center text-gray-400 text-sm">
              (Tidak ada gambar)
            </div>
          )}
          {!overlay_data_url && imgSrc && (
            <p className="text-xs text-gray-500 mt-2">
              * Overlay tidak tersedia—menampilkan gambar yang diupload.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
