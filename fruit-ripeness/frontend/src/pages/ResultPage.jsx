// src/pages/ResultPage.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';

function ResultPage() {
  const navigate = useNavigate();
  
  // Contoh data hasil deteksi (nanti ini diganti dari API)
  const detectionResult = "Hasil Deteksi: Buah Mangga Indramayu dengan Akurasi 98.5%";

  const handleReturn = () => {
    // Kembali ke halaman utama
    navigate('/'); 
  };

  return (
    <div className="min-h-screen bg-green-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold text-green-800 p-4 border-b-4 border-green-500 mb-8">
        🔍 Hasil Deteksi
      </h1>
      
      {/* 1. Kolom yang berisi output */}
      <div className="w-full max-w-xl p-8 bg-white rounded-lg shadow-xl border-l-4 border-green-500">
        <h2 className="text-xl font-semibold text-green-800 mb-4">
          Status Deteksi:
        </h2>
        <p className="text-gray-700 whitespace-pre-wrap">
          {detectionResult}
          {/* Nanti di sini bisa tampilkan gambar hasil bounding box dll. */}
        </p>
        <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded">
            <span className="font-medium text-green-700">Catatan:</span> Output akan dikirim dari model Machine Learning temanmu.
        </div>
      </div>

      {/* 2. Tombol untuk return ke Page Utama */}
      <button
        onClick={handleReturn}
        className="mt-8 px-8 py-3 border border-green-500 text-green-700 font-semibold rounded-full transition duration-300 ease-in-out hover:bg-green-200 focus:outline-none focus:ring-4 focus:ring-green-300"
      >
        Kembali ke Halaman Utama
      </button>
    </div>
  );
}

export default ResultPage;