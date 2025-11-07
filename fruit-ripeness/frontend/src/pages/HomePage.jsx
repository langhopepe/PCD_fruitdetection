// src/pages/HomePage.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';

function HomePage() {
  const navigate = useNavigate();

  const handleSubmit = () => {
    // Di sini nanti kamu akan kirim gambar ke backend/model ML temanmu.
    // Setelah berhasil upload, kita pindah ke halaman hasil.
    console.log("Gambar di-upload dan submit. Mengirim ke Page Hasil...");
    
    // Pindah ke halaman '/hasil'
    navigate('/hasil'); 
  };

  return (
    <div className="min-h-screen bg-green-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold text-green-800 p-4 border-b-4 border-green-500 mb-8">
        🍏 Deteksi Kematangan Buah - Upload
      </h1>

      {/* 1. Kolom/Tombol Upload Gambar */}
      <div className="w-full max-w-md p-6 bg-white rounded-lg shadow-xl border border-green-300">
        <label 
          htmlFor="file-upload" 
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          Pilih Gambar Buah (Max 5MB)
        </label>
        
        {/* Kolom Upload Sederhana */}
        <input 
          id="file-upload" 
          type="file" 
          accept="image/*" 
          className="w-full text-sm text-gray-500 
            file:mr-4 file:py-2 file:px-4 
            file:rounded-full file:border-0 
            file:text-sm file:font-semibold
            file:bg-green-50 file:text-green-700
            hover:file:bg-green-100
          "
        />
      </div>

      {/* 2. Tombol Submit Gambar */}
      <button 
        onClick={handleSubmit}
        className="mt-6 px-8 py-3 bg-green-500 text-white font-semibold rounded-full shadow-lg transition duration-300 ease-in-out transform hover:scale-105 hover:bg-green-600 focus:outline-none focus:ring-4 focus:ring-green-300"
      >
        Submit Gambar & Deteksi ➡️
      </button>
    </div>
  );
}

export default HomePage;