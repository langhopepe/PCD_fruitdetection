// src/App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ResultPage from './pages/ResultPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Rute untuk Page Utama */}
        <Route path="/" element={<HomePage />} />
        
        {/* Rute untuk Page Hasil */}
        <Route path="/hasil" element={<ResultPage />} />
        
        {/* Tambahan: Rute jika halaman tidak ditemukan (optional) */}
        <Route path="*" element={<h1 className="text-center mt-20 text-red-500">404 - Halaman Tidak Ditemukan</h1>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;