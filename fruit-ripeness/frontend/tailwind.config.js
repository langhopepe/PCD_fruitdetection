/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    // PENTING: Agar Tailwind meng-scan semua file JS/TS/JSX/TSX di dalam folder src/
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}

