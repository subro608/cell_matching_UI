import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative base so the build works from any prefix (local serve.py, Vercel, an R2 path).
export default defineConfig({ plugins: [react()], base: './', build: { chunkSizeWarningLimit: 4000 } });
