import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Keeps the browser on one origin in development, so no CORS
      // configuration is needed for the common local setup.
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  // `vite preview` serves the production build and does not inherit server.proxy,
  // so the same proxy is declared for it. In deployment nginx fills this role.
  preview: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // The map engine and the chart library are large and change rarely.
        // Splitting them keeps the app chunk small and cacheable across deploys.
        manualChunks: {
          maplibre: ['maplibre-gl'],
          charts: ['recharts'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
})
