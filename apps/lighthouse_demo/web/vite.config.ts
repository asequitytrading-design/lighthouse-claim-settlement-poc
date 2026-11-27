import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the API runs on 8787 (uvicorn); Vite proxies /api so the UI calls the same paths as in production.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": { target: "http://localhost:8787", changeOrigin: true } } },
  build: { outDir: "dist", emptyOutDir: true },
});
