import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Requests to /api are proxied to the FastAPI backend during development.
// This keeps the frontend code free of hardcoded hostnames and means the
// browser never makes a cross-origin request in the first place.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
