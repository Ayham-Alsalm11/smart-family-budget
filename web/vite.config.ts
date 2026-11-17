/**
 * Vite and Vitest configuration of the web interface (docs/ARCHITECTURE.md §10).
 * The dev server stays on 5173, the origin the backend allows by default (CORS_ORIGINS).
 */
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
