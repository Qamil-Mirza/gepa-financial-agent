// The dashboard build: `npm run build` writes the app into the Python package (src/pitstop/server/static), which
// `pitstop ui` serves and the wheel bundles. `npm run dev` serves on 5173 and proxies /api to a running `pitstop ui`.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    // changeOrigin sends Host: 127.0.0.1:8787, which the server's Host check allows
    proxy: { "/api": { target: "http://127.0.0.1:8787", changeOrigin: true } },
  },
  build: {
    outDir: "../src/pitstop/server/static",
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
    unstubGlobals: true,
  },
});
