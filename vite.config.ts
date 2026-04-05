import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { VitePWA } from "vite-plugin-pwa";
import { comlink } from "vite-plugin-comlink";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [comlink(), VitePWA({ registerType: "autoUpdate" }), svelte()],
  worker: {
    plugins: () => [comlink()],
  },
});
